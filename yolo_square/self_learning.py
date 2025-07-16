# yolo_square/self_learning.py

import os, glob
import shutil
import torch
import cv2
import numpy as np
from torchvision.ops import nms
from typing import List, Optional
from sklearn.model_selection import train_test_split
from .io_helpers import (
    ensure_dir,
    list_image_files,
    load_yolo_label,
    save_yolo_label,
    get_latest_yolo_weights,
    expand_to_tiles_for_images,
)
from .box_utils import square_to_xyxy, touches_border
from .data_preparation import (
    split_images_into_tiles_with_overlap, 
    prepare_selected_yolo_dataset, 
    select_top_k_tiles_per_image, 
    select_random_k_tiles_per_image
)
from .yolo_detection import detect_full_with_tiles, detect_with_yolo
from .yolo_training import train_yolo_model
from .evaluation import evaluate_iteration
from ultralytics import YOLO

def read_detection_labels(det_txt_path, img_shape):
    labels = []
    with open(det_txt_path, "r") as f:
        next(f)  # skip header
        for line in f:
            idx, x1, y1, x2, y2, conf = line.strip().split()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            side = max(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            labels.append([cx, cy, side])
    return labels

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

def merge_labels(existing_labels, new_labels, img_shape, iou_thresh=0.5):
    existing_xyxy = [square_to_xyxy(*lbl) for lbl in existing_labels]
    new_xyxy = [square_to_xyxy(*lbl) for lbl in new_labels]

    keep_existing = []
    for ex_lbl, ex_box in zip(existing_labels, existing_xyxy):
        ex_box_area = (ex_box[2]-ex_box[0])*(ex_box[3]-ex_box[1])
        overlap = False
        for new_box in new_xyxy:
            intersection = iou(ex_box, new_box)
            if intersection > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep_existing.append(ex_lbl)

    # return combined new labels + kept existing
    return new_labels + keep_existing

def _load_px_labels(txt_path, tile_path):
    # returns list of (cx,cy,side) in px
    out = []
    img = cv2.imread(tile_path)
    H, W = img.shape[:2]
    for cls, xcn, ycn, wn, hn in load_yolo_label(txt_path):
        cx, cy = xcn * W, ycn * H
        side   = wn * W  # wn==hn
        out.append((cx, cy, side))
    return out


def create_self_learning_dataset(
    inputnames_:         list[str],
    tile_dir:            str   = "images_split",
    previous_labels_dir: str   = "yolo_dataset_select/labels",
    output_dataset_dir:  str   = "yolo_self_learning/dataset",
    model_path:          str   = None,
    k:                   int   = 5,
    conf_threshold:      float = 0.25,
    device:              str   = "cpu",
    random_tiles:        bool  = True
) -> str:
    """
    1) Runs YOLO on each tile in `tile_dir`, writes YOLO .txt files under
       temp_pred/labels/{tile}.txt
    2) Merges those with previous_labels_dir/{tile}.txt—any old box overlapping
       a new one by >50% IOU is dropped—writes all results under
       merged_labels/{tile}.txt (always creates a file, even if empty)
    3) Picks top‐k tiles per original image (by merged‐label count) into
       `{output_dataset_dir}/selected/{images,labels}`
    4) Calls prepare_selected_yolo_dataset(...) on that selected folder
    """

    # ——— prep dirs ————————————————————————————————————————————————
    ensure_dir(output_dataset_dir)
    temp_pred = os.path.join("yolo_predictions", "self_learning_detection")
    temp_lbls = os.path.join(temp_pred, "labels")
    ensure_dir(temp_pred)
    ensure_dir(temp_lbls)

    merged_lbls = os.path.join(output_dataset_dir, "merged_labels")
    ensure_dir(merged_lbls)

    # ——— load model once —————————————————————————————————————
    if model_path is None:
        model_path = get_latest_yolo_weights("yolo_output")
    model = YOLO(model_path)

    # ——— 1) per‐tile detection → YOLO .txt —————————————————————————
    # expand inputnames_ into the tile list
    tiles = expand_to_tiles_for_images(image_files=inputnames_, split_dir=tile_dir)

    for tile in tiles:
        tile_path = os.path.join(tile_dir, tile)
        img = cv2.imread(tile_path)
        if img is None:
            print(f"⚠️  Skipping unreadable tile: {tile_path}")
            continue
        H, W = img.shape[:2]
        name = os.path.splitext(tile)[0]

        # 1.1 Inference A
        resultA = model.predict(source=tile_path,
                                conf=conf_threshold,
                                device=device,
                                verbose=False)[0]
        boxesA = resultA.boxes.xyxy.cpu().numpy()   # [[x1,y1,x2,y2],…]
        scoresA = resultA.boxes.conf.cpu().numpy()

        # 1.2 Inference B on flipped tile
        flipped = cv2.flip(img, 1)
        resultB = model.predict(source=flipped,
                                conf=conf_threshold,
                                device=device,
                                verbose=False)[0]
        boxesB = resultB.boxes.xyxy.cpu().numpy()
        scoresB = resultB.boxes.conf.cpu().numpy()
        # map B back to original coords (mirror x)
        boxesB[:, [0,2]] = W - boxesB[:, [2,0]]

        # 1.3 Ensemble: keep boxesA that match any boxB by IoU ≥ 0.5
        def iou_xyxy(a, b):
            xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
            xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            areaA = (a[2]-a[0])*(a[3]-a[1])
            areaB = (b[2]-b[0])*(b[3]-b[1])
            return inter / float(areaA + areaB - inter) if (areaA+areaB-inter)>0 else 0

        keep_idx = []
        for i, a in enumerate(boxesA):
            for b in boxesB:
                if iou_xyxy(a, b) >= 0.5:
                    keep_idx.append(i)
                    break
        boxes = boxesA[keep_idx]
        scores = scoresA[keep_idx]

        # 1.4 Class-agnostic NMS with IoU = 0.3
        if len(boxes):
            tboxes = torch.tensor(boxes, dtype=torch.float32)
            tscores = torch.tensor(scores, dtype=torch.float32)
            keep = nms(tboxes, tscores, 0.3)
            boxes = boxes[keep.numpy()]
            scores = scores[keep.numpy()]

        # 1.5 Write out pseudo-labels
        out_txt = os.path.join(temp_lbls, name + ".txt")
        with open(out_txt, "w") as f:
            for (x1,y1,x2,y2), conf in zip(boxes, scores):
                # convert to YOLO xywh normalized
                cx, cy = ((x1+x2)/2)/W, ((y1+y2)/2)/H
                wn, hn = (x2-x1)/W, (y2-y1)/H
                f.write(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")

    # ——— 2) merge with any existing labels ——————————————————————————
    for tile in tiles:
        tile_path = os.path.join(tile_dir, tile)
        name = os.path.splitext(tile)[0]

        new_txt = os.path.join(temp_lbls, name + ".txt")
        old_txt = os.path.join(previous_labels_dir, name + ".txt")
        new_lbls = _load_px_labels(new_txt, tile_path) \
                   if os.path.exists(new_txt) else []
        old_lbls = _load_px_labels(old_txt, tile_path) \
                   if os.path.exists(old_txt) else []
        
    # ─── prune any two boxes with IoU > 0.5, keeping only the smaller ─────
        merged = new_lbls + old_lbls
        to_remove = set()
        for i, (cx, cy, side) in enumerate(merged):
            xi1, yi1, xi2, yi2 = square_to_xyxy(cx, cy, side)
            area_i = (xi2 - xi1) * (yi2 - yi1)
            for j, (cx2, cy2, side2) in enumerate(merged):
                if j <= i: 
                    continue
                xj1, yj1, xj2, yj2 = square_to_xyxy(cx2, cy2, side2)
                inter_w = max(0, min(xi2, xj2) - max(xi1, xj1))
                inter_h = max(0, min(yi2, yj2) - max(yi1, yj1))
                inter_area = inter_w * inter_h
                area_j = (xj2 - xj1) * (yj2 - yj1)
                union   = area_i + area_j - inter_area
                iou      = inter_area / union if union > 0 else 0
                if iou > 0.5:
                    # mark the larger box for removal
                    if area_i > area_j:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

        merged = [b for idx, b in enumerate(merged) if idx not in to_remove]

        # write merged YOLO‐format file (even if merged_px is empty)
        outm = os.path.join(merged_lbls, name + ".txt")
        yo   = []
        img  = cv2.imread(tile_path)
        H, W = img.shape[:2]
        for cx, cy, side in merged:
            xcn, ycn = cx/W, cy/H
            sn       = side/W
            yo.append((0, xcn, ycn, sn, sn))
        save_yolo_label(outm, yo)

    # ——— 3) pick random-k or top‐k tiles per original image ——————————————————————
    selected = os.path.join(output_dataset_dir, "selected")
    if random_tiles == True:
        select_random_k_tiles_per_image(
            inputnames_,
            split_dir=tile_dir,
            label_dir=merged_lbls,
            k=k,
            output_sel=selected
        )    
    else:
        select_top_k_tiles_per_image(
            inputnames_,
            split_dir=tile_dir,
            label_dir=merged_lbls,
            k=k,
            output_sel=selected
        )

    # ——— 4) build train/val split & data.yaml ———————————————————————
    data_yaml = prepare_selected_yolo_dataset(
        selected_root=selected,
        val_split=0.2,
        class_names=["round"]
    )

    return data_yaml


def self_learning_loop(
    inputnames_:       list[str] = None,
    iterations:        int = 3,
    initial_model:     str = None,
    images_dir:        str = "images",
    initial_labels_dir:str = "yolo_dataset_select/labels",
    tile_dir:          str = "images_split",
    k:                 int = 5,
    random_tiles:      bool  = True,
    rows:              int   = 6,
    cols:              int   = 6,
    overlap:           float = 0.15,
    conf_threshold:    float = 0.25,
    iou_threshold:     float = 0.5,
    epochs_per_iter:   int = 50,
    device:            str = "cpu"
):
    # 0) checking if all split tiles exist
    missing = False
    for img_path in inputnames_:
        base = os.path.splitext(os.path.basename(img_path))[0]
        pattern = os.path.join(tile_dir, f"{base}_tile_r*_c*.jpg")
        if not glob.glob(pattern):
            missing = True
            break

    if missing:
        print(f"ℹ️  Some tiles missing in '{tile_dir}', running split_images_into_tiles_with_overlap")
        split_images_into_tiles_with_overlap(
            inputnames_,
            output_folder=tile_dir,
            rows=rows,
            cols=cols,
            overlap=overlap
        )

    # 1) Determine starting weights
    if initial_model:
        model_path = initial_model
    else:
        model_path = get_latest_yolo_weights(project_dir="yolo_output")

    # 2) Build list of base names (img1, img2, …)
    if inputnames_ is None:
        bases = [os.path.splitext(os.path.basename(p))[0]
                 for p in list_image_files(images_dir)]
    else:
        bases = [os.path.splitext(os.path.basename(n))[0]
                 for n in inputnames_]

# iteration 0: base‐model detection counts 
    print("ℹ️  Computing base‐model detections for iteration 0")
    init_eval = os.path.join("yolo_self_learning", "iter0_eval", "new_detect")
    ensure_dir(init_eval)

    # build filenames with their extensions
    full_names = []
    for b in bases:
        for ext in (".jpg",".png",".bmp"):
            fn = b + ext
            if os.path.exists(os.path.join(images_dir, fn)):
                full_names.append(fn)
                break

    # run tiled detection once with the base model
    detect_full_with_tiles(
        full_names,
        model_path=model_path,
        input_dir=images_dir,
        output_dir=init_eval,
        overlap=overlap,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )

    # History for evaluation plotting
    history_counts  = {b: [] for b in bases}
    history_overall = []
    history_confidences   = {b: [] for b in bases}
    history_mean_conf     = []

    # parse raw counts into history
    total0 = 0
    for b in bases:
        raw = os.path.join(init_eval, b, "detections_raw.txt")
        if os.path.exists(raw):
            with open(raw) as f:
                next(f)
                cnt = sum(1 for _ in f)
        else:
            cnt = 0
        history_counts[b].append(cnt)
        total0 += cnt
    history_overall.append(total0)

    # cache for faster “prev” in first iteration’s evaluation
    cached_prev = init_eval

    # record avg‐confidence for iteration 0 
    history_confidences = {b: [] for b in bases}
    history_mean_conf  = []
    sum_conf0 = 0.0
    for b in bases:
        raw_txt = os.path.join(init_eval, b, "detections_raw.txt")
        confs = []
        if os.path.exists(raw_txt):
           with open(raw_txt) as f:
                next(f)  # skip header
                for line in f:
                    *_, conf = line.split()
                    confs.append(float(conf))
        avg_conf0 = float(sum(confs)/len(confs)) if confs else 0.0
        history_confidences[b].append(avg_conf0)
        sum_conf0 += avg_conf0
    history_mean_conf.append(sum_conf0 / len(bases))

    # 3) Iterations
    # for it in range(1, iterations+1):
    # track last iteration’s “new_detect” for reuse
    cached_prev = None
    for it in range(1, iterations+1):
        print(f"\n🔄 Self‐learning iteration {it}/{iterations}")

        # capture the “before” weights
        prev_weights = model_path

        # a) Build new dataset
        ds_dir = f"yolo_self_learning/dataset_iter{it}"
        ensure_dir(ds_dir)
        data_yaml = create_self_learning_dataset(
            inputnames_=bases,
            tile_dir=tile_dir,
            previous_labels_dir=initial_labels_dir if it == 1
                                 else f"yolo_self_learning/dataset_iter{it-1}/selected/labels",
            output_dataset_dir=ds_dir,
            model_path=model_path,  # the “prev” model
            k=k,
            conf_threshold=conf_threshold,
            device=device,
            random_tiles = random_tiles
        )

        # b) Retrain
        proj = f"yolo_self_learning/model_iter{it}"
        new_model_path = train_yolo_model(
            data_yaml=data_yaml,
            model_name=     model_path,      # seed from previous iteration
            epochs=         epochs_per_iter,
            project=        proj,
            name=           f"exp_iter{it}",
            device=         device,
            initial_lr=1e-4,           
            final_lr_factor=0.2,      
            freeze=[0,1,2,3] 
        )
        model_path = new_model_path  # now points to best.pt of this iteration

        # # c) Evaluate “old vs new”
        # evaluate_iteration(
        # c) Evaluate “old vs new”, re-using prior detection when available
        evaluate_iteration(
            iteration=it,
            bases=bases,
            prev_weights=prev_weights,
            new_weights=model_path,
            images_dir=images_dir,
            output_dir=ds_dir,
            history_counts=history_counts,
            history_overall=history_overall,
            history_confidences=history_confidences,
            history_mean_conf=history_mean_conf, 
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            cached_prev_detect=cached_prev,
        )

        # d) Cache this iteration’s new_detect for next time
        cached_prev = os.path.join(ds_dir, f"iter{it}_eval", "new_detect")


    print("\n✅ Self‐learning loop completed.")
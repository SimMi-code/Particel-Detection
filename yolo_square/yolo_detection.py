# yolo_square/yolo_detection.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from .box_utils import snap_xywh_to_square, square_to_xyxy, touches_border
from .scale_bar import detect_scale_bar
from .io_helpers  import get_latest_yolo_weights, ensure_dir


def detect_full_with_tiles(
    inputnames_,
    model_path: str = None,
    input_dir: str = "images",
    output_dir: str = "yolo_predictions/full_tiled",
    tile_size: int = 640,
    overlap: float = 0.2,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    device: str = "cpu"
):
    """
    Runs tiled YOLOv8 over each full image, does NMS, then for each image writes:
      • <output_dir>/<base>/annotated_conf.jpg
      • <output_dir>/<base>/detections_raw.txt   (idx x1 y1 x2 y2 conf)
      • <output_dir>/<base>/conf_data.npz         (prop/post & accepted conf arrays)
    Finally writes global arrays:
      • <output_dir>/global_conf_data.npz
    """
    if model_path is None:
        model_path = get_latest_yolo_weights("yolo_output")
    ensure_dir(output_dir)
    model = YOLO(model_path)
    print(f"🔍 Loaded model from: {model_path}")

    # Global accumulators
    global_pre       = []
    global_post      = []
    global_pre_acc   = []
    global_post_acc  = []

    for img_name in inputnames_:
        img_path = os.path.join(input_dir, img_name)
        image    = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Skipping unreadable: {img_path}")
            continue

        H, W = image.shape[:2]
        step = int(tile_size * (1 - overlap))

        # 1) collect *all* raw proposals (conf=0)
        prop_boxes = []
        prop_confs = []
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                tile = image[y0:y0+tile_size, x0:x0+tile_size]
                th, tw = tile.shape[:2]
                if th < 16 or tw < 16:
                    continue

                preds = model.predict(source=tile, conf=0.0, device=device, verbose=False)[0]
                xyxy  = preds.boxes.xyxy.cpu().numpy()
                confs = preds.boxes.conf.cpu().numpy()

                for (x1,y1,x2,y2), conf in zip(xyxy, confs):
                    cx, cy        = (x1+x2)/2, (y1+y2)/2
                    w_, h_        = x2-x1, y2-y1
                    sq_cx, sq_cy, sq_side = snap_xywh_to_square(cx, cy, w_, h_)
                    if touches_border(sq_cx, sq_cy, sq_side, tile.shape):
                        continue
                    # shift back to global coords
                    sq_cx += x0; sq_cy += y0
                    bx1,by1,bx2,by2 = square_to_xyxy(sq_cx, sq_cy, sq_side)
                    prop_boxes.append([bx1,by1,bx2,by2])
                    prop_confs.append(float(conf))

        prop_boxes = np.array(prop_boxes, dtype=int)
        prop_confs = np.array(prop_confs, dtype=float)

        # Accumulate global
        global_pre.extend(prop_confs.tolist())
        global_pre_acc.extend(prop_confs[prop_confs >= conf_threshold].tolist())

        # 2) NMS on all proposals
        if prop_boxes.size:
            t_boxes = torch.tensor(prop_boxes, dtype=torch.float32)
            t_confs = torch.tensor(prop_confs, dtype=torch.float32)
            keep_all = nms(t_boxes, t_confs, iou_threshold).numpy()
            post_boxes = prop_boxes[keep_all]
            post_confs = prop_confs[keep_all]
        else:
            post_boxes = np.zeros((0,4), dtype=int)
            post_confs = np.zeros((0,), dtype=float)

        global_post.extend(post_confs.tolist())
        global_post_acc.extend(post_confs[post_confs >= conf_threshold].tolist())

        # 3) apply confidence threshold
        mask = post_confs >= conf_threshold
        final_boxes = post_boxes[mask]
        final_confs = post_confs[mask]

        # 4) per‐image output dir
        base   = os.path.splitext(img_name)[0]
        outdir = os.path.join(output_dir, base)
        ensure_dir(outdir)

        # 5) save annotated_conf.jpg
        vis = image.copy()
        for (x1,y1,x2,y2), conf in zip(final_boxes, final_confs):
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        cv2.imwrite(os.path.join(outdir, "annotated_conf.jpg"), vis)

        # 6) write detections_raw.txt
        raw_path = os.path.join(outdir, "detections_raw.txt")
        with open(raw_path, "w") as f:
            f.write("idx x1 y1 x2 y2 conf\n")
            for i, ((x1,y1,x2,y2), conf) in enumerate(zip(final_boxes, final_confs)):
                f.write(f"{i} {x1} {y1} {x2} {y2} {conf:.4f}\n")

        # 7) write conf_data.npz for evaluation
        np.savez(
            os.path.join(outdir, "conf_data.npz"),
            prop_confs=prop_confs,
            post_confs=post_confs,
            pre_acc_confs=prop_confs[prop_confs >= conf_threshold],
            post_acc_confs=post_confs[post_confs >= conf_threshold],
        )

        print(f"✅ [{base}] {len(final_boxes)} detections → {outdir}")

    # 8) write global confidences
    np.savez(
        os.path.join(output_dir, "global_conf_data.npz"),
        prop_confs_all=np.array(global_pre),
        post_confs_all=np.array(global_post),
        pre_acc_confs_all=np.array(global_pre_acc),
        post_acc_confs_all=np.array(global_post_acc),
    )
    print(f"\n✅ Saved global_conf_data.npz → {output_dir}")


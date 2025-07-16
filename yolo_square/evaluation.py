# yolo_square/visualization.py

import os
import glob
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.ops import nms
from .scale_bar import detect_scale_bar
from .box_utils import snap_xywh_to_square, square_to_xyxy
from .io_helpers import ensure_dir
from .yolo_detection import detect_full_with_tiles


def count_gt_labels(gt_label_dir: str, base: str) -> int:
    """
    Count all ground-truth boxes for a given image base across its tiles.
    Expects YOLO-style .txt files named like '{base}_tile_r0_c0.txt' in gt_label_dir.

    Args:
        gt_label_dir: Path to the folder containing your GT .txt label files.
        base:         The image basename (without extension or '_tile...' suffix).

    Returns:
        Total number of non-empty lines (i.e. boxes) across all matching files.
    """
    pattern = os.path.join(gt_label_dir, f"{base}_*.txt")
    total = 0
    for label_path in glob.glob(pattern):
        with open(label_path, "r") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total

def evaluate_detections(
    output_dir: str,
    original_dir: str = "images",
    selected_labels_dir: str = "yolo_dataset_select/labels",
    gt_label_dir: str = "yolo_dataset/labels",
    rewrite_eval: bool = True
):
    """
    Reads each `<output_dir>/<base>/detections_raw.txt`,
    (re)computes µm/px from the original image’s scale bar,
    writes/overwrites `<base>/detections_eval.txt` with diameters
      if rewrite_eval=True or file missing,
    otherwise reloads existing eval file.
    Produces per-image:
      • annotated_diam.jpg
      • diam_hist.png
      • performance.png    (#GT vs #det + avg_conf + ratio)
    And overall:
      • overall_counts.png      (bar chart GT vs detections)
      • overall_conf_ratio.png  (avg_conf and det/GT ratio)
    """
    bases = []
    train_counts = []
    det_counts = []
    avg_confs = []
    ratios = []

    for base in sorted(os.listdir(output_dir)):
        sub = os.path.join(output_dir, base)
        raw_txt = os.path.join(sub, "detections_raw.txt")
        if not os.path.isdir(sub) or not os.path.exists(raw_txt):
            continue

        # locate original image
        for ext in (".jpg", ".png", ".bmp"):
            orig = os.path.join(original_dir, base + ext)
            if os.path.exists(orig):
                break
        else:
            print(f"⚠️ Original image for {base} not found.")
            continue

        # compute µm/px
        um_per_px = 1.0
        sb = detect_scale_bar(orig)
        if sb[0] is not None:
            um_per_px, _ = sb
        else:
            print (f"Unsuccessful scale detection: default {um_per_px} µm/px")

        # read raw detections
        dets = []
        with open(raw_txt) as f:
            next(f)  # skip header
            for line in f:
                idx, x1, y1, x2, y2, conf = line.split()
                dets.append((
                    int(idx),
                    int(x1), int(y1), int(x2), int(y2),
                    float(conf)
                ))

        if not dets:
            print(f"⚠️ No raw detections for {base}")
            continue

        # (re)write or reload eval file
        eval_txt = os.path.join(sub, "detections_eval.txt")
        dets_eval = []
        if rewrite_eval or not os.path.exists(eval_txt):
            with open(eval_txt, "w") as f:
                f.write("idx x1 y1 x2 y2 conf diameter_um\n")
                for idx, x1, y1, x2, y2, conf in dets:
                    diam = (x2 - x1) * um_per_px
                    f.write(f"{idx} {x1} {y1} {x2} {y2} {conf:.4f} {diam:.2f}\n")
                    dets_eval.append((idx, x1, y1, x2, y2, conf, diam))
            print (f"rewriting {eval_txt}")
        else:
            with open(eval_txt) as f:
                next(f)
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 7:
                        continue
                    idx, x1, y1, x2, y2, conf, diam = parts
                    dets_eval.append((
                        int(idx),
                        int(x1), int(y1), int(x2), int(y2),
                        float(conf),
                        float(diam)
                    ))

        # annotate diameter image
        orig_img = cv2.imread(orig)
        vis = orig_img.copy()
        for idx, x1, y1, x2, y2, conf, diam in dets_eval:
            # box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            side = x2 - x1
            # idx on top
            txt = str(idx)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            tx = x1 + (side - tw)//2
            ty = y1 - 5
            cv2.putText(vis, txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
            # diameter at bottom
            txt = f"{diam:.2f}"
            (dw, dh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            dx = x1 + (side - dw)//2
            dy = y2 + dh + 5
            cv2.putText(vis, txt, (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(sub, "annotated_diam.jpg"), vis)

        # diameter histogram
        diams = np.array([diam for *_, diam in dets_eval])
        m = diams.mean() if diams.size else 0.0
        s = diams.std()  if diams.size else 0.0
        plt.figure(figsize=(6,4))
        plt.hist(diams, bins=20, edgecolor='black')
        plt.title(f"{base}   mean={m:.2f} um, σ={s:.2f} um")
        plt.xlabel("Diameter (um)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(sub, "diam_hist.png"))
        plt.close()
        print(f"📊 [{base}] Evaluated {len(dets_eval)} objects → diam_hist.png")

        # count GT labels for this image
        lbls_train = glob.glob(os.path.join(selected_labels_dir, "train", f"{base}_tile*.txt"))
        lbls_val   = glob.glob(os.path.join(selected_labels_dir, "val",   f"{base}_tile*.txt"))
        gt_count   = sum(1 for p in lbls_train+lbls_val for _ in open(p)) if (lbls_train+lbls_val) else 0

        det_count = len(dets_eval)
        avg_conf  = np.mean([conf for *_, conf, _ in dets_eval]) if dets_eval else 0.0
        ratio     = det_count / gt_count if gt_count else np.nan

        print(f"🔍 [{base}] GT={gt_count}, det={det_count}, avg_conf={avg_conf:.3f}, ratio={ratio:.3f}")

        # accumulate for overall
        bases.append(base)
        train_counts.append(gt_count)
        det_counts.append(det_count)
        avg_confs.append(avg_conf)
        ratios.append(ratio)

        # per-image performance plot
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.bar([0,0.5], [gt_count, det_count], width=0.4, color=["lightgray","skyblue"], tick_label=["GT","Det"])
        ax1.set_ylabel("Count")
        ax2 = ax1.twinx()
        ax2.plot([0.25], [avg_conf], "r*", label="Avg Conf")
        ax2.set_ylabel("Avg Confidence")
        for sp in ("top","right"):
            ax2.spines[sp].set_visible(False)
        plt.title(f"{base} performance\nratio={ratio:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(sub, "performance.png"))
        plt.close()

    # overall counts bar chart
    x = np.arange(len(bases))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x - w/2, train_counts,  width=w, label="GT labels", color="lightgray")
    ax.bar(x + w/2, det_counts,    width=w, label="Detections",  color="skyblue")
    ax.set_xticks(x); ax.set_xticklabels(bases, rotation=45, ha="right")
    ax.set_ylabel("Count"); ax.set_title("Ground‐Truth vs. Detections per Image")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_counts.png"))
    plt.close()
    print(f"✅ Saved overall counts → overall_counts.png")

    # overall confidence & ratio plot
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(x, avg_confs, marker="o", label="Avg confidence")
    ax1.set_xticks(x); ax1.set_xticklabels(bases, rotation=45, ha="right")
    ax1.set_ylabel("Average confidence"); ax1.set_title("Confidence & Det/GT Ratio per Image")
    ax2 = ax1.twinx()
    ax2.plot(x, ratios, marker="s", color="orange", label="Det/GT ratio")
    ax2.set_ylabel("Detection/GT ratio")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_conf_ratio.png"))
    plt.close()
    print(f"✅ Saved overall confidence & ratio → overall_conf_ratio.png")

def _compute_iou(boxA, boxB):
    """
    Compute IoU between two boxes in [x1,y1,x2,y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return 0.0 if union == 0 else inter / union


def evaluate_iteration(
    iteration: int,
    bases: list[str],
    prev_weights: str,
    new_weights: str,
    images_dir: str,
    device: str,
    history_counts: dict[str, list[int]],
    history_overall: list[int],
    history_confidences: dict[str, list[float]],
    history_mean_conf: list[float],    
    output_dir: str,
    tile_size: int = 640,
    overlap: float = 0.2,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    cached_prev_detect: str = None,
):
    """
    After iteration N, run `detect_full_with_tiles` on all `bases` images
    with both the previous and the newly trained weights, parse their
    detections_raw.txt, find which boxes are truly new (IoU≤0.5), and:

      • Write a composite image for each base (blue=old, green=new, red=added).
      • Update history_counts[base].append(n_new) and history_overall.
      • Plot per-image & total‐detections vs iteration.

    This never passes an ndarray to `detect_full_with_tiles`, only filenames.
    """

    # 1) prepare eval folders
    eval_dir = os.path.join(output_dir, f"iter{iteration}_eval")
    prev_dir = os.path.join(eval_dir, "prev_detect")
    new_dir  = os.path.join(eval_dir, "new_detect")
    ensure_dir(prev_dir)
    ensure_dir(new_dir)

    # 2) build list of actual filenames and remember each base's extension
    full_names = []
    ext_map    = {}
    for base in bases:
        for ext in (".jpg", ".png", ".bmp"):
            fn = base + ext
            if os.path.exists(os.path.join(images_dir, fn)):
                full_names.append(fn)
                ext_map[base] = ext
                break
        else:
            print(f"⚠️  [iter {iteration}] Image for {base} not found, skipping.")

    # # 3) run detection with the old weights
    # detect_full_with_tiles(
    #     full_names,
    #     model_path=prev_weights,
    #     input_dir=images_dir,
    #     output_dir=prev_dir,
    #     tile_size=tile_size,
    #     overlap=overlap,
    #     conf_threshold=conf_threshold,
    #     iou_threshold=iou_threshold,
    #     device=device,
    # )
    # 3) either copy cached prev results, or run detection
    if cached_prev_detect and os.path.isdir(cached_prev_detect):
        shutil.copytree(cached_prev_detect, prev_dir, dirs_exist_ok=True)
    else:
        detect_full_with_tiles(
            full_names,
            model_path=prev_weights,
            input_dir=images_dir,
            output_dir=prev_dir,
            tile_size=tile_size,
            overlap=overlap,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )

    # 4) run detection with the new weights
    detect_full_with_tiles(
        full_names,
        model_path=new_weights,
        input_dir=images_dir,
        output_dir=new_dir,
        tile_size=tile_size,
        overlap=overlap,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )

    counts_this = []

    # 5) for each image, parse detections_raw.txt from both runs
    for base in bases:
        prev_txt = os.path.join(prev_dir, base, "detections_raw.txt")
        prev_boxes = []
        if os.path.exists(prev_txt):
            with open(prev_txt) as f:
                next(f)  # skip header
                for line in f:
                    _, x1, y1, x2, y2, _ = line.split()
                    prev_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        new_txt = os.path.join(new_dir, base, "detections_raw.txt")
        new_boxes = []
        new_confs = []
        if os.path.exists(new_txt):
            with open(new_txt) as f:
                next(f)
                for line in f:
                    _, x1, y1, x2, y2, conf = line.split()
                    new_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    new_confs.append(float(conf))

        n_new     = len(new_boxes)
        avg_conf  = float(np.mean(new_confs)) if new_confs else 0.0

        history_counts.setdefault(base, []).append(n_new)
        history_confidences.setdefault(base, []).append(avg_conf)
        counts_this.append(n_new)

        # find truly *added* boxes (IoU ≤ 0.5 with every old box)
        added = [
            nb for nb in new_boxes
            if all(_compute_iou(nb, ob) <= 0.5 for ob in prev_boxes)
        ]

        # draw composite: blue=old, green=new, red=added
        img = cv2.imread(os.path.join(images_dir, base + ext_map.get(base, "")))
        vis = img.copy()
        for x1,y1,x2,y2 in prev_boxes:
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255, 0,   0), 2)
        for x1,y1,x2,y2 in new_boxes:
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,   255, 0), 2)
        for x1,y1,x2,y2 in added:
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,   0, 255), 2)

        out_img = os.path.join(eval_dir, f"{base}_iter{iteration}_compare.jpg")
        cv2.imwrite(out_img, vis)
        print(f"🔍 [iter {iteration}] {base}: {n_new} dets, +{len(added)} new → {out_img}")

    # 6) update overall
    total_this = sum(counts_this)
    history_overall.append(total_this)

    # overall mean confidence this iteration
    mean_conf_this = float(
        sum(history_confidences[b][-1] for b in bases) / len(bases)
    )
    history_mean_conf.append(mean_conf_this)

    # 7) plot per‐image counts vs iteration
    its = list(range(0, iteration + 1))
    plt.figure(figsize=(10, 6))
    for base in bases:
        plt.plot(its, history_counts[base], marker="o", label=base)
    plt.xlabel("Iteration")
    plt.ylabel("Detections")
    plt.title("Detections per Image over Iterations")
    plt.legend()
    plt.tight_layout()
    fn = os.path.join(eval_dir, f"iter{iteration}_per_image_counts.png")
    plt.savefig(fn)
    plt.close()
    print(f"✅ Saved per-image chart → {fn}")

    # 8) plot total detections vs iteration
    plt.figure(figsize=(6, 4))
    plt.plot(its, history_overall, marker="s", color="black")
    plt.xlabel("Iteration")
    plt.ylabel("Total Detections")
    plt.title("Total Detections over Iterations")
    plt.tight_layout()
    fn = os.path.join(eval_dir, f"iter{iteration}_total_counts.png")
    plt.savefig(fn)
    plt.close()
    print(f"✅ Saved total‐detections chart → {fn}")

    # 9) plot average confidence per image & mean vs iteration
    its = list(range(0, iteration + 1))  # or range(1, iteration+1) to match your counts
    plt.figure(figsize=(10, 6))
    for base in bases:
        plt.plot(its, history_confidences[base], marker="o", label=base)
    plt.plot(its, history_mean_conf, marker="s", linewidth=2,
             label="Mean Avg Confidence", color="black")
    plt.xlabel("Iteration")
    plt.ylabel("Average Confidence")
    plt.title("Average Confidence per Image over Iterations")
    plt.legend()
    plt.tight_layout()
    fn = os.path.join(eval_dir, f"iter{iteration}_avg_confidences.png")
    plt.savefig(fn)
    plt.close()
    print(f"✅ Saved avg‐confidence chart → {fn}")

# yolo_square/visualization.py

import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

#from .scale_bar import detect_scale_bar


def visualize_yolo_labels(image_dir, label_dir, output_dir, class_names=["round"], max_samples=20):
    """
    Overlays YOLO-format labels onto images and saves visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".bmp"))])

    for i, img_file in enumerate(img_files[:max_samples]):
        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        image = cv2.imread(img_path)
        if image is None or not os.path.exists(label_path):
            continue

        h, w = image.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_c, y_c, bw, bh = map(float, parts)
                x_c *= w
                y_c *= h
                bw *= w
                bh *= h

                # For square assumption
                side = max(bw, bh)
                x1 = int(x_c - side / 2)
                y1 = int(y_c - side / 2)
                x2 = int(x_c + side / 2)
                y2 = int(y_c + side / 2)

                color = (255, 0, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = class_names[int(cls)] if int(cls) < len(class_names) else f"class_{int(cls)}"
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(out_path, image)
        print(f"🖼️ Saved label visualization: {out_path}")


def plot_confidence_histogram(detections, save_path=None):
    """
    Plot histogram of confidence values from detection results.
    Each entry in detections is a tuple: (filename, confidence, diameter_um)
    """
    if not detections:
        print("⚠️ No detections to plot.")
        return

    confidences = [d[1] for d in detections]

    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Confidence Histogram of Detections")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Confidence histogram saved to {save_path}")
    else:
        plt.show()

def update_yolo_viz(
    dataset_dir,
    image_dir=None,
    label_dir=None,
    viz_dir=None,
    class_colors=None
):
    """
    Redraws YOLO‐format square labels onto their images and writes updated visualizations.

    Args:
        dataset_dir (str): root of your YOLO dataset.
        image_dir (str, optional): folder containing source images; defaults to dataset_dir/images.
        label_dir (str, optional): folder containing .txt labels; defaults to dataset_dir/labels.
        viz_dir   (str, optional): where to write annotated images; defaults to dataset_dir/viz.
        class_colors (dict): mapping from class_id to BGR color, e.g. {0:(0,255,0)}.
    """
    if image_dir is None:
        image_dir = os.path.join(dataset_dir, "images")
    if label_dir is None:
        label_dir = os.path.join(dataset_dir, "labels")
    if viz_dir is None:
        viz_dir = os.path.join(dataset_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # default color for class 0
    if class_colors is None:
        class_colors = {0: (0, 255, 0)}

    # for every label file
    for lbl_path in glob.glob(os.path.join(label_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(lbl_path))[0]

        # find matching image (jpg/png/bmp)
        img_path = None
        for ext in (".jpg", ".png", ".bmp"):
            p = os.path.join(image_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            print(f"⚠️  No image for {base}, skipping viz")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Failed to read {img_path}, skipping viz")
            continue
        H, W = img.shape[:2]

        # draw each square label
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xcn, ycn, wn, hn = map(float, parts)
                # assume wn==hn
                side = wn * W
                cx = xcn * W
                cy = ycn * H
                half = side / 2.0
                x1 = int(cx - half)
                y1 = int(cy - half)
                x2 = int(cx + half)
                y2 = int(cy + half)
                color = class_colors.get(int(cls_id), (0, 0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        out_path = os.path.join(viz_dir, base + "_viz.jpg")
        cv2.imwrite(out_path, img)
        print(f"✅ Wrote viz: {out_path}")


def compare_ground_truth_vs_pred(
    image_path,
    gt_txt,
    pred_txt,
    out_path,
    class_colors=None
):
    """
    Draw ground‐truth (green) and prediction (red) boxes onto full image.

    Args:
        image_path (str): path to image
        gt_txt (str): path to YOLO .txt ground‐truth
        pred_txt (str): path to YOLO .txt predictions
        out_path (str): where to save combined visualization
        class_colors (dict): override colors, keys 'gt' and 'pred'
    """
    if class_colors is None:
        class_colors = {"gt": (0, 255, 0), "pred": (0, 0, 255)}

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    H, W = img.shape[:2]

    def draw(txt, col):
        if not os.path.exists(txt):
            return
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xcn, ycn, wn, hn = map(float, parts)
                side = wn * W
                cx = xcn * W
                cy = ycn * H
                half = side / 2.0
                x1 = int(cx - half)
                y1 = int(cy - half)
                x2 = int(cx + half)
                y2 = int(cy + half)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

    draw(gt_txt, class_colors["gt"])
    draw(pred_txt, class_colors["pred"])

    cv2.imwrite(out_path, img)
    print(f"✅ Saved comparison: {out_path}")


def evaluate_detections(
    output_dir: str,
    original_dir: str = "images"
):
    """
    Reads each `<output_dir>/<base>/detections_raw.txt`,
    re-computes µm/px from the original image’s scale bar,
    writes `<base>/detections_eval.txt` with diameters,
    and produces:
      • annotated_diam.jpg   ← boxes + “idx:diam” under each
      • diam_hist.png        ← histogram (with “um” in labels)
    """
    for base in sorted(os.listdir(output_dir)):
        sub = os.path.join(output_dir, base)
        raw = os.path.join(sub, "detections_raw.txt")
        if not os.path.isdir(sub) or not os.path.exists(raw):
            continue

        # locate original image
        # assume same basename with any common extension
        for ext in (".jpg",".png",".bmp"):
            orig = os.path.join(original_dir, base + ext)
            if os.path.exists(orig):
                break
        else:
            print(f"⚠️  Original image for {base} not found.")
            continue

        # 1) detect scale bar once
        µm_per_px = 1.0
        sb = detect_scale_bar(orig)
        if sb is not None:
            µm_per_px, _ = sb

        # 2) read raw detections
        dets = []
        with open(raw) as f:
            next(f)  # skip header
            for line in f:
                idx, x1,y1,x2,y2, conf = line.split()
                dets.append((
                    int(idx),
                    int(x1), int(y1), int(x2), int(y2),
                    float(conf)
                ))
        if not dets:
            print(f"⚠️  No raw detections for {base}")
            continue

        # 3) compute diam and write eval file (or reload existing)
        eval_path = os.path.join(sub, "detections_eval.txt")
        dets_eval = []  # will hold (idx, x1, y1, x2, y2, conf, diam)

        if not os.path.exists(eval_path):
            # first time: compute diam, write file, collect into dets_eval
            with open(eval_path, "w") as f:
                f.write("idx x1 y1 x2 y2 conf diameter_um\n")
                for idx, x1, y1, x2, y2, conf in dets:
                    side = x2 - x1
                    diam = side * µm_per_px
                    f.write(f"{idx} {x1} {y1} {x2} {y2} {conf:.4f} {diam:.2f}\n")
                    dets_eval.append((idx, x1, y1, x2, y2, conf, diam))
        else:
            # reload stored diameters and confidences
            with open(eval_path, "r") as f:
                next(f)  # skip header
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

        # 4) annotated_diam.jpg
        orig_img = cv2.imread(orig)
        vis = orig_img.copy()
        for idx, x1,y1,x2,y2, conf in dets:
            side = x2 - x1
            diam = side * µm_per_px

            # draw square
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)

            # index at top-center of box
            text_idx = str(idx)
            ((tw, th), _) = cv2.getTextSize(text_idx, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            tx = x1 + (side - tw) // 2
            ty = y1 - 5
            cv2.putText(vis, text_idx, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

            # diameter at bottom-center of box
            text_d = f"{diam:.2f}"
            ((dw, dh), _) = cv2.getTextSize(text_d, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            dx = x1 + (side - dw) // 2
            dy = y2 + dh + 5
            cv2.putText(vis, text_d, (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(sub, "annotated_diam.jpg"), vis)

        # 5) histogram
        diams = np.array([ (x2-x1)*µm_per_px for _,x1,y1,x2,y2,_ in dets ])
        plt.figure(figsize=(6,4))
        plt.hist(diams, bins=20, edgecolor='black')
        m, s = diams.mean(), diams.std()
        plt.title(f"{base}   mean={m:.2f} um, σ={s:.2f} um")
        plt.xlabel("Diameter (um)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(sub, "diam_hist.png"))
        plt.close()

        print(f"📊 [{base}] Evaluated {len(dets)} objects → diam_hist.png")


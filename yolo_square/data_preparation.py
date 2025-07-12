# yolo_square/data_preparation.py

import os
import glob
import shutil
import cv2
import yaml
import re
import numpy as np
from sklearn.model_selection import train_test_split
from .io_helpers import list_image_files
from yolo_square.box_utils import (
    contour_to_square,
    square_to_xyxy,
    square_to_yolo
)
from .visualization import update_yolo_viz

def detect_contours_canny(image, canny_min=50, canny_max=150, blur_kernel=(5,5)):
    """
    Simple Canny‐based contour detector.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges   = cv2.Canny(blurred, canny_min, canny_max)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def compute_circularity(contour):
    """
    4π·area / perimeter²
    """
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    return 0 if peri == 0 else 4 * np.pi * area / (peri * peri)

def auto_tune_parameters(image, circularity_min=0.7, circularity_max=1.2):
    """
    Grid‐search over blur kernels & Canny thresholds to maximize
      (# round‐like contours) + 0.1·(sum circularity) − 0.01·(total contours).
    Returns: (best_blur_kernel, best_canny_min, best_canny_max)
    """
    blur_options  = [(5,5),(7,7),(9,9)]
    canny_options = [(20,60),(30,80),(50,120),(80,160)]
    best_score = -np.inf
    best_cfg   = (blur_options[0],) + canny_options[0]

    for bk in blur_options:
        for cmin, cmax in canny_options:
            cnts = detect_contours_canny(image, cmin, cmax, bk)
            rc = 0; cs = 0
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                if peri==0 or area<20:
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                circ2 = area / (np.pi * r * r)
                if circularity_min < circ2 < circularity_max:
                    rc += 1
                    cs += circ2
            score = rc + 0.1*cs - 0.01*len(cnts)
            if score > best_score:
                best_score = score
                best_cfg   = (bk, cmin, cmax)

    bk, cmin, cmax = best_cfg
    print(f"🧠 Auto‐tune → blur={bk}, canny=({cmin},{cmax})")
    return bk, cmin, cmax

def split_images_into_tiles_with_overlap(
    inputnames_,
    output_folder="images_split",
    rows=4,
    cols=4,
    overlap=0.2
):
    """
    Split each image into `rows × cols` tiles that overlap by `overlap` fraction.
    Tiles are arranged in a sliding‐window grid of size rows×cols,
    but shifted so that every row/column covers the full image.
    """
    os.makedirs(output_folder, exist_ok=True)

    for fname in inputnames_:
        img_path = os.path.join("images", fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Cannot read {fname}, skipping")
            continue

        H, W = img.shape[:2]

        # Compute nominal tile size
        tile_h = H / rows
        tile_w = W / cols
        # Compute step between tile top‐left corners
        step_h = int(tile_h * (1 - overlap))
        step_w = int(tile_w * (1 - overlap))
        # Ensure we cover the entire image: we'll clamp at the bottom/right edges
        tile_h = int(tile_h)
        tile_w = int(tile_w)

        base = os.path.splitext(fname)[0]
        count = 0

        for r in range(rows):
            for c in range(cols):
                # nominal top‐left
                y1 = r * step_h
                x1 = c * step_w
                # clamp so tile fits within image
                if y1 + tile_h > H:
                    y1 = H - tile_h
                if x1 + tile_w > W:
                    x1 = W - tile_w

                y1, x1 = int(max(0, y1)), int(max(0, x1))
                y2 = y1 + tile_h
                x2 = x1 + tile_w

                tile = img[y1:y2, x1:x2]
                out_name = f"{base}_tile_r{r}_c{c}.jpg"
                cv2.imwrite(os.path.join(output_folder, out_name), tile)
                count += 1

        print(f"✅ {fname} → {count} tiles ({rows}×{cols}, overlap={overlap})")

    print(f"\n🎉 All images split → {output_folder}")

def create_yolo_dataset_square(
    inputnames_,
    image_dir="images_split",
    dataset_dir="yolo_dataset",
    class_names=["round"],
    circularity_min=0.7,
    circularity_max=1.2,
    auto_tune=True
):
    """
    Heuristic YOLO‐style dataset creation using *square* bboxes.
    Writes:
      {dataset_dir}/labels/*.txt   ← YOLO .txt files
      {dataset_dir}/viz/*.jpg      ← visualization overlays
    """
    lbl_dir = os.path.join(dataset_dir, "labels")
    viz_dir = os.path.join(dataset_dir, "viz")
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    for fname in inputnames_:
        img_path = os.path.join(image_dir, fname)
        img      = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping {fname} (cannot read)")
            continue
        H, W  = img.shape[:2]
        # choose parameters
        if auto_tune:
            blur_k, cmin, cmax = auto_tune_parameters(img, circularity_min, circularity_max)
        else:
            blur_k, cmin, cmax = (5,5), 50, 150

        contours = detect_contours_canny(img, cmin, cmax, blur_k)
        labels   = []
        vis      = img.copy()
        base     = os.path.splitext(fname)[0]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri==0 or area<20:
                continue
            # second circularity check
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            circ2 = area / (np.pi * r * r)
            if not (circularity_min < circ2 < circularity_max):
                continue

            # get square
            cx, cy, side = contour_to_square(cnt)
            if side < 10:
                continue

            # YOLO label line
            yolo_line = square_to_yolo(cx, cy, side, img.shape)
            labels.append(yolo_line)

            # draw viz
            x1, y1, x2, y2 = square_to_xyxy(cx, cy, side)
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 2)

        # save labels
        if labels:
            lbl_file = os.path.join(lbl_dir, base + ".txt")
            with open(lbl_file, "w") as f:
                f.write("\n".join(labels))

        # save visualization
        viz_file = os.path.join(viz_dir, base + "_viz.jpg")
        cv2.imwrite(viz_file, vis)
        print(f"✅ {fname}: wrote {len(labels)} labels → {lbl_dir}, viz → {viz_file}")

def select_top_k_tiles_per_image(
    inputnames_,
    split_dir="images_split",
    label_dir="yolo_dataset/labels",
    k=5,
    output_sel="yolo_dataset_select"
):
    """
    For each base image in `inputnames_`, finds its tile-labels in `label_dir` matching
    "{base}_tile_r*_c*.txt", picks the top k by count, and copies both the .jpg tile
    and its .txt label into `output_sel/{images,labels}`.

    BEFORE doing so, if `output_sel` already exists, it is versioned:
    - yolo_dataset_select → yolo_dataset_select_v1 (or v2, v3, …)
    - then the old `output_sel` is removed.
    """

    # 1) Version the existing output_sel, if present
    if os.path.isdir(output_sel):
        parent, name = os.path.split(output_sel)
        # find siblings matching name_vN
        pattern = re.compile(re.escape(name) + r"_v(\d+)$")
        existing = []
        for entry in os.listdir(parent or "."):
            m = pattern.match(entry)
            if m:
                existing.append(int(m.group(1)))
        next_v = max(existing, default=0) + 1
        backup = f"{output_sel}_v{next_v}"
        print(f"📦 Backing up '{output_sel}' → '{backup}'")
        shutil.copytree(output_sel, backup)
        shutil.rmtree(output_sel)
        print("✅ Backup complete, old selection cleared.")

    # 2) Prepare fresh directories
    img_out = os.path.join(output_sel, "images")
    lbl_out = os.path.join(output_sel, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # 3) For each original image, pick its k tiles
    for inputname in inputnames_:
        base = os.path.splitext(inputname)[0]
        # gather all matching label files
        pattern = os.path.join(label_dir, f"{base}_tile_r*_c*.txt")
        all_lbls = sorted(glob.glob(pattern))
        if not all_lbls:
            print(f"⚠️  No tile-labels found for '{base}' in {label_dir}")
            continue

        # count lines in each label
        tile_counts = []
        for lp in all_lbls:
            with open(lp) as f:
                cnt = sum(1 for _ in f)
            tile_name = os.path.splitext(os.path.basename(lp))[0]
            tile_counts.append((tile_name, cnt))

        # sort descending by count
        tile_counts.sort(key=lambda x: x[1], reverse=True)

        # copy k
        for tile_name, cnt in tile_counts[:k]:
            src_img = os.path.join(split_dir, tile_name + ".jpg")
            src_lbl = os.path.join(label_dir,   tile_name + ".txt")
            dst_img = os.path.join(img_out,      tile_name + ".jpg")
            dst_lbl = os.path.join(lbl_out,      tile_name + ".txt")
            if os.path.exists(src_img) and os.path.exists(src_lbl):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_lbl, dst_lbl)
        print(f"✅ {base}: found {len(tile_counts)} tiles, selected top {min(k, len(tile_counts))}")

    print(f"\n🎉 Finished selecting top‐{k} tiles per image into: {output_sel}")


def prepare_selected_yolo_dataset(
    selected_root: str,
    val_split: float = 0.2,
    class_names: list = ["round"]
) -> str:
    """
    Given a folder containing:
      selected_root/images/*.jpg
      selected_root/labels/*.txt
    1) Prints per-original-image & overall label statistics
    2) Splits into train/val under:
         selected_root/images/train, images/val,
         selected_root/labels/train, labels/val
    3) Writes selected_root/data.yaml and returns its path.
    """
    img_src = os.path.join(selected_root, "images")
    lbl_src = os.path.join(selected_root, "labels")
    # sanity check
    if not os.path.isdir(img_src) or not os.path.isdir(lbl_src):
        raise FileNotFoundError(f"Expecting {img_src} and {lbl_src} to exist")

    # 1) gather all image basenames (without extension)
    img_paths = list_image_files(img_src)
    bases = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    if not bases:
        raise RuntimeError(f"No images found in {img_src}")

    # --- NEW BLOCK: compute per-original-image & overall stats ---
    # group tiles by original image name (prefix before '_tile')
    stats = {}
    for b in bases:
        orig = b.split("_tile", 1)[0]
        lbl_file = os.path.join(lbl_src, b + ".txt")
        count = 0
        if os.path.exists(lbl_file):
            with open(lbl_file) as f:
                count = sum(1 for _ in f)
        stats.setdefault(orig, []).append(count)

    total_tiles  = 0
    total_labels = 0
    print("\n🔍 Selected‐tile label counts by original image:")
    for orig, counts in stats.items():
        n_tiles = len(counts)
        sum_lbl = sum(counts)
        avg_lbl = sum_lbl / n_tiles if n_tiles else 0
        print(f"  • {orig}: {sum_lbl} labels over {n_tiles} tiles  (avg {avg_lbl:.2f}/tile)")
        total_tiles  += n_tiles
        total_labels += sum_lbl

    overall_avg = total_labels / total_tiles if total_tiles else 0
    print("\n=== Selected dataset summary ===")
    print(f"  • Original images: {len(stats)}")
    print(f"  • Total tiles:     {total_tiles}")
    print(f"  • Total labels:    {total_labels}")
    print(f"  • Avg labels/tile: {overall_avg:.2f}\n")

    # 2) clear out any old train/val splits
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        path = os.path.join(selected_root, sub)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # 3) split basenames
    train_b, val_b = train_test_split(bases, test_size=val_split, random_state=42)
    print(f"🔀 Splitting {len(bases)} tiles → {len(train_b)} train / {len(val_b)} val")

    # 4) copy into the proper folders
    def _copy(subset, split_name):
        for b in subset:
            # image: try any extension
            for ext in (".jpg", ".png", ".bmp"):
                src = os.path.join(img_src, b + ext)
                if os.path.exists(src):
                    dst = os.path.join(selected_root, "images", split_name, b + ".jpg")
                    shutil.copy(src, dst)
                    break
            # label
            src_lbl = os.path.join(lbl_src, b + ".txt")
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(selected_root, "labels", split_name, b + ".txt")
                shutil.copy(src_lbl, dst_lbl)

    _copy(train_b, "train")
    _copy(val_b,   "val")

    # 5) write data.yaml
    data = {
        "train": os.path.abspath(os.path.join(selected_root, "images/train")),
        "val":   os.path.abspath(os.path.join(selected_root, "images/val")),
        "nc":    len(class_names),
        "names": class_names
    }
    out_yaml = os.path.join(selected_root, "data.yaml")
    with open(out_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"\n✅ Prepared YOLO dataset at “{selected_root}”")
    print(f"   • train: {len(train_b)}, val: {len(val_b)}")
    print(f"   • data.yaml → {out_yaml}\n")

    return out_yaml
    return out_yaml

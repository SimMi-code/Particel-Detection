# yolo_square/io_helpers.py

import os
import glob
import cv2
import re

def ensure_dir(path):
    """
    Ensure that a directory exists (create it if necessary).
    """
    os.makedirs(path, exist_ok=True)

def list_images_in_folder(folder, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Return a sorted list of image *filenames* (basename only) in `folder`
    matching any of the provided extensions.
    """
    files = []
    for ext in extensions:
        pattern = os.path.join(folder, f"*{ext}")
        files.extend(glob.glob(pattern))
    # Return just the basenames, sorted
    return sorted(os.path.basename(f) for f in files)

# Alias to match older import in your code
get_image_filenames = list_images_in_folder

def list_image_files(folder, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Return a sorted list of full filepaths to images in `folder`
    matching any of the provided extensions.
    """
    files = []
    for ext in extensions:
        pattern = os.path.join(folder, f"*{ext}")
        files.extend(glob.glob(pattern))
    # return only the filename portion, so downstream code
    # can safely do os.path.join(input_dir, basename)
    return sorted(os.path.basename(p) for p in files)

def load_image(path):
    """
    Load an image from disk (or return None).
    """
    return cv2.imread(path)

def image_basename(path):
    """
    Return the basename (without extension) of a filepath.
    """
    return os.path.splitext(os.path.basename(path))[0]

def match_label_file(image_file, label_dir):
    """
    Given an image filename, return the corresponding .txt in label_dir
    if it exists, else None.
    """
    base = image_basename(image_file)
    candidate = os.path.join(label_dir, base + ".txt")
    return candidate if os.path.exists(candidate) else None

def save_yolo_label(label_path, boxes):
    """
    Save YOLO‐style labels to disk.  Supports two formats of `boxes`:
      - 4‐tuple: (xc, yc, w, h)  → writes "0 xc yc w h"
      - 5‐tuple: (idx, xc, yc, w, h) → writes "idx xc yc w h"
    All coords must already be normalized [0..1].
    """
    with open(label_path, "w") as f:
        for box in boxes:
            if len(box) == 5:
                cls, xc, yc, w, h = box
            elif len(box) == 4:
                cls = 0
                xc, yc, w, h = box
            else:
                raise ValueError(f"save_yolo_label: expected box of length 4 or 5, got {len(box)}")
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def load_yolo_label(label_path):
    """
    Read YOLO‐format boxes from disk. Returns list of tuples
    (class_id, x_center, y_center, w, h).
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, w, h = parts
                boxes.append((int(cls),
                              float(xc),
                              float(yc),
                              float(w),
                              float(h)))
    return boxes

def get_latest_yolo_weights(project_dir="yolo_output"):
    """
    Scan project_dir for subfolders named 'exp', 'exp2', 'exp3', …,
    pick the one with the highest number, and return its weights/best.pt path.
    """
    exps = []
    for name in os.listdir(project_dir):
        full = os.path.join(project_dir, name)
        if not os.path.isdir(full):
            continue
        m = re.fullmatch(r"exp(?:([0-9]+))?", name)
        if m:
            idx = int(m.group(1)) if m.group(1) else 1
            exps.append((idx, name))
    if not exps:
        raise FileNotFoundError(f"No exp* folders found in {project_dir}")
    _, latest = max(exps, key=lambda x: x[0])
    best = os.path.join(project_dir, latest, "weights", "best.pt")
    if not os.path.exists(best):
        raise FileNotFoundError(f"Couldn’t find best.pt in {latest}/weights")
    print(f"🔍 Using weights: {latest}/weights/best.pt")
    return best

def expand_to_tiles_for_images(
    image_files: list[str],
    split_dir: str,
    exts: tuple = (".jpg", ".png", ".bmp")
) -> list[str]:
    """
    Given a list of original image filenames (e.g. ["img1.bmp","img2.bmp"]),
    finds all tiles in `split_dir` named like "img1_tile_*.jpg" (or .png/.bmp),
    and returns the basenames of those tile files, sorted.
    """
    tiles = []
    for img in image_files:
        base, _ = os.path.splitext(img)
        for ext in exts:
            pattern = os.path.join(split_dir, f"{base}_tile*{ext}")
            for p in glob.glob(pattern):
                tiles.append(os.path.basename(p))
    return sorted(tiles)
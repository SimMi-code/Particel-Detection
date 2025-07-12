# yolo_square/box_utils.py

import cv2

def contour_to_square(contour):
    """
    From an OpenCV contour, compute the minimal enclosing square.
    Returns (cx, cy, side) in pixel coordinates.
    """
    x, y, w, h = cv2.boundingRect(contour)
    side = max(w, h)
    cx = x + w / 2
    cy = y + h / 2
    return cx, cy, side

def snap_tlwh_to_square(x, y, w, h):
    """
    Snap a top-left/width/height rectangle to its minimal enclosing square.

    Args:
        x, y: top‐left corner in pixels
        w, h: width and height in pixels
    Returns:
        (cx, cy, side): same center, side = max(w, h)
    """
    side = max(w, h)
    cx = x + w / 2
    cy = y + h / 2
    return cx, cy, side

def snap_xywh_to_square(cx, cy, w, h):
    """
    Snap a center/width/height box to its minimal enclosing square.

    Args:
        cx, cy: center coords in pixels
        w, h: width and height in pixels
    Returns:
        (cx, cy, side): same center, side = max(w, h)
    """
    side = max(w, h)
    return cx, cy, side

def square_to_xyxy(cx, cy, side):
    """
    Convert a square given by (center_x, center_y, side)
    to (x1, y1, x2, y2) pixel coordinates.
    """
    half = side / 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = int(cx + half)
    y2 = int(cy + half)
    return x1, y1, x2, y2

def square_to_yolo(cx, cy, side, img_shape):
    """
    Convert a square (cx, cy, side) in pixel coords to a YOLO label line:
      "0 x_center_norm y_center_norm width_norm height_norm"
    """
    H, W = img_shape[:2]
    x_norm = cx / W
    y_norm = cy / H
    w_norm = side / W
    h_norm = side / H
    return f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

def touches_border(cx, cy, side, img_shape, margin=1):
    """
    Returns True if the square (cx, cy, side) lies within `margin` pixels
    of any image border.
    """
    H, W = img_shape[:2]
    half = side / 2
    return (
        cx - half <= margin or
        cy - half <= margin or
        cx + half >= W - margin or
        cy + half >= H - margin
    )

def yolo_to_square(line, img_shape):
    """
    Parse a YOLO label line "cls xcn ycn wn hn" into a square (cx, cy, side).
    Assumes wn == hn (square). Returns floats.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO line: {line!r}")
    _, xcn, ycn, wn, hn = parts
    H, W = img_shape[:2]
    cx = float(xcn) * W
    cy = float(ycn) * H
    side = float(wn) * W
    return cx, cy, side

def touches_border(cx, cy, side, img_shape, margin=1):
    """
    Returns True if the square (center cx,cy, side) touches within
    `margin` pixels of any border of an image of shape `img_shape`.
    """
    H, W = img_shape[:2]
    half = side / 2
    return (
        cx - half <= margin or
        cy - half <= margin or
        cx + half >= W - margin or
        cy + half >= H - margin
    )


# yolo_square/scale_bar.py

import cv2
import numpy as np
import pytesseract
import re
import os

def detect_scale_bar(image_path, output_path=None, roi_frac=0.25, show=False):
    """
    Detects the scale bar in the bottom-left corner of the image and returns microns-per-pixel ratio.
    Also saves annotated image if output_path is given.

    Returns:
        (microns_per_pixel, unit) if detected, else (None, None)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Failed to read image: {image_path}")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    roi = gray[int(h * (1 - roi_frac)): h, 0: int(w * roi_frac)]
    roi_color = image[int(h * (1 - roi_frac)): h, 0: int(w * roi_frac)].copy()

    _, thresh = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scalebar_contour = None
    bar_length_px = None
    bar_box = None

    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
        area = cv2.contourArea(cnt)

        if area > 30 and aspect_ratio > 5 and h_box < 20:
            scalebar_contour = cnt
            bar_box = (x, y, w_box, h_box)
            bar_length_px = w_box
            break

    microns_per_pixel = None
    unit = None

    if scalebar_contour is not None and bar_box is not None:
        x, y, w_box, h_box = bar_box
        if output_path:
            cv2.rectangle(roi_color, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        y1 = max(y - 40, 0)
        y2 = min(y + h_box + 40, roi.shape[0])
        x1 = max(x - 20, 0)
        x2 = min(x + w_box + 20, roi.shape[1])
        text_roi = roi[y1:y2, x1:x2]

        config = "--psm 6"
        raw_text = pytesseract.image_to_string(text_roi, config=config)
        filtered_lines = [line for line in raw_text.splitlines() if re.search(r"\d", line)]
        filtered_text = " ".join(filtered_lines)

        match = re.search(r"(\d+)\s*(nm|µm|ym|um|mm)", filtered_text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
            if unit in ["ym", "um"]:
                unit = "µm"
            microns_per_pixel = value / float(bar_length_px)

            if output_path:
                text_x = max(x, 5)
                text_y = max(y - 10, 15)
                annotation_text = f"{value} {unit} = {bar_length_px}px ({microns_per_pixel:.2f} {unit}/px)"
                cv2.putText(
                    roi_color,
                    annotation_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

    else:
        print("⚠️ Scale bar not found.")

    if output_path:
        full_annotated = image.copy()
        full_annotated[int(h * (1 - roi_frac)): h, 0: int(w * roi_frac)] = roi_color
        cv2.imwrite(output_path, full_annotated)

    return microns_per_pixel, unit


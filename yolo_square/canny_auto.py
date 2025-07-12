# yolo_square/canny_auto.py

import cv2
import numpy as np

def auto_tune_parameters(image, circularity_min=0.7, circularity_max=1.2):
    """
    Heuristically tune blur and Canny edge thresholds based on image brightness and contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)

    # Adjust based on brightness/contrast heuristics
    if std_val < 20:
        blur_kernel = (7, 7)
        canny_min, canny_max = 30, 100
    elif mean_val < 100:
        blur_kernel = (5, 5)
        canny_min, canny_max = 40, 120
    else:
        blur_kernel = (3, 3)
        canny_min, canny_max = 50, 150

    return blur_kernel, canny_min, canny_max

def detect_contours_canny(image, canny_min, canny_max, blur_kernel=(5, 5)):
    """
    Apply Canny edge detection and return contours.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blurred, canny_min, canny_max)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


"""
Interim helper to strip the red hand-drawn annotation marks from the
reference photos so the ridge detector isn't just finding the pen ink.

This is a stopgap for testing only -- inpainting can't perfectly recover
whatever surface detail the ink directly covered, so results on
ink-masked images are indicative, not a substitute for clean
unannotated photos.
"""
import cv2
import numpy as np


def mask_red_ink(image_bgr, dilate_px=4):
    """Detect vivid red marker strokes and inpaint them out.

    Returns (clean_bgr, ink_mask) where ink_mask is the binary mask of
    pixels that were identified as ink and replaced.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around hue 0/180 in OpenCV's HSV. Vivid marker ink is
    # highly saturated and reasonably bright, which separates it from
    # the mostly-desaturated grey/silver bar surface.
    lower1 = np.array([0, 120, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 80])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel)

    clean = cv2.inpaint(image_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return clean, mask

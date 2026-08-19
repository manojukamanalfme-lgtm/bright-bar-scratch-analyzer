"""
Classical ridge-filter scratch/crack detector for bright-bar surfaces.

Pipeline:
  1. Flatten the smooth cylindrical specular gradient (the bar's own
     curvature creates a broad brightness roll-off that would otherwise
     swamp the fine scratch signal).
  2. Run a Frangi vesselness filter to highlight thin ridge/line-like
     structures (scratches show up as either bright or dark ridges
     depending on which side of the highlight band they fall on, so
     both polarities are checked).
  3. Threshold, clean up with morphology, and keep only elongated
     components (real scratches are long and thin; noise/pitting is not).
"""
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import remove_small_objects


def flatten_illumination(gray, blur_ksize=101):
    """Remove the broad specular/cylindrical brightness gradient via
    division by a heavily blurred version of itself, leaving local
    fine-scale contrast (scratches) intact."""
    gray_f = gray.astype(np.float32)
    baseline = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=blur_ksize / 3.0)
    baseline = np.clip(baseline, 1.0, None)
    flattened = gray_f / baseline
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)
    return flattened.astype(np.uint8)


def _normalize_robust(x, upper_percentile=99.5):
    """Scale by a high percentile rather than the true max -- a single
    hot pixel (sensor noise, a speck) shouldn't be able to hijack the
    whole map's normalization and drown out a real, spatially broader
    defect elsewhere."""
    hi = np.percentile(x, upper_percentile)
    hi = max(float(hi), 1e-6)
    return np.clip(x / hi, 0, 1)


def frangi_score_map(flattened, scale_min=1, scale_max=6, scale_step=1):
    """Frangi filter run on both polarities (bright ridge on dark bg and
    dark ridge on bright bg) since a scratch can appear either way
    depending on which side of the raking highlight it interrupts.
    Good at thin, sharp-edged scratches (narrow tubular ridge model)."""
    img_f = flattened.astype(np.float64) / 255.0
    scales = range(scale_min, scale_max + 1, scale_step)

    bright_ridges = frangi(img_f, sigmas=scales, black_ridges=False)
    dark_ridges = frangi(img_f, sigmas=scales, black_ridges=True)

    combined = np.maximum(bright_ridges, dark_ridges)
    return _normalize_robust(combined)


def _axis_kernel(length, thickness, angle_deg):
    """Elongated structuring element of given length/thickness, rotated
    to the expected bar-axis angle (0 = horizontal, matching the current
    capture rig)."""
    size = length + 4
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = size // 2
    x0 = (size - length) // 2
    canvas[y0 - thickness // 2: y0 - thickness // 2 + max(1, thickness),
           x0: x0 + length] = 1
    if angle_deg % 180 != 0:
        rot = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, 1.0)
        canvas = cv2.warpAffine(canvas, rot, (size, size),
                                 flags=cv2.INTER_NEAREST)
    return canvas


def directional_score_map(flattened, kernel_length=25, thickness=1, angle_deg=0):
    """Morphological top-hat/black-hat with a structuring element
    elongated along the bar axis -- responds to broad, low-contrast,
    axis-aligned grooves that a narrow-scale Frangi filter misses
    (it compares each pixel to its own elongated local neighborhood
    rather than assuming a specific ridge width)."""
    kernel = _axis_kernel(kernel_length, thickness, angle_deg)
    blackhat = cv2.morphologyEx(flattened, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(flattened, cv2.MORPH_TOPHAT, kernel)
    combined = np.maximum(blackhat, tophat).astype(np.float64)
    return _normalize_robust(combined, upper_percentile=99.9)


def ridge_score_map(flattened, scale_min=1, scale_max=6, scale_step=1,
                     tophat_length=25, tophat_thickness=1, axis_angle_deg=0):
    """Combine the Frangi (thin/sharp) and directional top-hat
    (broad/shallow) responses -- each catches defect shapes the other
    misses, per the image 1 vs image 6 comparison in the PoC dataset."""
    frangi_map = frangi_score_map(flattened, scale_min, scale_max, scale_step)
    tophat_map = directional_score_map(flattened, tophat_length, tophat_thickness, axis_angle_deg)
    return np.maximum(frangi_map, tophat_map)


def detect_scratches(image_bgr, config):
    """Run the full detection pipeline on a single frame.

    Returns (annotated_bgr, candidates) where candidates is a list of
    dicts: {bbox: (x, y, w, h), score, aspect_ratio, area}.
    """
    det_cfg = config.get("detection", {})
    score_threshold = det_cfg.get("score_threshold", 0.12)
    min_area = det_cfg.get("min_region_area_px", 25)
    min_aspect = det_cfg.get("min_aspect_ratio", 3.0)
    scale_min = det_cfg.get("frangi_scale_min", 1)
    scale_max = det_cfg.get("frangi_scale_max", 6)
    scale_step = det_cfg.get("frangi_scale_step", 1)
    tophat_length = det_cfg.get("tophat_kernel_length", 25)
    tophat_thickness = det_cfg.get("tophat_thickness", 1)
    axis_angle_deg = 0 if det_cfg.get("bar_axis", "horizontal") == "horizontal" else 90

    min_valid_intensity = det_cfg.get("min_valid_intensity", 12)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    flattened = flatten_illumination(gray)
    score_map = ridge_score_map(flattened, scale_min, scale_max, scale_step,
                                 tophat_length, tophat_thickness, axis_angle_deg)

    # Exclude near-black background/shadow -- flatten_illumination divides
    # by a blurred baseline that's unstable near zero there, which can
    # otherwise produce a large spurious connected blob (see PoC dataset
    # images 4/5 regression).
    foreground = gray >= min_valid_intensity

    # Exclude a border margin -- morphologyEx replicates pixels past the
    # image edge, which fabricates a high-contrast "ridge" right at the
    # frame boundary (see PoC dataset images 4/9 regression).
    border = max(tophat_length, 51) // 2
    edge_mask = np.zeros_like(foreground)
    edge_mask[border:-border, border:-border] = True

    score_map = score_map * foreground * edge_mask

    mask = (score_map >= score_threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask_bool = remove_small_objects(mask.astype(bool), min_size=min_area)
    mask = mask_bool.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates = []
    annotated = image_bgr.copy()

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < min_area:
            continue
        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect_ratio = long_side / short_side
        if aspect_ratio < min_aspect:
            continue

        region_score = float(score_map[labels == label_id].max())
        candidates.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "score": round(region_score, 3),
            "aspect_ratio": round(float(aspect_ratio), 2),
            "area": int(area),
        })
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated, f"{region_score:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return annotated, candidates

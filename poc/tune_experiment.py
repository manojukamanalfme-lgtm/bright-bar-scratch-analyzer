"""
Ad-hoc tuning experiment against the known miss in dataset/1.png --
there's an obvious horizontal groove-scratch around y=470-500 that the
default pipeline missed. Try a few flattening blur sizes and an
orientation-aware directional filter to see what actually picks it up.
"""
import cv2
import numpy as np
from pathlib import Path

from detection import flatten_illumination, ridge_score_map

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output" / "tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(BASE_DIR.parent / "dataset" / "1.png"))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Crop around the known scratch (full width, tight vertical band)
y0, y1 = 400, 560
crop = gray[y0:y1, :]
cv2.imwrite(str(OUT_DIR / "crop_raw.png"), crop)

# Intensity profile straight down a column through the scratch to see
# its actual width/contrast in raw pixel values
col = 600
profile = crop[:, col].astype(int)
print(f"Raw intensity profile at column {col} (row {y0}-{y1}):")
print(profile.tolist())

for blur in (15, 31, 51, 101, 151):
    flattened = flatten_illumination(crop, blur_ksize=blur)
    cv2.imwrite(str(OUT_DIR / f"flat_blur{blur}.png"), flattened)
    score = ridge_score_map(flattened, scale_min=1, scale_max=8, scale_step=1)
    score_img = (score * 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / f"score_blur{blur}.png"), score_img)
    print(f"blur={blur}: max score in crop = {score.max():.3f}")

# Directional filter tuned near-horizontal (bar axis in this crop is
# roughly horizontal) -- a simple elongated Sobel-like kernel via
# morphological top-hat with a long horizontal structuring element,
# which responds to thin horizontal ridges regardless of Frangi scale
# assumptions.
flattened_51 = flatten_illumination(crop, blur_ksize=51)
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
tophat = cv2.morphologyEx(flattened_51, cv2.MORPH_BLACKHAT, horiz_kernel)
tophat_bright = cv2.morphologyEx(flattened_51, cv2.MORPH_TOPHAT, horiz_kernel)
combined = np.maximum(tophat, tophat_bright)
cv2.imwrite(str(OUT_DIR / "directional_tophat.png"), combined)
print(f"directional top-hat: max={combined.max()}, mean={combined.mean():.2f}")

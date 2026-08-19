"""
Sanity test: run the classical ridge-filter detector against the
reference photos gathered so far (real scratches + one adversarial
stamped-marking image) and save annotated output for visual review.

This is not an automated pass/fail test (no ground-truth boxes yet) --
it's a quick way to eyeball whether the detector is even in the right
ballpark before any hardware/mechanical work starts.
"""
import json
import os
from pathlib import Path

import cv2

from detection import detect_scratches
from preprocess import mask_red_ink

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(BASE_DIR / "config.json") as f:
    config = json.load(f)

TEST_IMAGES = [
    "WhatsApp Image 2026-08-01 at 5.07.41 PM.jpeg",
    "WhatsApp Image 2026-08-01 at 5.07.41 PM (1).jpeg",
    "WhatsApp Image 2026-08-01 at 5.07.41 PM (2).jpeg",
    "Screenshot 2026-08-18 225200.png",  # adversarial: stamped marking, not a scratch
]

def main():
    for filename in TEST_IMAGES:
        path = PROJECT_DIR / filename
        if not path.exists():
            print(f"SKIP (not found): {filename}")
            continue

        image = cv2.imread(str(path))
        if image is None:
            print(f"SKIP (failed to read): {filename}")
            continue

        clean, ink_mask = mask_red_ink(image)
        if ink_mask.any():
            cv2.imwrite(str(OUTPUT_DIR / f"clean_{Path(filename).stem}.jpg"), clean)
            image = clean

        annotated, candidates = detect_scratches(image, config)

        out_name = f"detected_{Path(filename).stem}.jpg"
        out_path = OUTPUT_DIR / out_name
        cv2.imwrite(str(out_path), annotated)

        print(f"{filename}: {len(candidates)} candidate region(s) -> {out_path.name}")
        for c in candidates[:8]:
            print(f"    bbox={c['bbox']} score={c['score']} aspect={c['aspect_ratio']} area={c['area']}")

if __name__ == "__main__":
    main()

"""
Run the classical ridge-filter detector against the real dataset/
captures (clean, unannotated, taken under the raking-light rig).
"""
import json
from pathlib import Path

import cv2

from detection import detect_scratches

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_DIR / "config.json") as f:
    config = json.load(f)

def main():
    images = sorted(DATASET_DIR.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"SKIP (failed to read): {path.name}")
            continue

        annotated, candidates = detect_scratches(image, config)
        out_path = OUTPUT_DIR / f"detected_{path.name}"
        cv2.imwrite(str(out_path), annotated)

        print(f"{path.name}: {len(candidates)} candidate region(s) -> {out_path.name}")
        for c in candidates[:6]:
            print(f"    bbox={c['bbox']} score={c['score']} aspect={c['aspect_ratio']} area={c['area']}")

if __name__ == "__main__":
    main()

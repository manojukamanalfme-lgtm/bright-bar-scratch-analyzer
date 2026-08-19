"""
Live scratch detection PoC -- runs the Frangi + directional top-hat
detector (detection.py) continuously against a live camera feed (or a
recorded video file) instead of one-off images.

The Frangi filter is slow (~seconds per full-res frame, see the
dataset test runs), nowhere near video frame rates. So this runs the
raw camera feed live and smooth on screen, while detection runs
continuously in a background thread on a downscaled copy of whatever
the latest frame is -- the overlay updates as fast as the detector can
keep up (typically every 1-3s at default settings), not every frame.
That's an honest PoC of "detection from a live feed," not a claim of
real-time throughput -- speeding up the detector itself (smaller ROI,
a faster ridge measure, GPU) is a separate later step.

Controls while the preview window is focused:
    d        pause / resume detection
    s        save the current frame as a still image
    q / ESC  quit

Usage:
    python live_detect.py                          # live Basler camera
    python live_detect.py --source recordings/2026-08-19/bar_12-03-49.mp4
    python live_detect.py --max-width 480           # faster, coarser detection
"""
import argparse
import datetime
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from detection import detect_scratches
from record_bar_video import open_camera, make_converter, to_bgr

try:
    from pypylon import pylon
except ImportError:
    pylon = None

BASE_DIR = Path(__file__).resolve().parent


class DetectionWorker:
    """Runs detect_scratches continuously in the background on whatever
    the latest frame is, at a reduced resolution for speed. The display
    loop just reads back the most recent result -- no blocking."""

    def __init__(self, config, max_width):
        self.config = config
        self.max_width = max_width
        self.lock = threading.Lock()
        self.latest_frame = None
        self.candidates = []
        self.scale = 1.0
        self.last_duration = 0.0
        self.last_update = 0.0
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit_frame(self, frame_bgr):
        with self.lock:
            self.latest_frame = frame_bgr

    def _loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            with self.lock:
                frame = self.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            scale = min(1.0, self.max_width / w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

            start = time.time()
            try:
                _, candidates = detect_scratches(small, self.config)
            except Exception as e:
                print(f"Detection error: {e}")
                candidates = []
            duration = time.time() - start

            with self.lock:
                self.candidates = candidates
                self.scale = scale
                self.last_duration = duration
                self.last_update = time.time()

    def get_overlay_data(self):
        with self.lock:
            return list(self.candidates), self.scale, self.last_duration, self.last_update

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


def draw_overlay(display, candidates, scale, last_duration, last_update, paused):
    inv_scale = 1.0 / scale if scale > 0 else 1.0
    for c in candidates:
        x, y, w, h = c["bbox"]
        x, y, w, h = int(x * inv_scale), int(y * inv_scale), int(w * inv_scale), int(h * inv_scale)
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(display, f"{c['score']:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    age = time.time() - last_update if last_update else 0
    status = "PAUSED" if paused else f"detect: {last_duration:.2f}s  (updated {age:.1f}s ago)"
    color = (0, 165, 255) if paused else (0, 255, 255)
    cv2.putText(display, status, (15, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(display, f"{len(candidates)} candidate(s)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def frame_source_camera(user_set):
    camera, pixel_format = open_camera(user_set, None, None)
    converter = make_converter(pixel_format)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def read():
        if not camera.IsGrabbing():
            return False, None
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            return False, None
        image = converter.Convert(grab)
        frame = to_bgr(image.GetArray(), pixel_format)
        grab.Release()
        return True, frame

    def release():
        camera.StopGrabbing()
        camera.Close()

    return read, release


def frame_source_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video file: {path}")

    def read():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            ret, frame = cap.read()
        return ret, frame

    def release():
        cap.release()

    return read, release


def main():
    parser = argparse.ArgumentParser(description="Live scratch detection PoC")
    parser.add_argument("--source", default="camera",
                         help='"camera" for live Basler feed, or a path to a video file')
    parser.add_argument("--user-set", default="UserSet1")
    parser.add_argument("--max-width", type=int, default=480,
                         help="Downscale frames to this width before running detection (speed vs. detail)")
    parser.add_argument("--output-dir", default="recordings")
    args = parser.parse_args()

    with open(BASE_DIR / "config.json") as f:
        config = json.load(f)

    if args.source == "camera":
        if pylon is None:
            raise SystemExit("pypylon is not installed. Run: pip install pypylon")
        read_frame, release = frame_source_camera(args.user_set)
    else:
        read_frame, release = frame_source_video(args.source)

    worker = DetectionWorker(config, args.max_width)

    window_name = "Live Scratch Detection -- d:pause  s:snapshot  q:quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Live detection running. Detection updates in the background; "
          "raw feed stays live regardless of detector speed.")

    try:
        while True:
            ret, frame = read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            worker.submit_frame(frame)
            display = frame.copy()
            candidates, scale, duration, last_update = worker.get_overlay_data()
            draw_overlay(display, candidates, scale, duration, last_update, worker.paused)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('d'):
                worker.paused = not worker.paused
                print("Detection paused" if worker.paused else "Detection resumed")
            elif key == ord('s'):
                folder = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d"))
                os.makedirs(folder, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%H-%M-%S_%f")
                snap_path = os.path.join(folder, f"live_snapshot_{timestamp}.png")
                cv2.imwrite(snap_path, frame)
                print(f"Snapshot saved: {snap_path}")

    finally:
        worker.stop()
        release()
        cv2.destroyAllWindows()
        print("Closed.")


if __name__ == "__main__":
    main()

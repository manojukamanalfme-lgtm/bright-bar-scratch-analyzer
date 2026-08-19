"""
Basler bar-surface recorder for the raking-light test station (PoC).

Uses pypylon (Basler's official Python wrapper around the Pylon SDK you
already have installed for Pylon Viewer) instead of the GStreamer
pylonsrc pipeline used in basler_recorder.py -- that pipeline relied on
Jetson/Linux GST plugin paths (aarch64-linux-gnu) that don't exist on
this Windows dev box. pypylon works cross-platform against the same
Pylon runtime, so this is the direct equivalent for capture here; the
GStreamer approach comes back into play for the eventual Ubuntu
deployment (see camera.py / GstCamera in the main app).

Does NOT override exposure/gain/AOI by default -- it opens the camera
and uses whatever is currently active (i.e. what you tuned in Pylon
Viewer / saved to a User Set). Pass --exposure / --gain to override.

Controls while the preview window is focused:
    r        start / stop recording
    s        save the current frame as a still image
    q / ESC  quit

Usage:
    python record_bar_video.py
    python record_bar_video.py --output-dir recordings --user-set UserSet1
    python record_bar_video.py --exposure 10000 --gain 336
"""
import argparse
import datetime
import os
import time

import cv2
import numpy as np

try:
    from pypylon import pylon
except ImportError as e:
    raise SystemExit(
        "pypylon is not installed. Run: pip install pypylon\n"
        "(It binds to the Pylon SDK already installed for Pylon Viewer.)"
    ) from e


def get_node(camera, *names):
    """Return the first available GenICam node from a list of candidate
    names -- needed because older Basler ace cameras (like the
    acA1300-30gc) use legacy names (ExposureTimeAbs, GainRaw) while
    newer ones use the SFNC2 names (ExposureTime, Gain)."""
    for name in names:
        node = getattr(camera, name, None)
        if node is not None and node.IsReadable():
            return node
    return None


def open_camera(user_set=None, exposure_us=None, gain_raw=None):
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise SystemExit("No Basler camera found. Check power/network connection.")

    camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
    camera.Open()

    info = camera.GetDeviceInfo()
    print(f"Connected: {info.GetModelName()}  serial={info.GetSerialNumber()}")

    if user_set:
        camera.UserSetSelector.SetValue(user_set)
        camera.UserSetLoad.Execute()
        print(f"Loaded {user_set} from camera memory")

    exposure_node = get_node(camera, "ExposureTime", "ExposureTimeAbs")
    gain_node = get_node(camera, "Gain", "GainRaw")

    if exposure_us is not None:
        auto_node = get_node(camera, "ExposureAuto")
        if auto_node is not None:
            auto_node.SetValue("Off")
        exposure_node.SetValue(type(exposure_node.GetValue())(exposure_us))
        print(f"Exposure time set to {exposure_us} us")

    if gain_raw is not None:
        auto_node = get_node(camera, "GainAuto")
        if auto_node is not None:
            auto_node.SetValue("Off")
        gain_node.SetValue(type(gain_node.GetValue())(gain_raw))
        print(f"Gain set to {gain_raw}")

    width = camera.Width.GetValue()
    height = camera.Height.GetValue()
    pixel_format = camera.PixelFormat.GetValue()
    exposure = exposure_node.GetValue() if exposure_node else float("nan")
    gain = gain_node.GetValue() if gain_node else float("nan")
    print(f"AOI: {width}x{height}  PixelFormat={pixel_format}  "
          f"Exposure={exposure:.1f}us  Gain={gain:.1f}")

    return camera, pixel_format


def make_converter(pixel_format):
    """Convert whatever the camera hands us to BGR8 for OpenCV."""
    converter = pylon.ImageFormatConverter()
    if "Mono" in pixel_format:
        converter.OutputPixelFormat = pylon.PixelType_Mono8
    else:
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def to_bgr(frame, pixel_format):
    if "Mono" in pixel_format:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def date_folder(output_dir):
    folder = os.path.join(output_dir, datetime.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description="Record Basler bar-surface video for scratch PoC")
    parser.add_argument("--output-dir", default="recordings")
    parser.add_argument("--user-set", default="UserSet1",
                         help='Camera config to load before recording (default: UserSet1). Pass "" to skip.')
    parser.add_argument("--exposure", type=float, default=None, help="Exposure time in microseconds")
    parser.add_argument("--gain", type=float, default=None, help="Gain (raw units)")
    parser.add_argument("--fps", type=float, default=15.0, help="Output video container FPS")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec for output video")
    args = parser.parse_args()

    camera, pixel_format = open_camera(args.user_set, args.exposure, args.gain)
    converter = make_converter(pixel_format)

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    writer = None
    is_recording = False
    frame_count = 0
    record_start = None
    current_file = None

    window_name = "Bar Surface -- r:record  s:snapshot  q:quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Ready. Focus the preview window and press 'r' to start recording.")

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                grab.Release()
                continue

            image = converter.Convert(grab)
            frame = image.GetArray()
            frame_bgr = to_bgr(frame, pixel_format)
            grab.Release()

            display = frame_bgr.copy()
            if is_recording:
                elapsed = time.time() - record_start
                cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display, f"REC {elapsed:6.1f}s  frames={frame_count}",
                            (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                writer.write(frame_bgr)
                frame_count += 1
            else:
                cv2.putText(display, "Press 'r' to record", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # q or ESC
                break

            elif key == ord('r'):
                if not is_recording:
                    folder = date_folder(args.output_dir)
                    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
                    current_file = os.path.join(folder, f"bar_{timestamp}.mp4")
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*args.codec)
                    writer = cv2.VideoWriter(current_file, fourcc, args.fps, (w, h))
                    if not writer.isOpened():
                        print(f"Failed to open video writer for {current_file}")
                        writer = None
                        continue
                    is_recording = True
                    frame_count = 0
                    record_start = time.time()
                    print(f"Recording started: {current_file}")
                else:
                    is_recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print(f"Recording stopped: {current_file}  "
                          f"({frame_count} frames, {time.time() - record_start:.1f}s)")

            elif key == ord('s'):
                folder = date_folder(args.output_dir)
                timestamp = datetime.datetime.now().strftime("%H-%M-%S_%f")
                snap_path = os.path.join(folder, f"snapshot_{timestamp}.png")
                cv2.imwrite(snap_path, frame_bgr)
                print(f"Snapshot saved: {snap_path}")

    finally:
        if writer:
            writer.release()
            print(f"Recording finalized: {current_file}")
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()

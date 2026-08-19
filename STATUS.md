# Bright Bar Scratch Analyzer -- PoC Status

Last updated: 2026-08-19

## Goal

Feasibility PoC for automated crack/scratch detection on polished Bright
Bars (16-160mm dia) at Aarti Steels' checking station, using a Basler
area-scan camera + raking-light illumination + classical CV (with a
DL segmentation model as a possible later phase once a labeled dataset
exists). Assisted-inspection tool (operator makes the final call), not
an automated line replacement. Deployment target: GPU-based Ubuntu
workstation (not Jetson -- that's reserved for the existing Bar Counter
project, a separate system).

## What's reused from the Bar Counter codebase (`app_pylon_v1.4.11.py`)

- Camera capture pattern (`GstCamera` class / Pylon+GStreamer pipeline)
  -- reusable as-is for the eventual Ubuntu deployment.
- Flask/SocketIO dashboard skeleton, config.json convention, logging
  (atomic-write CSV, daily rotation), signal-handling/cleanup pattern.
- NOT reused: YOLO rod-counting model, SORT/zone-crossing counting
  logic, heat-number/genealogy workflow -- all specific to bar counting,
  irrelevant to surface inspection.

## Current architecture decisions

- **Illumination**: raking/grazing light (validated by hand with
  reference photos before any code was written). Multi-angle
  (photometric-stereo-style) stepping identified as likely necessary,
  not yet implemented -- see Open Items.
- **Camera**: single Basler acA1300-30gc (area-scan), roller-fed
  singulated bar, rotated by machine, slow throughput (~a few bars/min).
  Camera + light height will eventually be motorized (deferred -- not
  started).
- **Processing**: classical CV, not deep learning, for the PoC --
  no labeled dataset exists yet. Combination of two filters run on an
  illumination-flattened grayscale frame:
  - Frangi ridge/vesselness filter (skimage) -- catches thin, sharp
    scratches (validated: dataset image 6).
  - Directional top-hat (OpenCV morphology, elongated kernel aligned to
    bar axis) -- catches broad, shallow, low-contrast grooves that
    Frangi misses (validated: dataset image 1, previously a false
    negative, now correctly detected after adding this).
  - See `poc/detection.py` for the implementation; `poc/config.json`
    for tunable parameters (score_threshold, frangi scale range,
    tophat kernel length, min_aspect_ratio, min_valid_intensity).

## Known issues / fixed bugs (context for why the code looks the way it does)

- Red pen annotations in the earliest reference photos were being
  detected as ridges themselves -- `poc/preprocess.py` (`mask_red_ink`)
  is an interim ink-removal stopgap, unreliable, not part of the real
  pipeline (kept for reference only).
- Illumination-flattening (`flatten_illumination` in detection.py)
  divides by a blurred baseline -- unstable near-black background,
  which produced one giant false-positive blob covering most of the
  frame. Fixed via `min_valid_intensity` foreground mask.
- Morphological filters replicate pixels past the image border,
  fabricating a fake high-contrast "ridge" at the frame edge. Fixed via
  a border-margin exclusion in `detect_scratches`.
- **Still open**: on the current dataset (bundle photos, multiple bars
  in frame), an adjacent bar's edge in the same frame reads as a
  legitimate long linear feature and gets flagged. This is judged to be
  a dataset/capture-context artifact (bundle framing), not an algorithm
  bug -- expected to disappear once captures show a single bar only
  (the intended roller-feed production framing). NOT YET CONFIRMED --
  waiting on new single-bar captures to validate.

## Live-feed detection

`poc/live_detect.py` runs detection continuously against a live Basler
feed or a recorded video file, in a background thread, overlaying
results on the live (smooth) camera view as they become available.

**Performance reality check**: Frangi (skimage, CPU, pure Python) takes
~25s per full-res (~1238x785) frame in the batch tests
(`poc/test_dataset.py` output). This is NOT a hard limit of the
approach -- see Open Items for the speed plan. The actual throughput
requirement is per-bar latency within the multi-second roller-feed dwell
time, not per-frame video-rate analytics, which is a much more
achievable target than it first sounds.

## Files

- `poc/detection.py` -- core detection pipeline (flatten, Frangi,
  directional top-hat, combine, connected-components -> candidate boxes)
- `poc/record_bar_video.py` -- Basler camera capture/recording (pypylon).
  Loads `UserSet1` from camera by default.
- `poc/live_detect.py` -- live detection viewer (camera or video file source)
- `poc/test_dataset.py` -- batch-runs detection.py against `dataset/*.png`
- `poc/tune_experiment.py` -- ad-hoc tuning script used to diagnose the
  image-1 miss (kept for reference, not part of the pipeline)
- `poc/config.json` -- all tunable detection parameters
- `dataset/` -- 10 real, clean (unannotated), full-res bar-surface
  photos captured with `record_bar_video.py` under the raking-light rig.
  These are bundle photos (multiple bars in frame) -- see known issues.
- `pylon setting images/` -- reference screenshots of the camera's
  Pylon Viewer configuration (AOI, exposure, gain, trigger, etc.) and
  the IP Configurator (documents the GigE network fix below).

## Environment notes (for the machine transfer)

- Camera is GigE (Basler acA1300-30gc), Auto-IP/link-local
  (169.254.153.250/16). On this Windows dev machine, this required:
  1. Adding a secondary static IP in the 169.254.0.0/16 range to the
     camera's network adapter (the adapter's primary IP was on an
     unrelated subnet).
  2. Reclassifying that adapter's Windows network category from
     "Public" to "Private" -- Public-profile firewall silently drops
     the unsolicited inbound GigE Vision stream traffic even though
     device discovery/control still works.
  Both steps will likely be needed again on the new GPU machine if it's
  also Windows. If the GPU machine is Ubuntu, the equivalent is a
  static IP on the camera's subnet and no analogous "network profile"
  firewall concept -- just normal iptables/ufw rules if a firewall is
  active.
- `pypylon` (pip install) was used for capture on Windows instead of
  the GStreamer `pylonsrc` pipeline in `basler_recorder.py`, because
  that script hardcodes Jetson/Linux GStreamer plugin paths. On the
  Ubuntu GPU machine, GStreamer + `pylonsrc` becomes viable again (and
  is what the eventual production app should use, per the main app's
  `GstCamera` pattern) -- worth revisiting which capture approach to
  standardize on there.
- Camera uses **legacy GenICam parameter names**
  (`ExposureTimeAbs`/`GainRaw`), not the newer SFNC2 names
  (`ExposureTime`/`Gain`) -- `record_bar_video.py`'s `get_node()`
  helper handles both; keep this in mind if writing more camera code.

## Open items / next steps (in rough priority order)

1. **Waiting on user**: capture a proper dataset of single-bar-only
   frames (not bundle photos) under the raking light, including some
   with known scratches clearly in frame -- needed to properly validate
   detection accuracy without the bundle-edge-artifact confound.
2. Re-run `test_dataset.py` against the new single-bar dataset once
   available; confirm the bundle-edge false positives disappear.
3. **Speed optimization** (deferred until accuracy is validated -- see
   above): reduce Frangi scale range to match confirmed real defect
   widths (currently untuned, 1-6px, wider than needed); ROI-crop to
   the highlight band; consider reimplementing the ridge measure with
   OpenCV-native ops (compiled, much faster than skimage's pure-Python
   `frangi`) and/or GPU acceleration, since the target deployment
   machine has a discrete GPU currently unused by this CPU-only pipeline.
4. Multi-angle raking-light capture (photometric-stereo-style light
   stepping) -- deferred pending single-bar dataset, but empirically
   motivated: dataset image 1's scratch was only reliably caught after
   filter tuning, suggesting a single fixed light angle may not always
   give strong-enough contrast; stepping the light angle per bar was
   the original Step 2 recommendation for this reason.
5. Motorized camera/light height adjuster for the 16-160mm diameter
   range -- confirmed as motorized/software-controlled (per earlier
   discussion), not yet started.
6. Once a real labeled dataset exists (hundreds+ examples, likely
   bootstrapped by using this classical detector to pre-flag candidates
   for human confirm/reject): evaluate a lightweight DL segmentation
   model + TensorRT as a phase-2 replacement/supplement for the
   classical filter, per the original feasibility summary.

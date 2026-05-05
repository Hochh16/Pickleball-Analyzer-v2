# Stage 3 — Pose Estimation

## Purpose

For each player detection in `players.parquet` that's worth tracking (the user
plus other persistent on-court players), run MediaPipe Pose on the bbox crop
and emit a per-(frame, track_id) row of 33 landmarks in image-space pixels.

Skeletons are needed by Stage 5 (shot detection) and Stage 8 (metrics /
biomechanics).

## Inputs

All in the per-video folder:

- `video.mp4` — source video.
- `players.parquet` — from Stage 2. Stage 3 reads:
  - `frame`, `track_id`, `is_user`, `transient`
  - `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` — image-space pixel bbox
  - `t_sec` — copied through to `poses.parquet`
- `court.json` — from Stage 1. Stage 3 reads:
  - `video.fps` — for sanity-checking frame timing only.

No new user input is required.

## Scope (which detections get poses)

Stage 3 runs pose on a **filtered subset** of detections:

- All rows where `is_user == True`, OR
- All rows where `transient == False` AND `is_user == False`.

In other words: the user, plus all confirmed non-transient non-user tracks.
Transient tracks (lifetime < 30 frames or never inside the tracking zone) are
skipped — pose on them is wasted compute.

The scope is computed up-front from the loaded `players.parquet`. Detections
outside the scope are not present in `poses.parquet` at all (they're not
emitted as NaN rows).

## Outputs

Written to the per-video folder:

### `poses.parquet`

One row per in-scope (frame, track_id). Columns:

| Column | Type | Notes |
|---|---|---|
| `frame` | int | 0-indexed frame number |
| `t_sec` | float | frame time in seconds (copied from players.parquet) |
| `track_id` | int | from players.parquet |
| `is_user` | bool | from players.parquet |
| `pose_detected` | bool | True if MediaPipe returned a pose for this crop; False otherwise |
| `<landmark>_x_px` | float | image-space x in pixels; NaN if pose_detected=False |
| `<landmark>_y_px` | float | image-space y in pixels; NaN if pose_detected=False |
| `<landmark>_z` | float | MediaPipe-reported relative depth (smaller = closer to camera); NaN if pose_detected=False |
| `<landmark>_visibility` | float | MediaPipe visibility score in [0, 1]; NaN if pose_detected=False |

There are 33 landmarks, so 132 landmark columns plus the 5 metadata columns
= **137 columns total**.

The 33 landmark names follow MediaPipe's `PoseLandmark` enum. They are listed
in stable order at the top of `pose.py` as the `LANDMARK_NAMES` constant.
Examples: `nose`, `left_shoulder`, `right_shoulder`, `left_elbow`,
`right_wrist`, `left_hip`, `right_knee`, `left_ankle`, etc.

### `pose_summary.json`

Per-track diagnostic, written alongside the parquet:

```json
{
  "schema_version": 1,
  "total_in_scope_detections": 4321,
  "total_pose_detected": 3987,
  "overall_detection_rate": 0.923,
  "per_track": [
    {"track_id": 2, "is_user": true,  "n_detections": 511, "n_pose_detected": 487, "rate": 0.953},
    {"track_id": 7, "is_user": false, "n_detections": 280, "n_pose_detected": 250, "rate": 0.893}
  ],
  "warnings": []
}
```

## Process

1. Load `players.parquet`, `court.json`, and open the video.
2. Compute the scope subset (described above). Group by `frame` so all in-scope
   detections on a given frame are processed together (one video read per
   frame, multiple crops per read).
3. For each frame from 0 to last in-scope frame:
   1. Read the frame from the video.
   2. For each in-scope detection on this frame:
      1. Pad the bbox by `BBOX_PAD_FRAC` (default 0.10) on each side, then
         clip to image bounds.
      2. Crop the frame to the padded bbox.
      3. Run MediaPipe Pose on the crop.
      4. If a pose is returned: convert the 33 normalized `(x, y)` landmark
         coordinates back to image-space pixels using the crop's offset and
         size; record `z` and `visibility` as MediaPipe returned them; emit a
         row with `pose_detected=True`.
      5. If no pose is returned: emit a row with `pose_detected=False` and
         all 132 landmark values as NaN.
4. Write `poses.parquet` and `pose_summary.json`.

## Key behavioral notes

- **Back-facing players are expected.** The camera is typically behind the
  user. MediaPipe handles back-facing reasonably well, but expect lower
  `visibility` scores on chest/face landmarks. Downstream stages (Stage 5+)
  should weight landmarks by their visibility, not treat all landmarks as
  equally reliable. Recorded as a deferred consideration in `KNOWN_ISSUES.md`.
- **No pose smoothing.** Stage 3 emits raw per-frame poses. Temporal
  smoothing, if needed, is a downstream concern.
- **No interpolation across `pose_detected=False` rows.** Honest gaps.

## Configuration

Hardcoded constants at top of `pose.py`:

```python
MODEL_COMPLEXITY = 1            # 0=lite, 1=full, 2=heavy
MIN_DETECTION_CONFIDENCE = 0.5  # MediaPipe default
MIN_TRACKING_CONFIDENCE = 0.5   # ignored in static-image mode but accepted
BBOX_PAD_FRAC = 0.10            # pad bbox by 10% per side before cropping
USER_DETECTION_RATE_WARNING = 0.5   # warn if user's pose detection rate < 50%
```

## Failure modes (loud, never silent)

- `players.parquet` missing or malformed → fail with a clear message.
- `video.mp4` missing or unreadable → fail.
- MediaPipe import failure → fail with: *"MediaPipe not installed. Run: pip install mediapipe"*.
- A frame read failure mid-run → fail loudly. No silent truncation.
- Per-detection MediaPipe failures → silent NaN row (NOT a hard failure).
- High user-pose-failure rate → warning written to `pose_summary.json["warnings"]` AND logged. Does not fail the run.

## CLI

```
python -m stages.pose.pose <video_folder>
```

`<video_folder>` must contain `video.mp4`, `court.json`, and `players.parquet`.
Outputs are written into the same folder.

## Smoke test conditions

`test_pose.py` runs against `data/test_clip/` and verifies:

1. `poses.parquet` exists, is non-empty, and has all 137 expected columns
   with correct dtype kinds.
2. The number of rows in `poses.parquet` exactly equals the number of
   in-scope detections in `players.parquet`.
3. At least one row with `is_user=True` has `pose_detected=True` (we know
   the user is detected for at least 511 frames in the test clip).
4. For every row where `pose_detected=True`, the following landmarks have
   image-space pixel coordinates inside the bbox they came from (within a
   small tolerance for the 10% padding):
   `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`,
   `left_wrist`, `right_wrist`, `left_hip`, `right_hip`,
   `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.
5. Every row with `pose_detected=False` has all 132 landmark columns as NaN.
6. `pose_summary.json` exists, is valid JSON, and reports a `per_track`
   entry for every distinct track_id in `poses.parquet`.
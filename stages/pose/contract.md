# Stage 3 — Pose Estimation

## Purpose

For each player detection in `players.parquet` that's worth tracking (the user
plus other on-court players, after a strict scope filter), run MediaPipe Pose
on the bbox crop and emit a per-(frame, track_id) row of 33 landmarks in
image-space pixels.

Skeletons are needed by Stage 5 (shot detection) and Stage 8 (metrics /
biomechanics).

## Inputs

All in the per-video folder:

- `video.mp4` — source video.
- `players.parquet` — from Stage 2. Stage 3 reads:
  - `frame`, `track_id`, `is_user`, `transient`, `in_court`
  - `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` — image-space pixel bbox
  - `court_x_ft`, `court_y_ft` — for the strict scope filter
  - `t_sec` — copied through to `poses.parquet`
- `court.json` — from Stage 1. Stage 3 reads:
  - `video.fps` — for sanity-checking frame timing only.

No new user input is required.

## Scope (which detections get poses)

The "non-transient" filter from Stage 2 alone is too permissive — it admits
people on adjacent courts whose homography projections happen to land inside
the user's court rectangle. Stage 3 applies a strict per-track scope filter
on top of `transient`:

**When `track_roles.json` (Stage 2.5) is present (the normal path), scope is by
ROLE.** A track is in scope if its role is `user`, `partner`, `opp_left`, or
`opp_right`; `noise` tracks are excluded. `is_user` is taken from the role `user`
(not the click-only flag in `players.parquet`, empty in the no-clicks flow), so
every user segment — including re-identified / behind-baseline ones — is posed,
and **partner + opponents are posed too**. This replaced a geometric court_y
gate, which could not survive the far-side projection: foot points there jitter
past the baseline (the homography is ~4 px/ft near the horizon), so a `max ≤ 44`
gate deleted every real opponent from pose while a looser median gate admitted
in-court noise. The Stage 2.5 role classification is the right discriminator.
(See SYSTEM_DESIGN.md §3 Stage 2/3.)

**Fallback — no `track_roles.json`:** a track is in scope if `is_user`, OR all of
the conservative geometric gate hold (for non-user tracks):
  - `transient == False` (already lifetime ≥ 30 frames and at least one in-zone foot point)
  - `in_court_frac >= 0.50` (at least half the track's foot points project inside the 0×20 ft × 0×44 ft court rectangle)
  - `court_y_ft.max() <= 44.0` (never projects beyond the far baseline — adjacent-court contamination filter)
  - `court_y_ft.min() >= -8.0` (never projects more than 8 ft behind the user's near baseline — bystander/walk-on filter; matches `tracking_zone.behind_baseline_ft`)
  - `lifetime_seconds > 5.0` (track persists longer than 5 seconds)

Tracks out of scope are not present in `poses.parquet` at all (not
emitted as NaN rows). The user is always in scope, even if their other
metrics would fail the gate (e.g., user serves from far behind the
baseline).

This is a heuristic — it handles the common camera setup (player + partner +
2 opponents on the user's court, with adjacent-court activity visible) but
may need tuning for unusual footage. The cleaner long-term solution is a
dedicated "real player" classification stage between Stage 2 and Stage 3
(see KNOWN_ISSUES.md).

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
  "scope_filter": {
    "total_player_detections": 26272,
    "non_transient_detections": 20246,
    "in_scope_detections": 4211,
    "in_scope_tracks": 9
  },
  "total_pose_detected": 4107,
  "overall_detection_rate": 0.975,
  "per_track": [
    {"track_id": 2, "is_user": true,  "n_detections": 511, "n_pose_detected": 510, "rate": 0.998},
    {"track_id": 7, "is_user": false, "n_detections": 1280, "n_pose_detected": 1190, "rate": 0.930}
  ],
  "warnings": []
}
```

## Process

1. Load `players.parquet`, `court.json`, and open the video.
2. Apply the scope filter (described above) to compute the in-scope detection
   subset. Group by `frame` so all in-scope detections on a given frame are
   processed together.
3. Build the MediaPipe Pose detector via the new Tasks API
   (`mediapipe.tasks.vision.PoseLandmarker`). Image mode (one inference per
   crop, no temporal state).
4. For each frame from 0 to last in-scope frame:
   1. Read the frame from the video.
   2. For each in-scope detection on this frame:
      1. Pad the bbox by `BBOX_PAD_FRAC` (default 0.10) on each side, then
         clip to image bounds.
      2. Crop the frame to the padded bbox.
      3. **Mask other detections in the crop.** For each other detection
         on the same frame (whether in-scope or not), shrink its bbox by
         `OTHER_PERSON_MASK_SHRINK_FRAC` (default 0.05) on each side,
         intersect with the crop, and fill those pixels with
         `OTHER_PERSON_MASK_COLOR` (mid-grey BGR 128, 128, 128). The
         subject's own bbox region is then re-painted from the original
         crop in case any masked rectangle overlapped it. This prevents
         single-person MediaPipe from picking the wrong person when
         bboxes overlap.
      4. Run MediaPipe Pose on the masked crop.
      5. If a pose is returned: convert the 33 normalized `(x, y)` landmark
         coordinates back to image-space pixels using the crop's offset and
         size; record `z` and `visibility` as MediaPipe returned them; emit
         a row with `pose_detected=True`.
      6. If no pose is returned: emit a row with `pose_detected=False` and
         all 132 landmark values as NaN.
5. Write `poses.parquet` and `pose_summary.json`.

## Key behavioral notes

- **Back-facing players are expected.** The camera is typically behind the
  user. MediaPipe handles back-facing reasonably well, but expect lower
  `visibility` scores on chest/face landmarks. Downstream stages should
  weight landmarks by their visibility.
- **Crops are masked before pose extraction.** Other detections on the
  same frame (in-scope or not) are painted over with a neutral grey
  rectangle so MediaPipe sees only one person per crop. This is a
  workaround for MediaPipe being a single-person model; see KNOWN_ISSUES.md
  for the rationale and follow-up plan.
- **No pose smoothing.** Stage 3 emits raw per-frame poses. Temporal
  smoothing, if needed, is a downstream concern.
- **No interpolation across `pose_detected=False` rows.** Honest gaps.
- **MediaPipe's new Tasks API** uses a downloaded `.task` model bundle file
  rather than the bundled `mp.solutions.pose` API (which was removed in
  MediaPipe 0.10+). The model file (`pose_landmarker_full.task`, ~9 MB) is
  auto-downloaded on first run if not present.

## Configuration

Hardcoded constants at top of `pose.py`:

```python
MODEL_COMPLEXITY = 1            # 0=lite, 1=full, 2=heavy → maps to .task file
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5   # MediaPipe Tasks API uses this name
MIN_TRACKING_CONFIDENCE = 0.5   # ignored in image mode, accepted for API
BBOX_PAD_FRAC = 0.10
USER_DETECTION_RATE_WARNING = 0.5

# Scope filter constants
SCOPE_MIN_IN_COURT_FRAC = 0.50
SCOPE_MAX_Y_FT = 44.0
SCOPE_MIN_Y_FT = -8.0
SCOPE_MIN_LIFETIME_SEC = 5.0

# Crop-masking constants
OTHER_PERSON_MASK_COLOR = (128, 128, 128)   # BGR mid-grey
OTHER_PERSON_MASK_SHRINK_FRAC = 0.05
```

## Failure modes (loud, never silent)

- `players.parquet` missing or malformed → fail with a clear message.
- `video.mp4` missing or unreadable → fail.
- MediaPipe import failure → fail with: *"MediaPipe not installed. Run: pip install mediapipe"*.
- MediaPipe model file download failure → fail with the URL the user can fetch manually.
- A frame read failure mid-run → fail loudly. No silent truncation.
- Per-detection MediaPipe failures → silent NaN row (NOT a hard failure).
- High user-pose-failure rate → warning written to `pose_summary.json["warnings"]` AND logged. Does not fail the run.
- Empty in-scope set → fail with a message suggesting the scope filter may be too strict for this footage.

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
   in-scope detections in `players.parquet` (recomputed by the test using
   the same scope filter).
3. At least one row with `is_user=True` has `pose_detected=True`.
4. For every row where `pose_detected=True`, the following landmarks have
   image-space pixel coordinates within 40% of the bbox dimensions on each
   side (allowing for the 10% pad plus extrapolated landmarks near the
   image edge), counting only landmarks with
   `visibility >= 0.5`. Lower-visibility landmarks are extrapolated by
   MediaPipe and may legitimately project outside the bbox; downstream
   stages should weight by visibility rather than treat all landmarks as
   equally reliable.
   Checked landmarks: `left_shoulder`, `right_shoulder`, `left_elbow`,
   `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`,
   `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.
5. Every row with `pose_detected=False` has all 132 landmark columns as NaN.
6. `pose_summary.json` exists, is valid JSON, has a `per_track` entry for
   every distinct track_id in `poses.parquet`, and reports an in-scope
   detection count between 100 and 12000 (sanity-check that the scope
   filter neither dropped everything nor admitted everything).
# Stage 2 — Track Players

## Purpose

Detect and track people across the video, identify the user via a one-time click on the first frame they're visible, project foot positions onto court coordinates, and emit a per-frame parquet for downstream stages.

No silent fallbacks. If the user can't be identified or the user track is lost, the run still completes — but writes a `players_pending.json` listing exactly what's needed to resolve the gap on rerun.

## Inputs

All in the per-video folder:

- `video.mp4` — source video.
- `court.json` — from Stage 1. Stage 2 reads:
  - `homography.image_to_court` — 3×3 matrix (list-of-lists). Applied to image-space foot points to produce court-space coordinates in feet.
  - `court_geometry_feet.width_ft` and `court_geometry_feet.length_ft` — the court rectangle in court-space (typically 20×44 ft). Used directly as the in-court containment region; no polygon math needed since the rectangle is axis-aligned in court-space.
  - `video.fps` — for `t_sec` computation.
  - `user_inputs.court_corners_image` — read for diagnostic logging only. Foot-point containment is performed in court-space, not image-space.
- `court_zones.json` — from Stage 1. Stage 2 reads:
  - `tracking_zone.behind_baseline_ft` and `tracking_zone.beyond_sideline_ft` — scalar buffers (in feet) that extend the court rectangle outward to define the legitimate tracking zone. Stage 2 builds the tracking-zone rectangle as `[-beyond_sideline_ft, width_ft + beyond_sideline_ft] × [-behind_baseline_ft, length_ft + behind_baseline_ft]` in court-space and tests foot points against it.
- `user_clicks.json` — list of user-identification clicks. Schema:
```json
  {
    "clicks": [
      {"frame": 47, "x": 642, "y": 318}
    ]
  }
```
  Initial run requires at least one click. To resolve gaps reported in `players_pending.json`, append additional clicks and rerun.

## Outputs

Written to the per-video folder:

### `players.parquet`

One row per (frame, track_id). Columns:

| Column | Type | Notes |
|---|---|---|
| `frame` | int | 0-indexed frame number |
| `t_sec` | float | frame time in seconds |
| `track_id` | int | ByteTrack track ID |
| `is_user` | bool | True if this track is currently the identified user |
| `user_segment_id` | int (nullable) | Increments each time user identity is re-acquired after a gap. Null for non-user rows. |
| `bbox_x1, bbox_y1, bbox_x2, bbox_y2` | float | Image-space pixels |
| `foot_x, foot_y` | float | Bottom-center of bbox, image-space pixels |
| `court_x_ft, court_y_ft` | float | Foot point projected via homography. NaN if projection is non-finite. |
| `in_court` | bool | `0 ≤ court_x_ft ≤ width_ft AND 0 ≤ court_y_ft ≤ length_ft` |
| `transient` | bool | True if track lifetime < 30 frames OR foot points lie entirely outside the tracking-zone rectangle (which already contains the court) |

### `players_pending.json`

```json
{
  "gaps": [
    {"gap_id": 0, "last_user_frame": 412, "resumes_at_or_after": 442, "reason": "track_lost"}
  ],
  "warnings": []
}
```

Empty `gaps` array → clean run, no re-identification needed. `warnings` carries the doubles sanity-check message if triggered.

## Process

1. Load video, `court.json`, `court_zones.json`, `user_clicks.json`.
2. Run YOLO11s person detection + ByteTrack across all frames.
3. For each detection, compute foot point (bottom-center of bbox), project via `homography.image_to_court` → (`court_x_ft`, `court_y_ft`), and compute `in_court` as `0 ≤ court_x_ft ≤ width_ft AND 0 ≤ court_y_ft ≤ length_ft`.
4. **User identification**: For each entry in `user_clicks.json`, find the YOLO detection on that frame closest (Euclidean) to the click point. That detection's `track_id` becomes the user, starting `user_segment_id = N` where N is the click's index in the array (0 for the first click, 1 for the second, …).
5. **User track propagation**: While that `track_id` persists, mark its rows `is_user=True` with the current `user_segment_id`.
6. **Track-loss detection**: If the user track disappears for more than `TRACK_LOSS_TOLERANCE_FRAMES` (default 30) consecutive frames, close the segment, append a gap entry to `players_pending.json` with `last_user_frame` and `resumes_at_or_after` (last processed frame), and stop emitting `is_user=True` until a later click resolves it.
7. **Re-identification**: A click on a frame after a gap resolves that gap on rerun, with `user_segment_id = previous + 1`.
8. **Non-user tracks**: All other detected persons are recorded with `is_user=False`. Recording continues uninterrupted during user gaps.
9. **Gap rows are absent, not interpolated**: During a user gap, no `is_user=True` rows are emitted. If YOLO redetects the user under a new `track_id`, those rows appear as `is_user=False` until a click reclaims them.
10. **Transient flagging** (post-pass): mark `transient=True` for any track whose lifetime < 30 frames OR whose foot points lie entirely outside the tracking-zone rectangle (computed from `tracking_zone.behind_baseline_ft` and `beyond_sideline_ft`).
11. **Doubles sanity check** (post-pass): count tracks that (a) persist > 5 seconds AND (b) have ≥ 80% of their lifetime's foot points inside the court rectangle. If count > 4, append a warning to `players_pending.json["warnings"]` listing the count and offending `track_id`s. Likely cause: misconfigured `tracking_zone` or adjacent-court contamination.

## Configuration

Hardcoded constants at top of `track.py`:

```python
MODEL = "yolo11s.pt"            # auto-downloads via ultralytics on first run
TRACK_LOSS_TOLERANCE_FRAMES = 30
TRANSIENT_LIFETIME_FRAMES = 30
DOUBLES_PERSIST_SECONDS = 5.0
DOUBLES_IN_COURT_FRAC = 0.80
CLICK_MAX_DISTANCE_PX = 150     # max distance from click to nearest detection
```

## Failure modes (loud, never silent)

- `user_clicks.json` missing, malformed, or has empty `clicks` array → fail with: *"No user clicks provided. Add at least one click to user_clicks.json."*
- Closest detection to a click is > `CLICK_MAX_DISTANCE_PX` away → fail with: *"Click at frame N (x, y) has no detected person within 150 px. Re-click."*
- Homography projection produces non-finite values → write NaN for `court_x_ft`/`court_y_ft` and `False` for `in_court`. Do not interpolate.
- Frame read failure mid-run → fail loudly. No silent truncation.

## CLI

```
python -m stages.track_players.track <video_folder>
```

`<video_folder>` must contain `video.mp4`, `court.json`, `court_zones.json`, `user_clicks.json`. Outputs are written into the same folder.

## Smoke test conditions

`test_track.py` runs against a 30-second clip in `data/test_clip/` and verifies:

1. `players.parquet` exists, is non-empty, and has all 14 expected columns with correct dtypes.
2. At least one row has `is_user=True` (initial click resolved).
3. Within any single `user_segment_id`, all `is_user=True` rows share the same `track_id`.
4. For at least one frame, ≥ 2 distinct `track_id` values are present (multi-person tracking working).
5. Every row with `in_court=True` has finite `court_x_ft` and `court_y_ft`.
6. `players_pending.json` exists, is valid JSON, and every `gap` entry satisfies `last_user_frame < resumes_at_or_after`.
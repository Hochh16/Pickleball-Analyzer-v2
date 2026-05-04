# Stage 1 — Court Calibration

**Status:** Contract approved. Ready to build.

## Purpose

Establish a known geometric relationship between the camera's pixel coordinates and real-world court coordinates (in feet). Every downstream stage depends on this. Without it, "where on the court did this happen?" is unanswerable.

## Inputs

- **`video.mp4`** — the match video. Any resolution, any orientation. The video must show all four court corners and both kitchen lines in the chosen calibration frame.
- **User clicks** — 8 points marked through the calibration UI:
  - 4 court outer corners
  - 2 user-side kitchen line endpoints
  - 2 opponent-side kitchen line endpoints
  - 3 dropdown answers: user_baseline, user_starting_corner, dominant_hand

The user may scrub to a frame other than frame 0 if the first frame has obstructed corners or kitchen lines (e.g. a player blocking a corner during a serve).

## Outputs

Two files in the per-video data folder:

### `court.json`

~~~json
{
  "schema_version": 1,
  "video": {
    "path": "data/match_001/video.mp4",
    "frame_width": 1920,
    "frame_height": 1080,
    "fps": 30.0,
    "frame_used_for_calibration": 0
  },
  "user_inputs": {
    "court_corners_image": [
      [320, 980],
      [1640, 940],
      [1180, 220],
      [180, 240]
    ],
    "kitchen_line_user_image": [
      [420, 540],
      [1340, 530]
    ],
    "kitchen_line_opponent_image": [
      [380, 360],
      [1240, 355]
    ],
    "user_baseline": "near",
    "dominant_hand": "right",
    "user_starting_corner": "left"
  },
  "court_geometry_feet": {
    "width_ft": 20.0,
    "length_ft": 44.0,
    "kitchen_depth_ft": 7.0
  },
  "homography": {
    "image_to_court": [[0,0,0],[0,0,0],[0,0,0]],
    "court_to_image": [[0,0,0],[0,0,0],[0,0,0]]
  },
  "derived": {
    "user_half_polygon_image": [[0,0],[0,0],[0,0],[0,0]],
    "opponent_half_polygon_image": [[0,0],[0,0],[0,0],[0,0]],
    "user_kitchen_polygon_image": [[0,0],[0,0],[0,0],[0,0]],
    "opponent_kitchen_polygon_image": [[0,0],[0,0],[0,0],[0,0]],
    "pixels_per_foot_at_near_baseline": 21.4,
    "pixels_per_foot_at_far_baseline": 8.7
  },
  "validation": {
    "homography_rmse_pixels": 1.3,
    "kitchen_projection_error_user_px": 2.1,
    "kitchen_projection_error_opponent_px": 3.4,
    "warnings": []
  },
  "created_at": "2026-05-02T14:23:11Z"
}
~~~

### `court_zones.json`

~~~json
{
  "schema_version": 1,
  "policy_version": 1,
  "zones": {
    "kitchen_strict": {
      "depth_ft": 7.0,
      "description": "The actual non-volley zone, exactly as marked on the court"
    },
    "kitchen_effective": {
      "depth_ft": 9.0,
      "buffer_ft": 2.0,
      "description": "Kitchen + 2ft buffer counts as 'at the kitchen line' for stat purposes",
      "priority_rule": "If a player is in the buffer zone, count as kitchen, NOT as transition"
    },
    "transition": {
      "near_ft": 9.0,
      "far_ft": 32.0,
      "description": "Between effective kitchen and the rear of the court"
    },
    "baseline_zone": {
      "near_ft": 32.0,
      "far_ft": 44.0,
      "description": "Last 12 feet near the baseline"
    }
  },
  "tracking_zone": {
    "behind_baseline_ft": 8.0,
    "beyond_sideline_ft": 6.0,
    "description": "How far beyond the court lines the player tracker should look for the user. Players serve from behind the baseline and chase wide shots beyond sidelines."
  },
  "in_play_polygon_source": "court.json:derived.user_half_polygon_image and opponent_half_polygon_image",
  "in_play_description": "Ball-bounce IN/OUT determination uses the STRICT court polygons from court.json. A ball that bounces outside these is OUT. A ball hit BEFORE bouncing (player contacts mid-air) can happen anywhere — shot-impact location is not constrained by any court polygon.",
  "created_at": "2026-05-02T14:23:11Z"
}
~~~

## Field-by-field rationale

### `video`

Captured for downstream stages so they don't have to re-probe the video for dimensions/fps. `frame_used_for_calibration` records which frame the user actually marked corners on (may not be frame 0 if they scrubbed past obstructions).

### `user_inputs`

Raw user clicks, in original-video pixel coordinates. Stored verbatim so the calibration is reproducible and auditable.

- `court_corners_image` — 4 points in image-position order: bottom-left, bottom-right, top-right, top-left of the visible court (counter-clockwise starting from lower-left). Image position is unambiguous regardless of camera angle.
- `kitchen_line_user_image` — 2 points on the kitchen line on the user's side. Left endpoint then right endpoint (relative to image).
- `kitchen_line_opponent_image` — same but for the far kitchen line. Required because deriving it from the corners introduces small errors when the camera is at an extreme angle.
- `user_baseline` — `"near"` or `"far"`. Identifies which baseline corresponds to the user's side without ambiguity.
- `dominant_hand` — `"right"` or `"left"`. Used by the shot classifier (Stage 6).
- `user_starting_corner` — `"left"` or `"right"`. Used by the player tracker (Stage 2) to lock onto the correct player on the first valid frame.

### `court_geometry_feet`

Standard pickleball court dimensions. Stored explicitly so downstream code doesn't hardcode the numbers and so we have a place to override for non-regulation courts later.

### `homography`

Two 3x3 matrices.

- `image_to_court` — takes a pixel `(x, y)` and returns court feet `(x_ft, y_ft)` where `(0, 0)` is the user's near-left baseline corner, `(20, 44)` is the far-left baseline corner.
- `court_to_image` — inverse, used by the renderer (Stage 11) and the top-down preview to draw court coords back onto the image.

Computed via `cv2.findHomography` from the 4 corner points. The 4 kitchen points are NOT used to compute the homography — they validate it.

### `derived`

Pre-computed values stored alongside the homography for downstream convenience. Recomputable from the homography but stored to avoid every stage redoing the work.

- `user_half_polygon_image` — 4-point polygon in image coords covering the user's half of the court (from net to user's baseline). Downstream "is point inside user's half" becomes `cv2.pointPolygonTest(poly, (x,y), False) >= 0`.
- `opponent_half_polygon_image` — same for opposite half.
- `user_kitchen_polygon_image` — strict 7ft kitchen polygon on user's side. Note: uses the 7ft court-line definition, NOT the 9ft effective kitchen. The "effective kitchen" calculation is policy and lives in `court_zones.json`.
- `opponent_kitchen_polygon_image` — same for opponent.
- `pixels_per_foot_at_near_baseline` and `..._far_baseline` — perspective foreshortening means pixels/foot varies between the two ends. Knowing both supports bbox-size sanity checks and informs annotation-text sizing.

### `validation`

- `homography_rmse_pixels` — when we project the 4 clicked corners through the homography to court coords and back to pixel coords, the average pixel distance from the original click. Should be near zero. If >5, the user's clicks didn't form a valid quadrilateral and we surface a warning.
- `kitchen_projection_error_user_px` — distance between the user's clicked kitchen line and the kitchen line as derived from the homography. Validates that the corners and kitchen lines are mutually consistent.
- `kitchen_projection_error_opponent_px` — same for the opponent's kitchen line.
- `warnings` — list of strings. Examples: "kitchen line projection error >10px", "kitchen line shorter than 50px (extreme camera angle, accuracy may be reduced)".

### `court_zones.json`

Separate file because it contains policy, not geometry. The "kitchen + 2ft buffer counts as kitchen" rule, the tracking-zone margins, and the in/out-determination rule are all decisions that may be revisited.

- `zones` — court-position zones used by metric stages.
- `tracking_zone` — how far beyond court lines the player tracker should follow players. Used by Stage 2 to compute the actual tracking polygon (which is a larger polygon than the court itself).
- `in_play_polygon_source` and `in_play_description` — documents that ball in/out uses the STRICT court polygons from `court.json`, NOT the effective-kitchen or tracking-zone polygons.

Stages 8-10 read this file. Stage 1 always writes the same default; users can edit `court_zones.json` per video before running downstream stages if they want different thresholds.

## Coordinate semantics across stages

To prevent the kind of cross-stage confusion we saw in v1, here's the rule for every downstream stage:

| Question | Polygon to use | Source |
|---|---|---|
| Is the user IN the kitchen? (for stat purposes) | Effective kitchen (7 + 2 buffer) | Computed at runtime by Stage 8 from `court.json.derived.user_kitchen_polygon_image` + `court_zones.json.zones.kitchen_effective.buffer_ft` |
| Did the ball BOUNCE in/out? | Strict court polygon | `court.json.derived.user_half_polygon_image` + `opponent_half_polygon_image` |
| Should the player tracker LOOK for the user here? | Expanded tracking polygon | Computed at runtime by Stage 2 from `court.json.homography` + `court_zones.json.tracking_zone` |
| Where can a player HIT a ball? | No polygon — anywhere | Players can be anywhere; ball impact is not constrained by court geometry. Only ball-bounce location matters for in/out. |

This table is the contract for downstream stages. Any future stage that needs to reason about court geometry must pick from this table, not invent its own polygon.

## UI Flow (Stage 1 frontend)

A single-page React component, `CourtCalibrator.jsx`. Frame extraction happens in the browser; video upload happens at the end after corners are marked.

1. User picks a video file. Browser extracts the first frame locally using a `<video>` element + `<canvas>`.
2. User answers three quick form questions: dominant hand, user's baseline (near/far), starting corner (left/right). These can be set first or after marking — either order works.
3. User scrubs the timeline to find a frame where all corners and both kitchen lines are visible without obstruction. Default: frame 0. Slider supports frame-by-frame stepping. The currently-selected frame index is recorded for traceability.
4. With a good frame chosen, user marks 8 points in this order:
   1. Court bottom-left corner
   2. Court bottom-right corner
   3. Court top-right corner
   4. Court top-left corner
   5. User-kitchen line left endpoint
   6. User-kitchen line right endpoint
   7. Opponent-kitchen line left endpoint
   8. Opponent-kitchen line right endpoint
5. As points are added, lines connect them so the user can see the court outline forming.
6. Validation runs live. If `homography_rmse_pixels` > 5 or kitchen projection error > 10, a warning appears under the image but doesn't block the user.
7. "Looks right" button uploads the video plus the marker JSON to the backend in one POST. Backend computes homography and writes `court.json` + `court_zones.json` to the per-video data folder.
8. **Top-down preview confirmation step.** Backend returns the calibration result plus a top-down view of the court, generated by `cv2.warpPerspective` using `court_to_image`. This is essentially "what the camera would see if it were directly overhead." A correct calibration produces a clean rectangle with kitchen lines as horizontal lines. A bad calibration produces a visibly distorted court. User clicks "Confirm" to accept, or "Restart" to remark the corners.

### Edge cases handled

- **Misclicks** — Undo and Clear buttons.
- **Rotated video metadata** — frame extraction respects the `<video>` element's natural rendering, which honors orientation.
- **Court partially out of frame** — user clicks the visible corner even at image edges.
- **Extreme camera angle** — kitchen line endpoints close together produces a "shorter than 50px" warning but doesn't block.
- **Player blocking a corner** — user scrubs to a different frame. Calibrator records `frame_used_for_calibration`.
- **Self-intersecting quadrilateral or otherwise nonsensical clicks** — RMSE check catches this. "Looks right" disabled with hint text.
- **Players legitimately outside the court** — not a calibration concern. Captured by the `tracking_zone` margins in `court_zones.json` for use by downstream stages.

### What Stage 1 does NOT do

- No video processing beyond frame extraction.
- No AI/ML.
- No assumptions about court orientation or camera angle.
- No saving/reusing calibrations across videos.
- No database.

## Smoke Test

A pinned test video (the existing v1 sample) with known-correct calibration. Run the CLI tool on it with pre-recorded clicks and verify:

1. The 4 corners + 4 kitchen points produce a `court.json` and `court_zones.json`.
2. Loading the `court.json` and projecting court coord `(0, 0)` → image lands within 2 pixels of the original near-left click.
3. Projecting court coord `(10, 7)` (center of user's kitchen) → image lands inside `user_kitchen_polygon_image` per `cv2.pointPolygonTest`.
4. Projecting court coord `(0, 44)` → image lands within 2 pixels of the far-left click.
5. `validation.homography_rmse_pixels` is < 5.
6. Top-down warp produces a recognizable court rectangle with both kitchen lines visible as horizontal segments.

If all 6 pass, Stage 1 is done.

## Tech Stack

### Backend (Python CLI tool)

- File: `stages/01_calibrate/calibrate.py`
- Single CLI: `python -m stages.01_calibrate.calibrate --video <path> --markers <markers.json> --out-dir <dir>`
- `markers.json` is the file the frontend uploads. It contains the 8 user clicks + 3 form answers.
- Output: writes `court.json` and `court_zones.json` to `<out-dir>`.
- Pure logic, no FastAPI yet. Anyone can call it from a script.

### Backend (FastAPI wrapper)

- File: `stages/01_calibrate/api.py`
- Endpoint: `POST /calibrate`
- Body: multipart form with `video` (file) and `markers` (JSON string).
- Calls `calibrate.py` internally and returns the `court.json` content + a base64 top-down preview image.

### Frontend

- File: `frontend/src/CourtCalibrator.jsx`
- Single React component. Picks video → extract frame → mark points → submit → confirm preview.
- Native `<video>` and `<canvas>` for frame extraction. No video player libraries.
- Posts to `POST /calibrate` and renders the returned top-down preview.

### Dependencies (already in `pyproject.toml`)

- `opencv-python-headless` — `cv2.findHomography`, `cv2.warpPerspective`, `cv2.pointPolygonTest`, video frame I/O
- `numpy` — point math
- `fastapi`, `uvicorn`, `python-multipart` — API server
- `pydantic` — request/response models

No machine learning dependencies in Stage 1.

## Out of scope for Stage 1

- Authentication / user accounts.
- Persistence across videos.
- Editing an existing calibration (rerun = new calibration; old one is overwritten).
# Stage 4 — Track Ball

## Purpose

Detect the pickleball in every frame of a match video and emit a per-frame
record of its pixel-space position. Provides the raw ball-trajectory data
that downstream stages (shot detection, classification, rally segmentation)
consume.

## Inputs

| Arg | Type | Required | Description |
|---|---|---|---|
| `--video` | path | yes | Match video file (`.mp4`). Must be readable by OpenCV. |
| `--court` | path | yes | `court.json` produced by Stage 1 (calibrate). Used for pixel-space ROI filtering. |
| `--weights` | path | yes | TrackNetV2 weights file (`.pt`). No default — must be explicit so weights are swappable across runs. |
| `--out` | path | yes | Output `ball.parquet` path. Stage will not overwrite an existing file unless `--force` is set. |
| `--force` | flag | no | Overwrite existing output if present. |
| `--detection-threshold` | float | no | Heatmap-peak threshold below which a frame is treated as "no detection." Default: `0.5`. Range `[0.0, 1.0]`. Setting to `0.0` accepts every argmax (noisy but maximally permissive); setting to `1.0` accepts nothing. |
| `--max-gap-frames` | int | no | Maximum gap (in frames) to fill via linear interpolation. Default: `5`. Set to `0` to disable interpolation. |
| `--roi-buffer-ft` | float | no | Court ROI buffer in feet, projected to pixel space via the inverse homography. Default: `8.0`. Detections outside court polygon + buffer are discarded. |
| `--device` | str | no | `cpu` or `cuda`. Default: `cpu`. |
| `--log-level` | str | no | `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default: `INFO`. |

## Outputs

### `ball.parquet`

One row per video frame. No frames skipped, no rows missing — if the video
has N frames, the parquet has exactly N rows.

| Column | Type | Description |
|---|---|---|
| `schema_version` | int | `1`. Same value on every row (denormalized for ease of reading). |
| `frame_idx` | int64 | Zero-based frame index, monotonically increasing from 0. |
| `pixel_x` | float64 | Ball x-position in pixels. `NaN` if `visible=False` and `interpolated=False`. |
| `pixel_y` | float64 | Ball y-position in pixels. `NaN` if `visible=False` and `interpolated=False`. |
| `visible` | bool | `True` iff the detection on this frame is a real model output (not interpolated, not missing). |
| `confidence` | float32 | TrackNet heatmap peak value at the detected location, `[0.0, 1.0]`. `NaN` when `visible=False`. |
| `interpolated` | bool | `True` iff `pixel_x`/`pixel_y` were filled by linear interpolation across a gap. Mutually exclusive with `visible`. |

Invariants:
- Exactly one of `visible` and `interpolated` is `True` per row, OR both are
  `False` (true gap — frame had no detection and was not interpolatable).
- When `visible=True`: `pixel_x`, `pixel_y`, `confidence` are all non-NaN,
  and `confidence >= detection_threshold`.
- When `interpolated=True`: `pixel_x`, `pixel_y` are non-NaN, `confidence` is `NaN`.
- When both `False`: `pixel_x`, `pixel_y`, `confidence` are all `NaN`.

### `ball.meta.json`

Sidecar metadata file written next to `ball.parquet`.

```json
{
  "schema_version": 1,
  "video_path": "data/test_clip/video.mp4",
  "video_frame_count": 570,
  "video_fps": 30.0,
  "video_width": 1920,
  "video_height": 1080,
  "court_path": "data/test_clip/court.json",
  "weights_path": "data/models/tracknet_v2_dettor.pt",
  "weights_sha256": "abc123...",
  "device": "cpu",
  "detection_threshold": 0.5,
  "max_gap_frames": 5,
  "roi_buffer_ft": 8.0,
  "stats": {
    "frames_visible": 480,
    "frames_interpolated": 35,
    "frames_missing": 55,
    "detection_rate": 0.842,
    "detections_filtered_by_threshold": 23,
    "detections_filtered_by_roi": 12,
    "max_gap_observed_frames": 7
  },
  "wall_time_seconds": 412.3,
  "stage_version": "0.2.0"
}
```

`detection_rate = (frames_visible + frames_interpolated) / video_frame_count`.
This is the number reported against the smoke-test acceptance threshold.

## Detection method

**TrackNetV2** (heatmap-based small-object tracker). Single forward pass per
frame triple, output heatmap argmax → `(pixel_x, pixel_y)`, peak heatmap
value → `confidence`.

Implementation:
- Use a vendored PyTorch TrackNetV2 port (`mareksubocz/TrackNet`,
  MIT-licensed; see Implementation Notes).
- Load weights from `--weights` path. Architecture must match the weights
  file or load fails loudly.
- Input resolution: model-native (typically 512×288 or 640×360, depends on
  weights). Frames are resized in, coordinates scaled out to original video
  resolution before writing.
- Three-frame input window: `(frame[i-2], frame[i-1], frame[i])` predicts
  ball position in `frame[i]`. The first two frames of the video produce
  no detection (insufficient history); they get `visible=False` and may be
  interpolation-filled if a downstream detection arrives within
  `max_gap_frames`.

### Heatmap-to-coordinate conversion

For each frame's predicted heatmap:
1. Take argmax over the 2D spatial dimensions → `(pixel_x, pixel_y)` in
   model-input coordinates.
2. Read the heatmap value at that location → `confidence` in `[0.0, 1.0]`.
3. If `confidence < detection_threshold`, treat the frame as having no
   detection (`visible=False`, count toward `detections_filtered_by_threshold`).
4. Otherwise, scale `(pixel_x, pixel_y)` from model-input resolution back
   to original video resolution.
5. Apply ROI filter (next section). Detections passing both the threshold
   and ROI become `visible=True` rows.

Default `detection_threshold = 0.5` is a common starting point for sigmoid-
output heatmaps. The smoke-test report includes `detections_filtered_by_threshold`
so threshold tuning is data-informed: if many real-looking detections are
being dropped at 0.5, lower it; if the parquet contains many low-quality
detections that visibly aren't the ball, raise it.

## ROI filtering

Pixel-space polygon, derived once at startup from `court.json`:

1. Read four court corners in court-coords (feet) from `court.json`.
2. Expand by `roi_buffer_ft` in court-coord space (so polygon is the court
   plus a buffer for in-air balls and approaches behind the baseline).
3. Project each expanded corner to pixel space using the inverse homography
   from `court.json`.
4. Result is a pixel-space quadrilateral. Use `cv2.pointPolygonTest` per
   detection.

Detections whose `(pixel_x, pixel_y)` falls outside the polygon are
discarded **before** the visibility/interpolation logic runs. They count
toward `detections_filtered_by_roi` in stats but produce no row data.

The polygon is derived from court corners on the ground plane. In-air balls
high above the court could project to pixel locations outside this polygon
in extreme cases (very high lobs near the camera). The `roi_buffer_ft`
default of 8 ft is a pragmatic compromise — adequate for typical play,
tunable if smoke-test inspection shows real balls being filtered.

## Smoothing / gap-filling

Two-stage post-processing applied to the per-frame raw detections:

1. **Per-frame greedy selection.** For each frame, take the single highest-
   confidence detection that passed both threshold and ROI. Set `visible=True`
   for that row. If no qualifying detection, mark `visible=False` for now.

2. **Linear interpolation across short gaps.** Walk the time series. For
   any run of consecutive `visible=False` frames of length `L ≤ max_gap_frames`
   bounded on both sides by `visible=True` frames, fill the gap by linearly
   interpolating `pixel_x` and `pixel_y` from the bounding visible frames.
   Set `interpolated=True` and `confidence=NaN` for filled rows. Gaps longer
   than `max_gap_frames`, or gaps at the very start / very end of the video
   (no bounding visible frame on one side), remain unfilled (`visible=False`,
   `interpolated=False`, all numeric columns `NaN`).

No Kalman filter, no parabolic motion model. Trajectory smoothing
(physics-aware fits, bounce detection, etc.) is the responsibility of
downstream stages. Stage 4 emits clean per-frame data with explicit holes
where it didn't see the ball, and short gaps filled to avoid noisy holes
in the trajectory consumed by Stage 5+.

## Edge cases

- **Video unreadable / corrupt.** Raise `FileNotFoundError` or `RuntimeError`
  with the offending path. Do not write a partial parquet.
- **`court.json` missing or malformed.** Raise `ValueError` with a clear
  message naming the missing/invalid field. Do not write output.
- **Weights file missing or unloadable.** Raise `RuntimeError` with the
  weights path and the underlying load error. Do not silently fall back to
  any default or random weights.
- **Weights architecture mismatch.** Raise on the PyTorch state-dict load
  error. Do not attempt partial loads.
- **Video < 3 frames.** Raise `ValueError`. TrackNetV2 requires a 3-frame
  input window.
- **Output file exists and `--force` not set.** Raise `FileExistsError`.
- **All frames produce no detection.** Stage still completes successfully
  and writes a parquet with all rows `visible=False`, `interpolated=False`,
  numeric columns NaN. `detection_rate` in meta is `0.0`. Logs WARNING.
- **CUDA requested but unavailable.** Raise `RuntimeError`. Do not silently
  fall back to CPU.
- **Court polygon degenerate** (e.g., self-intersecting after projection).
  Raise `ValueError` and refuse to start. Court calibration is upstream and
  must be valid.
- **Interpolation across a sequence boundary** (camera cut, scene change).
  Stage 4 has no scene-change detection — it will interpolate across cuts
  if the gap is short enough. This is a known limitation; rallies are
  segmented in Stage 7, which has the context to reject pre-rally
  interpolation.
- **`detection_threshold` outside `[0.0, 1.0]`.** Raise `ValueError`.

## Failure mode summary

All failures raise loudly with descriptive messages. No silent fallbacks,
no dummy outputs, no partial files. Consistent with project rule "Failures
are loud" (ARCHITECTURE.md).

## Smoke test

**Clip:** `data/test_clip/video.mp4` (~2 min, indoor, mixed shots — dinks,
smashes/lobs, drops). Same clip used by Stages 1–3.

**Acceptance criteria:**
1. Stage runs to completion without exception.
2. `ball.parquet` and `ball.meta.json` are produced.
3. Schema invariants (above) hold on every row.
4. `detection_rate ≥ 0.80` measured against the **active-rally subset** of
   the clip. Active-rally frame ranges are listed in
   `data/test_clip/active_rally_frames.json` (a small hand-labeled
   sidecar — frame index ranges where the ball is in active play, excluding
   serves-not-yet-struck and dead time after a fault).
5. Visual sanity: the trajectory rendered onto a sample frame from each
   rally looks plausible (no obvious teleports, no obvious adjacent-court
   contamination).

**Reporting:** smoke-test run produces `data/test_clip/ball.smoke.txt` with
the meta.json stats summary plus pass/fail verdict. The smoke test is
implemented as a Python script at `stages/track_ball/smoke_test.py`,
re-runnable after weights swaps or parameter changes.

## Stage version

`0.2.0` for initial implementation. (Bumped from `0.1.0` during contract
review when `--detection-threshold` was added. No code was written against
`0.1.0`; bump preserves the convention that schema/contract changes get
visible version increments.)

Increment minor for behavior changes that preserve schema; increment
`schema_version` for any breaking parquet schema change.

## Implementation notes

(Not part of the contract proper, but recorded here so the next pass of
"code" doesn't re-litigate them.)

- **Choice of PyTorch TrackNetV2 port:** `mareksubocz/TrackNet`. Verified
  V2-architecture (3-in-3-out via `--one_output_frame=False` mode), MIT-
  licensed, parameterized cleanly enough for Dettor-weights compatibility.
  The repo is archived (Sep 2024) but that's fine: we vendor the model
  class into `stages/track_ball/_tracknet_model.py` rather than depending
  on the package. yastrebksv/TrackNet was rejected: V1-architecture,
  no LICENSE.
- **Weights conversion:** Dettor's published weights are Keras `.h5`. A
  one-time conversion script (`tools/convert_dettor_weights.py`) loads
  the `.h5`, maps tensors to PyTorch state-dict keys, saves `.pt`. This
  conversion script lives outside `stages/track_ball/` because it is a
  one-time tool, not a stage.
- **Frame batching:** for CPU inference, process frame-triples one at a
  time; batch size > 1 has minimal benefit on CPU. For CUDA, allow batch
  size up to 8.
- **Memory:** the full video does not need to be held in memory. Use a
  rolling 3-frame buffer; write parquet in chunks of 1000 rows.

## Out of scope (deferred to later stages)

- Bounce detection (Stage 5).
- Trajectory parabolic fitting / z-height inference (Stage 5+).
- Court-coordinate projection of the ball (Stage 5; only meaningful at
  ground-contact frames).
- Adjacent-court ball rejection beyond pixel-space ROI (downstream — same
  problem as adjacent-court player contamination noted in
  `KNOWN_ISSUES.md`, addressed there if it surfaces).
- Multi-ball handling (out of scope; pickleball is single-ball by rule).
- Fine-tuning weights on user's own footage (Stage 4.5).

## Known follow-ups

- **Generalization.** Dettor's weights were trained on PPA Tour broadcast
  footage. Real performance on user-shot footage (different lighting,
  court colors, camera position) is unknown until measured. Plan: after
  Stage 4 passes smoke test on `test_clip`, run on the other corpus
  videos (same setting/different opponents; same lighting/different
  court; outdoor/different lighting/different court). Document detection
  rates. If significant drops, trigger Stage 4.5 (fine-tuning).
- **TrackNetV3 weights.** No public pickleball-trained V3 weights as of
  contract drafting. Re-check quarterly. If they become public, evaluate
  against the V2 baseline established here.
- **Camera assumption.** Documented in `ARCHITECTURE.md` (Pipeline-wide
  assumptions § Camera placement). Stage 4 inherits this assumption and
  does not re-validate it.
## Candidate + continuity tracking (2026-07-21) — adjacent-court fix

**Symptom (match clip `pb_5_minute_outdoor-2`):** in-rally ball visibility only
**78.3%**, with **128 in-rally teleports** >200 px/frame. The track flip-flopped
between our ball and an object parked at ~(2631,1030) — *above* our court's image
region, i.e. a neighbouring court.

**Root cause:** inference took the heatmap's single **global `argmax`** per frame and
kept it only if it cleared `conf_thresh` (0.30). So (a) whenever a neighbouring
court's ball produced the stronger peak the track jumped there, with the real ball
never recorded as an alternative, and (b) a real-but-faint ball below 0.30 was
discarded outright. There was **no temporal continuity** at all — each frame decided
independently. The existing outlier filter only drops a detection far from *both*
neighbours, so alternating runs of 2+ frames survived.

**Fix:**
1. `topk_peaks()` — keep the top-`k` LOCAL maxima per frame (peak, suppress its
   neighbourhood, repeat) down to `--cand-floor` (0.15), *below* the accept
   threshold, so alternatives and faint balls are recorded.
2. `select_track()` — Viterbi-style DP over candidates choosing the single most
   plausible trajectory. Score = summed confidence − motion penalty (hard-gated at
   `--max-step-px`) − **stationarity penalty** − skipped-frame penalty, with a
   `--restart-cost` for re-acquiring after a real loss.
   > **The stationarity penalty is essential.** Penalising motion alone makes a
   > PARKED object the "smoothest" possible track, so the contaminant wins outright
   > (verified: it took 60/60 frames before this term was added). A ball in play is
   > never parked — measured median motion is ~6.7 src px/frame.
3. **Acceptance by temporal support** — a pick clearing `--conf` is trusted; a weaker
   pick is kept only when it sits on the track within `WEAK_SUPPORT_GAP` of an
   accepted one. That recovers faint real balls without promoting isolated noise.

Defaults preserve old behaviour at `--topk 1 --cand-floor 0.30`. Covered by 6 unit
tests (parked contaminant, weak-ball recovery, isolated-noise rejection, impossible
jump, top-k peak extraction/suppression). **Requires a GPU re-run to take effect** —
the change is in inference, so existing `ball.parquet` files are unaffected.

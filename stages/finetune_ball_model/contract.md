# Stage 4.5 - Ball-Detection Calibration

## STATUS: PAUSED (May 2026)

This contract describes the v3 (classical CV) approach. v3 was attempted
end-to-end on test_clip and produced a tune accuracy of ~1% (1/100
isolated-blob frames), well below the 80% acceptance criterion. See
KNOWN_ISSUES.md section 'Stage 4.5 - Ball detection PAUSED after three
failed attempts' for full diagnostic findings.

**Stage 4.5 is paused, not abandoned.** Two conditions would justify
re-activating it:

1. **Better source video** raises the SNR enough that the existing v3
   tooling produces usable detections. Specifically: higher camera mount
   (10-15 ft), 4K and/or 60 fps recording, faster shutter, simpler
   backgrounds. The existing `tune_ball_cv.py` can be re-run on improved
   footage without code changes; if tune accuracy rises above ~50% on
   a new test clip, v3 is likely viable.

2. **Algorithmic extension** beyond per-frame detection - specifically,
   multi-frame trajectory tracking that scores candidate detections
   across windows of 5-10 frames using ball physics (constant velocity
   between bounces, gravity-influenced arc, smooth trajectory). This
   is the standard fix for low-SNR ball tracking and is the next
   technical avenue if better video alone doesn't suffice.

**While Stage 4.5 is paused, downstream stages (Stage 5+) are being
built using a synthetic placeholder ball.parquet** (clean trajectories
generated from known shot patterns). This is documented in the Stage 5
contract (when written). The placeholder approach exposes what
downstream stages actually require from ball data, which will inform
the v4 ball detector design.

The rest of this document describes v3 as it was specified and
attempted. None of v3's outputs are currently in use downstream.

---


## Purpose

Produce per-video CV-pipeline parameters that let Stage 4 (track ball)
detect the pickleball reliably on the user's footage. Stage 4 has been
rewritten to use classical computer vision (background subtraction +
blob detection + court-ROI filter + trajectory smoothing) instead of
TrackNetV2 deep-learning inference. Stage 4.5 is the per-video
calibration step that produces the tuned parameters Stage 4 reads.

This stage replaces the original deep-learning fine-tuning Stage 4.5,
which failed twice (v1: fine-tune Dettor's weights; v2: train from
scratch). See "v1 and v2 lessons learned" at the bottom of this
document. The fundamental issue was that TrackNetV2 was the wrong
architectural choice for amateur phone-camera footage at low effective
resolution against busy backgrounds. Classical CV exploits the
static-camera assumption directly (background subtraction is nearly
free signal) and is better suited to this footage profile.

Stage 4.5 is run ONCE PER VIDEO (alongside `mark_court.py`), not once
per project. New videos at new venues require new calibration. The
calibration tool is interactive and takes ~2-3 minutes of operator
time per video.

## Place in the architecture

Slots between Stage 1 (calibrate) and Stage 4 (track ball). Per-video
setup workflow:
1. Stage 1 (`mark_court.py`) - produces `data/<video>/court.json`
2. Stage 4.5 (`tune_ball_cv.py`) - produces `data/<video>/ball_cv_params.json`
3. Stages 2-4 run on the video using the artifacts above.

Stage 4.5 does NOT modify Stage 4 or any other stage. Stage 4 reads
`ball_cv_params.json` from the video's folder; the only contract
between Stage 4.5 and Stage 4 is the schema of that file.

This stage is documented in ARCHITECTURE.md as Stage 4.5.

## Sub-pieces

| Sub-piece | Purpose | Inputs | Outputs |
|---|---|---|---|
| Labeling tool | Produce GT for validation | Video | `data/<video>/ball_labels.json` |
| Tuning tool   | Calibrate CV params       | Video + court.json + labels (optional) | `data/<video>/ball_cv_params.json` |
| Validation    | Measure detection rate    | Video + labels + cv_params | `data/<video>/ball_validation_report.json` |
| Stage 4 smoke test | End-to-end Stage 4 check | (already exists in Stage 4) | Pass/fail verdict |

The labeling tool (`tools/label_ball.py`) is unchanged from the
previous Stage 4.5 contract; its purpose has shifted from "produce
training data for the deep model" to "produce ground truth for
validating CV detection." Per-video minimum of 200 labels still
applies. The existing labels (10,701 across 4 videos) are reusable
without re-labeling.

## Sub-piece 1: Labeling tool

`tools/label_ball.py` - desktop Tkinter UI for clicking ball locations
in video frames.

Specification unchanged from v0.1.0. See git history or KNOWN_ISSUES
for the original schema. Output file: `ball_labels.json` per video.
The labels produced for v0.1.0 are valid input for the validation
sub-piece below.

## Sub-piece 2: Tuning tool (the heart of v3 Stage 4.5)

`stages/finetune_ball_model/tune_ball_cv.py` - interactive desktop tool
that calibrates the CV-pipeline parameters for a specific video.

### CLI args

| Arg | Type | Required | Description |
|---|---|---|---|
| --video | path | yes | Video file to calibrate against. |
| --court | path | yes | court.json for the video (from Stage 1). |
| --labels | path | no  | ball_labels.json for the video. If provided, the tool will skip the manual click step and seed thresholds from labels directly. |
| --out | path | yes | Output ball_cv_params.json path. |
| --n-calibration-frames | int | no | Number of frames to sample for calibration. Default: 20. |
| --force | flag | no | Overwrite --out if it exists. |

### What it does (interactive flow)

1. Opens the video and computes a median-frame background image from
   ~100 sampled frames (used as the static-background reference).
2. Samples N (default 20) candidate frames at roughly even intervals
   through the video.
3. For each sampled frame:
   - Subtracts the background to highlight moving objects.
   - Filters to connected components within the court ROI (court.json
     defines the polygon).
   - Highlights all candidate blobs that pass loose default filters
     (area in [3, 200] px^2; circularity > 0.3).
   - Asks operator: click on the actual ball if visible, or right-click
     for "ball not visible / off-court."
4. If --labels is provided, skips the interactive click loop and uses
   ground-truth labels for the sampled frames directly.
5. From the collected (frame, ball_pixel) pairs, computes:
   - Tight bounds on blob area (5th to 95th percentile observed).
   - Tight bounds on circularity (5th percentile observed).
   - Color characteristics of the ball (median HSV).
   - Background-subtraction sensitivity threshold (set so the ball
     consistently exceeds the threshold).
   - Motion threshold (frame-to-frame pixel displacement bounds for
     valid ball motion).
6. Validates the tuned parameters by re-running blob detection on the
   sampled frames and showing the operator a confirmation panel: for
   each frame, the tool's top candidate is shown vs the operator's
   click. Operator approves or aborts.
7. On approval, writes `ball_cv_params.json`.

### Output schema (ball_cv_params.json)

    {
      "schema_version": 1,
      "video_path": "data/test_clip/video.mp4",
      "video_width": 1920,
      "video_height": 1080,
      "video_fps": 30.0,
      "background_method": "median",
      "background_n_frames": 100,
      "bg_subtraction_threshold": 25,
      "blob_area_px_min": 4.0,
      "blob_area_px_max": 60.0,
      "blob_circularity_min": 0.45,
      "ball_color_hsv_median": [30, 180, 200],
      "ball_color_hsv_tolerance": [10, 60, 60],
      "motion_displacement_px_per_frame_min": 1.0,
      "motion_displacement_px_per_frame_max": 80.0,
      "calibration_method": "click",
      "n_calibration_frames_used": 20,
      "calibration_completed_at_utc": "2026-05-20T12:34:56Z",
      "stage_version": "0.3.0"
    }

### Constraints

- Coordinates and thresholds are in original-video pixel space.
- If --labels is provided, --n-calibration-frames is ignored; tool uses
  up to 50 labeled frames (chosen to span the video's time range).
- Tool fails loudly with no silent fallback if the video can't be
  opened or if court.json's homography is degenerate.
- Tool can be re-run with same --out to overwrite (with --force) and
  recalibrate.

## Sub-piece 3: Validation

`stages/finetune_ball_model/validate.py` - runs Stage 4's CV pipeline
against a video using the tuned parameters and compares predictions
against ground-truth labels.

### CLI args

| Arg | Type | Required | Description |
|---|---|---|---|
| --video | path | yes | Video to validate against. |
| --court | path | yes | court.json for the video. |
| --cv-params | path | yes | ball_cv_params.json for the video. |
| --labels | path | yes | ball_labels.json for the video. |
| --out | path | yes | Output ball_validation_report.json path. |
| --force | flag | no | Overwrite --out. |

### What it does

1. Loads court, cv_params, labels.
2. Runs Stage 4's CV inference path on every frame with a labeled
   entry.
3. For each labeled frame, compares predicted ball location to
   ground truth.
4. Reports detection rate (% of labeled ball_visible=true frames where
   prediction is within 10 px of GT) and false-positive rate (% of
   labeled ball_visible=false frames where any prediction was made).
5. Writes JSON report.

### Output schema (ball_validation_report.json)

    {
      "schema_version": 1,
      "video_path": "...",
      "cv_params_path": "...",
      "labels_path": "...",
      "n_labeled_frames": 320,
      "n_ball_visible": 240,
      "n_ball_invisible": 80,
      "detection_rate_at_10px": 0.85,
      "detection_rate_at_25px": 0.92,
      "false_positive_rate": 0.06,
      "median_pixel_error": 4.3,
      "p95_pixel_error": 18.7,
      "wall_time_seconds": 145.7,
      "evaluated_at_utc": "..."
    }

## Sub-piece 4: Stage 4 smoke test re-run

`python -m stages.track_ball.smoke_test` with the new CV-based Stage 4
in place. Pre-existing acceptance criteria from Stage 4's (rewritten)
contract apply.

## Acceptance criteria

Stage 4.5 passes if and only if:

1. All four videos in the corpus have ball_labels.json with >= 200
   labels each (already satisfied: 2708, 2365, 2607, 3021).
2. The tuning tool runs end-to-end on each video and produces a valid
   ball_cv_params.json.
3. The validation tool reports detection_rate_at_10px >= 0.80 on the
   held-out outdoor video using its tuned parameters.
4. Stage 4 smoke test passes (detection_rate >= 0.80 on
   data/test_clip/) using the new CV pipeline and the test_clip's
   tuned parameters.

If criterion 3 fails (outdoor doesn't hit 80%), the tuning tool needs
work or the CV pipeline parameters need additional dimensions
(e.g., per-region thresholds for outdoor lighting variation). Do NOT
declare Stage 4.5 done with parameters that fail acceptance.

If criterion 4 fails (test_clip doesn't hit 80%), the CV pipeline
itself in Stage 4 needs work. This points back to Stage 4's contract,
not Stage 4.5's.

## Smoke test

`stages/finetune_ball_model/smoke_test.py` - thin wrapper that:
1. Confirms all four ball_labels.json exist with >= 200 labels.
2. Confirms all four ball_cv_params.json exist and parse against
   schema.
3. Runs validate.py against the outdoor video.
4. Runs Stage 4 smoke test.
5. Pass if both validation report and Stage 4 smoke test pass.

## Stage version

0.3.0 for the CV-based rewrite. (v0.1.0 = fine-tune Dettor; v0.2.0 =
train from scratch; both failed. v0.3.0 abandons deep learning.)

## Out of scope

- A UI walkthrough that sequences calibrate -> tune -> process -> render
  as a single guided flow. This will be built as a wrapper on top of
  the CLI tools after all pipeline stages are complete and individually
  tested. CLI tools should expose clear error messages, consistent
  argument naming, and progress reporting so the wrapper has solid
  pieces to compose.
- Auto-tuning (no operator clicks). Possible future enhancement; the
  current interactive flow is the deliberate choice for v1.
- Per-region thresholds within a video. Currently parameters are
  global per video. If a video has wildly varying lighting across
  regions, this may need extension.
- Multi-camera or multi-court calibration. Per-video for now.

## Pipeline-wide assumptions inherited

Same as Stage 4 (camera placement, single ball, single match, etc.) -
documented in ARCHITECTURE.md section Pipeline-wide assumptions. The
"static camera" assumption is now load-bearing for the background
subtraction approach; previously it was load-bearing for the
homography but not for ball detection.

## v1 and v2 lessons learned (May 2026)

The original Stage 4.5 plan was deep-learning-based, in two attempts:

### v1: Fine-tune Dettor's TrackNetV2 weights

- 6 epochs on Colab T4, weighted BCE (pos_weight=100), Adam lr=1e-4.
- Best val loss at epoch 1; trained 5 more epochs with no improvement.
- Validation: detection_rate_at_10px=0.32, false_positive_rate=1.00.
- Diagnostic showed: on 20 visible-ball frames, the real ball was NOT
  in the model's top-5 candidate peaks in 13 cases (65%). Two fixed
  pixel locations (tree branches in the distant background) appeared
  as top-5 peaks on nearly every frame.
- Root cause: BCE with high pos_weight made "confidently wrong"
  locally optimal; static camera + no spatial augmentation let the
  model memorize fixed background features.

### v2: Train TrackNetV2 from scratch with MSE + spatial augmentation

- 15 epochs on Colab A100, MSE loss, Adam lr=1e-3 with cosine decay,
  random rotation +/-5deg, random translation +/-10%, weight decay
  1e-5.
- Val loss flatlined at ~0.000078 from epoch 10 onward.
- The MSE-on-zeros math: target heatmap is ~99.97% zeros; "predict 0
  everywhere" achieves MSE ~0.00007, which is exactly where the model
  settled. The model found the trivial all-zero local minimum and
  could not escape.
- Root cause: MSE on sparse positive targets has a "predict nothing"
  trivial solution. Symmetric to v1's failure: v1 made wrong-confident
  optimal; v2 made nothing-confident optimal.

### Why deep learning was the wrong choice

TrackNetV2 was designed for broadcast tennis and badminton:
mast-mounted cameras 15-30 feet up, 4K resolution where the ball
spans 8-12 pixels, clean uniform backgrounds (blue or green hard
court), tens of thousands of training labels per model, stable
lighting and ball appearance.

Our footage violates all of those:
- Camera at ~6 feet, near player head height.
- Phone camera 1080p; ball is 4-6 pixels.
- Backgrounds include trees, fences, light fixtures, windows -
  high-frequency content that looks "ball-like" at low resolution.
- ~2,500 labels per visual environment (thin for CNN feature
  learning at small scale).
- Lighting varies across indoor/outdoor and venue-to-venue.

Classical CV exploits the static-camera assumption directly via
background subtraction (the ball is the foreground; the background
is everything that doesn't move). This is essentially free signal
that deep learning cannot use. It also handles low resolution well
(a 4-pixel ball is a clear blob after background subtraction) and
extends to new venues via the calibration tool (Sub-piece 2) rather
than via collecting new labels and retraining.

### v1 and v2 artifacts retained

- `data/models/tracknet_v2_finetuned_v1.pt` - v1 failed weights.
- `data/training/validation_report.json` - v1 validate.py output.
- `data/training/diag_v1/*.png` - v1 diagnostic visualizations.
- `tracknet_v2_trained_v1.pt` - NOT saved; v2 was aborted before final
  save because training had collapsed to all-zeros.
- `train_log_v1.json` on Drive - v2 training log up to epoch 25.

All retained for reference but unused downstream. The
`_tracknet_model.py` vendored copy in `stages/track_ball/` will be
deleted when Stage 4 is rewritten in CV.

### General lessons

1. When fine-tuning, verify the source model's training data matches
   your problem's characteristics before assuming the priors will
   transfer. Dettor trained on PPA broadcast footage; we used amateur
   phone footage; the priors were actively misleading.
2. For sparse-positive detection problems, both raw weighted BCE
   (high pos_weight) and raw MSE have known failure modes that look
   different but produce equally useless models. Focal loss is the
   standard remedy if deep learning is required, but consider
   whether deep learning is the right tool at all before reaching
   for it.
3. Match the tool to the data, not the problem to the tool. We
   anchored on "TrackNet because it's pickleball" rather than asking
   "what does this footage actually look like and what method fits
   it." Static-camera amateur phone footage is closer to a 1990s
   surveillance CV problem than to a 2020s broadcast-sports CV
   problem.
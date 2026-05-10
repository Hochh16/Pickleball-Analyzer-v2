# Stage 4.5 — Fine-Tune Ball-Tracking Model

## Purpose

Produce TrackNetV2 weights that detect the pickleball reliably on the
user's footage. Stage 4 is mechanically correct but Andrew Dettor's
pre-trained pickleball weights do not generalize to the user's camera
setup, lighting, court colors, or ball appearance — verified by
multi-frame diagnostic inspection in May 2026. Stage 4.5 closes the
generalization gap by fine-tuning Dettor's weights on user-labeled
frames from the user's own videos.

This stage is offline (training runs on Colab GPU) and one-shot per
weights version. It is not part of the per-video processing pipeline.

## Place in the architecture

Slots between Stage 4 (track ball) and Stage 5 (detect shots) but does
not modify Stage 4 or Stage 5. Stage 4.5's only interface to the rest
of the pipeline is the .pt weights file at
data/models/tracknet_v2_dettor.pt (or another path passed to Stage 4
via --weights). When Stage 4.5 produces a new weights file, Stage 4's
--weights argument points at it; no other code changes.

This stage is added to ARCHITECTURE.md as Stage 4.5 (between 4 and 5),
extending the pipeline diagram from 11 to 12 stages.

## Inputs

The stage covers six sub-pieces, each with its own inputs and outputs.
The full pipeline is: label videos -> prepare training data -> train
-> validate -> swap weights -> re-run Stage 4 smoke test.

| Sub-piece | Inputs | Outputs |
|---|---|---|
| Labeling tool | Video file | data/<video>/ball_labels.json |
| Data prep | Multiple ball_labels.json + their videos | data/training/{train,val}/ |
| Training | Prepared dataset + Dettor's weights | data/models/tracknet_v2_finetuned_v1.pt |
| Validation | Held-out video + new weights | data/training/validation_report.json |
| Weights swap | New .pt file | (operator action; no code) |
| Stage 4 smoke test | (already exists in Stage 4) | Pass/fail verdict |

## Sub-piece 1: Labeling tool

tools/label_ball.py — desktop Tkinter UI for clicking ball locations
in video frames.

CLI args:

| Arg | Type | Required | Description |
|---|---|---|---|
| --video | path | yes | Video file to label. |
| --out | path | yes | Output ball_labels.json path. Resumes mid-label if file exists and is non-empty. |
| --sample-every | int | no | Label every Nth frame. Default: 3. Set to 1 to label every frame. |
| --start-frame | int | no | Frame to start labeling from. Default: 0. |
| --end-frame | int | no | Frame to stop labeling at (inclusive). Default: end of video. |

UX (per-frame):
- Window shows the current frame, downsampled if necessary to fit screen, with a reticle following the cursor.
- Frame index, total to label, and progress percentage shown in title bar.
- Left-click = mark ball at click position, advance to next sampled frame.
- Spacebar or right-click = mark "ball not visible," advance to next sampled frame.
- Backspace or left arrow = go back to previous labeled frame (in case of misclick).
- Esc or window-close = save and quit.
- Auto-saves to --out every 25 labels and on quit. Crash-safe.

Output schema (ball_labels.json):

    {
      "schema_version": 1,
      "video_path": "data/test_clip/video.mp4",
      "video_frame_count": 3843,
      "video_fps": 30.0,
      "video_width": 1920,
      "video_height": 1080,
      "sample_every": 3,
      "labels": [
        {"frame_idx": 0,   "ball_visible": true,  "pixel_x": 1234.5, "pixel_y": 678.9},
        {"frame_idx": 3,   "ball_visible": false, "pixel_x": null,    "pixel_y": null},
        {"frame_idx": 6,   "ball_visible": true,  "pixel_x": 1240.0, "pixel_y": 670.0}
      ],
      "started_at_utc": "2026-05-08T15:00:00Z",
      "last_saved_at_utc": "2026-05-08T15:42:00Z"
    }

Constraints:
- Coordinates are in original-video pixel space, not downsampled UI space.
- pixel_x / pixel_y are null iff ball_visible is false.
- labels array is sorted by frame_idx ascending.
- No duplicate frame_idx values.

## Sub-piece 2: Training data preparation

stages/finetune_ball_model/prepare_training_data.py — converts
ball_labels.json files plus their videos into TrackNetV2 training format.

CLI args:

| Arg | Type | Required | Description |
|---|---|---|---|
| --labels | path (multiple) | yes | One or more ball_labels.json files. Repeat the flag per file. |
| --out-dir | path | yes | Output directory for prepared dataset. |
| --val-video | str | yes | Substring matching exactly one of the input video paths. That video's data goes to validation; everything else to training. Example: --val-video outdoor. |
| --heatmap-sigma | float | no | Gaussian heatmap radius in pixels (model-input scale). Default: 2.0. |
| --force | flag | no | Overwrite --out-dir if it exists. |

What it produces: for each labeled frame, three consecutive video
frames are loaded and resized to the model's input resolution
(288x512). A 3-channel heatmap target is generated with a Gaussian
peak centered on the labeled ball position (or all-zero if
ball_visible=false). The frame triple and heatmap target are saved as
a single .npy file per training sample.

Output structure:

    data/training/
        train/
            000000.npy        # shape (12, 288, 512): 9 input + 3 target channels
            000001.npy
            ...
        val/
            000000.npy
            ...
        metadata.json         # per-sample lookup: source video, frame_idx, label

metadata.json schema:

    {
      "schema_version": 1,
      "n_train": 1240,
      "n_val": 320,
      "heatmap_sigma": 2.0,
      "model_input_h": 288,
      "model_input_w": 512,
      "val_source_video": "data/outdoor/video.mp4",
      "samples": [
        {"split": "train", "idx": 0, "source_video": "...", "source_frame_idx": 9, "ball_visible": true, "pixel_x": 1234.5, "pixel_y": 678.9},
        ...
      ],
      "created_at_utc": "..."
    }

Constraints:
- The labeled frame is the third of the triple (frame_idx-2, frame_idx-1, frame_idx), matching Stage 4's inference convention.
- Frames where frame_idx < 2 are skipped (insufficient history).
- Frames where ball_visible=false produce all-zero heatmap targets — these are NEGATIVE samples training the model to output near-zero where there's no ball. Critical for reducing false positives.
- Coordinates from the JSON are in original video resolution; data prep scales them to the 288x512 model input space when generating heatmaps.

## Sub-piece 3: Training (Colab notebook)

stages/finetune_ball_model/finetune.ipynb — Jupyter notebook designed
to run on Google Colab with a T4 GPU on the free tier.

Inputs (uploaded to user's Google Drive):

| Item | Drive location |
|---|---|
| Dettor's converted .pt weights | MyDrive/pickleball/tracknet_v2_dettor.pt |
| Prepared training data (zipped) | MyDrive/pickleball/training_data.zip |
| Vendored model code | MyDrive/pickleball/_tracknet_model.py |

What the notebook does:
1. Mounts Google Drive.
2. Unzips training data to Colab local disk.
3. Loads the vendored TrackNet model class.
4. Loads Dettor's weights as starting point.
5. Trains for N epochs with weighted BCE + Adam, validating each epoch.
6. Saves best-val-loss weights as tracknet_v2_finetuned_v<N>.pt.
7. Writes a training-log JSON with per-epoch loss/accuracy.
8. Copies weights and log back to user's Drive.

Hyperparameters (in the notebook, easy to edit):
- Optimizer: Adam, lr=1e-4 (lower than from-scratch training because we're fine-tuning).
- Loss: weighted binary cross-entropy with positive-weight scaling (heatmap pixels are sparse; positives need up-weighting).
- Batch size: 4 (constrained by T4 GPU memory at 288x512 input).
- Epochs: 30 default, with early stopping on val loss plateau (patience=5).
- Augmentation: random horizontal flip; brightness/contrast jitter; minor RGB shift. No spatial cropping (would change the camera-position assumption).

Outputs (back to Drive):

| Item | Drive location |
|---|---|
| Best fine-tuned weights | MyDrive/pickleball/tracknet_v2_finetuned_v<N>.pt |
| Training log | MyDrive/pickleball/finetune_log_v<N>.json |

Constraints:
- Notebook must run end-to-end in a single Colab session (~3 hours max on free tier).
- All logging is to the notebook output and to the JSON log; no wandb/tensorboard signup required.
- Notebook is VERSIONED IN GIT (committed as a .ipynb file). Re-running it should reproduce results modulo training stochasticity.

## Sub-piece 4: Validation

stages/finetune_ball_model/validate.py — runs the fine-tuned weights
against the held-out validation video and produces a detection-rate
report.

CLI args:

| Arg | Type | Required | Description |
|---|---|---|---|
| --weights | path | yes | Fine-tuned .pt. |
| --video | path | yes | Held-out validation video. |
| --court | path | yes | court.json for the validation video. |
| --labels | path | yes | ball_labels.json for the validation video. |
| --out | path | yes | Output validation_report.json path. |

What it does:
1. Runs Stage 4's inference path on the validation video with the new weights.
2. For each labeled frame, compares predicted ball location to ground-truth.
3. Reports detection rate (% of labeled ball_visible=true frames where prediction is within 10 pixels of ground truth at original resolution) and false-positive rate (% of labeled ball_visible=false frames where prediction has confidence > 0.5).
4. Writes a JSON report.

Output schema (validation_report.json):

    {
      "schema_version": 1,
      "weights_path": "...",
      "video_path": "...",
      "n_labeled_frames": 320,
      "n_ball_visible": 240,
      "n_ball_invisible": 80,
      "detection_rate_at_10px": 0.85,
      "detection_rate_at_25px": 0.92,
      "false_positive_rate": 0.04,
      "median_pixel_error": 4.3,
      "p95_pixel_error": 18.7,
      "evaluated_at_utc": "..."
    }

## Sub-piece 5: Weights swap

No script. Operator action: copy the new .pt file from Drive to
data/models/. Update Stage 4's --weights argument to point to the new
path (or rename the new file to tracknet_v2_dettor.pt to overwrite in
place — version history lives in git commits and meta sidecars).

## Sub-piece 6: Stage 4 smoke test re-run

python -m stages.track_ball.smoke_test with the new weights in place.
Pre-existing acceptance criteria from Stage 4's contract apply.

## Sampling and labeling targets

- Sample every 3rd frame (= --sample-every 3) on each video. At 30fps this is 10 labels per second of footage.
- Total target: 1000-1500 labels across all 4 videos.
- Per-video target: 250-400 labels. For a 2-minute video at 30fps with sample-every=3, that's 1200 candidate frames; user labels a contiguous portion or all of them depending on time.
- Per-video minimum: 200 labels. Below this, that video contributes too little signal.

## Train/val split

- Validation: outdoor video (most-different visual case). Stage 4.5 trains on the three indoor clips, validates on outdoor.
- No further train/test split inside training data. The validation set is the held-out video, full stop. Cross-validation within indoor clips is overkill at this scale.

## Acceptance criteria

Stage 4.5 passes if and only if:

1. All four videos have ball_labels.json with at least 200 labels each.
2. Training data prep produces a valid data/training/ directory with metadata.json validating against schema.
3. Training notebook runs to completion in a Colab session.
4. Validation report shows detection_rate_at_10px >= 0.80 on the held-out outdoor video.
5. Stage 4 smoke test passes (detection_rate >= 0.80 on data/test_clip/) using the new weights.

The 0.80 threshold on the held-out outdoor video is intentionally as
strict as the test_clip threshold. Reasoning: the user wants the
application to work on their full video corpus, not just on indoor
courts similar to where Dettor trained. Held-out outdoor performance
is the honest test of generalization.

If criterion 5 passes but criterion 4 fails (test_clip works, outdoor
doesn't), document as a known issue and decide whether to:
- Re-train with more outdoor labels.
- Accept reduced outdoor performance and document.
- Treat outdoor as out-of-scope until further data collection.

If criterion 4 passes but criterion 5 fails (outdoor works, test_clip
doesn't), the data prep or hyperparameters likely have a bug.
Diagnose before declaring Stage 4.5 done.

If both 4 and 5 fail, the fine-tune did not generalize. Increase
labeled data, re-train.

## Smoke test

The Stage 4.5 smoke test is implemented as a thin wrapper that:
1. Confirms all four ball_labels.json files exist with >= 200 labels.
2. Runs validate.py against the held-out video.
3. Runs python -m stages.track_ball.smoke_test.
4. Pass if both validation report and Stage 4 smoke test pass.

Implemented at stages/finetune_ball_model/smoke_test.py.

## Stage version

0.1.0 for initial implementation.

## Out of scope

- Active learning / label suggestion (e.g., labeling tool suggesting candidate ball locations to confirm). Future enhancement.
- Multi-pass training (e.g., train on all data, identify hard examples, re-label them, train again). Manual workflow if needed.
- Hyperparameter sweep. Single set of defaults; tune by hand if results are bad.
- TrackNetV3 architecture migration. Separate effort if/when V3 weights are public or we decide to retrain from scratch.
- Distributed training. Single Colab T4 only.
- TensorBoard / wandb logging. JSON log only.

## Known follow-ups

- Adjacent-court contamination still present. Stage 4.5 trains the model to detect the ball in user's footage; it does not teach the model to ignore adjacent-court balls. The labeling tool naturally excludes adjacent-court balls (user only clicks on user's-court ball or marks invisible), but this is implicit. If contamination remains a problem after fine-tuning, an explicit "negative sample" pass with adjacent-court balls labeled as ball_visible=false may help.
- Model is hard-coded to 288x512 input. If user's videos move to a different aspect ratio, retrain. Documented in _tracknet_model.py per BatchNormOverWidth's per-position width.
- Iteration speed. Stage 4 smoke test takes ~110 minutes on CPU. After Stage 4.5 completes, consider whether to commission a smaller fast-iteration smoke clip (the original 19s clip referenced in earlier conversations, if it can be located).

## Pipeline-wide assumptions inherited

Same as Stage 4 (camera placement, single ball, single match, etc.) —
documented in ARCHITECTURE.md § Pipeline-wide assumptions.
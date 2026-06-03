# Stage 4.5 v4 — Ball detection (TrackNet, done right on 4K footage)

**Status:** DRAFT for review. Supersedes the v1/v2/v3 approaches (kept in
`contract.md` as documented failure history). v4 re-attempts a **temporal
learned detector** — now justified because the new 4K/60fps footage solved the
SNR problem that doomed v1–v3 — and fixes the specific mistakes each prior
attempt made. On approval this becomes the active Stage 4.5 approach; the
v1/v2/v3 post-mortem stays for the record.

## Why now — the measured decision (not a guess)

The new footage is **3840×2160 @ 60fps, outdoor, baseline corner, full court.**
A quantitative SNR probe (`tools/diag_ball_snr.py`) on 40 mid-flight labeled
frames of `data/pb_2min/`:

| Metric | Value | Meaning |
|---|---|---|
| Ball abs-diff intensity | median **71/255** | bright (was "faint" at 1080p) |
| Local SNR (ball vs nearby clutter) | **61×** | ball dominates its neighborhood |
| Ball present as a foreground blob | **88%** | signal reliably there |
| Detected ball blob size | **137 px** (~13 px ⌀) | **big enough to learn/track** |
| Ball-sized distractors per FULL frame | median **372** | the remaining problem |

**Conclusion:** SNR is solved; the ball is now large and bright. The only
remaining problem is picking the ball out of ~372 per-frame look-alikes — a
**temporal** problem (the ball is the one candidate tracing a smooth, fast,
gravity-shaped path). A multi-frame learned detector + trajectory
post-processing is the right tool, and the ball is finally large enough for it
to work. This is the operator-approved path (Path A).

## How v4 fixes each prior failure

| Prior attempt | Failure | v4 fix |
|---|---|---|
| **v1** fine-tune Dettor PPA TrackNet | weighted BCE (pos_weight=100) → "confidently wrong"; memorized fixed background spots; broadcast priors | **focal loss**; train from scratch on the user's own footage (no broadcast priors); held-out-clip validation |
| **v2** TrackNet from scratch, MSE | MSE on ~99.97%-zero targets → collapsed to "predict nothing" | **focal loss** (no trivial-zero minimum); validate on **detection recall**, not loss |
| **v3** classical CV | 4–6 px ball at 1080p + hundreds of distractors → ~1% precision | **4K footage** (ball ~13 px) + **raised model input resolution** so the ball stays learnable + **temporal model** to beat the 372-distractor problem |
| all three | small/narrow training set; didn't generalize | **diverse multi-clip training** + heavy augmentation + a **held-out whole clip** to *measure* cross-background generalization |

> **The critical, easily-missed fix — input resolution.** Stage 4's existing
> TrackNet inference downscales to 512×288, at which our 4K ball shrinks back to
> ~2 px and the failure returns. v4 trains AND infers at **1280×720** (ball
> ~4–5 px, clean anti-aliased from 4K — *higher quality than native 720p*),
> escalating to 1920×1088 (~6–7 px) if held-out recall is low.

## Generalization strategy (first-class goal)

The Dettor failure was a generalization failure, so v4 designs for it:
- **Train on diverse backgrounds** (all 4 outdoor clips now; indoor clips folded
  in via a later fine-tune round) so the model learns the *invariant* (ball
  appearance + motion), not background shortcuts.
- **Heavy augmentation** (photometric + spatial) for lighting/color/position
  invariance.
- **Temporal/motion features** (3-frame input) generalize across courts better
  than single-frame appearance — motion is universal. If cross-court recall is
  weak, the upgrade path is a *more* motion-aware variant (TrackNetV3 / WASB),
  not a new paradigm.
- **Trajectory post-processing is court-agnostic** (physics) and recovers
  accuracy on any venue.
- **Held-out whole clip** as the test set → measures cross-background
  generalization, not just same-clip recall.
- **Cheap per-venue fine-tune loop:** a brand-new court needs ~200 labels + a
  short warm-start fine-tune (not a from-scratch retrain). The model improves
  cumulatively as venues are added.

> Honest expectation: generalizes well to similar outdoor courts/your gear,
> moderately to a new outdoor court, weakly to indoor until indoor labels are
> added — far better than Dettor because we train on the user's own
> distribution. Periodic cheap fine-tunes per new venue are the normal
> lifecycle, not a failure.

## Place in the architecture

```
4 outdoor clips (+ labels)
   │  [4.5] finetune_ball_model: label → prep → TRAIN (Colab) → validate
   ▼
ball_model_v4.pt (+ validation_report.json)
   │  [4] track_ball: inference @720p → detections → TRAJECTORY post-proc
   ▼
ball.parquet (real; synthetic=false)  → re-run Stages 5→11 on real ball
```

Stage **4.5** owns labeling, data prep, training, validation (produces the
weights + report). Stage **4** owns inference + trajectory post-processing
(produces `ball.parquet`). v4 repoints Stage 4 at the new weights, raises its
input resolution, and adds the trajectory stage.

## Inputs / outputs

**Stage 4.5 (training):**
- In: the 4 clips' `video.mp4` + `ball_labels.json` (from `tools/label_ball.py`).
- Out: `data/models/ball_model_v4.pt`, `validation_report.json` (recall /
  precision / FP on the held-out clip), training log.

**Stage 4 (inference):**
- In: `<folder>/video.mp4` + `ball_model_v4.pt`.
- Out: `<folder>/ball.parquet` (schema unchanged: `frame_idx, pixel_x,
  pixel_y, visible, confidence, interpolated`) + `ball.meta.json` with
  `synthetic: false` (so the Stage 11 watermark drops and downstream stops
  flagging placeholder).

## Method

### 1. Labeling (operator, `tools/label_ball.py`)
~400–600 sampled labels across a few rallies on **each** of the 4 clips
(`--sample-every 3`). Mix mid-flight (hard, most valuable) + some
between-rally. Mark not-visible honestly (negatives matter).

### 2. Data prep (`prepare_training_data.py`, adapted)
- **Densify:** between consecutive visible labels ≤ N frames apart, interpolate
  ball position (linear; the path is locally smooth) to per-frame labels —
  multiplies effective labels ~3×.
- **Targets:** Gaussian heatmap (σ ≈ ball radius) at the ball pixel, mapped to
  the 1280×720 processing grid; all-zero heatmap for not-visible/negative
  frames.
- **Samples:** 3 consecutive frames (channel-stacked, 9ch) → target heatmap for
  the middle frame. Frame stride tunable (start = consecutive).
- **Split BY CLIP:** train on 3 clips (with an internal rally-level val split
  for early stopping), **hold out the 4th whole clip as the test set** — pick
  the most visually distinct court as the held-out one if they differ.

### 3. Training (Colab GPU; `finetune.ipynb`, adapted)
- Model: existing `stages/track_ball/_tracknet_model.py` `TrackNet`,
  instantiated `input_shape=(720, 1280)`, `in_channels=9`, `out_channels=3`.
- **Loss: focal loss** on the per-pixel heatmap (α, γ tunable) — defeats both
  prior failure modes.
- Optimizer Adam, lr ~1e-3 cosine decay; batch size to GPU memory (~4–8 at
  720p); **early-stop on val detection-recall, not loss.**
- **Augmentation:** random translation, small scale, brightness/contrast/hue
  jitter, horizontal flip, optional cutout. Photometric aug is the key
  generalization lever.
- Save best-by-val-recall → `ball_model_v4.pt`.

### 4. Validation (held-out clip)
A detection is correct if the heatmap peak (above a confidence threshold) is
within `TOL_PX` of the label at processing resolution. Report on the held-out
clip:
- **detection recall** (visible-labeled frames localized within TOL),
- **false-positive rate** (peaks on not-visible-labeled frames),
- per-clip breakdown (train clips vs held-out) to quantify the generalization
  gap.

### 5. Inference + trajectory post-processing (Stage 4)
- Run the model at 720p over the clip → per-frame (peak, confidence).
- **Trajectory linking:** threshold confidence; link detections by motion
  gating (velocity continuity, gravity-consistent); fit local parabolas; fill
  short gaps (`interpolated=true`); drop peaks that don't lie on any plausible
  trajectory. (Court-agnostic physics — recovers recall + kills residual FPs.)
- Emit `ball.parquet` at native frame indices + `synthetic: false`.

### 6. Re-validate the whole pipeline
Re-run Stages 5→11 on the real `ball.parquet` for a labeled clip; sanity-check
shots/bounces/rallies are produced and plausible (accuracy re-tuning of 5–10 is
a follow-up; this confirms the chain runs on real ball and the watermark drops).

## Acceptance bars (tunable in review)

- **Held-out-clip detection recall ≥ 0.80** within `TOL_PX` (= ~ball diameter
  at 720p; start 6 px).
- **False-positive rate ≤ 0.10** on not-visible-labeled frames.
- **Generalization gap:** held-out recall ≥ 0.85 × best train-clip recall
  (model isn't just memorizing training courts).
- **Post-trajectory:** ≥ 0.90 of rally frames have a ball position (detected or
  interpolated) with no physically impossible jumps.
- If recall < bar at 720p → escalate input to 1080p before declaring failure.

## Smoke test

Training is offline (Colab), so CI can't retrain. The smoke test validates the
**inference + post-processing path** with the saved weights, runnable locally:
1. Stage 4 inference on a short labeled segment of the held-out clip →
   `ball.parquet` parses, schema correct, `synthetic: false`.
2. Detection recall/precision vs labels on that segment meets the bars.
3. Trajectory continuity: no impossible jumps; gaps flagged `interpolated`.
4. Integration: Stage 5 + 5.5 run on the real `ball.parquet` without crashing
   and emit a non-empty, schema-valid shots/bounces output.
`validation_report.json` (from the Colab run) is the training-time gate and is
checked into the report path for the record.

## Workflow / division of labor

| Step | Who |
|---|---|
| Label 4 clips (`label_ball.py`) | **operator** |
| Intake clips to `data/`, data-prep script, densification, splits | me |
| Colab training notebook + run | me (code) / **operator** (run on their Colab) |
| Validation report + bar check | me |
| Stage 4 inference + trajectory module + wiring | me |
| Re-run + re-validate Stages 5→11 | me |

## Decisions flagged for review

- **Input resolution:** start 720p, escalate to 1080p if recall < bar.
- **TOL_PX / acceptance bars:** start 6 px / recall 0.80 / FP 0.10 — adjust.
- **Held-out clip choice:** most distinct court (operator picks).
- **Frame stride** for the 3-frame input (start consecutive; raise to amplify
  motion if needed).
- **Focal α/γ**, augmentation strength — tune during training.
- **Architecture escalation:** if cross-court recall is weak even at 1080p,
  move to a motion-aware variant (TrackNetV3 / WASB) — flagged, not v4 default.

## Out of scope (v4)

- Indoor footage in the v4 training set (added later via fine-tune).
- Spin, multi-ball, occluded-ball recovery beyond trajectory interpolation.
- Re-tuning Stages 5–10 thresholds for real ball (separate follow-up after v4
  lands and the chain is re-run on real trajectories).
- A from-scratch retrain per venue (replaced by the cheap fine-tune loop).

## Known follow-ups

- **Per-venue fine-tune loop** — document the ~200-label warm-start procedure
  once v4 base weights exist.
- **Calibrate Stages 9/10** against real rallies once real ball flows through.
- **Indoor coverage** — fold the indoor Dropbox clips into a fine-tune round.

## Architecture note

On approval, this v4 approach is merged into
`stages/finetune_ball_model/contract.md` (preserving the v1/v2/v3 history),
`ARCHITECTURE.md`'s Stage 4/4.5 status moves from "paused" to "v4 in progress",
and KNOWN_ISSUES.md's Stage 4.5 PAUSED section gets a v4 note. When v4 lands and
`ball.parquet` is real, the synthetic-ball caveat across Stages 5–11 is lifted
(after re-validation), and the Stage 11 watermark drops automatically.

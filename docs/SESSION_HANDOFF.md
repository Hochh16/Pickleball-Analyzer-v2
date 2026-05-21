# Session Handoff: Stage 4.5 PAUSED, moving to Stage 5

This document captures the state of Pickleball-Analyzer-v2 at the end of
the May 20 2026 session. It supersedes the previous SESSION_HANDOFF.md
(mid-Stage-4.5, before the first Colab training run).

## Context for the next session

### Project conventions (unchanged)
- Repo: github.com/Hochh16/Pickleball-Analyzer-v2
- Local: `C:\Users\hochh\pickleball-analyzer-v2`
- Windows + PowerShell + Python 3.14
- Files sent one at a time as PowerShell heredoc blocks using
  `[System.IO.File]::WriteAllText` (UTF-8 no BOM; `Set-Content` forbidden)
- Working agreement: contract -> code -> smoke test -> commit
- Each stage is a standalone Python CLI with file-path I/O. No DB,
  no Celery, no shared global state.
- ARCHITECTURE.md and KNOWN_ISSUES.md are authoritative; read both before
  proposing anything.

### Stages complete and committed
- Stages 1, 2, 3: smoke-tested and committed in earlier sessions.
- Stage 4 (track ball, TrackNetV2 inference): code-complete but
  effectively obsolete; will be rewritten once ball detection
  (Stage 4.5) has a working v4.

### Stage 4.5 status: PAUSED after three failed attempts

See KNOWN_ISSUES.md section 'Stage 4.5 - Ball detection PAUSED after
three failed attempts' for the full record. Brief summary:

- v1: fine-tune Dettor's TrackNetV2 weights. detection_rate_at_10px=0.32,
  false_positive_rate=1.00. Model memorized fixed background pixels
  (tree branches) as 'always a ball'.
- v2: train TrackNetV2 from scratch with MSE + spatial aug. Training
  collapsed at epoch 10 to 'predict zero everywhere' trivial minimum.
  Aborted at epoch 25.
- v3: classical CV (background subtraction + connected components +
  per-blob scoring). Tune accuracy ~1% on test_clip. Root cause:
  signal-to-noise ratio of the source footage is below the floor for
  per-frame detection; the ball is detectable but indistinguishable
  from co-detected court/player noise without temporal trajectory
  information.

The fundamental issue across all three is the footage profile (4-6
pixel ball at 1080p, 6 ft camera height, busy backgrounds), not the
choice of algorithm. The contract for Stage 4.5 now has a STATUS:PAUSED
block at the top.

## What's queued for the next session

Two efforts run in parallel.

### Effort 1 (David, offline): improve source video quality

The goal is to record a new test clip with materially better SNR.
Specific changes to try, in priority order:

1. **Higher camera mount.** 6 ft is the worst possible height: above
   player heads but not high enough for clean top-down. Aim for
   10-15 ft. Possibilities: gym balcony, fence post mount, second-story
   window, light pole mount.
2. **Higher resolution and frame rate.** 4K and/or 60 fps if the phone
   supports it. A 4-6 px ball at 1080p becomes 8-12 px at 4K. 60 fps
   halves per-frame motion blur and doubles trajectory points.
3. **Faster shutter speed.** Most phones auto-expose with longer
   shutters than needed. Manual shutter at 1/500s or faster reduces
   blur substantially. iPhone: built-in Camera app's manual mode or
   apps like Filmic Pro / Halide. Android: Pro mode in the camera app.
4. **Simpler backgrounds.** Avoid waving trees behind court; avoid
   adjacent courts in frame; pick venues with single-color walls
   rather than patterned/brick.

Once a new clip exists:
1. Run `mark_court.py` to produce `data/<new_clip>/court.json`.
2. Run `label_ball.py` to produce `~100-200 mid-flight ball labels.
   Per-video minimum from the contract still applies (>= 200 labels)
   if we intend to use the clip as anything other than a quick SNR
   probe.
3. Run the existing v3 tooling on the new clip with no code changes:
   `python -m stages.finetune_ball_model.tune_ball_cv --video ... \
   --court ... --labels ... --out data/<new_clip>/ball_cv_params.json`
4. Report the resulting tune accuracy. If it's substantially higher
   than test_clip's 1%, v3 may be viable on improved footage and we
   re-activate Stage 4.5 v3. If still single-digit, we need
   algorithmic extension (trajectory tracking) regardless of footage
   quality.

### Effort 2 (next Claude session): begin Stage 5 with placeholder ball data

Stage 5 is shot detection (detecting when a player hits the ball, based
on ball trajectory inflections plus pose data from Stage 3). The
canonical input to Stage 5 is `data/<video>/ball.parquet` produced by
Stage 4. Since Stage 4 is currently producing unusable output, the
next session will:

1. Write a `tools/synth_ball.py` that generates a synthetic placeholder
   `ball.parquet` for a video. Inputs: video.mp4 (for frame count and
   dimensions), court.json (to constrain trajectories to plausible
   court positions). Outputs: a `ball.parquet` matching Stage 4's
   schema with `synthetic=true` flag in the metadata sidecar.
   Trajectories should be physically plausible (constant velocity
   between bounces, gravity-influenced arcs, bounces off ground at
   plausible heights). One "rally" per ~3-5 seconds of video.
2. Generate synthetic ball.parquet for the test_clip and at least one
   other video.
3. Write the Stage 5 contract (shot detection from ball trajectory +
   pose data).
4. Begin implementing Stage 5.

Two things future-David and future-Claude should bear in mind:

- The synthetic ball.parquet is a PLACEHOLDER. Stages 5+ built on it
  must not silently accept whatever the placeholder produces; they
  should validate ball trajectories against physical plausibility,
  detect impossible motion, fail loudly on bad input. The point of
  the placeholder is to develop and test downstream stages, not to
  cover for ball-detection problems.
- When real ball detection works in a future v4, the synthetic data
  gets removed and downstream stages are re-validated against real
  ball trajectories. Any stage that "only works with clean synthetic
  trajectories" will fail on real (noisy, gappy) data; build defenses
  in from the start.

## What's working as of session end

- Stage 4.5 contract committed with STATUS: PAUSED header.
- KNOWN_ISSUES.md updated with full v1+v2+v3 history and root-cause
  diagnosis.
- v3 tooling preserved in repo: `stages/finetune_ball_model/
  _ball_cv_pipeline.py`, `tune_ball_cv.py`, `validate.py`. Can be
  re-run on improved footage without code changes.
- Diagnostic tools preserved: `tools/diag_heatmaps.py` (TrackNet
  diagnostics, force-added over .gitignore), `tools/diag_fg_at_ball.py`
  (CV foreground diagnostics, also force-added).

## Things to NOT touch between sessions

- Stage 4 (`stages/track_ball/`): code-complete but obsolete. Will be
  rewritten when ball detection works. Don't modify in the interim.
- Stage 4.5 v3 code (`stages/finetune_ball_model/`): paused.
  Specifically don't delete - we'll re-run it on improved footage.
- v1 and v2 weights on Drive (`MyDrive/pickleball/v1_artifacts/`):
  retained for historical reference. Don't delete.
- Don't re-attempt v1 or v2 - those failure modes are well-understood.
  Any future v4 ball detector should either be improved-footage v3 or
  a trajectory-tracking extension.

## Bring this to the next session

Once a new test clip exists (if any), open a new Claude session and
paste this as the first message:

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, stages/finetune_ball_model/contract.md
    before proposing anything.

    Status: Stage 4.5 is paused per the handoff document. I have/have not
    recorded a new test clip with improved camera setup.

    [If new clip exists]: New clip is at data/<folder>/video.mp4 with
    court.json and ball_labels.json (N labels). Ready to run v3 tooling
    on it.

    [If no new clip yet]: No new clip yet. Ready to begin Stage 5 with
    placeholder ball data.

The next session's Claude will pick up from there. If a new test clip
exists, the first action is running tune_ball_cv.py on it and
reporting the tune accuracy. If not, the first action is writing
tools/synth_ball.py and the Stage 5 contract.

---

Generated at session end on May 20, 2026.
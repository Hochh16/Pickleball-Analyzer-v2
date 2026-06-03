# Session Handoff: Stage 4.5 v4 (real ball detection) — TRAINING INFRA BUILT, awaiting labeling + Colab run

State of Pickleball-Analyzer-v2 at the end of the June 2 2026 session. The full
13-stage pipeline was completed previously (Stages 1–11 implemented + smoke-
tested on synthetic-ball data; see git history). This session **un-paused Stage
4/4.5** and built the **v4 real-ball-detection** effort on new 4K/60fps footage.
v4 is **mid-flight**: training infrastructure is built + CPU-validated; it now
waits on (offline) operator labeling + a Colab GPU training run.

## Where the project is

- **Stages 1–11:** implemented, end-to-end runnable on the **synthetic**
  placeholder ball. Every ball-derived output is a validated scaffold until
  real ball detection lands.
- **Stage 4/4.5:** **v4 IN PROGRESS** (this session). Goal: replace the
  synthetic ball with a real TrackNet detector. When it lands, regenerate
  `ball.parquet`, re-run Stages 5→11 on real trajectories, re-validate, and the
  synthetic caveat + Stage 11 watermark lift automatically.

## What was done this session

### 1. Diagnosed the new footage (data-driven go decision)
- 4 new clips in `C:\Users\hochh\Dropbox\Pickleball Videos\PB {2,3,4,5} minute
  outdoor.mp4` — **3840×2160 @ 60fps, outdoor, baseline corner, full court.**
- Copied the 2-min clip to `data/pb_2min/video.mp4` (dev fixture). Operator
  labeled it (471 clicks).
- **SNR probe** (`tools/diag_ball_snr.py`, local-only): ball median intensity
  **71/255**, local SNR **61×**, **~13px** blob, present **88%** of mid-flight
  frames, but **~372 ball-sized distractors per full frame**.
- **Verdict:** the 4K footage SOLVED the SNR wall that doomed v1–v3. The only
  remaining problem is temporal disambiguation from the 372 distractors → a
  multi-frame learned detector + trajectory post-processing. (Operator chose
  Path A: TrackNet temporal DL.)

### 2. Approved + drafted the v4 contract
`stages/finetune_ball_model/contract_v4.md` (active; v1/v2/v3 history preserved
in `contract.md` with a banner pointing to v4). Key decisions:
- **TrackNet temporal detector**, input **1280×720** (escalate to 1080p if
  recall short). *Critical fix:* the old inference downscaled to 512×288, which
  reshrank the 4K ball to ~2px — v4 keeps it learnable at 720p.
- **Focal loss** (CenterNet penalty-reduced) — fixes v1 (BCE "confidently
  wrong") and v2 (MSE "predict-zero collapse").
- **Diverse multi-clip training:** label all 4 outdoor clips, **train on 3,
  hold out 1 whole clip** as the cross-background generalization test. Indoor
  clips + new venues via a **cheap ~200-label warm-start fine-tune** later
  (NOT a from-scratch retrain).
- **Trajectory post-processing** (court-agnostic physics) on the detector
  output.
- **Generalization** is a first-class goal (this was the Dettor failure):
  diversity + heavy augmentation + motion-temporal features + held-out-clip
  measurement + the per-venue fine-tune loop. Honest expectation: generalizes
  well to similar outdoor courts/your gear, weaker on very different
  venues until their labels are added — improves cumulatively.

### 3. Built + CPU-validated the training pipeline (committed)
Importable, testable code (deliberate fix for v1/v2's notebook-only opacity):
- `_v4_data.py` — label **densification** (interpolate between sampled clicks →
  per-frame labels; 474 clicks → 1158 labels on clip 1), Gaussian heatmap
  targets (peak==1.0), 3-frame 720p stacking, clip-based split.
- `train_v4.py` — **focal loss**, `TrackNet(in=9,out=1,input=(720,1280))`,
  training loop, **recall-based** validation (not loss-based).
- `_v4_sanity.py` — CPU end-to-end check. **Passed:** model instantiates clean
  at 720p (11.35M params, no odd-dim/BN issues), focal loss drops
  324840→174 over 4 steps (healthy, no collapse), ~220–285 s/step on CPU
  (confirms GPU/Colab is required).
- `prepare_v4.py` — **frame-cache** extractor: pulls the needed frames at 720p
  as JPEGs (sequential read) + `v4_manifest.json`. ~1–2 GB/clip to upload vs
  15 GB of 4K. Clip 1 done: 1422 JPEGs (86 MB), 1158 samples (1045 visible).
- `_v4_cache.py` — fork-safe JPEG `CacheDataset` + `train_from_manifests`
  (train clips → held-out clip) + `evaluate_cache`; saves best-by-recall
  weights + `validation_report.json`. **Validated:** reads 9ch stacks + builds
  heatmaps correctly.
- `finetune_v4.ipynb` — turnkey Colab notebook (mount Drive → set held-out clip
  → Run All → saves to Drive). GPU accelerator preset.

### Commits this session
- `9db66b4` docs: Stage 4.5 v4 contract approved (ball detection un-paused)
- `23d193c` Stage 4.5 v4: training pipeline (data + focal-loss loop), CPU-validated
- `e8a69d0` Stage 4.5 v4: frame-cache prep + cache dataset + Colab notebook

## NEXT STEPS

### Operator (offline) — the current blocker
1. **Label clips 3–5** (`tools/label_ball.py`, ~400–600 each):
   ```
   python tools\label_ball.py --video "C:\Users\hochh\Dropbox\Pickleball Videos\PB 3 minute outdoor.mp4" --out data\pb_3min\ball_labels.json --sample-every 3
   python tools\label_ball.py --video "C:\Users\hochh\Dropbox\Pickleball Videos\PB 4 minute outdoor.mp4" --out data\pb_4min\ball_labels.json --sample-every 3
   python tools\label_ball.py --video "C:\Users\hochh\Dropbox\Pickleball Videos\PB 5 minute outdoor.mp4" --out data\pb_5min\ball_labels.json --sample-every 3
   ```
   **Flag the most visually-distinct clip → held-out test clip.**
2. **Prep each clip:** `python -m stages.finetune_ball_model.prepare_v4 data\pb_3min --clip pb_3min` (and pb_4min, pb_5min).
3. **Upload to Drive** `MyDrive/pb_v4/`: `repo/` + `data/<clip>/{v4_manifest.json, frames_720/}` for all 4 (NOT the 4K videos). Layout in the notebook's first cell.
4. **Run `finetune_v4.ipynb`** on Colab GPU (set `HOLDOUT`). **Bar: held-out recall ≥ 0.80** (notebook explains 1080p escalation if short).
5. **Download `ball_model_v4.pt`** → `data/models/`, report the held-out recall.

### Then me (next session, once weights exist)
1. Build + smoke-test **Stage 4 inference @720p** (repoint `track_ball` at
   `ball_model_v4.pt`, raise input res from 512×288 → 1280×720) **+ trajectory
   post-processing** (Kalman/parabola linking; fill gaps; reject outliers) →
   real `ball.parquet` (`synthetic: false`). Write its contract/smoke first.
2. **Re-run Stages 5→11** on the real ball; re-validate; re-tune Stage 5–10
   thresholds for real (vs synthetic) trajectories as needed.
3. **Calibrate Stages 9/10** against real rallies (uncalibrated until now).
4. The Stage 11 synthetic-ball watermark drops automatically once
   `ball_source != synthetic`.

## Key facts / gotchas
- **Local is CPU-only** (`torch 2.11.0+cpu`). Training MUST run on Colab GPU
  (operator-run; the notebook mounts the operator's Drive). I can't execute
  Colab — I prepare/validate the code; the operator runs the GPU step.
- **The 512×288 trap:** Stage 4's existing inference downscales to 512×288; at
  4K that reshrinks the ball to ~2px. v4 inference MUST run at 1280×720 — don't
  let Stage 4 silently downscale.
- **Diagnostics are local-only** (`tools/diag_*.py` are gitignored by
  convention); the SNR numbers are recorded in `contract_v4.md` + KNOWN_ISSUES.
- **frames_720/ + v4_manifest.json are gitignored** (regenerable, under
  `data/`). Re-run `prepare_v4.py` to rebuild.

## Things to NOT touch
- Don't re-attempt ball-detection v1/v2; failures well understood.
- v1/v2 weights on Drive retained for reference.

## Bring this to the next session

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, stages/finetune_ball_model/contract_v4.md
    before proposing anything.

    Stage 4.5 v4 training infra is built + CPU-validated. I have [labeled
    clips 3-5 / run prepare_v4 / run the Colab training] and the held-out
    recall was [X]. ball_model_v4.pt is in data/models/. Build Stage 4
    inference @720p + trajectory post-processing and re-run Stages 5-11 on the
    real ball. [or: held-out recall was low — let's escalate to 1080p.]

---

Generated at session end on June 2, 2026.

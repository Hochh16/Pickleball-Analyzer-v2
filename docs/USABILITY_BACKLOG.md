# Usability backlog — setup wizard + vision hand-off

Deferred by David to AFTER the first full 5-minute end-to-end run
(`pb_5_minute_outdoor-2`). Captured 2026-07-15. These are the "much to be
dramatically improved" items — the app works end-to-end, but the operator
experience (setup → Colab → resume) is fragile and confusing. Fix as a focused
cleanup pass once the run is validated.

Priority order: **A (setup confusion) → B (hand-off speed/robustness) → C (code
robustness) → D (nice-to-haves).**

## A. Setup wizard — direct operator feedback  ✓ DONE 2026-07-15

- **A1. Ask starting position ONCE.** ✓ Removed the Court-step near/far dropdown.
  The camera protocol puts the camera in the corner nearest the analyzed player,
  and the court marking already treats points 5-6 as the user's (near/bottom)
  kitchen line, so the player is always on the NEAR baseline. Position is now
  asked once, visually, on the "You" step (which side).
- **A2. Make the step nav clickable.** ✓ Any reached step can be clicked in the
  step bar to jump back and re-edit (gated by a furthest-reached index + session
  presence). Was locked before.
- **A3. Never emit `user_baseline='far'`.** ✓ The wizard now always sends `near`
  (see A1), so `far` can't reach Stage 2.5. Backend contract stays general
  (still accepts near/far) for a future baseline-agnostic Stage 2.5.

## B. Vision hand-off — "far quicker and easier"

**Pass 1 (reliability & usability) ✓ DONE 2026-07-16** — orchestration moved into
`tools/colab_vision.py` (real, unit-tested module). The notebook is a tiny
**git-clone bootstrapper**: it `git pull`s this (public) repo on Colab and calls
`run_all(REPO)`. So a code change is just `git push` -> **Run All** (pulls latest)
— no bundle to rebuild, nothing to re-upload. The notebook can be opened straight
from GitHub. The old `pb_vision_upload.zip` bundle + `build_vision_bundle.py` are
gone.

- **B1. One knob, or zero.** ✓ `run_all` auto-derives CLIP from the single
  `*_vision_input.zip` on Drive (clear errors on zero/multiple); `CLIP=None` by
  default, set only to disambiguate.
- **B2. Notebook self-manages the GPU.** ✓ `free_gpu()` between stages + ball OOM
  auto-fallback down a batch ladder [8,4,2,1] with `expandable_segments:True`.
- **B3. Pull the clip to local disk ONCE, robustly.** ✓ `robust_copy` (sequential
  + force-remount retries) for the bundle + weights; stages read local disk.
- **B4. Auto-backup each stage's outputs to Drive as it finishes.** ✓ Backs up to
  `My Drive/<clip>_outputs/` and RESUMES from there — a reset re-runs only the
  outstanding stages (and skips the whole bundle copy if all outputs exist).
- **B7. Target flow:** ✓ *download bundle → Run All → upload outputs* (outputs
  also downloadable straight from `<clip>_outputs/` on Drive).
- **C2 done:** the `no_grad` fix is pulled from GitHub like the rest of the code —
  no bundle, no runtime patch. (Superseded the "re-upload the bundle" action.)

**Pass 1.5 (Drive-for-Desktop auto-sync) ✓ DONE 2026-07-16** — with Google Drive
for Desktop installed, the app auto-detects the synced `My Drive` (`?:\My Drive`,
or `PB_DRIVE_DIR`). On the GPU hand-off it **writes the clip bundle into the synced
folder** (Drive uploads it; clears stale `*_vision_input.zip`) and **watches
`<clip>_outputs/`**, ingesting the results + auto-resuming the moment they sync
back — no manual download/upload/unzip. Falls back to the manual buttons when no
synced folder is present; the manual upload also accepts a `.zip`. Operator's only
action is **Run All** on Colab. (`app/drivesync.py`, watcher in `pipeline.py`.)
Caveat: wait for Drive to finish uploading the (large) clip before running Colab.

**Pass 2 (stage-level performance):**
- **B5. Pose on GPU ✓ DONE + GPU-VALIDATED 2026-07-18.** Swapped MediaPipe
  (CPU-bound, the single slowest stage — ~43 min on the 5-min CPU run) for
  Ultralytics YOLO-pose (CUDA; already a Stage-2 dep). YOLO's COCO-17 keypoints
  map onto the existing BlazePose-33 column schema (16 unused points → NaN);
  every landmark any stage reads is in COCO-17, so it's drop-in (no Stage 5/6/8/
  render changes). Crops batched per frame. Validated vs the MediaPipe run on
  pb_5min_test_20s-7: detection 100% (was 96.5%), user keypoint drift 5-9 px
  median, skeleton overlays track tightly, **rating stable (3.2 vs 3.16, same
  band/confidence)**. **Measured end-to-end on Colab T4: pose = 115 s** (4537
  detections). Dropped the mediapipe dependency.
- **B6. Decode speed — RESOLVED via classify fix + A100 (NVDEC deprioritized) 2026-07-18.**
  The T4 poles turned out to be (a) `classify_tracks` **random-seeking** to sampled
  frames (`cap.set(POS_FRAMES)` re-decodes from a keyframe each time) — fixed with
  a single sequential pass + one decode for near+far (**341 s → 11 s on A100**,
  roles byte-identical), and (b) a modest T4. **Measured A100 (Pro+) vs original
  T4, 20 s clip:** track 106→43 s, classify 341→11 s, pose 115→47 s, ball 389→115 s
  (batch 8, no OOM) — **~16 min → ~3.6 min of stage time (~4.5×)**. Projected A100
  compute: 2-min ~25 min, 5-min ~55-60 min, 10-min ~1.9 hr. Notebook now recommends
  A100 + warns on lesser cards. **NVDEC judged not worth it** (marginal ~20% vs
  A100, real integration risk). Remaining speed levers if revisited: ball batch
  16/32 on A100, bundle-size reduction (cuts upload+decode), half-fps for
  track/pose. **Deferred — speed is "reasonable for now" (David).**

## C. Code / pipeline robustness (from this run)

- **C1. `no_grad` in ball inference — FIXED in repo** (`stages/track_ball/track_ball_v4.py`,
  committed). The batched path retained the autograd graph → ~38 GB OOM. Both
  autocast blocks now wrapped in `torch.no_grad()`.
- **C2. Re-upload the fixed `pb_vision_upload.zip` to Drive** so the notebook needs
  no on-Colab patch cell. (Bundle rebuilt locally; Drive copy is still the buggy
  pre-fix version — currently patched at runtime as a stopgap.)

## E. Session resume UX  ✓ DONE 2026-07-18

Re-picking the same video created a NEW session and a full re-upload; resuming
dropped you on the Court step with points blanked. Fixed: loading a fully-
configured session hydrates handedness/side from disk, unlocks the whole step
nav, and jumps straight to the Run screen (or Review). Bundle push also skips if
the same clip is already synced (no needless re-upload).

## D. Nice-to-haves

- Per-stage progress + time estimates on the app run screen.
- Validate throughput for clips longer than 5 minutes (David wants longer to work).

## Video (product decision)  ✓ DONE 2026-07-15

Decided to **drop the box-overlay render and link the original clip** (the boxes
added little value). Pipeline no longer renders/compresses an annotated video;
the report's "Match video" section and the app done-card link the original.
**Revisit** once the shot stats are solid: the valuable form is a ball-trail +
per-shot-label render, and especially short **evidence clips tied to each
improvement-plan point** — not a full annotated replay.

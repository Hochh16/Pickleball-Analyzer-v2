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

**Pass 2 (stage-level performance) — TODO:**
- **B5. Pose on GPU.** Pose ran 43 min on CPU — the single biggest time sink.
  Likely a pose-model swap (MediaPipe CPU → a CUDA model), which changes keypoint
  format and needs Stage 6 validation.
- **B6. GPU-decode (NVDEC)** for video reads to cut ball/decode time.

## C. Code / pipeline robustness (from this run)

- **C1. `no_grad` in ball inference — FIXED in repo** (`stages/track_ball/track_ball_v4.py`,
  committed). The batched path retained the autograd graph → ~38 GB OOM. Both
  autocast blocks now wrapped in `torch.no_grad()`.
- **C2. Re-upload the fixed `pb_vision_upload.zip` to Drive** so the notebook needs
  no on-Colab patch cell. (Bundle rebuilt locally; Drive copy is still the buggy
  pre-fix version — currently patched at runtime as a stopgap.)

## E. Session resume UX (found during Drive auto-sync testing)

Re-picking the same video creates a NEW session (`-3`, `-4`, …) and a full multi-GB
re-upload; "Or continue a previous setup" exists but resuming currently drops you
on the Court step with the points blanked — effectively forcing a full re-setup,
which is why the operator kept creating new sessions. Fix: on session load, unlock
the step nav from the on-disk state (calibration/roster done → Review/Run
reachable) and let a fully-configured session jump straight to Run / re-run.

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

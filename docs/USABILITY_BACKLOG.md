# Usability backlog — setup wizard + vision hand-off

Deferred by David to AFTER the first full 5-minute end-to-end run
(`pb_5_minute_outdoor-2`). Captured 2026-07-15. These are the "much to be
dramatically improved" items — the app works end-to-end, but the operator
experience (setup → Colab → resume) is fragile and confusing. Fix as a focused
cleanup pass once the run is validated.

Priority order: **A (setup confusion) → B (hand-off speed/robustness) → C (code
robustness) → D (nice-to-haves).**

## A. Setup wizard — direct operator feedback

- **A1. Ask starting position ONCE.** Today the Court step has a "Which baseline
  are you on? (near/far)" dropdown AND the You step asks which side you start on
  visually. Two places for one fact → confusing, and they can disagree. Decide a
  single source of truth. *Recommendation:* drop the Court-step baseline dropdown
  entirely; infer near/far + corner from the single visual "which side do you
  start on" click combined with the marked court geometry. One question, visual,
  unambiguous.
- **A2. Make the step nav clickable.** Clicking a prior step (e.g. "Court") in the
  top step bar must jump back to it. David got stuck — the court field was
  "locked" and he couldn't return to re-mark. Every completed step should be
  re-editable.
- **A3. Never emit `user_baseline='far'`.** Stage 2.5 v1 only supports the analyzed
  player on the NEAR baseline. The wizard must guarantee near — either by
  construction (always orient so the analyzed player's baseline is near) or by
  validating before the run and blocking with a plain-language message. A `far`
  value must never reach the pipeline (it silently killed Stage 2.5 this run).

## B. Vision hand-off — "far quicker and easier"

- **B1. One knob, or zero.** The notebook has multiple textual `CLIP` references;
  David didn't know which to edit. Reduce to a single obvious knob — better,
  auto-derive `CLIP` from the one `*_vision_input.zip` present on Drive (glob it)
  so there is nothing to type.
- **B2. Notebook self-manages the GPU.** On start: clear any prior allocation,
  verify free memory, and on OOM auto-retry at a smaller batch instead of failing.
  (This run OOM'd repeatedly and needed manual restarts.)
- **B3. Pull the clip to local disk ONCE, robustly.** Sequential copy + remount
  retries; never random-read the 4.6 GB bundle over Drive FUSE (that dropped with
  `Errno 107` mid-run). Now done by hand in the recovery cell — bake it in.
- **B4. Auto-backup each stage's outputs to Drive as it finishes.** A runtime reset
  wiped ALL local state this run (video, code, outputs) and forced a re-run. If
  every stage backs up on completion, a reset costs "re-run one cell," never
  "start over."
- **B5. Pose on GPU.** Pose ran 43 min on CPU — the single biggest time sink.
  Move it to GPU.
- **B6. GPU-decode (NVDEC)** for video reads to cut ball/decode time.
- **B7. Target flow:** *download bundle → Run All → upload outputs.* No editing,
  no manual patch cells, no per-stage babysitting.

## C. Code / pipeline robustness (from this run)

- **C1. `no_grad` in ball inference — FIXED in repo** (`stages/track_ball/track_ball_v4.py`,
  committed). The batched path retained the autograd graph → ~38 GB OOM. Both
  autocast blocks now wrapped in `torch.no_grad()`.
- **C2. Re-upload the fixed `pb_vision_upload.zip` to Drive** so the notebook needs
  no on-Colab patch cell. (Bundle rebuilt locally; Drive copy is still the buggy
  pre-fix version — currently patched at runtime as a stopgap.)

## D. Nice-to-haves

- Per-stage progress + time estimates on the app run screen.
- Validate throughput for clips longer than 5 minutes (David wants longer to work).

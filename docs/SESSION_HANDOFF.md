# Session Handoff: Stage 5 (detect shots) DONE; Stage 4.5 still PAUSED

This document captures the state of Pickleball-Analyzer-v2 at the end of the
May 22 2026 session. It supersedes the previous handoff (Stage 4.5 paused,
"begin Stage 5 with placeholder ball data"), whose plan is now executed.

## Context for the next session

### Project conventions (unchanged)
- Repo: github.com/Hochh16/Pickleball-Analyzer-v2
- Local: `C:\Users\hochh\Pickleball-Analyzer-V2`
- Windows + PowerShell + Python 3.14 (3.14.3 verified this session;
  mediapipe 0.10.35, ultralytics 8.4.46, torch 2.11 cpu all import fine)
- Working agreement: contract -> code -> smoke test -> commit. Each stage's
  `contract.md` is the source of truth and is approved before code.
- Each stage is a standalone Python CLI with file-path I/O. No DB, no shared
  global state. Outputs are sidecar files in one folder per video under `data/`.
- `ARCHITECTURE.md` and `KNOWN_ISSUES.md` are authoritative; read both before
  proposing anything.
- Implemented stages use importable folder names (`stages/detect_shots`, not
  `stages/05_detect_shots`) because Python modules can't start with a digit.

### Stage status
- **Stages 1, 2, 3**: implemented, smoke-tested. Re-verified this session on
  `data/test_clip/` (Stage 2 6/6, Stage 3 6/6).
- **Stage 4** (TrackNetV2 ball): code-complete and not broken in itself; its
  weights just don't generalize to this footage, so it currently yields
  unusable output. Whether it needs only new weights (stays as-is) or a rewrite
  depends on the eventual v4 approach (TrackNet-with-better-weights vs the
  classical-CV direction explored in Stage 4.5 v3). Not obsolete by itself.
- **Stage 4.5** (ball detection): PAUSED after v1/v2/v3 failures. Root cause is
  the footage profile (4-6 px ball, ~6 ft camera, busy backgrounds), not the
  algorithm. See `KNOWN_ISSUES.md`. Unchanged this session.
- **Stage 5** (detect shots): NEW — implemented, smoke-tested (7/7), committed
  this session. Details below.
- **Stages 6-11**: not started. Stage 6 (classify shots) is the natural next.

## What was done this session

### 1. Re-verified Stages 2 & 3 and improved user labeling
- No parquet artifacts existed on disk (all gitignored; earlier outputs were
  never committed). Regenerated them for `data/test_clip/`.
- Wrote `data/test_clip/user_clicks.json` with **5 clicks** (frames 1000, 2800,
  4468, 6000, 7400) identifying the user (a woman, lavender top / white skirt).
  Stage 2 user labeling went from 253 -> **4876 frames (~60% of the clip)**.
  ByteTrack drops the user's track on side-switches, so the user spans 4
  track_ids (2, 1393, 2857, 4074); coverage is sparse by design of single
  clicks. Remaining unlabeled gaps: 0-999, 2380-2800, 4721-6000, 6849-7400.
- Fixed a stale assertion in `stages/pose/test_pose.py`: the in-scope upper
  bound was an absolute 12000 (tuned to the old 2-min clip); it's now relative
  (in-scope < 90% of non-transient), which scales with clip length. Committed
  separately (`7f08997`).

### 2. Built the synthetic placeholder ball generator
- `tools/synth_ball.py`: generates `ball.parquet` matching Stage 4's schema
  exactly, plus `ball.meta.json` (with `synthetic: true`) and
  `ball_synth_truth.json` (ground-truth hit list). Impacts are placed at real
  player **wrists** (from poses.parquet), with gravity-flavored arcs between
  hits, follow-throughs after the last hit, and an optional `--gap-frac` to
  simulate detection gaps. Deterministic via `--seed`. Each truth hit is
  flagged `is_serve` (first hit of a rally).
- Run: `python tools/synth_ball.py data/test_clip --seed 1234 --force`

### 3. Implemented Stage 5 (detect shots)
- Contract: `stages/detect_shots/contract.md` (approved). Code:
  `stages/detect_shots/detect_shots.py`. Smoke test:
  `stages/detect_shots/test_detect_shots.py`.
- Run: `python -m stages.detect_shots.detect_shots data/test_clip --force`
- Smoke: `python -m stages.detect_shots.test_detect_shots` (7/7 pass).
- **How it works:**
  - Rally hits are detected by an **impulse** signal — a single-frame turn-rate
    spike and/or sudden speed jump — NOT a raw windowed angle. This is the key
    design choice: free-flight gravity arcs (e.g. a lob's apex passing over a
    player's head) bend the path *gradually* and are correctly NOT counted as
    shots, while a paddle strike (sharp, ~1-frame change) is.
  - Players are associated to an impact by **nearest wrist** (poses.parquet),
    falling back to bbox/foot, within a **perspective-scaled radius**
    (0.5 x bbox height, clamped 30-120 px) — a flat pixel threshold fails
    because near players are ~600 px tall and far players ~150 px.
  - **Serves are detected** by a separate signal: the ball appearing near a
    player after a long not-visible gap (dead time), with an outgoing launch.
    They are flagged `is_serve: true`, `detection_method: "serve_appearance"`.
    (A serve has no incoming ball, so the impulse detector alone is blind to
    it — this was added after review specifically so serve quality can be
    tracked downstream.)
  - Defenses (placeholder/bad data): requires `ball.meta.json`; emits a loud
    `ball_source: "synthetic"` warning; raises on impossible motion (teleport
    check); honestly misses impacts inside ball gaps rather than fabricating.
- **Output `shots.json`**: ordered shots with `shot_id`, `frame`, `t_sec`,
  striking `track_id`/`is_user`, `is_serve`, `detection_method`,
  `impact_pixel_xy`, `impact_court_xy_ft`, pre/post velocity + speed,
  `direction_change_deg`, `turn_rate_deg`, `speed_change_ratio`, `confidence`,
  plus `stats` and `warnings`.
- **Smoke-test results on test_clip (synthetic ball):** non-serve recall 0.988,
  player-match 0.959, precision 0.990, serve recall 0.894; gap variant degrades
  gracefully.

### Commits this session
- `7f08997` Stage 3: relative scope bound in pose smoke test
- `f68132a` Stage 5: detect shots (impulse hits + serve appearance) +
  synthetic ball fixture

## IMPORTANT caveats for the next session

- **The ball is synthetic.** Everything Stage 5 produces is derived from a
  PLACEHOLDER ball. Downstream stages must keep validating ball plausibility and
  must not silently trust it. When a real ball detector (v4) exists, delete the
  synthetic ball, re-run Stage 5 on real (noisy, gappy) trajectories, and
  re-validate. The synthetic-only smoke-test bars (recall 0.80 / player-match
  0.80 / precision 0.70 / serve recall 0.70) will need a real-data counterpart.
- **Serve detection is detection, not quality.** Stage 5 now tells you a serve
  happened, who served, from where, and the launch velocity — enough for Stage 6
  to classify and Stage 8 to count. But serve *quality* (placement/depth, e.g.
  "deep serve to the backhand") needs the serve's **landing/bounce location**,
  and **bounce detection is still deferred** (Stage 7 territory). On real gappy
  ball data, serve recall will be lower than the synthetic 0.894; a "server
  behind the baseline" check is suggested to harden it (see Stage 5 contract
  Known follow-ups).
- **User labeling is ~60%.** User-attributed shot stats will undercount until
  coverage improves. The right fix is the dedicated track-classification stage
  ("Stage 2.5") noted in `KNOWN_ISSUES.md`, not manual clicks-per-gap. Defer
  until user-centric metrics are actually being computed on real data.

## Local-only artifacts (gitignored — regenerate, don't expect in git)

`data/` and `*.parquet` are gitignored. To reproduce `data/test_clip/` state:
1. `python -m stages.track_players.test_track`  (needs `user_clicks.json`)
2. `python -m stages.pose.test_pose`
3. `python tools/synth_ball.py data/test_clip --seed 1234 --force`
4. `python -m stages.detect_shots.detect_shots data/test_clip --force`

`user_clicks.json` IS gitignored too (under `data/`), so the 5 clicks are
local-only. If lost, re-identify the user in a few frames and rebuild it.

## What's queued for the next session

In rough priority order (none blocking):

1. **Stage 6 — classify shots.** The natural next stage. Input: `shots.json`
   + poses + ball + court. Output: per-shot forehand/backhand (from pose) and
   drive/dink/lob/serve/volley (from trajectory + position). Serves are already
   flagged by Stage 5. Write the contract first (the stub is 3 lines).
2. **Better source video for Stage 4.5** (offline, David). Higher mount
   (10-15 ft), 4K/60fps, faster shutter, simpler backgrounds. Once a clip
   exists, run the v3 tooling to measure SNR (see prior handoff in git history).
3. Deferred (revisit when relevant): fill user-label gaps / build the Stage 2.5
   track-classification stage; generate synth ball + Stage 5 on a second clip
   to test generalization; bounce detection (needed for in/out + serve
   placement).

## Things to NOT touch between sessions
- Stage 4 (`stages/track_ball/`) and Stage 4.5 (`stages/finetune_ball_model/`):
  paused/obsolete; don't modify or delete.
- v1/v2 weights on Drive: retained for reference.
- Don't re-attempt ball-detection v1/v2; those failures are well-understood.

## Bring this to the next session

Open a new Claude session and paste:

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, and the relevant stage contract.md
    before proposing anything.

    Stage 5 (detect shots) is done and committed; the ball is still a
    synthetic placeholder (Stage 4.5 paused). I'd like to start Stage 6
    (classify shots).   [or whatever you choose]

---

Generated at session end on May 22, 2026.

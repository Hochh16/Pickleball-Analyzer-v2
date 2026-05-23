# Session Handoff: Stages 5, 6 & 2.5 DONE; Stage 4.5 still PAUSED

This document captures the state of Pickleball-Analyzer-v2 at the end of the
May 22 2026 session. It supersedes the previous handoff (Stage 4.5 paused,
"begin Stage 5 with placeholder ball data"), whose plan is now executed and
extended through Stage 6 (classify shots) plus a NEW Stage 2.5 (classify tracks
into player roles). The pipeline is now 12 stages.

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
- **Stage 2.5** (classify tracks): NEW — implemented, smoke-tested (5/5),
  committed this session. Maps track_ids to roles (user/partner/opp_left/
  opp_right/noise). Raised user coverage 60% -> 98.6% on test_clip. Details below.
- **Stage 5** (detect shots): implemented, smoke-tested (7/7), committed.
- **Stage 6** (classify shots): implemented, smoke-tested (8/8), committed.
- **Stages 7-11**: not started. Stage 7 (segment rallies) is the natural next.

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

### 4. Implemented Stage 6 (classify shots)
- Contract: `stages/classify_shots/contract.md`. Code:
  `stages/classify_shots/classify_shots.py`. Smoke test:
  `stages/classify_shots/test_classify_shots.py` (8/8).
- Run: `python -m stages.classify_shots.classify_shots data/test_clip --force`
- **Output `classified.json`**: a 1:1 superset of `shots.json` adding per shot:
  `stroke_side` (+conf), `shot_type` (+conf), `is_volley` (+conf), and a
  `features` block. Propagates `ball_source` + the synthetic warning.
- **Stroke side** (forehand/backhand): real only for the USER (handedness from
  `roster.json`, mapped via `is_user`); `unknown` for others until role
  classification exists. Handles LEFT-handed users (handedness flips the side;
  camera-facing mirror handled) — verified by unit test.
- **Shot type**: serve (from Stage 5) / drive / dink / drop / lob / overhead /
  reset / unknown, by a rule tree over hitter court-zone, post/pre ball speed
  (px→ft/s via local px/ft), arc height, contact height.
- **Volley** (`is_volley`): bounce-based — scans the inter-shot ball trajectory
  for a non-player ground-bounce kink; none => volley.
- `roster.json` (NEW per-video setup input): handedness per logical role
  (user/partner/opp_left/opp_right). v1 uses only the `user` entry.
- **Smoke results:** rule logic (shot type + L/R stroke side) all correct;
  end-to-end is_volley accuracy 0.95, serves->serve, schema 1:1, unknown 11%.

### 5. Implemented Stage 2.5 (classify tracks into player roles) — NEW stage
- Contract: `stages/classify_tracks/contract.md`. Code:
  `stages/classify_tracks/classify_tracks.py`. Smoke test (5/5):
  `stages/classify_tracks/test_classify_tracks.py`.
- Run: `python -m stages.classify_tracks.classify_tracks data/test_clip --force`
- **Output `track_roles.json`**: maps each track_id -> role
  (user/partner/opp_left/opp_right/noise) + confidence + basis; aggregates
  track_ids per role; stats incl. `user_frame_coverage`.
- **Method (v1, video-free):** noise filter (835 -> ~47 non-noise) -> near/far
  side -> seed user from clicks -> **simultaneity** ("two people at once" =>
  the simultaneous near track is the partner) + **click-anchored continuity** +
  **perspective-normalized height** to link user gap-segments and split
  user/partner even in matching kit -> provisional opponent L/R by court-x.
- **Result:** user coverage **60% -> 98.6%**; all clicked tracks confirmed user.
- **v1 limitations:** multi-region clothing-color matching NOT yet implemented
  (fast-follow; height+continuity already cover matching-kit). Opponent roles
  are contaminated by far-side adjacent-court players (low confidence) — needs
  a tighter far-side filter.
- **Why this matters:** it's REAL data (no synthetic-ball dependency) and fixes
  the user-labeling debt that capped every downstream user metric. Stage 3's
  scope filter and Stage 6's is_user-only handedness mapping should later switch
  to consuming `track_roles.json`.

### Commits this session
- `7f08997` Stage 3: relative scope bound in pose smoke test
- `f68132a` Stage 5: detect shots (impulse hits + serve appearance) + fixture
- `94a08a9` docs: ARCHITECTURE + handoff for Stage 5
- `fbd22f0` Stage 6: classify shots + synth_ball typed/bounce truth
- `b5570e1` docs: ARCHITECTURE + handoff for Stage 6
- `ef0e52e` Stage 2.5: classify tracks into player roles (pipeline 11->12)

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
- **Lob detection is weak in this footage (low headroom).** The play sits ~250
  px below the top of the frame, so tall lob arcs (synthetic OR real) clip at
  the top edge and `arc_height_frac` collapses. Stage 6's lob *rule* is
  validated by unit test, but end-to-end lob accuracy is not gated.
  **David will provide future footage with more headroom above the play**, which
  directly fixes this (the apex stays in-frame). When that footage arrives,
  re-tune `LOB_MIN_ARC_FRAC` and re-validate lob detection end-to-end. (More
  headroom comes naturally from the higher camera mount already wanted for
  Stage 4.5 ball SNR — one better setup helps both.)
- **Non-user handedness / opponent analysis** (e.g. "hit to the opponent's
  backhand") needs the role-classification stage to map handedness (collected in
  `roster.json`) to the right tracks. Left-handed *players generally* are
  supported; we just can't yet attribute hands to non-user players.

## Local-only artifacts (gitignored — regenerate, don't expect in git)

`data/` and `*.parquet` are gitignored. To reproduce `data/test_clip/` state:
1. `python -m stages.track_players.test_track`  (needs `user_clicks.json`)
2. `python -m stages.pose.test_pose`
3. `python tools/synth_ball.py data/test_clip --seed 1234 --force`
4. `python -m stages.detect_shots.detect_shots data/test_clip --force`
5. `python -m stages.classify_shots.classify_shots data/test_clip --force`
6. `python -m stages.classify_tracks.classify_tracks data/test_clip --force`
   (independent of the ball; can run any time after step 1)

`user_clicks.json` and `roster.json` are gitignored too (under `data/`), so
they're local-only. If lost: re-identify the user in a few frames to rebuild
`user_clicks.json`, and recreate `roster.json` (`{"schema_version":1,
"handedness":{"user":"right","partner":"unknown","opp_left":"unknown",
"opp_right":"unknown"}}` — set `user` to match `court.json.dominant_hand`).

## What's queued for the next session

Two threads: continue the linear pipeline, and the infrastructure investments
that several stages now want.

**Linear pipeline:**
1. **Stage 7 — segment rallies.** The natural next stage. Input:
   `classified.json` + ball. Output: `rallies.json` (start/end frame, shot_ids,
   end_reason). NOTE it wants **bounce detection** for end reasons like
   "ball-out" / "ball bounced twice" — see below. Write the contract first
   (the stub is ~5 lines).

**Infrastructure:**
- **Stage 2.5 follow-ups (DONE this session, but v1 has gaps):**
  - Multi-region clothing-color matching (deferred; helps different-colour
    user/partner & opponent separation).
  - Tighten the far-side filter — opponent roles are contaminated by
    adjacent-court players (~19 tracks each).
  - Re-wire Stage 3 (scope filter) and Stage 6 (is_user-only handedness) to
    consume `track_roles.json` instead of re-deriving / using sparse is_user.
    Then Stage 6 can give opponents/partner real forehand/backhand.
- **Bounce detection** (likely its own small stage, or folded into Stage 7).
  Needed for: ball in/out + rally end reasons (Stage 7), serve/shot landing &
  placement quality (Stage 6 reset, serve quality), and would consolidate the
  inter-shot bounce check Stage 6 currently does itself. Stage 5 already
  computes the raw signal (non-player trajectory inflections it discards).

**Footage (offline, David):**
- **Better source video for Stage 4.5** AND **more headroom above the play**.
  Higher mount (10-15 ft), 4K/60fps, faster shutter, simpler backgrounds, and
  framing with room above the action. The higher mount serves ball SNR (Stage
  4.5) AND the headroom fixes Stage 6 lob detection. Once a clip exists: run v3
  tooling to measure ball SNR, and re-validate/re-tune lob detection.
- Deferred: fill user-label gaps (better solved by Stage 2.5); second clip for
  Stage 5/6 generalization.

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

    Stages 2.5 (classify tracks), 5 (detect shots) and 6 (classify shots) are
    done and committed; the ball is still a synthetic placeholder (Stage 4.5
    paused). I'd like to start Stage 7 (segment rallies).   [or: bounce
    detection, or wire Stage 3/6 to consume track_roles.json — see What's queued]

---

Generated at session end on May 22, 2026.

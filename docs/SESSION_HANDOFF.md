# Session Handoff: Stage 9 (rate — USAPA) DONE

This document captures the state of Pickleball-Analyzer-v2 at the end of the
May 29 2026 session. It supersedes the previous handoff (Stage 8 done), now
extended through **Stage 9 (rate — USAPA skill rating)**. The pipeline is now
13 stages, **10 implemented** (1, 2, 2.5, 3, 5, 5.5, 6, 7, 8, 9); Stage 4/4.5
remain paused; Stages 10–11 not started.

## Context for the next session

### Project conventions (unchanged)
- Repo: github.com/Hochh16/Pickleball-Analyzer-v2
- Local: `C:\Users\hochh\Pickleball-Analyzer-V2`
- Windows + PowerShell + Python 3.14 (mediapipe 0.10.35, ultralytics 8.4.46,
  torch 2.11 cpu, pandas + pyarrow all import fine)
- Working agreement: **contract → code → smoke test → commit**, with design
  decisions / limitations flagged for the operator (inputs asked, not guessed)
  BEFORE coding. Each stage's `contract.md` is the source of truth.
- Each stage is a standalone Python CLI with file-path I/O. No DB, no shared
  global state. Outputs are sidecar files per video under `data/`.
- `ARCHITECTURE.md` and `KNOWN_ISSUES.md` are authoritative; read both first.
- Implemented stages live in **importable** folder names (`stages/rate`,
  `stages/compute_metrics`, `stages/segment_rallies`, …) — module names can't
  start with a digit. Per-stage contracts live IN the implementation folder.
  Numbered stub folders are deleted on approval.

### Stage status (post-session)
- **Stages 1, 2, 3**: implemented, smoke-tested. Unchanged.
- **Stage 2.5** (classify tracks): implemented; unchanged. Opponent
  contamination follow-ups still pending (flows into S8/S9 — flagged).
- **Stage 4** (TrackNetV2 ball): code-complete, weights don't generalize.
  **Do not touch.**
- **Stage 4.5** (ball detection): **PAUSED** after v1/v2/v3 failures.
  **Do not touch / don't re-attempt v1/v2.**
- **Stages 5, 5.5, 6, 7**: implemented; unchanged.
- **Stage 8** (compute metrics): implemented; unchanged. Produces `metrics.json`
  with a `reliability` block (real vs synthetic_gated vs pending families).
- **Stage 9** (rate — USAPA): **NEW** — implemented + smoke-tested (9/9) +
  committed (`2ba21ca`). Details below.
- **Stages 10–11**: not started. Stage 10 (plan improvement) is the natural
  next.

## What was done this session

### Built Stage 9 (rate — USAPA skill rating) — NEW stage
- Contract: `stages/rate/contract.md`. Code: `stages/rate/rate.py`. Smoke test
  (9/9): `stages/rate/test_rate.py`.
- Run: `python -m stages.rate.rate data/test_clip --force`
- **Input:** `metrics.json` (Stage 8) only — S8 already aggregated everything.
- **Output `rating.json`** for the **user**:
  - `rating`: `estimate` (continuous 1.0–5.5) + `band` (nearest USAPA
    half-step) + `range` (confidence interval) + `confidence` [0,1].
  - `dimensions[]`: six USAPA-anchored skill axes, each with
    `subscore_level`, `weight`, `confidence`, `data_source` (real/synthetic),
    `driver_metrics`. **net_play (0.20) + movement (0.10) are REAL**
    (positioning from players.parquet); **error_control (0.25) + shot_skill
    (0.25) + serve (0.10) + rally_consistency (0.10) are synthetic-derived**.
    Real weight = 0.30, synthetic = 0.70.
  - `skill_coverage`: covered / proxy_or_pending / not_captured_yet /
    out_of_scope buckets (see "skill gaps" below).
  - `reliability`: synthetic_ball + real_weight/synthetic_weight.
- **Scope decisions settled with the operator before coding:**
  1. **Rate the USER only** in v1 (others deferred; structure extensible).
  2. **Output = continuous estimate + USAPA band + range.**
  3. **Full rating, loudly flagged** (operator's explicit choice): the point
     estimate uses ALL dimensions with NO synthetic down-weighting. Honesty is
     carried by (a) a loud placeholder warning, (b) lowered `confidence`
     (synthetic dims get `synth_confidence_factor=0.35` data-confidence), and
     (c) a wide `range`. `data_source` is evidence only — it does NOT gate the
     score. This is a deliberate departure from the stricter
     "reliability-gated sub-scores" option, documented in the contract.
- **ACCEPTANCE BAR (important):** Stage 9 is validated for **logical
  correctness assuming trustworthy inputs**, NOT real-world accuracy. Per the
  operator: nothing from Stages 5–9's ball-derived output is useful until
  **Stage 4/4.5 (real ball) is complete.** The smoke test checks schema,
  banding, range-vs-confidence monotonicity, reliability propagation,
  **directional monotonicity** (each scorer + end-to-end: stronger inputs →
  higher rating), confidence-drops-with-synthetic, degradation, and skill
  coverage — never "is the number right."
- **Thresholds are UNCALIBRATED** heuristics anchored to the published USA
  Pickleball Player Skill Rating Definitions — there is no corpus of rated
  amateur footage. This is the dominant limitation (always-on warning). The
  scorers are monotonic linear-interpolation maps (documented constants),
  chosen so the directional smoke checks are meaningful.

### Skill-gap analysis (operator question: "are all skills accounted for?")
The 6 dimensions do NOT cover everything a full assessment considers. The
`skill_coverage` block makes this explicit so Stage 10 / the UI don't imply
full coverage:
- **Covered (a dimension):** net_play, movement, error_control, shot_skill,
  serve, rally_consistency.
- **Proxy / pending** (inside a dimension; await ball v4): serve depth &
  placement, third-shot-drop outcome, dink tolerance, forced/unforced,
  shot placement/targeting, pace/power control.
- **Not captured yet (NEW metric/stage needed; ABSENT from rating):** return
  of serve, volleys/hands-battles, attack conversion, reset under pressure,
  defense/scrambling, partner stacking/poaching, footwork/split-step (the
  Tier-C pose stage), shot-selection IQ.
- **Out of scope (single corner cam):** spin, score/situational decisions.

### Docs updated
- `ARCHITECTURE.md`: Stage 9 status (not-started → implemented), Stages 10–11
  line, importable-folders list (+`stages/rate`), and a NEW
  **"Presentation / UI (deferred to post-4.5)"** entry in the future-stages
  section (operator decision — see below).
- `KNOWN_ISSUES.md`: the "Synthetic ball" section retitled to **Stages 5–9**
  with a Stage 9 bullet (rating is ~0.70 synthetic-weighted + uncalibrated =
  a scaffold until v4).

### UI / presentation decision (operator question: "when do we build the UI?")
The pipeline is headless (JSON in/out); Stage 11 renders video overlays. A
product UI — input/setup flow + an output **dashboard** (metrics display, USAPA
rating + criteria table, heatmaps, improvement plan) — is a **separate
workstream, not one of the 13 stages.** **DECISION: defer the output dashboard
to post-4.5** (build on trustworthy numbers; the locked JSON schemas are the
UI data contract). Input/setup UI is data-source-independent and could come
earlier. Captured in ARCHITECTURE.md future section.

### Commits this session
- `2ba21ca` Stage 9: rate (USAPA skill rating; NEW stage implemented, 10/13 done)

## IMPORTANT caveats for the next session

- **The ball is still synthetic.** Stages 5–9's ball-derived output is
  placeholder. Stage 9's estimate is **0.70 synthetic-weighted** — treat it as
  a scaffold, not a measured rating, until ball v4. The real-data dimensions
  (net_play, movement) are only 0.30 of the weight, so even the trustworthy
  part of the rating is a minority of the number.
- **Rating thresholds are uncalibrated.** When any rated footage exists,
  calibrate the per-dimension tables + weights and replace the directional
  smoke checks with accuracy bars.
- **Pending inputs** (`pending_real_ball` in metrics.json) — forced/unforced,
  drop outcome, dink tolerance, targeting — are recorded as `null` driver
  metrics in Stage 9 and swapped in at v4 with no structural change.

## Local-only artifacts (gitignored — regenerate, don't expect in git)

`data/` and `*.parquet` are gitignored. To reproduce `data/test_clip/` state:
1. `python -m stages.track_players.test_track`  (needs `user_clicks.json`)
2. `python -m stages.pose.test_pose`
3. `python tools/synth_ball.py data/test_clip --seed 1234 --force`
4. `python -m stages.detect_shots.detect_shots data/test_clip --force`
5. `python -m stages.detect_bounces.detect_bounces data/test_clip --force`
6. `python -m stages.classify_shots.classify_shots data/test_clip --force`
7. `python -m stages.segment_rallies.segment_rallies data/test_clip --force`
8. `python -m stages.classify_tracks.classify_tracks data/test_clip --force`
9. `python -m stages.compute_metrics.compute_metrics data/test_clip --force`
10. `python -m stages.rate.rate data/test_clip --force`

Or run the Stage 9 smoke test, which regenerates the whole chain:
`python -m stages.rate.test_rate`

`user_clicks.json` and `roster.json` are gitignored too (under `data/`),
local-only. If lost: re-identify the user to rebuild `user_clicks.json`, and
recreate `roster.json` (`{"schema_version":1,"handedness":{"user":"right",
"partner":"unknown","opp_left":"unknown","opp_right":"unknown"}}` — set `user`
to match `court.json.dominant_hand`).

## What's queued for the next session

**Linear pipeline:**
1. **Stage 10 — plan improvement.** The natural next stage. Input:
   `rating.json` (+ `metrics.json`). Output: `improvement_plan.json` — keyed
   off the weakest rating dimensions, suggesting drills/focus areas. **Write
   the contract first.** Must respect `skill_coverage`: don't prescribe for
   skills the pipeline can't yet measure (not_captured_yet), and flag
   synthetic-derived weaknesses as provisional.
2. **Stage 11 — render annotated video** (`annotated.mp4` + `timeline.json`):
   the video-overlay presentation layer. Consumes all upstream JSON +
   heatmap grids from Stage 8.

**Infrastructure / follow-ups:**
- **Re-wire Stages 3 + 6 to consume `track_roles.json`** (duplicate logic).
- **Stage 2.5 v2:** multi-region color + tighter far-side filter (improves S8
  opponent stats + S9 confidence on opponent-derived signals when multi-role
  rating arrives).
- **Calibrate Stage 9** once rated footage exists.
- **Populate Stage 8 `pending_real_ball` + wire into Stage 9** at ball v4.
- **Tier-C future stages** (ARCHITECTURE.md): pose-technique analysis
  (footwork/split-step — also a future S9 dimension), cross-video trends,
  presentation/UI (post-4.5).

**Footage (offline, David):**
- Better source video for Stage 4.5 + more headroom (higher mount 10–15 ft,
  4K/60fps, faster shutter, simpler backgrounds, fewer adjacent courts).

## Things to NOT touch between sessions
- Stage 4 (`stages/track_ball/`) and Stage 4.5
  (`stages/finetune_ball_model/`): paused/obsolete; don't modify or delete.
- v1/v2 weights on Drive: retained for reference.
- Don't re-attempt ball-detection v1/v2.

## Bring this to the next session

Open a new Claude session and paste:

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, and the relevant stage contract.md
    before proposing anything.

    Stage 9 (rate — USAPA) is done and committed. The ball is still a
    synthetic placeholder (Stage 4.5 paused), so Stage 9's rating is a
    scaffold (0.70 synthetic-weighted, uncalibrated thresholds) — validated
    for logical correctness, not accuracy. I'd like to start Stage 10
    (plan improvement). [or: Stage 11 render, Stages 3/6 role re-wire, or a
    Tier-C stage — see What's queued]

---

Generated at session end on May 29, 2026.

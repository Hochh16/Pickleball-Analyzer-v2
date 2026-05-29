# Session Handoff: Stage 8 (compute metrics) DONE

This document captures the state of Pickleball-Analyzer-v2 at the end of the
May 29 2026 session. It supersedes the previous handoff (Stages 5.5 + 7 done,
Stage 6 rewired), which is now extended through **Stage 8 (compute metrics)**.
The pipeline is now 13 stages, **9 of them implemented** (1, 2, 2.5, 3, 5, 5.5,
6, 7, 8); Stage 4/4.5 remain paused; Stages 9–11 not started.

## Context for the next session

### Project conventions (unchanged)
- Repo: github.com/Hochh16/Pickleball-Analyzer-v2
- Local: `C:\Users\hochh\Pickleball-Analyzer-V2`
- Windows + PowerShell + Python 3.14 (mediapipe 0.10.35, ultralytics 8.4.46,
  torch 2.11 cpu, pandas + pyarrow all import fine)
- Working agreement: **contract → code → smoke test → commit**. Each stage's
  `contract.md` is the source of truth and is approved before code. Design
  decisions / limitations are flagged for the operator BEFORE coding (inputs
  are asked, not guessed).
- Each stage is a standalone Python CLI with file-path I/O. No DB, no shared
  global state. Outputs are sidecar files in one folder per video under `data/`.
- `ARCHITECTURE.md` and `KNOWN_ISSUES.md` are authoritative; read both before
  proposing anything.
- Implemented stages live in **importable** folder names (`stages/detect_shots`,
  `stages/segment_rallies`, `stages/classify_tracks`, `stages/compute_metrics`,
  etc.) — Python module names can't start with a digit. **Per-stage contracts
  live IN the implementation folder.** Numbered stub folders are deleted on
  approval.

### Stage status (post-session)
- **Stages 1, 2, 3**: implemented, smoke-tested. Unchanged this session.
- **Stage 2.5** (classify tracks): implemented; unchanged. Maps track_ids →
  user/partner/opp_left/opp_right/noise. Opponent contamination follow-ups
  still pending (now flows into Stage 8 opponent stats — flagged, not hidden).
- **Stage 4** (TrackNetV2 ball): code-complete, weights don't generalize.
  Unchanged. **Do not touch.**
- **Stage 4.5** (ball detection): **PAUSED** after v1/v2/v3 failures.
  Unchanged. **Do not touch / don't re-attempt v1/v2.**
- **Stage 5** (detect shots): implemented; unchanged.
- **Stage 5.5** (detect bounces): implemented; unchanged.
- **Stage 6** (classify shots): implemented (consumes bounces.json); unchanged.
- **Stage 7** (segment rallies): implemented; unchanged. Emits `end_reason`
  (7 categories) with a documented `end_reason → who-lost` implication that
  Stage 8 consumes for error attribution.
- **Stage 8** (compute metrics): **NEW** — implemented + smoke-tested (15/15)
  + committed (`90b860c`). Details below.
- **Stages 9–11**: not started. Stage 9 (rate USAPA) is the natural next.

## What was done this session

### Built Stage 8 (compute metrics) — NEW stage
- Contract: `stages/compute_metrics/contract.md`. Code:
  `stages/compute_metrics/compute_metrics.py`. Smoke test (15/15):
  `stages/compute_metrics/test_compute_metrics.py`.
- Run: `python -m stages.compute_metrics.compute_metrics data/test_clip --force`
- **Inputs:** classified.json (S6) + rallies.json (S7) + bounces.json (S5.5) +
  players.parquet (S2) + track_roles.json (S2.5) + roster.json + court.json +
  court_zones.json. **First consumer of track_roles.json**, and first to read
  **real player positions** for durable (non-ball) metrics.
- **Output `metrics.json` families:**
  - `match`: rally length/duration stats (+ distribution buckets),
    `by_end_reason` (passthrough from S7), serve-fault rate, shot mix,
    third-shot drop rate, bounce in/out.
  - `error_attribution`: S7's `end_reason → owner`. Server/hitter errors →
    a specific **role** (via track_id); receiver errors → a **team**
    (team_near/team_far, since the specific receiver of two players isn't
    identifiable in v1). `by_owner` reconciles exactly to `n_rallies`.
  - `players` (per role, all 4 best-effort): shot mix, serve stats,
    errors_committed, mean post-speed, and a REAL `position` block —
    depth (`zone_time_frac` kitchen/transition/baseline), lateral
    (`lateral_time_frac` left/center/right), 3×3 `area_time_frac`,
    `court_coverage_frac`, and `movement` (distance total/per-rally/per-min).
    Plus an `unattributed` bucket for shots whose track_id maps to no role.
  - `team`: near + far — `both_at_kitchen_frac`, partner `spacing_ft`,
    per-player transition time.
  - `heatmaps`: numeric grids (player-position per role + ball-landing),
    2 ft bins → 10 cols × 22 rows, row-major. **Stage 11 renders these**;
    Stage 8 stays pure-data.
  - `pending_real_ball`: 4 Tier-B ball-derived metrics
    (`forced_vs_unforced_errors`, `dink_shot_tolerance`,
    `third_shot_drop_outcome`, `opponent_backhand_targeting`) emitted as
    `value: null` + `status` + a `description` of exactly what each will hold
    once ball v4 lands. NOT computed against synthetic ball.
  - `reliability`: names which families are `synthetic_gated` vs `real_data`
    vs `pending`.
- **Scope decisions settled with the operator before coding:**
  1. **Attribution = all roles, best-effort.** Consume track_roles.json,
     emit all 4 roles with `role_confidence` + a `role_contaminated` flag
     (Stage 2.5 opponents are known-contaminated). Surface, don't hide.
  2. **Heatmaps = grid data here, render in Stage 11.**
  3. **Position stats included in v1** (REAL data, ball-independent, durable).
- **Tier-A / B / C metric triage (operator design discussion):**
  - **Tier A (REAL, computed now):** team positioning (both-at-kitchen,
    spacing, transition) + movement work-rate.
  - **Tier B (structure only, synth-gated, `pending_real_ball`):**
    forced/unforced errors, dink/shot tolerance, third-shot-drop outcome,
    opponent-backhand targeting. Documented in KNOWN_ISSUES to be populated
    when real ball lands (forced/unforced is highest priority — feeds S9).
  - **Tier C (future stages, written up in ARCHITECTURE.md):** a pose-derived
    **technique-analysis** stage (split-step timing, posture, contact
    consistency, ready-position recovery — mostly REAL data, the standout
    coaching feature) and **cross-video trend tracking** (the SQLite
    retention layer).
- **Correctness model:** Stage 8 is aggregation, so the smoke test gates on
  **reconciliation invariants**, NOT ball-derived accuracy (no per-player
  ground truth on real tracks): per-role + unattributed shots sum to total;
  `by_owner` sums to `n_rallies`; `by_end_reason` equals S7 exactly; position
  fractions sum to 1 and area marginals equal the depth zones; heatmap grids
  reconcile with in-extent counts (recomputed via the stage's own helpers).
- **Robustness to L/R movement + side-switching** (operator question): per-role
  and per-half aggregation makes position/error metrics robust to players
  moving left↔right between serves or switching sides mid-rally. The only soft
  spot is the `opp_left` vs `opp_right` label split (inherited from Stage 2.5
  median-x); team-level and combined-opponent numbers are unaffected.
  Documented in the contract + KNOWN_ISSUES.

### Docs updated
- `ARCHITECTURE.md`: Stage 8 status (not-started → implemented), Stages 9–11
  line, importable-folders list, and a NEW "Future / proposed stages (not in
  v1)" section writing up Tier C (technique analysis + cross-video trends).
- `KNOWN_ISSUES.md`: NEW "Synthetic ball — Stages 5–8 consume PLACEHOLDER ball
  data" section (downstream consequence + workaround + v4 re-validation plan +
  the `pending_real_ball` procedure), and a Stage 8 opponent L/R split entry.

### Commits this session
- `90b860c` Stage 8: compute metrics (NEW stage implemented; pipeline 13
  stages, 9 done)

## IMPORTANT caveats for the next session

- **The ball is still synthetic.** Everything Stages 5, 5.5, 6, 7, and the
  ball-derived families of Stage 8 produce is derived from
  `tools/synth_ball.py`'s placeholder. Stage 8's `reliability` block names
  exactly which families are synthetic-gated vs real — **do not erase it.**
  When a real ball detector (v4) lands: regenerate ball.parquet, re-run
  S5→5.5→6→7→8 on real (noisy, gappy) trajectories, re-validate, and populate
  the 4 `pending_real_ball` metrics per their descriptions.
- **Stage 8 position / team / movement / player-position heatmaps are REAL
  value now** (from players.parquet, ball-independent). These are durable.
- **Opponent role contamination (Stage 2.5) flows into opp_left/opp_right
  stats** — surfaced via `role_contaminated` + warnings (on test_clip both
  opponents flag at confidence 0.5 < 0.55 floor; their movement/spacing is
  visibly inflated by adjacent-court ID-swaps). Team-level (near/far) numbers
  hold. Tightens with Stage 2.5 v2.
- **end_reason diversity on the synth clip is skewed** (test_clip:
  ~29/42 ball-off-frame, 0 serve-fault, 0 double-bounce this seed) — so
  Stage 8's serve-fault rate and some error buckets read low. This is an
  upstream synth-skew (documented in the previous handoff), not a Stage 8 bug.

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
   (independent of ball; can run any time after step 1)
9. `python -m stages.compute_metrics.compute_metrics data/test_clip --force`

Or just run the Stage 8 smoke test, which regenerates the whole chain:
`python -m stages.compute_metrics.test_compute_metrics`

`user_clicks.json` and `roster.json` are gitignored too (under `data/`), so
they're local-only. If lost: re-identify the user in a few frames to rebuild
`user_clicks.json`, and recreate `roster.json` (`{"schema_version":1,
"handedness":{"user":"right","partner":"unknown","opp_left":"unknown",
"opp_right":"unknown"}}` — set `user` to match `court.json.dominant_hand`).

## What's queued for the next session

**Linear pipeline:**
1. **Stage 9 — rate USAPA.** The natural next stage. Input: `metrics.json`
   (+ maybe rallies.json). Output: `rating.json` — a rule-based USAPA-style
   skill rating anchored in published level descriptions, reading Stage 8's
   shot mix, third-shot drop rate, error attribution, kitchen/positioning
   stats, etc. **Write the contract first.** Note: the highest-signal rating
   input (unforced-error rate) is in Stage 8's `pending_real_ball` and stays
   null until ball v4 — Stage 9 v1 must rate on what's available and flag the
   gap.

**Infrastructure / follow-ups:**
- **Re-wire Stage 3 (scope filter) and Stage 6 (is_user → role mapping) to
  consume `track_roles.json`.** Long-standing; both work today but duplicate
  logic. Light refactor; smoke tests must still pass.
- **Stage 2.5 v2:** multi-region clothing-color matching + tighter far-side
  filter to drop adjacent-court opponent contamination. Directly improves
  Stage 8's opponent stats and the opp_left/opp_right split.
- **Tier-C future stages** (now in ARCHITECTURE.md): pose-derived technique
  analysis; cross-video trend tracking (SQLite). Mostly REAL data — could be
  built before ball v4.
- **Populate Stage 8 `pending_real_ball`** once ball v4 exists.
- **Diagonal service-box validation** + **net-hit vs short-shot split**
  (deferred from Stage 7).

**Footage (offline, David):**
- Better source video for Stage 4.5 AND more headroom: higher mount (10–15 ft),
  4K/60fps, faster shutter, simpler backgrounds, fewer adjacent courts in
  frame. Serves ball SNR (4.5), fixes Stage 6 lob detection, and reduces the
  adjacent-court contamination that skews Stage 7 end_reasons + Stage 8
  opponent stats.

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

    Stage 8 (compute metrics) is done and committed. The ball is still a
    synthetic placeholder (Stage 4.5 paused), and Stage 8's ball-derived
    families + the 4 pending_real_ball metrics reflect that. I'd like to
    start Stage 9 (rate USAPA). [or: re-wire Stages 3/6 to track_roles.json,
    or build a Tier-C technique stage — see What's queued]

---

Generated at session end on May 29, 2026.

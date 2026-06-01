# Session Handoff: Stage 10 (plan improvement) DONE

State of Pickleball-Analyzer-v2 at the end of the May 29 2026 session.
Supersedes the Stage 9 handoff; now extended through **Stage 10 (plan
improvement)**. The pipeline is 13 stages, **11 implemented** (1, 2, 2.5, 3, 5,
5.5, 6, 7, 8, 9, 10); Stage 4/4.5 remain paused; **Stage 11 (render) is the
only remaining stage.**

## Context for the next session

### Project conventions (unchanged)
- Repo: github.com/Hochh16/Pickleball-Analyzer-v2 · Local:
  `C:\Users\hochh\Pickleball-Analyzer-V2`
- Windows + PowerShell + Python 3.14 (mediapipe 0.10.35, ultralytics 8.4.46,
  torch 2.11 cpu, pandas + pyarrow).
- Working agreement: **contract → code → smoke test → commit**, decisions /
  limitations flagged for the operator (inputs asked, not guessed) BEFORE
  coding. Each stage's `contract.md` is the source of truth.
- Standalone Python CLIs, file-path I/O, sidecar JSON per video under `data/`.
  No DB, no shared state. `ARCHITECTURE.md` + `KNOWN_ISSUES.md` authoritative.
- Implemented stages live in **importable** folders (`stages/plan_improvement`,
  `stages/rate`, `stages/compute_metrics`, …); module names can't start with a
  digit. Per-stage contracts live IN the folder; numbered stubs deleted on
  approval.

### Stage status (post-session)
- **Stages 1, 2, 2.5, 3**: implemented; unchanged. (2.5 opponent contamination
  follow-ups still pending — flows into S8/S9/S10, flagged.)
- **Stage 4** (TrackNetV2 ball): code-complete, weights don't generalize.
  **Do not touch.**
- **Stage 4.5** (ball detection): **PAUSED** after v1/v2/v3. **Do not touch.**
- **Stages 5, 5.5, 6, 7, 8, 9**: implemented; unchanged.
- **Stage 10** (plan improvement): **NEW** — implemented + smoke-tested (8/8) +
  committed (`10c8c6f`). Details below.
- **Stage 11** (render annotated video): not started — the last stage.

## What was done this session

### Built Stage 10 (plan improvement) — NEW stage
- Contract: `stages/plan_improvement/contract.md`. Code:
  `stages/plan_improvement/plan_improvement.py`. Smoke (8/8):
  `stages/plan_improvement/test_plan_improvement.py`.
- Run: `python -m stages.plan_improvement.plan_improvement data/test_clip --force`
- **Input:** `rating.json` (S9) + `metrics.json` (S8, optional context).
- **Output `improvement_plan.json`** for the **user**:
  - `current` → `target` (next USAPA half-step, cap 5.0; `--target-band`
    overrides).
  - `focus_areas[]`: dimensions below target, prioritized. Each has
    `gap_to_target`, `priority_score`, a **data-grounded finding** (built from
    rating.json's `driver_metrics` so it can't drift from the rating),
    `why_it_matters`, **1–3 drills** (from a built-in library, some selected
    conditionally on driver values), and `data_source`/`confidence`/
    `provisional_note`.
  - `strengths[]`: dimensions already ≥ target.
  - `developing_capability`: **forward-looking scaffold** — the
    `proxy_or_pending` + `not_captured_yet` skills from Stage 9's
    `skill_coverage`, each with `unlocked_by` / `will_assess` /
    `will_recommend`; plus `out_of_scope`.
  - `reliability`: real vs provisional focus-area counts.
- **Scope decisions (operator, before coding):**
  1. **Depth = focus areas + drills/cues** (no scheduling/dosage in v1).
  2. **Include synthetic weaknesses, flagged `provisional`**; real-data
     weaknesses rank higher-confidence via
     `priority_score = gap · weight · (0.5 + 0.5·dim_confidence)` (real ~1.0,
     synthetic ~0.35 → mild lift, doesn't bury large synthetic gaps).
  3. **Forward-looking developing_capability block** (the operator's explicit
     ask: "account for skills measured via synthetic + those developed
     downstream, so full capability is there once 4/4.5 + critical skills
     complete"). v1 emits NO recommendations for unmeasured skills — it only
     documents what completes the plan.
- **Self-healing flags:** when real ball detection (v4) lands, Stage 9's
  `data_source` flips synthetic→real, so Stage 10's `provisional` flags clear
  automatically and those focus areas become high-confidence — no Stage 10
  change. developing_capability entries migrate into focus_areas as each
  skill's metric lands (the descriptor table is the migration checklist).
- **Same honesty model:** loud synthetic + uncalibrated warnings; a warning if
  the #1 focus area is provisional; reliability counts.

### Docs updated
- `ARCHITECTURE.md`: Stage 10 status (not-started → implemented), Stage 11 line,
  importable-folders list (+`stages/plan_improvement`).

### Commits this session
- `10c8c6f` Stage 10: plan improvement (NEW stage implemented, 11/13 done)

## IMPORTANT caveats for the next session
- **The ball is still synthetic.** Stages 5–10's ball-derived output is
  placeholder. On test_clip the plan's #1 focus area is `net_play` (REAL,
  trustworthy now); the provisional (synthetic) focus areas are a scaffold
  until ball v4. The real-data part of every output is the durable value today.
- **Uncalibrated thresholds** (Stages 9 + 10): which drills move which metric
  is unvalidated. Calibrate with real-data + outcome tracking later.
- **developing_capability is the roadmap** for full rating/plan capability —
  it enumerates exactly what each downstream skill needs (ball v4, new metric
  stages, the Tier-C pose stage).

## Local-only artifacts (gitignored — regenerate, don't expect in git)
`data/` and `*.parquet` are gitignored. To reproduce `data/test_clip/`:
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
11. `python -m stages.plan_improvement.plan_improvement data/test_clip --force`

Or run the Stage 10 smoke test, which regenerates the whole chain:
`python -m stages.plan_improvement.test_plan_improvement`

`user_clicks.json` / `roster.json` are gitignored (local-only). If lost:
re-identify the user to rebuild clicks; recreate `roster.json`
(`{"schema_version":1,"handedness":{"user":"right","partner":"unknown",
"opp_left":"unknown","opp_right":"unknown"}}`, `user` = `court.json.dominant_hand`).

## What's queued for the next session

**Linear pipeline — the LAST stage:**
1. **Stage 11 — render annotated video.** Output: `annotated.mp4` +
   `timeline.json`. Consumes the video + ALL upstream JSON (court, players,
   poses, shots, bounces, classified, rallies, track_roles, metrics, rating,
   improvement_plan) + Stage 8's heatmap grids. The video-overlay presentation
   layer (court lines, player roles/boxes, shot/bounce markers, rally
   end_reasons, optional rating/plan overlay). **Write the contract first.**
   Decide: what to overlay, how to render heatmaps, whether to burn in the
   rating/plan or keep them in `timeline.json` for the (deferred) dashboard.

**Infrastructure / follow-ups:**
- **Re-wire Stages 3 + 6 to consume `track_roles.json`** (duplicate logic).
- **Stage 2.5 v2:** multi-region color + tighter far-side filter (improves S8
  opponent stats + S9/S10 confidence).
- **Calibrate Stages 9 + 10** once rated footage exists.
- **Ball v4 → populate Stage 8 `pending_real_ball`**, which auto-firms S9
  dimensions + clears S10 provisional flags; then re-validate the whole chain.
- **Tier-C future stages** (ARCHITECTURE.md): pose-technique (also a future S9
  dimension + S10 focus area), cross-video trends, presentation/UI (post-4.5).

**Footage (offline, David):** better source video for Stage 4.5 + more headroom
(higher mount 10–15 ft, 4K/60fps, faster shutter, simpler backgrounds, fewer
adjacent courts).

## Things to NOT touch between sessions
- Stage 4 (`stages/track_ball/`) + Stage 4.5 (`stages/finetune_ball_model/`):
  paused/obsolete; don't modify or delete.
- v1/v2 weights on Drive: retained. Don't re-attempt ball-detection v1/v2.

## Bring this to the next session

Open a new Claude session and paste:

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, and the relevant stage contract.md
    before proposing anything.

    Stages through 10 are done and committed. The ball is still a synthetic
    placeholder (Stage 4.5 paused), so Stages 5-10 are validated for logical
    correctness, not accuracy. Stage 11 (render annotated video) is the last
    pipeline stage — I'd like to build it. [or: Stages 3/6 role re-wire,
    Stage 2.5 v2, or a Tier-C stage — see What's queued]

---

Generated at session end on May 29, 2026.

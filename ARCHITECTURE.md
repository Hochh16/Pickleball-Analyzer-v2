# Architecture

## Implementation status (2026-06-14)

This is a design doc; per-stage status lives here as a quick pointer (full
detail in `docs/SESSION_HANDOFF.md` and `KNOWN_ISSUES.md`).

- **Stages 1–3** (calibrate, track players, pose): implemented, smoke-tested.
- **Stage 2.5** (classify tracks): NEW stage (added 2026-05-22; pipeline is now
  13 stages). Maps ByteTrack track_ids to logical roles
  (user/partner/opp_left/opp_right/noise) via multi-cue re-identification
  (click-anchored continuity + height + clothing color). Runs on real player
  tracking (no ball dependency). Unlocks complete user labeling, per-player
  stats, and non-user handedness. See `stages/classify_tracks/contract.md`.
- **Stage 4** (track ball, TrackNetV2): **v4 WORKING** (2026-06-11/12). Rewritten
  as `stages/track_ball/track_ball_v4.py` (720p inference + trajectory
  post-processing) against the v4-trained `data/models/ball_model_v4.pt`.
  Validated vs ground truth on pb_2min frames 300–420: 39/40 labeled balls,
  **median 4.9px error at 4K**, 100% within 25px. The **first real full-clip
  `ball.parquet`** (`synthetic: false`) was produced for `data/pb_2min/` via the
  GPU (Colab) path `stages/track_ball/infer_v4.ipynb`: 7164 frames, detect_frac
  0.676, all coords in-bounds. Production inference is GPU-only (CPU ~11 s/frame).
- **Stage 4.5** (ball-detection calibration): **v4 LANDED** (2026-06-11). New
  4K/60fps footage solved the SNR wall that doomed v1–v3 (ball ~13px, median
  intensity 71/255, present 88% of frames). v4 trained a temporal TrackNet done
  right — focal loss (fixes v1 BCE/v2 MSE failures), input raised to 1280×720
  (old 512×288 reshrank the ball to ~2px), diverse multi-clip training, and
  court-agnostic trajectory post-processing. Trained weights: **val recall 0.90
  same-court, 0.54 cross-court** — cross-court generalization is the open work
  (see `KNOWN_ISSUES.md`), required for varied indoor/outdoor venues. See
  `stages/finetune_ball_model/contract_v4.md`. The synthetic caveat across
  Stages 5–11 lifts **per-stage as each is re-run** on the real ball. **DONE so
  far: Stages 1–3, 5, 5.5, 6 run on the real ball for pb_2min** (operator-validated);
  Stages 7–11 pending.
- **Stage 5** (detect shots): **implemented + smoke-tested; v0.2.0 real-ball
  adapted** (`8aa9164`). Run on the real v4 ball for pb_2min (304→45 real
  strikes, operator-validated): 4K resolution + 60fps scaling, teleport-drop,
  is_user-from-roles, and a net-side ball-handling filter (rejects catch/bounce
  between points). Still runs on synthetic for the smoke test (real-ball filters
  gated off). Synthetic caveat lifted for Stage 5 on real clips.
- **Stage 5.5** (detect bounces): NEW stage (added 2026-05-27; pipeline now 13
  stages); **v0.2.0 real-ball adapted** (`740fac9`, run on pb_2min, 135→16
  bounces, operator-validated). Reuses Stage 5's impulse signal with the opposite
  proximity rule to emit every ground bounce, plus a y-velocity-flip that (on
  real ball) is required for ALL bounces — a real bounce reverses vertical
  direction; a mid-air wobble doesn't. Real-ball adds resolution/fps scaling, an
  apex/off-court filter, and ground-contact refinement (accurate far-court zones).
  Outputs `bounces.json` with `between_shots`, in/out classification, and
  `is_at_feet` per bounce. Consumed by Stage 6 (`is_volley`) and downstream by
  Stage 7 (rally-end reasons). Synthetic caveat lifted for Stage 5.5 on real clips. See
  `stages/detect_bounces/contract.md`.
- **Stage 6** (classify shots): **implemented + smoke-tested; v0.3.0 real-ball
  adapted** (run on pb_2min, 45 shots, 0 unknown, operator-validated). Stroke side
  (user only until role classification), shot type, volley. On the real ball the
  **volley flag is decoupled from Stage 5.5's precision bounce list** (which
  under-detects → false volleys) and uses a recall-focused **local trajectory
  scan** (ground bounce = interior pixel_y local peak with descent-in + rebound-out;
  the bounce list is an occlusion fallback). Plus lob-requires-slow, a tweener
  arc-shape drive/drop tiebreak, and fps/resolution scaling. **Known real-ball
  limits** (KNOWN_ISSUES): pixel-speed underestimates depth/height → some
  down-court drives mistype as drops (needs court-plane/3D speed); serve labeling
  depends on Stage 5 `is_serve`; courtesy feeds read as volleys (Stage 7 to
  exclude). Synthetic caveat lifted for Stage 6 on real clips. See
  `stages/classify_shots/contract.md`.
- **Stage 7** (segment rallies): **implemented + smoke-tested** (2026-05-29).
  Groups shots into rallies by `is_serve` and tags each with an
  `end_reason` from a 7-category set (`serve-fault`, `double-bounce`,
  `ball-out`, `net-or-short`, `ball-not-returned`, `ball-off-frame`,
  `unknown`). Uses court-side reasoning from `impact_court_xy_ft` (shots)
  and `court_xy_ft` (bounces) to classify hitter-side vs receiver-side
  events and detect kitchen-fault serves. **Role-blind v1**: no
  `winner_side`, no `track_roles.json` dependency — winner attribution
  deferred to Stage 8. Same synthetic-ball caveat. See
  `stages/segment_rallies/contract.md`.
- **Stage 8** (compute metrics): **implemented + smoke-tested** (2026-05-29).
  Aggregates `classified.json` + `rallies.json` + `bounces.json` +
  `players.parquet` + `track_roles.json` into `metrics.json`: match summary
  (rally lengths, serve-fault rate, shot mix, third-shot drop rate, bounce
  in/out), per-role player stats, `error_attribution` (Stage 7's
  `end_reason → owner` mapping; server/hitter errors to a role, receiver
  errors to a team), team positioning + movement, and numeric heatmap grids
  (player-position per role + ball-landing) for Stage 11 to render. First
  consumer of `track_roles.json` (all-roles, best-effort, contamination
  flagged) and first to read real player positions for **durable** (non-ball)
  position/coverage metrics. Tier-B ball-derived metrics are emitted as
  null `pending_real_ball` placeholders with descriptions. A `reliability`
  block names which families are synthetic-gated vs real. Same synthetic-ball
  caveat for ball-derived families. See `stages/compute_metrics/contract.md`.
- **Stage 9** (rate — USAPA): **implemented + smoke-tested** (2026-05-29).
  Maps `metrics.json` to a USA Pickleball skill rating for the **user**:
  continuous `estimate` + nearest half-step `band` + confidence `range`, from
  six USAPA-anchored dimensions (net_play + movement are REAL, ~0.30 of the
  weight; error_control/shot_skill/serve/rally_consistency are
  synthetic-derived, ~0.70). **Full rating, loudly flagged** (operator's
  choice): the estimate uses all dimensions, while a placeholder warning,
  lowered `confidence`, and a wide `range` carry the synthetic-ball honesty.
  Emits a `skill_coverage` map (covered / proxy_or_pending / not_captured_yet /
  out_of_scope) so the rating doesn't imply full competency coverage.
  **Thresholds are uncalibrated heuristics** (no rated-footage corpus). Smoke
  test gates on schema + banding + directional monotonicity, not accuracy. See
  `stages/rate/contract.md`.
- **Stage 10** (plan improvement): **implemented + smoke-tested** (2026-05-29).
  Turns `rating.json` (+ `metrics.json`) into `improvement_plan.json` for the
  **user**: the gap to the next USAPA half-step, prioritized **focus areas**
  (each with a data-grounded finding + 1–3 drills/cues from a built-in
  USAPA-anchored library), and a forward-looking **developing_capability**
  block that scaffolds in the skills not yet measurable (from Stage 9's
  `skill_coverage`) — so the plan reaches full capability once ball v4 + the
  new metric/pose stages land. Synthetic-ball-derived focus areas are flagged
  `provisional` and mildly down-weighted in the priority score (real-data
  weaknesses rank higher-confidence). Smoke test gates on schema + focus
  correctness + directional behavior, not accuracy (uncalibrated, like Stage
  9). See `stages/plan_improvement/contract.md`.
- **Stage 11** (render annotated video): **implemented + smoke-tested**
  (2026-05-29). **Pure consumer** — draws upstream decisions onto the actual
  source video and emits `annotated.mp4` + `timeline.json` + standalone heatmap
  PNGs; recomputes nothing. Overlays: court lines (homography-projected; verified
  pixel-exact against the calibrated corners), player boxes + roles, ball marker
  + trail, shot/bounce markers, rally/end_reason banner, HUD card (rating + top
  focus area), top-down minimap inset, and a persistent synthetic-ball watermark.
  `timeline.json` is the scrubbable event stream (the deferred dashboard's data
  contract). Built/tested on the existing `data/test_clip/video.mp4`; re-runs
  unchanged on real footage (watermark drops when `ball_source != synthetic`).
  See `stages/render/contract.md`.

**Pipeline status: all 11 logical stages (13 numbered) are implemented and
smoke-tested. Ball detection (Stage 4/4.5) is no longer paused — v4 is working
and has produced a real, validated full-clip `ball.parquet` for `data/pb_2min/`.
The chain has run end-to-end on synthetic-ball data; every ball-derived output
remains a validated scaffold until Stages 5–11 are re-run on the real ball
(pending: pb_2min needs Stages 1–3 first). Two open items gate broad reliance on
the detector: cross-court generalization (0.54 cross-court recall) and inference
throughput (~2.9 fps, CPU-decode-bound) — see `KNOWN_ISSUES.md`.**

Implemented stages live in **importable** folders (`stages/calibrate`,
`stages/track_players`, `stages/pose`, `stages/track_ball`,
`stages/detect_shots`, `stages/detect_bounces`, `stages/classify_shots`,
`stages/segment_rallies`, `stages/classify_tracks`, `stages/compute_metrics`,
`stages/rate`, `stages/plan_improvement`, `stages/render`) — Python modules
can't start with a digit, so the numbered folders in the pipeline diagram below
are illustrative, not import paths.

## Pipeline-wide assumptions

These assumptions apply to the entire pipeline. Any stage may rely on them
without re-validating. Violating them is undefined behavior — stages may
fail loudly, produce wrong outputs, or both.

### Camera placement

The camera is positioned in **one of the two far corners of the court,
approximately 6 feet high, with the entire court visible in frame**. Side
views, baseline-center views, low-angle views, and partial-court framings
are not supported.

This single constraint underpins:
- Stage 1 (calibrate) — the homography assumes a quadrilateral court
  visible from a corner-elevated angle.
- Stage 2 (track players) — player detection and ground-projection assume
  feet are visible and the camera is above the action.
- Stage 4 (track ball) — TrackNet weights and the pixel-space ROI assume
  this viewpoint distribution.
- Downstream stages — shot attribution, side-of-court reasoning, etc.

If we ever support other camera positions, this assumption needs to be
revisited stage-by-stage. Not in scope for v1.

### Single ball, single match

One pickleball in active play at a time. One continuous match per video
file (no edited highlight reels, no multi-game compilations). Camera does
not pan, zoom, or cut during the match.

### Frame rate and resolution

Any reasonable phone-camera output is acceptable: 24-60 fps, 720p or
higher. Stages may resize internally for model input but operate on the
native frame index from the source video.

## Layered pipeline

~~~
video.mp4
    │
    ▼
[1] Calibrate ──► court.json, court_zones.json
    │
    ▼
[2] Track players (uses court.json) ──► players.parquet
    │
    ▼
[2.5] Classify tracks (players + clicks + roster) ──► track_roles.json
    │   (maps track_ids -> user/partner/opp_left/opp_right/noise)
    ▼
[3] Pose (uses players bbox crops) ──► poses.parquet
    │
    ▼
[4] Track ball (uses court.json) ──► ball.parquet
    │
    ▼
[5] Detect shots (players + pose + ball) ──► shots.json
    │
    ▼
[5.5] Detect bounces (shots + ball + players) ──► bounces.json
    │   (consumed by Stage 6 for is_volley, by Stage 7 for end_reasons)
    ▼
[6] Classify shots (shots + bounces + pose + ball) ──► classified.json
    │
    ▼
[7] Segment rallies (classified + bounces + ball) ──► rallies.json
    │
    ▼
[8] Compute metrics (everything above) ──► metrics.json
    │
    ▼
[9] Rate USAPA (metrics) ──► rating.json
    │
    ▼
[10] Plan improvement (rating) ──► improvement_plan.json
    │
    ▼
[11] Render annotated video (video + all JSON) ──► annotated.mp4, timeline.json
~~~

## Stage rules

1. **Each stage is a standalone Python CLI script.** It can be run from the command line on any video without invoking other stages. Example: `python -m stages.01_calibrate --video data/match_001/video.mp4`

2. **Inputs are file paths. Outputs are file paths.** No in-memory pipelines, no shared global state, no class hierarchies that span stages.

3. **Schemas are versioned.** Every output file has a `schema_version` field. Breaking schema changes increment it. Non-breaking additions (new optional column, new optional metadata field) do not. Stages that consume a schema must check `schema_version` and fail loudly on a version they were not written for.

4. **No stage modifies its inputs.** Output files are always new files.

5. **Each stage has a `contract.md`** in its folder. The contract is the source of truth — it specifies inputs, outputs, schema, edge cases. The contract is reviewed and approved before code is written.

6. **Each stage has a smoke test.** A short test clip (≤30 seconds) with known-correct expected outputs. The stage is "done" only when the smoke test passes.

7. **Failures are loud.** No silent fallbacks. If a stage can't produce its output, it raises with a clear message.

## Build vs. third-party

Use these (don't reinvent):
- `ultralytics` — YOLO + ByteTrack
- `opencv-python` — homography, video I/O, drawing
- `mediapipe` — pose
- `pandas` + `pyarrow` — parquet I/O

Build these (no off-the-shelf alternative):
- Court calibration UI (camera-angle dependent)
- Shot impact detection (sport-specific heuristics)
- Shot classification (pickleball-specific shot taxonomy)
- USAPA rating engine (rule-based, anchored in published descriptions)
- Improvement plan engine

## Storage

Sidecar files. No database. One folder per video under `data/`.

If we ever need cross-video queries, we add a thin SQLite layer that indexes the JSON files. Not in v1.

## Future / proposed stages (not in v1)

These are deferred product directions, captured so the roadmap is explicit.
They are NOT part of the current 13-stage pipeline and have no contract yet.
They emerged from the Stage 8 metrics design (the "Tier C" differentiators):
deep value that leans on the pipeline's real (non-ball) data assets — pose and
position — and so is mostly unaffected by the synthetic-ball pause.

### Proposed — Technique analysis (pose-derived)

A stage consuming `poses.parquet` (Stage 3, REAL) + `classified.json`
(shot frames) + `players.parquet`, emitting `technique.json`: per-player
biomechanical / footwork coaching signals that almost no consumer app offers.
Candidate metrics:
- **Split-step timing** — did the player load/hop as the opponent struck?
  (vertical foot/hip motion aligned to opponent contact frames).
- **Contact-point consistency** — variance of impact position relative to the
  body across shots of a given type.
- **Posture on dinks** — knee bend / hip height when hitting from the kitchen
  (low-and-balanced vs reaching high).
- **Ready-position recovery** — time to return to a neutral paddle-up stance
  after a shot.

Mostly REAL data (pose), so durable now; deferred only because it's its own
modeling effort, not because it needs the ball. Would feed the Stage 9 rating.

### Proposed — Presentation / UI (deferred to post-4.5)

The pipeline is headless today: inputs are hand-authored/clicked JSON (court
calibration UI + `user_clicks.json` + `roster.json`), outputs are sidecar JSON,
and Stage 11 renders *video overlays* (`annotated.mp4` + `timeline.json`). A
product UI — the input/setup flow (upload → calibrate → identify players →
roster) and the **output dashboard** (metrics display, USAPA rating + criteria
table, heatmap visualization, improvement plan) — is a **separate workstream,
not one of the 13 stages.**

**DECISION (timing):** defer the output dashboard until the analytical pipeline
produces trustworthy numbers (post real-ball v4), to avoid building on
placeholder data. The locked JSON schemas (`metrics.json`, `rating.json`,
`timeline.json`) ARE the UI's data contract, so careful schema design now is the
real UI prep. The input/setup UI is data-source-independent and could be built
earlier if needed.

### Proposed — Cross-video trend tracking

The retention layer: aggregate `metrics.json` across sessions to show
improvement curves per metric (kitchen-line %, unforced-error rate, third-shot
drop rate, movement work-rate, …) over weeks/months. This is the concrete use
case for the thin SQLite index mentioned under Storage — it indexes each
video's sidecar JSON and serves per-metric time series to a dashboard / report.
Needs a stable `metrics.json` schema (hence Stage 8's `schema_version`) and a
notion of player identity across videos.
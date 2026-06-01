# Session Handoff: Stage 11 (render) DONE — PIPELINE COMPLETE

State of Pickleball-Analyzer-v2 at the end of the May 29 2026 session.
Supersedes the Stage 10 handoff; extended through **Stage 11 (render annotated
video)**. **The 13-stage pipeline is now end-to-end runnable** — every stage is
implemented + smoke-tested EXCEPT ball detection (Stage 4/4.5), which is paused.
The whole chain runs today on synthetic-ball data; every ball-derived output is
a validated scaffold until real ball detection (v4) lands.

## Context for the next session

### Project conventions (unchanged)
- Repo: github.com/Hochh16/Pickleball-Analyzer-v2 · Local:
  `C:\Users\hochh\Pickleball-Analyzer-V2`
- Windows + PowerShell + Python 3.14 (mediapipe 0.10.35, ultralytics 8.4.46,
  torch 2.11 cpu, pandas + pyarrow, opencv-python 4.13).
- Working agreement: **contract → code → smoke test → commit**; decisions /
  limitations flagged for the operator (inputs asked, not guessed) BEFORE
  coding. Each stage's `contract.md` is the source of truth.
- Standalone Python CLIs, file-path I/O, sidecar outputs per video under
  `data/`. No DB, no shared state. `ARCHITECTURE.md` + `KNOWN_ISSUES.md`
  authoritative.
- Implemented stages live in **importable** folders (`stages/render`,
  `stages/plan_improvement`, `stages/rate`, …); module names can't start with a
  digit. Per-stage contracts live IN the folder; numbered stubs deleted on
  approval.

### Stage status (post-session) — ALL IMPLEMENTED except 4/4.5
- **1 calibrate, 2 track_players, 2.5 classify_tracks, 3 pose** — implemented.
- **4 track_ball** — code-complete, weights don't generalize. **PAUSED, do not
  touch.**
- **4.5 finetune_ball_model** — **PAUSED** after v1/v2/v3 failures. **Do not
  touch / don't re-attempt v1/v2.**
- **5 detect_shots, 5.5 detect_bounces, 6 classify_shots, 7 segment_rallies,
  8 compute_metrics, 9 rate, 10 plan_improvement** — implemented.
- **11 render** — **NEW** this session, implemented + smoke-tested (9/9) +
  committed (`5b15815`). Details below.

## What was done this session

### Built Stage 11 (render annotated video) — NEW stage, completes the pipeline
- Contract: `stages/render/contract.md`. Code: `stages/render/render.py`. Smoke
  (9/9): `stages/render/test_render.py`.
- Run: `python -m stages.render.render data/test_clip --force`
  (full clip; slow). Range-limit for dev:
  `... --start-frame 1000 --max-seconds 5`.
- **Pure consumer** — recomputes nothing; draws what upstream JSON decided.
- **Inputs:** `video.mp4` + `court.json` required; everything else optional
  (missing → that layer skipped + warning, so it tolerates a partial pipeline).
- **Outputs:**
  - `annotated.mp4`: the ACTUAL footage + AR overlays — court lines
    (homography-projected; verified pixel-exact vs the calibrated corners),
    player boxes + role labels, ball marker + fading trail, shot markers,
    bounce markers (in/out), rally banner with `end_reason`, HUD card
    (rating band + estimate + range + #1 focus area, provisional-tagged),
    top-down **minimap inset** (role dots + bounces in court-feet), and a
    persistent **SYNTHETIC-BALL watermark**. Optional `--pose` skeleton,
    `--labels` shot-type text (off by default).
  - `timeline.json`: synchronized event stream (`rally_start`/`rally_end`/
    `shot`/`bounce`, sorted by frame) + a `summary` carrying rating + plan.
    The deferred dashboard's data contract; emitted for the full clip even when
    the video render is range-limited.
  - `heatmap_*.png`: standalone position (per role) + ball-landing heatmaps over
    a court diagram (Stage 8 deferred heatmap *rendering* to here).
- **Scope decisions (operator, before coding):** annotate real video + minimap
  inset (not a schematic-only video); heatmaps as standalone PNGs; rating/plan
  as HUD card + full data in timeline; default optional layer = ball trail
  (pose + labels flag-gated off).
- **Synthetic discipline:** the watermark + `summary.synthetic_ball` +
  warnings make the placeholder status unmissable on the most real-looking
  output. The minimap deliberately does NOT project the mid-air ball to court
  (geometrically wrong) — ball appears on the minimap only at bounce frames.
- **Self-updating:** re-runs unchanged on real footage; the watermark drops
  automatically once `ball_source != synthetic`.
- **Perf:** full-clip render is slow (CPU draw + encode, ~8125 frames);
  `--start-frame/--end-frame/--max-seconds/--fps-out` bound it. Smoke renders
  ~60 frames.
- **Verified visually:** extracted an annotated frame — banner, HUD, watermark,
  role-colored boxes, ball trail, shot marker, minimap all render correctly;
  court-line projection confirmed pixel-exact against `court.json`'s clicked
  corners.

### Docs updated
- `ARCHITECTURE.md`: Stage 11 status (not-started → implemented) + a
  pipeline-complete note; importable-folders list (+`stages/render`).

### Commits this session
- `5b15815` Stage 11: render annotated video (NEW stage; pipeline complete)

## The full pipeline (runnable today)

```
video.mp4
 → [1] calibrate → court.json, court_zones.json
 → [2] track_players → players.parquet
 → [2.5] classify_tracks → track_roles.json
 → [3] pose → poses.parquet
 → [4] track_ball → ball.parquet            (PAUSED → tools/synth_ball.py placeholder)
 → [5] detect_shots → shots.json
 → [5.5] detect_bounces → bounces.json
 → [6] classify_shots → classified.json
 → [7] segment_rallies → rallies.json
 → [8] compute_metrics → metrics.json
 → [9] rate → rating.json
 → [10] plan_improvement → improvement_plan.json
 → [11] render → annotated.mp4 + timeline.json + heatmap_*.png
```

## IMPORTANT caveats for the next session
- **The ball is still synthetic.** Stages 5–11's ball-derived output is
  placeholder; the annotated video carries a burned-in watermark. The durable
  real value today is positioning/movement (players.parquet) — net_play +
  movement metrics, the player-position heatmaps, and the minimap.
- **Stages 9 + 10 thresholds are uncalibrated** (no rated-footage corpus).
- **Stage 2.5 opponent contamination** flows into S8/S9/S10 (flagged).
- The pipeline being "complete" means **structurally** complete + logically
  correct given trustworthy inputs — NOT validated for real-world accuracy.
  That gate is ball v4 + new footage.

## Local-only artifacts (gitignored — regenerate, don't expect in git)
`data/` and `*.parquet` are gitignored (so are all Stage 11 outputs:
annotated.mp4, timeline.json, heatmap_*.png — under `data/`). To reproduce
`data/test_clip/`:
1. `python -m stages.track_players.test_track`  (needs `user_clicks.json`)
2. `python -m stages.pose.test_pose`
3. `python tools/synth_ball.py data/test_clip --seed 1234 --force`
4–11. detect_shots → detect_bounces → classify_shots → segment_rallies →
   classify_tracks → compute_metrics → rate → plan_improvement → render
   (each `python -m stages.<folder>.<module> data/test_clip --force`).

Fastest full-chain check: `python -m stages.render.test_render` (regenerates
the whole chain, renders a short range).

`user_clicks.json` / `roster.json` are gitignored (local-only). If lost:
re-identify the user; recreate `roster.json`
(`{"schema_version":1,"handedness":{"user":"right","partner":"unknown",
"opp_left":"unknown","opp_right":"unknown"}}`, `user` = `court.json.dominant_hand`).

## What's queued for the next session

**No linear pipeline stages remain.** The work is now: make the outputs TRUE,
and harden/extend.

1. **Ball detection v4 (Stage 4/4.5) — the critical unlock.** Needs better
   footage (see below) and/or a temporal multi-frame approach. When it lands:
   regenerate `ball.parquet`, re-run 5→11 on real trajectories, re-validate
   every stage, and CALIBRATE Stages 9/10 against rated footage. Stage 8's
   `pending_real_ball` auto-feeds S9 dims + clears S10 provisional flags.
- **Re-wire Stages 3 + 6 to consume `track_roles.json`** (duplicate logic).
- **Stage 2.5 v2:** multi-region color + tighter far-side filter (cleans
  opponent stats/confidence across S8/S9/S10).
- **Tier-C future stages** (ARCHITECTURE.md): pose-technique analysis (also a
  future S9 dimension + S10 focus area + S11 overlay), cross-video trend
  tracking, and the **presentation/UI dashboard** (deferred to post-4.5;
  `timeline.json` + `metrics.json` + `rating.json` are its data contract).
- **Render polish** (optional): HUD/banner overlap nit, overlay colors/sizes,
  GPU/parallel encode for full-clip speed.

**Footage (offline, David):** better source video for Stage 4.5 + more headroom
(higher mount 10–15 ft, 4K/60fps, faster shutter, simpler backgrounds, fewer
adjacent courts). This is the single highest-leverage next input.

## Things to NOT touch between sessions
- Stage 4 (`stages/track_ball/`) + Stage 4.5 (`stages/finetune_ball_model/`):
  paused/obsolete; don't modify or delete.
- v1/v2 weights on Drive: retained. Don't re-attempt ball-detection v1/v2.

## Bring this to the next session

Open a new Claude session and paste:

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, and the relevant stage contract.md
    before proposing anything.

    The full 13-stage pipeline is implemented + committed (Stage 11 render
    done) and runs end-to-end on the synthetic-ball placeholder. Every
    ball-derived output is a scaffold until real ball detection (Stage 4/4.5).
    I'd like to work on [ball detection v4 / better-footage intake / Stage 2.5
    v2 / Stages 3+6 role re-wire / a Tier-C stage / render polish] — see
    What's queued.

---

Generated at session end on May 29, 2026.

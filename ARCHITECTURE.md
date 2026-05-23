# Architecture

## Implementation status (2026-05-22)

This is a design doc; per-stage status lives here as a quick pointer (full
detail in `docs/SESSION_HANDOFF.md` and `KNOWN_ISSUES.md`).

- **Stages 1–3** (calibrate, track players, pose): implemented, smoke-tested.
- **Stage 4** (track ball, TrackNetV2): code-complete, but its detector's
  weights don't generalize to amateur footage, so it currently produces
  unusable output. The code is not broken in itself — depending on the
  eventual v4 ball-detection approach it will either be re-pointed at new
  weights (stays as-is) or rewritten (e.g. around classical CV). Undecided.
- **Stage 4.5** (ball-detection calibration): **PAUSED** after three failed
  approaches; awaiting better source video. See `KNOWN_ISSUES.md`.
- **Stage 5** (detect shots): **implemented + smoke-tested**. Because real ball
  detection is paused, Stage 5 currently runs against a **synthetic placeholder
  `ball.parquet`** produced by `tools/synth_ball.py` (impacts placed at real
  player positions, flagged `synthetic: true`). Downstream stages must treat
  ball data as placeholder until a real ball detector (v4) exists.
- **Stages 6–11**: not started.

Implemented stages live in **importable** folders (`stages/calibrate`,
`stages/track_players`, `stages/pose`, `stages/track_ball`,
`stages/detect_shots`) — Python modules can't start with a digit, so the
numbered folders in the pipeline diagram below are illustrative, not import
paths.

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
[3] Pose (uses players bbox crops) ──► poses.parquet
    │
    ▼
[4] Track ball (uses court.json) ──► ball.parquet
    │
    ▼
[5] Detect shots (players + pose + ball) ──► shots.json
    │
    ▼
[6] Classify shots (shots + pose + ball trajectory) ──► classified.json
    │
    ▼
[7] Segment rallies (classified + ball) ──► rallies.json
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
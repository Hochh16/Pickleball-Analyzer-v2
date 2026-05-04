# Architecture

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

3. **Schemas are versioned.** Every output file has a `schema_version` field. Breaking schema changes increment it.

4. **No stage modifies its inputs.** Output files are always new files.

5. **Each stage has a `contract.md`** in its folder. The contract is the source of truth — it specifies inputs, outputs, schema, edge cases. The contract is reviewed and approved before code is written.

6. **Each stage has a smoke test.** A 30-second test clip with known-correct expected outputs. The stage is "done" only when the smoke test passes.

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
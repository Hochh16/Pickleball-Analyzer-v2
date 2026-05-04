# Pickleball Analyzer v2

Analyze amateur pickleball match video to produce skill metrics, USAPA rating estimates, and improvement plans.

## Architectural principle

A linear pipeline of independent stages. Each stage:
- Reads files from disk (the previous stage's output)
- Writes files to disk (its own output)
- Knows nothing about how its inputs were produced
- Knows nothing about who will consume its outputs

Each stage can be run, tested, replaced, or rewritten independently. New analyses are added by reading existing stage outputs — never by modifying upstream code.

See `ARCHITECTURE.md` for the full pipeline.

## Stages

| # | Stage | Input | Output |
|---|---|---|---|
| 1 | Calibrate | video.mp4 + 10 user clicks | court.json + court_zones.json |
| 2 | Track players | video.mp4 + court.json | players.parquet |
| 3 | Pose | video.mp4 + players.parquet | poses.parquet |
| 4 | Track ball | video.mp4 + court.json | ball.parquet |
| 5 | Detect shots | players + ball + poses | shots.json |
| 6 | Classify shots | shots + poses + ball | classified.json |
| 7 | Segment rallies | classified + ball | rallies.json |
| 8 | Compute metrics | all of the above | metrics.json |
| 9 | Rate (USAPA) | metrics.json | rating.json |
| 10 | Plan improvement | rating.json | improvement_plan.json |
| 11 | Render annotated video | video.mp4 + all JSON | annotated.mp4 + timeline.json |

## Per-video data layout

Each analyzed video gets its own folder under `data/`:

~~~
data/match_001/
├── video.mp4
├── court.json
├── court_zones.json
├── players.parquet
├── ball.parquet
├── poses.parquet
├── shots.json
├── classified.json
├── rallies.json
├── metrics.json
├── rating.json
├── improvement_plan.json
├── annotated.mp4
└── timeline.json
~~~

Sidecar files only. No database in v1.

## Status

Stage 1 in progress.
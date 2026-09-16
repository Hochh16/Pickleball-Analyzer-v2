# Pickleball Analyzer v2

Analyze amateur pickleball match video to produce skill metrics, USAPA rating estimates, and
improvement plans, from an ordinary camera on an ordinary tripod.

## Architectural principle

A linear pipeline of independent stages. Each stage:
- Reads files from disk (the previous stage's output)
- Writes files to disk (its own output)
- Knows nothing about how its inputs were produced
- Knows nothing about who will consume its outputs

Each stage can be run, tested, replaced, or rewritten independently. New analyses are added by
reading existing stage outputs — never by modifying upstream code.

See `ARCHITECTURE.md` for the pipeline and `SYSTEM_DESIGN.md` for the authoritative design and
accuracy position.

## Stages

Each is a folder under `stages/`, run as `python -m stages.<name>.<name> data/<clip>`.

| Stage | Input | Output |
|---|---|---|
| `calibrate` | video + the operator's court clicks | court.json, court_zones.json |
| `track_players` | video + court.json | players.parquet |
| `classify_tracks` | players.parquet | track_roles.json (user / partner / opp_a / opp_b) |
| `pose` | video + players.parquet | poses.parquet |
| `track_ball` | video + court.json | ball.parquet |
| `ball_trajectory` | ball_3d.parquet + shots | trajectory.json (ground-anchored ball speed) |
| `detect_shots` | players + ball + poses | shots.json (+ shot_discards.json, a debug trace) |
| `detect_bounces` | ball + court | bounces.json |
| `classify_shots` | shots + bounces + poses + ball | classified.json |
| `segment_rallies` | classified + bounces + rally ends | rallies.json |
| `compute_metrics` | all of the above | metrics.json |
| `rate` | metrics.json | rating.json |
| `plan_improvement` | rating.json | improvement_plan.json |
| `render` | video + all JSON | annotated.mp4, timeline.json |
| `aggregate` | N analysed videos | union.json + merged inputs — one virtual session, so the stages above run on it unchanged |

`finetune_ball_model` trains the ball detector and is not part of a normal run.
`ball_3d.parquet` (the 3-D ball, read by `detect_bounces`, `ball_trajectory` and
`classify_shots`) is built by `python -m tools.build_ball_3d`, which is ~98% of local
post-processing time.

## Per-video data layout

Each analyzed video gets its own folder under `data/` holding the stage outputs above, plus
`_labeling/` (the annotated review video and the operator's review sheet) and `_gemini/` (saved
video-model responses). `data/` is not tracked by git.

Sidecar files only. No database in v1.

## The operator's reviews are the acceptance test

Accuracy claims are scored against what the operator says happened in his own videos, collated in
`docs/truth/<SOURCE VIDEO>.json`. See `docs/TRUTH_STORE.md` for how a review is produced and
imported, and what the store holds. Four videos are reviewed shot by shot; one of them (court A) is
HELD OUT and must never be tuned on.

    python -m tools.annotate_full data/<clip>        # the whole video, numbered and labelled
    python -m tools.shot_review_sheet data/<clip>    # the workbook he corrects
    python -m tools.truth_store --import-review data/<clip>
    python -m tools.truth_store --report

## Checking a change

    python -m pytest -q                              # the suite
    python tools/regression.py                       # scores four clips against docs/REGRESSION_BASELINE.json
    python tools/regression.py --rerun               # re-runs stages 5-11 first; exits 1 if numbers moved
    python tools/regression.py --save                # accept the new numbers as the baseline

Never `--rerun` a clip whose review video is still being used: re-detection renumbers its shots.

## Status (2026-09-15)

The pipeline runs end to end on real video and produces a rating, a plan and a rendered timeline.
Measured against the operator's reviews of four videos: 74% of real shots detected, 72% of those
typed correctly, 65% of serves found end to end. `SYSTEM_DESIGN.md` holds the current accuracy
block and what has been tried against that plateau; `docs/ACCURACY_LEDGER.md` is the running
record; `docs/SESSION_HANDOFF.md` says where work stands.

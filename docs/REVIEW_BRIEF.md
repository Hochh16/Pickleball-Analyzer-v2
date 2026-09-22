# Brief for an outside code review

Updated 2026-09-22. Paste this in before asking another model to review the repo. It exists
because a review arrived built on numbers this project had already retracted — the code is ahead
of the older sections of `SYSTEM_DESIGN.md` and of the un-marked parts of `KNOWN_ISSUES.md`.

Authoritative, in this order: this brief → `docs/ACCURACY_LEDGER.md` (running record, newest
entries last) → the code. Anything else may be stale.

## What the app is

A Windows desktop app that turns ordinary pickleball video into a USA Pickleball-aligned skill
report for ONE player. A linear pipeline of independent stages under `stages/`, each reading the
previous stage's files from disk: calibrate → track_players → classify_tracks → pose → track_ball
→ ball_trajectory → detect_shots → detect_bounces → classify_shots → segment_rallies →
compute_metrics → rate → plan_improvement → render, plus `aggregate` for many videos.

Two constraints that rule out whole classes of advice:
- **Ordinary camera.** A phone on a normal tripod behind one baseline, low and off-centre, often
  with adjacent courts in frame. "Mount the camera higher / centred / add a second camera" is not
  available.
- **No user correction pass.** The analysis must be automatic. Asking the end user to review or
  fix shots defeats the product.

## Ground truth and how accuracy is measured

The operator reviewed four videos shot by shot; his answers are collated per SOURCE VIDEO in
`docs/truth/*.json` (see `docs/TRUTH_STORE.md`). Totals: 340 real shots, 82 of which we never
detected, 133 junk detections, 56 rally ends. One video (`pb_5_min_indoor_1_court_a`) is HELD
OUT — nothing is ever tuned on it. `tools/regression.py` scores four clips against
`docs/REGRESSION_BASELINE.json`.

Every number below is one-to-one against that truth, not against another model's output.

## Current measured accuracy

| | measured |
|---|---|
| shots detected | 252 / 340 (74%), plus 111 junk |
| shot type, of those detected | 181 / 252 (72%); drop is worst at 14/43 |
| volley vs not | 159 / 194 (82%) |
| serves, end to end | 34 / 52 (65%), plus 13 false serves |
| rally end within 2 s | 12/19 on the held-out video |
| end reason (net / out / not-returned) | 14 / 23 |
| side of the net, of detected shots | 178 / 180 |
| ball position when the ball IS visible | recall 94-96%, median error ~4 px |

## What the code already does (do not propose these)

- **calibrate** already computes and stores a 3×3 homography (`court.json` → `homography.image_to_court`)
  plus court polygons and pixels-per-foot at each baseline. Stages convert pixels to feet with it.
- **track_ball (v4)** is a TrackNet heatmap model, not a box detector, so box trackers
  (ByteTrack/SORT) do not apply. It already keeps the top-k peaks per frame INCLUDING
  sub-threshold ones, then chooses one per frame by a Viterbi-style search over the whole clip:
  confidence minus a motion penalty, a hard per-frame step limit, gap linking, and a restart cost
  so it will not hop to a neighbouring court's ball. Post-processing drops detections that are an
  impossible jump from both neighbours and interpolates short gaps.
- **Kinematic filtering exists in three places**: the step limit inside track selection, the
  velocity-outlier drop in post-processing, and a "wrong object latch" gate in detect_shots.
- **detect_shots** records EVERY pre-filter candidate to `shot_discards.json` for measurement.
- **build_ball_3d** is ~5 s per clip (it stopped re-decoding video when Stage 4 began measuring
  the ball blob). It is no longer a throughput problem.
- **rate** confidence-weights every category, and `tools/rating_leverage.py` measures what each
  driver metric is worth by nudging it and re-running the real scorers.

## The actual failure, stated precisely

Not recall. **95 of the held-out video's 97 real shots are already present in the Stage 5
candidate stream before any filter runs.** Two failures dominate:

1. **Choosing among nearby candidates.** A real contact and a spurious kink 0.3-1.0 s later look
   alike; the filters often keep the wrong one. On the held-out video, 12 of 25 recoverable
   missed shots die in the same-side-run ("you can't hit twice in a row") filter.
2. **False positives when no ball is there.** Where the operator labelled the ball NOT visible,
   the detector still claims one in 49% of frames outdoors, 35% and 25% indoors.

A third, smaller one: a far player's hit sometimes gets attached to a near player who is in front
of them in the image, which then looks like the near side hitting twice.

## Tried and measured against video it was NOT tuned on — all failed

| attempt | result |
|---|---|
| more hand-built rejection filters | plateaued; do not generalise to an unseen court |
| learned per-candidate classifier (24 features, gradient boosting) | 64/97 found with 26 junk vs the rules' 72/36 |
| a third fully reviewed video added to training | no improvement on the held-out video |
| paddle-sound onset, mono | separates real from junk (AUC 0.78) but adds no net gain |
| stereo channel difference to reject adjacent courts | does not separate (AUC 0.50-0.62) |
| sound choosing which nearby contact is the hit | 180 found / 75 junk, identical to today |
| Gemini 3.8 Flash and Pro on the video directly, 720p and 4K, whole-video / 20 s windows / per rally | all below the pipeline; best was Pro at 96 found / 155 junk against the pipeline's 101 / 42 |
| lower ball-detector threshold | rejected: recall is already 94-96%; it adds false positives, the actual failure |

Also rejected with numbers, in `docs/ACCURACY_LEDGER.md` and `KNOWN_ISSUES.md`: net-crossing
splits, dead-ball splits, resting-side rules, serve rotation, 1080p escalation, negative-mining
retrain, dead-ball suppression from ball height, predicting where a volleyed ball would land.

## Where a review could actually help

1. Separating two contacts 0.3-1.0 s apart when only one is a paddle strike, from a low camera —
   any signal we are not using.
2. Suppressing detections in frames with no ball, without costing the 94-96% recall.
3. Attributing a contact to the correct player when a far player is occluded by a near one.
4. Shot TYPE from this data (72%, drop 14/43) — least explored of the open problems.
5. Anything in the list above where the measurement, not the idea, was the weak part.

Ground any suggestion in what the truth store can score, and say which measured number it should
move.

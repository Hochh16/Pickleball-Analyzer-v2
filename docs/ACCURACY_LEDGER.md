# Accuracy ledger — real-clip validation

Foundations-first accuracy tracking: validate each stage by RENDERING its output
against reality, not by smoke tests. Confidence ≠ correctness. Started 2026-07-18
on `pb_5min_test_20s-7`.

## Reference clip: `pb_5min_test_20s-7`

A **20 s drill** (ball cart present, players feeding from deep — NOT a normal
match, so positioning/rally structure aren't representative; good for finding
detector bugs, not for validating the final rating). 4K @ 60 fps, 1200 frames.

**Operator ground truth (David watched it):** **11 paddle strikes** — 1 before the
rally; within the rally 3 dinks + a 4th dink that netted, and 1 drop from the
transition zone into the kitchen (the rest are the far-side returns).

## Per-stage verdicts

| Stage | Verdict | Evidence / notes |
|---|---|---|
| 1 Court calibration | 🟡 roughly OK | Far kitchen reads correct (players' median court-y ≈ 30 ft = far kitchen). Near players read *behind* the baseline (y≈−2 to −3 — drill positioning or small offset). Far-side foot-y noisy (up to 57 ft — known homography sensitivity; roles, not geometry, drive far scope). Worth an operator eyeball. |
| 2 Player tracking | ✅ good | Correct 4 players by role; background/adjacent-court excluded; user = left-near. |
| 2.5 Roles | ✅ good | Sensible, byte-identical to reference; single-pass decode fix kept output identical. |
| 3 Pose | ✅ good | YOLO-pose 100% detect, 5–9 px median drift vs MediaPipe, skeletons track tightly. |
| 4 Ball | 🟡 ok / jittery | 87% visible, 37 gaps (mostly 2–6 frames), median jerk 3 px (p90 9.5), a few >800 px teleport outliers. Decent; the jitter only bit shot detection. |
| **5 Shots** | ✅ **FIXED 2026-07-19** | Was 2/11 (~18% recall). Root cause: the adjacent-court teleport-in gate rejected real shots (ball occluded at the paddle strike → reappears "teleported"). Fix: gate rejects a teleport only if the run is a short BLIP. Now **13 shots, hitter side alternates near/far all rally**, recall ~100% (13 vs 11, +1 pre-rally, ~2 extra). |
| **6 Shot type** | 🔴 **NEXT** | With shots now correct, types are wrong: labeled drives 5 / drops 4 / dinks 2 / overhead 1 / lob 1 for a clip that was **mostly dinks + 1 drop**. Over-calls drives/drops, under-calls dinks; a dink/drop drill shouldn't yield drives+overhead+lob. Stroke side is fine (5 user shots → 4 BH/1 FH; opponents "unknown" by design). |
| 5.5 Bounces | 🔴 wrong kind | All 6 detected are `at_feet` events near the baselines (5/6 "OUT"), not true mid-court ground bounces. Detecting ball-near-feet, missing real bounces. Revisit with the ball-trajectory work. |
| 7 Rally / 8 Metrics | 🟡 improving | 1 rally of 13 (was 1 of 2). Position/heatmaps from tracking are plausible; shot-derived metrics now rest on a correct shot layer. |
| 9 Rating | 🟡 improving | 3.2 → 3.23, confidence 0.223 → 0.267 after the shot fix. Still rests on imperfect shot-type + bounces. |

## Fix priority (remaining)

1. **Stage 6 shot-type** — dinks mislabeled as drives/drops. Biggest remaining
   accuracy gap; drives the dink/third-shot/volley categories of the rating.
2. **Stage 5.5 bounces** — detecting ball-at-feet, not ground bounces.
3. **Court calibration** — confirm the near-baseline offset with the operator.
4. **Validate on a real MATCH clip** — this is a drill; a real doubles match would
   test positioning/rally/shot-mix representatively.

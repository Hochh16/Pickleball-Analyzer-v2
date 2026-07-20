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
| 1 Court calibration | ✅ good (corrected) | Homography RMSE ~0; 4 corners map exactly to the court rectangle; kitchen lines project to y=15.5 / 28.5 (≈15/29 ✓). Earlier "near side off" was WRONG — the near players read behind the baseline early because they genuinely **feed from deep then move up** (drill; late dinks read y≈13 = kitchen edge). Possible *minor* ~2 ft near-side foot-projection under-read makes some kitchen dinks borderline (zone needs y≥13), but not a calibration bug. |
| 2 Player tracking | ✅ good | Correct 4 players by role; background/adjacent-court excluded; user = left-near. |
| 2.5 Roles | ✅ good | Sensible, byte-identical to reference; single-pass decode fix kept output identical. |
| 3 Pose | ✅ good | YOLO-pose 100% detect, 5–9 px median drift vs MediaPipe, skeletons track tightly. |
| 4 Ball | 🟡 ok / jittery | 87% visible, 37 gaps (mostly 2–6 frames), median jerk 3 px (p90 9.5), a few >800 px teleport outliers. Decent; the jitter only bit shot detection. |
| **5 Shots** | ✅ **FIXED 2026-07-19** | Was 2/11 (~18% recall). Root cause: the adjacent-court teleport-in gate rejected real shots (ball occluded at the paddle strike → reappears "teleported"). Fix: gate rejects a teleport only if the run is a short BLIP. Now **13 shots, hitter side alternates near/far all rally**, recall ~100% (13 vs 11, +1 pre-rally, ~2 extra). |
| **6 Shot type** | 🟡 **improved 2026-07-19** | **6/10 rally shots correct** (return✓, drive✓✓, drop✓, 2/5 dinks✓). Fixes: overhead→stroke axis, lob→receiver-at-kitchen, volley rules, dink/drop by distance-from-net + a slow-ball speed guard. Remaining 3 dink misses + serve are UPSTREAM: airborne-ball **speed inflation** (Stage 4 geometry) reads dinks as 20–27 ft/s, and **serve detection** never fires (Stage 5). Stroke side fine (4 BH/1 FH). |
| 5.5 Bounces | ✅ **FIXED 2026-07-19** | Was 5 (~55% recall), missing soft near-kitchen dink bounces. Root cause: candidates from the generic impulse signal (fired at arc apexes + jitter). Fix: detect candidates as **pixel_y descent-peaks** (an apex is a pixel_y *minimum*, so apexes are ignored) + y-flip re-check on the smoothed trajectory. Now **11 bounces** matching the operator landing map; **9/13 shots get a `landing_y`** (was 0), restoring shot-type's primary signal. |
| 7 Rally / 8 Metrics | 🟡 improving | 1 rally of 13 (was 1 of 2). Position/heatmaps from tracking are plausible; shot-derived metrics now rest on a correct shot layer. |
| 9 Rating | 🟡 improving | 3.2 → 3.23, confidence 0.223 → 0.267 after the shot fix. Still rests on imperfect shot-type + bounces. |

## Stage 5 shots — FIX RECORD (do not regress) — commit 734afe1, 2026-07-19

**Symptom:** recall 2/11 (~18%). **Root cause:** the adjacent-court "teleport-in
contamination gate" rejected REAL shots — the ball is occluded at the paddle strike
and reappears a few frames later, which looks like a teleport-in. **Fix:** the gate
rejects a teleport only when the reappearance run is a short BLIP (< `min_serve_run`
frames); a sustained run is a real shot, kept. In `detect_shots.py`:
```python
if contam_filter and teleport_in_pxpf(f) > teleport_thresh:
    a_run, z_run = run_bounds(f)
    if (z_run - a_run + 1) < min_serve_run:
        n_rejected_teleport += 1
        continue
```
Thresholds (×`res_scale`=2.0 @4K): MIN_TURN_RATE_DEG=45, MIN_DIRECTION_CHANGE_DEG=45,
ASSOC_MAX_PX=120, TELEPORT_IN_PX_PER_FRAME=40. **Result:** 13 shots, hitter side
alternates near/far all rally, ~100% recall (13 vs operator 11 = +1 pre-rally feed,
~1 extra). **Still open (upstream):** serve never fires (shot 2) — dead-time+launch
detector at clip start.

## Stage 5.5 bounces — FIX RECORD (do not regress) — commit 67c6ecf, 2026-07-19

Recall 5→11. Candidates are now pixel_y **descent-peaks** (an arc apex is a pixel_y
*minimum* so apexes are correctly ignored) with `BOUNCE_PROMINENCE_PX=9.0*res_scale`;
y-flip re-check runs on the smoothed trajectory with `yflip_floor=0.3*res_scale`
(the old 4px floor rejected soft dink rebounds ~0.75px/f). Restored `landing_y` on
9/13 shots (was 0), which is shot-type's primary signal.

## Fix priority (remaining) — foundations first

1. ~~Stage 5.5 bounces~~ ✅ DONE (pixel_y descent-peak detection; 5→11).
2. **Stage 6 shot-type** ← NEXT. Now has landings; fix the type LOGIC — design
   notes below (dink/drop zone dependency, lob receiver-position, overhead-as-
   stroke, serve detection, volley rules). Ground truth still 7 drives / 2 dinks
   vs 5 dinks. (Possible minor near-side foot-projection under-read to check —
   NOT calibration — flips borderline kitchen dinks to transition→drop.)
3. **Validate on a real MATCH clip** — this is a drill; a real doubles match would
   test positioning/rally/shot-mix representatively.

## Stage 5.5 bounces — ground truth (operator, 20 s clip)

**≈ 9–10 real ground bounces** (volleys don't bounce):
- **A. Out-of-rally, near side, behind the baseline (feeds):** ~2–3.
- **B. Opponent hit → landed on the NEAR side (in-court):** 4 — 2 dinks in the near
  **kitchen**, 1 return-serve in near **transition**, 1 drive in near transition.
- **C. You/partner hit → landed on the FAR side:** 3 — 1 drop far **kitchen**,
  1 serve far **transition**, 1 dink just outside the far kitchen (~within 2 ft).
- **Not bounces (volleys, no landing):** opponent air-hit your dink; opponent
  air-hit your drive at the kitchen line.
- **Ambiguous:** 1 attempted dink that hit the net.

**Detected 5 of ~9–10 (~55% recall):** f72/f307 = the feed bounces (A ✓),
f794 = a near-transition (B ✓), f856 = far kitchen (C ✓), f730 = a far one.
**Systematically MISSING the soft near-KITCHEN dink bounces (B) + some transition
bounces.** Likely: soft kitchen bounce = small far-ish ball + weak vertical
rebound (low y-flip) + the same 234-candidate single-frame noise. Fix like shots:
cleaner (windowed) candidates + a ground-landing test that tolerates soft rebounds.

## Stage 6 shot-type — per-shot ground truth + dink finding (2026-07-19)

Operator per-shot truth (aligned to detected shots by side-alternation + ~1 s
offset; my shots 0–1 = the 2 pre-rally feeds): 2=serve, 3=return, 4/5=drive,
**6=drop**, **7/8/9/10/11=dink** (11 netted), 12=post-net.
After the overhead/lob/volley fixes: 4/5 drive ✓, **6 drop ✓**, serve✗(not
detected→drive), dinks only 9 ✓ (a volley) — 7/8/10/11 → drive.

**KEY FINDING (verified, NOT a bug):** the near players dink from ~2–7 ft BEHIND
the kitchen line (their feet project to court_y ≈ 8–13 = transition; the homography
is correct — kitchen line projects to 15.5). So requiring the hitter *at* the
kitchen (`zone=="kitchen"`, y≥13) for a dink is too strict — real dinks come from a
step back. **DEFINITIONAL DECISION NEEDED (operator):** should a soft shot from the
near transition (a step behind the kitchen line) be a **dink** (operator labeled
7–11 as dinks) or a **drop**? That decides the dink/drop split (likely: dink =
soft + hitter in kitchen OR near-transition + part of a net exchange; drop = soft +
hitter deep/baseline, e.g. the third-shot drop = shot 6). Also: depth-corrupted
speed (no-landing shots read fast→drive) and the near-side landing under-read
(a far dink landing near-kitchen reads deep) still hurt 7/8/11.

**FIX APPLIED 2026-07-19 (operator chose "distance from net"):** dink = soft/slow +
hitter at kitchen OR transition (a step behind the line still dinks); drop = soft +
hitter at baseline (third-shot drop). Plus a **speed guard**: a slow ball
(post ≤ DINK_MAX) near the net is a dink even if its landing read a bit deep — a
drive requires real pace. Result on the clip: **6/10 rally shots correct** (was
~2 before Stage-6 work, 5 after overhead/lob/volley): serve✗, return✓, drive✓✓,
drop✓, dinks 9✓ 10✓, 7/8/11✗.

**FRONT-FOOT rule (operator, 2026-07-19):** a dink is called by the **front foot**
(the ankle nearest the net) being within ~2 ft of the kitchen line — NOT the rear
foot. The bbox-bottom foot point is, on the NEAR side, the REAR foot (nearer the
camera) and reads several feet too deep, mis-reading a kitchen dink as
transition/drop. Fix (`front_foot_court_y`, `classify_shots.py`): project both pose
ankles to court_y and take whichever of {bbox foot, ankle projections} is CLOSEST to
the net (seeded with the bbox foot so it can never read DEEPER — protects the FAR
side, where the bbox-bottom is already the front foot and a noisy far ankle would
otherwise push it deeper; that regression cost shot 9 before the seed was added).
Near dinks now read front foot ≈ 13–16 ft (kitchen) vs rear 10–13.

**Remaining Stage-6 errors are UPSTREAM, not Stage-6 logic:**
- **Airborne-ball speed inflation** (shots 7/8/11): a dink reads post ≈ 20–27 ft/s
  because the ground-homography projects the ball while it's mid-air, inflating its
  court-speed; can't loosen the drive threshold without flipping real drives (4/5)
  to dinks. **Fix at Stage 4/geometry** (estimate ball height / use apex-relative
  speed), not here.
- **Serve detection** (shot 2 → drive): serve never fires — upstream in Stage 5
  (dead-time gap + launch at clip start).
These two are the next foundations for shot accuracy.

## Stage 6 shot-type — design notes

Ground truth for the 20 s clip (operator): **1 serve, 1 return, 2 drives (1 hard,
1 soft), 1 drop, 5 dinks (4 + 1 netted).** Pipeline gave drives 5 / drops 4 /
dinks 2 / overhead 1 / lob 1, **0 serves**. All 13 shots had `landing_y = None`
(bounces broken) → classifier ran entirely in its low-confidence speed/arc
fallback. Fix the inputs first; then:

**Known Stage-6 logic bugs (operator-confirmed):**
- **Lob** (`classify_type` line ~360) must require the **receiver at the kitchen**
  — a lob is a soft ball lofted *over a player's head while they're at the net*.
  Currently a soft high shot to baseline opponents is mislabeled a lob.
- **"Overhead" is a STROKE, not a shot type.** It belongs on the stroke axis with
  forehand/backhand (how the ball was struck — above the head), not in
  `shot_type`. An overhead is tactically usually a drive/put-away. Split the axes.
- **Serve** was not detected (0 vs 1) — check the serve detector (dead-time gap +
  launch) at the clip start.

**Volley classification (no bounce → no landing).** Operator rules — decide type
from **ball speed + receiver location + where it WOULD have landed**:
- slow ball taken out of the air **at the kitchen** → **dink**
- fast ball taken out of the air **at the kitchen** → **drive** (speed-up)
- ball taken out of the air from **transition/baseline** → **drive**
- if a player started at the baseline, the ball went **over their head**, and they
  ran back to hit it out of the air from deep → the PRIOR shot was a **lob**.

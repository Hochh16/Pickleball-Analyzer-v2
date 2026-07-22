# Accuracy ledger — real-clip validation

## ACCEPTANCE TEST — operator ground truth (`pb_5_minute_outdoor-2`, 2026-07-22)

**These counts are the acceptance test. Validate every change against THEM, never
against the previous run.** (Lesson learned the hard way: per-shot type accuracy was
tuned on one rally for days while the whole-clip COUNTS — what the operator actually
sees in the report — were never checked. A single operator count exposed a 25%
adjacent-court contamination bug in minutes.)

| item | operator truth |
|---|---|
| Shots | **98** |
| Serves = rallies = returns | **13** |
| Dinks | **18** |
| Volleys | **17** |
| Bounces | **81** |

**Identity that must hold: `shots = volleys + bounces`** (98 = 17 + 81). Every shot is
either volleyed out of the air or lands exactly once. This is the single best
self-check in the system — enforce it in code, not just in review.

### Scorecard (2026-07-22, after the ground-truth-driven fixes)

| item | truth | session start | now | error |
|---|---|---|---|---|
| Shots | 98 | 155 | **108** | +10% |
| Bounces | 81 | 146 | **75** | −7% |
| Serves | 13 | 4 | **11** | −15% |
| Volleys | 17 | 51 | **27** | +59% |
| Dinks | 18 | 8 | **35** | +94% |
| Rallies | 13 | 19 | 18 | +38% |
| Forehand+backhand | — | 26 shots | **76 shots** | — |

### CAPABILITY ASSESSMENT — what this camera can and cannot deliver

Camera: ONE camera, ~6 ft, corner mount. Operator will not change it now (a side
mount was tried before and caused other problems; a higher mount is possible later).

**ACHIEVABLE (errors here are bugs, not physics — proven 2026-07-22):**
- shot counts, serve counts, rally counts, return counts
- dink/drive/drop VOLUME and the third-shot drop-vs-drive CHOICE
- forehand / backhand per player (roster already carries all four players' handedness)
- court positioning, kitchen-line time, team movement, court coverage
- volley COUNT — not by seeing height, but **derived from the identity**
  (`volleys = shots − bounces`), which sidesteps the height problem entirely

**BLOCKED by the single low camera (needs ball HEIGHT; three independent height-free
methods tested and defeated — see "TESTED ON CLEAN DATA" below):**
- dink QUALITY (pop-up height, height/depth control)
- true shot speed
- direct bounce-vs-volley discrimination at a player's feet
- spin (not feasible at this resolution)

**Product consequence:** the report can honestly cover the USAPA rubric on VOLUME,
CHOICE and POSITIONING, but not STROKE QUALITY. **Operator directive (2026-07-22):
keep every USA-Pickleball-rated item listed in the "what's behind each category"
chart — do not delete rows — and let the filled/unfilled circles tell the truth about
what is measured, partial, or not yet available.**

### Bugs found from the operator counts (all fixed 2026-07-22)

1. **Adjacent-court contamination.** `detect_shots` loaded `track_roles.json` but used
   it ONLY to set `is_user`; it never excluded `role='noise'`. On a multi-court venue a
   ball wobble near someone on the NEXT court became a shot — **38 of 155 shots (25%)**.
   `detect_bounces` had the same hole in its at-feet test. Both now restrict
   association to the four participants.
2. **Serves detected then discarded.** When the serve-appearance test fired at a frame
   already captured as an impulse shot, the code `continue`d — leaving `is_serve=False`.
   11 of 18 rallies had NO serve. Now it PROMOTES that shot to a serve.
3. **Handedness thrown away.** `roster.json` carries handedness for all four players
   and `stroke_side()` derives facing from the pose, but the code passed handedness
   only for the user, leaving 74 of 108 shots "unknown".
4. **No physical bounce constraint.** Intervals held up to 12 bounces. Now capped at
   ONE landing per shot, per the identity above.

### Remaining work (priority order, validate each against the counts above)

1. **Dinks 35 vs 18** — over-calling; recalibrate thresholds against the operator's 18.
2. **Volleys 27 vs 17** — should fall out once bounces are exact (identity).
3. **Rallies 18 vs 13** — 11 serve-starts + 7 deep-shot restarts. NOTE: sweeping the
   stall / gap / dead-ball thresholds does NOT move the count, so the current theory is
   wrong — trace the 7 actual restart points rather than tuning blind.
4. **Then** rebuild the report.


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
| **6 Shot type** | 🟡 **improved 2026-07-20** | Drill **7/10**, match rally 10 **7/12**. Fixes: overhead→stroke axis, lob→receiver-at-kitchen, volley rules, dink/drop by distance-from-net, front-foot zone, slow-ball guard, + **Stage 5.7 ground-anchored horizontal speed wired in** (physical, replaces 261/117 ft/s garbage) with a volley/phantom-bounce consistency guard. Residual errors = camera-limited volley cases (phantom bounces) + upstream serve/return region + serve detection — NOT the speed. |
| **5.7 Ball trajectory** | 🟡 **NEW 2026-07-20** | Ground-anchored horizontal ball speed (Phase 1, 8/8 tests). Physical on clean-bounce shots; match coverage limited by bounce quality. Phase 2 (height) shelved — monocular precision floor can't resolve bounce-vs-volley (z=0 vs z≈1.5 ft). See stages/ball_trajectory/contract.md. |
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

## Match-clip validation — `pb_5_minute_outdoor-2` rally 10 (2026-07-19)

First validation on a REAL doubles match (11 rallies, 10 serves; not a drill).
Operator gave per-shot ground truth for rally 10 (12 shots, a full point: serve →
baseline drives → kitchen dink exchange). Rendered annotated video
(`tools/render_rally.py`, `_rally_10_check.mp4`). **Score: types 7/12, sides 9/12,
volleys 2/8.**

**Errors all trace to ONE root — unreliable ball trajectory/bounce/height:**
1. **Soft-shot → drive (all 5 type errors):** #2 drop, #3/#5/#6 dink, #11 reset all
   mis-typed; driven by airborne-ball **speed inflation**, which on match data
   produces GARBAGE values (#5 post = 261 ft/s, #1 = 117 ft/s — physically
   impossible). Confirms the Stage-4 speed finding below, and worse than the drill.
2. **Volley detection BROKEN on match play (2/8).** Barely mattered in the drill (4
   volleys); a real kitchen exchange has many (operator: 5/12 shots were volleys).
   Pipeline MISSES real volleys #4/#5/#6/#10 (phantom bounce → "not volley") and
   FALSE-flags #2/#3 (missed a real bounce → "volley"). Volley = "did it bounce
   since the last shot," so these are **bounce-detection errors**.
3. **Sides** perfect #4–#11 (settled dink rally) but scrambled #1–#3
   (serve/return/third-shot), where ball speeds are garbage (unreliable track).

**Re-prioritisation:** the deferred **3-D projectile-trajectory fit** now addresses
the THREE biggest error sources at once — soft-vs-drive, volley (bounce) detection,
AND garbage speeds. Match data justifies building it next. What's already SOLID:
serve detection, settled-rally sides, and dinks that bounce & land in the kitchen
(#4/#7/#8/#9 all correct).

## Landing-depth investigation + OPERATOR DEFINITION (2026-07-20)

**OPERATOR DECISION: shot type is decided by WHERE THE BALL LANDED, not by how it
was struck.** A softly-hit ball that lands well past the kitchen line is NOT a dink —
it's "a dink that got away", typed by outcome. This confirms the existing
landing-first logic is the intended behaviour, and makes the LANDING POSITION the
authoritative signal (so its accuracy now matters most).

Findings (drill shot 7, the canonical "deep landing" case):
- **The bounce PROJECTION is correct** — the bounce pixel (py 1494) sits clearly past
  the kitchen-line pixel (py 1396) at that x. Not a projection bug.
- **That ball genuinely landed ~6 ft past the kitchen line** (far court → there in
  0.4 s = firm). Under the operator definition it is correctly NOT a dink, so drill
  shot 7 is **not an error** — drill effectively **8/10**.
- **Bounce positions are real, not interpolated:** bounces land on a genuinely
  visible frame 99–100% of the time (drill 0% interpolated, match 1%).
- **BUT ball occlusion around bounces is common on match play:** 24% of match bounces
  have a ≥3-frame occlusion within ±5 frames (match ball visibility 71.5% vs drill
  87.3%). Shot 7 showed 6 consecutive interpolated frames through the landing window
  (a perfectly linear +42.87 px/frame ramp).
- **Residual real gap: missed SOFT near-kitchen bounces.** Operator truth says 2 dinks
  landed in the near kitchen; the whole drill yielded only ONE near-kitchen bounce. A
  missed soft bounce leaves a shot with no landing, or lets it grab a later, deeper
  bounce → mis-typed.

**Planned fix:** do NOT globally lower the bounce prominence (that worsens the already
high match false-positive rate). Instead use the operator's volley idea (below) to
learn which shots were volleyed; every NON-volleyed shot MUST have a bounce, so search
harder for one only where a bounce is required. Targeted recall, no global precision cost.

## OPERATOR IDEA — positive volley detection by across-court REVERSAL (2026-07-20)

Detect a volley DIRECTLY from a direction change at a player with no bounce, instead
of inferring it from the ABSENCE of a detected bounce (fragile — bounce detection is
noisy). **Height-independent, which matters because height is the monocular precision
floor we hit.** The discriminator:
- **Bounce:** the ball's VERTICAL direction reverses (falling → rising) but it
  **continues across the court** in the same direction.
- **Volley / paddle contact:** the ball **REVERSES across the court** (heads back over
  the net the way it came).
A "bounce" candidate showing an across-court reversal is really a paddle contact →
kills the phantom bounces. Implementation caveat: an airborne ball's raw pixel
direction is confounded by its arc, so compare NET DISPLACEMENT over a short window
before vs after the event and test whether the across-court component flips sign.
Speed then comes from bounce→volley or volley→volley — exactly the Stage 5.7 anchor
model, so the two fixes compound.

**Refined design (ready to build, 2026-07-21).** A first prototype fired on junk, and
the reasons are now fixed or known — build it with these guards:
1. **Windowed, not per-frame.** Compare NET DISPLACEMENT over ~5 frames before vs
   after the event. Per-frame turn rate reads exactly 0.0 through interpolated
   stretches (measured), so a contact hidden by occlusion is invisible to it while
   the windowed measure still sees the reversal (112-177°).
2. **Require REAL detections on both sides** (not interpolated) and a minimum ball
   speed — the prototype's false positives were interpolated fill and slow drift
   (~3 px/frame), not contacts.
3. **Track jumps are no longer a source of false positives** — the Stage 4
   candidate+continuity fix (2026-07-21) eliminated them (teleports 20→0, max step
   1531→144 px/f). The prototype's 207/360 px "reversals" cannot occur now.
4. **Discriminator:** bounce = vertical reverses but the ball CONTINUES across the
   court; volley/paddle contact = the ball REVERSES across the court. Beware: for this
   camera (behind the near baseline) the across-court axis maps mostly to image Y,
   which the ball's ARC also moves — so judge direction over a horizon long enough
   that court travel dominates the arc, not frame-to-frame.
5. **Then:** every NON-volleyed shot MUST have a bounce → search harder for a missed
   bounce only where one is required (targeted recall, no global precision cost), and
   reject "bounces" that show an across-court reversal (they are paddle contacts).

**Fast test rig:** `data/pb_outdoor2_excerpt` (bundle already on Drive) = source
frames 16200-18861, excerpt f == source f+16200, rally 10 = excerpt f1684-2544, full
vision pass ~2 min. Operator per-shot truth for rally 10 is in this ledger.

### TESTED ON CLEAN DATA (2026-07-21) — do NOT retry these

Volley detection improved **2/8 → 5/10 from the Stage-4 ball fix alone**. Two further
signals were then tested directly against operator rally-10 volley truth, on the clean
track. **Both fail; the cause is the camera angle, not the implementation.**

1. **py direction-reversal (the reversal idea as a bounce test): CONFOUNDED.** For this
   camera (behind the near baseline) image-y mixes ball HEIGHT with COURT TRAVEL, so
   the signature is direction-dependent:
   - a real VOLLEY interval (shot 4→5) showed a strong interior py-max, prominence
     **+90 px** — looks exactly like a bounce (checked: nearest player 139 px away, so
     not a missed contact);
   - a real BOUNCE interval (shot 1→2) showed **−48 px** (no interior max) — because
     the ball was travelling AWAY from camera, and falling py from travel cancels the
     bounce's rise.
   Toward-camera travel amplifies noise into false bounces; away-camera travel erases
   real ones.
2. **Energy loss at the bounce (speed drop): NO SEPARATION.** Mid-interval speed ratio,
   volley intervals `[0.50, 0.53, 0.72, 1.17, 1.30]` vs bounced `[0.27, 0.71, 0.72,
   0.79, 2.24]` — **identical medians (0.72)**, fully overlapping.

Together with the earlier height-reconstruction result (precision-floored), that is
**three independent height-free attempts at bounce-vs-volley, all defeated by the same
monocular limit.** Treat volley as a ~50%-reliable SOFT signal, not a per-shot fact.
Revisit only if the camera angle changes (operator: possible higher mount in future).

## Stage 4 geometry / ball SPEED — investigation (2026-07-19)

**Goal:** fix the "airborne-ball speed inflation" that made dinks 7/8/11 read as
drives. **Finding: instantaneous ball speed has NO robust monocular fix.** The ball
is airborne, its height is unknown, and every candidate method was tested and fails:

| method | result |
|---|---|
| project ball px → court via ground homography, court-distance/time | **explodes** — airborne ball near the image horizon projects to court_y = 75–150 ft (off-court), shot 1 gave 1.2e7 ft/s. The ground plane is meaningless for a raised point. |
| ppf at the **ball's pixel row** (local optical scale) | **inflates** — an airborne ball sits high in the image where px/ft is small, so px/ppf blows up (dinks → 12–44 ft/s). |
| contact→landing **travel distance** (ground points) | contact point is ALSO airborne (paddle height) → same explosion (contact court_xy = 120k / 149 / 58 ft for several shots). Only the LANDING (a real ground bounce) projects reliably. |
| current: ppf at **hitter's ground court_y** | least-bad, but conflates: **shot 5 (real DRIVE) reads 18.6 ft/s while shot 8 (real DINK) reads 26.9** — the drive reads SLOWER than the dink. No threshold separates them. |

**Conclusion:** the only reliable court measurement for an airborne ball is its
**landing** (ground bounce) — already used. The remaining misses (8, 11) have NO
landing (volleyed away / netted) so they fall back to the unreliable speed; shot 7's
landing reads deep (a genuinely deepish far-side dink, or a near-side bounce
under-read). **Speed is a weak discriminator by physics, not by a fixable bug.**

**Real fix (a feature, not a patch):** fit the ball's 3-D **projectile trajectory**
(parabola under gravity, anchored by the detected ground bounces + apex) between
consecutive contacts to recover true launch speed AND height. That would also fix
the deep-landing reads. Significant effort; deferred pending operator direction.
**Short-term:** lean on landing + arc + rally-context; treat speed as low-weight.

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

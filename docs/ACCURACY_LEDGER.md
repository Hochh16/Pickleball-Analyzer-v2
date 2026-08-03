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
| Serves = rallies = returns | **14** |
| Dinks | **18** |
| Volleys | **17** |
| Bounces | **81** |

> **CORRECTED 2026-08-01 — serves/rallies are 14, not 13.** The operator re-checked
> all 14 listed serve timestamps against the rendered frames and confirmed **every one
> is a real serve**, so 14 serves = 14 rallies. Two timestamps also shift: the serve
> listed at **1:29 is really at 1:33**, and **4:58 is really at 5:01** (in both cases
> the pipeline detects the player's pre-serve ball-handling ~3.5 s early and misses the
> serve itself — see KNOWN_ISSUES "Pre-serve ball handling"). Also confirmed: **a
> faulted serve counts as a point AND a serve, with no re-serve** — and there are no
> faulted serves in this clip.
> Corrected serve times: 0:03 · 0:33 · 0:47 · 1:04 · 1:16 · **1:33** · 2:08 · 2:32 ·
> 2:45 · 3:06 · 3:39 · 4:04 · 4:43 · **5:01**.
> `tools/score_acceptance.py` scores every count in this table in one command.
>
> **OPEN: the +1 shot (99 detected vs 98).** Operator (2026-08-01): "could be my count
> is off by one OR you included a feed back to server." The second is now a concrete
> suspect — feeds are confirmed present in this clip (the 0:55 ball) and the baseline
> pipeline does NOT exclude them. **Resolve by listing the 99 detected shots and finding
> the one that is a feed**, rather than by assuming either count is wrong. Do this before
> treating 98 as a hard target — a 1-shot error is within the noise of everything else
> right now, but it also silently sets the bar for `shots = volleys + bounces`.

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

## BOUNCE RECALL — FIVE THINGS TESTED AND REJECTED (2026-08-02). DO NOT RETRY.

Bounce precision was identified as the highest-leverage target (it defines volleys, and
the operator's ruling makes the LANDING the primary shot-type signal). The deficit is
**71 bounces vs 81 truth**, i.e. ~10 missing, with 23 non-volley shots carrying no
landing. Five candidate causes were measured on `pb_5_minute_outdoor-2`. **All five are
NOT the cause** — recall is not threshold-limited:

| # | hypothesis | result | verdict |
|---|---|---|---|
| 1 | the one-per-interval cap keeps the WRONG bounce | in **33/33** intervals with a choice, the most-confident candidate **is** the earliest — i.e. already the physically correct landing | cap is sound |
| 2 | multi-candidate intervals hide a MISSED SHOT (so 2 landings are real and one is deleted) | single-candidate intervals median gap **1.49 s** (a normal exchange); multi-candidate **3.8–7.9 s** → they are DEAD TIME, not missed shots | not the cause |
| 3 | the away-from-camera confound cancels the bounce signature (per the 2026-07-21 finding) | missing-landing rate **near-side 27%** vs **far-side 34%** — roughly equal, and if anything the opposite of the prediction | not dominant here |
| 4 | `BOUNCE_MAX_OUT_OF_COURT_FT = 8.0` rejects real out-balls | widening to 12/16 ft gained **+1 bounce**; mean error unchanged | no effect — and see the operator rule below |
| 5 | occlusion — the ball is invisible at the bounce | intervals with NO bounce have **HIGHER** ball visibility (median **92%**) than those with one (**83%**) | **not the cause** |

Prominence was also swept (9.0 → 6.0 → 4.5 → 3.0 px): bounces 71 → 74, identity gap +5 →
+2, but **mean error got WORSE** (22.9% → 23.1%) as false dinks rose. The 2026-07-20 note
"do NOT globally lower the bounce prominence" still holds.

> **OPERATOR RULE (2026-08-02):** the **15 ft beyond-the-baseline envelope applies to
> PLAYERS ONLY** — where they may run. **The BALL must bounce within the court
> parameters.** So the ball's out-of-court margin must NOT be widened to match the player
> envelope; if anything it wants tightening. (Corrects an assumption made while testing
> hypothesis 4.) See [[project_pickleball_domain_rules]].

**Conclusion:** the ~10 missing bounces sit in intervals where the ball is clearly
visible, the cap is choosing correctly, and no threshold recovers them. They need a
DIFFERENT SIGNAL, not tuning — the documented candidate is the deferred 3-D
projectile/parabola trajectory fit, which the 2026-07-19 analysis already flagged as
addressing soft-vs-drive, volley detection AND speed at once. Do not spend more time on
thresholds here.

## IDENTITY VALIDATED BY RENDER (2026-08-01) — root cause is the FAR side, not user/partner

The 2026-07-27 handoff named user/partner identity (Stage 2.5) as the next foundation,
on the theory that track fragmentation swapped user↔partner at the who-served errors.
**Rendering the roles (`tools/verify_identity.py`) disproved that theory and found the
real cause.** This is the `feedback_consumer_output_validation` discipline paying off
again: no smoke test could see either result.

**1. The near-side axis is CORRECT — the ambiguous seed did NOT flip user/partner.**
Stage 2.5 warns `near players are close in the opening window (dx=0.2ft); user/partner
seed by starting corner is ambiguous`, so the fear was a global flip that would make
every "your" stat the partner's. It didn't happen: at **1:16 the render shows the woman
serving, labelled `partner`, matching operator truth**; the man is consistently `user` at
0:06 / 1:29 / 3:39. Automatic check agrees — *duplicate-role* frames (one role on two
tracks at once, provably impossible) are rare: **user 8, partner 53 of 18,862 frames**.
⚠ Note a fully-swapped assignment is self-consistent, so duplicates near zero is NOT
proof of correctness — only the render is.

**2. OPEN with the operator:** at **0:48 the render unambiguously shows the MAN (`user`)
serving from behind the baseline**, but operator truth says partner. Timestamps align
tightly elsewhere (0:33→0:33.9, 1:16→1:16.3, 1:29→1:29.5) so this is not drift.

**3. ROOT CAUSE of the who-served errors: Stage 2.5 discards real OPPONENTS as noise.**
The noise filter cuts on **median `court_y_ft` ∈ [-8, 44]**. The documented far-side
foot-point drift (±5 ft zone-precision, SYSTEM_DESIGN §3 Stage 2) pushes real opponents'
median just past the 44 ft baseline, so they are thrown away as adjacent-court
contamination:

| track | median court_y | p10 | in_court | verdict |
|---|---|---|---|---|
| 3 | 45.3 | 30.6 | 50% | **noise** ← real opponent |
| 3454 | 45.4 | 29.6 | 40% | **noise** ← real opponent |
| 3443 | 47.7 | 44.6 | 10% | **noise** ← real opponent |
| 965 | 36.5 | 29.2 | 70% | opp_a ✓ (median merely happened to land < 44) |

**Measured blast radius: `opp_a` absent in 41% of frames, `opp_b` in 36%** (present
11,051 / 11,971 of 18,862). **35 noise tracks / 19,034 rows** sit in the 22–60 ft band.
At **3:39 there are ZERO tracked players in the far half of the court**, so the
opponent's serve was attributed to the near-side man — the same mechanism the 07-27
handoff spotted at 2:24 ("the front thrower isn't tracked, so the behind player is the
only near track"). This ONE cause explains the wrong-side serves (3:39, 4:43), the
missed far-side shots (2:32, 4:04), and the contract's own open follow-up ("opponent
roles are contaminated").

**Adjacent-court players ARE separable from real opponents** — the discriminator is
range, not median: a real opponent works between the far kitchen and the far baseline
(**p10 ≈ 29–31**), while an adjacent-court player sits in a tight deep band (**p10 ≈
51–57**) and never comes forward.

**Consequence for the roadmap:** the Stage 2.5 *near-side* rebuild the handoff proposed
would not have fixed any observed error. Latent risk there is real but unproven — **72%
of `user` frames come from two tracks assigned at confidence 0.53–0.56** (tid 1452
covering 1:28–4:19, tid 4127 covering 4:23–5:14). Revisit after the far side.

## THE PLAY ENVELOPE (operator, 2026-08-01) — the fix, and two wrong turns on the way

**THE RULE: players are NOT confined to the 20×44 court.** They serve from BEHIND the
baseline and chase balls wide. Operator's real play envelope:
**5 ft beyond each sideline, 15 ft beyond each baseline** → `court_x ∈ [-5, 25]`,
`court_y ∈ [-15, 59]`. Any "is this person playing on our court" test must use THIS,
not the court rectangle.

**The defect this exposes.** Stage 2.5's noise filter cut at `med_y ≤ 44` AND required a
floor on `in_court_frac` — measured against the strict rectangle. Together these
**discarded far-side servers by construction**: standing behind the baseline means
out-of-rectangle. Operator-confirmed by eye, then confirmed in the data — at EVERY
operator-identified opponent serve, BOTH opponents were detected just behind the far
baseline and BOTH were classified `noise`:

| frame | opponent 1 | opponent 2 | role before |
|---|---|---|---|
| 0:47.5 | tid 797 (14.8, 50.4) | tid 583 (6.7, 51.3) | both `noise` |
| 3:39.8 | tid 3443 (4.7, 45.4) | tid 3454 (14.3, 46.2) | both `noise` |
| 4:05 | tid 3590 (5.0, 51.6) | tid 3713 (15.7, 52.0) | both `noise` |
| 4:43.7 | tid 3912 (7.3, 48.5) | tid 4399 (15.9, 52.0) | both `noise` |

Opponents sit at **court_y 45–54**; genuinely adjacent-court people separate at
**59–115 ft**. **v0.2.0 fix:** noise is judged against the play envelope, and the
`in_court_frac` floor becomes an **`in_env_frac`** floor.

### Two wrong turns — recorded so they are not repeated

**Wrong turn 1 — a too-clever rule.** First attempt made `(44, 52]` a "drift zone" a
track could occupy only if it *reached* the far kitchen line and kept a player-sized
*span*, reasoning from the documented far-side foot-point drift. It passed smoke 6/6 and
raised `opp_a` frame presence 59%→87% — but the window was too tight and the extra tests
too strict: it recovered only **one of the two** opponents at 0:47/3:39/4:05 and **neither**
at 4:43. Superseded by the envelope, which is simpler and operator-grounded.

**Wrong turn 2 — misreading which court is ours, then reverting a correct fix.** Zoomed
crops of the far side were read as "the recovered opponents are past the fence on the next
court; our far half is empty", and the fix was reverted on that basis. **This was a visual
error.** Our court's far half is a *thin foreshortened sliver* near the top of frame
(image x 1801→2739, y 1217→1370) while a NEIGHBOURING court dominates the view. The boxes
judged "past the fence" were on our court. **`tools/verify_identity.py` now projects the
court outline onto every frame** — never judge "is that player on our court" by eye again.

**The downstream regression that seemed to confirm the revert** (shots 99→118, serves
13→15) was misread too: the 0:47 serve moving to `opp_a`/far is **CORRECT** (operator: "at
:47 the opponent is serving"), and 0:48 is the operator **returning** it, not serving. The
residual over-count is real and traced to remaining contamination — an adjacent-court
figure reads `dist_from_net` 21.8–33.8 ft, satisfying the serve rule's "behind the baseline
(≥21 ft)", so contaminating tracks can steal serve status. **Track that as the open item,
not as a reason to revert.**

**Method lesson (reinforces `feedback_consumer_output_validation`):** across both wrong
turns, every wrong conclusion came from judging geometry by eye or from a coordinate that
was itself the thing in question. **Project the court, then look. Render before building.**


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

## REPORT VALIDATED PER-USER + ROADMAP (2026-07-22b)

Operator did a detailed 12-question review of the built report; all addressed. USER
counts (rating is per-user) now match operator truth well: dink 6=6, serve 4=4,
volley 6=6, returns 3=3 EXACT; drive 14 (12), drop 1 (2), FH 13 (15), BH 10 (9).
Report is internally consistent (header 96 in-rally shots, categories agree),
per-user where it should be (third shot, returns), and honestly labelled
(measurement coverage not "confidence"; volley % = share of YOUR shots; bounce map
states net/volley shots aren't shown). Third shot is gated (<4 user decisions), so
it is not coached off n=1.

**ROADMAP (operator priority: technique/enrichment BEFORE multi-clip):**
1. **Technique / body mechanics from POSE** (NEXT) — ready position, split-step
   timing, athletic stance/knee bend, contact point (front vs late), shoulder turn,
   balance, reach-vs-move, follow-through. Pose is 94% detected, 33 joints/frame, and
   these need NO ball height -> achievable on the 6ft camera. Adds camera-feasible
   shot QUALITY.
2. Report enrichment (surface more of what we compute).
3. Remaining doable accuracy: bounce recall (identity gap), fewer "unknown" strokes,
   opponent-side dink over-count (match totals only).
4. Net-hit detection (ball stops at the net; show net errors).
5. Multi-clip aggregation over time (AFTER technique).
6. HEIGHT-LIMITED quality (true speed, dink height, return depth, volley sub-types,
   spin) — deferred; needs a camera change (height-free methods all defeated).

## TECHNIQUE / BODY MECHANICS from pose (2026-07-22c) — quality layer

Camera-feasible shot QUALITY from pose (no ball height). Calibrated to OPERATOR
coaching standards.

- **Contact point (front vs late): SHIPPED.** Paddle-wrist net-ward of the hip at
  contact. Operator standard: contact "in front of your hip" (~1:00 forehand / 11:00
  backhand). Feeds Forehand/Backhand quality. User: FH 90% in front, BH 62%.
- **Knee bend / athletic stance: SHIPPED, calibrated to per-shot-type BANDS.** bend =
  180 - knee angle. Operator bands: serve/return 10-30, drive 20-35, drop 30-45, dink
  35-50 (soft shots need a deeper, lower base). Measured means land in-band
  (validated). Feeds Forehand/Backhand (drives) + Dink + Serve.
- **Shoulder turn / rotation: REMOVED.** Operator standard is peak BACKSWING rotation
  in degrees (drive 60-90, drop 20-45, dink 5-15). Absolute 3-D rotation is NOT
  reliably recoverable from one corner camera: recovering it from shoulder-width
  foreshortening is noise-dominated (dinks measured 62 deg vs the true 5-15). SAME
  monocular-3D limit as ball height. Don't ship what we can't measure.

- **Ready position (paddle up): SHIPPED, ZONE-AWARE.** Operator standard: paddle
  high at the kitchen (chest), dropping to waist/ankles as you move back (a high
  paddle deep in the court sends balls out). Reported per court zone. CAVEAT: we track
  the WRIST, not the paddle TIP (no paddle detection), so absolute chest/waist/ankle
  isn't reliable -- what IS reliable is the zone TREND (higher at net, lower back).
  User: trend correct (kitchen 0.14 > baseline 0.04) but LOW at the net (hands at
  waist, should be chest). Feeds Strategy + a "paddle up & ready" drill.
- **Split-step: NOT shipped.** A split-step is a few-inch vertical hop timed to the
  opponent's contact = a few pixels at this distance, at the pose-jitter floor; the
  "59% detection" was noise-firing, unverifiable. Same sub-pixel limit as shoulder
  turn. Deferred (needs a closer/higher camera).

**Principle reinforced:** contact-frame + ground-plane + relative quantities are
robust; absolute 3-D quantities (ball height, rotation degrees) are monocular-limited.
Technique now fills previously-empty quality circles + drives body-mechanics coaching
(late contact, not enough knee bend) in the improvement plan; Backhand surfaces as a
focus (weakest: 62% contact, 0% in-band drive knee bend).

## USER-LEVEL acceptance test (operator, `pb_5_minute_outdoor-2`, 2026-07-22)

The USAPA rating is PER-USER, so the user's counts are the real acceptance test.
User = near-left player. Operator counted, for the user only:

| type | op | pipeline | | stroke | op | pipeline |
|---|---|---|---|---|---|---|
| drive | 12 | 16 | | forehand | 15 | 14 |
| serve | 4 | **4** | | backhand | 9 | 11 |
| dink | 6 | **6** | | (unknown) | 0 | 3 |
| drop | 2 | 1 | | | | |
| lob | 0 | 1 | | volleys | 6 | **6** |
| **total** | **24** | 28 | | | | |

**Operator identities (must hold, per user):** drive+serve+dink+drop = FH+BH = total
(24). Every shot is a forehand or backhand (serves are forehands); a dink is soft +
at the kitchen whether it bounced OR was volleyed (dink & volley overlap); volley vs
bounce is the exclusive axis (shots = volleys + bounces).

**KEY RESULT: the user-level classification is GOOD** — dink 6=6, serve 4=4, volley
6=6 EXACT; drive/drop/stroke within ~1-4. The earlier MATCH dink over-count (35 vs
18) is entirely the OPPONENTS (far side, seen poorly by the corner camera), NOT the
user. **So the per-user rating is trustworthy.** Remaining user gaps: +4 total (2 are
between-point drives not in any rally = droppable; ~2 drive/drop confusion), 1 false
lob, unknown strokes 3.

**Report directives (operator):** show BOTH match total and the user's share per row
("Dinks: 22 in the match, 6 by you"); keep every USAPA item in the chart with
filled/unfilled circles; the USAPA rating is for the selected user only.

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

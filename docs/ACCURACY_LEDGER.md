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

## PER-SHOT OPERATOR LABELS (2026-08-03) — the counts were hiding two large errors

The operator labelled the first 30 detected shots individually (`tools/label_shots.py`,
`_labeling/labels.csv`). **This is the first per-shot ground truth for shot TYPE, and it
overturns the count-based picture.**

### 1. 30% of detected "shots" are NOT SHOTS

| | n |
|---|---|
| real shots | 21 |
| **NOT a shot** — feeds, balls rolled back at the net, ball handling, picking the ball up | **9 (30%)** |

The pipeline typed those 9 non-shots as **drive x5, dink x2, serve x2** — so between-point
junk is inflating the shot, dink AND serve counts simultaneously.

**The arithmetic that matters.** We report 99 shots against a truth of 98 and called that
"+1". If ~30% are junk, only ~69 detections are real, so we are **missing ~29 real
shots**. The +1 agreement was two large errors cancelling. ⚠ Caveat: shots 1–30 cover
0:06–1:44, which contains a lot of clip-start dead time; the junk rate over the whole clip
may be lower. Label another block before treating 30% as the global rate.

### 2. Shot-type accuracy on the REAL shots is 33%, not ~58–73%

| type | n | precision | recall |
|---|---|---|---|
| dink | 3 | 40% | 67% |
| drive | 7 | 30% | 43% |
| drop | 3 | 50% | 33% |
| **lob** | 2 | **0%** | **0%** |
| **reset** | 1 | **0%** | **0%** |
| **return** | 3 | **0%** | **0%** |
| serve | 2 | 33% | 50% |
| **overall** | **21** | **7/21 = 33%** | |

`is_volley` was 4/4 correct (small sample, but the one signal that held up).
**`return` scores 0% because our taxonomy has no `return` type** — two returns were called
`serve`. That is a taxonomy gap, not only a classifier error; the ledger counts returns
separately (14 returns = 14 serves), so Stage 6 should emit it.

### 3. The "in-play" filter would have made things WORSE — do not ship it

The cross-net-exchange test (a shot is in play if an opposite-side shot falls within 1.5 s)
was validated against these labels for the first time:

- it **killed 9 of the 21 real shots**,
- and **missed 4 of the 9 junk ones**.
- precision on junk **36%**, recall **56%**.

Why: **a ball rolled back at the net, or fed to the server, DOES cross the net** — it looks
exactly like an exchange. And real shots look isolated because their cross-net partner is
one of the ~29 shots we never detect. Defeated in both directions. This is the filter that
looked fine on COUNTS (it produced 99 shots) — another case of a count agreeing for the
wrong reason.

### 4. OPERATOR TYPE RULES (2026-08-03) — implementable, currently not implemented

From the kitchen area: **soft to the baseline = LOB · hard to baseline or transition =
DRIVE · into the kitchen, or soft into the transition zone, = DINK.** A volley struck near
the opponent's baseline the operator defines as a **volley drop** (else volley dink) —
explicitly a fuzzy area he has chosen a convention for. So type is a
**(hitter zone, landing zone, pace)** table. Our rule only uses landing-distance-from-net
plus hitter zone, which is why **both lobs were called drives**.

### Labelling-tool follow-ups the operator asked for
- Needs **broader context**: could not always tell a return-of-serve from a drive from a
  2 s window. Show the preceding shot / rally context.
- **Multiple shots fall inside one segment**; make the labelled contact unmistakable.
- Some shots have **no hard rule** — record them as genuinely ambiguous rather than forcing.

## DINKS — half junk, half a LANDING-COVERAGE problem (2026-08-26)

Court C emits 17 dinks against the operator's 5; the acceptance clip is nearly exact (34 vs
their 32). The gap splits cleanly in two.

**Half of the extras are junk we emit as dinks.** Of the dinks matching no operator label, 4
of 6 on court C and 6 of 8 on the acceptance clip are on their NOT-A-SHOT list. Those are the
between-point problem wearing a dink label, not a typing error.

**The rest is drive <-> dink confusion, and it tracks LANDING COVERAGE:**

| | landing available |
|---|---|
| shots we get RIGHT on drive/dink | 13/24 and 20/30 (54-67%) |
| shots we CONFUSE | 2/6 and 5/14 (33-36%) |

Without a landing the classifier falls back to SPEED, and the speed is not fit for the job.
On shots the operator calls drives we measure **3.5-14 ft/s** (a drive at the kitchen is 30+),
and on shots they call dinks we measure **27-65**. Wrong by 3-10x in both directions, from
both speed sources (`ppf_instantaneous` and `trajectory_horizontal`). Tuning the 16/25 ft/s
thresholds cannot fix a measurement that far out.

### REJECTED: ball travel as the fallback

A dink travels a few feet into the kitchen, a drive travels the court -- a displacement over
a window, the kind this reconstruction handles. It does not separate them:

    all labelled dinks/drives      dink median 31.5 ft   drive median 31.0 ft
    only those with NO landing     dink median 22.3 ft   drive median 26.7 ft

The apparent "77% at an 11 ft split" on the no-landing subset is **the majority class**: 30 of
those 39 shots are drives, so predicting "drive" for everything scores 77%. Not a signal.

### Four ways to replace the missing landing, all measured, none sufficient

The operator proposed three of these (2026-08-26); the fourth was the standing lead.

**1. "The shots with no landing are volleys."** They are not, and this is worth knowing: of
the landing-less shots only **2 of 29** (court C) and **8 of 50** (acceptance clip) are
volleys. The rest DID land -- we failed to detect or attribute the bounce.

**2. Extrapolate where a volleyed ball would have landed.** Validated on shots that DID bounce
(so the answer is checkable): **median error 25.6 ft on a 44 ft court**, and the landing ZONE
-- kitchen / transition / baseline, the split that decides the type -- correct **50%** of the
time against 33% for a coin. Extrapolating the reconstructed position instead gives 13.9 ft
median but a p90 of 138.7 ft. Not usable.

**3. Contact height against the waist.** Half of it is true. A contact ABOVE THE SHOULDER is
almost never a dink (0 of 9 on the landing-less shots, 3 of 18 overall) -- a real one-way
signal. But BELOW the waist says nothing: 20 dinks against 18 drives. As a classifier the
whole rule scores 69% on the landing-less shots where **always guessing "drive" scores 77%**.
Applying only the half that holds moves 7 shots overall and 2 on the landing-less subset --
inside the noise.

**4. Height-derived floor contacts as landings.** `bounces_height.json` has 131 and 175 floor
contacts against 43 and 76 in `bounces.json`, and their `court_xy_ft` is the reconstructed
position (it reads `[-5.7, 180.5]` on a 44 ft court) -- but at a floor contact the ball is at
z = 0, exactly where the ground homography is EXACT, so the PIXEL projects correctly.
Validated against known landings: **median 1.3 ft outdoor, 5.9 ft on court C**. The problem is
coverage, not accuracy: a height contact exists before the next shot for only **9 of 29** and
**10 of 50** landing-less shots. The "3x more floor contacts" are mostly between points and in
intervals that already have a bounce, so they do not become landings.

### What this means

Drive-vs-dink WITHOUT a landing is at or near the limit of this camera position. Four
independent substitutes -- speed, ball travel, extrapolation, contact height -- all fail, and
the one that works (a real bounce) is available for 60% of shots and cannot be raised much.

**The productive half of the dink gap is the junk**, and that is a serve/return problem: about
half the over-count is between-point balls typed as dinks, which the rally gate removes once
the serves are right. Fixing serve detection pays twice.

## THE MISLABELLED RETURN WAS BLOCKING THE REAL SERVE (2026-08-27)

Asking WHICH condition blocked each remaining serve, rather than tuning, found a guard doing
the opposite of its job. The five the acceptance clip still missed:

| serve | what was happening |
|---|---|
| 1:34.22 | contact discarded; has-ball 0.57, **and a kept shot at 1:35.50 within 2 s** |
| 2:32.50 | contact discarded; has-ball 0.80, **kept shot at 2:33.92 within 2 s** |
| 4:42.30 | contact discarded; has-ball 0.82, **kept shot at 4:41.60 within 2 s** |
| 3:13.00 | contact KEPT, simply not accepted as a serve |
| 5:01.90 | no contact at all, even with the handling filter off |

**Those "kept shots within 2 s" are the RETURNS we had mislabelled as serves.** The guard
exists to stop adding a duplicate of the same contact -- but a duplicate is on the SAME side.
A shot on the OPPOSITE side 1.2 s later is the return, and the serve legitimately belongs
before it. Guarding against any nearby shot let the mislabelled return block the real serve
behind it: the same "a false serve blocks the real one" shape already recorded in serve
acceptance, in a new place.

Making the guard same-side-only was measured before shipping, against also lowering the
has-ball threshold to reach the 0.57-0.82 cases:

| change | admitted | serves caught |
|---|---|---|
| **guard -> same side only, threshold stays 0.85** | **2** | **2** |
| ...and threshold 0.70 | 7 | 3 |
| ...and threshold 0.55 | 11 | 4 |

The guard fix is free -- everything it admits is a real serve. Lowering the threshold to chase
the rest costs four junk contacts per two serves, so it was not taken.

    serves FLAGGED           18/25 -> 20/25 (80%)     contacts never detected  5 -> 2
    servers NAMED            12/20 -> 16/20           serving side       19/20 (held)
    shot types               107/191 -> 110/191       volleys        93/114 -> 96/117
    missed shots recovered   31/53 -> 34/53           real shots outside rallies  4 -> 3
    court B serve timing     1.19s -> 0.38s

### What is left, and it is only two things

* **3:13.00** -- the contact is detected and kept; `structure_points` does not accept it as a
  serve. A rule question, not a detection one.
* **5:01.90** -- no contact exists even with the filter off. Genuinely invisible to the
  impulse detector.

Serve detection went 10/25 -> 20/25 across this work. Six routes were rejected on the way and
are recorded above; what finally worked was the operator's own rule, and the correction that
made it work was dropping the underhand test from it.

## THE SERVER HAS THE BALL — the serve restoration that worked (2026-08-27)

Operator, after four failed attempts: *"once I know which side is serving, I can tell which
shot is a serve by a) they have the ball and b) ball moves forward toward the net (to
distinguish hitting it underhand as a feed to their partner). ignore whether underhand or
not."*

Dropping the underhand test was the correction that made it work.

**"Has the ball" is ball-NEAR-THAT-PLAYER over a window, not ball-at-rest.** That distinction
is the whole thing. Ball-at-rest was tested and failed (83 px vs 118 px) because a player
bounces the ball before serving -- it moves plenty. But it stays WITH THEM, and no other shot
does: in a rally the ball arrives from the opponent and is near the hitter only at the last
instant.

The radius has to be tight. Measured on the discarded contacts, share of the 1.5 s before
contact with the ball on that player:

| radius (fraction of player height) | on serves | on everything else |
|---|---|---|
| 1.2 | 0.93 | **0.97** -- separates nothing |
| 0.6 | 0.93 | 0.55 |
| **0.4** | **0.81-0.93** | **0.28-0.37** |
| 0.25 | 0.50-0.67 | 0.06-0.17 |

At 1.2 body-heights everything is "near the player", which is why the first attempt read a
flat 0.97 for serves and junk alike and looked like the operator's rule failing.

### The gate, and why all three parts are needed

| gate applied to the discards | admitted | serves caught |
|---|---|---|
| formation + serving side | 16 | 7 |
| + underhand (ball below the hip) | 56 | 9 |
| **+ has the ball + moves forward** | **13** | **7** |

Shipped with a fourth condition the offline test did not have: never restore a contact within
2 s of one already kept. We are recovering a MISSING serve, not adding a second contact beside
one we have.

### Result

    serves FLAGGED            17/25 -> 18/25        contacts never detected   6 -> 5
    court C serve recall      0.80 -> 1.00          court C serve precision   0.80 -> 0.91
    court B serve recall      0.80 -> 0.90          court B servers named      5/10 -> 6/10
    court C shot types        41 -> 43              court C junk in rallies    12 -> 10
    court C rally-end error   6.83s -> 3.85s        court B rally-end error   2.87s -> 2.34s

Cost, all on the acceptance clip: one extra shot, junk in rallies 20 -> 21, serve precision
0.76 -> 0.72. Every acceptance bar still passes, including MAX_FALSE_POSITIVES and
MAX_WRONG_PLAYER.

Five routes were rejected before this one. What separated it: it asks a SUSTAINED, RELATIVE
question (was the ball with this player for a second and a half) rather than a per-frame one
(was the ball below the hip AT the contact frame). That is the same distinction that governs
everything else built on this footage.

## REJECTED: identifying the serve by underhand swing + forward ball (2026-08-27)

Operator's rule, and it is correct as a description of the game: once you know which side is
serving, the serve is the contact where **(a)** they have the ball, **(b)** the swing is
**underhand**, and **(c)** the ball then moves **forward** -- (c) separating a serve from an
underhand feed to their own partner.

Measured on the 231 and 125 contacts the handling filter discards, which is where 7 of the 8
missing serves live:

| test | on the operator's serves | on everything else |
|---|---|---|
| **(b)** ball below the hip line at contact | 6/15 and 10/18 (**~50%**) | 83/216 and 40/107 (**~38%**) |
| **(c)** ball moves toward the other side | 12/15 and 10/18 (**~70%**) | 99/216 and 42/107 (**~45%**) |
| **(b) AND (c)** | 4 and 5 serves caught | **56 and 23 contacts admitted** |

Fifteen and eighteen serves wanted; the combined rule admits fifty-six and twenty-three, and
catches fewer than a third of them. It does not work.

**Why, and it is the same reason as everything else today.** The rule is about the MOMENT of
contact, and our measurements at that moment are too coarse. The detected impulse frame is
approximate and the ball moves fast, so by the frame we call "contact" the ball has often
risen above the hip -- a real serve reads "below the hip" only about half the time, barely
above the 38% of everything else.

One implementation note worth keeping: contact height must be measured from the BALL against
the hip line, not from a wrist. The wrist version picks whichever hand is higher -- often the
tossing arm -- and read False for **every** real serve, which looks like the rule failing when
it is the code failing.

### Serve DETECTION: the routes tried and closed

| route | outcome |
|---|---|
| relax the invisible-gap threshold | no serves gained, more false ones |
| ball at rest before the serve | 83 px vs 118 px -- not separable |
| bound the excursion window at the next contact | +1 serve, and 5 other metrics worse |
| restore discards where the formation agrees | 16 admitted for 7 wanted |
| underhand swing + forward ball | 56 admitted for 15 wanted |

Serve detection stands at **17/25 flagged**. Every route through the BALL has failed because
the ball cannot see the serve; every route through PLAYER POSITION has failed because it
cannot pin down the MOMENT. What is still untried is a serve-specific motion signature over a
WINDOW rather than at a frame -- the toss-and-strike as a trajectory, not a single contact
height. That is a different kind of measurement from anything attempted, and it should not be
started without deciding it is worth the build.

## REJECTED: restoring discarded serves where the formation agrees (2026-08-27)

The plan looked sound and the arithmetic killed it. `reject_same_side_runs` keeps one contact
per same-side run, and 7 of the 8 serves we miss still exist among its discards. The serve
formation independently says a point begins at all 8. So: make the filter record what it
drops, and put a discard back only where the formation agrees.

Built and measured, tightening three times:

| gate | restored on the acceptance clip | serves FLAGGED | serve precision |
|---|---|---|---|
| (before) | -- | 17/25 | 0.76 |
| formation + serving side + no kept shot within 2 s | 18 | **15/25** | **0.62** |
| ...and struck from behind the baseline | 18 | -- | -- |
| ...and at most ONE per formation window | 16 | -- | -- |

**Seven serves were wanted and the tightest gate still admitted sixteen.** The depth test
barely filters, because pre-serve handling IS deep -- that is where serving happens. The
one-per-window rule barely filters either, because the formation has **32 windows for 14
serves**.

That is the whole problem in one line: the formation's RECALL is excellent (every serve falls
in a window) and its PRECISION is not (it holds 26-36% of the clip). A high-recall,
low-precision cue cannot authorise putting shots back -- it says "a point begins somewhere
around here", which is not the same as "this contact is the serve".

Reverted. Shots went 125 -> 143 on the acceptance clip and serves flagged went DOWN, which is
the signature of adding junk that then competes for the serve slot.

**What would actually help**: something that narrows a formation window to the MOMENT of the
serve. The formation cannot do it (identical at the serve and its return), and neither can
the ball (every ball route has now failed). A serve-specific pose signature -- the toss, the
underhand swing -- is the obvious untried candidate, and pose is already computed.

## THE SERVE FORMATION — a rally-start cue that needs no ball (2026-08-26)

Operator, reviewing the eight serves we miss: *"serve starts by the players hitting hit behind
the baseline along with their partner, and one of the opposing players are behind the baseline
as well"*. Measured, and it is the strongest signal found today -- and the only one that does
not touch the ball, which matters because every ball-based route to these serves has failed.

| | players behind a baseline | someone behind BOTH baselines |
|---|---|---|
| **serves** (acceptance clip) | median 3 | **93%** |
| **serves** (court C) | median 2 | 60% |
| any other shot | median 1 | **15-16%** |

**As a rally-START detector it has near-perfect recall**: every operator serve falls inside a
formation window -- 14/14 on the acceptance clip, 9/10 on court C. Precision is the weak side:
32 windows for 14 serves, and the formation holds 26-36% of the clip.

**It CANNOT separate a serve from its return.** The formation is identical 1.2s later (93% vs
100%, 60% vs 78%) because nobody has moved. So it does not fix the serve/return confusion
directly -- but it says, independently of the ball, *a point begins about here*.

### How to use it, and why it matters now

The eight missing serves all sit inside a formation window, and 7 of the 8 contacts still
EXIST but are discarded by `reject_same_side_runs` (which keeps one contact per run). Two
independent cues therefore agree at those moments: the formation says a serve happened, and a
discarded candidate says a contact happened. Promoting a discarded contact ONLY where the
formation agrees is far safer than loosening the filter for everyone -- the filter removes
~200 junk detections per clip and must keep doing so.

That needs the filter to expose what it drops, which it currently does not. That plumbing is
the next piece of work.

### A process note

The list of missing serves sent for review was keyed to the CURRENT pipeline while the
annotated video the operator watched was rendered 2026-08-24, before that day's changes. Shot
numbers had shifted, so "#32 is annotated correctly" describes a different shot. **Re-render
the video whenever a review list is produced from it** -- the numbers are the only link
between the two, and they are not stable across a detection change.

## SERVE / RETURN — where it stands, and one fix REJECTED after measuring (2026-08-26)

    serves FLAGGED as serves      10/25 -> 17/25
    serve/return confusions       27 -> 15 across the three reviewed clips
    "return called a serve"       6 -> 0 outdoor, 4 -> 2 court C
    shot types correct            95/191 -> 107/191

Two fixes landed: the same-side run now keeps the contact the BALL LEAVES ON (above), and a
"serve" the ball demonstrably reached from the OTHER SIDE is retyped as the return it is.

### The residual is one thing, named

Of the 8 serves still not flagged, **7 are the same shape**: the shot we call the serve is the
one the operator labelled the RETURN, 1.0-1.4s later -- the serve-to-return interval. Every
remaining `return -> drive` error is that same missing serve one step downstream, because
Stage 6 derives the return structurally (previous shot is_serve, opposite side), so a missed
serve costs the return too.

**7 of those 8 contacts still exist with the handling filter switched off.** They are being
discarded, not missed.

### REJECTED: bounding the excursion window at the next contact

The excursion rule measures how far the ball gets in the second after a contact, and that
window BLEEDS: in a tight cluster an earlier contact inherits the travel of the real shot
that follows it and then outscores it. The serve at 1:34.22 lost to a wobble 0.7 s earlier
credited with 909 px of travel, all of it the serve's own flight:

    run [(93.20, 268), (93.53, 909), (93.75, 806), (94.00, 646), (94.22, 619)]
                                                       the operator's serve is 94.22

The diagnosis is right and the fix is obvious -- stop the window at the next contact, so only
a shot the ball actually leaves on scores. **Measured, it is net negative:**

| | change |
|---|---|
| serves flagged | 17 -> 18 |
| shot types correct | +6 |
| volleys correct | **-2** |
| rally ends within 2s | **-3** |
| serving side correct | **-2** |
| wrong_player | **1 -> 2** (trips a hard acceptance bar) |
| false positives | 24 -> 25 |

Reverted. Bounding the window makes every contact in a tight cluster score near zero, so the
choice among them becomes arbitrary. A version that falls back to impact strength when all
excursions collapse is the obvious next thing to try, and has not been tried.

## THE SERVE WAS BEING DELETED BY THE HANDLING FILTER — FIXED (2026-08-26)

Two compounding faults, not one. The inert serve detector (below) was the second.

**The first: `reject_same_side_runs` was discarding the serve itself.** A serve is preceded
by the server's own bouncing, so those contacts form one same-side run and only one survives.
The rule branched on the run's DURATION -- keep the last if the run spans >= 8 s, else keep
the strongest impact. Real pre-serve runs are 2-6 s, so the "keep the last" branch **never
fired**, and the strongest impact in a handling run is a BOUNCE, whose direction reversal is
far sharper than a serve's. The serve was the one shot reliably thrown away:

    RUN span=4.32s rule=strongest kept=3.80 of [3.8, 4.23, 4.6, 5.22, 5.47, 5.74, 5.99,
                                                6.45, 7.55, 7.77, 8.12]
                                                       the operator's serve at 8.25 is here

Confirmed by disabling the filter: **11 of the 12 missing serve contacts reappeared.** The
filter still has to run -- it removes ~200 junk detections per clip -- so the question is
which shot it keeps.

**The fix asks what the run is actually about: which contact SENT THE BALL AWAY.** That is
the shot; everything else is handling. It settles both cases the duration rule was straining
to separate, without caring how long the run happened to be: in bounce-bounce-SERVE the ball
leaves on the LAST contact, in STRIKE-then-wobble it leaves on the FIRST.

| | before | after |
|---|---|---|
| serve contacts detected | 13/25 | **19/25** |
| serves FLAGGED as serves | 10/25 | **17/25** |
| shot types correct | 95/191 | **104/191** |
| volleys correct | 83/114 | **93/114** |
| servers named | 10/20 | **12/20** |
| previously-missed real shots recovered | | **+8** |
| false positives (acceptance bar) | 23 | 24 |

### Two things that were tried first and did NOT work

* **"The ball is at rest before a serve."** It is not. Pre-contact ball motion is 83 px for
  serves against 118 px for every other shot -- the best split reaches 87% accuracy only by
  calling almost nothing a serve (0% serve recall). Measured before building on it.
* **Lowering the invisible-gap threshold** (0.7 s -> 0.15 s): no serves gained, more false
  ones. The condition is wrong, not mistuned.

### Residuals, honestly

* Four shots on the acceptance clip now sit 0.24-0.65 s BEFORE the rally that should contain
  them -- the run's chosen contact shifted by one. `real_outside_rallies` 0 -> 4.
* Court C over-emits dinks badly: 17 against the operator's 5. Outdoor is close (34 vs 32).
* Court B moved slightly the wrong way on several counts.

### A stale truth that cost me an hour of wrong conclusions

`dinks_truth = 18` in the regression is the 2026-07-22 acceptance figure for
`pb_5_minute_outdoor-2`. The operator's own shot-by-shot review of `outdoor-7` counts **32**.
Reading our 34 against the stale 18 looks like a threefold over-count; against the real 32 it
is nearly exact, and the change that produced it was an improvement. The review now
supersedes those constants wherever it covers the same video -- **but only when it covers the
WHOLE clip**: court B's typed shots are every one of them a shot we MISSED, so its "2 dinks"
is two among the misses, and printing that beside our 20 invents a tenfold error out of
nothing.

## THE SERVE DETECTOR HAS GONE INERT (2026-08-26)

Operator asked how we know who hits each shot, given they identified themselves at setup.
Two separate mechanisms, and measuring them separated the problem cleanly.

**Attribution is not the problem.** A shot is credited to the closest player at contact --
wrist from pose first, then bbox, then foot. Scored against the operator's labelled hitters
on the two reviewed clips: **135/139 = 97% correct**, with 3 spatial misses and 1 frame where
the right player was not tracked. Identity is a setup CLICK (confidence 0.95) propagated to
the tracker's other fragments by appearance+height, simultaneity and continuity; it works.

**The serve detector fires on a condition that no longer occurs.** A serve has no incoming
ball trajectory, so the impulse detector is blind to it and a dedicated rule handles it: the
ball REAPPEARS after >= 0.7 s of not being visible (dead time), with an outgoing launch, near
a player. That was right when the ball was untracked between points. **TrackNet v4 now tracks
the ball straight through the dead time** -- held, bounced, carried -- so the gap never opens:

| | |
|---|---|
| operator serves where a >= 0.7 s invisible gap still exists | **9 of 24** |
| on the acceptance clip alone | **2 of 14** |
| serves we actually flag, at a 0.75 s match window | **9 of 24** |

The two numbers are the same nine. The detector fires exactly when its trigger holds and
nowhere else. Lowering the gap does not recover them (swept 0.7 -> 0.15 s: no gain, more
false serves) -- the condition is wrong, not mistuned.

**This is the third time a filter has gone inert under a better ball track**, after the
ground-ball filter. The lesson stands and is now cheap to act on: every closed accuracy claim
needs a standing score, because an improvement upstream can silently disable the thing that
depended on the old weakness.

### What it costs, downstream

A missed serve is not one missing shot. The rally then opens on the RETURN, which is
indistinguishable on every axis Stage 5 can see -- both are struck from behind the baseline,
and after a missed serve the return also has a long clear gap in front of it. So:

* the receiving side is credited as the server (half of all server errors),
* the rally START shifts a shot later, which drags the measured rally END with it,
* the third shot -- a core USAPA item -- is the wrong shot.

Beware the match window when reading serve recall: **9/24 at 0.75 s, 13/24 at 1.0 s, 21/24 at
1.5 s.** The extra eight only appear once the window spans the serve-to-return interval, so
they are returns. Any serve figure quoted without its tolerance is meaningless.

### The next fix

Trigger the serve on the ball being AT REST on the server's side rather than ABSENT. The
signal is already validated for the neighbouring question: the ball's positional span in the
second before contact separates the operator's 51 labelled serves and returns at 82%, and at
the same 82% for every window from 0.6 s to 1.5 s. It is a sustained, relative measurement,
which is the kind this reconstruction answers well.

Not attempted yet: span alone is not sufficient as a detector (real serves range 5-52 ft of
pre-contact span, because the previous point's ball is sometimes still moving), so it needs
pairing with the rally-end gate -- after a trusted end, the next launch from the serving side
is the serve.

## SERVER ATTRIBUTION — 50%, and the cause is the RETURN being flagged as the serve (2026-08-24)

Newly measurable: the operator recorded who served every point on both indoor courts, and
nothing had ever scored it. **10 of 20 correct** (court B 6/10, court C 4/10).

Nine of the ten errors get the SIDE wrong, which looked at first like the rally opening on a
serve we had missed. It is not that. Split by `serve_is_inferred`:

| | servers correct |
|---|---|
| rallies where we DETECTED a serve | **8/18 (44%)** |
| rallies where the serve was inferred | 2/2 |

The errors are concentrated where we DID flag a serve. So we are not failing to find the
opening shot — **we are flagging the RETURN as the serve** whenever the real serve went
undetected, and then crediting the receiving side as the server.

**Depth and gap cannot separate the two, by the operator's own domain rule: a return is hit
from behind the baseline.** After a missed serve the return also has a long clear gap in
front of it, so it satisfies both serve conditions exactly. `structure_points` is not
misjudging anything — it is being handed a shot that is identical on every axis it can see.

**What could separate them, and how far it got.** Before a serve the ball is on the server's
own side; before a return it has just crossed the net. That is a sustained, relative question,
the kind the reconstruction answers well. Measured — flipping the server whenever the ball
crossed into the striker's side just before:

| look-back | fires | fixes | breaks | net |
|---|---|---|---|---|
| 1.2 s | 5 | 4 | 1 | **+3** |
| 1.8 s | 8 | 4 | 4 | 0 |
| 2.5 s | 4 | 2 | 2 | 0 |
| 3.5 s | 2 | 2 | 0 | +2 |

**Not shipped.** +3 on 20 samples, from 5 fires, at a window that gives nothing one step
either side — non-monotonic in the threshold is the shape of noise, not of signal. The
principle is right and the route is worth returning to; the sample is far too small to accept
a swept threshold on, and this codebase has been burned by exactly that before.

What it needs is more labelled rallies, not more tuning. `server_correct` / `server_judged`
are in the regression table so the number cannot quietly drift while that is arranged.

## WHAT IS ACTUALLY LABELLED — a full audit (2026-08-24)

Before asking the operator for more labels, an inventory of every labelled file under
`data/`. Ten distinct videos have been touched; the labelling is far more uneven than the
folder count suggests.

| axis | coverage |
|---|---|
| shot-by-shot review (a type per shot) | **1 video** — PB 5 minute outdoor, 111 typed |
| rally-level truth (points, servers, shot counts, ends) | **3 videos** — outdoor, indoor court B (10 pts / 82 shots), court C (10 pts / 59 shots) |
| hand-labelled ball PIXELS | **~10 videos**, 18,862 frames, 12,576 with a visible ball |
| nothing at all | PB 5 min indoor 1 court A |

Three defects found in the audit, all fixed without asking for a single new label:

**1. The operator's vocabulary was being discarded.** They type the shot the way a player
says it — `drive/volley`, `backhand drive`, `3rd shot drive`. Eight of court B's twenty-one
typed shots were not in the canonical set, and `score_shot_types` treats an unrecognised type
as a NON-SHOT label, so **six real shots were being reported as false positives over a
wording difference**. `norm_type()` now splits the phrasing into the fields it carries —
`drive/volley` is a drive AND a volley, `3rd shot drop` records where in the rally it fell.
Court B: **4/12 = 33% -> 8/20 = 40%**, and its 6 phantom false positives are gone.

**2. Court B's typed shots are ALL shots we MISSED.** Every one carries `detected: False` —
they came from a missed-shot review, so they are the hardest cases by construction and the
rate is NOT comparable to a clip reviewed shot by shot. Reading "court B 33%" beside "outdoor
47%" invites exactly the wrong conclusion, and I did that once. `shot_type_sample_was_missed`
now sits next to the rate: **20 of 20 for court B, 21 of 111 for outdoor.**

**3. Per-rally SERVER truth existed and nothing measured it.** `truth.json` records `server`
and `n_shots` for every point on both indoor courts; only the end time was being read. Now
imported as `rally_truth` and scored:

| clip | server correct |
|---|---|
| court B | 6/10 |
| court C | 4/10 |
| **total** | **10/20 = 50%** |

A coin flip, on the axis a known-weak subsystem depends on. That is a new open finding, not a
regression — it was simply never looked at.

**The real gap needing new labels is shot TYPES on a second video.** Court C is the best
target: it already has rally windows, servers, per-rally shot counts and outcome notes, so a
review sheet drops into a frame that is already anchored.

## THE TRUTH STORE — one home for the operator's answers (2026-08-24)

`docs/truth/<VIDEO>.json`, keyed by SOURCE VIDEO so it survives re-analysis into a new clip
folder. Built by `python -m tools.truth_store --import-all`, read by every scorer through
`known_shots(clip)`. Inspect with `--report`.

Its whole purpose is that the operator never reviews the same thing twice. That failed
silently in five ways at once, all found while chasing "6 real shots sit outside a rally" —
of which **only one was a real pipeline problem**.

| defect | what it did |
|---|---|
| an untouched sheet was importable | filed **109 of our own detections** as operator truth at top authority |
| import was not idempotent | re-running appended clones; **8 duplicate pairs** had accumulated |
| a higher-authority source never corrected the TIME | **35 shots** carried an older hand-typed time; every scorer matches on a window, so a stale time silently moves a shot |
| the not-a-shot path retracted without CLAIMING | two of the operator's rows became one record — "not a shot" merged with the next row's real shot |
| row-order greedy matching | once row #26 took the entry nearest IT, #27 took the one belonging to #28; three notes landed on the wrong shots |
| an OLDER review overruled the latest | `shot_review.json` names SHOT NUMBERS ("#18 is mislabeled") and the numbering changed between reviews — **7 times counted as junk AND as a confirmed shot** |

Rules now enforced, all of them the operator's own words:

* **"Use the last one I built as the truth if there is a conflict between any reviews."** An
  older false-positive claim cannot stand where the latest sheet confirms a shot, and within
  the span a sheet covers it is the complete account — the sheet lists every detection and
  lets the operator add the ones we missed, so an older label with no row of its own is
  contradicted by it. Overruled entries move to `superseded_shots` /
  `superseded_false_positives`; nothing the operator said is ever deleted.
* **"If I did not mark it as wrong, then I deliberately considered it to be correct."** A
  blank row is a confirmation — but ONLY in a sheet that carries at least one mark.
* Matching is **shortest-pair-first and one-to-one**, never greedy in row order. Greedy-in-
  order matching invents a story; it read rally-end recall as 0/16 when the answer was 12/16.

**Counts after the corrections: 112 known shots** (was 118 before the stale labels were
demoted, 131 before junk retraction), 38 false positives, 16 rally ends, 21 shots we missed.
Shot type 51/111 = **46%**.

### The one real finding: a whole point was missing — FIXED 2026-08-24

| video | xls row | shot # | operator marked |
|---|---|---|---|
| **0:46.57** | 25 | #17 | `serve` |
| **0:48.62** | 27 | #19 | `return` — *"bounced on far side and was missed by opponent. Rally ended"* |

We DETECTED both and typed the serve as a `drive`. `is_serve` was therefore false, no rally
opened, and the rally-end gate discarded the pair as dead-time ball-handling. Rally 1 ended
0:36.83 and rally 2 started 1:04.60 — the point between them was simply absent.

**Cause: junk immediately before a serve hides it.** `structure_points` asks for a >= 3 s gap
to the previous detection before a deep shot can "open a point". The serve is 31.6 ft deep —
plainly a serve — but sits 2.45 s after a detection the operator calls *"not a shot. Looks
like it picked up something from court behind"*. This is the operator's own most-reported
pattern ("false shots often before serves") and it was costing whole points, not just a
false positive.

**Fix: measure the gap to the last contact from the OTHER side.** "Opens a point" means
nothing was in play, and the ball is only in play if it came from the opponent — so junk on
the server's own side cannot hide their serve. Against the last opposite-side contact the gap
at 0:46.57 is 9.7 s. Every rally shot crosses the net, so this is a rule of the game, and it
**adds no new threshold** — it reuses the same 3 s.

Two restrictions, each measured, without which the change is a wash:

* **A relaxed candidate may FILL a slot the strict rule left empty, never TAKE one it
  filled.** Letting it displace cost court B its 1:44 serve to a candidate 2.9 s later —
  recall +1 outdoors, −1 indoors.
* **A relaxed candidate must be ANSWERED.** A serve is played back; a ball handled in dead
  time is not. Without it the relaxation opened a rally at 3:31 built from four junk shots
  and no real ones.

| | recall | precision |
|---|---|---|
| shipped | 27/34 (79%) | 82% |
| relaxed, may displace | 27/34 (79%) | 77% |
| relaxed, no displace | 28/34 (82%) | 80% |
| **+ must be answered** | **28/34 (82%)** | **82%** |

End to end: `real_outside_rallies` **2 → 0**, serve recall 0.86 → 0.93, serve precision
0.80 → 0.81, `junk_in_rallies` unchanged at 14, shot type 51 → 52, volleys invented 2 → 1,
rally ends within 2 s 10 → 11.

`serve_timing_median_s` rose 0.02 → 0.73 s and that is not a regression: all four of the
operator's strike marks on the acceptance clip are matched now (it was three), and the
recovered one is 0.73 s early, which at n = 4 lands in the middle. A median over four
samples is a poor summary — read it with the recall figure beside it.

## RALLY END — three of the operator's four rules are noise; one is not (2026-08-24)

Scored every detected point-end against the operator's **36 point-ends across three clips**
(`pb_5_minute_outdoor-7`, and indoor courts B and C — the indoor ones were sitting unused in
per-clip `truth.json` until the truth store absorbed them):

| reason | fires | correct | precision |
|---|---|---|---|
| **net** | 20 | **17** | **85%** |
| out | 35 | 10 | 29% |
| not-returned | 6 | 1 | 17% |

The operator's rules are all correct as rules. Only one of them is a question our
measurement can answer. A **net** end asks a *sustained, relative* question — did the ball go
to the floor and stay there — which the reconstruction handles. **out** asks for an *absolute
position at one instant*, which it does not. That is the same split that governs volleys,
bounces and everything else built on `ball_3d.parquet`; it is not specific to rally ends.

**Recomputing the bounce from the PIXEL did not rescue `out`.** The ground homography is
exact at z=0, which is precisely where a bounce is, so projecting the bounce pixel should
beat the reconstructed position. Measured: precision 50% vs 44%, and the gate it feeds still
cost 30 real shots. Rejected. The untrusted ends are still emitted — they are the bar the
next attempt has to clear.

### What the trusted ends buy

Gating shots on `[serve, rally end]` — the operator's ask, *"every shot outside the serve to
rally can be ignored"* — on the acceptance clip:

| | known junk inside rallies | real shots outside |
|---|---|---|
| gate off | 34 | 4 |
| **gate on (net ends only)** | **19** | **6** |
| operator's own point boundaries | 17 | 3 |

**15 junk removed for 2 real shots**, which is most of what perfect boundaries would give.
A MISSED end costs nothing; a FALSE end marks the live play behind it as dead. That asymmetry
is why precision, not recall, is the thing to optimise here — and why taking all ends (44%
precise) measured net negative and got the feature switched off for weeks.

Rally-end timing: median error **6.33s → 1.47s**, 10 of 16 within 2s.

**Method note worth keeping.** This was judged on END PRECISION for weeks because the
regression table had no metric for what the gate is *for*. `junk_in_rallies` and
`real_outside_rallies` exist now. Same lesson as the ground-ball filter that went inert:
build the scorer for the thing you actually want before tuning the thing you can see.

## RALLY END + 3-D FIT — the dependency chain resolved (2026-08-03)

### 3-D projectile fit is NOT the unlock — the operator was right, it is camera-blocked

I proposed the 3-D/parabola fit as the remaining option. **That was wrong** — I cited the
2026-07-19 note recommending it without checking the 2026-07-20 investigation that
SUPERSEDED it (`stages/ball_trajectory/contract.md` "Phase 2 (height) — investigation
result: MARGINAL, precision-floored"). What it actually found:
- **apex height IS robustly recoverable** (3–7 ft) — keep as a lob feature;
- **bounce-vs-volley is at the noise floor** — z=0 vs z≈1–2 ft needs ~0.5 ft resolution,
  the method delivers ±0.5–1 ft. Self-calibration made it WORSE (7/9 → 3/9).
- its own conclusion: *"monocular-precision-limited on this low camera angle… highest
  real leverage is (a) better INPUTS or (b) an OPERATOR camera-angle change — not more
  trajectory math."*

**Do not spend on 3-D height without a second camera / higher mount.**

### The rally-end RULE is correct; our INPUTS cannot support it

Operator's definition (2026-08-03) — a rally ends on **two bounces**, a **first bounce
outside the court**, or a **net hit**; plus the serve exception (a serve must land in the
diagonal service box, kitchen line to baseline, else the rally is just the bad serve).
Rarely a catch. This is right, and it is mechanically checkable. Two things blocked it:

**1. A double bounce was IMPOSSIBLE to detect by construction.**
`MAX_BOUNCES_PER_INTERVAL = 1` — its comment dismisses the second bounce as *"after the
rally is already over"*, when that second bounce IS the rally-end signal. At cap 1 we found
**zero** double bounces. At cap 2, 25 appear and 13/14 rallies get a candidate.

**2. But those candidates are not real double bounces.** Applying physics — a genuine 2nd
bounce follows the 1st in ~0.4–1.0 s, on the SAME side, a few feet along:

| | candidates | median gap | same side | physically plausible |
|---|---|---|---|---|
| current recall (71%) | 30 | **2.17 s** | 12/30 | **1** → covers 1/14 rallies |
| with far-side fix | 31 | **1.35 s** | 14/31 | **6** → covers 4/14 rallies |

**Root cause: a bounce pair with "no shot between" usually means we MISSED the shot
between them.** At 71% shot recall that happens constantly. Double-bounce detection is
therefore gated on SHOT RECALL, not on bounce tuning.

### The chain, and why the far-side fix is upstream of everything

```
far-side player retention  ->  shot recall (recovers 4/4 opponent misses)
        -> double-bounce detection becomes physical (1 -> 6 plausible)
        -> rally END locatable
        -> between-point balls FLAGGABLE by rally structure
        -> clean stats + serve anchoring
```
Each step measurably improves the next. The intermediate state looks WORSE on raw counts
(shots 83 -> 121 vs truth 98) because between-point balls are counted until flagging works
— which is exactly the bootstrap the operator anticipated when he said to fix false
positives first, then recall. The order has to invert here: **recall unlocks the machinery
that makes flagging possible.**

## BETWEEN-POINT BALLS — the operator's two pose hypotheses TESTED AND REJECTED (2026-08-03)

A second, ENRICHED operator labelling round (22 isolated candidates from 1:45 on;
`_labeling/labels.csv`) gave 11 real / 11 junk — 50% junk, so the enrichment worked.
The composition kills both hypotheses, for reasons that are physical, not statistical.

### What the between-point balls actually ARE

| delivery | n | consequence |
|---|---|---|
| **hit WITH A PADDLE** | **7** | hypothesis (a) — thrown-leaves-the-hand — cannot catch these; they *are* paddle shots |
| thrown (or possibly) | 2 | (a) applies, but this is the minority |
| **ADJACENT-COURT ball** | **3** | a completely different failure — see below |

### Hypothesis (a) — thrown ball leaves the HAND, not the paddle
Dead on arrival: **7 of 11 feeds are struck with a paddle** ("opponent picked up ball and
hit it to user/partner with paddle", "partner picks up and hits ball with paddle").
A hand-vs-paddle test cannot separate a paddle-fed ball from a paddle-struck shot.

### Hypothesis (b) — the RECEIVER catches it or does not swing through
Tested two ways on the combined label set (32 real / 20 junk):

| measure | REAL | JUNK | verdict |
|---|---|---|---|
| receiver wrist displacement | med 14.3 | med 12.7 | overlapping |
| receiver **FORWARD** swing (wrist motion projected on the outgoing ball direction — the operator's actual wording, "little forward motion") | med 9.00 | med 5.74 | overlapping; **every threshold costs as many real shots as it catches junk** (e.g. <3 catches 4/12 junk, costs 4/21 real) |

**The physical reason it cannot work:** a real **block, reset or dink IS a non-swing
shot.** "Didn't swing through" does not distinguish a feed from a soft defensive shot —
the two are the same gesture. This is a property of the sport, not of our pose data
(coverage is 83% at the needed frames, so data is not the limit).

### What the labels DID surface

1. **ADJACENT-COURT contamination is live: 3 of 11 junk.** *"a ball being picked up and hit
   with paddle by a player from the NEXT COURT — their ball rolled onto our court"*,
   *"looks like shot on other court"*. Known issue C1, still biting.
2. **5 near/far ATTRIBUTION errors** — we say user/partner (near), the operator says
   opponent. Independent confirmation of the far-side gap that the play-envelope fix
   addresses (it recovers 4 of 4 opponent misses).
3. **4 ball-track failures flagged by eye:** *"circle is on my paddle handle, but I don't
   have the ball"*, *"no circle around anything during this segment"*, *"circle jumps
   around a bit"*. Stage 4 errors visible to the operator.

### Conclusion — stop trying to classify feeds PER SHOT

A paddle-fed between-point ball is, physically, a real paddle shot. The only thing making
it "not a shot" is GAME CONTEXT (it happens between points). Three independent per-shot
signals have now failed on it (ground projection catches only the ground-level subset;
hand-vs-paddle; receiver swing). **Recommended: stop deleting them and FLAG them instead**
— `is_between_point`, excluded from statistics but retained and rendered, per the original
`SERVE_ANCHORED_RALLIES_DESIGN` §2c ("logged, never silently removed"). That converts an
unsolved per-shot classification problem into a rally-segmentation one, where the operator's
dead-time framing applies.

## RECALL MEASURED (2026-08-03) — and it is the FAR SIDE

The operator marked every shot in a continuous 0:40–1:45 render (`label_shots.py
--continuous`, `_labeling/missed_shots.csv`). First direct recall measurement:

| | |
|---|---|
| real shots in span | 21 |
| detected | 15 → **recall 71%** |
| precision | 15/17 → **88%** (was 70% before the ground-ball filter) |

**4 of the 6 misses are OPPONENTS.** Testing the stashed Stage 2.5 play-envelope fix
against them: **it recovers 4 of 4 opponent misses** (0:47 serve, 1:08 dink, 1:20
return, 1:38 volley drive). The two it does not recover are near-side (1:34 user serve,
1:38 partner lob). **This validates the day-one far-side diagnosis with per-shot truth.**

**Why it still cannot land:** clip-wide it takes shots 83 → 121 (truth 98), mean error
25.6% → 42.4%, and it re-introduces the 0:55 ball the operator identified as *"the
opponent throwing it back to my partner"*. The residual junk class is **airborne
between-point balls (throws / feeds)**, which the ground-projection filter cannot catch
by construction — a thrown ball really is in flight.

### Signals tested for THROWN/FED balls — all UNDERPOWERED, none adopted

Operator's hypothesis (2026-08-03), and a good one: *(a)* a thrown ball leaves the HAND,
not the paddle; *(b)* the RECEIVER handles a fed ball differently — they catch it, or do
not swing through. He explicitly warned that raw speed will not work ("a fed ball can be
thrown quickly and a paddle shot can be slow"), and the data confirms that warning.

| signal | real | junk | verdict |
|---|---|---|---|
| ball-to-wrist px at contact | med 77 | med 148 | **reversed** overall (rolling balls sit far from anyone). The 3 actual THROWS read 13/50/65 — closer than real, i.e. directionally right for (a), but **n=3** |
| hitter wrist speed px/f | 1.4–36.4 | 2.3–56.1 | fully overlapping, exactly as the operator predicted |
| **receiver swing-through** (b) | med **14.3** | med **15.3** | does not separate — but only **5** junk had usable pose |
| ball comes to rest afterwards | med 125 px | med 269 px | **reversed** — junk travels MORE (rolling/fed balls keep going). My idea, not the operator's; his (b) is about the receiver's POSE, not the ball |

**Pose coverage is NOT the blocker:** wrist data exists at ±6 frames for **83%** of shots
(user 100%, partner 81%, opp_a 72%, opp_b 75%). A pose-based test is measurable; we
simply lack labelled throws to validate one.

**Status: not enough labelled feeds (3–5) to separate signal from noise.** Do not build
on this sample — that is precisely how the in-play filter got through, and the operator's
labels later showed it killed 9 of 21 real shots. Next step is a small ENRICHED labelling
set (isolated shots in the unlabelled region, ~40% junk) rather than labelling everything.

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

---

## FUTURE CAPABILITY — DETECT THE PADDLE (deferred, 2026-09-05)

**Why it is on the list.** Junk detections are the largest remaining accuracy cost: they
inflate every count in the report and every shot type. The operator, on why he can spot
them instantly: *"I clearly see the shot before and after and there is noone hitting the
ball on the not-a-shot between those correct shots."* He is reading a paddle. We are not —
pose gives wrists, shoulders and ankles, and **there is no paddle landmark anywhere in the
pipeline**, so "did the ball touch a paddle" has no direct measurement to make.

Seven substitutes have been measured against the operator's own adjudications and all sit
at roughly a one-for-one trade of real play for junk:

| substitute | junk cut : real lost |
|---|---|
| ball-to-wrist distance, absolute px | 1.1 |
| ball-to-wrist distance, in body-heights | 0.6–1.0 |
| ball direction change at the contact | 1.1–1.3 |
| impact confidence | 1.0 |
| ball height at the contact | 1.2 (and junk sits HIGHER than real play) |
| a bounce coincident with the contact | inert — Stage 5.5 suppresses these by construction |
| the ball retracing its incoming line | 0.1–0.4, worse than chance |
| wrist motion (did they swing) | strong on the pre-filter population, nothing left after it |

Proximity cannot work **in principle**, and the numbers say so plainly: junk detections sit
a median 0.171 body-heights from the wrist against 0.205 for real shots — the junk is
CLOSER. A ball passing near a player is exactly what manufactures the false positive, so
the cue that would reject it is the cue that created it.

**It is feasible.** A paddle is about 0.23 body-heights long, which in this footage is:

| clip | near-side paddle | far-side paddle | ball, for scale |
|---|---|---|---|
| pb_5_minute_outdoor-12 | ~86 px | ~35 px | 6–14 px |
| pb_3_min_indoor_1_court_c | ~162 px | ~65 px | 10–26 px |

So a paddle is roughly **6x the size of the ball**, and TrackNet already locates the ball to
a 4.9 px median. Resolution is not the obstacle.

**What it would cost.** A second detector (paddle bounding boxes) and the labelled data to
train it — which is different work from the shot review the operator already does, since it
needs boxes rather than timestamps. Inference is added to a pipeline where build_ball_3d is
already ~98% of local post-processing, so throughput has to be part of the design.

**What it would buy.** A direct answer to the one question every substitute is trying to
approximate, and with it the residue of junk that no ball-side rule can reach. It would also
give the swing a rigid object to track, which the wrist alone does not provide at this
distance.

**Do it as its own stage, after the current accuracy work settles.** Note that the
association radius today is 0.5 body-heights (ASSOC_BBOX_HEIGHT_FRAC), more than twice a
paddle's actual reach — deliberately generous to absorb tracking error. Tightening it toward
real paddle reach was measured as part of the table above and does not pay on its own; it
would only pay alongside an actual paddle position.

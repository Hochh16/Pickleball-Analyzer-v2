# Design Review — the counting layer (2026-08-02)

**Scope:** why single-stage fixes keep breaking other stages, whether that is fundamental,
and what to do about it. Commissioned by the operator after the 2026-08-01 session ended
with the pipeline measurably worse (mean error across the acceptance counts 26.9% → 45.9%).

**Verdict up front:** the problem is real, it is structural, and it is **fixable**. It is
not a sensing limit and not a vision-model limit. It is confined to the *labelling* layer —
Stages 5, 5.5, 6 and 7 — which currently exchange **derived labels instead of physical
events**, so each stage's tuning silently encodes the other stages' errors.

---

## 0. What this review is grounded in

Everything below rests on measurement, not intuition. The key input is the **recall census**
(`tools/recall_census.py`, commit `a0024eb`), run because a redesign built on a sensing limit
would be worthless:

| | result |
|---|---|
| Operator serves with **no ball signal** | **0 / 14** |
| Operator serves where the **ball is present but we produce no shot / no serve label** | **6 / 14** |
| Ball visible **during rally play** | **84.0%** (67.1% during dead time) |
| Ball gaps ≥ 0.5 s **inside rally play** | **3**, totalling **2.5 s** across a 5-minute clip |

**The ball is not the ceiling.** At most a handful of contacts in the whole clip are
physically unrecoverable. Every other error is ours to fix. That is what justifies a design
change rather than a capture change.

Second input — the same three configurations scored end to end
(`tools/score_acceptance.py`):

| config | serves correct (±1 s) | shots (truth 98) | mean error, 14 counts |
|---|---|---|---|
| baseline (`main`) | 7 / 14 | 99 | **26.9%** |
| + far-side player retention | **8 / 14** | 136 | 56.5% |
| + far-side + in-play filter | 4 / 14 | 99 | 45.9% |

Note the shape of that table: **no configuration is good at both.** That is the symptom this
review explains.

---

## 1. The finding — stages exchange derived labels, not events

Verified in code, not inferred:

**(a) The bounce set is a function of the shot set.**
`detect_bounces.py`: `MAX_BOUNCES_PER_INTERVAL = 1`, and every bounce is bucketed by the
shot pair it falls between (`between_shots = [prev_id, next_id]`). Intervals are defined by
shots, so **changing the shot list changes which bounces can exist.** Measured on 08-01:
deleting 37 shots deleted **14 bounces**.

**(b) The volley set is a function of the shot set — twice.**
`classify_shots.py`: *"a shot is a volley iff the ball did NOT bounce since the previous
shot."* The primary signal scans the trajectory **between two shot frames**; the fallback
counts `bounces_between[(prev_shot_id, shot_id)]`. Both windows are defined by shots. So
**volleys move when shot detection moves, even though no volley changed.** Measured: those
same 37 deletions turned 23 volleys into 35.

**(c) Rallies and serves are mutually defined.**
`segment_rallies.py` derives rallies from `is_serve` plus gaps. `detect_shots.structure_points`
derives `is_serve` from point boundaries — and a point boundary is "a shot nobody returned,
followed by dead time." An unreturned **spurious** serve satisfies that trivially, so a false
serve manufactures the very evidence that admits it. The operator identified this circularity
independently: *"assuming you know when a rally is over and the next serve. but how will you
know the next serve?"*

**(d) The system already reports that these layers disagree.**
The ledger calls `shots = volleys + bounces` "the single best self-check in the system." On
`main` it reads **99 vs 23+71 = 94 — off by 5.** It has been quietly failing because those
three numbers come from three independent detectors rather than being three *views of one
thing*.

---

## 2. Why this produces "every fix breaks something else"

Each stage was tuned against the acceptance counts **with the other stages held fixed**. The
code says so plainly — `reject_same_track_repeats`: *"Measured on pb_5_minute_outdoor-2:
108 → 99 shots (operator truth 98)."*

That filter collapses two same-player impacts with nobody hitting in between, on the sound
logic that a player cannot legally strike twice in a row. **But far-side opponents were being
discarded upstream**, so real cross-court exchanges *looked* like same-player repeats and were
collapsed. The filter was silently compensating for a missing-opponent bug.

Fix the far side, and the compensation becomes an over-correction. This is the general
mechanism, and it is why yesterday looked like whack-a-mole:

> **Every threshold in the counting layer is calibrated against the other stages' current
> errors. Correcting any one stage invalidates the tuning of the rest.**

That is a design property, not a run of bad luck — and it will recur on *any* future fix
until the coupling is removed. It also predicts something worth stating: the counting layer
will resist incremental improvement indefinitely. Each correct fix will look like a
regression.

---

## 3. The alternative — one ball-event timeline

**Physical model.** The ball's path is a sequence of flight arcs separated by exactly two
kinds of event, plus dead time:

- **CONTACT** — a paddle strikes the ball (impulsive direction change *at a player*)
- **BOUNCE** — the ball hits the ground (impulsive vertical reversal *away from players*)

Stage 5 and Stage 5.5 already detect both from the *same* impulse signal, with opposite
proximity rules — they are two halves of one detector that were split into two stages and
then tuned against each other.

**Proposal.** Build **one ordered event timeline** — `(frame, type, position, actor,
confidence)` — validated once against operator truth. Then every statistic becomes a **view**
over it, not a separate detector:

| statistic | view over the timeline |
|---|---|
| shots | CONTACT events |
| bounces | BOUNCE events |
| **volleys** | CONTACT with no BOUNCE since the previous CONTACT |
| rallies | runs of events separated by dead time |
| serves | first CONTACT of a run |
| returns | second CONTACT of a run |
| dink / drive / drop | CONTACT typed by the *next* BOUNCE's position + hitter zone |
| in-play vs feed | events inside a run vs isolated events |

**The decisive property:** `shots = volleys + bounces` becomes **true by construction**. It
is no longer a check that can fail — it is the definition. Today it is an aspiration that is
off by 5.

Two further consequences worth naming:

- **The circularity dissolves.** Runs are found from event timing alone; serves are then read
  off the runs. Neither needs the other. The operator's question — *how will you know the
  next serve* — has a non-circular answer: it is the first contact of the next run.
- **One place to be right, one place to validate.** Today an error in shot detection surfaces
  as wrong volleys, wrong dinks and wrong rallies, in three different stages, each of which
  invites its own compensating patch.

---

## 4. What this actually buys, from the census

The 6 algorithm-limited serves are all this class:

- **0:47, 4:05, 5:01** — far-side serves. The contact exists in the ball track; the *actor*
  was discarded upstream (Stage 2.5 play-envelope issue, fix understood and measured).
- **1:04, 1:33** — the contact is missed because the player's **pre-serve ball handling**
  ~3.5 s earlier is detected instead (KNOWN_ISSUES). On a timeline, handling and serve are
  both contacts, and the *run* structure decides which opens the point — rather than a
  same-track collapse rule guessing.
- **2:32, 4:05** — a contact is found but never labelled a serve, because the mutual
  constraint rejected it. On a timeline this is not a decision at all: it is the first contact
  of a run.

None of these need better vision. All are reachable.

---

## 5. Against the three critical focus areas

**1 — Accuracy and completeness.** Directly addressed: identities hold by construction, one
validation surface, and the compensating-tuning trap is removed. On *completeness*, every stat
previously judged camera-feasible is a view over the timeline: counts (shots, serves, returns,
rallies, dinks), the volley count via the identity, drop-vs-drive choice, FH/BH from pose,
positioning and kitchen time from tracking. The camera-blocked set is unchanged and unaffected
by this design — ball height, true speed, spin, dink quality, shoulder rotation, split-step
all remain out (three independent height-free methods already tested and defeated; do not
retry).

**2 — Multi-video accumulation.** Today's per-clip counts are a poor substrate: they carry no
per-event record, so they cannot be re-aggregated, re-weighted or audited after the fact. A
timeline of confidence-carrying events is the natural unit to accumulate across videos, and it
makes thin per-category samples (dink 6, serve 4 in five minutes) poolable. **Still required
and not solved by this design:** stable cross-video player identity (F28). That is a separate
piece of work and should not be conflated with it.

**3 — Cross-court, indoor and outdoor.** This design is *neutral* on generalization, and it is
important not to oversell it: the binding constraint there is the **ball detector** (court2
0.69 / indoor 0.61 against a 0.80 bar), which is **data-limited** — the operator has 3
unlabeled indoor clips, and that is the highest-value unblocked input to the project. What the
design *does* contribute is a discipline the current code already mostly follows and must keep:
**derive every threshold from calibration geometry, never per-court constants.** The far-kitchen
line used in yesterday's work was computed as `length/2 + kitchen_depth`, not hardcoded — that
is the pattern.

---

## 6. Migration cost — what changes and what does not

**Unchanged:** Stages 1 (calibrate), 2 (track players), 2.5 (roles), 3 (pose), 4 (ball). The
entire vision layer is untouched. Stages 8–11 (metrics, rating, plan, render) keep their shape
and schemas.

**Merged:** Stages 5, 5.5, 6, 7 become a **timeline builder** plus a thin set of views. Most of
the existing detection logic is reusable — the impulse detector, the player association, the
bounce descent-peak detector, the landing-based type rules are all sound in isolation. What is
discarded is the *inter-stage compensation*: the same-track collapse, the volley-by-absence
inference, the serve/point mutual constraint.

**Honest costs and risks:**
- This is a refactor of the labelling layer, not new sensing. **It cannot exceed the census
  ceiling** — expect a handful of unrecoverable contacts to remain.
- Contact detection is still the same underlying signal; the timeline does not make a missed
  impulse appear. It changes how errors *propagate*, not how many are made at the source.
- There is real work in re-deriving shot *type* on the new structure, and the type rules are
  the least-validated part of the current system.
- **What would falsify this:** if, after building the timeline for one rally with known truth,
  the identity holds but the counts are still wrong, the problem is contact detection rather
  than coupling — and the effort should go to the impulse detector instead.

---

## 7. Recommended sequence

1. **Prove it on one rally** (small, cheap, falsifiable). Build the timeline for rally 10,
   where operator per-shot truth already exists (12 shots: serve → baseline drives → kitchen
   dink exchange). Show shots, bounces and volleys all fall out consistently with the identity
   holding. **Gate: if it does not reproduce the operator's 12, stop and reconsider.**
2. **Land the far-side fix on its own.** It is measured, it is the right direction (serves
   7→8), and it is independent of the redesign. **Discard the in-play filter** — the census
   shows it is the harmful half (serves 4/14).
3. **Then** build the timeline for the full clip and cut the views over to it, validating
   against all 14 acceptance counts with `tools/score_acceptance.py` at every step.
4. **In parallel, independent:** indoor clip labeling (Focus Area 3). Genuinely decoupled from
   all of the above.

**Open questions that do not block step 1:** the 99-vs-98 shot discrepancy (may be a feed we
count — resolve by listing the 99); feed vs faulted serve (deferred by the operator, no fault
exists in this clip to test against).

---

## 8. What this review does not claim

- It does not claim the redesign reaches 98/14. The census says the *ceiling* allows it; it
  does not promise the contact detector achieves it.
- It does not address shot **type** accuracy (dink 36 vs 18 is the largest single error on
  `main` and is a *type* problem, not a coupling problem — though the coupling makes it harder
  to see, because dink type depends on landing, which depends on the bounce set, which depends
  on the shot set).
- It does not solve cross-video identity, or cross-venue ball recall. Both remain open and
  both are prerequisites for focus areas 2 and 3 respectively.

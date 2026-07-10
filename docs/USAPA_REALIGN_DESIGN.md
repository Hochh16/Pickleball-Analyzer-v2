# USAPA Realign — Stage 9 rewrite design (scoped 2026-07-09)

Design for **build-program step 2: REALIGN** — rewrite Stage 9's 6 homegrown
dimensions to USA Pickleball's **7 official rating categories**. Scoped with the
operator; two decisions locked (below). Implement against this doc.
Source of the standard: `docs/PRODUCT_VISION.md` + the USAPA level pages.

## Operator decisions (2026-07-09)

1. **Full 7-category structure, hard-gated.** Build all 7 categories now. Strategy
   scores from real position data; the other 6 score from what exists but are
   confidence-gated and mostly surface as "developing / not assessable yet." The
   confidence-weighted estimate leans on Strategy automatically.
2. **Single headline estimate + band, heavily caveated.** Keep one
   confidence-weighted USAPA estimate + band + range (as today), with a loud new
   caveat that it currently rests almost entirely on positioning (Strategy).

## The data reality (why 6 of 7 are gated)

For the **user**, only positioning is high-confidence. Everything shot-based rests
on ~5 detected user shots at confidence 0.15–0.23 (pb_2min):

| Category | Current source metric | Today | Note |
|---|---|---|---|
| **Strategy** | zones + both-at-kitchen + movement (+ unforced errors) | ● real, conf ~1.0 | the anchor |
| **Third Shot** | `match.third_shot.drop_rate` | ◐ conf 0.23 | **match-level, not per-user** |
| **Dink** | `by_shot_type.dink` + rally sustain | ◐/○ conf ~0.16 | tiny sample |
| **Volley** | `shot_mix.volley` (rate) | ◐ conf 0.21 | volley flag heuristic |
| **Serve/Return** | `serve.serve_fault_rate` | ○ conf 0 | **n_serves=0 for user** |
| **Forehand** | `by_stroke_side.forehand` | ○ conf 0.20 | count only, no pace/depth |
| **Backhand** | `by_stroke_side.backhand` | ○ conf 0.20 | count only, no pace/depth |

The realign is a **legitimacy/naming + presentation** win, not a new-signal win.
New signal comes in the later ADD step (unblocked by: ball recall C4, serve
detection C3, stroke-side F16, court-plane speed F7, landing depth C4, per-user
third-shot attribution).

## Target output — 7 categories

Replace `WEIGHTS`/`REAL_DIMS`/`DIM_DRIVERS` + the six `score_*` fns with seven
categories. Each dimension entry keeps the existing shape (`name`,
`subscore_level`, `weight`, `confidence`, `limited_by`, `data_source`,
`driver_metrics`) plus a new **`coverage_status`** ∈ {`measured`, `partial`,
`not_assessable`} derived from confidence (measured ≥0.5, partial ≥ASSESS_FLOOR,
else not_assessable). Near-zero-confidence categories still route to Stage 10's
`developing_capability.not_assessable_now` (existing mechanism).

### Category → scorer → drivers

1. **Strategy** (anchor; ● real). Reuse `score_net_play` + `score_movement` and
   fold in the unforced-error rate. Drivers: `kitchen_time_frac`,
   `both_at_kitchen_frac`, `transition_time_frac`, `distance_ft_per_min`,
   `unforced_error_rate` (when available). Confidence = min over position/team/errors.
2. **Third Shot** (◐). Reuse the `drop_rate` half of `score_shot_skill`. Driver:
   `third_shot_drop_rate` (+ soft/power mix). **Flag: match-level until per-user
   third-shot attribution lands.** Confidence from `match.third_shot`.
3. **Dink** (◐/○). New small scorer from `by_shot_type.dink` count + a rally-sustain
   proxy (`rally_length` / dink-rally length when available). Low confidence.
4. **Volley** (◐). New small scorer from `shot_mix.volley.volley_rate`. Low confidence.
5. **Serve/Return** (○). Reuse `score_serve` (`serve_fault_rate`). Return not
   separately detected → note. Near-zero confidence when `n_serves=0` → not_assessable.
6. **Forehand** (○). New: `by_stroke_side.forehand` count/consistency. No pace/depth
   → near-zero confidence → not_assessable.
7. **Backhand** (○). New: `by_stroke_side.backhand` count (+ avoids-BH proxy later).
   Near-zero confidence → not_assessable.

### Weights (heuristic, uncalibrated — tunable)

Encode rough skill importance at rec levels; the confidence-weighting already
prevents low-confidence categories from inflating the estimate, so weights mainly
shape the reported coverage/confidence. Proposed (sum 1.00):
Strategy 0.20 · Third Shot 0.18 · Dink 0.15 · Volley 0.13 · Serve/Return 0.12 ·
Forehand 0.12 · Backhand 0.10. Keep the existing "UNCALIBRATED heuristics" warning.

### Headline + caveat

Keep the confidence-weighted estimate + band + range. Add a loud warning +
a `reliability.assessable_categories` count, e.g. *"This USAPA estimate currently
rests almost entirely on court positioning (Strategy); 6 of 7 categories are not
yet reliably measured (need ball recall / serve detection / stroke-side / shot
speed)."*

## Downstream impact (coordinated — do NOT ship Stage 9 alone)

Stage 10 (`plan_improvement`) keys `WHY`, `DRILLS` selection, `UNMEASURED_REASON`,
and the `finding_and_drills` branches on the **dimension names**. Rewriting Stage 9
to 7 categories requires re-keying all of those to the 7 category names (findings +
why + drills per category), or the plan breaks. Sequence: **Stage 9 rewrite → Stage
10 re-key → re-run 9→10→11 on pb_2min → operator-validate the rendered report.**
Stage 11/render reads dimension names too — check the HUD/report labels.

Schema: `rating.json` dimensions list changes names (additive `coverage_status`);
bump Stage 9 `STAGE_VERSION` (→0.4.0) and Stage 10 accordingly. The test fixtures
(`test_rate.py`, `test_plan_improvement.py`) assert on the 6 old names / DIM_NAMES —
update them to the 7.

## Suggested implementation order

1. Stage 9: new 7-category `compute_rating` (scorers + weights + coverage_status +
   caveat), update `test_rate.py` (monotonicity per new scorer, 7-name checks).
2. Stage 10: re-key WHY/DRILLS/UNMEASURED_REASON/finding branches to the 7; update
   `test_plan_improvement.py` DIM_NAMES.
3. Re-run 8→9→10→11 on pb_2min; render; operator-validate (Strategy is the only
   category that should read high-confidence).
4. Update `stages/rate/contract.md` + `stages/plan_improvement/contract.md` +
   KNOWN_ISSUES (close the "dimensions don't match USAPA" entry) + PRODUCT_VISION
   build-program (mark REALIGN done).

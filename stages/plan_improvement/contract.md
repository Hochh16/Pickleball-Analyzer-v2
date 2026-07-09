# Stage 10 — Plan Improvement

**Status:** DRAFT for review. Turns the Stage 9 `rating.json` (+ Stage 8
`metrics.json` for concrete numbers) into an `improvement_plan.json` for the
**user**: the gap to the next USAPA half-step, a prioritized set of **focus
areas** (each with a data-grounded finding + 1–3 drills/cues from a built-in
USAPA-anchored library), and a forward-looking **developing-capability** section
that scaffolds in the skills not yet measurable — so the plan reaches full
capability once real ball detection (Stage 4/4.5) and the missing-skill metrics
land. Rule-based, same pipeline philosophy (documented mappings, honest
confidence, loud failures).

## Scope decisions (settled with the operator before drafting)

> **DECISION (depth = focus areas + drills/cues).** Per weak dimension: the gap
> to the next level, a short data-grounded finding + rationale, and 1–3 concrete
> drills/coaching cues from a built-in library. No practice scheduling/dosage in
> v1 (no evidence basis on placeholder data).

> **DECISION (include synthetic weaknesses, flagged provisional).** Prioritize
> across ALL dimensions, but mark recommendations derived from synthetic-ball
> dimensions (`error_control`, `shot_skill`, `serve`, `rally_consistency`) as
> `"provisional"` (pending real ball v4), and let real-data weaknesses
> (`net_play`, `movement`) rank as higher-confidence via a mild
> confidence weighting in the priority score. Full plan now, honest about which
> parts are placeholder.

> **DECISION (v0.3.0, 2026-07-07 — gate focus/strengths by PER-DIMENSION
> confidence).** The 06-19 rule gated "provisional" on the coarse `ball_source`
> (synthetic vs real). The first real-ball run exposed the failure mode: with a
> real ball, a dimension at **confidence 0** (a data gap, not a signal — `serve`
> with 0 detected serves, `error_control` with errors undetectable because
> end_reasons are all `unknown`) was rendered as a confident coaching signal —
> "work on your serve," "great error control." Now a REAL dimension whose
> confidence is below `ASSESS_CONF_FLOOR` (0.1) is treated as **not assessable**:
> it is routed OUT of `focus_areas`/`strengths` into
> `developing_capability.not_assessable_now` (with its `limited_by` reason), never
> coached as a weakness or celebrated as a strength. Dimensions above the floor
> carry an honest per-dim confidence label (`high`/`moderate`/`low`, or
> `provisional` on synthetic) instead of a blanket `high`. Priority still scales
> with confidence, so low-but-nonzero dims rank below high-confidence ones.

> **DECISION (forward-looking developing-capability section — operator's
> explicit ask).** The plan accounts for BOTH the currently-measured skills
> (synthetic-flagged) AND the skills to be developed downstream. The skills in
> Stage 9's `skill_coverage.proxy_or_pending` and `not_captured_yet` are emitted
> as a `developing_capability` block — each with what unlocks it, what it will
> assess, and what it will recommend — so when ball v4 + the new metric/pose
> stages land, Stage 10 reaches full capability. v1 does NOT invent
> recommendations for skills it can't yet measure.

## Purpose

Answer "what do I work on next, and how." Stage 10 reads the rating, finds where
the user is furthest below the next USAPA half-step (weighted by how much each
dimension matters), and produces a short, prioritized, actionable plan with
drills — while being explicit that synthetic-derived advice is provisional and
that several real skills aren't measured yet. Feeds Stage 11 (the annotated
report surfaces the plan).

Rule-based — the rating dimensions, a curated drill library, and the skill
coverage map are the inputs; there is no learned recommendation model.

## Place in the architecture

```
rating.json (S9) + metrics.json (S8)
        │
        ▼
   [10] plan_improvement ──► improvement_plan.json
```

Per-video, file-path I/O, standalone CLI:
`python -m stages.plan_improvement.plan_improvement <video_folder>`.

> **DECISION (folder name).** Code + contract live at
> `stages/plan_improvement/` (importable). This contract sits at the numbered
> stub `stages/10_plan_improvement/` for review; on approval it moves to
> `stages/plan_improvement/contract.md` and the stub is deleted.

## Inputs

| File | From | Stage 10 reads |
|---|---|---|
| `rating.json` | Stage 9 | `rating` (estimate, band, confidence), `dimensions` (subscore, weight, confidence, data_source, driver_metrics), `skill_coverage`, `ball_source` |
| `metrics.json` | Stage 8 | optional extra context for findings text (the dimension `driver_metrics` in rating.json already carry the key values; metrics.json is read defensively for any extra detail) |

CLI flags (defaults in Configuration): `--force`, `--log-level`,
`--max-focus-areas`, `--target-band` (override the auto next-half-step).

**Degradation:** `metrics.json` missing → warn, proceed on rating.json alone
(findings fall back to the rating's `driver_metrics`). `rating.json`
missing/malformed → fail loudly.

## Output — `improvement_plan.json`

```json
{
  "schema_version": 1,
  "source_rating": "data/test_clip/rating.json",
  "source_metrics": "data/test_clip/metrics.json",
  "ball_source": "synthetic",
  "rated_role": "user",
  "current": {"estimate": 3.69, "band": "3.5", "confidence": 0.545},
  "target": {
    "band": "4.0",
    "level": 4.0,
    "rationale": "Next USAPA half-step. Closing the focus-area gaps below moves the user toward 4.0."
  },
  "focus_areas": [
    {
      "priority": 1,
      "dimension": "net_play",
      "data_source": "real",
      "confidence": "high",
      "current_subscore": 2.78,
      "gap_to_target": 1.22,
      "priority_score": 0.244,
      "finding": "Only 13% of rally time at the kitchen line and 32% in the transition zone; partners are both at the line just 2% of the time.",
      "why_it_matters": "USAPA 3.5–4.0 players win the net: they get to the kitchen line, hold it together, and avoid being caught in the transition zone.",
      "drills": [
        {"name": "Get-to-the-line", "cue": "After every return, sprint to the NVZ line and freeze before the next ball — 'get to the line, then play.'"},
        {"name": "Move as a unit", "cue": "Shadow your partner across the kitchen keeping ~8–10 ft spacing; close the middle together."},
        {"name": "Transition resets", "cue": "From mid-court, reset a hard feed softly into the kitchen, then advance to the line."}
      ],
      "provisional_note": null
    },
    {
      "priority": 2,
      "dimension": "error_control",
      "data_source": "synthetic",
      "confidence": "provisional",
      "current_subscore": 3.64,
      "gap_to_target": 0.36,
      "priority_score": 0.061,
      "finding": "Attributed error rate ≈0.36 per rally. (Forced/unforced split unavailable until real ball.)",
      "why_it_matters": "Cutting unforced errors is the fastest way up the rating ladder.",
      "drills": [
        {"name": "Cooperative dink count", "cue": "Cross-court dink rally to 20 without an error before adding pace."}
      ],
      "provisional_note": "Derived from the synthetic ball; revisit when real ball detection (v4) lands."
    }
  ],
  "strengths": [
    {"dimension": "serve", "current_subscore": 4.2, "data_source": "synthetic",
     "note": "At/above the 4.0 target (provisional — synthetic)."}
  ],
  "developing_capability": {
    "_comment": "Skills not yet fully measured. v1 emits NO recommendations for these; they scaffold in once their data source lands, giving the plan full capability post Stage 4/4.5 + the listed new stages.",
    "proxy_or_pending": [
      {
        "skill": "forced_vs_unforced",
        "unlocked_by": "real ball detection v4 (Stage 8 pending_real_ball)",
        "will_assess": "Split errors into forced vs unforced.",
        "will_recommend": "Targeted consistency vs shot-selection drills based on which dominates."
      }
    ],
    "not_captured_yet": [
      {
        "skill": "return_of_serve",
        "unlocked_by": "new return-quality metric (needs real ball + positioning)",
        "will_assess": "Return depth and whether the user gets to the net behind it.",
        "will_recommend": "Deep-return + split-step-and-advance drills."
      },
      {
        "skill": "footwork_split_step",
        "unlocked_by": "pose-technique stage (Tier-C; mostly real pose data)",
        "will_assess": "Split-step timing and ready-position recovery.",
        "will_recommend": "Split-step timing drills vs opponent contact."
      }
    ],
    "out_of_scope": ["spin", "score_situational_decisions"]
  },
  "reliability": {
    "synthetic_ball": true,
    "n_focus_real": 1,
    "n_focus_provisional": 1,
    "note": "1 of 2 focus areas is provisional (synthetic-ball-derived). The plan's real-data focus areas (positioning/movement) are trustworthy now; the rest firm up at ball v4."
  },
  "operator_considerations": {
    "_comment": "Analysis-reliability notes for the OPERATOR (separate audience, lower priority than coaching). Surfaced only when a real-data limiter bites; empty otherwise (here: synthetic ball -> suppressed).",
    "items": []
  },
  "warnings": [
    "ball_source is 'synthetic': provisional focus areas are derived from PLACEHOLDER ball data — treat as a scaffold until ball detection v4.",
    "Rating + plan thresholds are UNCALIBRATED heuristics (no rated-footage corpus); see KNOWN_ISSUES.md."
  ],
  "params": {"max_focus_areas": 4, "confidence_weight_floor": 0.5},
  "stage_version": "0.2.0",
  "completed_at_utc": "2026-05-29T..."
}

// On a REAL-ball plan with a low-confidence dimension, operator_considerations
// instead reads e.g.:
//   "items": [
//     {"category": "more_data", "limiters": ["sample_size"], "affects": ["serve"],
//      "action": "Record longer sessions, or combine clips across sessions - these assessments currently rest on few rallies."},
//     {"category": "capture_quality", "limiters": ["measurement"], "affects": ["shot_skill"],
//      "action": "Capped by single-camera 2D (no ball height/depth) - a higher-mounted or second camera would improve precision."}
//   ]
```

### Field notes

- **`target`** — the next USAPA half-step above `current.band`, capped at 5.0
  (a 5.0 player gets a "maintain/refine" target). `--target-band` overrides.
- **`focus_areas`** — dimensions whose `subscore_level < target.level`, sorted
  by `priority_score` descending, then `priority` assigned 1..n, capped at
  `max_focus_areas`.
  - `gap_to_target = max(0, target.level − current_subscore)`.
  - `priority_score = gap_to_target · weight · (confidence_weight_floor + (1 −
    confidence_weight_floor) · dim_confidence)`. The confidence term (real ≈
    1.0, synthetic ≈ 0.35) gently lifts real-data weaknesses above provisional
    ones of similar leverage **without burying** a genuinely large synthetic
    gap. `confidence_weight_floor` (default 0.5) bounds how much synthetic
    advice is down-weighted.
  - `data_source` / `confidence` — `"real"`/`"high"` for net_play + movement;
    `"synthetic"`/`"provisional"` for the ball-derived dimensions (becomes
    `"real"`/`"high"` automatically when `ball_source == "real"`, since
    rating.json's `data_source` already flips).
    > **Player coaching stays clean (Foundation #3 decision, David 2026-06-21).**
    > Focus areas carry NO capture/app advice — analysis-reliability actions live
    > in the separate `operator_considerations` block below, a different audience
    > (the operator who records/configures) and lower priority than coaching.
  - `finding` — a plain-language sentence built from the dimension's
    `driver_metrics` (e.g. kitchen %, transition %, error rate). Numbers come
    straight from rating.json so the finding can't drift from the rating.
  - `drills` — 1–3 entries selected from the built-in library for that
    dimension, some chosen conditionally on the driver values (e.g. high
    transition time → include the "Transition resets" drill).
  - `provisional_note` — non-null only for synthetic-derived focus areas.
- **`strengths`** — dimensions already `≥ target.level`, listed briefly (with a
  provisional tag if synthetic), so the plan acknowledges what's working.
- **`developing_capability`** — built from rating.json's
  `skill_coverage.proxy_or_pending` + `not_captured_yet`, each annotated from a
  static descriptor table (`unlocked_by` / `will_assess` / `will_recommend`).
  `out_of_scope` passes through. **This is the forward-looking scaffold** — v1
  produces zero recommendations here; it documents what completes the plan.
- **`reliability`** — counts of real vs provisional focus areas + a note;
  makes the placeholder dependence machine-readable for Stage 11 / the UI.
- **`operator_considerations`** — **NEW (Foundation #3).** Analysis-reliability
  notes for the **OPERATOR** (who records / configures the capture), kept
  SEPARATE from and lower-priority than the player coaching, and intended for a
  possibly-different audience. `items` aggregates the limiters that materially
  bite, each `{category, limiters, affects, action}`:
  - `category` — `more_data` (limiter `sample_size` → record longer / combine
    clips) or `capture_quality` (`measurement`/`known_limit` → higher-mounted /
    second camera; also a `detection_floor` home, though counts rarely bite).
  - `affects` — the dimension names whose reliability the limiter caps.
  - `action` — the operator-facing instruction (descriptive about the *capture*,
    never coaching-imperative).
  **Surfaced only when a limiter actually bites:** a limiter contributes only for
  a **real-data** dimension whose `confidence < OPERATOR_CONF_FLOOR` (0.6). On the
  synthetic ball, ball-derived dims are `data_source: synthetic` and excluded
  (their gated-low confidence is the placeholder ball, already warned, not an
  operator-fixable limiter), so `items` is empty — the UI hides the section. The
  block ties to the two capture-side future levers in KNOWN_ISSUES (throughput
  for more footage; higher/2nd camera for speed/precision).

There is **no separate `.meta.json`** — metadata lives inside the file.

## Built-in drill library (USAPA-anchored; documented constants)

A small curated `dimension → [drill]` table (each drill = `{name, cue}`),
plus light conditional selection on driver values. v1 content (tunable):

- **net_play:** Get-to-the-line; Move as a unit (if `both_at_kitchen_frac`
  low); Transition resets (if `transition_time_frac` high).
- **movement:** Split-step + recover; Court-coverage ladder (if coverage low).
- **error_control:** Cooperative dink count; Reset under pressure.
- **shot_skill:** Third-shot-drop reps (if `drop_rate` low); Shot-variety
  ladder (if `shot_variety` low); Soft-game targets.
- **serve:** Deep-serve targets (if `serve_fault_rate` high); Pre-serve routine.
- **rally_consistency:** Sustained-rally game; Volley exchanges (if
  `volley_rate` low).

> The library is intentionally modest and rule-selected — not a content engine.
> It expands as dimensions are added (developing_capability) and as real-data
> calibration tells us which drills move which metric.

## Method

1. **Load + validate.** Read `rating.json` (`schema_version == 1`; fail loudly
   otherwise) and `metrics.json` if present (`schema_version == 2` — Stage 8 v2;
   read defensively, not consumed for any number). Pull `dimensions`,
   `skill_coverage`, `ball_source`, `rating`.
2. **Target.** `target.level` = next half-step above `current.band` (cap 5.0),
   or `--target-band`.
3. **Focus areas.** For each dimension with `subscore < target.level`: compute
   `gap_to_target`, `priority_score`; build `finding` from `driver_metrics`;
   select drills; set `data_source`/`confidence`/`provisional_note` (coaching
   only — no capture/app fields). Sort by `priority_score` desc, cap at
   `max_focus_areas`, assign `priority`.
3b. **Operator considerations.** While iterating dimensions, collect each
   **real-data** dim whose `confidence < OPERATOR_CONF_FLOOR`, bucket its
   `limited_by` into an operator `category` (`more_data` / `capture_quality`),
   and emit one item per category present (empty when none bite).
4. **Strengths.** Dimensions `≥ target.level` → brief list.
5. **Developing capability.** Map `skill_coverage.proxy_or_pending` +
   `not_captured_yet` through the static descriptor table; pass `out_of_scope`.
6. **Reliability + warnings.** Count real vs provisional focus areas; loud
   synthetic warning + uncalibrated warning; note if the top priority is
   provisional.
7. **Write** `improvement_plan.json` (refuse to overwrite without `--force`).

## Defenses against placeholder / bad data

- **Propagates `ball_source`** + a loud warning; per-focus-area
  `provisional` flags + `reliability` counts make the synthetic dependence
  explicit. The plan is action-prompting, so the placeholder caveat is
  front-and-center.
- **No recommendations for unmeasured skills** — `developing_capability` only
  documents them; it never fabricates a drill for a skill with no data.
- **Uncalibrated-thresholds warning** always present (inherited from Stage 9).
- **Empty plan** (user already ≥ target on every dimension, or degraded input)
  → `focus_areas: []` + a note, valid file, no crash.
- **`rating.json` schema mismatch / missing** → fail loudly naming the file.
- **Output exists without `--force`** → `FileExistsError`.

## Edge cases

- **All dimensions ≥ target** (or a 5.0 player) → empty `focus_areas`, a
  "maintain/refine" target note, strengths populated.
- **A dimension at very low confidence** (tiny clip) → still eligible as a
  focus area, but its `priority_score` is reduced by the confidence term and
  it's flagged; if synthetic, also `provisional`.
- **Top priority is provisional** → add a warning so the user knows the #1 item
  rests on placeholder data.
- **Missing `driver_metrics` value** → finding uses a generic phrasing for that
  dimension (no fabricated number).
- **Required input missing/malformed** → fail loudly naming the file.

## Configuration (defaults; tuned against smoke-test directional checks only)

```python
MAX_FOCUS_AREAS         = 4
CONFIDENCE_WEIGHT_FLOOR  = 0.5   # min multiplier applied to synthetic leverage
TARGET_CAP              = 5.0    # USAPA top band
USAPA_BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# DRILL_LIBRARY and DEVELOPING_DESCRIPTORS are documented constants in code.
```

## Smoke test

`stages/plan_improvement/test_plan_improvement.py`, against `data/test_clip/`.
No ground-truth plan exists, so the test gates on **schema + internal
consistency + directional behavior** (mirrors Stage 9), via the end-to-end
chain (synth → … → S8 → S9 → S10) plus pure-function checks on synthesized
ratings.

1. **Schema.** `improvement_plan.json` parses, `schema_version == 1`, all
   documented top keys; `current`/`target` valid; `target.band` is the next
   half-step above `current.band` (or capped 5.0).
2. **Focus-area correctness.** Every focus area's `dimension` has
   `current_subscore < target.level`; no dimension `≥ target.level` appears in
   `focus_areas`; `focus_areas` sorted by `priority_score` desc with contiguous
   `priority` 1..n; count ≤ `max_focus_areas`; each has 1–3 drills.
   `strengths` are exactly the dimensions `≥ target.level`.
3. **Provisional flags.** Every synthetic-`data_source` focus area has
   `confidence == "provisional"` + non-null `provisional_note`; every real one
   has `confidence == "high"` + null note.
3b. **Operator considerations.** Player focus areas carry NO operator/capture
   fields. `operator_considerations.items` is empty on the synthetic ball
   (suppressed); on a real-ball rating with a low-confidence dimension it fires
   the right `category` with the canonical `action`, and a high-confidence real
   dim does NOT trigger.
4. **Developing capability.** `developing_capability.proxy_or_pending` +
   `not_captured_yet` skill names exactly match rating.json's
   `skill_coverage`; each entry has `unlocked_by` / `will_assess` /
   `will_recommend`; `out_of_scope` matches. No skill in `developing_capability`
   also appears as a `focus_areas` dimension.
5. **Reliability + propagation.** `synthetic_ball == true`; `n_focus_real +
   n_focus_provisional == len(focus_areas)`; placeholder + uncalibrated
   warnings present.
6. **Directional (pure functions).** (a) Lowering one dimension's subscore (via
   a synthesized rating) increases its `gap_to_target` and its `priority_score`,
   and—if it drops below target—adds it to `focus_areas`. (b) A real-data
   weakness ranks above a synthetic weakness of equal gap+weight (confidence
   term). (c) Rating the same dims with `ball_source == "real"` removes the
   provisional flags.
7. **Degradation.** A rating with every dimension ≥ target → empty
   `focus_areas`, populated `strengths`, valid file, no crash.

## Stage version

`0.2.0` (Foundation #3): accepts Stage 8 `metrics.json` `schema_version 2` (the
metrics-present guard was bumped 1 → 2; metrics content is still read defensively,
not consumed for any number) and adds a separate, lower-priority
`operator_considerations` block (analysis-reliability notes for the operator,
surfaced only when a real-data limiter bites) — kept cleanly OUT of the player
coaching. `improvement_plan.json` output `schema_version` stays `1` (additive
field). `0.1.0` was the initial version.

## Out of scope (deferred)

- **Practice scheduling / dosage / progressions** (frequency, sets, week plan)
  — no evidence basis on placeholder data; future.
- **Recommendations for `developing_capability` skills** — documented only
  until their data lands (ball v4 + new metric/pose stages).
- **Personalized drill content / video links / a real content engine** — v1 is
  a small curated library.
- **Opponent-specific game-planning** (exploit opponent's backhand, etc.) —
  needs the targeting metric (pending) + reliable opponent roles.
- **Multi-role plans** (partner/opponents) — v1 is user-only, matching Stage 9.

## Known follow-ups

- **Uncalibrated mapping** (inherited from Stage 9): which drills actually move
  which metric is unvalidated. Calibrate with real-data + outcome tracking.
- **Most v1 focus areas may be provisional** (0.70 of the rating is synthetic),
  so the trustworthy part of the plan is the positioning/movement items until
  ball v4. Surfaced in `reliability`.
- **developing_capability becomes focus_areas** incrementally as each skill's
  metric lands; the descriptor table is the migration checklist.

## Architecture note

Stage 10 was in the original pipeline diagram; this contract takes it from "not
started" to "implemented + smoke-tested". Pipeline stays at 13 stages. On
approval, `ARCHITECTURE.md`'s "Stages 10–11: not started" becomes "Stage 10
implemented; Stage 11 not started".

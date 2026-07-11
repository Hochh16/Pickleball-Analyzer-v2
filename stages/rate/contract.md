# Stage 9 — Rate (USAPA skill rating)

> **REALIGNED to the 7 official USAPA categories (v0.4.0, 2026-07-09).** This
> contract body now describes the 7-category rating. Companion design +
> weights + reality table: `docs/USAPA_REALIGN_DESIGN.md`.

**Status:** DRAFT for review. Maps the aggregated `metrics.json` (Stage 8) to a
**USA Pickleball (USAPA) skill rating** for the **user**: a continuous point
estimate, the nearest official half-step band, a confidence range, and
per-category **evidence** across the **7 official USA Pickleball rating
categories** — `strategy`, `third_shot`, `dink`, `volley`, `serve_return`,
`forehand`, `backhand`. Rule-based, anchored in the published USA Pickleball
Player Skill Rating Definitions — same pipeline philosophy (documented
thresholds, honest confidence, loud failures).

## Scope decisions (settled with the operator before drafting)

> **DECISION (rate the USER only in v1).** One `rating.json` for the user. The
> user is the only role with durable real positioning + known handedness;
> opponents are contamination-flagged and partner handedness is unknown. Rating
> structure is built so other roles can be added later, but v1 emits the user
> only. (Alternatives considered: user+partner, all-four-roles — deferred.)

> **DECISION (output = continuous estimate + band + range).** Emit a continuous
> point estimate (e.g. `3.4`), the nearest official USAPA half-step `band`
> (`3.5`), and a `range` (e.g. `[3.0, 3.5]`). The range is the honest carrier of
> synthetic-ball + calibration uncertainty; the band matches the published
> self-rating sheet for readability.

> **DECISION (v0.3.0, 2026-07-07 — CONFIDENCE-WEIGHTED estimate).** The point
> estimate weights each category by `static_weight × confidence` (renormalized),
> so a category we can't trust no longer inflates the headline number. On the
> real ball this matters enormously: only **Strategy** rests on high-confidence
> data, so the estimate leans on it while the six shot-based categories (low
> confidence, small detected-shot sample) contribute little. If NO category
> carries any confidence, it falls back to the plain static-weight sum. The
> reported `confidence` stays the static-weight-average of category confidences
> (how much of the INTENDED skill picture is trustworthy) so the `range` stays
> honestly wide — a confident read of INCOMPLETE coverage. Honesty is carried by
> (a) the loud uncalibrated-threshold + USAPA-coverage warnings, (b) the lowered
> `confidence` + wide `range`, (c) per-category `data_source` + `confidence` +
> `limited_by` + `coverage_status`, and (d) `reliability.{measured,not_assessable}_
> categories`. Rating thresholds remain uncalibrated heuristics — do not treat the
> number as a measured rating.

> **DECISION (v0.4.0, 2026-07-09 — REALIGN to 7 USAPA categories; hard-gated,
> single caveated estimate).** Replaced the 6 homegrown dims with the 7 official
> categories. Operator-scoped (see `docs/USAPA_REALIGN_DESIGN.md`): score all 7,
> but honestly gate them — each carries `coverage_status` ∈ {`measured`,
> `partial`, `not_assessable`} from its confidence, and **count-only** categories
> (`forehand`/`backhand`: the stroke is counted but its quality — pace/depth/
> consistency — is unmeasured) are confidence-capped to `not_assessable`.
> `serve_return` is `not_assessable` when no serves are detected. Today only
> Strategy is `measured`; a loud USAPA-COVERAGE warning says so. This is a
> **legitimacy/naming win, not new signal** — the shot-based categories carry
> real signal only once ball recall (C4) / serve detection (C3) / stroke-side
> (F16) / shot speed (F7) land.

## Purpose

Turn the numbers into the thing a player actually asks for: *what level am I,
and why.* Stage 9 reads `metrics.json`, scores the user across a set of
skill dimensions (each anchored to USAPA level language), combines them into a
single rating with confidence, and reports the **evidence** behind it (which
metric drove each dimension, and whether that metric is real or
synthetic-derived). It feeds Stage 10 (improvement plan), which keys off the
weakest dimensions, and Stage 11 (annotated report).

Rule-based, not ML — there is no corpus of rated amateur footage to learn from,
so thresholds are documented heuristics anchored to published definitions and
**explicitly uncalibrated** (see Known follow-ups).

> **ACCEPTANCE BAR (read first).** Stage 9 v1 is validated for **logical
> correctness assuming the inputs were trustworthy** — i.e. *if the metrics
> were real, would the rating be computed and combined correctly?* It is NOT
> validated for real-world rating accuracy, and **none of its output (nor
> Stages 5–8's ball-derived output) is to be treated as useful until Stage 4 /
> 4.5 (real ball detection) is complete.** The smoke test therefore checks
> schema, banding, range, reliability propagation, and directional
> monotonicity — never "is the number right." This caveat is the whole reason
> for the loud warnings, the lowered confidence, and the wide range.

## Place in the architecture

```
metrics.json (S8)
        │
        ▼
   [9] rate ──► rating.json
```

Per-video, file-path I/O, standalone CLI:
`python -m stages.rate.rate <video_folder>`.

> **DECISION (folder name).** Code + contract live at `stages/rate/`
> (importable). This contract sits at the numbered stub `stages/09_rate/` for
> review; on approval it moves to `stages/rate/contract.md` and the stub folder
> is deleted.

## Inputs

Per-video folder positional argument.

| File | From | Stage 9 reads |
|---|---|---|
| `metrics.json` | Stage 8 | the entire metrics object — `players.user` (position, shot_mix, serve, errors_committed), `team.near`, `match` (rally length, third_shot, serve), `reliability` (which dimensions are real vs synthetic), `ball_source` |

**Only `metrics.json`.** Stage 9 does not re-read upstream files — Stage 8
already aggregated everything. (court.json is not needed; fps/geometry are not
used in the rating.)

CLI flags (defaults in Configuration): `--force`, `--log-level`,
`--synth-confidence-factor`, plus the contract's threshold knobs are constants
in v1 (documented, not all exposed) with `--role` reserved for future
multi-role rating (defaults to `user`).

## Output — `rating.json`

```json
{
  "schema_version": 1,
  "source_metrics": "data/pb_2min/metrics.json",
  "ball_source": "real",
  "rated_role": "user",
  "rating": {
    "estimate": 3.95,
    "band": "4.0",
    "range": [3.0, 5.0],
    "confidence": 0.30
  },
  "dimensions": [
    {
      "name": "strategy",
      "subscore_level": 4.08,
      "weight": 0.20,
      "confidence": 0.997,
      "limited_by": "sample_size",
      "data_source": "real",
      "coverage_status": "measured",
      "driver_metrics": {
        "user_kitchen_time_frac": 0.336,
        "both_at_kitchen_frac": 0.333,
        "user_transition_time_frac": 0.149,
        "distance_ft_per_min": 191.7,
        "unforced_error_rate": 0.0
      }
    },
    {
      "name": "third_shot",
      "subscore_level": 4.0,
      "weight": 0.18,
      "confidence": 0.234,
      "limited_by": "measurement",
      "data_source": "real",
      "coverage_status": "partial",
      "driver_metrics": {"third_shot_drop_rate": 0.5,
                         "third_shot_by_type": {"drive": 2, "drop": 3, "reset": 1},
                         "per_user": false}
    },
    {
      "name": "dink",
      "subscore_level": 2.91,
      "weight": 0.15,
      "confidence": 0.155,
      "limited_by": "measurement",
      "data_source": "real",
      "coverage_status": "partial",
      "driver_metrics": {"dink_count": 0, "dink_frac": 0.0, "mean_rally_length": 5.67}
    },
    {
      "name": "volley",
      "subscore_level": 4.2,
      "weight": 0.13,
      "confidence": 0.211,
      "limited_by": "measurement",
      "data_source": "real",
      "coverage_status": "partial",
      "driver_metrics": {"volley_rate": 0.6, "n_volley": 3}
    },
    {
      "name": "serve_return",
      "subscore_level": 3.0,
      "weight": 0.12,
      "confidence": 0.0,
      "limited_by": "sample_size",
      "data_source": "real",
      "coverage_status": "not_assessable",
      "driver_metrics": {"serve_fault_rate": null, "n_serves": 0, "return_metric": null}
    },
    {
      "name": "forehand",
      "subscore_level": 3.0,
      "weight": 0.12,
      "confidence": 0.05,
      "limited_by": "measurement",
      "data_source": "real",
      "coverage_status": "not_assessable",
      "driver_metrics": {"forehand_count": 2, "forehand_frac": 0.4,
                         "pace_mph": null, "depth": null, "consistency": null}
    },
    {
      "name": "backhand",
      "subscore_level": 3.0,
      "weight": 0.10,
      "confidence": 0.05,
      "limited_by": "measurement",
      "data_source": "real",
      "coverage_status": "not_assessable",
      "driver_metrics": {"backhand_count": 2, "backhand_frac": 0.4,
                         "pace_mph": null, "depth": null, "consistency": null}
    }
  ],
  "reliability": {
    "synthetic_ball": false,
    "real_weight": 1.0,
    "synthetic_weight": 0.0,
    "measured_categories": ["strategy"],
    "assessable_categories": ["strategy", "third_shot", "dink", "volley"],
    "not_assessable_categories": ["serve_return", "forehand", "backhand"],
    "note": "ball_source is real; all categories count as real data."
  },
  "skill_coverage": {
    "covered": ["strategy", "third_shot", "dink", "volley", "serve_return",
                "forehand", "backhand"],
    "proxy_or_pending": ["serve_depth_placement", "third_shot_drop_outcome",
                         "dink_tolerance", "forced_vs_unforced",
                         "shot_placement_targeting", "pace_power_control"],
    "not_captured_yet": ["return_of_serve", "volleys_hands_battles",
                         "attack_conversion", "reset_under_pressure",
                         "defense_scrambling", "partner_stacking_poaching",
                         "footwork_split_step", "shot_selection_iq"],
    "out_of_scope": ["spin", "score_situational_decisions"],
    "note": "not_captured_yet skills are NOT reflected in the rating; they need new metrics/stages. Surfaced so Stage 10 / the UI don't imply full coverage."
  },
  "usapa_anchor_version": "2024-self-rating",
  "params": {
    "synth_confidence_factor": 0.35,
    "weights": {"strategy": 0.20, "third_shot": 0.18, "dink": 0.15, "volley": 0.13,
                "serve_return": 0.12, "forehand": 0.12, "backhand": 0.10}
  },
  "warnings": [
    "Rating thresholds are UNCALIBRATED heuristics anchored to USAPA descriptions — no corpus of rated footage exists yet (see KNOWN_ISSUES.md).",
    "USAPA COVERAGE: this estimate currently rests almost entirely on strategy — 3 of 7 categories are not yet reliably measured (serve_return, forehand, backhand). The 6 shot-based categories need ball recall / serve detection / stroke-side / shot speed before they carry signal; they are confidence-gated, not guessed."
  ],
  "stage_version": "0.4.0",
  "completed_at_utc": "2026-07-09T..."
}
```

### Field notes

- **`rating.estimate`** ∈ [1.0, 5.5] (continuous) = `Σ weight_i ·
  subscore_level_i` over all dimensions. Clamped to [1.0, 5.5].
- **`rating.band`** = the nearest official USAPA half-step to `estimate` in
  `{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}` (estimate clamped to ≤5.0 for
  banding; >5.0 bands to "5.0").
- **`rating.range`** = `[estimate − h, estimate + h]` (each end rounded to the
  nearest 0.5, clamped to [1.0, 5.0]), where the half-width `h` grows as
  confidence falls: `h = RANGE_MIN_HALF + RANGE_SPAN · (1 − confidence)`. With
  v1's synthetic data, `confidence` is low so the range is wide — by design.
- **`rating.confidence`** ∈ [0,1] = `Σ weight_i · dim_confidence_i` (weighted
  mean of per-dimension confidences). Because synthetic dimensions carry low
  `dim_confidence`, overall confidence is low while the ball is synthetic.
- **`dimensions[]`** — one per USAPA category (7; below). Each has:
  - `subscore_level` — the category's standalone level estimate (same 1.0–5.5
    scale), from its driver metrics via documented thresholds. For count-only
    categories (forehand/backhand) this is a neutral prior (quality unmeasured).
  - `weight` — fixed contribution to the estimate (weights sum to 1.0).
  - `data_source` — `"real"` (strategy is position-derived, ball-independent) or
    `"synthetic"` (ball/shot-derived on the synthetic ball). Evidence only; it
    gates `confidence` while the ball is synthetic and is shown to the user.
  - `confidence` — per category, the **minimum inline `.confidence` of its driving
    Stage 8 metrics** (only as reliable as its weakest evidence), × the synthetic
    gate (`synth_confidence_factor` when the category is ball-derived AND
    `ball_source == "synthetic"`, else 1.0). **Count-only categories
    (forehand/backhand) are then capped to `QUALITY_UNMEASURED_CONF`** (the stroke
    is counted but its quality is unmeasured). Sample-size is carried by Stage 8's
    `penalty(n)` inside each metric's `.confidence` (single source of truth).
    Driver→metric map: `strategy`←{user.position, team.near}; `third_shot`←{match.third_shot};
    `dink`←{user.shot_mix.by_shot_type, match.rally_length_shots}; `volley`←{user.shot_mix.volley};
    `serve_return`←{user.serve}; `forehand`/`backhand`←{user.shot_mix.by_stroke_side}.
  - `coverage_status` — **NEW (v0.4.0).** How well we can actually assess the
    category, from its confidence: `measured` (≥`MEASURED_CONF_FLOOR` 0.5),
    `partial` (≥`ASSESS_CONF_FLOOR` 0.1), else `not_assessable`. Stage 10 routes
    `not_assessable` categories to `developing_capability.not_assessable_now`.
  - `limited_by` — the `limited_by` tag of the binding (min-confidence) driver
    (`sample_size` → "record more"; `measurement`/`known_limit` → camera limit).
  - `driver_metrics` — the exact metric values used; `null` for any not-yet-
    available input (e.g. `pace_mph`, `depth` on the strokes; `return_metric`).
- **`reliability`** — `real_weight` / `synthetic_weight` make the synthetic
  dependence explicit; `measured_categories` / `assessable_categories` /
  `not_assessable_categories` make the coverage explicit + machine-readable for
  Stage 10/11 and the UI.

There is **no separate `.meta.json`** — metadata lives inside `rating.json`.

## The 7 USAPA categories

The 7 official USA Pickleball rating categories; weights sum to 1.0 (uncalibrated
heuristics for rough skill importance — the confidence-weighting, not the weight,
keeps low-confidence categories from inflating the estimate). Each maps driver
metric(s) to a `subscore_level` via a documented piecewise threshold table
(constants in code; tuned only against the smoke test's directional checks). The
**"today"** column is the coverage reality for the user on real data (pb_2min).

| Category | Weight | Drivers (from metrics.json) | Higher level ⇐ | Today |
|---|---|---|---|---|
| **strategy** | 0.20 | `position.zone_time_frac.{kitchen,transition}`, `team.near.both_at_kitchen_frac`, `movement.distance_ft_per_min` (+ unforced errors, exposed not scored) | more kitchen-line time, partners both at line, less no-man's-land | ● **measured** — the anchor |
| **third_shot** | 0.18 | `match.third_shot.drop_rate` | drops the 3rd more than drives it | ◐ partial (match-level, not per-user) |
| **dink** | 0.15 | `shot_mix.by_shot_type.dink` share + `rally_length` sustain | more soft-game dinking + longer rallies | ◐ partial (tiny shot sample) |
| **volley** | 0.13 | `shot_mix.volley.volley_rate` | more controlled net volleys | ◐ partial |
| **serve_return** | 0.12 | `serve.serve_fault_rate` (return not separately detected) | lower serve-fault rate | ○ not_assessable (no serves detected) |
| **forehand** | 0.12 | `shot_mix.by_stroke_side.forehand` COUNT only | (quality unmeasured → neutral) | ○ not_assessable (count-only) |
| **backhand** | 0.10 | `shot_mix.by_stroke_side.backhand` COUNT only | (quality unmeasured → neutral) | ○ not_assessable (count-only) |

> **Strategy is the anchor.** It's the only category on high-confidence real
> position data. Unforced errors are a USAPA Strategy sub-element but are exposed
> as a driver only (NOT scored / not a confidence driver) — errors are
> undetectable until bounce recall improves, and folding them in would either zero
> Strategy's confidence or read "no errors = perfect."

> **USAPA anchoring.** Threshold tables track the published USA Pickleball Player
> Skill Rating Definitions (directional + uncalibrated — no rated-footage corpus;
> honest confidence + range carry that). `usapa_anchor_version` records the set.

> **Legitimacy vs signal.** The realign gives the recognized *category structure*
> now, but 6 of 7 categories carry real signal only once their metrics land:
> depth/landing ← bounce recall (C4); pace ← court-plane speed (F7); FH/BH quality
> ← stroke-side (F16) + speed; per-user third-shot attribution; serve detection
> (C3). Until then they are honestly `partial`/`not_assessable`, not faked.

## Skill coverage map

A complete USAPA-style assessment considers more competencies than v1 can
measure. To avoid implying full coverage, Stage 9 classifies every
rating-relevant skill into one of four buckets and emits this in
`rating.json.skill_coverage`. **Skills in `not_captured_yet` are NOT reflected
in the rating at all** — they need new metrics/stages; surfacing them keeps
Stage 10 (improvement plan) and the UI honest about blind spots.

**A — A USAPA category (structure present; signal varies — see `coverage_status`):**
- Court positioning / NVZ approach / move-as-a-team → `strategy` (real, ● measured)
- Third-shot drop-to-net → `third_shot` (drop_rate; ◐ partial, match-level)
- Soft game at the kitchen → `dink` (dink share + sustain; ◐ partial)
- Net volleys → `volley` (volley_rate; ◐ partial)
- Serve + return → `serve_return` (fault rate only; ○ not_assessable w/o serves)
- Forehand → `forehand` (COUNT only; ○ not_assessable, quality unmeasured)
- Backhand → `backhand` (COUNT only; ○ not_assessable, quality unmeasured)

**B — Covered via proxy / pending input (inside an existing dimension):**
- Serve **depth & placement** — needs ball landing; v1 uses fault rate only.
- Third-shot-drop **outcome/quality** — `pending_real_ball`.
- **Dink consistency / patience** — `pending_real_ball` (dink_shot_tolerance).
- **Forced vs unforced** errors — `pending_real_ball`.
- **Shot placement / targeting opponent backhand** — `pending_real_ball`.
- **Pace / power control** — `mean_post_speed` exists (synthetic) but isn't yet
  its own dimension.

**C — Not captured yet (needs a NEW metric or stage; absent from the rating):**
- **Return of serve** (depth + recovery to the net afterward).
- **Volleys / hands battles / speed-ups + counters** at the net (needs fast
  ball + reaction timing).
- **Attack conversion / put-aways / overheads** (needs pop-up detection).
- **Reset under pressure** from the transition zone (Stage 6 has a `reset`
  shot-type but it isn't a rating input yet).
- **Defense / scrambling / re-resetting** under attack.
- **Partner strategy** — stacking, poaching, switching, communication (team
  spacing is partly captured; stacking/poaching detection is not built).
- **Footwork quality** — split-step timing, ready-position recovery (this is
  the Tier-C **pose-technique** stage in ARCHITECTURE.md; mostly REAL data).
- **Shot selection / strategic IQ** under pressure (hard; only weak proxies).

**D — Out of scope for a single corner camera (likely permanent):**
- **Spin** (topspin/slice) — not recoverable from this viewpoint.
- **Score / situational decision-making** — no game-state/score model in v1.

> So: the rating's *structure* is complete and correct, but its *competency
> coverage* is partial — buckets B and C are the roadmap. As ball v4 + the
> pose-technique stage land, B's pending inputs and several of C's skills become
> new dimensions; weights are re-normalized then. v1 does not invent scores for
> uncovered skills.

## Method

1. **Load + validate.** Read `metrics.json`; check `schema_version == 2` (fail
   loudly otherwise — Stage 8 now emits inline `{value, confidence, n,
   limited_by}` wrappers). Pull `ball_source` and `reliability`. **Unwrap** each
   consumed metric to its `.value` for the scorers (which operate on raw values),
   and read each metric's `.confidence` / `.limited_by` for the dimension
   confidence. Select the rated role block (`players.user`); if absent/empty →
   degrade (see below).
2. **Per-dimension subscore.** For each dimension, read its driver metric(s),
   map to a `subscore_level` via the threshold table, set `dim_confidence` from
   the **minimum of its drivers' inline `.confidence`** (× synthetic gate), record
   the binding driver's `limited_by`, and record `driver_metrics` + `data_source`.
3. **Combine.** `estimate = Σ weight·subscore`; `confidence = Σ
   weight·dim_confidence`; `band` = nearest half-step; `range` from confidence.
4. **Reliability + warnings.** Emit `real_weight`/`synthetic_weight`; loud
   placeholder warning when `ball_source == "synthetic"`; the uncalibrated-
   thresholds warning always.
5. **Write** `rating.json` (refuse to overwrite without `--force`).

## Defenses against placeholder / bad data

- **Propagates `ball_source` + the synthetic dependence** (`real_weight` /
  `synthetic_weight`) and a loud warning. The rating is the most
  over-interpretable output in the pipeline, so the placeholder caveat is
  front-and-center.
- **Honest confidence + range** rather than a falsely precise number.
- **Missing/empty user block** (no shots, no position) → emit a rating with
  `confidence ≈ 0`, maximal range, `band` from whatever dimensions exist, and a
  warning; never crash, never fabricate a confident level.
- **`metrics.json` schema mismatch** → fail loudly naming the file.
- **Output exists without `--force`** → `FileExistsError`.

## Edge cases

- **A dimension's driver metric is missing/null** (e.g. user has zero serves →
  no serve_fault_rate) → that dimension's `subscore_level` falls to a neutral
  prior (the dataset-wide midpoint, ≈3.0) at low `dim_confidence`, flagged in
  `driver_metrics`. It still carries its weight (so the estimate stays on
  scale) but contributes little confidence.
- **All dimensions low-confidence** (e.g. tiny clip) → very low overall
  confidence, very wide range, warning.
- **estimate at the scale edges** — clamp to [1.0, 5.5]; band clamps to [1.0,
  5.0].
- **Required input missing/malformed** → fail loudly naming the file.

## Configuration (defaults; tuned against smoke-test directional checks only)

```python
WEIGHTS = {"strategy": 0.20, "third_shot": 0.18, "dink": 0.15, "volley": 0.13,
           "serve_return": 0.12, "forehand": 0.12, "backhand": 0.10}
SYNTH_CONFIDENCE_FACTOR = 0.35   # gate for synthetic-ball-derived categories
NEUTRAL_PRIOR_LEVEL     = 3.0    # subscore when a driver is missing / quality unmeasured
ASSESS_CONF_FLOOR       = 0.10   # below this a category is 'not_assessable'
MEASURED_CONF_FLOOR     = 0.50   # at/above this a category is 'measured' (else 'partial')
QUALITY_UNMEASURED_CONF = 0.05   # cap for count-only categories (forehand/backhand)
RANGE_MIN_HALF          = 0.25   # min half-width of the confidence range
RANGE_SPAN              = 1.25   # extra half-width at confidence 0
USAPA_BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
REAL_DIMS   = {"strategy"}       # position-derived -> real regardless of ball_source
COUNT_ONLY_DIMS = {"forehand", "backhand"}
# Per-category threshold tables (metric -> subscore_level) live as documented
# constants next to each category's scorer; see code.
```

## Smoke test

`stages/rate/test_rate.py`, against `data/test_clip/`. There is **no
ground-truth rating** (and the ball is synthetic), so accuracy cannot be graded.
The test gates on **schema + internal consistency + directional monotonicity**
(the engine must move the right way), mirroring how Stage 6 unit-tested its rule
logic and Stage 8 gated on reconciliation.

Pipeline prefix: regenerate the chain (synth → 5 → 5.5 → 6 → 7 → 2.5 → 8) then
run Stage 9, plus pure-function unit checks that don't need the pipeline.

1. **Schema.** `rating.json` parses, `schema_version == 1`; `estimate` ∈
   [1.0, 5.5]; `band` ∈ `USAPA_BANDS`; `range` is `[lo, hi]` with `lo ≤
   estimate ≤ hi`; `confidence` ∈ [0,1]; the **7 USAPA categories** present in
   `WEIGHTS` order; weights sum to 1.0 (±1e-6); every `subscore_level` ∈
   [1.0, 5.5]; every `confidence` ∈ [0,1]; every category carries a `limited_by`
   ∈ {`sample_size`, `measurement`, `known_limit`, `detection_floor`} and a
   `coverage_status` ∈ {`measured`, `partial`, `not_assessable`}. (Input
   `metrics.json` must be `schema_version == 2`.)
2. **Banding.** `band` is the nearest half-step to `estimate` (clamped).
3. **Range vs confidence.** Range half-width decreases monotonically as
   confidence increases (check the helper directly at a few confidence values).
4. **Reliability propagation.** `reliability.synthetic_ball == true`;
   `real_weight + synthetic_weight == 1.0`; the placeholder + uncalibrated
   warnings are present; synthetic dimensions have lower `dim_confidence` than
   real ones (with equal sample sufficiency).
5. **Directional monotonicity (engine sanity, pure functions).** For each
   scorer with a live driver, a clearly-stronger driver value yields a
   `subscore_level ≥` a clearly-weaker one (more kitchen time → strategy; higher
   drop rate → third_shot; higher dink fraction → dink; higher volley rate →
   volley; lower serve-fault → serve_return). Count-only strokes return a valid
   neutral level with the count surfaced. End-to-end: a synthesized "strong"
   metrics dict yields `estimate >` a "weak" one.
6. **Confidence drops with synthetic ball.** Rating the same metrics with
   `ball_source` forced to `"real"` (test hook) yields a higher `confidence`
   and narrower `range` than with `"synthetic"` — proving the honesty machinery
   actually engages.
7. **Degradation.** An empty/zeroed user block produces a valid file with
   `confidence ≈ 0`, a maximal range, and a warning, without crashing.
8. **Skill coverage.** `skill_coverage` is present with the four buckets;
   `covered` equals the seven category names; no skill appears in two buckets.
   (Guards against silently implying full competency coverage.)

## Stage version

`0.4.0` (USAPA REALIGN): the 6 homegrown dims are replaced by the **7 official
USAPA categories**, each gaining a `coverage_status`; count-only strokes
(forehand/backhand) are confidence-capped to `not_assessable`; `reliability`
gains `measured`/`assessable`/`not_assessable`_categories and a loud
USAPA-COVERAGE warning fires when ≤2 categories are measured. `rating.json`
output `schema_version` stays `1` (additive — `coverage_status` +
reliability fields). Prior: `0.3.0` confidence-WEIGHTED estimate; `0.2.0`
Foundation #3 (inline Stage-8 confidence + `limited_by`); `0.1.0` initial.
Increment minor for behavior changes preserving the schema; bump
`schema_version` for breaking schema changes.

## Out of scope (deferred)

- **Calibration against real rated players.** v1 thresholds are heuristics
  anchored to text definitions. Calibrating to DUPR/USAPA-rated footage is a
  major future effort (needs labeled data).
- **Multi-role rating** (partner/opponents). Structure allows it; v1 is
  user-only.
- **Return-of-serve, dink tolerance, forced/unforced, third-shot-drop
  outcome** as rating inputs — they live in Stage 8's `pending_real_ball`;
  wired in at ball v4. v1 uses available proxies.
- **DUPR-style dynamic rating** (match-result/Elo across games) — Stage 9 is a
  single-match capability estimate, not a ladder rating.
- **Sub-level reasoning text generation** beyond the short per-dimension
  `rationale` — richer coaching prose is Stage 10/11 territory.

## Known follow-ups

- **Uncalibrated thresholds are the dominant limitation.** Once any rated
  footage exists, calibrate the per-category tables + weights and replace the
  directional smoke checks with accuracy bars.
- **Only Strategy is `measured` today** — the estimate leans almost entirely on
  one category. The six shot-based categories become trustworthy only as their
  metrics land (ball recall C4, serve detection C3, stroke-side F16, shot speed
  F7, per-user third-shot attribution). This is the build-program **ADD** step;
  the category structure + weights stay, and confidence rises across the board.

## Architecture note

Stage 9 was already in the pipeline diagram (one of the original 11 stages);
this contract takes it from "not started" to "implemented + smoke-tested".
Pipeline count stays at 13. On approval, `ARCHITECTURE.md`'s "Stages 9–11: not
started" line becomes "Stage 9 implemented; Stages 10–11 not started".

# Stage 9 — Rate (USAPA skill rating)

**Status:** DRAFT for review. Maps the aggregated `metrics.json` (Stage 8) to a
**USA Pickleball (USAPA) skill rating** for the **user**: a continuous point
estimate, the nearest official half-step band, a confidence range, and
per-dimension **evidence**. Rule-based, anchored in the published USA Pickleball
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

> **DECISION (full rating now, loudly flagged — operator's explicit choice).**
> The point estimate is computed from ALL skill dimensions, including the ones
> derived from the synthetic ball, **without down-weighting** the synthetic
> dimensions in the score. Rationale: the operator wants the complete rating
> engine built and exercised end-to-end now, so that when real ball detection
> (v4) lands the same engine simply gets more accurate — not a weak
> positioning-only stand-in replaced later. **The honesty is carried by (a) a
> loud top-level placeholder warning, (b) a lowered `confidence` and widened
> `range` driven by data reliability, and (c) per-dimension `data_source`
> evidence.** This is a deliberate departure from the stricter
> "reliability-gated sub-scores" option; it is documented here and in
> KNOWN_ISSUES.md so the trade-off is explicit. **The point estimate leans on
> placeholder data until v4 — do not treat the number as a measured rating.**

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
  "source_metrics": "data/test_clip/metrics.json",
  "ball_source": "synthetic",
  "rated_role": "user",
  "rating": {
    "estimate": 3.4,
    "band": "3.5",
    "range": [3.0, 3.5],
    "confidence": 0.38
  },
  "dimensions": [
    {
      "name": "net_play",
      "subscore_level": 3.0,
      "weight": 0.20,
      "confidence": 0.85,
      "data_source": "real",
      "driver_metrics": {
        "user_kitchen_time_frac": 0.133,
        "both_at_kitchen_frac": 0.02,
        "user_transition_time_frac": 0.320
      },
      "rationale": "Limited time at the kitchen line and high transition-zone time indicate a developing (≈3.0) net game; 3.5+ players hold the line together."
    },
    {
      "name": "error_control",
      "subscore_level": 3.5,
      "weight": 0.25,
      "confidence": 0.30,
      "data_source": "synthetic",
      "driver_metrics": {"errors_per_rally": 0.36, "unforced_rate": null},
      "rationale": "Moderate attributed-error rate. NOTE: forced/unforced split is pending real ball; v1 uses total attributed errors (synthetic-derived)."
    },
    {
      "name": "shot_skill",
      "subscore_level": 3.5,
      "weight": 0.25,
      "confidence": 0.30,
      "data_source": "synthetic",
      "driver_metrics": {"third_shot_drop_rate": 0.375, "shot_variety": 5,
                         "soft_game_frac": 0.42, "unknown_type_frac": 0.12}
    },
    {
      "name": "serve",
      "subscore_level": 4.0,
      "weight": 0.10,
      "confidence": 0.30,
      "data_source": "synthetic",
      "driver_metrics": {"serve_fault_rate": 0.0}
    },
    {
      "name": "rally_consistency",
      "subscore_level": 3.5,
      "weight": 0.10,
      "confidence": 0.30,
      "data_source": "synthetic",
      "driver_metrics": {"mean_rally_length": 5.74, "volley_rate": 0.12}
    },
    {
      "name": "movement",
      "subscore_level": 3.0,
      "weight": 0.10,
      "confidence": 0.85,
      "data_source": "real",
      "driver_metrics": {"court_coverage_frac": 0.55, "distance_ft_per_min": 56.98}
    }
  ],
  "reliability": {
    "synthetic_ball": true,
    "real_weight": 0.30,
    "synthetic_weight": 0.70,
    "note": "0.70 of the rating weight comes from synthetic-ball-derived dimensions; estimate is PLACEHOLDER until ball v4. confidence + range reflect this."
  },
  "skill_coverage": {
    "covered": ["net_play", "movement", "error_control", "shot_skill",
                "serve", "rally_consistency"],
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
    "weights": {"net_play": 0.20, "error_control": 0.25, "shot_skill": 0.25,
                "serve": 0.10, "rally_consistency": 0.10, "movement": 0.10}
  },
  "warnings": [
    "ball_source is 'synthetic': the rating point estimate is PLACEHOLDER (0.70 of its weight is synthetic-ball-derived). Treat as a scaffold until ball detection v4; confidence is reduced and range widened accordingly.",
    "Rating thresholds are UNCALIBRATED heuristics anchored to USAPA descriptions — no corpus of rated footage exists yet (see KNOWN_ISSUES.md)."
  ],
  "stage_version": "0.1.0",
  "completed_at_utc": "2026-05-29T..."
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
- **`dimensions[]`** — one per skill axis (below). Each has:
  - `subscore_level` — the dimension's standalone level estimate (same 1.0–5.5
    scale), from its driver metrics via documented thresholds.
  - `weight` — fixed contribution to the estimate (weights sum to 1.0).
  - `data_source` — `"real"` (positioning/movement, ball-independent) or
    `"synthetic"` (derived from the synthetic ball). **Evidence only — does NOT
    down-weight the subscore in the estimate** (per the full-rating decision);
    it gates `dim_confidence` while the ball is synthetic, and is shown to the user.
  - `confidence` (`dim_confidence`) — **CHANGED in v0.2.0 (consumes Stage 8
    schema_version 2 inline confidence).** Per dimension, take the **minimum of
    the inline `.confidence` of its driving Stage 8 metrics** (the dimension is
    only as reliable as its weakest evidence), then apply the synthetic gate:
    `dim_confidence = min(driver .confidence) × (synth_confidence_factor if the
    dimension is ball-derived AND ball_source == "synthetic" else 1.0)`. The old
    `data_conf · sample_conf` formula is **RETIRED** — sample-size is now carried
    by Stage 8's `penalty(n)` inside each metric's `.confidence` (single source
    of truth), not recomputed here against `SAMPLE_FLOORS`. The synthetic **gate**
    is retained because Stage 8 inline confidence is artificially clean on the
    synthetic ball (see Stage 8 contract § "Synthetic-ball interaction"); on the
    real ball the gate is inactive and inline confidence fully drives the rating.
    Driver→metric map: `net_play`←{user.position, team.near}; `movement`←{user.position};
    `error_control`←{user.errors_committed}; `shot_skill`←{user.shot_mix.by_shot_type,
    match.third_shot}; `serve`←{user.serve}; `rally_consistency`←{match.rally_length_shots,
    user.shot_mix.volley}.
  - `limited_by` — **NEW.** The `limited_by` tag of the binding (min-confidence)
    driver, threaded through so Stage 11 can show the per-dimension remedy
    (`sample_size` → "record more"; `measurement`/`known_limit` → camera limit).
  - `driver_metrics` — the exact metric values used (so Stage 10/11 and the
    user see the basis); `null` for any not-yet-available input (e.g.
    `unforced_rate`, which is `pending_real_ball`).
  - `rationale` — short USAPA-anchored sentence (optional per dimension).

  > **`error_control` confidence caveat (v0.2.0):** it draws from
  > `user.errors_committed.confidence`, whose sample size is the error-event
  > *count*, so a clean (few-error) player gets conservatively low confidence.
  > Acceptable for v1; refine to a rally-opportunity sample later.
- **`reliability`** — `real_weight` / `synthetic_weight` make the synthetic
  dependence explicit and machine-readable for Stage 10/11.

There is **no separate `.meta.json`** — metadata lives inside `rating.json`.

## Skill dimensions (USAPA-anchored)

Six dimensions; weights sum to 1.0. Real-data dimensions total 0.30 of the
weight (net_play 0.20 + movement 0.10); synthetic 0.70. Each maps driver
metric(s) to a `subscore_level` via a documented piecewise threshold table
(constants in code; tuned only against the smoke test's directional checks,
NOT against real ratings).

| Dimension | Weight | Source | Drivers (from metrics.json) | Higher level ⇐ |
|---|---|---|---|---|
| **net_play** | 0.20 | real | `players.user.position.zone_time_frac.kitchen`, `team.near.both_at_kitchen_frac`, `…transition` | more kitchen-line time, partners both at line, less no-man's-land |
| **movement** | 0.10 | real | `players.user.position.court_coverage_frac`, `movement.distance_ft_per_min` | efficient coverage (not frantic, not static) |
| **error_control** | 0.25 | synthetic | `players.user.errors_committed` ÷ rallies-involved; `unforced_rate` (pending=null) | fewer errors per rally |
| **shot_skill** | 0.25 | synthetic | `match.third_shot.drop_rate`, user `shot_mix` variety + soft-game fraction + unknown rate | reliable third-shot drops, varied + controlled shot mix |
| **serve** | 0.10 | synthetic | `players.user.serve.serve_fault_rate` | low serve-fault rate |
| **rally_consistency** | 0.10 | synthetic | `match.rally_length_shots.mean`, user `shot_mix.volley_rate` | sustains longer rallies, comfortable at net exchanges |

> **USAPA anchoring.** Threshold tables are set so the dimension language tracks
> the published USA Pickleball Player Skill Rating Definitions — e.g. a player
> who rarely reaches the kitchen line and makes frequent errors scores ≈2.5–3.0;
> consistent dinking/drops + low errors + holding the line scores ≈4.0; forcing
> errors with varied, controlled shots ≈4.5. `usapa_anchor_version` records the
> definition set used. The mapping is **directional and uncalibrated** (no
> rated-footage corpus) — honest confidence + range carry that.

> **Pending inputs flagged, not faked.** `error_control` would ideally use the
> forced/unforced split, and `shot_skill` the third-shot-drop *outcome* + dink
> tolerance — all in Stage 8's `pending_real_ball`. v1 uses the available
> proxies (total errors, drop *rate*) and records the missing inputs as `null`
> driver_metrics with a note. When v4 lands, swap the proxies for the real
> inputs; the dimension structure is unchanged.

## Skill coverage map

A complete USAPA-style assessment considers more competencies than v1 can
measure. To avoid implying full coverage, Stage 9 classifies every
rating-relevant skill into one of four buckets and emits this in
`rating.json.skill_coverage`. **Skills in `not_captured_yet` are NOT reflected
in the rating at all** — they need new metrics/stages; surfacing them keeps
Stage 10 (improvement plan) and the UI honest about blind spots.

**A — Covered now (a dimension):**
- Net-play / kitchen-line discipline → `net_play` (real)
- Court movement / coverage → `movement` (real)
- Consistency / error control → `error_control` (synthetic; forced/unforced pending)
- Third-shot + shot variety / soft game → `shot_skill` (synthetic; outcome pending)
- Serve consistency → `serve` (synthetic; fault rate only)
- Rally consistency / net comfort → `rally_consistency` (synthetic)

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
WEIGHTS = {"net_play": 0.20, "movement": 0.10, "error_control": 0.25,
           "shot_skill": 0.25, "serve": 0.10, "rally_consistency": 0.10}
SYNTH_CONFIDENCE_FACTOR = 0.35   # data_conf for synthetic-derived dimensions
NEUTRAL_PRIOR_LEVEL     = 3.0    # subscore when a driver is missing
RANGE_MIN_HALF          = 0.25   # min half-width of the confidence range
RANGE_SPAN              = 1.25   # extra half-width at confidence 0
USAPA_BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
SAMPLE_FLOORS = {"position_frames": 1500, "shots": 40, "rallies": 15}
# Per-dimension threshold tables (metric -> subscore_level) live as documented
# constants next to each dimension's scorer; see code.
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
   estimate ≤ hi`; `confidence` ∈ [0,1]; six dimensions present; weights sum to
   1.0 (±1e-6); every `subscore_level` ∈ [1.0, 5.5]; every `dim_confidence` ∈
   [0,1]; every dimension carries a `limited_by` ∈ {`sample_size`, `measurement`,
   `known_limit`, `detection_floor`}. (Input `metrics.json` must be
   `schema_version == 2`.)
2. **Banding.** `band` is the nearest half-step to `estimate` (clamped).
3. **Range vs confidence.** Range half-width decreases monotonically as
   confidence increases (check the helper directly at a few confidence values).
4. **Reliability propagation.** `reliability.synthetic_ball == true`;
   `real_weight + synthetic_weight == 1.0`; the placeholder + uncalibrated
   warnings are present; synthetic dimensions have lower `dim_confidence` than
   real ones (with equal sample sufficiency).
5. **Directional monotonicity (engine sanity, pure functions).** For each
   dimension's scorer, a clearly-stronger driver value yields a `subscore_level
   ≥` a clearly-weaker one (e.g. more kitchen time, fewer errors, lower
   serve-fault rate, higher drop rate, longer rallies → higher or equal). And
   end-to-end: a synthesized "strong" metrics dict yields `estimate ≥` a
   "weak" one.
6. **Confidence drops with synthetic ball.** Rating the same metrics with
   `ball_source` forced to `"real"` (test hook) yields a higher `confidence`
   and narrower `range` than with `"synthetic"` — proving the honesty machinery
   actually engages.
7. **Degradation.** An empty/zeroed user block produces a valid file with
   `confidence ≈ 0`, a maximal range, and a warning, without crashing.
8. **Skill coverage.** `skill_coverage` is present with the four buckets;
   `covered` equals the six dimension names; no skill appears in two buckets.
   (Guards against silently implying full competency coverage.)

## Stage version

`0.2.0` (Foundation #3 — confidence propagation): consumes Stage 8
`schema_version 2` (inline metric wrappers); per-dimension `dim_confidence` now
derives from the **minimum inline `.confidence`** of each dimension's driving
metrics (synthetic gate retained) instead of `data_conf · sample_conf`; each
dimension gains a `limited_by` tag. `rating.json` output `schema_version` stays
`1` (additive — only the new `limited_by` field). `0.1.0` was the initial
version. Increment minor for behavior changes preserving the `rating.json`
schema; bump `schema_version` for breaking schema changes.

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
  footage exists, calibrate the per-dimension tables + weights and replace the
  directional smoke checks with accuracy bars.
- **Point estimate leans on synthetic ball** (operator's explicit full-rating
  choice). Re-validate the whole rating on real trajectories at v4; the estimate
  will shift and confidence should rise. Swap the `pending_real_ball` proxies
  for their real inputs then.
- **Real-data dimensions are only 0.30 of the weight**, so even the
  "high-confidence" part of v1 is a minority of the estimate. As real ball lands
  and synthetic dimensions become trustworthy, the weighting stays but the
  confidence rises across the board.

## Architecture note

Stage 9 was already in the pipeline diagram (one of the original 11 stages);
this contract takes it from "not started" to "implemented + smoke-tested".
Pipeline count stays at 13. On approval, `ARCHITECTURE.md`'s "Stages 9–11: not
started" line becomes "Stage 9 implemented; Stages 10–11 not started".

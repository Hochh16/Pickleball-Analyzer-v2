# Stage 8 — Compute Metrics

**Status:** DRAFT for review. Aggregates every upstream per-shot / per-rally /
per-bounce / per-frame stream into one `metrics.json`: match-level summary,
per-player (per-role) breakdowns, error attribution, position/coverage stats,
and numeric heatmap grids. Pure aggregation + arithmetic — no new detection,
same pipeline philosophy (documented formulas, loud failures, honest gaps).

This is the first stage to **consume `track_roles.json`** (Stage 2.5) for
per-player attribution, the first to read **real player positions**
(`players.parquet`) for durable position metrics, and the place where Stage 7's
`end_reason → error-owner` implication is finally turned into per-player /
per-team error counts.

## Scope decisions (settled with the operator before drafting)

> **NOTE (2026-06-19):** opponents are now identity-based **`opp_a` / `opp_b`**
> (Stage 2.5), not position L/R. This contract's older `opp_left`/`opp_right`
> prose and any left/right-by-court_x opponent semantics below are **stale** and
> belong to the deferred real-ball Stage 8 rework (SYSTEM_DESIGN.md #7); the code
> uses `opp_a`/`opp_b`.
>
> **DECISION (attribution = all roles, best-effort).** Stage 8 consumes
> `track_roles.json` and emits per-role stats for all four roles
> (`user`, `partner`, `opp_a`, `opp_b`). Each role carries its
> `role_confidence` (mean of its tracks' confidences from Stage 2.5) and a
> `role_contaminated` flag when confidence is below a floor, because Stage 2.5
> opponent roles are known-contaminated. Shots/positions whose `track_id` maps
> to `noise` or to no role go into an `unattributed` bucket — never silently
> dropped, never force-assigned. (Alternative considered: user-only, mirroring
> Stage 6 handedness. Rejected — the operator wants all-player stats now, with
> contamination surfaced rather than hidden.)

> **DECISION (heatmaps = grid data here, render in Stage 11).** Stage 8 emits
> numeric 2-D occupancy / landing grids in court coordinates inside
> `metrics.json`. It does NOT draw images — rendering is Stage 11's job
> (architecture's dedicated render stage). Keeps Stage 8 pure-data and testable
> by array reconciliation.

> **DECISION (position stats included in v1).** Kitchen/transition/baseline
> time fractions and court coverage are computed from `players.parquet` —
> **real tracking data, independent of the synthetic ball** — so they are
> durable real value now and are flagged as NOT synthetic-gated in the
> output's reliability map.

## Purpose

Take everything the pipeline has produced and compute the numbers a player
actually wants: how long were the rallies, what's my shot mix, how often do I
fault my serve, who's losing the points and how, where do I stand on the court.
These feed Stage 9 (USAPA rating) and Stage 11 (annotated video / report).

Stage 8 is **rule-based aggregation** — every metric has a documented formula
and reconciles with its inputs. Where a metric can't be computed honestly
(missing role, synthetic ball, ambiguous receiver), it's surfaced as a flag or
an `unattributed` bucket, not faked.

## Place in the architecture

```
classified.json (S6) + rallies.json (S7) + bounces.json (S5.5)
   + players.parquet (S2) + track_roles.json (S2.5) + roster.json
   + court.json + court_zones.json (S1)
        │
        ▼
   [8] compute_metrics ──► metrics.json
```

Per-video, file-path I/O, standalone CLI:
`python -m stages.compute_metrics.compute_metrics <video_folder>`.

> **DECISION (folder name).** Code + contract live at
> `stages/compute_metrics/` (importable; Python modules can't start with a
> digit — same convention as `detect_shots/`, `segment_rallies/`, etc.). This
> contract sits at the numbered stub path `stages/08_compute_metrics/` for
> review; on approval it moves to `stages/compute_metrics/contract.md` and the
> stub folder is deleted.

## Inputs

Per-video folder positional argument.

| File | From | Stage 8 reads |
|---|---|---|
| `rallies.json` | Stage 7 | rally records (`shot_ids`, `n_shots`, `duration_sec`, `server_track_id`, `end_reason`, `end_signals.hitter_side`) — rally-length + serve + error-attribution metrics |
| `classified.json` | Stage 6 | per-shot (`shot_id`, `track_id`, `is_user`, `is_serve`, `stroke_side`, `shot_type`, `is_volley`, `features.contact_zone`, `features.post_speed_ftps`) — shot-mix + third-shot + per-role shot stats; also `ball_source` for the synthetic flag |
| `bounces.json` | Stage 5.5 | per-bounce (`court_xy_ft`, `is_in_court`, `court_zone`, `out_side`) — ball-landing heatmap + in/out rate |
| `players.parquet` | Stage 2 | per-frame `track_id`, `court_x_ft`, `court_y_ft`, `in_court`, `transient` — position / coverage stats + position heatmap |
| `track_roles.json` | Stage 2.5 | `track_roles[tid].role` + `.confidence`, `roles[role].track_ids` — maps every shot/track to a logical player |
| `roster.json` | setup | `handedness` per role — carried into per-role output for downstream (not used in any formula here) |
| `court.json` | Stage 1 | `court_geometry_feet` (`length_ft=44`, `width_ft=20`), `video.fps` — net line, grid extent, time base |
| `court_zones.json` | Stage 1 | zone depth bands — documented; the actual binning reuses Stage 6's `zone_from_court_y` constants (single source of truth) |

CLI flags (defaults in Configuration): `--force`, `--log-level`,
`--heatmap-bin-ft`, `--role-conf-floor`.

**Degradation (loud, not silent):**
- `track_roles.json` missing/malformed → **warn**, fall back to user-only
  attribution via `classified.json.is_user` (`user` role from is_user shots;
  everyone else `unattributed`), match-level metrics unaffected. Per-role
  opponent/partner blocks emitted empty with a warning.
- `bounces.json` empty → ball-landing heatmap + in/out rate emitted empty/zero
  with a note; everything else unaffected.
- `rallies.json` with zero rallies → rally + error + serve families emitted
  empty; shot-mix + position families still computed from their own inputs.
- Required structural input missing/malformed (`classified.json`,
  `players.parquet`, `court.json`) → **fail loudly** naming the file.

## Output — `metrics.json`

```json
{
  "schema_version": 1,
  "sources": {
    "classified": "data/test_clip/classified.json",
    "rallies": "data/test_clip/rallies.json",
    "bounces": "data/test_clip/bounces.json",
    "players": "data/test_clip/players.parquet",
    "track_roles": "data/test_clip/track_roles.json"
  },
  "ball_source": "synthetic",
  "fps": 30.0,
  "params": {
    "heatmap_bin_ft": 2.0,
    "role_conf_floor": 0.55,
    "net_y_ft": 22.0,
    "kitchen_max_dist_ft": 9.0,
    "baseline_min_dist_ft": 17.0
  },

  "match": {
    "n_rallies": 41,
    "n_shots": 262,
    "n_bounces": 142,
    "match_span_sec": 612.4,
    "rally_length_shots": {
      "mean": 6.39, "median": 5, "max": 18,
      "distribution": {"1": 4, "2-4": 12, "5-8": 18, "9+": 7}
    },
    "rally_duration_sec": {"mean": 4.12, "median": 3.6, "max": 11.2},
    "by_end_reason": {
      "serve-fault": 4, "double-bounce": 5, "ball-out": 11,
      "net-or-short": 4, "ball-not-returned": 12, "ball-off-frame": 3,
      "unknown": 2
    },
    "serve": {
      "n_serves": 41, "n_serve_faults": 4, "serve_fault_rate": 0.098
    },
    "shot_mix": {
      "by_shot_type": {"serve": 41, "dink": 120, "drive": 50, "drop": 28,
                       "lob": 14, "overhead": 6, "unknown": 3},
      "by_stroke_side": {"forehand": 150, "backhand": 96, "unknown": 16},
      "n_volley": 31, "volley_rate": 0.118
    },
    "third_shot": {
      "n_rallies_ge_3_shots": 30,
      "by_shot_type": {"drop": 14, "drive": 9, "dink": 4, "unknown": 3},
      "drop_rate": 0.467
    },
    "bounce_in_out": {"n_in": 130, "n_out": 12, "in_rate": 0.915}
  },

  "error_attribution": {
    "by_owner": {
      "user": 5, "partner": 3,
      "opp_left": 4, "opp_right": 6,
      "team_near": 2, "team_far": 4,
      "unattributed": 1, "unknown": 2
    },
    "by_end_reason_and_owner": [
      {"end_reason": "ball-out", "owner": "user", "owner_kind": "hitter", "count": 2}
    ],
    "notes": [
      "Server / hitter errors attribute to a specific role via track_id.",
      "Receiver errors (double-bounce, ball-not-returned) attribute to the receiving TEAM (team_near/team_far) — the specific receiver of two players is not identifiable in v1.",
      "unknown end_reason -> 'unknown' owner; shots whose track_id maps to no role -> 'unattributed'."
    ]
  },

  "players": {
    "user": {
      "role_confidence": 0.95,
      "role_contaminated": false,
      "handedness": "right",
      "track_ids": [2, 1393, 2857],
      "n_shots": 78,
      "shot_mix": {
        "by_shot_type": {"serve": 12, "dink": 30, "drive": 14, "drop": 18, "unknown": 4},
        "by_stroke_side": {"forehand": 44, "backhand": 30, "unknown": 4},
        "n_volley": 9, "volley_rate": 0.115
      },
      "serve": {"n_serves": 12, "n_serve_faults": 1, "serve_fault_rate": 0.083},
      "errors_committed": 5,
      "mean_post_speed_ftps": 17.3,
      "position": {
        "n_frames": 7080,
        "zone_time_frac": {"kitchen": 0.52, "transition": 0.31, "baseline": 0.17},
        "lateral_time_frac": {"left": 0.30, "center": 0.34, "right": 0.36},
        "area_time_frac": {
          "kitchen-left": 0.16, "kitchen-center": 0.18, "kitchen-right": 0.18,
          "transition-left": 0.09, "transition-center": 0.11, "transition-right": 0.11,
          "baseline-left": 0.05, "baseline-center": 0.05, "baseline-right": 0.07
        },
        "court_coverage_frac": 0.41,
        "mean_court_xy_ft": [9.8, 7.4],
        "movement": {
          "distance_ft_total": 2840.0,
          "distance_ft_per_rally": 69.3,
          "distance_ft_per_min": 278.0
        }
      }
    },
    "partner":   { "...": "same shape" },
    "opp_left":  { "...": "same shape, role_contaminated likely true" },
    "opp_right": { "...": "same shape" },
    "unattributed": {
      "n_shots": 6,
      "note": "shots whose track_id is noise or maps to no role"
    }
  },

  "team": {
    "near": {
      "roles": ["user", "partner"],
      "n_frames_both_present": 6800,
      "both_at_kitchen_frac": 0.38,
      "spacing_ft": {"mean": 9.4, "median": 9.1, "min": 2.1, "max": 18.7},
      "transition_time_frac": {"user": 0.31, "partner": 0.28}
    },
    "far": {
      "roles": ["opp_left", "opp_right"],
      "role_contaminated": true,
      "n_frames_both_present": 5100,
      "both_at_kitchen_frac": 0.29,
      "spacing_ft": {"mean": 10.2, "median": 9.8, "min": 1.4, "max": 21.0},
      "transition_time_frac": {"opp_left": 0.34, "opp_right": 0.30}
    }
  },

  "heatmaps": {
    "grid": {
      "bin_ft": 2.0,
      "x_min_ft": 0.0, "x_max_ft": 20.0, "n_cols": 10,
      "y_min_ft": 0.0, "y_max_ft": 44.0, "n_rows": 22,
      "row_major": true,
      "note": "cell [r][c] covers x in [c*bin, (c+1)*bin), y in [r*bin, (r+1)*bin). Counts only; Stage 11 normalizes + renders."
    },
    "player_position": {
      "user":      [[0, 0, "..."], ["... 22x10 int grid ..."]],
      "partner":   "...",
      "opp_left":  "...",
      "opp_right": "..."
    },
    "ball_landing": [[0, 0, "..."], ["... 22x10 int grid of in+out bounces clipped to extent ..."]]
  },

  "pending_real_ball": {
    "_comment": "Tier-B metrics. STRUCTURALLY present so the output shape is stable and downstream/UI can bind to it now, but VALUE is null in v1: they need trustworthy ball trajectories, and computing them against the synthetic ball would be placeholder-only. Each entry documents exactly what it will contain once ball detection v4 lands. See KNOWN_ISSUES.md 'Synthetic ball' section. To populate: drop the null, compute per the description, and move the key from reliability.pending to reliability.synthetic_gated->real once validated on real data.",
    "forced_vs_unforced_errors": {
      "status": "pending_real_ball",
      "value": null,
      "description": "Splits each committed error into 'forced' (error off a fast incoming ball: shot.features pre-speed >= FORCED_MIN_INCOMING_FTPS) vs 'unforced' (off a slow/neutral ball). Will populate {by_owner: {<role>: {forced: int, unforced: int, unforced_rate: float}}, match: {forced: int, unforced: int, unforced_rate: float}}. High-value input to the Stage 9 USAPA rating (unforced-error rate is a core rating signal)."
    },
    "dink_shot_tolerance": {
      "status": "pending_real_ball",
      "value": null,
      "description": "Patience/consistency: average number of consecutive dinks, and average total shots, sustained before the rally-ending error. Will populate {match: {mean_dinks_before_error: float, mean_shots_before_error: float}, players: {<role>: {mean_dinks_before_error: float, ...}}}. Needs reliable shot-type (dink) + rally sequencing from real ball."
    },
    "third_shot_drop_outcome": {
      "status": "pending_real_ball",
      "value": null,
      "description": "Beyond third_shot.drop_rate: whether each third-shot drop SUCCEEDED (the hitting team won the ensuing kitchen approach / next shot was a controlled dink rather than a forced reset or pop-up). Will populate {n_drops: int, n_successful: int, success_rate: float, by_server_role: {...}}. Needs trustworthy post-drop trajectory + bounce."
    },
    "opponent_backhand_targeting": {
      "status": "pending_real_ball",
      "value": null,
      "description": "Uses roster.json handedness + shot-direction geometry to measure how often a player targets an opponent's BACKHAND, and the rate that wins the point. Will populate {by_role: {<role>: {n_shots_to_opp_backhand: int, frac_to_backhand: float, point_win_rate_when_to_backhand: float}}}. Needs shot direction (impact -> next bounce/contact vector) from real ball + reliable opponent roles (Stage 2.5 v2)."
    }
  },

  "reliability": {
    "synthetic_ball": true,
    "synthetic_gated": ["match.by_end_reason", "match.serve", "match.shot_mix",
                        "match.third_shot", "match.bounce_in_out",
                        "error_attribution", "heatmaps.ball_landing",
                        "players.*.shot_mix", "players.*.serve",
                        "players.*.errors_committed", "players.*.mean_post_speed_ftps"],
    "real_data": ["players.*.position", "players.*.position.movement",
                  "heatmaps.player_position", "team.near", "team.far",
                  "match.rally_length_shots", "match.rally_duration_sec"],
    "pending": ["pending_real_ball.forced_vs_unforced_errors",
                "pending_real_ball.dink_shot_tolerance",
                "pending_real_ball.third_shot_drop_outcome",
                "pending_real_ball.opponent_backhand_targeting"]
  },

  "warnings": [
    "ball_source is 'synthetic': all ball-derived metrics are PLACEHOLDER. See reliability.synthetic_gated.",
    "opp_left role_confidence 0.50 < floor 0.55: opponent stats may be contaminated by adjacent-court tracks (Stage 2.5 known issue)."
  ],
  "stage_version": "0.1.0",
  "completed_at_utc": "2026-05-29T..."
}
```

### Field notes

- **`match.rally_length_shots.distribution`** buckets: `"1"`, `"2-4"`,
  `"5-8"`, `"9+"` (fixed buckets; documented, not configurable in v1).
- **`match.third_shot`** looks at `shot_ids[2]` (0-indexed third shot) of every
  rally with `n_shots >= 3`; `drop_rate` = fraction whose `shot_type == "drop"`.
  The third-shot drop is the canonical pickleball strategy metric.
- **`match.serve.serve_fault_rate`** = `n_serve_faults / n_serves`, where
  `n_serve_faults` = count of rallies with `end_reason == "serve-fault"`,
  `n_serves` = `n_rallies` (every rally starts with exactly one serve).
- **`error_attribution.by_owner`** sums to `n_rallies`. Mapping (from Stage 7's
  documented `end_reason → who-lost` implication):

  | end_reason | error owner_kind | attributed to |
  |---|---|---|
  | `serve-fault` | server | `server_track_id` → role |
  | `ball-out` | hitter | last shot's `track_id` → role |
  | `net-or-short` | hitter | last shot's `track_id` → role |
  | `ball-off-frame` | hitter | last shot's `track_id` → role |
  | `double-bounce` | receiver | receiving team (`team_near`/`team_far`) |
  | `ball-not-returned` | receiver | receiving team (`team_near`/`team_far`) |
  | `unknown` | — | `unknown` |

  Receiving team = the team on the side opposite `end_signals.hitter_side`
  (`hitter_side == "near"` → receiver is `team_far`, and vice-versa). If a role
  lookup yields `noise`/no-role for a server-or-hitter error, owner is
  `unattributed`. `team_near` = {user, partner}; `team_far` = {opp_left,
  opp_right}.
- **`players.<role>.errors_committed`** counts only errors attributable to a
  *specific role* (server/hitter errors mapping to that role). Receiver
  (team-level) errors are NOT split into per-role here — they live in
  `error_attribution.by_owner.team_*`. Documented so the per-role number
  doesn't silently absorb ambiguous attribution.
- **`players.<role>.position` — court-area time (answers "% time in each area
  of the court").** Three views of the same per-frame foot positions
  (`court_x_ft`, `court_y_ft`), each a set of fractions summing to ~1.0 over
  the role's valid in-extent frames:
  - **`zone_time_frac`** — DEPTH (kitchen / transition / baseline) via Stage
    6's `zone_from_court_y` (`dist_from_net = abs(court_y − 22)`; ≤9 kitchen,
    ≥17 baseline, else transition). The strategically important axis.
  - **`lateral_time_frac`** — LEFT / CENTER / RIGHT by `court_x_ft` thirds of
    the 20 ft width: left `[0, 20/3)`, center `[20/3, 40/3)`, right
    `[40/3, 20]`. **Convention:** left = low `court_x_ft`, right = high —
    court-coordinate left/right, matching Stage 2.5's `opp_left`/`opp_right`
    split, NOT player-egocentric. Documented so it's unambiguous.
  - **`area_time_frac`** — the 3×3 cross-product (`<depth>-<lateral>`, 9
    cells), for a full court-area breakdown without going to the fine grid.
  This makes the position metric robust to a player who roams left↔right: it
  reports where they actually spent time rather than assuming a fixed spot.
- **`players.<role>.position.court_coverage_frac`** = fraction of grid cells in
  that role's near/far HALF of the court (10 cols × 11 rows = 110 cells per
  half) that the role's foot position visited at least once. Coverage is over
  the player's own half (a near player isn't expected on the far half).
- **`players.<role>.position.movement`** (REAL): per-frame path length of the
  role's foot court position (`court_x_ft`, `court_y_ft`), summed over valid
  frames. A per-step jitter floor (`MOVE_MIN_STEP_FT`) drops sub-threshold
  deltas so tracking noise isn't integrated as distance. `distance_ft_per_min`
  normalizes by the role's active wall-clock; `distance_ft_per_rally` divides
  rally-frame distance by the number of rallies the role was present for. A
  work-rate / effort signal; ball-independent.
- **`team`** (REAL): team-level positioning, near (`user`+`partner`) and far
  (`opp_left`+`opp_right`). Computed over frames where BOTH of the team's
  players have a valid position (`n_frames_both_present`):
  - **`both_at_kitchen_frac`** — fraction of those frames where BOTH players
    are within kitchen depth (`abs(court_y−22) ≤ KITCHEN_MAX_DIST_FT`). The
    headline "are we holding the line together" doubles metric.
  - **`spacing_ft`** — Euclidean court distance between the two partners
    (mean/median/min/max). Flags being stretched apart or stacked too tight.
  - **`transition_time_frac`** — each player's no-man's-land time, surfaced
    from their per-role `zone_time_frac.transition` for convenience (not a new
    computation). The far team carries `role_contaminated` from Stage 2.5; its
    split between opp_left/opp_right is imprecise but the team aggregate holds
    (see Robustness section).
- **`pending_real_ball`** (Tier B): structurally present, `value: null` in v1.
  Each entry's `description` states exactly what it will contain once real
  ball detection (v4) lands. Listed under `reliability.pending`. Emitting the
  keys now keeps the output shape and any UI bindings stable; the null + status
  make it unambiguous they are NOT yet measured. See KNOWN_ISSUES.md. **No
  Tier-B value is computed against the synthetic ball** — a placeholder number
  here would be more misleading than an explicit null.
- **`role_contaminated`** = `role_confidence < role_conf_floor`. A surfaced
  flag, not a filter — the stats are still computed.
- **`mean_post_speed_ftps`** averages `features.post_speed_ftps` over the
  role's non-serve shots where it's non-null. Synthetic-gated.
- Heatmap grids are plain nested-int arrays (`n_rows × n_cols`), row-major,
  origin at court `(0,0)`. Bounce landings outside `[0,20]×[0,44]` are clipped
  OUT of the grid (not clamped to the edge) and counted in
  `match.bounce_in_out.n_out` regardless. Player positions outside the extent
  (chasing wide / behind baseline) are likewise dropped from the grid but
  still counted in `position.n_frames`.

There is **no separate `.meta.json`** — run metadata lives inside
`metrics.json`, consistent with every other stage output.

### Robustness to left/right movement and side-switching

Players legitimately change their left/right position **between serves** (the
serving team switches service boxes by score) and **during a rally** (partners
rotate / poach, a lob sends them switching sides). Stage 8 is built so this
does not corrupt the metrics — by aggregating over *roles* and *halves*, never
over instantaneous left/right position:

- **Per-role aggregation is position-agnostic.** All per-player stats (shot
  mix, serves, errors, position/area) group `classified.json` shots and
  `players.parquet` frames by `role(track_id)` over the WHOLE clip. A player
  who roams or switches sides simply contributes to both areas in
  `area_time_frac` and to a wider `court_coverage_frac` — no assumption that a
  role sits on one side.
- **User vs. partner identity is continuity/height-based (Stage 2.5), not
  L/R.** A user moving to the other side of their own half keeps the `user`
  role and is not swapped with the partner. So user/partner stats are robust
  to side-switching.
- **Error attribution never uses left/right.** Server/hitter errors attribute
  by the actual striking `track_id`; receiver errors attribute by *half*
  (`team_near` / `team_far`), which is set by `hitter_side` (near vs far) and
  does not change when players shuffle left↔right within a half. A mid-rally
  side-switch therefore can't misattribute the error.
- **The one place L/R matters is the `opp_left` vs `opp_right` LABEL.** That
  split is inherited from Stage 2.5's median-court-x assignment; under frequent
  opponent side-switching the two opponent buckets blur into each other.
  **Team-level (`team_far`) and the *combined* opponent numbers are unaffected**
  — only the split between the two opponents is imprecise. Stage 8 trusts the
  Stage 2.5 roles as given and does NOT re-derive them (no cross-stage
  re-classification). Surfaced via the opponents' `role_confidence` /
  `role_contaminated` flags and noted in Known follow-ups + KNOWN_ISSUES.md.

## Method

1. **Load + validate.** Read all inputs; check `schema_version` on each JSON
   (fail loudly on a version not written for). Pull `ball_source` from
   `classified.json` (cross-check `bounces.json` agrees). Build:
   - `tid → role` map and `role → track_ids` from `track_roles.json`.
   - `shot_id → shot` index from `classified.json`.
2. **Match summary.** Aggregate rally lengths/durations from `rallies.json`;
   `by_end_reason` copied from rally records; shot-mix + volley from
   `classified.json`; third-shot from `shot_ids[2]`; bounce in/out from
   `bounces.json`.
3. **Per-role shot stats.** Group `classified.json` shots by
   `role(shot.track_id)`. Unmapped/noise → `unattributed`. Compute shot mix,
   volley rate, serve counts, mean post-speed per role.
4. **Error attribution.** For each rally, resolve owner via the table above
   (server/hitter → role; receiver → team; unknown → unknown). Tally
   `by_owner` and `by_end_reason_and_owner`; add `errors_committed` to each
   role for the role-attributable kinds.
5. **Position / coverage (real data).** From `players.parquet`, for each
   role's `track_ids`, over rows with `transient == False` and a valid
   (non-NaN, in-extent) foot position: bin `court_y_ft` into
   kitchen/transition/baseline via Stage 6's `zone_from_court_y`
   (`dist_from_net = abs(court_y - 22)`; ≤9 kitchen, ≥17 baseline, else
   transition) for `zone_time_frac`; bin `court_x_ft` into left/center/right
   thirds for `lateral_time_frac`; tally the depth×lateral cross-product for
   `area_time_frac`; accumulate the position heatmap; compute
   `court_coverage_frac` over the role's own half.
   > Reuses Stage 6's zone constants verbatim (`KITCHEN_MAX_DIST_FT=9.0`,
   > `BASELINE_MIN_DIST_FT=17.0`, `NET_Y_FT=22.0`) so a shot's `contact_zone`
   > and a player's position zone always agree. Constants are duplicated with
   > a citation comment rather than imported, per the no-cross-stage-coupling
   > rule (architecture rule #2).
6. **Movement (real data).** For each role, sort its valid foot positions by
   frame, sum per-frame Euclidean deltas above `MOVE_MIN_STEP_FT`; normalize
   to per-minute and per-rally.
7. **Team positioning (real data).** For near (`user`+`partner`) and far
   (`opp_left`+`opp_right`): over frames where both players have a valid
   position, compute `both_at_kitchen_frac` and `spacing_ft`; copy each
   player's transition fraction into `transition_time_frac`.
8. **Ball-landing heatmap.** Bin each bounce's `court_xy_ft` into the grid
   (clip out-of-extent).
9. **Pending (Tier B).** Emit the `pending_real_ball` block verbatim with
   `value: null` (no computation against synthetic ball).
10. **Reliability + warnings.** Emit the `reliability` map (incl. `pending`);
    loud synthetic warning when `ball_source == "synthetic"`; per-role
    contamination warnings.
11. **Write** `metrics.json` (refuse to overwrite without `--force`).

## Defenses against placeholder / bad data

- **Propagates `ball_source`** and a `reliability` map that names exactly which
  metric families are synthetic-gated vs real. Loud warning + WARNING log when
  synthetic. This is the most important defense — Stage 8 is where a reader is
  most tempted to trust the aggregate numbers.
- **Reconciliation is the correctness contract.** Per-role + unattributed shot
  counts MUST sum to `match.n_shots`; `error_attribution.by_owner` MUST sum to
  `n_rallies`; `by_end_reason` MUST equal `rallies.json.stats.by_end_reason`.
  Asserted in code (loud failure on drift) and in the smoke test.
- **Never force-assigns** a contaminated role or fabricates a receiver. Honest
  `unattributed` / `team_*` / `unknown` buckets.
- **Missing optional inputs degrade loudly** (see Inputs → Degradation), never
  silently.
- **Output exists without `--force`** → `FileExistsError`.

## Edge cases

- **A shot's `track_id` not in any role** (noise / out-of-court contamination)
  → `unattributed`; counted, not dropped.
- **A rally's server/hitter track maps to `noise`** → that error → owner
  `unattributed` (we don't guess a role).
- **`hitter_side` null in `end_signals`** (degenerate court projection) → a
  receiver error can't pick a team → owner `unknown`.
- **Zero rallies / zero shots** → empty families, valid file, warnings.
- **A role with zero tracks** (e.g. singles clip, or Stage 2.5 found <4 roles)
  → that player block emitted with zeroed stats + a warning; no crash.
- **Player rows with `court_y_ft` NaN / out of `[0,44]`** → excluded from zone
  fractions and heatmap, but the row is counted in `n_frames` only if it has a
  valid position (NaN positions skipped entirely, logged in a debug count).
- **Required input missing/malformed** → fail loudly naming the file.

## Configuration (defaults; tuned against smoke test)

```python
HEATMAP_BIN_FT       = 2.0    # court grid bin -> 10 cols (x) x 22 rows (y)
ROLE_CONF_FLOOR      = 0.55   # role_confidence below this -> role_contaminated
NET_Y_FT             = 22.0   # net line (= length_ft / 2)   [from Stage 6]
KITCHEN_MAX_DIST_FT  = 9.0    # effective kitchen depth from net  [from Stage 6]
BASELINE_MIN_DIST_FT = 17.0   # within ~5ft of own baseline -> baseline [from S6]
COURT_LEN_FT         = 44.0
COURT_WID_FT         = 20.0
RALLY_LEN_BUCKETS    = ["1", "2-4", "5-8", "9+"]
MOVE_MIN_STEP_FT     = 0.25   # per-frame foot delta below this = jitter, not movement
# Tier B (pending real ball; documented, not used in v1):
# FORCED_MIN_INCOMING_FTPS = 25.0  # incoming speed above which an error is 'forced'
```

## Smoke test

`stages/compute_metrics/test_compute_metrics.py`, against `data/test_clip/`.
Stage 8 is aggregation, so the test is mostly **reconciliation + schema +
reliability**, plus a couple of ground-truth ties to the synth truth.

Pipeline prefix (reuse the chain from the handoff): regenerate synth ball →
Stage 5 → 5.5 → 6 → 7 → 2.5 → **8**.

Assertions:

1. **Schema.** `metrics.json` parses, `schema_version == 1`, all documented
   top-level keys present, dtypes correct, every `by_*` value an int ≥ 0.
2. **Shot reconciliation.** `sum(players.<role>.n_shots) +
   players.unattributed.n_shots == match.n_shots == len(classified.shots)`.
3. **Error reconciliation.** `sum(error_attribution.by_owner.values()) ==
   match.n_rallies`. Every `owner` is a known role / `team_near` / `team_far`
   / `unattributed` / `unknown`.
4. **End-reason passthrough.** `match.by_end_reason ==
   rallies.json.stats.by_end_reason` exactly.
5. **Serve metric.** `match.serve.n_serves == match.n_rallies`;
   `n_serve_faults == by_end_reason["serve-fault"]`; rate in [0,1].
6. **Rally-length stats** recomputed from `rallies.json` match
   `match.rally_length_shots` (mean within float tol; distribution buckets
   exact); distribution counts sum to `n_rallies`.
7. **Position stats (real data).** For each non-empty role,
   `zone_time_frac`, `lateral_time_frac`, and `area_time_frac` each have
   values in [0,1] and sum to ~1.0 (±1e-6); `area_time_frac` is consistent
   with its marginals (summing the 3 lateral cells of a depth row equals that
   row's `zone_time_frac`, ±1e-6); `court_coverage_frac` in [0,1];
   `position.n_frames > 0` for the `user` role (clicks guarantee user frames).
   Independent of `ball_source`.
   - **Movement:** `distance_ft_total >= 0`; `distance_ft_per_min` and
     `distance_ft_per_rally` finite and ≥ 0 for the `user` role.
   - **Team:** `team.near.both_at_kitchen_frac` in [0,1]; `spacing_ft.mean ≥
     0` and `min ≤ median ≤ max`; `n_frames_both_present > 0` for near (user
     + partner both present somewhere). All independent of `ball_source`.
8. **Heatmap integrity.** Each `player_position[role]` grid is `n_rows ×
   n_cols`; its sum equals the count of that role's in-extent valid foot
   positions (cross-checked against the same filter used for `n_frames`).
   `ball_landing` sum == count of bounces whose `court_xy_ft` falls in extent.
9. **Reliability + propagation.** `reliability.synthetic_ball == true`;
   synthetic warning present; `players.*.position`, `team.near`/`team.far`,
   and `heatmaps.player_position` listed under `reliability.real_data`, NOT
   under `synthetic_gated`.
9b. **Pending block.** Every `pending_real_ball.*` entry has `value == null`,
    `status == "pending_real_ball"`, and a non-empty `description`; every
    pending key is listed under `reliability.pending`. (Guards against a
    synthetic value leaking into a Tier-B metric.)
10. **Role-confidence flag.** Any role with `role_confidence <
    role_conf_floor` has `role_contaminated == true` and a warning naming it.
11. **Degradation variant.** Re-run with `track_roles.json` temporarily
    hidden (or a `--no-roles` test hook): completes, `user` populated from
    `is_user`, opponents empty, warning present, match-level metrics
    unchanged from the full run.
12. **Truth tie (light).** `match.n_rallies` equals the detected rally count
    from Stage 7; `match.bounce_in_out.n_in + n_out == len(bounces.json
    bounces with court projection)`. (We don't grade per-player accuracy —
    no per-player ground truth on real tracks; reconciliation is the gate.)

## Stage version

`0.1.0` (initial). Increment minor for behavior changes preserving the
`metrics.json` schema; bump `schema_version` for breaking schema changes.

## Out of scope (deferred)

- **Winner-side / point-win attribution beyond error owner.** v1 attributes
  the *error* (who lost the rally); it does not credit the *winner* of each
  point or compute serve-hold / break stats. Needs reliable role tracking +
  score modeling. Stage 9 territory or a later metrics pass.
- **Per-receiver error splitting.** Receiver errors attribute to a team, not
  one of the two players. Splitting needs shot-target geometry (which half the
  ball went to) + role positions at the rally-ending frame.
- **Rally-by-rally timeline export.** Stage 11 can join `rallies.json` itself;
  Stage 8 emits aggregates, not a per-rally table.
- **Tier-B ball-derived metrics** (forced/unforced errors, dink/shot
  tolerance, third-shot-drop outcome, opponent-backhand targeting) — structure
  is emitted (`pending_real_ball`, null) but values wait for real ball
  detection v4. See KNOWN_ISSUES.md.
- **Pose-derived technique metrics + cross-video trends** (Tier C) — split-step
  timing, posture, contact consistency, improvement curves across sessions.
  Their own future stages; written up in ARCHITECTURE.md "Future / proposed
  stages". Not part of Stage 8.
- **Ball-speed accuracy.** `mean_post_speed_ftps` is synthetic-gated and
  perspective-approximate (inherits Stage 6's coarse px/ft scaling). Re-tune
  on real ball detection (v4).
- **Configurable rally-length buckets / heatmap normalization.** Fixed in v1.

## Known follow-ups

- **Opponent role contamination** (Stage 2.5) directly degrades `opp_left` /
  `opp_right` stats; surfaced via `role_contaminated` + warnings. Tightens once
  Stage 2.5 v2 (multi-region color, far-side filter) lands.
- **Everything ball-derived is placeholder** until real ball detection (v4).
  Re-validate the whole metric suite on real (noisy, gappy) trajectories then;
  the synth-derived numbers will shift.
- **`is_user`/role coverage** bounds per-user completeness — Stage 2.5 raised
  user coverage to ~0.82–0.99; gaps still send some user shots to
  `unattributed`. Improves with Stage 2.5 follow-ups + the queued Stage 3/6
  role re-wire.

## Architecture note

Stage 8 was already in the pipeline diagram (one of the original 11 stages);
this contract takes it from "not started" to "implemented + smoke-tested".
Pipeline count stays at 13. On approval, `ARCHITECTURE.md`'s "Stages 8–11: not
started" line becomes "Stage 8 implemented; Stages 9–11 not started".

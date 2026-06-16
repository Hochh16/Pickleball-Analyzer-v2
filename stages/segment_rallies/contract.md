# Stage 7 — Segment Rallies

**Status:** IMPLEMENTED; **v0.2.0 real-ball adapted** (run on pb_2min,
operator-validated rally boundaries). Groups the shot stream into rallies and
tags each rally with how it ended. NOTE: the v0.1.0 assumption that "boundaries
are trivial because Stage 5 flags every serve" proved **false on the real ball**
(serves under-detected, shots missed) — boundaries now come from the **ball
going out of play** (see "Real-ball adaptations (v0.2.0)" below). end_reason
leans on `bounces.json`, but real bounce recall is low, so most real-ball
end_reasons are currently `unknown` (honest).

**Role-blind v1** (per the scope decision at the start of the session): no
winner_side attribution, no track_roles.json consumption. Winner-side and
per-player error stats are Stage 8's job.

## Purpose

Take the per-shot stream from Stage 6 (`classified.json`) and the per-bounce
stream from Stage 5.5 (`bounces.json`), and emit `rallies.json`: one record
per rally with `start_frame`, `end_frame`, the `shot_ids` it contains, the
`server_track_id`, and a categorical **`end_reason`** describing why play
stopped. Rallies are the natural unit for downstream metrics: serve-fault
rate, average rally length, point-ending shot mix, error rate by player.

Stage 7 is **rule-based on the shot+bounce stream**, same pipeline
philosophy as the rest. Honest `unknown` when signals are weak — better than
a confident guess.

## Place in the architecture

```
classified.json (S6) + bounces.json (S5.5) + court.json (S1)
        │
        ▼
   [7] segment_rallies ──► rallies.json
```

Per-video, file-path I/O, standalone CLI:
`python -m stages.segment_rallies.segment_rallies <video_folder>`.

> **DECISION (folder name).** Code and contract live at
> `stages/segment_rallies/` (importable; Python module names can't start
> with a digit — same convention as `detect_shots/`, `detect_bounces/`,
> `classify_shots/`, `classify_tracks/`). The contract goes directly in
> the implementation folder, matching the post-approval convention used
> for every implemented stage.

> **DECISION (role-blind v1).** Stage 7 does NOT consume `track_roles.json`
> and does NOT emit `winner_side`. Stage 2.5's v1 opponent classification
> has known contamination (low confidence), so building winner attribution
> on top of it would propagate that uncertainty into rally-level stats.
> Winner-side is deferred to Stage 8 (metrics), where it can be added once
> the role-classification follow-ups land. v1's `server_track_id` is
> still emitted because it comes from the serve shot's `track_id`
> directly — no role inference needed.

## Inputs

Per-video folder positional argument.

| File | From | Stage 7 reads |
|---|---|---|
| `classified.json` | Stage 6 | the shot list (shot_id, frame, t_sec, track_id, is_user, is_serve, **impact_court_xy_ft**) — boundaries + assignment + serve attribution + hitter side for net-or-short detection + server side for serve-fault kitchen check |
| `bounces.json` | Stage 5.5 | bounces (bounce_id, frame, between_shots, is_in_court, **court_xy_ft**) — end_reason classification (in/out, side, kitchen) |
| `court.json` | Stage 1 | `video.fps` (fallback if classified.json doesn't expose it) |

**Not** read: `ball.parquet` (don't need raw trajectory; bounces are
already extracted), `players.parquet` (no role inference), `poses.parquet`
(no body landmarks needed).

CLI flags (defaults in Configuration): `--force`, `--log-level`,
`--serve-fault-max-frames`, `--rally-end-grace-frames`.

## Output — `rallies.json`

```json
{
  "schema_version": 1,
  "source_classified": "data/test_clip/classified.json",
  "source_bounces": "data/test_clip/bounces.json",
  "ball_source": "synthetic",
  "fps": 30.0,
  "params": {
    "serve_fault_max_frames": 60,
    "rally_end_grace_frames": 45
  },
  "rallies": [
    {
      "rally_id": 0,
      "start_frame": 1052,
      "end_frame": 1183,
      "start_t_sec": 35.07,
      "end_t_sec": 39.43,
      "duration_sec": 4.37,
      "shot_ids": [0, 1, 2, 3, 4],
      "n_shots": 5,
      "serve_shot_id": 0,
      "server_track_id": 2,
      "server_is_user": false,
      "end_reason": "ball-out",
      "end_reason_confidence": 0.85,
      "ending_bounce_id": 12,
      "end_signals": {
        "n_bounces_after_last_shot": 1,
        "last_bounce_in_court": false,
        "last_bounce_out_side": "far",
        "last_bounce_side": "far",
        "last_bounce_in_kitchen": false,
        "hitter_side": "near",
        "server_side": "near",
        "frames_to_next_serve": 87
      }
    }
  ],
  "stats": {
    "n_rallies": 41,
    "by_end_reason": {
      "serve-fault": 4,
      "double-bounce": 5,
      "ball-out": 11,
      "net-or-short": 4,
      "ball-not-returned": 12,
      "ball-off-frame": 3,
      "unknown": 2
    },
    "total_shots_in_rallies": 262,
    "unassigned_shots": 0,
    "mean_rally_length": 6.39,
    "mean_rally_duration_sec": 4.12
  },
  "warnings": [
    "ball_source is 'synthetic': rally end_reasons are derived from PLACEHOLDER ball data."
  ],
  "stage_version": "0.1.0",
  "completed_at_utc": "2026-05-28T..."
}
```

Field notes:

- `rallies` is **ordered by `start_frame` ascending**; `rally_id` is the
  index in that order.
- `start_frame` = the serve shot's frame. `end_frame` = `max(last shot's
  frame, ending bounce's frame)` — extends through the rally-ending bounce
  if one exists, so the rally's frame span covers the visible
  end-of-point event.
- `shot_ids`: shot_ids belonging to this rally, in frame order. Always
  contains the serve as the first element.
- `serve_shot_id`: the `shot_id` of the rally's serve (always
  `shot_ids[0]` in v1).
- `server_track_id` / `server_is_user`: carried through from the serve
  shot. Stable real attribution (the serve's striking player is always
  trustworthy from Stage 5).
- `end_reason` ∈ `{"serve-fault", "double-bounce", "ball-out",
  "net-or-short", "ball-not-returned", "ball-off-frame", "unknown"}`.
  See Classification below. Each carries a *who-lost-the-point*
  implication used downstream by Stage 8 (error attribution):
  - **Server's error:** `serve-fault`
  - **Hitter's error** (the last shot's striker): `ball-out`,
    `net-or-short`, `ball-off-frame` (the ball flew off-frame, most
    likely because the hitter sent it wide)
  - **Receiver's error** (the player who failed to return): `double-bounce`,
    `ball-not-returned`
  - **Ambiguous:** `unknown`
- `end_reason_confidence` ∈ [0, 1]: reflects signal clarity (out-of-court
  bounce → high; no bounce signal → low).
- `ending_bounce_id`: the `bounce_id` (from bounces.json) of the bounce
  that drove the classification, or `null` if no bounce was decisive
  (e.g., end_reason="ball-not-returned" with zero bounces after last shot).
- `end_signals`: the raw features used for classification. Exposed for
  downstream debugging and for Stage 8 to reuse without re-deriving.
  - `n_bounces_after_last_shot`: count of bounces with
    `between_shots[0] == last_shot_id` (i.e., bounces between the
    rally's last shot and the next shot in shots.json, whether casual
    or serve).
  - `last_bounce_in_court`: `is_in_court` of the **last** such bounce
    (or null if none).
  - `last_bounce_out_side`: `out_side` of the last bounce (null if
    in-court or no bounce).
  - `last_bounce_side`: `"near"` | `"far"` | null. Which side of the
    net the last post-last-shot bounce was on (derived from bounce
    `court_xy_ft.y` vs net at 22 ft). Used for `net-or-short` detection.
  - `last_bounce_in_kitchen`: bool | null. Whether the last bounce
    sits in the receiver's kitchen (used for serve-fault detection).
  - `hitter_side`: `"near"` | `"far"` | null. The last shot's hitter
    side (from its `impact_court_xy_ft.y` vs 22). Null if degenerate.
  - `server_side`: `"near"` | `"far"` | null. The serve shot's hitter
    side. (Same player throughout the rally as the server.)
  - `frames_to_next_serve`: frames from this rally's last shot to the
    next rally's serve (or null if this is the last rally in the clip).

There is **no separate `.meta.json`** — run metadata lives inside
`rallies.json`, consistent with `shots.json` / `bounces.json` /
`classified.json`.

## Classification method

### Step 1 — Boundary segmentation

1. Sort shots by frame (they should already be sorted; assert).
2. For each shot `s` with `s.is_serve == true`, start a new rally.
   Subsequent shots (in frame order) up to but not including the next
   serve all belong to this rally.
3. If shots exist BEFORE the first serve, emit a warning and **drop**
   them as pre-rally orphans (counted in `unassigned_shots`). This is
   rare — the start of the analyzed range usually catches the first
   serve via Stage 5's appearance signal — but possible.

### Step 2 — Per-rally end_reason

For each rally, find all bounces `b` with
`b.between_shots[0] == last_shot_id` — bounces between the rally's
last shot and the **next shot in shots.json** (whether casual hit or
serve; both are handled identically). Derive side data from court
projections:
- `hitter_side` = side of last_shot's `impact_court_xy_ft.y` vs net (22 ft).
- `server_side` = side of serve_shot's `impact_court_xy_ft.y` vs net.
- `last_bounce_side` = side of last bounce's `court_xy_ft.y` vs net.
- `last_bounce_in_kitchen` = bounce is in the **receiver's** kitchen,
  where the receiver's kitchen is `y in [22, 29]` if server is near,
  or `y in [15, 22]` if server is far.

Apply rules in order (first match wins):

1. **`serve-fault`** — `n_shots == 1` (rally has only the serve, no
   return). Sub-classify by signal strength:
   - Post-serve bounce is `is_in_court == false` → confidence 0.9
     (clear out-of-court serve).
   - Post-serve bounce is in the receiver's kitchen
     (`last_bounce_in_kitchen == true`) → confidence 0.9 (clear
     kitchen fault).
   - `frames_to_next_serve` is positive AND
     `<= serve_fault_max_frames` → confidence 0.7 (quick next serve
     implies fault).
   - Else (n_shots == 1 with no further signal) → still
     `serve-fault`, confidence 0.5.

   > A pickleball serve has only one attempt and must land in the
   > receiver's court past the kitchen line. v1 detects the
   > **kitchen-fault** signal (bounce in receiver's kitchen) and the
   > **out-of-court** signal. It does NOT check the diagonal
   > service-box (right vs left), which would need server-alternation
   > tracking across points — see "Out of scope" below.

2. **`double-bounce`** — `n_bounces_after_last_shot >= 2`. Two or more
   bounces between this rally's last shot and the next shot in
   shots.json. Receiver let the ball bounce twice and failed to play
   it. Confidence 0.85.

3. **`net-or-short`** — `n_bounces_after_last_shot >= 1` AND the last
   such bounce is **in-court** AND **on the hitter's side of the net**
   (`last_bounce_in_court == true` AND
   `last_bounce_side == hitter_side`). The ball didn't make it
   across — either it hit the net and fell back, or it was short and
   bounced before reaching the receiver. Hitter's error. Confidence
   0.8. (Out-of-court bounces that happen to project to the hitter's
   side fall through to rule 4 / `ball-out` — they're still a hitter
   error but the CAUSE is "wide" not "net".)

4. **`ball-out`** — `n_bounces_after_last_shot == 1` AND that bounce
   has `is_in_court == false` AND (`last_bounce_side != hitter_side`
   OR sides indeterminate). The hitter sent it wide/long on the
   receiver's side or beyond. Hitter's error. Confidence 0.85.

5. **`ball-not-returned`** — `n_bounces_after_last_shot >= 1` AND the
   last such bounce is `is_in_court == true` AND on the receiver's
   side (`last_bounce_side != hitter_side`). Legal landing on the
   receiver's side, but no return came. Receiver's error.
   Confidence 0.75.

6. **`ball-off-frame`** — `n_bounces_after_last_shot == 0` AND
   `frames_to_next_serve` is positive (play actually stopped). The
   ball flew off-frame without producing a detectable bounce — most
   commonly because the hitter sent it wide and it left the camera
   frame before bouncing. Likely **hitter's** error but confidence is
   capped because the no-bounce signal is ambiguous (could also be a
   missed volley by the receiver, or the ball going behind the camera).
   Confidence 0.5.

7. **`unknown`** — every other case: last rally of the clip with no
   follow-up signal; degenerate court projection so sides can't be
   determined; ambiguous data. Confidence 0.3.

`ending_bounce_id` is the `bounce_id` of:
- The decisive out-of-court bounce for `ball-out` and `serve-fault`
  (out-of-court branch);
- The kitchen bounce for `serve-fault` (kitchen branch);
- The last bounce for `double-bounce`;
- The single bounce for `net-or-short` and `ball-not-returned`;
- `null` for `ball-off-frame`, `unknown`, and `serve-fault` (no-bounce
  signal branches).

> **DECISION (split zero-bounce from one-bounce).** Per review,
> zero-bounce-no-return (`ball-off-frame`) and one-bounce-no-return
> (`ball-not-returned`) have different error attribution (likely
> hitter's vs receiver's), so they're separate end_reasons. Stage 8
> uses this split for per-player error stats.

> **DECISION (net-or-short as its own end_reason).** Per review, a
> ball bouncing on the hitter's side (net hit or short shot) is a
> distinct hitter error and worth surfacing now — accounting for it
> later would mean re-deriving the side check from `end_signals`.
> Detected via `last_bounce_side == hitter_side`.

> **DECISION (no diagonal service-box check in v1).** v1 only checks
> kitchen + out-of-court for serve-faults. The wrong-half check (serve
> hit the receiver's right box when it should have hit the left) needs
> server-alternation tracking across points, which Stage 7 doesn't
> have. This is a follow-up.

## Defenses against placeholder / bad data

- **Requires `classified.json` AND `bounces.json`.** Missing either →
  fail loudly. The pipeline order means both should be present.
- **Surfaces ball_source.** Propagated from `classified.json` (or
  `bounces.json` — they should agree). When `"synthetic"`, loud warning
  in `warnings[]` and a WARNING log line. Downstream metrics must keep
  treating end_reasons as placeholder-derived until real ball data
  exists.
- **No shots at all** → complete with `rallies: []` and a warning.
- **No serves detected** → complete with `rallies: []`, all shots in
  `unassigned_shots`, warning. (Probably a degraded Stage 5 run; we
  don't fabricate a rally.)
- **Empty `bounces.json`** (zero bounces) → still run; every rally
  falls into `ball-off-frame` or `unknown` (no bounce signal to
  classify).
- **`bounces.json` schema_version mismatch** → fail loudly naming the
  file.
- **Output exists without `--force`** → `FileExistsError`.

## Edge cases

- **Pre-first-serve shots.** Dropped with a warning, counted in
  `unassigned_shots`. Common when the clip's analyzed range starts
  mid-rally.
- **Last rally has no next serve.** `frames_to_next_serve = null`. End
  classification still works from post-last-shot bounces alone; if
  there are no bounces either, `end_reason = "unknown"`.
- **A serve in the middle that's actually a Stage-5 false serve.** We
  trust `is_serve` from classified.json. If Stage 5 mis-flagged a
  shot, the rally split here is incorrect — out of scope for Stage 7.
- **`between_shots[0] == null` bounces** (bounce before any shot).
  Ignored — they're not "after a shot in any rally".
- **Bounces with `between_shots[0]` pointing at a shot in a different
  rally** (because the next-shot bridge crosses a serve boundary).
  These are post-last-shot bounces of the *earlier* rally, so they
  correctly belong to that rally — the index keys on
  `between_shots[0]`, not `[1]`.
- **Required input missing/malformed** → fail loudly naming the file.

## Configuration (defaults; tuned against smoke test)

```python
SERVE_FAULT_MAX_FRAMES  = 60   # quick-next-serve = serve fault (~2s at 30fps)
RALLY_END_GRACE_FRAMES  = 45   # how far past the last shot we scan for bounces
                               # belonging to this rally (~1.5s at 30fps)
NET_Y_FT                = 22.0  # net line in court coordinates
KITCHEN_DEPTH_FT        = 7.0   # kitchen extends 7 ft from net (each side)
END_REASONS = {"serve-fault", "double-bounce", "ball-out",
               "net-or-short", "ball-not-returned", "ball-off-frame",
               "unknown"}
```

> `RALLY_END_GRACE_FRAMES` isn't strictly needed when bounces.json's
> `between_shots[0] == last_shot_id` already identifies them
> structurally — but it documents the implicit assumption that a bounce
> further than ~1.5s after the last shot is unlikely to be from this
> rally and may be background motion. Reserved for v2 if needed; v1
> uses the structural between_shots index without a frame cap.

## Test fixture: synthetic ball ground-truth extension

`tools/synth_ball.py` currently builds rallies of 3-9 shots and ends
them with a follow-through (the last shot's outgoing trajectory in a
new direction, then the ball goes invisible). It does **NOT** model
rally-ending bounces — so today, every synthetic rally would land in
`ball-not-returned` (no-bounce variant), giving the smoke test no
basis to grade end_reason accuracy.

### Extension — rally-ending events

Two parts: serve-fault rallies (single-shot) and end-pattern rendering
for multi-shot rallies.

#### Part A — Serve-fault rallies (~10% of all rallies)

Some fraction of rallies (`SERVE_FAULT_RALLY_PROB`, default 0.10) are
single-shot rallies: only the serve, no return. The serve's bounce is
rendered as either out-of-court or in the receiver's kitchen,
producing a clear serve-fault signal:

| Pattern | Prob (of serve-faults) | Truth `end_reason` | Bounce location |
|---|---|---|---|
| `serve_fault_out` | 0.50 | `serve-fault` | Out-of-court (beyond receiver's baseline) |
| `serve_fault_kitchen` | 0.50 | `serve-fault` | In receiver's kitchen (y in [22, 29] if server is near; y in [15, 22] if server is far) |

For these rallies, the existing rally-building logic is short-circuited
to `n_shots == 1` and the serve's outgoing trajectory ends at a single
ground bounce in the fault location. No follow-up shots are added.

#### Part B — End-pattern rendering for multi-shot rallies (~90%)

After each multi-shot rally's last shot's follow-through, render one
of five rally-ending patterns chosen by weighted random:

| Pattern | Prob | Truth `end_reason` | Bounces added |
|---|---|---|---|
| `in_court_bounce_receiver` | 0.33 | `ball-not-returned` | 1 in-court, receiver's side |
| `out_of_court_bounce` | 0.28 | `ball-out` | 1 out-of-court (receiver side or beyond) |
| `double_bounce` | 0.17 | `double-bounce` | 2 in-court on receiver's side (~10 frames apart) |
| `hitter_side_bounce` | 0.11 | `net-or-short` | 1 in-court, hitter's side (near net) |
| `no_bounce` | 0.11 | `ball-off-frame` | 0 (ball flies off; current behavior) |

The bounces are appended to the existing `out_bounced`/`bounce_pt`
machinery — same trajectory rendering (two linear legs with a kink),
just placed AFTER the last shot's follow-through endpoint, in a
location consistent with the trajectory direction (e.g.,
`out_of_court_bounce` lands beyond the court polygon in the
trajectory's forward direction; `hitter_side_bounce` lands a few feet
from the net on the hitter's side, simulating a net hit). They become
part of `ball_synth_truth.json["bounces"]` exactly like other bounces.

### Extension — `rallies[]` in `ball_synth_truth.json`

Truth file gains a top-level `rallies` block:

```json
{
  "schema_version": 3,
  "n_rallies": 49,
  "rallies": [
    {
      "rally_id": 0,
      "start_frame": 1052,
      "end_frame": 1183,
      "shot_ids_truth": [0, 1, 2, 3, 4],
      "end_reason": "ball-out",
      "end_pattern": "out_of_court_bounce",
      "ending_bounce_id_truth": 12
    },
    {
      "rally_id": 1,
      "start_frame": 1240,
      "end_frame": 1268,
      "shot_ids_truth": [5],
      "end_reason": "serve-fault",
      "end_pattern": "serve_fault_kitchen",
      "ending_bounce_id_truth": 14
    }
  ]
}
```

`schema_version` bumps 2 → 3 (additive). `shot_ids_truth` is the list
of generator hit_ids in the rally (each hit becomes a shot in
shots.json with high but not perfect correspondence — Stage 5
sometimes misses). `ending_bounce_id_truth` is the `bounce_id` of the
rally-ending bounce in the truth `bounces[]` list, or `null` for
patterns that don't produce a bounce (`no_bounce`).

## Smoke test

`stages/segment_rallies/test_segment_rallies.py`, against
`data/test_clip/`:

1. Regenerate ball+truth (extended): `synth_ball.py --seed 1234 --force`
2. Run Stage 5 → `shots.json`
3. Run Stage 5.5 → `bounces.json`
4. Run Stage 6 → `classified.json`
5. Run Stage 7 → `rallies.json`
6. Assert:
   1. **Schema:** `rallies.json` parses, `schema_version=1`, fields/
      dtypes correct, rallies sorted by `start_frame`, `rally_id`
      contiguous from 0, every `end_reason` ∈ `END_REASONS`.
   2. **Propagation:** `ball_source == "synthetic"` and placeholder
      warning present.
   3. **Boundary correctness:** every shot with `is_serve == true` in
      `classified.json` appears as some rally's `serve_shot_id`. Zero
      "missing" serves.
   4. **Shot assignment:** every non-pre-rally shot appears in
      exactly one rally's `shot_ids`. No overlaps, no orphans.
       Pre-rally shots (before first serve) accounted for in
       `stats.unassigned_shots`.
   5. **Boundaries match truth:** for each truth rally, find the
      detected rally whose `start_frame` is within ±MATCH_WINDOW
      (6 frames) of the truth `start_frame`. Boundary recovery ≥ 0.90.
   6. **end_reason accuracy ≥ 0.70:** on matched (detected, truth)
      rally pairs, the detected `end_reason` matches truth ≥ 70% of
      the time. Bar is set lower than overall recall/precision bars
      because end_reason classification depends on every upstream
      stage being right (Stage 5 catches the last shot, Stage 5.5
      catches the rally-ending bounce, classify_shots doesn't
      reorder anything). 0.70 is the realistic floor given
      synthetic noise.
   7. **Internal consistency:**
      - Every `end_reason == "ball-out"` rally has at least one
        bounce in `bounces.json` with `between_shots[0] ==
        last_shot_id` AND `is_in_court == false`.
      - Every `end_reason == "double-bounce"` rally has 2+ such
        bounces.
      - Every `end_reason == "serve-fault"` rally has `n_shots == 1`.
      - Every `end_reason == "net-or-short"` rally has a bounce
        with `last_bounce_side == hitter_side` in `end_signals`.
      - Every `end_reason == "ball-off-frame"` rally has
        `n_bounces_after_last_shot == 0`.
   8. **`stats.by_end_reason`** has ≥ 4 non-zero categories (sanity:
      not collapsing all rallies into one or two buckets; the 6
      generated patterns should produce at least 4 distinct
      end_reasons after detection noise).
   9. **Gap variant:** with `--gap-frac 0.2`, the full pipeline
      completes without crash and Stage 7 still emits a non-empty
      `rallies[]`.

## Real-ball adaptations (v0.2.0)

The v0.1.0 contract above assumes Stage 5 flags every serve and emits clean
shots. On the real v4 ball neither holds: serves are under-detected, shots are
missed (ball-detection recall), and the shot **court projection is unusable**.
v0.2.0 adapts (all gated to the real ball; synthetic keeps the v0.1.0 path,
smoke unchanged):

1. **Rally boundaries from the ball going OUT OF PLAY, not `is_serve`.** Pure
   `is_serve` boundaries merge many points into mega-rallies (serves missed),
   and a raw inter-shot **time-gap** falsely splits a rally wherever a hit was
   missed. The robust, **general physical signal** is ball visibility: during a
   point the ball is in flight (`visible|interpolated` almost every frame, tiny
   <~0.25s absences); between points it is dead (picked up / reset) for 3–4s. A
   new rally starts when the ball had a sustained not-in-play run
   (`>= ball_dead_run_frames` = `BALL_DEAD_RUN_SEC`·fps, default 1.5s) since the
   previous shot, OR at a flagged serve. A missed shot leaves the ball flying →
   **no false split**. Stage 7 therefore **reads `ball.parquet`** on the real
   ball (added input). Validated on pb_2min (operator-confirmed rally banners).
2. **Side from `hitter_side` (Stage 5), not the ball projection.**
   `impact_court_xy_ft` is garbage for an airborne contact (see Stage 5 v0.3.0);
   `hitter_side`/`server_side` now come from Stage 5's `hitter_side` (the hitting
   player's ground position), falling back to the old projection only if absent.
3. **Zero-bounce end_reason → `unknown` (not `ball-off-frame`).** With real bounce
   recall, no detected rally-ending bounce almost always means the bounce was
   *missed*, not that the ball flew off-frame — so the off-frame inference (and
   its hitter-error attribution) isn't warranted. Synthetic (clean ball) keeps
   `ball-off-frame`. **Most real end_reasons are currently `unknown`** — honest;
   end_reason becomes useful only when bounce recall improves (Stage 4/5.5).
4. **Courtesy/non-rally feed drop.** A single non-serve shot bounded by dead time
   is dropped (counted in `unassigned_shots`). *Limitation:* a courtesy feed that
   flows straight into the serve (no dead-ball between) stays in the rally as its
   first shot — separating it needs serve detection (deferred). `serve_is_inferred`
   marks rallies whose start was a gap, not a flagged serve.

## Stage version

`0.2.0`. (0.1.0 initial → 0.2.0 real-ball: ball-out-of-play rally boundaries,
`hitter_side`-based sides, real-ball `unknown` end_reason, feed drop.) Increment
minor for behavior changes preserving the `rallies.json` schema. (v0.2.0 adds
the additive field `serve_is_inferred`; schema kept.)

## Out of scope (deferred)

- **Winner-side attribution.** Needs `track_roles.json` plus per-shot
  side reasoning; Stage 8 territory.
- **Diagonal service-box validation** (was the serve's bounce in the
  *correct* receiver's box — right vs left — given the server's
  alternation count?). v1 handles the kitchen + out-of-court parts of
  serve-fault; the diagonal check needs server-alternation tracking
  across points (which side has served how many times for the current
  server). Stage 8 or a dedicated serve-quality stage.
- **Per-shot error attribution beyond end_reason.** Which player hit
  the rally-ending shot, was it forced or unforced, was the lost
  point a winner or unforced error. Stage 8 metrics.
- **Casual trailing shots inside a rally.** After a rally truly ends,
  a player sometimes hits the ball casually back to the server before
  the next serve. Stage 5 sees this as a shot, Stage 7 v1 includes it
  in the rally's `shot_ids` (which inflates `n_shots`). end_reason is
  still correct since it's bounce-driven, but `n_shots` and mean
  rally length are slightly noisy. Synth doesn't generate these; real
  data may. Follow-up: detect rally-ended-here using bounces, then
  truncate trailing shots.

## Known follow-ups

- **End_reason accuracy depends on Stage 5/5.5 recall.** Stage 5 at
  ~0.95 hit recall and Stage 5.5 at ~0.87 bounce recall together set
  the ceiling for end_reason accuracy — if either misses the
  rally-ending event, Stage 7 falls into `unknown`. Re-tune when real
  ball detection (v4) lands.
- **Pre-rally orphan shots.** Currently dropped with a warning; in
  longer clips this might lose actual rally data if the first serve
  isn't detected. Consider a fallback: if pre-rally shots exist AND
  the first shot is plausible-as-serve (e.g., follows a long ball-
  gap), promote it to a synthetic serve. Real-data signal needed
  before deciding.
- **Last-rally end_reason without a next-serve boundary.** Works in
  v1 (uses bounces alone), but may overreport `unknown` when the
  clip cuts before any rally-ending bounce. Stable behavior, just
  noted.
- **Distinguishing net-hit from short-shot.** v1's `net-or-short`
  bucket conflates "ball hit the net and fell back" with "ball was
  short and bounced before reaching the receiver". Splitting them
  would need either the bounce's distance-from-net (a derived feature
  from `court_xy_ft`) or a net-strike detector. Easy follow-up once
  real-data tuning starts.

## Architecture note

Stage 7 was already in the pipeline diagram (one of the original 11
stages); this contract takes it from "not started" to
"implemented + smoke-tested". Pipeline count stays at 13. The
ARCHITECTURE.md `Stages 7-11` line becomes "Stage 7 implemented;
Stages 8-11 not started" on approval.

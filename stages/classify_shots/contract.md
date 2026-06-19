# Stage 6 — Classify Shots

**Status:** Contract APPROVED (2026-05-22), IMPLEMENTED. Two axes (stroke side,
shot type) + a bounce-based volley flag. v1 gives real forehand/backhand for the
USER only (non-user sides are `unknown` until a player-role-classification stage
exists); full shot-type taxonomy incl. reset; volley via inter-shot bounce
detection. Smoke test: rule logic by deterministic unit checks; end-to-end
schema + serve + is_volley (0.95) + unknown-rate via an extended `synth_ball`.
Lob-by-arc is weak in this low-headroom footage (Known follow-ups).

## Purpose

Take each shot Stage 5 found and label *what kind of shot it was* — both the
**stroke side** (forehand / backhand) and the **shot type** (serve / drive /
dink / drop / lob / overhead / …), plus whether it was a **volley** (hit out of
the air). These labels feed rally segmentation (Stage 7), per-shot metrics and
shot-mix breakdowns (Stage 8), and ultimately the USAPA-style rating (Stage 9),
which cares a lot about *which* shots a player hits well (third-shot drops,
dinks, resets).

Stage 6 is rule-based on geometric/kinematic features, not ML — same philosophy
as the rest of the pipeline (sport-specific heuristics, documented thresholds,
loud failures, honest "unknown" when the signal isn't there).

## Place in the architecture

```
shots.json (S5) + players.parquet (S2) + poses.parquet (S3)
        + ball.parquet (S4/synth) + court.json + court_zones.json (S1)
        │
        ▼
   [6] classify_shots ──► classified.json
```

Per-video, file-path I/O, standalone CLI:
`python -m stages.classify_shots.classify_shots <video_folder>`.

> **DECISION (folder name).** Like Stage 5, the code must live in an importable
> folder `stages/classify_shots/` (not `06_classify_shots`, which isn't a valid
> module). This contract sits at the numbered stub path for review; on approval
> it moves to `stages/classify_shots/contract.md`.

## Inputs

From the per-video folder:

| File | From | Stage 6 uses |
|---|---|---|
| `shots.json` | Stage 5 | the shot list (frame, track_id, is_user, is_serve, impact_*, pre/post velocity, etc.) and `ball_source` |
| `players.parquet` | Stage 2 | hitter **foot/court position** at the shot frame → which court zone they hit from (reliable; the ball's in-air court projection is not) |
| `poses.parquet` | Stage 3 | shoulders/hips/wrists at the shot frame → stroke side + contact height |
| `ball.parquet` | Stage 4 / synth | post-impact **trajectory** (arc shape) for a few frames after contact |
| `court.json` | Stage 1 | homography, `derived.pixels_per_foot_at_*` for speed scaling |
| `court_zones.json` | Stage 1 | kitchen/transition/baseline depth bands |
| `roster.json` | setup (NEW, asked up front) | per-player **handedness** by logical role (user / partner / opp_left / opp_right) |

CLI flags (defaults in Configuration): `--force`, `--log-level`, plus the
threshold knobs (`--dink-max-speed-ftps`, `--drive-min-speed-ftps`,
`--lob-min-arc-frac`, `--overhead-contact-frac`, …).

## Outputs

### `classified.json`

Carries each shot through from `shots.json` (so Stage 7 reads one file) and adds
classification. `ball_source` and the synthetic warning are **propagated**.

```json
{
  "schema_version": 1,
  "source_shots": "data/test_clip/shots.json",
  "ball_source": "synthetic",
  "fps": 30.0,
  "params": { "...thresholds...": 0 },
  "shots": [
    {
      "shot_id": 7,
      "frame": 2105,
      "t_sec": 70.17,
      "track_id": 1393,
      "is_user": true,
      "is_serve": false,
      "stroke_side": "backhand",
      "stroke_side_confidence": 0.71,
      "shot_type": "dink",
      "shot_type_confidence": 0.66,
      "is_volley": false,
      "is_volley_confidence": 0.4,
      "features": {
        "contact_zone": "kitchen",
        "post_speed_ftps": 14.2,
        "arc_height_frac": 0.08,
        "contact_height": "low",
        "landing_court_y": 18.3,
        "type_from_landing": true,
        "handedness_used": "right",
        "handedness_known": true,
        "facing": "away"
      }
    }
  ],
  "stats": {
    "n_shots": 286,
    "by_shot_type": {"serve": 42, "dink": 120, "drive": 50, "drop": 28,
                     "lob": 14, "overhead": 6, "unknown": 26},
    "by_stroke_side": {"forehand": 150, "backhand": 96, "unknown": 40},
    "n_volley": 31,
    "n_unknown_type": 26,
    "n_unknown_side": 40
  },
  "warnings": [
    "ball_source is 'synthetic': classifications are derived from PLACEHOLDER ball data."
  ],
  "stage_version": "0.1.0",
  "completed_at_utc": "..."
}
```

Every input shot produces exactly one output shot (same `shot_id`), so the file
is a 1:1 superset of `shots.json`.

## Classification method

Two **independent** axes plus a volley flag. Each gets a confidence in [0,1];
when the signal is too weak/ambiguous, emit `"unknown"` rather than guess.

### Axis A — stroke side (forehand / backhand)

1. **Handedness — asked up front for all players.** A new `roster.json`
   captures handedness per **logical role** (user, partner, opp_left,
   opp_right), collected during setup (operator answers; safer and more
   accurate than auto-detecting paddle hand on the serve). Knowing all four
   enables high-value analysis later (e.g. "hit to the opponent's backhand").
   **But applying it needs role identity:** Stage 6 must know which track_id is
   which logical player to attach a hand. Today only the **user** is mapped (via
   `is_user`). So:
   - **User shots:** real handedness → real forehand/backhand.
   - **Non-user shots:** handedness is `unknown` (and so is stroke side) **until
     a player-role-classification stage** ("Stage 2.5" in `KNOWN_ISSUES.md`)
     maps roles to tracks. We do NOT assume right-handed — the operator gave us
     truth for the user, and guessing others would corrupt opponent-targeting
     stats later.
   > RESOLVED (review): collect `roster.json` now (all four roles), no
   > assumed-right fallback (non-user side = `unknown` until roles exist). This
   > makes the role-classification stage higher priority: it unlocks non-user
   > handedness, per-player stats, the court-switch ID fix, and target-placement
   > analysis.

   `roster.json` schema (hand-authored during setup; `unknown` allowed):
   ```json
   {
     "schema_version": 1,
     "handedness": {
       "user":     "right",
       "partner":  "unknown",
       "opp_left": "unknown",
       "opp_right":"unknown"
     }
   }
   ```
   The `opp_left` / `opp_right` labels are provisional until the role stage
   pins which person is which. v1 reads only `handedness.user` (must match
   `court.json.dominant_hand`); the rest is stored for when roles exist.
2. **Facing.** From pose, decide if the player faces toward or away from the
   camera (camera is usually behind the user → user faces away). Use the
   left/right shoulder x-order and landmark `z`/visibility. This determines the
   sign convention for "which side of the body."
3. **Side of contact.** Compute the impact point's horizontal offset from the
   player's body midline (shoulder/hip center) in the player's *egocentric*
   frame (flip by facing). Contact on the dominant side → **forehand**; on the
   non-dominant side → **backhand**.
4. **Confidence** falls with low pose visibility, ambiguous facing, or
   near-midline contact; below a floor → `stroke_side="unknown"`.

> Forehand/backhand is the genuinely hard axis (facing + handedness). v1 is
> best-effort with honest `unknown`; see Known follow-ups.

### Axis B — shot type

Features per shot:
- `contact_zone` ∈ {kitchen, transition, baseline} — from the **hitter's foot
  position** (`players.parquet` court_x/y_ft) and `court_zones.json` depth
  bands, by distance from the net (net at length/2).
- `post_speed_ftps` — `post_velocity_px_per_frame` scaled to ft/s using the
  local pixels-per-foot at the contact location (interpolated from
  `court.json.derived.pixels_per_foot_at_near/far_baseline`) × fps.
  > **DECISION (speed).** Pixel speed is perspective-dependent, so we normalize
  > by local px/ft for an approximate true speed. It's approximate (in-air
  > ball, foreshortening); thresholds are coarse buckets, not precise speeds.
- `pre_speed_ftps` — the **incoming** ball speed (`speed_pre_px_per_frame` from
  shots.json), scaled the same way. Distinguishes a soft reply to a *hard*
  incoming ball (reset) from an unforced soft shot (dink/drop). Null for serves.
- `arc_height_frac` — from `ball.parquet` over ~0.5 s after contact: peak upward
  excursion above the contact→end chord, as a fraction of the chord length.
  High = lob.
- `contact_height` ∈ {low, mid, high} — `impact_pixel_y` vs the hitter's
  shoulder/hip y from pose (above shoulders = high; below hips = low).
- `landing_court_y` (v0.4.0) — the court_y of the **first bounce after the shot**
  (from `bounces.json`, `between_shots[0] == shot_id`), or null if no landing was
  detected. **Bounces are on the ground**, so this projects through the homography
  reliably — unlike the airborne ball contact, whose ground projection explodes
  (see DECISION below). `features.type_from_landing` records which path was taken.

> **DECISION (v0.4.0 — landing-aware type).** Pixel-speed is depth-corrupted (a
> down-court drive covers few pixels → reads slow), and projecting the airborne
> ball to court-feet for a court-plane speed was **validated DEAD** (it explodes
> toward the horizon — court_y in the hundreds/thousands). The *sound* signal is
> **where the ball lands**: a drive lands deep, a drop/dink lands in the kitchen
> (+~2 ft). So when a real bounce landing exists, it drives the type. **Coverage
> is only ~21% of shots** (bounce recall, Stage 5.5) and inherits its
> association errors; the other ~79% fall back to the speed/arc rules below at
> lower confidence. The deep-landing bucket still blurs drive/serve/lob/feed —
> only `is_serve` (Stage 5) and arc separate them. See SYSTEM_DESIGN.md §3 Stage 6.

Rule order (first match wins; thresholds tunable, tuned on the smoke test):
1. `is_serve` (from Stage 5) → **serve**.
2. `arc_height_frac ≥ lob_min_arc_frac` AND `post_speed_ftps < drive_min_speed_ftps`
   → **lob**. (A lob is lofted **and** slow; resolved before the landing split
   because a lob also lands deep. The speed gate keeps a fast flat drive with a
   noisy-high measured bow from being mislabeled a lob.)
3. **Landing-aware path (v0.4.0) — when `landing_court_y` is not null:**
   - soft landing (within `kitchen + ~2 ft` of the net, `|landing − net| ≤ 9 ft`):
     `contact_zone == kitchen` → **dink**, else → **drop**.
   - deep landing + `contact_height == high` → **overhead**; else → **drive**.
   - (confidence 0.70–0.78 — the reliable path.)
4. **Fallback (no landing): speed + arc, lower confidence (0.40–0.60):**
   - `contact_height == high` AND `post_speed_ftps ≥ drive_min_speed_ftps` → **overhead**.
   - `post_speed_ftps ≥ drive_min_speed_ftps` (flat, fast) → **drive**.
   - **reset** — `pre_speed_ftps ≥ reset_min_incoming_ftps` (hard incoming) AND
     `post_speed_ftps ≤ dink_max_speed_ftps` (soft reply) AND `contact_zone != baseline`.
   - `post_speed_ftps ≤ dink_max_speed_ftps` AND `contact_zone == kitchen` → **dink**.
   - `post_speed_ftps ≤ dink_max_speed_ftps` AND `contact_zone in {transition, baseline}` → **drop**.
   - **tweener tiebreak** (`dink_max < post_speed_ftps < drive_min`, 16–25 ft/s):
     `arc_height_frac ≥ drive_drop_arc_split` → **drop** (lofted), else → **drive** (flat).
5. otherwise → **unknown** (recorded honestly, counts toward `n_unknown_type`).

> **Reset** (per review): a fast ball returned softly, valid only when the
> players are off the baseline and the reply settles into the kitchen. v1 keys
> on fast `pre_speed` + soft `post_speed` + hitter not at baseline. The full
> definition's "lands in the kitchen" and "opponent also off the baseline"
> conditions need the ball's **landing point** (bounce detection) and
> **opponent role/position** (role classification) respectively — both deferred;
> v1 approximates with the hitter-position + speed signal and lowers confidence
> when those can't be confirmed.

### Volley flag (orthogonal, bounce-based)

`is_volley` = the ball was struck **out of the air** — i.e. it did **not bounce**
between the previous shot and this one. Volleys are common in pickleball (net
exchanges often never bounce), and volley-ness is orthogonal to stroke type (you
can volley a dink or a drive), so it's a **separate boolean**, not a `shot_type`.

**Mechanism — `bounces.json` lookup (since Stage 6 v0.2.0, 2026-05-27).**
Stage 6 reads `bounces.json` (Stage 5.5's output) and computes
`is_volley = (count of bounces with between_shots == [prev_shot_id, this_shot_id]) == 0`.
At-feet bounces count as bounces — a dink that lands at the receiver's feet
means the receiver's return is NOT a volley. The first shot of a rally
(serve) is always `is_volley = false` with high confidence.

> **DECISION (volley, post-rewire).** Originally Stage 6 v0.1.0 scanned the
> inter-shot ball trajectory itself to find a non-player inflection. With
> Stage 5.5 (`detect_bounces`) landed, that scan is duplicate logic that can
> drift from the canonical bounce signal. Stage 6 now consumes `bounces.json`
> directly — single source of truth, includes at-feet bounces the old scan
> missed. Output schema unchanged; `stage_version` bumped 0.1.0 → 0.2.0.

> **DECISION (volley on the REAL ball, v0.2.0 → v0.3.0, 2026-06-15).** On the
> real (v4) ball the `bounces.json` lookup **under-detects volleys' opposite**:
> Stage 5.5's bounce LIST is tuned for *precision* (exact zone stats; it filters
> apex/in-air wobble), so it misses real bounces → those shots wrongly read as
> volleys. The volley *flag* wants **recall** ("did the ball bounce at all
> between these two shots?"), a different objective than the bounce *list*. So
> v0.3.0 **decouples** them: `is_volley`'s primary signal is a recall-focused
> **local trajectory scan** between the two shots — a ground bounce is an
> *interior local peak in pixel_y* (ball momentarily lowest on screen) with the
> ball descending into it and rebounding up out of it. (Crucially NOT the global
> pixel_y max: the segment starts at a high pixel_y because the previous contact
> is low on screen, and the outgoing arc's apex is a pixel_y *minimum* — both are
> ignored.) `bounces.json` is kept only as a **fallback** when the inter-shot ball
> is too occluded to judge locally. Validated on pb_2min: operator-confirmed
> volley/bounced on a 7-shot spot-check; pipeline volleys 27 → 9. Output schema
> unchanged; `stage_version` 0.2.0 → 0.3.0.

## Defenses against placeholder / bad data

- Propagate `ball_source` from `shots.json`; if `"synthetic"`, add the loud
  warning and log it. Downstream must keep treating labels as placeholder-
  derived.
- If `shots.json` is missing/empty → complete with `shots: []` and a warning
  (not a crash).
- Missing pose for a shot → stroke side falls to `unknown` (don't fabricate).
- Missing/short post-impact ball trajectory (gap) → arc feature unavailable;
  shot_type falls back on speed+zone only, with reduced confidence.

## Edge cases

- **Shot near a court boundary / zone edge** — zone by hitter foot position; ties
  resolved toward the net-ward zone (kitchen > transition > baseline), matching
  `court_zones.json`'s effective-kitchen priority rule.
- **Degenerate/low-visibility pose** — `stroke_side="unknown"`, low confidence.
- **Non-user player, handedness unknown** — see DECISION (handedness).
- **Serve** — `shot_type="serve"`; stroke side still computed if pose allows.
- **Required input missing/malformed** → fail loudly naming the file. Output
  exists without `--force` → `FileExistsError`.

## Configuration (defaults; tuned on smoke test)

```python
LOB_MIN_ARC_FRAC        = 0.35   # arc peak / chord -> lob
DRIVE_MIN_SPEED_FTPS    = 25.0   # fast, flat -> drive/overhead
DINK_MAX_SPEED_FTPS     = 16.0   # soft -> dink/drop/reset
RESET_MIN_INCOMING_FTPS = 25.0   # hard incoming + soft reply + not baseline -> reset
OVERHEAD_CONTACT_FRAC   = 0.0    # contact above shoulder line -> high
POST_TRAJ_FRAMES        = 15     # ~0.5s window for arc + inter-shot bounce scan
SIDE_CONF_FLOOR         = 0.5    # below -> stroke_side = unknown
# No non-user handedness default: unknown until role classification exists.
```

## Smoke test

`stages/classify_shots/test_classify_shots.py`, against `data/test_clip/`.

The hard part: there is **no ground truth for shot type or stroke side** on
real footage, and the ball is synthetic. `tools/synth_ball.py` is extended to
tag each hit with an intended `shot_type` demo and to **insert ground bounces**
(for `is_volley` truth) in `ball_synth_truth.json`.

**Discovered during build:** end-to-end **shot_type accuracy is not a reliable
gate** for arc-based types in this footage. The play sits high in the frame
(~250 px of headroom), so any tall lob arc — synthetic OR real — clips at the
top edge and its measured `arc_height_frac` collapses. So lob detection by arc
fraction is inherently weak here (a real-data limitation, see Known follow-ups).
Testing therefore has **two layers**:

1. **Unit checks of the rule logic** (deterministic, footage-independent):
   feed clear-cut feature tuples to `classify_type` and assert the expected
   label for every type (serve/lob/overhead/drive/reset/dink/drop/unknown).
   This validates rule order + thresholds without depending on whether the
   synthetic ball can physically render a clean lob.
2. **End-to-end** (`synth_ball` → Stage 5 → Stage 6) graded on what IS reliable:
   schema/consistency, serve handling, `is_volley` accuracy, unknown rate.

Conditions:
1. **Rule logic**: all clear-cut `classify_type` cases produce the expected type.
2. `classified.json` parses; 1:1 with `shots.json` (same shot_ids); every shot
   has `stroke_side`, `shot_type`, `is_volley` + confidences in [0,1]; all
   categories from the allowed sets.
3. `ball_source` propagated; synthetic warning present.
4. Every `is_serve=true` shot has `shot_type="serve"`.
5. **is_volley accuracy ≥ 0.70** vs synthetic bounce-derived truth.
6. `unknown` shot_type rate < 40% of shots (rules aren't bailing out wholesale).
7. An injected-gap variant completes without crashing.

Stroke side has no synthetic ground truth (pose is real), so it's validated for
schema/consistency only.

## Stage version

`0.3.0`. (0.1.0 → 0.2.0: volley via `bounces.json` lookup. 0.2.0 → 0.3.0,
real-ball: volley via recall-focused local trajectory scan (bounce list = fallback);
lob requires below-drive speed; tweener (16–25 ft/s) arc-shape tiebreak drains the
"unknown" dead-zone; fps + resolution scaling. Output schema unchanged.)

## Out of scope (deferred)

- **Bounce detection** (reliable volley, and serve/shot landing point) — a
  future bounce stage / Stage 7.
- **Spin** — not recoverable from a single corner camera.
- **Per-shot in/out** — needs bounce location (Stage 7).
- **Learned classification** — rule-based for v1.

## Known follow-ups

- **Forehand/backhand robustness** depends on facing detection; revisit if the
  smoke test / inspection shows systematic side errors, especially for
  camera-facing opponents (mirrored) and back-facing user.
- **Player-role classification is the key unlock.** Non-user handedness
  (collected in `roster.json`) and per-player analysis like "hit to the
  opponent's backhand" need a stage that maps logical roles (user / partner /
  opp_left / opp_right) to track_ids over the match — the "Stage 2.5" noted in
  `KNOWN_ISSUES.md`. Until it exists, non-user stroke side is `unknown`.
- **Reset & volley depend on bounce/landing + opponent position.** v1 uses the
  ball-trajectory bounce check (volley) and hitter-position+speed (reset); the
  full definitions ("lands in the kitchen", "opponent off the baseline") want a
  dedicated bounce/landing stage and role positions.
- **Real-data thresholds** (speeds, arc) are tuned on synthetic arcs; re-tune
  against real ball trajectories when a real detector (v4) exists.
- **Lob detection by arc fraction is weak in low-headroom footage.** When the
  play sits high in the frame (this test_clip: ~250 px above the action), a tall
  lob arc clips at the top edge and `arc_height_frac` collapses — so real lobs
  may be missed/under-arced. Revisit with a height-aware or court-projected arc
  measure, or rely on speed + apex-time, when better-framed (higher-mounted)
  footage exists. Until then, lob is best-effort; the smoke test validates the
  lob *rule* by unit check rather than end-to-end accuracy.
- **Depth/height corrupts pixel-speed → drive-vs-drop confusion (real ball).**
  Shot type leans on `post_speed_ftps`, derived from pixel-speed × a planar
  `ppf` scalar at the contact point. A drive hit straight **down-court (in
  depth)** covers almost no pixels/frame, so it reads as slow and is
  pixel-for-pixel indistinguishable from a slow drop (observed on pb_2min f3541:
  a true drive measured 4.2 px/f → labeled "drop"). The tweener arc-tiebreak
  (rule 8) helps the 16–25 ft/s band, but a depth-drive that reads *below* dink
  speed still can't be separated in 2D. Proper fix: **homography-projected
  court-plane ball speed** (project ball pixel → court feet per frame; handles
  depth, residual height bias) — which also feeds Stage 8 metrics — or full 3D
  ball tracking. Deferred to when ball speed matters for metrics.
- **Serve labeling depends on Stage 5 `is_serve`.** If Stage 5 misses a serve
  (serve-detection gap), Stage 6 classifies it by features (often "drive"/"lob"),
  never "serve". Fix belongs in Stage 5.
- **Courtesy/between-point feeds read as volleys.** A pre-serve feed (opponent
  hands the ball over) has no bounce → `is_volley=true`, which is literally
  correct but pollutes rally stats. Exclude such non-rally shots in **Stage 7
  (rally segmentation)**, not here.

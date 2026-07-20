# Stage 5.7 — Ball Trajectory (physics)

**Status:** Contract DRAFT (2026-07-20), pending approval. NEW stage. Slots
**between Stage 5.5 (detect bounces) and Stage 6 (classify shots)**. Built
**incrementally** (operator-approved 2026-07-20):

- **Phase 1 (this contract): ground-anchored horizontal ball speed.** Replaces the
  airborne-ball speed that today explodes to physically-impossible values (261, 117
  ft/s on match rally 10) and drives the soft-shot→drive mis-types.
- **Phase 2 (later): height / apex reconstruction + physics-based bounce
  prediction** → fixes volley detection (2/8 on match play). Vertical scale from
  tracked players calibrated by **operator-entered player heights** (nominal ~5'9"
  fallback) — operator chose known-heights 2026-07-20.
- **Phase 3 (later): wire into Stage 6, recalibrate dink/drive/drop thresholds
  against ground truth, re-validate** on match rally 10 + the 20 s drill.

## Why this stage exists

Match-clip validation (`pb_5_minute_outdoor-2` rally 10, 2026-07-20) showed every
significant shot-type/volley error traces to ONE root: **the ball is airborne and
its ground-homography projection is meaningless** for speed/position. Four speed
methods were tested (see `docs/ACCURACY_LEDGER.md` → "Stage 4 geometry / ball
SPEED"); all projection-based ones explode or inflate. The robust insight:

> Never measure speed from the airborne ball's raw projection. Anchor on the
> **ground points we can trust** — the hitter's foot (on the ground, where the ball
> was struck) and the bounce (z = 0) — and derive the ball's motion between them.

A pickleball between two contacts is a projectile: horizontal motion ≈ constant,
vertical motion a parabola under gravity. Phase 1 recovers the **horizontal**
component, which alone separates a slow dink/drop from a fast drive.

## Place in the architecture

```
shots.json (S5) + bounces.json (S5.5) + players.parquet (S2)
    + court.json (S1) + poses.parquet (S3, front-foot)
        │
        ▼
   [5.7] ball_trajectory ──► trajectory.json   (per-shot physics features)
        │
        ▼  (consumed by Stage 6 classify_shots: horizontal_speed replaces the
            ppf-scalar post_speed as the primary dink/drive discriminator)
```

Per-video, file-path I/O, standalone CLI:
`python -m stages.ball_trajectory.ball_trajectory <video_folder>`.

> **DECISION (folder name).** `stages/ball_trajectory/` — importable (no leading
> digit), same convention as `detect_bounces/`, `classify_shots/`.

> **DECISION (placement at 5.7).** Downstream of bounces (it consumes them as
> ground anchors) and upstream of Stage 6 (which consumes its speed). A dedicated
> stage — not a helper inside Stage 6 — because Phase 2/3 grow it into the full
> "ball physics" layer (height, apex, bounce prediction) and it deserves its own
> tests + contract.

> **DECISION (ground-anchor model, not camera calibration).** Full 3-D
> reconstruction from a ground homography needs the vertical vanishing point / camera
> intrinsics (fragile, per-clip). Instead we anchor on ground points (hitter foot +
> bounce) and use gravity — no per-clip calibration, fully automatic. Operator
> approved 2026-07-20.

## Phase 1 algorithm

Inputs per shot `i` (shots are processed in rally order; `f_i` = contact frame,
`H_i` = hitter's front-foot court position `(x, y)`):

1. **Hitter ground anchor `H_i`** = the hitter's **front foot** (ankle nearest the
   net) projected to court `(x, y)` at `f_i`. Reuses the front-foot logic already in
   Stage 6 (`front_foot_court_y`, generalised to return `(x, y)`). Fallback: the
   player's `court_x_ft`/`court_y_ft` (bbox foot) when the pose ankle is missing or
   `court_pos_reliable` is false.

2. **Far ground anchor** (where the ball's flight ends):
   - **Bounced shot** — a bounce `B` with `between_shots[0] == i` exists →
     `far = B.court_xy_ft`, `f_far = B.frame`, `anchor = "bounce"`.
   - **Volley** (no bounce before the next contact) → `far = H_{i+1}` (next hitter's
     front foot), `f_far = f_{i+1}`, `anchor = "next_contact"`.
   - **Rally-final shot, no bounce** → no far anchor → `horizontal_speed = null`,
     `confidence = 0`, `anchor = "none"` (Stage 6 falls back to today's logic).

3. `range_ft = ‖far − H_i‖` (Euclidean, court feet).
   `airtime_s = (f_far − f_i) / fps`.
   `horizontal_speed_ftps = range_ft / airtime_s`  (only if `airtime_s ≥
   MIN_AIRTIME_S`).

4. **Confidence** ∈ [0, 1]:
   - base `0.85` (anchor = bounce) or `0.6` (anchor = next_contact);
   - × `0.5` if either anchor's `court_y` is outside `[−3, 47]` ft (homography
     stretch — the far-side y = 51 we saw on this clip);
   - × `0.5` if `airtime_s < 0.15` s (too few frames to trust);
   - × `0.5` if `range_ft > MAX_RANGE_FT` (44) — impossible single-shot range.

## Outputs — `trajectory.json`

```json
{
  "schema_version": 1,
  "phase": 1,
  "params": { ... echoed ... },
  "shots": [
    {
      "shot_id": 3,
      "horizontal_speed_ftps": 34.2,
      "range_ft": 15.8,
      "airtime_s": 0.46,
      "anchor_type": "bounce",          // bounce | next_contact | none
      "hitter_court_xy_ft": [7.1, 3.2],
      "anchor_court_xy_ft": [6.4, 18.9],
      "confidence": 0.85
    }
  ]
}
```

One entry per shot in `shots.json`, same order/ids. `horizontal_speed_ftps` is
`null` when `anchor_type == "none"` or `airtime_s < MIN_AIRTIME_S`.

## Params (× nothing — all in court feet / seconds, resolution-independent)

| name | default | meaning |
|---|---|---|
| `min_airtime_s` | 0.10 | below this the frame gap is too small to trust |
| `max_range_ft` | 44.0 | a single shot can't travel more than court length |
| `court_y_valid_min` / `_max` | −3 / 47 | homography-reliable band (court is 0–44) |
| `bounce_conf` / `next_contact_conf` | 0.85 / 0.6 | base confidence by anchor type |

## Edge cases
- **No bounce and last shot of rally** → `null` speed, confidence 0 (documented).
- **Serve** (`is_serve`) — treated like any shot; anchor is the serve bounce
  (return-of-serve must bounce) → almost always `anchor = "bounce"`.
- **Missing hitter pose/player row** → bbox-foot fallback, confidence × 0.5.
- **Double bounce between two shots** → use the FIRST bounce (the ball's landing);
  the second belongs to the receiver's miss.
- **`between_shots[0]` is null** (bounce before the first tracked shot) → ignored.

## Limitations (on the record)
- **Average, not instantaneous.** `horizontal_speed` is the mean over the flight;
  it's lower than the launch speed. Stage 6 thresholds get **recalibrated** in
  Phase 3 against ground truth (today's `DINK_MAX=16`, `DRIVE_MIN=25` assume
  instantaneous px-derived speed and will change).
- **Linear horizontal motion** assumed (no air resistance / spin). Fine for
  dinks/drives; weakest for long lobs.
- **Contact ≈ hitter foot** horizontally (~1–2 ft error vs true contact point;
  negligible on 7–40 ft ranges, vs today's 261 ft/s error).
- **Far-side homography** stretch (court_y = 51 observed) inflates far-anchor range;
  handled by the confidence penalty, not yet corrected (Phase 2 height helps).
- Height, apex, launch angle, and **bounce/volley determination are Phase 2** — this
  stage does not touch `is_volley` yet.

## Phase 1 result (2026-07-20) + the bounce-quality dependency it exposed

Implemented + 8/8 tests. Added two PHYSICAL FILTERS beyond the contract draft (an
anchor is *wrong*, not slow/fast, if it violates these), each rejecting the anchor →
fall back (bounce → next_contact → none):
- **range ≤ court length** (rejected impossible 53/61/63 ft "ranges");
- **net-crossing** — a shot's landing / next contact must be on the OPPOSITE side of
  the net from the hitter (rejected the serve "landing" 2.9 ft from the server).

**On clean bounces (drill): works.** Dinks 15–22 ft/s, drives 27–31, drop 31, serve
45 — physical and separable; the old garbage (261, 117) is gone (→ null, low conf).

**On match play (rally 10): exposed the real bottleneck — BOUNCE QUALITY.** Only
**47 % of shots (73/155) get a reliable anchor**, and spurious FAR-side bounces still
corrupt #2/#5/#7 (a drop + two volleys read 34–48 ft/s from a phantom landing). Root
cause is upstream: **the match clip's bounces are 79 % "at-feet" (116/146)** — the
Stage 5.5 at-feet tiebreaker that correctly recovered dink bounces in the 20 s drill
massively over-fires on match play (players are always near the ball). Ground-anchor
speed can only be as clean as the bounces it anchors on.

**→ New foundation before Phase 3 pays off: clean up Stage 5.5 at-feet
over-triggering on match play.** Phase 2 (physics-based bounce PREDICTION from the
arc) also directly attacks this — a predicted bounce time/place validates or rejects
each detected bounce, which is exactly what would kill the phantom far-side bounces.

## Validation plan
Smoke test on `data/test_clip/` (schema + no crash + sane ranges). Then re-derive on
`pb_5_minute_outdoor-2` rally 10 and the 20 s drill and compare `horizontal_speed`
against the operator ground truth: the garbage speeds (261, 117 ft/s) must become
physical, and the soft-shot→drive mis-types must shrink once Phase 3 wires it in.

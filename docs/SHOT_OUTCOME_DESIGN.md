# Shot-Outcome Detection + Rally Trim — Design / Contract

**Status:** proposed (awaiting operator approval before code)
**Date:** 2026-07-27
**Supersedes the shelved bounce-based net-hit attempt** (see docs/ACCURACY_LEDGER.md,
"NET-HIT DEFERRED"). Rendering proved net-or-short rests on dead-time bounces; and the
operator found between-point balls (returns to the server) leaking into the shot stream.
Both are the same root: **dead-time activity absorbed into rallies.** This design attacks
it from the trajectory instead of the bounce.

---

## 1. Goal

For **every detected shot**, classify what happened to the ball from its **image-space
trajectory** right after contact (not from a landing bounce, which is missing/dead-time
15% of the time at rally ends). Then use those outcomes to **trim between-point balls**
out of rallies, so all downstream shot evaluation runs on real rally play only.

## 2. Outcome taxonomy (operator definition, 2026-07-27)

After a shot, exactly one of:

| outcome | meaning | ends point? | fault |
|---|---|---|---|
| `continues` | opponent returned it (a next opposite-side shot follows soon) | no | — |
| `net` | ball did **not** clear the net (died on the hitter's own side / at the net) | yes | **hitter** (unforced) |
| `out` | ball crossed the net but landed **outside** the court boundary | yes | **hitter** (unforced) |
| `winner` | ball crossed and landed **in** the court but the opponent **missed** it | yes | receiver (a hitter *winner*) |
| `unknown` | trajectory too short / broken to tell | — | — |

Operator's phrasing: "a ball is out [the point is over] if it hits the net, lands outside
the court boundary, OR can bounce in the court but is missed by the player." The first two
are hitter errors; the third (`winner`) is a good shot. We keep them **distinct** because
coaching/rating must not treat a winner as an error.

## 3. Method — read the outcome from a CLEAN trajectory

**3a. Clean trajectory extraction.** From the contact point (`impact_pixel_xy`), walk the
ball forward through `ball.parquet`, accepting only visible detections within a continuity
radius (`STEP≈280px`) of the last accepted point; stop after `MISS_STOP≈10` consecutive
misses or `MAXF≈90` frames (1.5s). This **rejects the parked-background jumps** the tracker
makes when the real ball is lost. Validated: produces a clean arc on ~59% of all shots
(54% of rally-enders) — vs 15% bounce-landing recall. Rendered examples in `_traj_check/`
(r1 shows a clean drop crossing to the far kitchen — correcting a wrong bounce-based call).

**3b. Net crossing (did it clear?).** Project the net line to the image
(`court_to_image` at court_y=22). A shot **cleared** if the clean trajectory's descent
carries the ball to the **far side** of the net line (opposite the hitter) — judged on the
DESCENDING leg, where the ball is lower and its projection is more trustworthy, NOT the
airborne apex. If the trajectory rises toward the net and comes back down on the hitter's
own side (or dies with no far-side descent) → did **not** clear → `net`.

**3c. Landing estimate (in vs out).** For a cleared ball, project the **last clean
trajectory point** (end of the tracked descent) through `image_to_court`. This is a
mid-descent estimate (the ball is lost before it lands), so it is biased slightly toward
the net and carries a real error bar — but it reliably separates *inside the court* from
*beyond the baseline / outside the sideline*. If the endpoint is out of bounds → `out`;
in bounds → candidate `in`.

**3d. continues vs winner.** For a ball that cleared and landed in bounds: if a **next
shot by the opposing side** follows within a plausible window (≈ RALLY gap), it
`continues`; otherwise it's a `winner` (landed in, not returned).

**3e. Confidence + `unknown`.** Emit `unknown` (never guess) when: clean trajectory <
MIN_PTS (~15) or has no clear rise+descent; or the endpoint sits in the compressed
far-court sliver where in/out is within the noise floor. Each outcome carries a confidence
from trajectory length, descent clarity, and endpoint reliability.

## 4. Rally trim (remove between-point balls)

Within each provisional rally (serve → next serve), scan shots from the serve; the **first
shot whose outcome is a point-ender (`net` / `out` / `winner`)** marks the TRUE end of the
point. Shots after it (up to the next serve) are **between-point** → dropped from the rally
and from shot evaluation. Guards:
- Only trim on a **confident** point-ender (low-confidence/`unknown` → do NOT trim; leave
  the rally as-is rather than truncate real play).
- Never trim below the serve + its return (a point needs ≥ the serve).
- Log every trimmed shot (frame, why) so the render can show between-point balls dropping.

This directly fixes the operator's finding (r5 return-to-server, r10's 5.7s-late ball, and
rally 2's merged dead time) and tightens the shot count toward the operator's 98 (24 user).

## 5. Pipeline placement

Extend **Stage 5.7 `ball_trajectory`** (already per-shot; runs after bounces, before
classify + segment). It gains a per-shot `outcome` block. **Stage 7 `segment_rallies`**
consumes it for the trim. No new stage. Ball is real (GPU); gate on real ball as elsewhere.

## 6. Outputs (additive)

`trajectory.json` per shot gains:
```
"outcome": {
  "label": "continues|net|out|winner|unknown",
  "cleared_net": true/false/null,
  "landing_court_xy_ft": [x, y] | null,   // mid-descent estimate, has error bar
  "landing_in_bounds": true/false/null,
  "clean_traj_pts": <int>,
  "confidence": <0..1>
}
```
`rallies.json` gains, per rally: `trimmed_shot_ids` (between-point balls removed) and the
trim reason; `end_reason` becomes derivable from the point-ender's outcome
(net/out/winner) instead of the dead-time bounce.

## 7. Acceptance test (validate against operator truth, by RENDERING)

On `pb_5_minute_outdoor-2` (operator counts):
- **98 shots (24 user)** after trim — between-point balls removed, not real shots.
- **13 rallies / serves.**
- **17 volleys (5 user)** unchanged (volleys have no landing; outcome from next-contact).
- **8 net hits total, 3 by the user (a dink, a forehand drive, a backhand drive).**
- Every `net` / `out` / `winner` call and every trimmed shot **rendered** for operator
  eyeball before the number is trusted (`tools/draw_shot_trajectory.py` extended).
  Confidence ≠ correctness — the render is the gate.

## 8. Honest limits (stated before coding, per project practice)

- Works only when the ball is cleanly tracked (~59% of shots). The rest → `unknown`, not a
  guess. Recall of net/out/winner will be **partial**; we report what we can stand behind.
- The landing estimate is mid-descent (ball lost before it lands) → in-vs-out near the
  baseline is fuzzy; deep-out and clearly-in are reliable, boundary-line calls are not.
- A true net hit still often produces no bounce — but the **trajectory** (rise toward the
  net, no far-side descent) is the signal now, so this is no longer fatal.
- Between-point trim depends on a confident point-ender; if the point-ending shot is
  `unknown`, dead-time balls may still leak (logged, not silently absorbed).

## 9. Phasing (contract → code → smoke → commit, per phase)

1. **Outcome classifier** in `ball_trajectory` + unit tests + render validation vs the 8
   net hits / out / winner. **Gate: operator confirms the renders before proceeding.**
2. **Rally trim** in `segment_rallies` using confident point-enders; re-check shot count
   → 98/24, render the dropped between-point balls.
3. **Wire downstream** (metrics/rating/report): net/out as unforced errors (total + by
   you), winners as a positive, honest coverage badges. Re-validate counts end-to-end.

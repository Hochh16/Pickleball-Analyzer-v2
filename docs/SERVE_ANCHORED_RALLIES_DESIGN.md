# Serve-Anchored Rallies + Between-Point Trim — Design / Contract

**Status:** proposed (awaiting operator approval before code)
**Date:** 2026-07-27
**Goal (operator directive):** remove between-point balls (returns to the server, tidy-up
taps) from the shot set — "those are meaningless" — and fix the rally boundaries they
expose. Does NOT depend on ball-outcome detection (shelved as camera-blocked; see
docs/SHOT_OUTCOME_DESIGN.md, docs/ACCURACY_LEDGER.md).

---

## 1. The problem (diagnosed on pb_5_minute_outdoor-2)

Rally boundaries are wrong because **serve detection is unreliable**, so between-point
balls get absorbed and points get merged/split:
- **Merged points:** rally 2 contains **3 serves** (3 points), rally 5 and rally 12 each
  contain 2. The 6s intra-rally gap glued separate points together.
- **Missing serves:** **7 of 13 rallies have no detected serve.** 5 of those start with a
  clear deep baseline shot (21–28 ft from the net) that was never *flagged* as a serve.
- **Between-point balls:** e.g. r5 `sid56` (a return-to-server 3.6s after real play), r10
  `sid83` (a serve mis-attached to the previous rally's end).

**Root cause of the missed serves:** the serve detector triggers on the ball *reappearing
after a not-visible gap* (dead time). But during dead time the tracker keeps emitting a
**background** ball (visible-frac 0.7–1.0 before the missed serves), so the gap never
appears and the serve isn't flagged. Same background-contamination that breaks bounces.

Operator truth: **13 rallies = 13 serves, 98 shots (24 user), 17 volleys (5 user).**

## 2. Fix — three parts

### 2a. Robust serve detection (depth-anchored, not ball-gap)
A serve is **a shot struck from BEHIND the baseline that opens a point** (operator rule,
2026-07-27: "all serves must be hit behind the baseline"). Replace the fragile
ball-visibility-gap trigger with the **hitter's ground depth** — `hitter_court_xy_ft`
distance-from-net — which is robust (ground plane, not airborne/occluded ball):
- Candidate serve = a shot with the hitter **behind the baseline**
  (`dist_from_net > ~21–22 ft`; baseline is 22 ft from the net) **AND** preceded by a
  real-play gap (no rally shot for ≥ SERVE_GAP_S, previous point over).
- **Dedup the between-point feed.** The operator confirmed the "extra" serves are NOT real
  serves: at 2:24 the opponent *feeds the ball to the server* between points (rally 5),
  and rally 12's second "serve" is spurious (`sid87`, only 14 ft from the net → the
  behind-baseline rule alone rejects it). Rule: of two behind-baseline serve candidates
  within ~SERVE_DEDUP_S with no cross-net rally exchange between them, keep the one that
  **starts sustained play**; the other is a feed/artifact, not a serve.
- Validated on this clip: the rule **rejects** the false serves (`sid4` 10 ft, `sid65`
  22 ft-far, `sid87` 14 ft) and **recovers 4 missed serves** (`sid25/40/57/84`, 23–28 ft),
  landing near the operator's 13.
- Keep the existing sustained-ball-run gate as *secondary* confirmation where available,
  but do not *require* the ball gap.

### 2b. Serve-anchored rally segmentation
Each serve **starts** a new rally; rally *i* = [serve_i, serve_{i+1}). This:
- splits merged points (rally 2 → 3, rally 5 → 2, rally 12 → 2),
- folds no-serve fragments into their serve's rally,
- makes rally count fall out of serve count (target: 13).

Replaces the gap-primary segmentation with serve-primary; the gap rule stays only as a
*within-span* aid for the trim (2c) and for the rare span with no clean serve.

### 2c. Between-point (dead-time) trim
Within a serve-to-serve span, the real point is the **continuous exchange** after the
serve. Trailing shots separated from that exchange by a large gap (≥ DEADTIME_GAP_S) and
sitting in the run-up to the next serve are **between-point** → dropped from the rally and
from shot evaluation (e.g. r5 `sid56`). Trimmed shots are logged (frame + reason) and
rendered, never silently removed.

## 3. Design decisions (resolved with operator 2026-07-27)

1. **Fault/feed serves — RESOLVED.** The close second "serves" are NOT real serves: at
   2:24 it's the opponent feeding the ball to the server between points; at 4:54 there's
   only one serve. So **one serve per point.** The behind-baseline rule + the feed-dedup
   (2a) handle it; no fault-serve special case needed for now.
2. **Depth threshold — RESOLVED to the operator's hard rule:** behind the baseline
   (`dist_from_net > ~21–22 ft`). Calibrate the exact cut on this clip, validate by render.

## 4. Pipeline placement
- **Stage 5 `detect_shots`** — depth-anchored serve detection (2a).
- **Stage 7 `segment_rallies`** — serve-anchored segmentation (2b) + dead-time trim (2c).
Real ball only (gated as today). No new stage.

## 5. Outputs (additive)
- `shots.json`: serves flagged by the new rule; `serve_fault` events preserved.
- `rallies.json`: serve-anchored boundaries; per rally `trimmed_shot_ids` + reason.
- Everything downstream (metrics/rating/report) re-runs on the cleaned rallies/shots.

## 6. Acceptance test — validated by RENDERING
- **13 rallies, each starting with a serve.**
- **98 shots / 24 user** after trim (between-point balls removed) — vs 99 now with
  contamination.
- **17 volleys / 5 user** unchanged.
- Every detected **serve frame** and every **trimmed between-point ball** rendered
  (`tools/verify_*`) for operator eyeball before the counts are trusted.
- Downstream counts (dinks 18, volleys 17, etc.) must not regress against operator truth.

## 7. Honest limits
- Depth-anchored serves rely on `hitter_court_xy` — robust near the camera, zone-only far
  side; a serve from the far baseline is the far-depth case, but "deep vs kitchen" is a
  coarse call well within reach even far-side.
- The dead-time trim is a **heuristic** (isolated trailing shot), not outcome-based; a
  between-point ball hit *quickly* after the point (small gap) may still slip through —
  logged, not hidden. Serve-anchoring fixes the big structural errors; the trim cleans the
  residual.
- No claim about *why* a point ended (net/out/winner) — that stays camera-blocked. This is
  purely about **which shots are real rally play.**

## 8. Phasing (contract → code → smoke → commit)
1. **Depth-anchored serve detection** in `detect_shots` + tests; render all serve frames;
   **gate: operator confirms 13 serves land on real serves.**
2. **Serve-anchored segmentation + trim** in `segment_rallies`; re-check 13 rallies /
   98 shots; render trimmed balls.
3. **Re-run downstream** (metrics/rating/report); re-validate all counts vs operator truth.

## 9. OUTCOME — attempted 2026-07-27, REVERTED (bounded by foundation)

Phase 1 was built and validated against operator ground truth, and it **did not clear the
gate.** Findings:
- **Depth-anchored serves (behind-baseline rule) is a good PRINCIPLE:** it recovered the
  missed far-side serves the ball-appearance heuristic dropped, and every detected serve
  was behind the baseline. But it produced **15 serves with ~5 false positives** — deep
  shots that are NOT serves (deep returns, deep dead-time taps). Depth+gap cannot reject
  these; the "a serve gets returned" check that could relies on `hitter_side`, which is
  part of the same noisy attribution and made it WORSE (10 serves, dropping real ones).
- **Operator validation exposed a second, deeper problem — WHO served.** Corrections:
  8 serve-detection errors + 3 player-attribution errors. Root cause: **track
  fragmentation** (over 5 min the partner alone is 29 track-IDs; user 8, opp_a 19,
  opp_b 27) → re-assigning fragments to roles churns, and user↔partner (same side) swaps.
  Far-side opponent serves are also under-detected (distance/compression).
- **Serve-anchored segmentation on ~65-70% serves made the rally count WORSE:** splitting
  at interior serves over-split on the false positives → **18 rallies vs the gap-based 13
  (operator truth 13).** So the cleanup is bounded by serve quality, which is bounded by
  the tracking/identity foundation.

**Conclusion:** reliable serves / rally boundaries / who-served / per-shot outcome all bottom
out on the SAME foundation — single-camera per-shot signal + track fragmentation + identity.
Heuristics on top just trade one error for another. The honest path to these is fixing the
FOUNDATION (stable per-player identity: de-fragment tracks, pin user vs partner, far-side
recall), OR a camera change. Reverted to the gap-based baseline (13 rallies). Design kept as
the record; revisit once identity is stable.

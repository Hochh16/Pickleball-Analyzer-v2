# Pickleball-Analyzer-v2 — System Design & Accuracy Ledger (AUTHORITATIVE)

> **This document is the single source of truth. Read it before touching any stage.**
> It exists to stop the failure pattern that sank v1–v3 and was recurring in v4:
> decisions deferred downstream that became blockers, rationale lost across
> sessions, and stats reported without honest accuracy accounting.
>
> `ARCHITECTURE.md` = the system frame. `KNOWN_ISSUES.md` = the detailed issue
> log. **This file = the dependency map, the per-stage accuracy ledger, the
> honest trust map, the fundamental-limits decisions, and the foundations-first
> roadmap.** When a design decision is made, record it here.

_Last full audit: 2026-06-19 (whole-system parallel deep-read of all 13 stages +
real `data/pb_2min` outputs)._

---

## 0. Process rules (the discipline this document enforces)

1. **A stage is not "done" until it meets the accuracy its downstream consumers
   need, validated on REAL data.** No "good enough for now, fix later."
2. **No deferral without recording its blast radius here.** If we defer, we write
   down exactly which downstream stages/stats it corrupts, and consciously accept it.
3. **Every session reads this file first** and updates it when decisions change.
4. **Fundamental limits are decided, not deferred.** Where single-camera 2D can't
   deliver, we choose: accept-and-report-with-confidence, fix-at-capture, or
   scope-out — and record the choice in §5.

---

## 1. Pipeline & dependency graph

```
video + operator setup
 [1]  calibrate        → court.json, court_zones.json     (REAL ✓ ground-plane only)
 [2]  track_players    → players.parquet                  (REAL ✓ near / ✗ far-drift)
 [2.5]classify_tracks  → track_roles.json                 (REAL ~ user only)
 [3]  pose             → poses.parquet                    (REAL ✓ user+partner / ✗ no opp)
 [4]  track_ball       → ball.parquet     [4.5 trains the model]  (REAL ~ same-court / ✗ cross-court)
 [5]  detect_shots     → shots.json                       (REAL ~ in-rally / ✗ serves, fast-ball)
 [5.5]detect_bounces   → bounces.json                     (REAL ~ precision-high / ✗ ~21% recall)
 [6]  classify_shots   → classified.json                  (REAL ~ partial — see ledger)
 [7]  segment_rallies  → rallies.json                     (REAL ✓ boundaries / ✗ end_reason)
 [8]  compute_metrics  → metrics.json                     (SYNTHETIC-ONLY — never run on real ball)
 [9]  rate             → rating.json                      (SYNTHETIC-ONLY)
 [10] plan_improvement → improvement_plan.json            (SYNTHETIC-ONLY)
 [11] render           → annotated.mp4, timeline.json     (SYNTHETIC-ONLY)
```

Legend: ✓ trustworthy · ~ partial/conditional · ✗ broken/unvalidated on real data.

**The cascade in one sentence:** calibration is sound only on the ground plane →
the airborne ball has no height → ball speed and airborne position are
unrecoverable → shot *type*, bounce in/out, and end_reason degrade → every
ball-derived metric, the rating, and the plan inherit that noise; meanwhile a
parallel chain (far-side player drift → no opponent pose → no opponent stats) and
the role-awareness gap starve all per-opponent analysis.

---

## 2. THE TRUST MAP — what is real vs noise *right now* (read this first)

This is the section the reporting UI and every stat consumer must honor.

### Trustworthy on real data (durable value today)
- **Player position / movement / court-coverage** (Stage 2, **near side**): zone-time,
  lateral spread, area heatmap, distance/min. The single most durable real signal.
- **User identification** (Stage 2.5): ~85% frame coverage from the no-click seed.
- **User pose** (Stage 3): 99% detection; torso + right-side limbs reliable.
- **Rally boundaries** (Stage 7): ball-out-of-play segmentation, operator-validated.
- **In-rally shot detection** (Stage 5): ~92% in-rally ball recall → most in-rally
  strikes caught.
- **Calibration ground-plane geometry** (Stage 1): rmse ~0 for points on the court.
- **Bounce *positions when a bounce is truly detected*** (Stage 5.5): on-ground → projects correctly.

### Noise on real data (do NOT report as fact without the reliability gate)
- **Ball speed** (depth-foreshortened; court-plane projection is DEAD — explodes for airborne ball).
- **Shot *type* for drive/lob/volley** (deep bucket blurs; only serve + drop/dink-via-landing are sound, and landing covers ~21% of shots).
- **Serve stats** (serve detection under/over-detects → serve_fault_rate is noise).
- **Bounce in/out & ball-landing heatmap** (~21–38% bounce recall; rates computed over a biased subsample).
- **End_reason** (100% `unknown` on real pb_2min — bounce-recall-gated).
- **Error attribution** (depends on end_reason → mostly `unknown` owner).
- **Third-shot-drop rate** (drive/drop type confusion flips it directly).
- **The rating band & the improvement priorities** (≈70% of rating weight is the noisy synthetic dimensions, folded in without down-weighting).
- **Any per-opponent stat** (far-side drift + role-awareness gap).

### The single most important architectural finding
**No stage propagates per-shot / per-event confidence.** Every stat is computed
and rendered as a precise, certain number even when it rests on noise. Uncertainty
exists only coarsely (Stage 8 `reliability` family map, Stage 9 per-dimension
confidence/range, Stage 11 watermark). **Fixing confidence propagation end-to-end
is a cross-cutting foundational task (§6).**

---

## 3. Per-stage accuracy ledger

> Format: **Status · Produces · Key upstream need · Achievable-on-real · Top limitation → blast radius.**
> Evidence is from the 2026-06-19 audit against `data/pb_2min`.

### Stage 1 — calibrate  ·  REAL ✓ (ground plane only)
- Produces `court.json` (homography, ppf near/far, half/kitchen polygons), `court_zones.json`.
- Achievable: exact for **ground-plane** points (rmse 3.3e-13; kitchen cross-check ~8–9 px ≈ 0.2 ft). Near-baseline ppf 61.6 vs far 29.0 → **2.1× foreshortening**.
- **Top limitation → blast radius:** the homography is valid for **z=0 only**. Any
  point above the plane (airborne ball/contact) explodes toward the horizon
  (observed court_y ~1900 ft). → ROOT CAUSE of dead ball-speed (Stage 6/8) and
  garbage `impact_court_xy_ft` (Stage 5/7). Calibration emits **no intrinsics /
  camera height**, so there is nothing to build 3D on.
- Secondary: 4-point fit has **zero redundancy** (a misclick → confidently-wrong-but-clean cal; RMSE warning can't fire). Operator UI (`mark_court.py`) **lacks the top-down warp confirmation** — the best human sanity check is absent.

### Stage 2 — track_players  ·  REAL ✓ near / ✗ far
- Produces `players.parquet` (foot court_x/y_ft, bbox, in_court, transient).
- Achievable: **near side sound** (user court_y median 0.5, 0 rows > 44 ft). **FAR SIDE BROKEN:** 79% of far-half rows read court_y > 44, **max 150 ft on a 44-ft court**. Root cause: `foot_y = bbox bottom` under 2.1× far perspective.
- **Top limitation → blast radius:** far-side drift **deletes every opponent from
  Stage 3 pose** (the `y_max ≤ 44` scope gate rejects all 10 opp tracks) → no
  opponent pose → no opponent body-mechanics, no opponent position stats. This is
  a **foundational, tractable bug** (foot-point refinement / ankle landmark / per-depth correction).

### Stage 2.5 — classify_tracks  ·  REAL ~ (user only)
- Produces `track_roles.json` (user / partner / opp_left / opp_right / noise).
- Achievable: **user coverage 0% → 85.5%** (no-click geometric seed + appearance re-id) — the headline win. Partner good where simultaneity anchors it (conf 0.8); **opponents are a fixed conf-0.5 geometric guess**; 2 frames show an impossible double-user (re-id merge glitch); one user track sits off-court.
- **Top limitation → blast radius:** **role-awareness (F12) is the keystone blocker.**
  Only the `user` role is honored downstream (Stage 3 scopes by geometry, Stage 6
  applies handedness for user only). Blocks winner-side, per-receiver errors,
  opponent targeting, non-user stroke side, multi-role rating. Tractable (extend
  re-id to partner/opponents + wire roles into 3/6/8).

### Stage 3 — pose  ·  REAL ✓ user+partner / ✗ opponents
- Produces `poses.parquet` (33 landmarks × x/y/z/visibility, image space).
- Achievable: **99.2% detection**; torso + right-side limbs reliable; **left wrist/elbow ~0.4 visibility** (back-facing camera — and that's the *paddle* arm for a left-handed user). **Zero opponents posed** (cascaded from Stage 2 drift).
- **Top limitation → blast radius:** no opponent pose → opponent body-mechanics
  (F17) impossible until Stage 2 drift is fixed *or* Stage 3 scopes by Stage-2.5
  roles instead of geometry. Output is image-space px + relative z — **no metric
  3D**, so 2D joint angles are computable but true biomechanics (rotation, weight
  transfer) need depth/multi-view.

### Stage 4 — track_ball (+ 4.5 training)  ·  REAL ~ same-court / ✗ cross-court
- Produces `ball.parquet` (pixel_x/y, visible, confidence, interpolated). **2D only — no height.**
- Achievable: same-court recall **0.90** / cross-court **0.54** / held-out indoor **0.13**. In-rally recall ~0.92 (overall detect_frac 0.68 is dragged down by legit dead-time, not misses). Median 4.9 px error on clean frames.
- **Top limitations → blast radius:**
  - **(a) Cross-court 0.54 / indoor 0.13** vs the product requirement of varied
    courts → BLOCKER on any new venue (poisons all of 5–11). Fix = cross-venue
    training (Run-2, **blocked on Colab compute units**). Indoor *must* be in
    training (augmentation alone got 0.13).
  - **(b) Fast-ball / motion-blur misses** → the ball vanishes at the hardest hit
    (the most important frame) → missed shots. Likely a **hard limit** at this
    shutter/frame-rate; retrain did not recover it; gap-based impact-recovery was
    built and REVERTED (wrong failure mode).
  - **(c) No height (2D)** → ROOT of shot-speed/type errors (a down-court drive
    covers few pixels → reads slow).
  - **(d) Adjacent-court contamination** (single-ball argmax grabs a neighbor
    court's ball) → phantom shots; mitigated only downstream (Stage 5 gates).
  - **(e) Throughput ~2.9 fps, CPU-decode-bound** vs "many ≥5-min videos" → scaling blocker (GPU-decode task spawned).

### Stage 5 — detect_shots  ·  REAL ~ in-rally / ✗ serves
- Produces `shots.json` (frame, is_serve, hitter_court_xy_ft, hitter_side, speeds).
- Achievable: operator-validated in-rally; shot recall is **ball-detection-limited** (fast-ball misses). `hitter_side` from player ground position is the sound fix for garbage `impact_court_xy_ft`.
- **⚠ DATA HYGIENE:** the on-disk `data/pb_2min/shots.json` is **STALE** — it contains
  11 `impulse_recovered` shots from the reverted experiment (39 shots / 4 serves);
  **current code reproduces ~28 shots.** Re-run before trusting any count.
- **Top limitation → blast radius:** **serve detection** (the recurring blocker).
  Under-detects (serve missed when ball is blurred at launch) and can over-flag
  (mid-rally occlusion looks like dead-time). → Stage 6 mislabels serves as drives,
  Stage 7 server attribution / courtesy-feeds, Stage 8 serve-fault rate. Mitigation
  available: "server-behind-baseline" court-position gate (data exists, not built).

### Stage 5.5 — detect_bounces  ·  REAL ~ precision-high / ✗ recall
- Produces `bounces.json` (court_xy_ft, court_zone, is_in_court, between_shots).
- Achievable: **15 bounces / 39 shots ≈ 38%, but only ~8 distinct shots have a landing → ~21% shot coverage.** Tuned hard for precision (operator 4/4 correct); the y-flip-required gate is the dominant recall killer. **Association can be wrong** (bounce_id 11 → court_y 40.8 = far baseline for a soft shot → an airborne point mis-accepted).
- **Top limitation → blast radius:** **low recall is the cross-cutting lever.** Gates
  shot-type-via-landing (Stage 6, ~21% coverage) AND end_reason (Stage 7, 100%
  unknown). Most tractable fix: **displacement-based reversal at the refined
  contact frame** (the windowed y-flip smears the sharp 1-frame bounce); make the
  apex/off-court filter symmetric to catch in-court airborne points.

### Stage 6 — classify_shots  ·  REAL ~ (partial)  ·  ⚠ uncommitted landing changes
- Produces `classified.json` (stroke_side, shot_type, is_volley, confidences, features).
- Achievable per type: **serve** reliable (gated by Stage-5 `is_serve`); **drop/dink
  via landing** reliable but only ~21% coverage (and in pb_2min *zero* dinks/drops
  actually hit the landing path — all 16 used the corrupted fallback); **drive/lob**
  blur (arc-only separation, noisy); **volley sub-type** unreliable. **stroke_side
  89.7% unknown** (role-gated to user).
- **Top limitation → blast radius:** depth-corrupted speed + deep-bucket blur +
  ~21% landing coverage → shot-mix feeding Stage 8/9/10 is unreliable for fast/deep
  shots. **The uncommitted landing-aware classifier** (built this session) is a net
  positive for drop/dink but inherits bounce-association errors and does NOT fix the
  drive/lob/serve blur — see §7 decision.
- **⚠ Contract is STALE** (doesn't document the landing path) and stage_version was not bumped.

### Stage 7 — segment_rallies  ·  REAL ✓ boundaries / ✗ end_reason
- Produces `rallies.json` (boundaries, server attribution, end_reason).
- Achievable: **boundaries GOOD** (ball-out-of-play, operator-validated; robust to
  missed shots/serves — 4/8 rallies started by inferred serve). **end_reason 100%
  `unknown`** on real pb_2min (every rally has 0 detected bounces after the last
  shot → honest-unknown by design).
- **Top limitation → blast radius:** end_reason unbuildable until bounce recall
  rises → Stage 8 cannot compute serve-fault, error-by-player, or point-ending mix.
  The honest-unknown discipline is correct (failures visible, not guessed).

### Stages 8–11 — compute_metrics / rate / plan / render  ·  SYNTHETIC-ONLY
- **Never run on the real ball.** Scaffolds whose real-data accuracy is unverified.
- **Durable now:** position/movement/team stats + player-position heatmaps (Stage 2 derived).
- **Noise now:** serve_fault_rate, error_attribution, bounce_in_out, third_shot.drop_rate, shot_mix → and therefore the **rating band** (≈70% synthetic weight, no down-weighting) and the **plan priorities**.
- **Architectural gap:** no per-event confidence propagation; rates divide by
  *detected* counts (low recall silently biases the subsample, not the rate);
  Stage 8 reliability map even **over-claims** rally-length as `real_data`.
- **Honesty machinery that DOES work:** `ball_source`/synthetic watermark,
  `reliability` family map, per-dimension confidence + range, `pending_real_ball`
  nulls, provisional focus-area flags. A reporting UI MUST bind to these.

---

## 4. Cross-cutting limitations (recur across many stages)

| # | Limitation | Stages hit | Nature |
|---|---|---|---|
| C1 | Ball-detection recall (cross-court, fast-ball) | 4,5,5.5,6,7,8,9 | dominant downstream limiter |
| C2 | **No ball height / 3D** (single corner cam) | 1,5,6,8 | **fundamental** — see §5 |
| C3 | Serve detection | 5,6,7,8 | tractable |
| C4 | Bounce recall | 5.5,6,7,8 | partly tractable (gate) / partly C1 |
| C5 | Role-awareness (only user tracked) | 2.5,3,6,8,9 | keystone blocker, tractable |
| C6 | Far-side player position drift | 2,3,8 | foundational, tractable |
| C7 | Cross-court / cross-venue generalization | 4 | required (product), data problem |
| C8 | Throughput (CPU-decode-bound) | 2,4,11 | required (product), engineering |
| C9 | No per-event confidence propagation | 8,9,10,11 | architectural, tractable |
| C10 | Uncalibrated rating thresholds | 9,10 | needs rated-footage corpus |

---

## 5. Fundamental-limits reckoning — DECISIONS REQUIRED

These cannot be "fixed later." Each needs an explicit choice; record it here.

| Limit | What it costs | Options | Status |
|---|---|---|---|
| **Ball height / 3D** (C2) | exact ball speed; certain lob-vs-drive; per-shot in/out at contact; airborne position | (a) accept + report shot-type with confidence, lean on landing/arc/zone; (b) add capture (2nd camera or higher mount) → unlocks 3D; (c) parabola/gravity fit to recover z (partial, research) | **DECIDED 2026-06-19 (David): investigate (c) parabola/gravity z-recovery feasibility FIRST, then choose accept-vs-capture.** Roadmap proceeds without it meanwhile. |
| **Fast-ball motion blur** (C1) | missed hard-hit shots → undercounts rally length & drive mix | (a) accept (flag undercount); (b) capture-side (higher shutter / frame rate) | **OPEN** |
| **Spin** | no topspin/slice metric | scope OUT (permanent single-cam limit) | **Recommend: scope out** |
| **Rating calibration** (C10) | rating band is heuristic, not measured | report as directional + range until a rated-footage corpus exists | **Accept (directional-only) for now** |

---

## 6. Foundations-first roadmap (dependency-correct order)

Fix upstream accuracy *before* anything depends on it. Each item lists its lever
and what it unblocks. (Tractability from the audit; none requires abandoning 2D.)

1. **Stage 2 far-side player drift (C6).** ✅ **DONE 2026-06-19** (commit pending).
   Implemented: Stage 2 temporal foot-point smoothing + `court_pos_reliable` flag;
   **Stage 3 scope is now role-based** (pose user + partner + opponents by Stage-2.5
   role, geometric gate only as fallback) — the old `court_y.max() ≤ 44` gate
   deleted every opponent from pose. Validated on pb_2min: **all 10 opponents
   restored, 0 noise admitted.** Unblocks opponent pose → opponent body-mechanics
   (F17). *Honest findings:* (a) foot smoothing is **marginal** — far-side error is
   correlated bbox imprecision × ~4 px/ft horizon compression, not spikes; far-side
   absolute position stays **zone-precision (~±5 ft)**, a camera-geometry limit
   (flagged via `court_pos_reliable`). (b) **Role-based scope's robustness now
   DEPENDS on Stage-2.5 role quality** — clean on pb_2min, messy on the test fixture
   (38 opp fragments). So it pulled a slice of #2 forward; #2 must be hardened for
   role-based scope to be robust on messy real videos. (c) Posing all opponents is
   ~5× the pose compute (throughput note, C8). (d) Far-opponent poses are
   lower-fidelity (small crops) — relevant to opponent body-mechanics quality.
2. **Stage 2.5 role-awareness (C5).** 🔄 **IN PROGRESS 2026-06-19.** Done: opponents
   are now grouped into **two stable IDENTITIES `opp_a` / `opp_b`** by the same
   two-anchor appearance + continuity re-id as user/partner (NOT position L/R —
   they switch sides), at honest moderate confidence (cap 0.75; far crops noisier).
   System-wide rename opp_left/opp_right → opp_a/opp_b (classify_tracks, pose,
   compute_metrics, render + tests + contracts). Validated on pb_2min: same correct
   5/5 partition as the old geometric split, now identity-based + appearance-grounded
   (conf 0.6–0.75 vs flat 0.5). **This hardens the role-based pose scope (#1).**
   *Remaining #2:* non-user handedness into Stage 6 stroke-side (needs operator
   roster input — currently all `unknown`); per-opponent stats wait for the real-ball
   Stage 8 work (#7); tighten opponent appearance on harder/side-switching clips.
   Unblocks: opponent body-mechanics, winner-side, per-receiver errors, targeting.
3. **Confidence propagation (C9).** Carry per-shot/-event confidence + sample sizes
   through 6→7→8→9→10→11 so every reported number has an honest reliability. *The
   architectural fix that makes "honest stats" real, not coarse.*
4. **Stage 5.5 bounce recall (C4).** Displacement-based reversal at the contact
   frame + symmetric apex filter. Unblocks: shot-type-via-landing coverage (Stage
   6) AND end_reason (Stage 7) — the cross-cutting stats lever. *Tractable, no GPU.*
5. **Stage 5 serve detection (C3).** Server-behind-baseline gate + real-data
   gap-tuning. Unblocks: serve labels (6), server attribution (7), serve-fault (8).
6. **Stage 4 cross-venue (C7) + throughput (C8).** Run-2 indoor training (blocked on
   compute units) + GPU-decode. Required for the product's varied-court / many-video
   reality. *Run-2 set up; awaiting compute.*
7. **Then** re-run Stages 8–11 on the real ball and lift the synthetic caveat —
   only after the above, so the stats layer is validated against real, confidence-
   carrying inputs rather than re-scaffolded on noise.

Ball **height/3D (C2)** sits across all of this as the §5 decision; the roadmap is
designed to extract maximum value *without* it (landing/zone/arc for type,
ground-anchored positions for everything else), so 3D becomes an enhancement, not a
prerequisite.

---

## 7. Operator decisions — RESOLVED 2026-06-19 (David)

1. **Ball height / 3D (§5):** **investigate parabola/gravity z-recovery feasibility
   first**, then decide accept-with-confidence vs capture change. (Parallel research track.)
2. **Foundations order (§6): CONFIRMED.** **Start with Stage 2 far-side drift** (widest
   reach, no GPU), then proceed down the dependency order.
3. **Stage 6 landing-aware classifier:** **commit now** (net positive for drop/dink +
   honest confidence), revisit Stage 6 fully after bounce-recall + confidence land.
4. **UI: build the input/setup UI for real now + a thin reporting skeleton** that
   round-trips the locked schemas and honors confidence/reliability. Defer only the
   rich reporting content. (Parallel track.)

**Active work order:** (1) Stage 2 far-side drift → (2) role-awareness → (3) confidence
propagation → (4) bounce recall → (5) serve detection → (6) cross-venue ball → (7) re-run
8–11 on real ball. Parallel: z-recovery feasibility spike; input-UI + reporting skeleton.

---

## 8. Future-items register (feasibility + prereqs)

Condensed from the audit (full detail in KNOWN_ISSUES + stage contracts). F# are stable IDs.

**Vision/ball:** F1 cross-venue training (feasible, data; *blocked on compute*) · F2
fast-ball recall (hard / capture-limited) · F4 court-aware multi-ball detector
(feasible) · F5 GPU-decode throughput (feasible, required) · F6 TrackNetV3/WASB
escalation (feasible).

**Geometry/physics:** F7 court-plane ball speed (feasible but **validated DEAD** for
airborne ball — landing/zone replaces it) · **F8 3D/ball-height (fundamental — §5)** ·
F9 spin (permanent-out) · F10 per-shot in/out (feasible for ground bounces) · F11
diagonal service-box (feasible; needs serve-alternation state).

**Role-awareness/attribution:** **F12 partner/opp role-awareness (keystone, feasible)**
→ unblocks F13 winner-side, F14 per-receiver errors, F15 opponent-backhand targeting,
F16 non-user stroke side, F25 multi-role rating.

**Metrics/coaching:** **F17 body-mechanics / pose-technique stage** — David-named;
**feasible NOW for the USER** (pose is real + durable, ball-independent); **blocked for
opponents** until Stage 2 drift (C6) is fixed; needs handedness wiring (esp.
left-handed → low-visibility paddle arm) + visibility-weighting. F18 forced/unforced
errors (feasible w/ real ball; highest-priority Tier-B) · F19 dink tolerance / 3rd-shot
outcome · F20 courtesy-feed exclusion · F22 robust serve detection (C3) · F23 rating
calibration (needs rated corpus — C10) · F24 uncaptured skills (mixed).

**Platform:** F27 UI (§7.4) · F28 cross-video trend tracking (feasible; needs stable
schema + cross-video identity) · F29 render perf · F30 multi-person pose model · F31
per-venue fine-tune loop · F32 ball-label auto-tuning.

---

## 9. The failure pattern (why v1–v3 died, and what we changed)

- **v1/v2/v3 ball detection** failed with *different mechanisms but the same outcome*
  (memorized background / collapsed-to-zero / sub-SNR). **Lesson: when repeated
  approaches fail differently but identically, fix the DATA, not the technique.** v4
  succeeded by changing inputs (4K/60fps) + matching the tool (temporal TrackNet,
  focal loss, 1280×720 — the "512×288 trap" was the easily-missed fix).
- **The deferral pattern this document stops:** adjacent-court contamination deferred
  from Stage 2 → resurfaced as phantom shots; role classification deferred → Stages
  3/5/6/8 each re-derived geometry ("duplication waiting to drift"); `impact_court_xy_ft`
  built airborne → unusable, salvaged late by `hitter_side`. Each was a locally-
  reasonable "later" that became a downstream blocker. §0 rules + this ledger are the
  countermeasure.

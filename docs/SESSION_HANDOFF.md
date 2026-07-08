# Session Handoff — Pickleball-Analyzer-v2 (updated 2026-07-07)

## 2026-07-07 — Stage 4 Run 2 relaunched (warm-start + per-venue eval) — READ FIRST

**Work order item #3 (Stage 4 cross-venue detector) is IN PROGRESS.** David bought
Colab Pro+; Run 2 is training on Colab now (RTX PRO 6000). This session:

- **Reconciled the notebook.** The live Colab `finetune_v4.ipynb` had DIVERGED from the
  repo copy (stronger photometric aug + a from-scratch training loop + no resume block;
  the repo copy was stale and even had a `resume_best` NameError bug). Downloaded the
  live copy, made it the repo source of truth, and rebuilt it for Run 2.
- **Two findings from the live notebook's cached Run-1 output:** (a) the documented
  **"0.90→0.858 same-court regression" is largely a model-SELECTION artifact** —
  recall actually hit **0.96**, but the `score = recall − fp` selector kept a
  low-recall/low-fp epoch. There is *also* a real precision cost (fp 0.10–0.24 for ≥0.90
  recall vs baseline 0.018). (b) **held-out indoor recall maxed at 0.126** → augmentation
  alone doesn't generalize indoor; indoor must be trained on.
- **Run 2 design (commit `ed3d02d`):** warm-start from the clean 0.90 baseline
  (`MyDrive/ball_model_v4_base.pt`) at LR 1e-4; **all 3 venues in training with per-venue
  held-out slices** (pb_2min home guardrail; court2 + indoor 88/12 leakage-free split);
  **fp-capped selection** (max mean per-venue recall s.t. home fp ≤ 0.05) → saves
  `ball_model_v4_run2.pt` + `validation_report_run2.json` (0.90 baseline never overwritten).
  Data-split logic validated locally; warm-start confirmed loading the 0.90 base on the
  live run (recall 0.9024).

**NEXT (resume here):** read `validation_report_run2.json` when the run finishes.
- **home_rec** must hold **≥ 0.90 at fp ≤ 0.05** (warm-start preserved precision).
- **court2_rec / indoor_rec** (held-out slices) are the cross-venue numbers that matter.
- If home_rec collapsed or indoor stayed low, **LR (1e-4) is the first knob to revisit**;
  next options are more epochs, or per-venue LR.
- If Run 2 beats the baseline across venues, download `ball_model_v4_run2.pt` →
  `data/models/`, then proceed to work-order **#4** (lock real-ball upstream 5→5.5→6→7,
  one stage at a time). SYSTEM_DESIGN §6 item 3 + §7 updated with this state.

**Deployment note:** Colab needs, in `MyDrive/` root: `pb_v4_upload.zip` (bundle, current)
+ `ball_model_v4_base.pt` (= local `data/models/ball_model_v4.pt`, the 0.90 model). The
G4/RTX-PRO-6000 GPU auto-sizes BATCH to 12. Pro+ background execution runs it 24h even if
the tab closes.

---

## 2026-06-21 — COURSE CORRECTION + plan reset (read this first)

**What happened:** this session built **Foundation #3 (confidence propagation)** as
inline `{value, confidence, n, limited_by}` wrappers across Stages **8→9→10→11**
(commits `0d116b2` S8, `8350724` S9, `39b2c41` S10; **S11 uncommitted**), plus the
**operator-vs-player separation** in Stage 10 (`operator_considerations`, surfaced
only when a real limiter bites; David's call). The mechanism is sound and smoke-tested.

**BUT it was validated only on the SYNTHETIC `test_clip`, and the session started
declaring stages "done" + validating Stage 11 while Stage 7 was still unvalidated on
real data** — rebuilding the v1–v3 compounding-error failure. **Operator stopped it.**

**Corrections committed this session:**
- **SYSTEM_DESIGN §0 rules 5–9** added (synthetic ≠ validation; real = all venues;
  strict dependency order; one stage at a time; downstream-sufficiency review is part
  of "done"). **Read §0 before any work.**
- **§6/§7 roadmap REORDERED:** the real foundation gap is the **ball detector across
  venues**, not confidence. v4 trained on **outdoor same-court only** — different-court
  (0.54) + indoor (0.13) **never trained** (contract_v4.md). Confidence work is
  reframed as **built-but-UNVALIDATED**, deferred to after the real-ball upstream is locked.

**NEXT (agreed work order):**
1. **Stage 4 cross-venue retrain — NEEDS COMPUTE (operator funding Colab).** All 6
   clips already LABELED; **Run 2 (3-venue) is CONFIGURED in `finetune_v4.ipynb`**,
   blocked on compute. So: re-run Run 2, achieve per-venue recall on all 6 clips
   **WITHOUT regressing same-court** (Run 1 regressed pb_2min 0.90→0.858 — the open
   challenge; may need venue-balancing / 1080p / per-venue heads). Production detector
   today = original same-court v4. *Gates real validation of Stages 5–11 on 2 of 6 clips.*
2. Lock real-ball upstream **5→5.5→6→7**, one stage at a time, operator-validated,
   each with a downstream-sufficiency review.
3. Stats layer **8→9→10→11 + confidence**, re-validated on real (reuse this session's code).

**Decide before resuming:** whether to commit the Stage 11 confidence code as
work-in-progress (smoke-passed, unvalidated) or hold it.

---

# Session Handoff — Pickleball-Analyzer-v2 (prior: updated 2026-06-19)

> **READ `SYSTEM_DESIGN.md` (repo root) FIRST.** It is the authoritative source of
> truth: dependency map, per-stage accuracy ledger, the honest trust-map (what's
> real vs noise *today*), fundamental-limits decisions, the foundations-first
> roadmap, and the F1–F32 future register. This handoff is the session *log*;
> SYSTEM_DESIGN.md is the live *state*. It's also pinned in auto-memory.

## The design philosophy (NEW — how we work now)

v1–v3 (and v4 was repeating it) failed from **deferred decisions becoming
downstream blockers, lost cross-session rationale, and stats reported with no
honest accuracy accounting.** The countermeasure is SYSTEM_DESIGN.md §0:
1. A stage isn't "done" until it meets the accuracy its downstream needs,
   **validated on REAL data**. No "good enough for now, fix later."
2. **No deferral without recording its blast radius** in the ledger.
3. **Every session reads SYSTEM_DESIGN.md first** and updates it when decisions change.
4. **Fundamental limits are decided, not deferred** (accept-with-confidence /
   fix-at-capture / scope-out).

## Status (end of 2026-06-19 session)

This session pivoted from symptom-patching to a **full whole-system parallel audit**
→ SYSTEM_DESIGN.md, then began the **foundations-first roadmap**. Commits:
- `fcddcb6` — **SYSTEM_DESIGN.md** (the audit / source of truth).
- `94d5b1f` — **Stage 6 v0.4.0**: landing-aware shot type (drive/drop/dink from the
  bounce landing — sound where the airborne ball's depth-corrupted speed is not;
  ~21% landing coverage, honest confidence).
- `4b9c25d` — **Foundation #1**: Stage 2 far-side drift + **role-based pose scope**.
  Opponents were deleted from pose by a `court_y.max()≤44` gate (far-side jitter
  spikes past the baseline); now Stage 3 scopes by Stage-2.5 role. Opponents
  restored (validated pb_2min). Far-side absolute position is **zone-precision
  (~±5 ft)** — a camera-geometry limit, flagged via `court_pos_reliable`.
- `736b567` — **Foundation #2 (core)**: opponents grouped into two stable
  IDENTITIES **`opp_a`/`opp_b`** by appearance + continuity re-id (NOT position
  L/R — they switch sides), honest moderate confidence. System-wide rename
  opp_left/opp_right → opp_a/opp_b.

**Roadmap (SYSTEM_DESIGN §6): #1 done · #2 core done · #3 = NEXT.**

## NEXT SESSION: Foundation #3 — confidence propagation (design LOCKED)

Thread honest per-event confidence through Stages 6→11 so **every reported number
carries its reliability** (the audit's #1 architectural finding: no stage
propagates per-event confidence; every stat renders as certain even when it rests
on noise). **Decided with David 2026-06-19:**
- **Option 2 — inline `{value, confidence, n}` wrappers** on every metric
  (confidence inseparable from the value; no orphan numbers; chosen over a parallel
  block because that drifts).
- **All three stages in one pass: 8 → 9 → 11.**
- **Stage 8:** `conf_n()` aggregator (mean per-event confidence × small-sample
  penalty) + `mv()` wrapper; a per-metric **confidence-source map** — clean sources
  exist (`shot_mix`←`shot_type_confidence`, `bounce_in_out`←bounce confidence,
  `serve_fault`←`end_reason_confidence`); metrics with **no** per-event source
  (`rally_length_shots`, `match_span`) need a deliberate decision on what their
  confidence *means*. That per-metric mapping is the real work.
- **Stage 9:** read `.value`; set per-dimension confidence from each metric's real
  `.confidence` (retire the coarse synthetic/real heuristic).
- **Stage 11/timeline:** surface per-metric confidence so the report gates each number.
- **Honest caveat to document:** captures classification-noise + sample-size, **NOT
  recall-bias** (Stage 8 can't see missed events — that stays a documented limit).

## Parallel tracks (David chose; queued, not started)
- **z-recovery feasibility spike** — parabola/gravity ball-height from a single
  camera; informs the SYSTEM_DESIGN §5 ball-height/3D decision (currently "investigate first").
- **Input-UI + reporting skeleton** — #2 surfaced a concrete need: non-user
  handedness needs a UI to show the operator "who is `opp_a`" before they can label it.

---

_Below: the prior session log (history; superseded by SYSTEM_DESIGN.md for current state)._

## DONE 2026-06-16: foundation hardening (Stage 5 v0.3.0 + Stage 7 v0.2.0)

Operator review of the FULL pb_2min clip (not just the per-shot overlay) exposed
data-quality issues that corrupt stats. Fixed + validated + committed:
- **Stage 5 v0.3.0** (`066d63e`): **adjacent-court contamination gates**
  (serve-must-launch-a-sustained-run + impulse-impact-must-not-teleport-in —
  rejects neighbouring-court phantom shots/serves the single-ball detector grabs
  when ours is occluded); **reliable `hitter_court_xy_ft`/`hitter_side`** from the
  hitting player's GROUND position (the airborne `impact_court_xy_ft` projection
  is garbage — court_y up to ~1900 ft on a 44-ft court); **serve de-duplication**.
- **Stage 7 v0.2.0** (`14def29`): **rally boundaries from the ball going OUT OF
  PLAY** (sustained not-in-play run), NOT `is_serve`/time-gap. KEY INSIGHT: during
  a point the ball is in flight (visible ~every frame, <0.25s absences); between
  points it's dead 3-4s. A missed shot leaves the ball flying → no false split.
  This is a **general physical signal**, the thing that finally made David's rally
  boundaries correct ("top labels are correct"). Side from `hitter_side`;
  zero-bounce end_reason → `unknown` on real ball.

**Residuals (all tied to deferred work, NOT new hacks):** serve→drive labels +
courtesy-feed-as-rally-start (need serve detection); drive↔drop/dink type errors
(2D depth-speed limit, need court-plane/3D ball speed); missed shots + mostly-
`unknown` end_reason (ball-detection recall). See `KNOWN_ISSUES.md`.

**Real-vs-synthetic adaptation pattern (applies to every remaining stage 6–11):**
1. **`is_user` from `track_roles.json`** (role 'user'), NOT players.parquet's
   click-only flag (empty in the no-clicks flow). Every stage reading is_user.
2. **Resolution scaling**: px thresholds × `frame_width/1920` (4K = 2×).
3. **fps scaling**: frame-count windows × `fps/30` (60fps = 2×).
4. **Real-world-phenomenon filters gated to real ball** (`ball_source=="real"`):
   the synthetic placeholder lacks the noise/handling, so gating keeps the
   synthetic smoke bars valid (e.g. Stage 5 net-side ball-handling rejection;
   Stage 5.5 y-flip-for-all + apex filter).
5. **Validation = operator spot-check overlays** (render markers on the video,
   David confirms) — there is no real-data ground truth to auto-grade against.
6. **Gotcha:** `(x > thresh)` on numpy floats yields a **numpy bool**; `if b is
   not True` then rejects everything (numpy True ≠ Python True). Wrap in `bool()`.

## Where the project is

- **Stages 1–11:** implemented, end-to-end runnable. Last run on the **synthetic**
  placeholder ball — every ball-derived output is a validated scaffold until
  re-run on the real ball.
- **Stage 4/4.5:** **v4 WORKING.**
  - Trained detector `data/models/ball_model_v4.pt` (720p TrackNet, 3-frame/9-ch):
    **val recall 0.90 same-court / 0.54 cross-court**, fp 0.02.
  - Inference: `stages/track_ball/track_ball_v4.py` (720p + trajectory
    post-processing) + smoke `test_track_ball_v4.py`. Validated vs ground truth on
    pb_2min frames 300–420: 39/40 balls, **median 4.9px at 4K**, 100% within 25px.
  - Production (full-clip) inference: `stages/track_ball/infer_v4.ipynb` (GPU/Colab,
    built by `tools/build_infer_v4_nb.py`). **Real full-clip `ball.parquet`
    produced for `data/pb_2min/`**: 7164 frames, 4418 visible + 426 interp + 2320
    not-visible, detect_frac 0.676, coords in-bounds, conf mean 0.78. Visually
    spot-checked on the longest rally — looks good.

## What was done this session (2026-06-11/12)

- **Drove `infer_v4.ipynb` on Colab end-to-end via the Claude-in-Chrome browser
  MCP** (Claude *can* drive Colab once the Chrome extension is connected) and
  produced the first real full-clip `ball.parquet` + `ball.meta.json` for pb_2min;
  downloaded + validated them locally; rendered a local overlay spot-check.
- **Fixed a T4 OOM:** the notebook hardcoded `BATCH=16`, which OOMs a 15GB T4 at
  720×1280. The builder now scales BATCH to GPU memory (T4→4, >20GB→8, >32GB→16).
  Committed **`1621541`** ("Stage 4 v4: T4-safe GPU batch size").
- **Logged two now-first-class requirements** (product reality: many ≥5-min videos,
  varied courts) → see `KNOWN_ISSUES.md`:
  - **Throughput:** full-clip inference is **CPU-decode-bound at ~2.9 fps**
    (~40 min for 2 min of 4K/60; ~100 min for 5 min). A background task to switch
    to GPU/hardware decode (NVDEC) was spawned.
  - **Cross-court generalization:** 0.90 same-court vs **0.54 cross-court** — must
    close before relying on the detector across indoor/outdoor venues.
- **Updated docs** (this commit): ARCHITECTURE.md (Stage 4/4.5 + pipeline status no
  longer "paused"), KNOWN_ISSUES.md (v4-landed update, synthetic-caveat-still-applies
  note, two new issues), this handoff.

## DONE 2026-06-13: Stages 1–3 on pb_2min (no clicking) + user-tracking fixes

pb_2min now has the full Stage 1–3 set: `court.json`/`court_zones.json` (operator
court clicks via `tools/mark_court.py`), `players.parquet`, `track_roles.json`,
`poses.parquet`. Three improvements landed, driven by "track the user extremely
well" + "no user-clicking":
- **No-click user ID** (`4b8e4b8`): operator only sets handedness/baseline/
  starting-corner; Stage 2.5 seeds the user geometrically from
  `user_starting_corner`. `user_clicks.json` is now an optional override (Stage 2
  + 2.5). `tools/mark_user.py` is the override clicker.
- **Appearance re-id** (`b348d98`): Stage 2.5 follows the user across ByteTrack
  ID swaps / gaps / side-switches by clothing-color + height. pb_2min user
  coverage 68% -> **85.5%** (visually verified on a role-overlay clip).
- **Role-aware pose** (`f349141`): Stage 3 takes `is_user` from the role `user`
  and poses every user track (incl. behind-baseline 1663). pb_2min: user pose
  6125 rows @ 99.1% detection, all 3 user tracks `[1, 1554, 1663]`.

**Product note (David):** the final app's input/setup flow must let the user
select their own handedness; the court/user inputs are product UI, not dev
fixtures (see memory `project_product_requirements`).

## DONE 2026-06-14: Stages 5 + 5.5 on the real ball (pb_2min)

Both operator-validated via spot-check overlays and committed (real-ball
adaptations per the pattern above):
- **Stage 5 (detect shots)** `8aa9164`, v0.2.0: 304 → 45 real shots (all real
  over-net strikes by David's eye). Adaptations: teleport-drop (don't crash on
  outliers), 4K resolution + 60fps scaling (the fps scaling collapsed 2–3
  duplicate detections/strike), is_user-from-roles, and a **net-side
  ball-handling filter** (real players catch/bounce/hold the ball between points
  = a sharp dir-change at a hand; every rally shot crosses the net, so consecutive
  same-side impacts = handling — keep the LAST of each run).
- **Stage 5.5 (detect bounces)** `740fac9`, v0.2.0: 135 → 16 bounces (4/4
  validated). Adaptations: scaling, **apex/off-court filter** (reject bounces
  projecting far off-court = ball in the air, not on the ground),
  **ground-contact refinement** (snap to lowest pixel_y for accurate far-court
  zones), and **y-flip-for-all on real ball** (a real bounce reverses vertical
  down→up; impulse-with-no-reversal = mid-air wobble). Deferred: bounce occluded
  behind the net is missed (ball-quality cap); is_at_feet edge case.

## DONE 2026-06-15: Stage 6 (classify shots) on the real ball (pb_2min)

Operator-validated via spot-check overlay and committed. `classified.json`: 45
shots, types {drive:14, drop:12, serve:8, dink:6, lob:4, reset:1}, **0 unknown**,
volleys 9. Real-ball adaptations (v0.2.0 → **v0.3.0**):
- **Volley decoupled from the bounce LIST → recall-focused local trajectory scan.**
  The precision-tuned Stage 5.5 bounce list under-detects → false volleys. The
  volley flag now scans the inter-shot ball directly for a **ground bounce =
  interior local peak in pixel_y** (ball momentarily lowest on screen, descends
  in + rebounds out). **Gotcha that cost a retry:** do NOT use the *global*
  pixel_y max — the segment starts at high pixel_y (previous contact is low on
  screen) and the arc apex is a pixel_y *minimum*; both must be ignored. Bounce
  list kept only as an occlusion fallback. Pipeline volleys 27 → 9; operator
  confirmed volley/bounced on a 7-shot window.
- **Lob requires below-drive speed** (a lob is lofted AND slow; fixed fast drives
  reading as lobs on the noisy ball).
- **Tweener arc-shape tiebreak** (16–25 ft/s dead-zone): flat=drive, lofted=drop;
  drained all 7 "unknown" types into the right bucket.
- **fps + resolution scaling** of the px/frame thresholds (4K/60fps).

**Three residuals logged in KNOWN_ISSUES (NOT fixable in Stage 6):**
1. **Depth/height corrupts pixel-speed** → a drive hit down-court reads as slow
   and mistypes as a drop (f3541: a real drive measured 4.2 px/f). Proper fix =
   **homography-projected court-plane ball speed** (also feeds Stage 8 metrics)
   or 3D. The arc-tiebreak only covers the 16–25 ft/s band.
2. **Serve labeling** depends on Stage 5 `is_serve` (f3470 missed → "drive"). Fix
   in **Stage 5**.
3. **Courtesy/between-point feeds** read as volleys (f3148) — correct but not a
   rally shot. Exclude in **Stage 7 (rally segmentation)**.

## NEXT STEPS (me, next session)

1. **Improve Stage 4 ball detection — the agreed high-impact "do it now" work**
   (before building Stages 8–11, to avoid compounding errors into every stat).
   **Refined diagnosis (2026-06-16):** the "62% ball-visible" was misleading —
   **in-rally recall is ~92%**; the dead-time drags the average down. The real
   miss is **FAST-BALL under-detection**: at a hard hit the ball moves ~250 px/f
   and is **lost to motion blur** (tracked max only ~67 px/f in the missed region
   vs 252 in clean rallies), so the shot has no ball at impact. An **impact-recovery
   experiment (gap-based) was built and REVERTED** — it only recovers *gap-hidden*
   shots, not these fast-ball misses (the ball is "visible" but slow/jittery, not
   absent). So the fix is genuinely **detector quality**, not a Stage 5 heuristic.
   **Plan = ONE combined retrain:**
   (a) **fast-ball / motion-blur recall** — label more hard-hit / blurred-ball
       frames (the same-court outdoor clips are a rich source);
   (b) **cross-court generalization** — add the different-court + indoor clips
       (closes the 0.90 same-court vs **0.54 cross-court** gap, a hard product
       requirement). Do both in one training pass, not several.
   **Operator data on hand (David, 2026-06-16):** 4 outdoor videos of the SAME
   (pb_2min) court + 1 outdoor video at a DIFFERENT court + 1 indoor video. The
   different-court + indoor are the generalization set; the same-court clips add
   fast-ball examples. (Training is GPU/Colab, operator-driven, like `infer_v4`.)
   Also fold in the **adjacent-court contamination** root cause (Stage 4
   single-ball — KNOWN_ISSUES); a court-aware/continuity detector helps recall +
   contamination together.
   **Resume here:** quantify the fast-ball failure modes on pb_2min (speed/blur
   correlation, where misses cluster) to target labeling, set up the
   label→retrain→validate loop, then drive Colab.
2. **Then Stages 8 → 9 → 10 → 11** on the real ball. In **Stage 8**, build
   **court-plane / height-aware ball speed** (KNOWN_ISSUES Stage 6 depth-speed) —
   the right speed signal for metrics; retro-improves Stage 6 drive/drop typing.
3. **Calibrate Stages 9/10** against real rallies (uncalibrated until now).
4. Stage 11 synthetic-ball watermark drops automatically once `ball_source != synthetic`.

**Real-ball boundary lesson (carry forward):** the GENERAL rally-boundary signal
is **ball-out-of-play** (sustained not-in-play run), not serves or time-gaps —
robust to missed shots. The same "use a physical signal, gate real-only, validate
by operator spot-check" pattern applies to every remaining real-ball adaptation.

Notes carried forward: Stage 2.5 user coverage 85.5% (rest is genuine off-frame
time); partner/opponent role-awareness + opp L/R continuity still geometric
heuristics (KNOWN_ISSUES) — revisit if downstream opponent stats look off.

Parallel / larger efforts (tracked in KNOWN_ISSUES + a spawned task):
- **Inference speedup** (GPU decode) — required for the real ≥5-min workload.
- **Cross-court training diversity** — add more indoor/outdoor courts.

## Key facts / gotchas

- **Production inference is GPU-only.** Local CPU torch is ~11 s/frame (~23h for a
  2-min clip). Use `infer_v4.ipynb` on Colab. Claude can drive it via Chrome MCP
  once the extension is connected.
- **`data/` is gitignored.** `ball.parquet`, `ball.meta.json`, `video.mp4`,
  `frames_720/`, the `ball.val_300-419.*` backups — all local, regenerable, NOT
  committed. The 120-frame validation slice is preserved as
  `data/pb_2min/ball.val_300-419.parquet`.
- **Colab upload gotcha:** uploading the whole `data\pb_2min` folder into
  `MyDrive/pb_infer/pb_2min/` doubles the path → `.../pb_2min/pb_2min/video.mp4`.
  Either upload just `video.mp4`, or set `CLIP='pb_2min/pb_2min'` in the notebook.
- **Chrome downloads** land in `C:\Users\hochh\Dropbox\My PC (DESKTOP-94DNBCT)\Downloads`
  (symlinked from `~/Downloads`); Dropbox briefly renames in-flight files to
  `<guid>.tmp` then restores the real name.
- **The 512×288 trap** (still true): never let inference silently downscale 4K to
  512×288 — it reshrinks the ball to ~2px. v4 runs at 1280×720.

## Things to NOT touch

- Don't re-attempt ball-detection v1/v2/v3; failures well understood (KNOWN_ISSUES).
- v1/v2 weights on Drive retained for reference.

## Bring this to the next session

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, stages/finetune_ball_model/contract_v4.md
    before proposing anything.

    pb_2min has real ball (synthetic:false) run through Stages 1,2,2.5,3,5,5.5,6
    (court.json, players.parquet, track_roles.json, poses.parquet, shots.json,
    bounces.json, classified.json), each operator-validated. Next: Stage 7 (segment
    rallies) on the real ball, then 8-11. Follow the real-vs-synthetic adaptation
    pattern at the top of this handoff (is_user-from-roles, 4K/fps scaling,
    real-only filter gating, spot-check validation, numpy-bool gotcha). In Stage 7
    own the courtesy-feed exclusion; in Stage 8 build homography-projected
    court-plane ball speed (see KNOWN_ISSUES Stage 6 depth-speed). Then calibrate
    Stages 9/10. Also open: inference throughput (GPU decode), cross-court
    generalization, partner/opponent role-awareness, Stage 5 serve-flagging.

---

Generated at session end on June 14, 2026.

# Known Issues and Deferred Decisions

> **⚠ `SYSTEM_DESIGN.md` (repo root) is the authoritative whole-system accuracy +
> dependency ledger as of 2026-06-19** — read it first. It carries the per-stage
> accuracy, the blast-radius of each limitation, the fundamental-limits decisions,
> and the foundations-first roadmap. This file remains the detailed issue log.
> **New discipline (SYSTEM_DESIGN §0):** no deferral without recording its blast
> radius; a stage isn't done until it meets its downstream's accuracy on REAL data.
>
> **Recently addressed (2026-06-19):** far-side player court-position drift —
> Stage 3 now scopes pose by Stage-2.5 *role* (was a brittle `court_y.max()≤44`
> gate that deleted all opponents); far-side absolute position is accepted as
> zone-precision (camera-geometry limit, flagged `court_pos_reliable`). Opponent
> role classification — opponents are now identity-based `opp_a`/`opp_b` via
> appearance+continuity re-id (was geometric far-side-x at flat conf 0.5).

Issues observed during development that are not yet resolved, with notes on
when/where they should be addressed. Update as issues are resolved or as new
ones are discovered.

## Stage 2 - Adjacent-court contamination

**Observed:** May 2026, Stage 2 smoke test on `data/test_clip/`.

**Problem:** People playing on courts adjacent to the user's are detected by
YOLO and projected through the homography onto the user's court coordinate
system. When their projected positions happen to fall inside the
`0 <= court_x_ft <= 20, 0 <= court_y_ft <= 44` rectangle, they register as
`in_court=True` even though they are physically on a different court.

The doubles sanity check in Stage 2 correctly flags this: 11 tracks were
flagged in the smoke-test run. Inspection showed `court_y_ft` values up to
69 ft for some flagged tracks - clearly off-court.

**Why not fix in Stage 2:** Stage 2's job is detection and tracking. Filtering
which tracks to count toward stats is an adjudication decision that belongs
downstream, where shot attribution and player-role assignment happen.

**Where to fix:** Stage 4 or 5 (whichever stage first does shot attribution
or per-player stats). Likely filter: only count non-user tracks whose
`court_y_ft` stays within `0..44` for >= 95% of their lifetime, OR whose
court coordinates are physically plausible given the homography's pixel
density at that location.

## Stage 2 - Court switches cause user track loss

**Observed:** May 2026, Stage 2 smoke test on `data/test_clip/`.

**Problem:** When the user switches sides with their partner (a routine
event in doubles, happens many times per match), ByteTrack's ID gets
swapped between the two players who cross paths. The user's track ID is
then attached to the partner, the user becomes a non-user track, and
Stage 2 reports a gap requiring re-identification.

The contract's `click again to re-identify` mechanism works, but is not
viable as a UX for real matches: a 30-minute match could have 50+ side
switches, each requiring a click.

**Why not fix in Stage 2:** Stage 2's contract explicitly defers
re-identification to user clicks. Changing this would require additional
logic (visual appearance matching, location-based heuristics, etc.) that
expands Stage 2's scope.

**Where to fix:** A new dedicated stage between Stage 2 and downstream
consumers, or expanded Stage 2 logic. Options to consider:
- Visual appearance matching (compare jersey colors / clothing across IDs).
- Position-based heuristic (after a side-switch event, the user is the
  player closest to the previous user position on the *opposite* side
  of the net).
- Operator-confirmed re-identification at fewer key moments rather than
  per-frame click-fixing.

This needs design before implementation.

> **UPDATE (2026-06-13): substantially addressed for the USER by Stage 2.5
> appearance re-id.** Stage 2.5 now follows a person across ByteTrack ID
> swaps/gaps/side-switches by upper+lower-body clothing-color match + height
> (anchored on the user seed), not per-frame clicks — so a side-switch or a
> >4s gap no longer loses the user. Validated on pb_2min: after ByteTrack
> dropped the user's ID at frame 4868, the re-appearances (tids 1554, 1663,
> ~5.8s gap) are re-attached to the user, lifting user coverage 68% -> 85.5%
> (the remainder is genuine off-frame time). Commit `b348d98`. Still open: the
> same appearance matching is not yet extended to keep the two OPPONENTS
> continuity-tracked (opp L/R remains provisional), and partner gap-recovery
> relies on the same cue.

## Stage 3 - Scope filter is a heuristic, not the right architectural answer

**Observed:** May 2026, while drafting Stage 3 (pose).

**Problem:** Stage 2's 	ransient flag (lifetime < 30 frames OR no in-zone foot points) is too permissive as a filter for "real on-court players." On the test clip, 178 of 486 tracks were non-transient — far more than the ~4 actual players. The extras were people on adjacent courts whose homography projections occasionally landed inside the user's court rectangle, with track lifetimes well above 30 frames.

Running pose on every non-transient detection (~20,000 detections in a 2-minute clip) would have wasted the bulk of MediaPipe inferences on people who aren't on the user's court.

**Workaround in place:** Stage 3 applies a strict per-track scope filter on top of 	ransient:
- `in_court_frac >= 0.50`
- `court_y_ft.max() <= 44.0` (no adjacent-court contamination)
- `court_y_ft.min() >= -8.0` (no people behind the gym)
- `lifetime > 5 seconds`

Plus the user is always in scope unconditionally. This brings detections down to a manageable count and keeps real players in scope, including a player serving from behind the baseline or chasing a wide shot.

**Why this is a heuristic, not the right answer:** The scope filter is hard-coded in Stage 3. Stage 4 (ball tracking) does not need it. Stages 5+ may want to know about all real players for shot attribution. Each stage re-deriving this filter independently is duplication waiting to drift.

**The right answer (deferred):** A dedicated stage between Stage 2 and downstream consumers — call it Stage 2.5 or Stage 2b — that classifies each track in players.parquet into one of: `user`, `partner`, `opp_left`, `opp_right`, `noise`. Output is a small JSON file (`track_classification.json`) that downstream stages read instead of re-doing geometric heuristics. This is also where the court-switch ID-swap problem (already in this file) is most naturally addressed: `user` is a logical role across multiple ByteTrack IDs, not a single track ID.

Adding this stage would change ARCHITECTURE.md from 11 stages to 12. Worth doing, but should wait until we have at least one downstream consumer that proves the filter set we settle on. For now, Stage 3's hard-coded filter is the pragmatic option.

> **UPDATE (2026-06-13): Stage 3 now consumes `track_roles.json` for the USER.**
> The dedicated classification stage (2.5) exists, so the "right answer" is
> partially realized: when `track_roles.json` is present, Stage 3 takes `is_user`
> from the role `user` and poses every user-role track regardless of the
> geometric gate — fixing the case where a real user track was dropped (pb_2min
> tid 1663, `in_court_frac` 0.40 < 0.50, the user serving/retrieving behind the
> baseline). Commit `f349141`. Still a heuristic for **partner/opponents**, which
> remain on the geometric gate; extending role-awareness to them (and having
> Stages 5+ read roles instead of re-deriving filters) is the remaining work.

## Stage 3 - Single-person pose model picks wrong person when bboxes overlap

**Observed:** May 2026, Stage 3 smoke test on `data/test_clip/`.

**Problem:** MediaPipe Pose is a single-person model. When given a bbox crop that contains more than one person (a partner standing close, an opposing player on the far side of the net within the frame, an adjacent-court player visible behind the subject), the model picks one pose to return - and it is not always the person the YOLO bbox was drawn around. The returned landmarks are then mis-attributed to the wrong track_id.

**Workaround in place (May 2026):** Before running pose, the crop is masked - regions of all OTHER detections on the same frame are painted with a neutral grey rectangle, with the subject's own bbox region preserved. This forces MediaPipe to see only one person.

**Why this is a workaround, not the right answer:** Masking with a flat grey rectangle is unusual visual input and may slightly lower MediaPipe's pose detection rate on otherwise-good crops. The smoke test should compare the post-masking detection rate against the pre-masking rate (97.5%) to flag regressions. A more sophisticated approach would mask only the body region of the other person (using a person-segmentation model), not the entire bbox rectangle. Even better would be a multi-person pose model.

**Where to revisit:** If the masked detection rate drops below 90%, or if downstream stages report incorrect landmarks even with masking enabled, consider switching to a multi-person model (MediaPipe `num_poses > 1` plus pick-by-distance, or a different model altogether). Track this as a Stage 3 follow-up.

## Stage 4 - Dettor's pre-trained weights do not generalize to user footage

**Observed:** May 2026, Stage 4 first end-to-end run with Andrew Dettor's
pickleball-trained TrackNetV2 weights converted from his TF SavedModel.

**Problem:** Stage 4 ran end-to-end without exception against the
2-minute test_clip. All schema invariants validated. But the detection
rate on active-rally frames was 4.5%, far below the 80% threshold.
Diagnostic inspection on 4+ frames showed:

- The model produces near-uniform low-confidence output (heatmap
  values mostly in 0.001-0.005 range; p99.9 around 0.05-0.1).
- On lucky frames where an adjacent-court ball is well-lit, the model
  locks onto it (frame 250: peak value 0.48 on adjacent-court ball,
  not user's-court ball).
- On all other frames tested, the model's argmax landed on incidental
  bright/circular features: window glare, wall objects, court lines,
  player heads. Never on the user's-court ball.

**Verified mechanically correct:**
- Weight conversion sanity checks: all 5 passed (layer count, Conv/BN
  count match, per-layer shape parity, forward-pass not-NaN, output
  range plausible).
- BatchNorm-over-width adaptation working as designed (Dettor used
  axis=-1 BN on NCHW data — see `_tracknet_model.py` for details).
- Forward pass on dummy zero input returns the expected near-uniform
  sigmoid 0.5 output.

**Why generalization failed (likely causes):**
- Camera placement different. Dettor trained on PPA Tour broadcast
  footage (high boom, professional venues, stable lighting, 4K). User
  footage is amateur — corner-mounted phone at ~6 ft, indoor and
  outdoor venues with variable lighting and court colors.
- Dettor's training set was small (~1 PPA Tour match) and his own
  writeup acknowledged overfitting concerns.

**Path forward (Stage 4.5):** Fine-tune Dettor's weights on
user-labeled frames from the user's own videos. Stage 4.5 contract at
`stages/finetune_ball_model/contract.md` codifies this effort.

**Stage 4 itself is code-complete.** No code changes required to
Stage 4. When Stage 4.5 produces new weights, Stage 4's `--weights`
argument points at them; smoke test re-runs without other changes.

## Stage 4.5 - Ball detection PAUSED after three failed attempts

**Observed:** May 2026, across three distinct ball-detection approaches.

**Outcome:** All three attempts produced detectors that failed acceptance.
Stage 4.5 is currently paused. Downstream stages (Stage 5+) are being
built against a placeholder ball.parquet so the rest of the pipeline can
progress. Ball detection will be revisited when (a) better source video
is available from updated camera setups, and/or (b) algorithmic options
beyond per-frame detection are explored.

> **UPDATE (2026-06-02): UN-PAUSED → v4 in progress.** Both conditions are now
> met. New **4K/60fps outdoor** footage arrived, and a measured SNR probe
> (`tools/diag_ball_snr.py` on `data/pb_2min/`) confirmed the SNR wall is gone:
> ball median intensity **71/255**, local SNR **61×**, **~13px** blob, present
> in **88%** of mid-flight frames. The one remaining problem is temporal
> disambiguation from **~372 per-frame distractors** — exactly the "needs
> multi-frame trajectory info" conclusion below. v4 (temporal TrackNet + focal
> loss + **raised input resolution** + trajectory post-processing) is the
> approved approach: `stages/finetune_ball_model/contract_v4.md`. The
> resolution point is critical — Stage 4's old inference downscaled to 512×288,
> reshrinking the 4K ball to ~2px; v4 infers at 1280×720.

> **UPDATE (2026-06-11/12): v4 LANDED — real ball detection works.** Training
> finished (`data/models/ball_model_v4.pt`): **val recall 0.90 same-court, 0.54
> cross-court**, fp 0.02. Inference rewritten as
> `stages/track_ball/track_ball_v4.py` (720p + trajectory post-processing) and
> validated vs ground truth on pb_2min frames 300–420 — 39/40 balls, **median
> 4.9px at 4K**, 100% within 25px. The first **real full-clip `ball.parquet`**
> (`synthetic: false`) was produced for `data/pb_2min/` via the GPU notebook
> `stages/track_ball/infer_v4.ipynb` (7164 frames, detect_frac 0.676, coords
> in-bounds, conf mean 0.78). Two open items remain (new sections below): the
> **0.54 cross-court** gap and **inference throughput**. The synthetic caveat
> does NOT fully lift yet — Stages 5–11 must be re-run on the real ball first
> (pb_2min needs Stages 1–3 done first).

### Attempt 1 (v1, contract v0.1.0): Fine-tune Dettor's PPA TrackNetV2 weights

- Setup: Adam lr=1e-4, weighted BCE pos_weight=100, 10,701 user labels
  across 4 videos, T4 GPU.
- 6 epochs in 2.8h; best val loss at epoch 1.
- Validation: detection_rate_at_10px=0.32, false_positive_rate=1.00.
- Diagnostic showed the model memorized two fixed pixel locations (tree
  branches in the distant background) as 'always a ball', present as
  top-5 peaks on nearly every frame.
- Root cause: BCE with high pos_weight made 'confidently wrong' locally
  optimal; static camera + no spatial augmentation enabled positional
  memorization; Dettor's PPA-broadcast features anchored on the wrong
  visual prior.

### Attempt 2 (v2, contract v0.2.0): Train TrackNetV2 from scratch with MSE + spatial aug

- Setup: random init, MSE loss, Adam lr=1e-3 cosine decay, rotation
  +/-5deg, translation +/-10%, A100 GPU, batch_size=8.
- Aborted at epoch 25; training had collapsed by epoch 10 to the
  trivial 'predict zero everywhere' solution.
- Val loss flatlined at ~0.000078 (exactly matches the MSE of zero-
  prediction on a 99.97%-zero target heatmap).
- Root cause: MSE on sparse positive targets has a stable trivial
  minimum at 'predict nothing.' Symmetric to v1's failure mode.

### Attempt 3 (v3, contract v0.3.0): Classical CV with background subtraction

- Setup: median-background subtraction + connected-component filtering
  + per-blob scoring (motion + circularity + color). Tunable per-video
  via tune_ball_cv.py interactive calibration.
- Validation on test_clip: tune accuracy ~1% (1/100 frames) even after
  multiple rounds of threshold tuning and adding an isolated-blob
  filter to exclude held-ball labels from measurement.
- Diagnostic visualizations (tools/diag_fg_at_ball.py and the approval
  grid PNG) showed:
  - The ball IS faintly visible as foreground at the labeled position
    in most mid-flight frames (thresholding at 8-20 produces a small
    white blob at the click).
  - However, hundreds of other small foreground blobs (court line
    glints, player limb edges, fence shadows) survive the same
    thresholds, producing a signal-to-noise ratio too low for the
    per-frame scoring function to discriminate the ball.
  - 86% of supposedly-clean isolated-blob labels still failed to
    produce a measurable ball blob within 12 px of the click, due to
    centroid offset from motion-blur streaks and component merging
    with nearby foreground.
- Root cause: a 4-6 pixel ball in 1080p amateur phone footage at 6ft
  camera height with busy backgrounds is at or below the SNR floor
  for per-frame CV. Per-frame algorithms (CV or DL) cannot
  discriminate the ball from co-detected distractors without
  temporal trajectory information across multi-frame windows.

### Why all three approaches share a root cause

The fundamental issue is the *footage profile*, not the algorithm:
- Camera at ~6 ft (just above player heads, frequently sees the ball
  passing through head height where it blends with shirts/faces).
- Phone camera at 1080p; ball is 4-6 pixels in diameter.
- Backgrounds include trees, fences, light fixtures, windows -
  high-frequency content that looks ball-like at low resolution.
- Lighting varies across venues.

These characteristics violate the assumptions baked into both
broadcast-trained DL models (TrackNetV2 expects high-mounted 4K
broadcast feeds with simple backgrounds) and standard per-frame CV
(expects either high SNR or temporal coherence to disambiguate
candidates).

### Path forward (planned)

Two parallel efforts:

1. **Better source video.** Higher camera mount (10-15 ft if possible),
   4K and/or 60 fps recording, faster shutter to reduce motion blur,
   simpler backgrounds (avoid trees behind court), avoid adjacent
   courts in frame. Even partial improvements should raise the SNR
   substantially and may make the current v3 CV approach viable
   without algorithmic changes. New test footage should be labeled
   (~100 mid-flight frames is enough) and run through the existing
   v3 tooling to measure the SNR improvement.

2. **Pipeline development continues without ball detection.** Stage 5
   (shot detection) and later stages will be built against a synthetic
   placeholder ball.parquet (clean trajectories generated from known
   shot patterns). This lets the rest of the pipeline progress, exposes
   what downstream stages actually need from ball data (precision vs
   recall vs zone-accuracy), and creates real pressure to inform a
   future v4 ball-detection attempt.

### Artifacts retained but unused downstream

- v1: data/models/tracknet_v2_finetuned_v1.pt, validation_report.json,
  diag_v1/*.png.
- v2: train_log_v1.json on Drive (training was aborted before save).
- v3: stages/finetune_ball_model/_ball_cv_pipeline.py,
  tune_ball_cv.py, validate.py, tools/diag_fg_at_ball.py,
  tools/diag_heatmaps.py. All retained as documentation of what was
  tried; tune_ball_cv.py specifically may be re-run on improved
  footage without code changes.
- All TrackNet code in stages/track_ball/ remains in place pending
  Stage 4 rewrite; that rewrite is deferred until ball detection has
  a working v4.

### Generalizable lessons

1. When fine-tuning, verify the source model's training data matches
   your problem's characteristics. Dettor's PPA priors were actively
   misleading.
2. For sparse-positive heatmap detection, both raw weighted BCE and
   raw MSE have symmetric failure modes (confidently-wrong vs
   nothing-confident). Focal loss is the standard remedy if DL is
   the right tool.
3. Per-frame CV with background subtraction is appropriate for
   high-SNR scenarios (large object vs simple background) but fails
   when the object is small and the background is busy. Temporal
   trajectory tracking across multi-frame windows is the standard
   next step for low-SNR ball tracking.
4. Match the tool to the data, but ALSO match the data to the tool
   where possible. Improving source video is often higher-leverage
   than improving algorithms.
5. When repeated approaches fail with different mechanisms but the
   same outcome, the problem may be the data, not the technique.
   Step back and reassess inputs before assuming the next algorithm
   will work.

## Synthetic ball — Stages 5–9 consume PLACEHOLDER ball data

**Observed:** May 2026, ongoing. The *cause* was the Stage 4.5 pause above;
this section documents the *downstream consequence and workaround* that every
ball-consuming stage (5, 5.5, 6, 7, 8) inherits, because it's easy to forget
when reading those stages' outputs.

> **UPDATE (2026-06-12): a real `ball.parquet` now exists for `data/pb_2min/`
> but this caveat STILL APPLIES.** v4 landed (above), but Stages 5–11 have NOT
> yet been re-run on the real ball — they were last run on the synthetic
> placeholder. Until that re-run happens (and pb_2min first gets Stages 1–3:
> court.json / players.parquet / poses.parquet), every ball-derived output below
> remains synthetic-scaffold. The caveat lifts per-stage only as each is re-run
> and re-validated on the real (noisy, gappy) trajectory.
>
> **UPDATE (2026-06-14): lifting in progress.** pb_2min now has real Stages 1–3,
> and **Stages 5 (shots, `8aa9164`) and 5.5 (bounces, `740fac9`) have been re-run
> and operator-validated on the real ball** — caveat **lifted** for those two on
> real clips. Each needed real-ball adaptations (4K/fps scaling, is_user-from-
> roles, real-only filter gating, ground-contact refinement) and real-world
> phenomena the synthetic never had (ball-handling between points; arc apexes vs
> ground bounces). **Still synthetic-scaffold: Stages 6, 7, 8, 9, 10, 11** — re-run
> next, same per-stage approach. The stages still run on synthetic for their smoke
> tests (real-ball filters gated off so the synthetic bars hold).

**Problem:** Because real ball detection is paused, the pipeline runs against a
**synthetic placeholder `ball.parquet`** generated by `tools/synth_ball.py`
(clean, gap-free trajectories with impacts/bounces placed at real player
positions, flagged `synthetic: true` in `ball.meta.json`). Every metric and
label derived from the ball is therefore **placeholder, not measured**:

- **Stage 5** (detect shots) — shot frames/impacts.
- **Stage 5.5** (detect bounces) — bounce locations, in/out, at-feet.
- **Stage 6** (classify shots) — shot type, volley flag, speeds.
- **Stage 7** (segment rallies) — `end_reason`, serve-fault detection.
- **Stage 8** (compute metrics) — everything in `reliability.synthetic_gated`:
  `by_end_reason`, serve stats, shot mix, third-shot, bounce in/out, error
  attribution, ball-landing heatmap, and all per-player ball-derived stats.
- **Stage 9** (rate — USAPA) — the rating point estimate is ~0.70
  synthetic-weighted (error_control, shot_skill, serve, rally_consistency
  dimensions). Only net_play + movement (~0.30) are real. The rating is a
  SCAFFOLD until v4: validated for logical correctness, not accuracy, and on
  top of that its thresholds are uncalibrated (no rated-footage corpus).

**What is NOT affected (durable real value now):** anything derived from
`players.parquet` / `poses.parquet` / `track_roles.json` — i.e. Stage 8's
**position / court-area time fractions, court coverage, and the player-position
heatmaps**, plus rally length/duration (frame-counting, not ball physics).
Stage 8's `reliability` block names exactly which families are synthetic-gated
vs real; do not erase that block.

**Workaround in place:**
- `ball.meta.json` carries `synthetic: true`; each stage propagates it as
  `ball_source` and emits a loud `warnings[]` entry + WARNING log line. No
  stage silently trusts the ball.
- Acceptance bars in Stages 5–8 smoke tests are calibrated against the
  *synthetic* ground truth in `ball_synth_truth.json`, NOT against real
  footage. They prove the *logic* is correct given clean ball data; they do
  **not** prove real-world accuracy.
- Stage 8 specifically gates correctness on **reconciliation invariants**
  (counts sum correctly, `by_end_reason` matches Stage 7 exactly) rather than
  on ball-derived accuracy, precisely because the ball is synthetic.
- **Stage 8 Tier-B metrics are emitted as explicit `null` placeholders.** The
  `metrics.json.pending_real_ball` block lists four ball-derived metrics
  (`forced_vs_unforced_errors`, `dink_shot_tolerance`,
  `third_shot_drop_outcome`, `opponent_backhand_targeting`) with `value: null`,
  `status: "pending_real_ball"`, and a `description` of exactly what each will
  contain. They are deliberately NOT computed against the synthetic ball (a
  placeholder number would mislead more than a null). **When v4 lands:**
  implement each per its `description`, drop the null, and move its key from
  `reliability.pending` to `reliability.real_data` (or `synthetic_gated`→real)
  once validated. `forced_vs_unforced_errors` is the highest priority — it
  feeds the Stage 9 USAPA rating.

**Where/when to fix:** When real ball detection (v4) lands (see Stage 4.5 path
forward), regenerate `ball.parquet`, re-run Stages 5→5.5→6→7→8 on the real
(noisy, gappy) trajectories, and re-validate every stage. Expect the
synth-derived numbers to shift; the synthetic acceptance bars will need
real-data counterparts.

## Stage 8 — opponent left/right split inherits Stage 2.5 labeling imprecision

**Observed:** May 2026, while drafting Stage 8 (compute metrics).

**Problem:** Players change left/right position between serves (service-box
switches by score) and during rallies (partners rotate/poach). Stage 8's
per-player and position metrics are robust to this because they aggregate per
*role* (over track_ids, all frames) and attribute errors by *track_id* or by
*half* (near/far) — none of which depend on left/right. The one exception is
the `opp_left` vs `opp_right` split, which is inherited from Stage 2.5's
median-court-x assignment; under frequent opponent side-switching the two
opponent buckets blur. **Team-level (`team_far`) and combined-opponent numbers
are unaffected — only the split between the two opponents is imprecise.**

**Workaround in place:** Stage 8 trusts the Stage 2.5 roles as given (no
cross-stage re-classification) and surfaces opponent uncertainty via
`role_confidence` / `role_contaminated` flags + warnings.

**Where to fix:** Stage 2.5 v2 (far-side simultaneity/continuity + appearance
matching for opponents), already queued. Once opponents are continuity-tracked
like the near side, the L/R split tightens with no Stage 8 change.

## Stage 4 (v4) — inference throughput is CPU-decode-bound, too slow at scale

**Observed:** 2026-06-11/12, full-clip Colab run of `stages/track_ball/infer_v4.ipynb`
on a T4 (driven via the Claude-in-Chrome browser MCP).

**Problem:** The full pb_2min clip (7164 frames @ 3840×2160/60fps) ran at only
**~2.9 frames/sec** — about **40 minutes** for a 2-minute clip. The GPU is mostly
idle; the bottleneck is **single-threaded CPU video decode** (`cv2.VideoCapture.read`
of 4K frames + `cv2.resize` to 1280×720), not the TrackNet forward pass.

**Why it matters:** this is a per-player analysis app whose real workload is
**many videos, each ≥5 minutes** (longer = better feedback — see the product
requirements). At ~2.9 fps a 5-minute 4K/60 clip is **~100 minutes**, which does
not scale.

**Where to fix:** A background task was spawned to switch decode to GPU/hardware
(NVDEC via `decord`/PyAV) and/or a threaded prefetch reader so decode overlaps
the GPU forward pass — target ~5–10× speedup. Optionally pre-transcode clips to a
lower working resolution (note: outputs are in SOURCE-resolution pixel coords, so
that changes the coordinate scale and is a deliberate, not free, decision).
Acceptance: match the current detection output (validate against the pb_2min
[300,420] ground truth — 39/40 balls, median 4.9px). Regenerate the notebook via
`tools/build_infer_v4_nb.py`. (A separate, already-fixed gotcha: the notebook
hardcoded `BATCH=16`, which OOMs a 15GB T4 at 720×1280 — the builder now scales
BATCH to GPU memory: T4→4, >20GB→8, >32GB→16. Commit `1621541`.)

## Stage 4 (v4) — detector does not yet generalize across courts

**Observed:** 2026-06-11, v4 training validation (cross-court holdout).

**Problem:** The v4 detector is **0.90 recall on the training court but 0.54 on a
held-out cross-court test** (fp 0.02 both). It learned the training venue well,
not pickleball-ball-in-general.

**Why it matters:** the app must analyze footage from **different indoor AND
outdoor courts** (product requirement). A detector that only works on the
training court can't be relied on across the venues real users will film. This is
**required**, not optional polish.

**Where to fix:** Extend training-set diversity — add more indoor and outdoor
courts (the v4 contract already anticipates a cheap ~200-label warm-start
fine-tune per new venue, NOT a from-scratch retrain). Re-measure with a
whole-clip cross-court holdout each time. Track recall per venue type. Until the
gap closes, treat real-ball results on any not-yet-trained court as provisional.
See `stages/finetune_ball_model/contract_v4.md`.

## Stage 6 — shot type confused by depth/height (pixel-speed limitation)

**Observed:** 2026-06-15, real-ball validation of Stage 6 on pb_2min (operator
spot-check).

**Problem:** Shot **type** leans on `post_speed_ftps`, computed from the ball's
*pixel*-speed × a planar pixels-per-foot scalar at the contact point. A ball
moving in **depth** (a drive hit straight down-court) or at **height** covers few
pixels per frame, so its real speed is badly underestimated. On pb_2min f3541, a
true **drive** measured **4.2 px/f** and was indistinguishable from a slow
**drop** — the drop one shot later (f3740) actually measured *faster* (19 px/f)
because it moved laterally. Speed thresholds can't separate readings that are
backwards. (Volley, type-by-arc, and fast lateral shots all validated correctly;
this affects slow/depth groundstrokes only.)

**Mitigation in place (v0.3.0):** a **tweener arc-shape tiebreak** (16–25 ft/s →
flat=drive, lofted=drop) drains the old "unknown" dead-zone and fixes the cases
where speed lands in that band. A depth-drive reading *below* dink speed still
mistypes as a drop.

**Where to fix:** **homography-projected court-plane ball speed** — project the
ball pixel → court feet per frame (via `court.json` `image_to_court`) and measure
displacement in feet, which handles depth (residual bias = ball height above the
plane). This is also the right speed signal for **Stage 8 metrics**, so do it
once, there or as a shared helper, rather than patching Stage 6. Full fix = ball
height / 3D tracking. Deferred until ball speed materially drives a metric.

## Stage 6 — serve labeling & courtesy feeds are upstream/downstream concerns

**Observed:** 2026-06-15, same validation.

**Problem (serve):** if Stage 5 misses a serve (`is_serve` not set), Stage 6
classifies it by features (e.g. "drive"/"lob") and can never say "serve" — seen
on pb_2min f3470. **Fix belongs in Stage 5** (serve detection), not Stage 6.

**Problem (courtesy feed):** a between-points feed (opponent hands the ball over
before a serve) has no bounce, so `is_volley=true` — literally correct but it's
not a rally shot and would skew rally/volley stats (f3148). **Fix belongs in
Stage 7** (rally segmentation), which should scope stats to actual rallies and
exclude pre-serve feeds. Flagged here so the downstream stages own it.

## Stage 4 — adjacent-court ball contamination (single-ball assumption)

**Observed:** 2026-06-16, operator review of pb_2min via Stage 6/7 overlays.

**Problem:** On a **multi-court venue** the single-ball detector locks onto a
**neighbouring court's ball** when ours is occluded/absent. Those detections
become phantom shots/serves/rallies (e.g. a "serve" before the point starts; a
"lob" from the court behind the far baseline, which overlaps our airborne-ball
image zone and is NOT separable by position).

**Mitigation in place (Stage 5 v0.3.0):** trajectory-coherence gates reject the
phantoms at the shot level — a serve must launch a *sustained* run, and an
impulse impact's ball run must not *teleport in*. This removed the operator-flagged
phantoms on pb_2min without touching real shots.

**Root cause / proper fix:** the **ball detector** (Stage 4) is single-ball and
court-blind. A court-aware or multi-ball-disambiguating detector (track our ball
as the trajectory continuous with our players' play) would fix it at the source
and also help recall. Until then the Stage 5 gates are the safety net.

## Stage 4 — ball-detection recall is the dominant downstream limiter

**Observed:** 2026-06-16, foundation review.

**Problem:** On pb_2min the ball is detected (`visible|interpolated`) in only
**~62% of frames**. Some is genuine occlusion (ball behind a player, motion blur,
off-frame) that no detector recovers; some is detector miss. This **cascades**:
missed ball at an impact → **missed shot** (rally looks shorter, sides incomplete);
missed bounce → **`unknown` rally end_reason** + missed volleys; missed serve
launch → **serve under-detection**. The rally *boundaries* are now robust to this
(Stage 7 uses the ball-out-of-play signal), but shot/bounce/serve **completeness**
is capped by detector recall.

**Where to fix:** improve **Stage 4 v4 recall** (retrain with more data + the
cross-court diversity already tracked above; consider longer-gap trajectory
interpolation). This is the highest-leverage foundation investment — it improves
shots, bounces, serves, and end_reason at once. Forcing detections out of gaps in
Stage 5 instead is rejected: it reintroduces the contamination above.

## Stages 5/7 — airborne ball-contact projection is unusable (resolved by hitter_side)

**Observed:** 2026-06-15/16.

**Problem:** A shot's `impact_court_xy_ft` (ball-contact pixel → court via the
ground homography) is **physically meaningless**: the contact is airborne, and an
elevated point projects toward the horizon (observed court_y up to ~1900 ft on a
44-ft court). Any side/zone/in-out logic built on it is noise.

**Resolved:** Stage 5 v0.3.0 emits `hitter_court_xy_ft`/`hitter_side` from the
hitting **player's ground position**; Stage 7 uses `hitter_side`. `impact_court_xy_ft`
is retained for debugging only. (Ground-truth **bounce** positions remain valid —
bounces are on the ground, so their court projection is sound.) A true ball court
position (for shot speed → Stage 6 types, Stage 8 metrics) still needs ball
height / 3D — see the Stage 6 depth-speed entry.

## Confidence propagation (Foundation #3) — the two capture-side levers it exposes

**Observed:** 2026-06-21, while designing Stage 8 confidence propagation
(SYSTEM_DESIGN.md §6 #3, C9). The confidence model decomposes every metric's
reliability into `base × penalty(n)` — per-event measurement quality × a
sample-size term — and tags each metric with a `limited_by` reason
(`sample_size` / `measurement` / `known_limit`) so the report tells the user the
*right* remedy. Two of those remedies are **future capture/throughput enhancements,
not in scope now**, recorded here so they aren't lost:

1. **Processing-speed enhancement is a prerequisite for the `sample_size` lever.**
   When a metric is `limited_by: sample_size`, the honest user-facing remedy is
   "capture more rallies" — either a longer video or **multiple cumulative clips**
   (see next entry). But the real product workload is already **many videos, each
   ≥5 min**, and Stage 4 inference is **CPU-decode-bound at ~2.9 fps** (the
   throughput issue above, C8). So the confidence model's headline advice
   ("record more") is only *usable at scale once throughput is fixed*. Longer /
   more clips → more rallies → higher sample-size confidence is the lever; app
   processing speed is the gate on actually pulling it. Future enhancement; ties to
   F5 (GPU/NVDEC decode) + the throughput entry above.

2. **A higher-mounted or second camera is the only real fix for the `measurement`
   / `known_limit` lever.** When a metric is `limited_by: measurement` (depth-
   corrupted shot speed, ambiguous shot type) or `known_limit` (`mean_post_speed_ftps`,
   stamped low via `SPEED_CONF`), **more footage does NOT help** — you just get a
   more stable estimate of a fuzzy/biased number. The reliability ceiling there is
   set by single-camera 2D having **no ball height** (C2 / §5). The future capture-
   side enhancement that raises that ceiling is a **higher camera mount and/or a
   second camera** (enabling depth / parabola-z / true 3D ball speed). Until then
   the report must say "limited by single-camera video, not by how much you record"
   — never imply more clips will sharpen speed. Future enhancement; ties to
   SYSTEM_DESIGN §5 (Ball height/3D, option (b) add-capture) + F8.

**Honesty banner (orthogonal to both):** the confidence model is **blind to
recall** — a missed (motion-blurred) fast shot leaves no record to attach low
confidence to, so `n` is *detected*-n, not *true*-n. Neither lever above is
visible in any per-metric confidence; the recall undercount is surfaced as a
standing caveat (shot counts / rally length are a **lower bound**), not folded
into a number. The fast-ball recall fix is itself partly capture-side (faster
shutter / higher frame rate — F2).

## Cumulative multi-clip stats — can pooling raise confidence?

**Observed:** 2026-06-21, operator question during Foundation #3 design.

**Question:** can multiple video clips be combined so that stats which depend on
the *number of rallies* become more reliable?

**Answer — yes, for the sample-size half only, with conditions.** Pooling rallies
across clips grows `n`, which raises `penalty(n)` and therefore the confidence of
**count/rate/average metrics** (rally length, rally duration, shot-mix rates). It
is the *same lever* as recording one longer video — more events, more statistical
stability. **Caveats that bound it:**
- **Only sample-size-limited metrics improve.** Measurement-limited stats (shot
  speed/type — `limited_by: measurement`/`known_limit`) do **not** get more
  accurate from pooling; you get a more stable estimate of the same fuzzy number.
- **Per-player pooling needs cross-video identity.** Stage 2.5 roles are
  per-clip; pooling `user`/`opp_a`/… stats across clips requires matching the
  same logical players across videos (**F28 cross-video identity/trend tracking** —
  feasible, not built). Match-level pooling (rally lengths) is easier than
  per-player pooling.
- **Conditions must be comparable.** Position/heatmap pooling needs the same court
  + camera calibration; rate metrics tolerate venue differences better.
- **Recall bias persists.** Pooling clips that all share the same fast-ball miss
  rate gives a more stable estimate of a biased number — confidence rises, the
  undercount does not shrink (see honesty banner above).
- **Semantic shift.** Pooling answers "this player *across sessions*" (typical
  behavior / trend), not "this single match." That's a feature for trend tracking
  (F28) but a caveat if a single-session readout was intended.
- **Throughput-gated** at scale, like lever #1 above.

## Stage 8 — net-play / court-zone metric is systematically WRONG (2026-07-07)

**Observed:** 2026-07-07, operator viewing the first rendered consumer report of
pb_2min. The net-play dimension's drivers read **`user_kitchen_time_frac` 0.054
(~5%)** and **`both_at_kitchen_frac` 0.0033 (~0.3%)** — while the operator watched
both partners live at the kitchen line for much of the match. The position→zone
mapping (which court positions count as kitchen / transition / baseline) is
systematically off.

**Why it matters (the big one):** net_play is stamped **confidence 0.998 ("high")**
because it rests on real position data with a large sample — but it is **wrong**.
Confidence measures noise + sample size, **not correctness**, so a systematic bug
renders as *confidently wrong*. Worse, the Stage 9 v0.3.0 confidence-weighting
**leans the rating toward** high-confidence dims → it leans on this wrong number.
This is the core lesson: **confidence ≠ correctness; only operator-eyes-on-rendered-
output catches it** (smoke tests didn't). See `feedback_consumer_output_validation`.

**Where to fix:** inspect `court_zones.json` kitchen/transition/baseline polygons +
how Stage 8 maps player foot positions to zones (and whether far-side drift or a
polygon/threshold error is the cause). FIRST item in the fix program (it drags the
rating). Re-validate against the rendered report.

> **RESOLVED (2026-07-09, Stage 8 v0.3.0, commit pending).** The zone *mapping*
> (`zone_from_court_y`) was NOT the bug — it correctly maps the marked kitchen line
> (court_y≈15) to "kitchen", and when a player genuinely stands at the line the
> pipeline reads it right. The real root cause: **court position was taken from the
> bounding-box bottom = the BACK foot.** For a net-facing near player with a
> staggered stance (step the back foot back to dig out a low ball), the back foot
> sits several feet behind where the player is playing, so a kitchen-line player
> was mis-classified as transition. Operator's rule: judge position by the FRONT
> foot ("front foot within ~2 ft of the kitchen line = at the kitchen"). Fix: Stage
> 8 now derives court position from the **net-most ankle** (`poses.parquet`,
> projected via `court.json` `image_to_court`), bbox-foot fallback per frame. The
> 2-ft tolerance is already the buffer in `KITCHEN_MAX_DIST_FT` (9 = 7 NVZ + 2).
> **pb_2min result:** user kitchen 5.4%→**26.2%**, partner 33%→**50.4%**,
> both-at-kitchen (near) 0.3%→**22.6%**; opponents unchanged (far-side bbox bottom
> already coincides with the front foot — no regression). Operator-validated on the
> rendered frame-532 overlay (front-foot at the line vs back-foot in transition).
> **Two follow-ups noted, not yet done:** (1) the metric averages over the WHOLE
> clip incl. ~42% dead-time (between-points baseline standing) — rally-scoping would
> further sharpen it (rally-only lifts user kitchen to ~33%); (2) confirm which near
> player is the user vs partner (Stage 2.5 role stability) if the split still looks
> off to the operator.
>
> **Operator review 2026-07-09 — front-foot calls confirmed correct on all 6 rally
> snapshots. Two review notes:**
> - **Between-point frames dilute the metric (follow-up #1, operator-confirmed).**
>   A snapshot inside rally 1's window (f1650) is visibly a *between-points* moment
>   (player reset to baseline) — it is classified correctly but should NOT be counted
>   toward zone-time when computed over the full clip. Fix = **rally-scope the Stage 8
>   position metrics** to frames inside (clean) rally windows. Sequenced AFTER the
>   Stage 7 rally over-segmentation fix, because rally-scoping must use correct
>   boundaries (else the spurious micro-rallies 6/7 = between-point net-gathering
>   would still be counted as play).
> - **Near-side user↔partner role gap under-counts user kitchen time (follow-up #2,
>   = Stage 2.5).** At some frames BOTH near tracks resolve to a single role (pb_2min
>   f6420: both labeled `partner`, user unidentified), so wherever the user is
>   temporarily mislabeled their kitchen frames are attributed to partner (or
>   dropped). Consequence: the user's 26.2% kitchen is a slight UNDER-count; the
>   near-team aggregate (both-at-kitchen) is unaffected. Root cause is Stage 2.5
>   near-side continuity (same appearance re-id that handles the USER's ID swaps is
>   not yet keeping the user/partner *split* stable frame-by-frame — cf. the "Court
>   switches cause user track loss" entry above). Deferred to a Stage 2.5 pass.

## Stage 7 — rally over-segmentation (micro-rallies) (2026-07-07)

**Observed:** 2026-07-07, same consumer-report review. Stage 7 segmented **8 rallies**
on pb_2min; the operator counts **6**. Rallies 0–5 are real (5.5–19.3s); rallies **6
and 7 are 0.8s and 1.1s** (2 shots each, 1.9s apart) — spurious micro-splits from the
ball-out-of-play splitter.

**Where to fix:** a minimum-rally filter in Stage 7 (min duration and/or min shots —
a real rally isn't 0.8s). Easy. Re-validate count against the operator's eye.

## Rating — dimensions do not match the official USAPA standard (2026-07-07)

**Observed:** 2026-07-07. Stage 9 rates 6 homegrown dimensions (net_play, movement,
error_control, shot_skill, serve, rally_consistency). The **official USA Pickleball
framework uses 7 categories** — forehand, backhand, serve/return, dink, third shot,
volley, strategy — with published per-level criteria. The homegrown dims are not the
standard, so the rating lacks legitimacy.

**Where to fix:** rewrite Stage 9 to USAPA's 7 categories, scoring each from the
metrics available and confidence-gating the not-yet-measured ones. The full
criteria→metric alignment (most metrics still planned = the legitimacy gap) and the
build program are in `docs/PRODUCT_VISION.md`. Body mechanics is NOT a USAPA category
(footwork lives inside Strategy) — kept as a planned supporting pose layer.

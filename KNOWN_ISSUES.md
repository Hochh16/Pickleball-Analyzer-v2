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

## Tooling - LABEL FRAME INDICES were silently corrupted by per-frame seeking (FIXED, 2026-08-11)

**Status: FIXED (a98158b) and all affected labels recovered (f99ad9e, 2ebb45b, 3a20c72).
Recorded because the failure mode is invisible and will recur if the pattern returns.**

**The bug.** `tools/label_ball.py` displayed a frame with
`cap.set(cv2.CAP_PROP_POS_FRAMES, idx); cap.read()`. On long-GOP H.264 that returns a
frame NEAR `idx`, not `idx`. The operator clicked an accurate ball position and it was
filed against a frame that was never on screen — by up to **12 frames**.

**Why it was hard to see.** The label file looks perfect: plausible positions, smooth
trajectories, correct counts. Nothing in the labels, the manifest, or the smoke tests can
detect it. The only symptom is downstream and easy to misread — a model that will not
learn a venue. Run 3 reported `indoor_seen_rec 0.035` and was nearly written up as "the
model cannot generalise indoors". It was being trained to find the ball where it wasn't.

**Blast radius.** 3 of 8 clips in the training bundle: `indoor_B1_3min` (92% of labels
wrong), `indoor_C1_3min` (85%), and `pb_3min_indoor` (97%) — the last being the
**held-out unseen-venue val clip**, so the 0.03 recall attributed to it is not a valid
measurement and must be re-taken. The other five clips measure clean. Drift is NOT
predictable from resolution or source: four clean clips were labelled from raw 4K
Dropbox files exactly like the corrupted ones. **Measure per file.**

**The rule.** Never seek per frame for anything whose frame index is recorded. Read
sequentially (see the `VideoReader` docstring in `label_ball.py`). A frame cache built by
sequential decode is trustworthy; a seek is not.

**The standing check.** `python -m tools.audit_labels data/<clip>` scores labels against
a detector that never saw them. A healthy clip reads near **1.5px median / 97% within
6px / 0% confidently-wrong** (pb_2min). Run it on any newly labelled clip BEFORE training.

**Its one blind spot, learned the hard way:** the audit cannot judge a clip the detector
is blind on. `pb_3min_indoor` read 241px median with **0% confidently-wrong** — never
confident anywhere. A blind model and a corrupt label are indistinguishable there. Use
`tools/measure_seek_drift.py`, which never looks for the ball, to settle those.

**Not yet measured:** `indoor_b`, `indoor_c`, `outdoor`, `test_clip` were labelled with
the buggy tool but have no `frames_720` cache, so drift could not be measured. The first
two are superseded 1080p/30 and unused; `test_clip` backs the Stage 2.5 smoke test — if
that test ever behaves oddly around ball timing, suspect this first.

## Stage 7 - RALLY END is undetectable; between-point balls are counted (ACCEPTED, 2026-08-03)

**Status: ACCEPTED LIMITATION, operator decision 2026-08-03.** Deliberately parked so
the rest of the work can proceed. Come back to it — it is not solved, only quantified.

**The problem.** Shot counts include between-point balls (feeds, balls rolled back at the
net, ball handling, a ball picked up). On the operator-labelled sample these are **~30% of
detected "shots"** in one block and **50%** in a deliberately enriched block. They inflate
shots, dinks, volleys and serves simultaneously.

**Why it is not fixed.** Removing them requires knowing where a rally ENDS. The operator's
definition is correct and complete — a rally ends on **two bounces**, a **first bounce
outside the court**, or a **net hit** (plus the serve exception: a serve must land in the
diagonal service box, else the rally is just the bad serve). Every route to detecting that
has been measured and failed:

| route | result | why |
|---|---|---|
| **net hit** | **no detector exists** | Deferred July 2026 as camera-limited. Operator: *"a very common way of a rally ending"* — so this alone caps any rally-end approach |
| **double bounce** | 1 physically plausible candidate in the whole clip | A bounce pair with "no shot between" usually means we MISSED the shot between them. Gated on shot recall (71%), not on bounce tuning |
| **out-bounce** | 29 candidates for 14 rallies | Too noisy to select from |
| **ball-comes-to-rest** | reversed (junk travels MORE) | Rolling/fed balls keep moving |
| **cross-net "in-play" test** | precision 36% / recall 56% | Kills 9 of 21 real shots; a ball rolled at the net DOES cross the net |
| **player FORMATION window** | precision 44% / **recall 20%** | Feeds happen right after the point ends, while players are still at the kitchen — BEFORE anyone walks back to serve position. The window opens too late |

**Per-shot classification is also dead** (see "BETWEEN-POINT BALLS" in ACCURACY_LEDGER):
**7 of 11 feeds are struck WITH A PADDLE**, so they are physically real paddle shots — only
GAME CONTEXT makes them not count. Hand-vs-paddle and receiver-swing signals both overlap;
a real block/reset/dink is itself a non-swing shot.

**RETRACTED 2026-08-18 — this filter is inert and the "solved" claim below was wrong.**
It formerly read: *"ground-level between-point balls ARE removed by the Stage 5 v0.4.0
ground-ball filter — 6 of 9 on the labelled block with zero real shots lost … what remains
is the AIRBORNE, paddle-struck subset."* Measured against the operator's full per-shot
review of pb_5_minute_outdoor-7 (34 labelled false positives), `grounded_fraction` scores
them at a **median 0.25 — identical to the median 0.25 of the 91 real shots** — and rejects
**2 of 34**, one of those by luck. There is no gap left to threshold.

The cause is a lesson, not a bug: the filter was tuned on 2026-08-03 against the OLD ball
track, and its discriminating power was a property of that track's failure modes. Replacing
it with TrackNet removed the signal. Because the issue was marked solved and
not-to-be-re-litigated, nothing re-ran the measurement for two weeks. **A filter tuned
against one detector does not transfer to another; any filter marked solved needs a standing
score or it is only assumed to work.** That standing score now exists —
`python -m tools.score_shots data/pb_5_minute_outdoor-7`. Run it before believing any claim
in this section.

**Blast radius (SYSTEM_DESIGN §0 rule 2):** shot / dink / volley / serve counts run high by
the number of between-point balls; every rate computed over them inherits it. The USAPA
rating is per-user, so it is affected wherever the user is the one feeding.

**Where to resume.** The two candidates not yet exhausted are (a) a **net-hit detector**,
which would unlock the single most common ending and is the highest-value missing piece,
and (b) improving **shot recall** past 71%, which is what makes double-bounce detection
physical (measured: plausible double bounces 1 -> 6 when far-side players are retained).
Do NOT retry the six routes in the table above.

## Stage 5 - Two serves missed at 1:04 and 1:33 (OPEN lead: adjacent-court players)

**Observed:** 2026-08-03, after the Stage 2.5 play envelope landed. Serve detection is
**12/14** against operator truth. The two failures have DIFFERENT causes:

| serve | what happened | cause |
|---|---|---|
| **1:04** | the shot IS detected (1:01.5, partner, 25.3 ft from net, 6.5 s gap — it passes both the depth and gap tests) but the serve rule rejects it | serve LOGIC |
| **1:33** | **no shot detected at all** within 2.5 s | shot DETECTION |

**Operator observation (the lead):** *"both are very clear serves by my partner and then
myself. And all players are positioned exactly as the rule states. There are people
playing on the court BEHIND the opponents so perhaps that is the cause."*

That is a plausible and specific hypothesis, and it is consistent with independent
evidence from the same session: **3 of 11 labelled between-point balls were adjacent-court
balls** (*"a ball being picked up and hit with paddle by a player from the NEXT COURT —
their ball rolled onto our court"*, *"looks like shot on other court"*). So adjacent-court
activity is demonstrably reaching our data. Two mechanisms could produce these misses:
- the **ball tracker** locks onto the neighbouring court's ball at that moment (would
  explain 1:33, where no shot is detected at all);
- extra far-side **tracks** perturb role assignment or the deep-player count.

**Operator decision: move on.** Two cases is too thin to tune against — that is how the
in-play filter got built and later killed. Re-examine when more videos are analysed and
the pattern either repeats or does not.

**Also open, larger than the 2 misses:** **5 FALSE serves** (0:13, 0:55, 1:35, 2:24, 4:27)
— detected serve count is 17 vs truth 14. **0:55 and 2:24 are balls the operator already
labelled as FEEDS**, so 2 of the 5 are the accepted between-point limitation surfacing in
the serve count, not a separate serve bug.

## Stage 5 - A FAULTED serve is indistinguishable from a FEED (operator-accepted, deferred)

**Observed:** 2026-08-01, `pb_5_minute_outdoor-2`, while building the in-play /
between-point separation.

**Problem:** Balls sent back to the server ("feeds") must be excluded — operator: a
ball thrown back to the next server "is Not a serve or a shot", and separating balls
in play from returning-the-ball is a stated product requirement. The working rule is
**IN PLAY = part of a cross-net exchange** (an opposite-side shot within ~1.5 s); a
feed is isolated, with dead time either side. That rule is right for feeds and gets
shots to 99 vs the operator's 98.

**But a FAULTED serve is also isolated** — nobody returns it, by definition — so the
same rule drops it. Per the operator (2026-08-01): **a faulted serve counts as a point
and a serve, and there is NO re-serve** (unlike tennis). So a fault should be kept and
a feed dropped, and the pipeline currently cannot tell them apart:

| | struck from | returned? | followed by |
|---|---|---|---|
| faulted serve | behind the baseline | no | the next point's serve |
| feed | often behind the baseline | no | the next serve |

**Signals tested and rejected:** ball speed (the known feed reads 3.0 px/frame — so do
the real serves at 1:16, 3:06 and 4:04; no separation, consistent with the established
finding that ball speed is unreliable on this camera) and timing/isolation (identical
by construction).

**Operator decision (2026-08-01): DEFER.** There are **no faulted serves in this clip**
to test against, it is uncommon, and the genuinely hard case is narrower still — only
when the **feed is struck from behind the baseline**. A feed from anywhere else is
separable on position. Other work is higher priority.

**Where to fix, when it matters:** the most promising untested signal is direction
relative to who serves next (a feed travels TO the player about to serve), or the fact
that a feed is often *thrown/rolled* rather than paddle-struck (operator's description
of the 0:55 ball), which pose or a ground-contact trace might expose. Needs a clip that
actually contains a faulted serve.

## Stage 5 - Pre-serve ball handling detected instead of the serve

**Observed:** 2026-08-01, from the operator's corrected serve timestamps.

**Problem:** The operator corrected two serve times — the serve listed at 1:29 is really
at **1:33**, and the one at 4:58 is really at **5:01**. In both cases the pipeline
detects a shot ~3.5 s EARLIER (1:29.5 and 4:58.1, both by the user from behind the near
baseline) and detects **nothing at the actual serve**. The pattern is the player
bouncing/handling the ball before serving: the handling is caught, the serve is missed.

**Why it matters:** the handling ball gets flagged as the serve, so the rally is
anchored ~3.5 s early, and the real serve is absent from the shot set.
`reject_same_track_repeats` exists for exactly this ball-handling case but only collapses
repeats within `rally_gap_frames`; at ~3.5 s apart these fall outside it, and the real
serve is not detected at all so there is nothing to collapse *to*.

**Where to fix:** Stage 5 — either widen the same-track handling window when the later
shot is a serve candidate, or fix the underlying miss (the serve strike itself is not
detected, which is ball-recall-limited).

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

## App/pipeline — local CPU can't process real clips; Stages 2 & 3 must move to GPU/Colab (2026-07-14)

**Observed:** 2026-07-13/14, first real end-to-end run through the Phase-2 setup-UI
runner on a **5-minute 4K/60fps outdoor clip** (`PB 5 minute outdoor.mp4`, 18,862
frames), on a machine with **no CUDA GPU** (`torch.cuda`=False).

**Problem:** **Stage 2 (YOLO player tracking) ran at ~1.05 frames/sec on CPU** —
measured from the live log (frame 0 → frame 10,300 in ~161 min). Extrapolated:
- **Stage 2 alone ≈ ~5 hours** for the full clip; **Stage 3 (MediaPipe pose)** is a
  similar order on top → **~8–12 h just for tracking + pose**, before the ball step.
- An outdoor multi-court venue makes it worse (~8 person-detections/frame → heavier
  YOLO+ByteTrack + more pose crops).

A **20-second trimmed clip** (1,200 frames) confirmed the pipeline runs end-to-end
correctly through the app: Stages 2 → 2.5 → 3 finished in **~17 min** once the CPU was
free (≈2.2 fps with nothing competing), producing valid `players.parquet`
(8,297 rows / 56 tracks), `track_roles.json` (all 4 roles), `poses.parquet` (4,637
rows), then paused at the ball hand-off as designed. So the **logic is correct; the
wall is purely CPU throughput** on real-length clips.

**Blast radius:** the setup UI's "run locally" path (UI_PLAN Phase 2) is **not viable
for real clips** on a CPU machine — a user would wait many hours for Stages 2/3. This
is the dominant practicality blocker for the app, alongside the Stage-4 GPU need.

**Where to fix (next work item):** **move the heavy vision stages — Stage 2
(track_players) and Stage 3 (pose) — to GPU/Colab**, exactly like Stage 4 already is.
The app would offload 2/3/4 to the GPU step (one Colab pass, or a cloud GPU later) and
keep only the **light analytical stages (5→11 + report) local** (those run in seconds,
proven on real data). Both YOLO (ultralytics) and MediaPipe support GPU; on Colab's GPU
Stage 2 would go from ~1 fps to real-time-ish. This also pairs with the Stage-4
GPU-decode throughput item below (C8) and the cloud-GPU direction in UI_PLAN. Until
then: the app is only practical on **short clips** locally, or needs a local CUDA GPU.

**Interim mitigations (partial, not the real fix):** track at reduced frame rate
(players don't need 60 fps), a smaller YOLO model (`yolo11n`), and/or GPU-decode —
each helps ~2–4× but doesn't close the gap and carries accuracy trade-offs to validate.

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

> **TESTED + REJECTED (2026-07-11) — the "residual bias" is fatal, not residual.**
> When the USAPA-realign made pace the blocker for 4 categories, the court-plane
> approach was validated empirically on pb_2min before building: project each
> shot's post-contact ball pixels through `image_to_court`, measure ft/frame ×
> fps. Result = **physically impossible garbage**: a *drop* read 157 ft/s, drives
> 362 ft/s, one shot 5626 ft/s (a hard drive is ~40–60 ft/s), with the airborne
> ball projecting to `court_y` up to **902 ft** on a 44-ft court. Cause: a ball at
> even 2–3 ft of height moving horizontally has its ground-intersection race toward
> the horizon, so real motion becomes hugely amplified projected distance. The
> ground homography **cannot place an airborne ball** — the same wall as the
> unusable airborne contact-projection (Stages 5/7 entry). **So court-plane speed
> is NOT a doable-now helper; real pace requires ball HEIGHT** (parabola-z / 3D
> reconstruction = F8, the z-recovery spike — a research effort, partly gated on
> ball-detection recall). Pace deferred; do NOT re-attempt the naive 2D→ground
> projection. Stroke-side FH/BH (F16, pose-based, no height needed) was done next
> instead.

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

## Stage 5.5 — bounce recall is ~50% (undercounts landings, caps depth/end_reason) (2026-07-11)

**Observed:** 2026-07-11, building the consumer report — the operator noticed the
counts don't reconcile. A clean way to see it: **every groundstroke is hit right
after a bounce**, so bounces should roughly equal the number of non-volley shots.
On pb_2min there are **39 shots − 9 volleys = 30 groundstrokes**, so we'd expect
**~30 bounces** (plus a few rally-ending ones) — but only **15 bounces are
detected** (~50% recall). The shot count makes the miss legible where the raw
bounce list doesn't.

**Blast radius:** the ball-landing diagram in the report is sparse (13 in-rally
dots vs the ~30 real bounces); rally `end_reason` is mostly `unknown` (no
rally-ending bounce detected → Stage 7 depth-speed entry); the `● depth/landing`
and `dink-rally-length` metrics that feed Third Shot / Dink stay `partial`/`○`;
in/out and net-or-short attribution are thin. This is the **same root cause** as
the Stage 4 ball-detection-recall limiter above (a missed ball at ground contact =
a missed bounce), surfaced as its own entry because it's the specific gate on the
report's landing map + the depth/landing metrics for the USAPA ADD step.

**Where to fix:** upstream ball-detection recall (Stage 4 v4 retrain — same lever
as above), plus possibly a more permissive Stage 5.5 bounce detector once recall
improves (the current precision-tuned detector deliberately under-detects to avoid
false bounces on the noisy ball). Do NOT loosen Stage 5.5 thresholds before the
detector improves — it would trade the recall gap for false landings.

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
> - **Between-point frames dilute the metric (follow-up #1) — RESOLVED (2026-07-09,
>   Stage 8 v0.4.0).** Position stats are now **rally-scoped** (in-rally frames
>   only) via `scope_to_rally_frames`, using Stage 7 v0.3.0's clean boundaries;
>   movement never integrates a step across a rally boundary or a >`MOVE_MAX_GAP_SEC`
>   gap. `position.scope` = `in_rally` (or `whole_clip` + warning if rallies.json is
>   absent). pb_2min: user kitchen 26.2%→**33.6%**, partner→**56.9%**, both-at-kitchen
>   22.6%→**33.3%**; net_play subscore 3.55→**3.89**, rating 3.71→**3.81** (band 4.0).
>   Smoke 16/16.
> - **Near-side user↔partner role gap under-counts user kitchen time (follow-up #2,
>   = Stage 2.5).** At some frames BOTH near tracks resolve to a single role (pb_2min
>   f6420: both labeled `partner`, user unidentified), so wherever the user is
>   temporarily mislabeled their kitchen frames are attributed to partner (or
>   dropped). Consequence: the user's 26.2% kitchen is a slight UNDER-count; the
>   near-team aggregate (both-at-kitchen) is unaffected. Root cause is Stage 2.5
>   near-side continuity (same appearance re-id that handles the USER's ID swaps is
>   not yet keeping the user/partner *split* stable frame-by-frame — cf. the "Court
>   switches cause user track loss" entry above). Deferred to a Stage 2.5 pass.

## Stage 8 — movement distance uses an un-fps-scaled jitter floor (CONFIDENTLY WRONG) (2026-07-09)

**Observed:** 2026-07-09, while rally-scoping the position metrics on pb_2min (4K/60fps).

**Problem:** `MOVE_MIN_STEP_FT = 0.25` (the per-frame step below which motion is
treated as jitter) is **not fps-scaled**. At 60 fps, 0.25 ft/frame = **15 ft/s
(~10 mph)** — so the movement integrator **rejects 84.5% of real steps** (the median
in-rally step is 0.078 ft ≈ 4.7 ft/s, normal walking/shuffling) and sums mostly the
noise spikes that DO exceed 15 ft/s (p99 step = 2.37 ft/frame ≈ 142 ft/s, physically
impossible). So `distance_ft_total` / `distance_ft_per_min` measure jitter, not
movement. Like net-play, the **Stage 9 `movement` dimension is stamped confidence
1.00** on this — confidently wrong. (Same class as the front-foot and net-play bugs:
a systematic error a confidence number can't see.)

**Root cause:** the floor was tuned at the 30 fps design point (where 0.25 ft/frame =
7.5 ft/s, already high) and never scaled by `fps/30`. Cf. the real-vs-synthetic
adaptation pattern (SESSION_HANDOFF): "fps scaling — frame-count/per-frame params ×
fps/30".

**Where to fix:** define the jitter floor in **ft/second** (e.g. ~1–2 ft/s) and
convert per-frame via `fps`, OR scale `MOVE_MIN_STEP_FT` by `30/fps`. Also cap
per-frame steps at a physical max (e.g. ~25 ft/s sprint) to reject the tracking
spikes rather than count them. Re-validate `distance_ft_per_min` against a plausible
range for a rec player.

> **RESOLVED (2026-07-09, Stage 8 v0.5.0).** A per-frame speed floor turned out to be
> the wrong tool (jitter has high *instantaneous* speed, so a low floor doesn't
> reject it and a high one rejects real slow movement). Fix = integrate from a
> **noise-robust downsample** (`compute_movement`): bin frames into
> `MOVE_SAMPLE_DT_SEC` (0.2s) windows, take each window's MEAN position (averages
> out high-frequency jitter), integrate displacement between temporally-adjacent
> same-rally windows, gated by `MOVE_MIN_STEP_FT` (0.3 ft ≈ 1.5 ft/s floor) and
> `MOVE_MAX_SPEED_FTPS` (24 ft/s cap, rejects teleports / front-foot L↔R switches).
> All thresholds fps-independent. pb_2min user `distance_ft_per_min` **492 → 192**
> (plausible ~3 ft/s average for rec play); per-rally ~34 ft. Smoke 16/16.

## Stage 7 — rally over-segmentation (micro-rallies) (2026-07-07)

**Observed:** 2026-07-07, same consumer-report review. Stage 7 segmented **8 rallies**
on pb_2min; the operator counts **6**. Rallies 0–5 are real (5.5–19.3s); rallies **6
and 7 are 0.8s and 1.1s** (2 shots each, 1.9s apart) — spurious micro-splits from the
ball-out-of-play splitter.

**Where to fix:** a minimum-rally filter in Stage 7 (min duration and/or min shots —
a real rally isn't 0.8s). Easy. Re-validate count against the operator's eye.

> **RESOLVED (2026-07-09, Stage 7 v0.3.0, commit `13b629c`).** Added a
> minimum-rally filter: a segment is dropped only when it is **BOTH** shorter than
> `MIN_RALLY_SEC` (2.0s) **AND** has fewer than `MIN_RALLY_SHOTS` (3) shots
> (conservative AND-logic — either long *or* many-shot always survives). Note
> rally 7 starts on a **falsely detected serve**, so a serve-flag guard alone could
> not catch it; size is the only clean separator (real rallies here ≥5.45s / ≥4
> shots). A lone serve-fault (`n_shots==1`) is guarded and never dropped. Dropped
> shots roll into `unassigned_shots` (accounting reconciles) with an explicit
> warning listing each dropped span. Real ball only; synthetic bars unmoved.
> **pb_2min: 8 → 6 rallies** (matches the operator's count); mean rally length
> 5.19 → 5.67 shots; `rally_consistency` 3.52 → 3.73. Smoke: Stage 7 9/9, Stage 8
> 16/16. *Accepted limitation:* a genuine ultra-short point (2 shots, <2s) would
> also drop — rare, and not separable from between-point taps on the noisy real
> ball. Tunable via `--min-rally-sec` / `--min-rally-shots`.

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

> **RESOLVED (2026-07-09, Stage 9 v0.4.0 `52c9a56` + Stage 10 v0.5.0 `bc85e80`).**
> Stage 9 now rates the **7 official USAPA categories** (strategy, third_shot, dink,
> volley, serve_return, forehand, backhand); Stage 10 findings/why/drills re-keyed to
> match. Design + weights in `docs/USAPA_REALIGN_DESIGN.md`. Operator decisions: full
> 7-category structure, hard-gated, single heavily-caveated estimate. Each category
> carries `coverage_status` (measured/partial/not_assessable); count-only strokes
> (forehand/backhand) and serve_return-with-no-serves route to `not_assessable`, and
> Stage 10 adds a zero-event guard (0 dinks → not assessable, not a #1 focus). The
> confidence-weighted estimate leans on Strategy; a loud USAPA-COVERAGE warning +
> `reliability.{measured,not_assessable}_categories` surface that only ~1 of 7 is
> measured today. pb_2min: estimate 3.95 / band 4.0, confidence 0.30; Strategy
> `measured`, third_shot/dink/volley `partial`, serve_return/forehand/backhand
> `not_assessable`. Smoke: Stage 9 9/9, Stage 10 9/9. **This is a legitimacy/naming
> win, not new signal** — the 6 shot-based categories stay data-limited until ball
> recall (C4) / serve detection (C3) / stroke-side (F16) / shot speed (F7) land
> (build-program ADD step). Also fixed in passing: `score_volley` read the wrong
> (nested) shot_mix path and always scored NEUTRAL (`c2a703f`).

## Stage 5/6 - "wrong player" is a CONTACT-TIMING error, not an identity error (2026-08-18)

Recorded because the obvious diagnosis is wrong and cost a wasted plan. The operator flagged
7 shots as attributed to the wrong near-side player. The natural reading -- role assignment
or track identity -- was **measured and refuted**:

- **Tracking is stable.** Track 1452 carries 23 shots over 171s and is on the same person in
  every sampled frame (2s, 60s, 120s, 200s renders). No ID swap, no flip.
- **Roles never change.** The near-side labels are consistent for the whole clip.
- **Association is not ambiguous.** At 6 of the 7 disputed shots the chosen player's box is
  **0-90px** from the ball while the other near-side player is **229-927px** away. The
  associator picks the obviously-nearest player and is right to.

The actual cause is that the CONTACT FRAME is late. Rendered at t=200s: at **199.60s** the
partner is bent low with her paddle on the ball (a kitchen dink); at **200.13s**, where the
shot is detected, the ball has travelled up to the user's head and is associated with him.
The operator independently reported the same thing -- *"#73 at 3:18.3 was hit by partner,
not the user. And shot looked closer to 3:17.5"* -- flagging a ~0.8s timing error in the
same note as the attribution error.

Measured across all 7: the partner's closest approach to the ball is 4-51px at a frame
**17-60 frames EARLIER** than the detected contact in 5 of 7 cases. So fixing attribution
means fixing WHEN the contact is detected, not WHO it is assigned to. Reassigning the hitter
without moving the frame would only trade one wrong answer for another.

`python -m tools.show_attribution data/pb_5_minute_outdoor-7` renders the disputed frames
with both near-side boxes so the operator can adjudicate directly.

### Latent risk found alongside: the user was never identified

`session.json` has `steps.user_clicks: false`, and `track_roles.json` seeds the user from
`user_starting_corner: "left"` with **confidence 0.5** and basis `"starting-corner"`; the
partner is then derived as `"simultaneous-with-user"`. On this clip the guess is correct and
stable, but nothing verifies it. If that coin-flip lands wrong on another video, EVERY
user-specific metric and the USAPA rating belong to the partner, with no signal that anything
is amiss. See docs/USABILITY_BACKLOG.md.

## Cross-venue check of the 2026-08-18 Stage 5/7 changes (CLOSED; indoor RECALL is the real gap)

Everything in that session was tuned on ONE outdoor clip, so it was re-run against
`pb_3_min_indoor_1_court_b`, which has independent operator truth (10 points, 82 shots).

**Generalised well — rally structure:**

| indoor | before the session | after |
|---|---|---|
| serve recall | 50% | **70%** |
| serve precision | 62% | **88%** |

**OPEN — the indoor shot count moved from over to under truth:** 88 → 71 against a truth of
82. Isolated to the wrong-object latch gate (85 shots with it off, 71 with it on); the
same-side strength rule is not involved. Narrowing `LATCH_WINDOW_S` recovers only part of it
(79 at 0.25s) and costs outdoor serve accuracy badly (86%/86% → 64%/69%), so the committed
1.0s stands.

The count alone cannot settle whether those 14 removals were real shots or junk — the
outdoor clip is the cautionary case, where 125 detected contained 34 non-shots AND 17 missed
shots at the same time, so a count near truth can hide large errors in both directions. The
independent evidence says the gate helps: with it ON, indoor serve recall is 70% vs 50% OFF
and precision 88% vs 83%. A better-structured rally sequence from fewer shots means the
removals were mostly junk.

**CLOSED 2026-08-18 — no new labelling was needed.** The operator's existing
`_truth_worksheet.csv` already lists every point with its start, end, server and shot count,
so a detected shot outside every rally window is a between-point false positive *by the
operator's own account of when play was live*. `tools/score_rally_shots.py` splits the count
on that boundary:

| indoor | inside rallies (truth 82) | outside = junk |
|---|---|---|
| latch OFF | 62 | 23 |
| latch ON | 60 | **11** |

The gate removes 14 shots: **12 junk and 2 real, a 6:1 ratio**. The apparent regression
(88 → 71 against a truth of 82) was the total moving AWAY from truth while becoming more
correct — precisely the failure mode that made a raw count untrustworthy in the first place.
The gate stays.

**The real indoor gap is RECALL, not precision:** 60 of 82 shots inside rallies = 73%, and
22 missing. It is concentrated — rally 0 (−4), rally 3 (−7), rally 5 (−3) and rally 9 (−6)
hold 20 of the 22, in about 55 seconds of the clip:

| rally | window | truth | got | delta |
|---|---|---|---|---|
| 0 | 3–20s | 17 | 13 | −4 |
| 3 | 63–77s | 14 | 7 | **−7** |
| 5 | 104–116s | 10 | 7 | −3 |
| 9 | 166–178s | 11 | 5 | **−6** |

Outdoor recall is ~85% against ~73% here, so the ball track is weaker indoors and this is
the next thing to work. A per-shot review is still worth having eventually, but it is no
longer needed to answer the precision question.

## Stage 5 - the HANDLING filter compounds every missed shot (OPEN, cause confirmed)

The operator reviewed the annotated indoor video (`tools/annotate_full.py`) and reported
21 timestamped missed shots, frozen in `data/pb_3_min_indoor_1_court_b/missed_review.json`.
Their summary led with the pattern: *"if a serve is missed, often 2 or 3 other shots after
it are missed."*

That is confirmed, with a named cause. Checking each of their timestamps against detection
with and without `reject_same_side_runs`:

| of the 21 confirmed missed shots | |
|---|---|
| **the handling filter deleted it** | **9** |
| never detected at all | 4 |
| actually detected (timing/attribution, not a miss) | 8 |

The deleted ones come in consecutive blocks — 67.6, 70.4, 71.2, 73.9, 74.3, 74.8s in rally 3
— all through one fast drive/volley exchange.

**The mechanism.** The filter collapses consecutive same-net-side impacts because a team
cannot legally hit twice in a row. That is true of REAL play, but it assumes our detection
is complete. When a shot is missed, the two real shots either side of it become
"consecutive same-side", and the filter deletes one of THOSE. Every miss costs a second
shot, so recall failures compound instead of adding up.

The filter cannot simply be removed. On the indoor clip that lifts in-rally shots from 60 to
89 (truth 82), but on the outdoor clip it produces **174 shots against ~108 true**, because
outdoor has real between-point ball-handling and an adjacent court.

### Two fixes attempted and REJECTED — both defeated by ball height

1. **Did the ball cross the NET LINE between the two impacts?** (image-space line, the edge
   the half-court polygons share.) Recovered 1 real shot and added 3 junk indoors, changed
   outdoor by nothing. It fails exactly where it is needed: in a kitchen volley exchange the
   ball stays ABOVE net height throughout, so against a ground-projected net line it never
   changes sides. Ball visibility was ruled out as the explanation — the ball is visible for
   a median 58 of 62 frames between in-rally impacts.

2. **Did the ball come within paddle reach of an OPPOSING-side player?** Players stand on the
   ground, so their side needs no height estimate. Excellent indoors — in-rally shots 60 →
   **79** of 82, confirmed misses recovered 8/21 → **16/21**, serve recall 70% → 80%. But
   outdoors it emits **160 shots** (fp 23 → 27, misattribution 1 → 6, serve 86%/86% →
   79%/79%), and tightening the reach to 0.15x barely helps (still 148 shots, misattribution
   5/7). A high ball merely LOOKS near a far-side player in image space — the same
   z-ambiguity, third time.

Neither shipped; the standing rule is that a change must not regress another venue.

**What this needs.** Both rejected fixes ask "did the ball go to the other side", and both
fail because that question needs the ball's HEIGHT, which one 6ft camera cannot give. A fix
has to either avoid the question or get height another way. The one untried angle that
avoids it: instead of deleting all-but-one impact in a same-side run, keep them and let
Stage 7's point structure arbitrate, since a rally with the wrong parity is detectable
downstream where a single run is not.

## Indoor review - operator observations NOT yet investigated (2026-08-18)

From the same review that produced `missed_review.json`. The handling-filter cascade was
chased because it accounted for 9 of the 21 missed shots; these were recorded and NOT
investigated, and are listed so they are not mistaken for closed.

1. **Wrong player, ~5 cases.** "#30 opponent" was the partner (1:15.6); "#47 says user but
   it's the partner"; 2:52.7 "says #68 user when opponent hits it"; "#70 is user dink but
   says partner". Note the outdoor clip's wrong-player class turned out to be the same-side
   filter deleting the real shot, not an attribution bug — check that first here.

2. **End-of-rally shot missed, 2 cases.** :19.57 partner drive that went out of bounds, the
   last shot of the rally; 1:55.6 opponent drive hit out. Both are shots that END the point
   by going out. Plausibly the same root as the outdoor "6 of 17 missed shots went into the
   net" — a shot that terminates a rally has no return after it, so anything relying on a
   following contact has nothing to anchor to.

3. **Phantom shots after the point ended.** At 1:18.3 two circles appear at once — "#32
   opponent, #33 partner … #31 by user was a winning shot missed by opponent. So there is
   no #32 and #33. Ball is not visible but a few yellow circles appear (not around ball)."
   Also #65, between points, with the marker on the partner's paddle handle. These are
   detections with NO BALL, which the ball-visibility precondition is supposed to prevent.

4. **The serve is detected at the BALL TOSS, not the strike.** Rally 9: "#67, opponent as
   1st shot, however it does it while she is about to throw the ball up for the serve and
   not when she actually hits the ball." `serve_appearance` fires when the ball first appears
   after dead time — which is the toss, not the paddle contact. Consistent with the measured
   serve offsets: one serve early by 0.58s, four LATE by 0.6-3.6s where a later shot
   inherited the serve label after the real serve was missed.

5. **Shot numbering goes off by one after a miss** (:7.98). A display consequence of 1-4
   rather than its own defect, but it makes any future numbered review harder to read.

## Ball HEIGHT from apparent size — feasibility PASSED (2026-08-19)

The operator asked whether knowing the camera angle would let us position a volley. It would
not: calibrating the court already determines the camera pose, and the homography IS that
pose. The shortfall is arithmetic and per-frame — a ball position has 3 unknowns (X, Y, Z)
and a pixel gives 2 equations. A pixel specifies a RAY; camera pose fixes the ray exactly but
cannot say where along it the ball sits. "z = 0" is the third equation, which is why the
ground homography works at all, and why it is exact at a bounce and nonsense in the air.

Demonstrated on the indoor volley exchange (rally 3), ball projected through the homography:

| time | pixel | → court (ft) |
|---|---|---|
| 73.9s | (1704, 744) | (−30.5, **67.4**) |
| 74.8s | (2433, 657) | (−42.7, **178.8**) |
| 76.3s | (1851, 681) | (−57.5, **120.3**) |

On a 44 ft court. Every one reads "far side". Against that, 40 of 41 detected BOUNCES project
inside the court (median y = 22.5 ft, right at the net) — same camera, same ball, same maths.
The only difference is z.

**Apparent ball size supplies the third equation, and is the only candidate that helps a
VOLLEY.** A ballistic fit (the textbook answer) needs curvature, so it is weakest during the
short flat exchanges where this problem bites. A pickleball is a known 2.9 inches, so its
pixel diameter fixes range directly with no assumption about motion.

`tools/measure_ball_size.py` tests it, with nothing tuned per court — px/ft comes from each
clip's own calibration, and the measurement bias from each clip's own bounces:

| clip | bounce (control) | in flight | AUC |
|---|---|---|---|
| indoor | 1.52 | 4.66 | 0.85 |
| outdoor | 1.72 | 2.42 | 0.84 |

**Both courts agree despite the outdoor ball being half the pixel size** (15 px near / 7 px
far, vs 24 / 11 indoors) — so the signal is not a property of one venue, which was the
operator's stated requirement. The bias sits at 1.5-1.7 rather than the ideal 1.0 because FWHM
over-reads a small bright blob (motion blur, halo); it is stable, and each clip's bounces
calibrate it out.

**Caveats before building.** The AUC estimate is noisy — only 8-20 bounces in the control, and
the same clip moves between 0.78 and 0.85 depending on the window; call it ~0.8. It is also
per-FRAME, whereas the real question ("did the ball reach the far side between these two
impacts?") aggregates over 30-60 frames, which should do much better — but that is an
expectation, not yet measured. Turning a ratio into a court POSITION additionally needs the
camera's ground position and height; those are recoverable per-clip from player boxes (feet
at z=0, head at ~5.5 ft) without asking the operator for anything.

## Ball 3-D recovered from one camera — WORKING (2026-08-19)

`tools/ball_3d.py`. Two measurements close the z gap, neither needing anything from the
operator beyond the calibration already on disk.

**Camera ground position, from player boxes.** For a camera at height H and a person of
height h, the head projects through the homography to `C + k*(P_feet - C)`, so C, the feet
and the projected head are COLLINEAR — every player in every frame is a line through the
camera. Solved on both clips with a **median line residual of 0.0 ft**; the model is exact,
not approximate. Camera height comes out at **6.7 ft indoor / 6.6 ft outdoor**, against a rig
that is a ~6 ft camera — an independent check nobody supplied.

**Ball range, from apparent size**, as in `tools/measure_ball_size.py`.

    P_true = C + (P_ground - C) / k,   k = (measured_px / predicted_px) / bias
    z      = H * (1 - 1/k)

Validated indoor, frames 3600-4400:

| | raw ground projection | reconstructed |
|---|---|---|
| z at bounces (should be 0) | — | **0.14 ft** |
| court y in flight | 80.4 ft | **23.0 ft** (net is at 22) |
| within the play envelope | 26% | **89%** |

x,y do NOT depend on the assumed person height — only z in feet does.

### Net hits: the signature is SUSTAINED z≈0 near the net

Confirmed against the operator's timestamped net hits. At the 19.45s net hit the ball falls
from 4.33 ft to the floor and stays pinned there beside the net for over two seconds
(t=19.20-21.82, z=0.00, 0.1-3 ft from the net line) — matching their note exactly: *"from
:19.45 until :24 ball was bouncing and rolling along net."*

The discriminator is **not** z≈0 alone: ordinary play reaches z=0 on every bounce (control
p10 = 0.00). It is z≈0 SUSTAINED. A bounce touches the floor for a frame or two; a dead ball
stays there for seconds. Near the net that is a net hit; away from it, a ball hit out or
rolling. This is the first route to net-hit detection that has produced a signal — the
July 2026 deferral was made without height.

### Weakest link

`bias` is estimated from only 5-20 bounces and ranges 1.59-2.03 across clips and windows. It
scales k directly, so absolute heights inherit that error. Widen the estimation window, or
find a bounce-independent calibration, before anything depends on absolute z. Relative
comparisons within one clip are far safer than cross-clip thresholds.

## Height separates "a shot was missed" from "ball-handling" — AUC 0.95 (2026-08-19)

The test the ball-3D work was built for. `reject_same_side_runs` collapses consecutive
same-side impacts, which is right for handling and catastrophic when a shot between them was
MISSED — 9 of the operator's 21 confirmed misses. Two earlier attempts to tell the two cases
apart died on ball height (net-line crossing, opposing-player reach).

With height available, for every same-side pair the filter would collapse, take the maximum
reconstructed height between the two impacts. Ground truth is the operator's missed-shot
list, restricted to the four rallies they reviewed:

| | n | median max height | clears the net |
|---|---|---|---|
| **CROSSED** (operator lists a shot between) | 7 | **5.21 ft** | 100% |
| handling (no shot between) | 12 | 4.24 ft | 67% |

**Separability AUC = 0.95**, well above the ~0.8 per-frame figure — aggregating over 30-60
frames of flight does exactly what was predicted.

Two things to note before building on it. The sample is small (7 vs 12 pairs), and the
"handling" label is only as complete as the operator's review. And the discriminator is NOT
"did the ball clear the net" — 67% of handling pairs clear it too, because they are mostly
duplicate detections inside a live rally where the ball is legitimately airborne. It is the
APEX MAGNITUDE that separates them, which is a weaker physical story than a clean geometric
rule and deserves a larger labelled sample before it becomes a threshold.

Reproduce: `python -m tools.probe_crossing_signal <off-filter classified.json>`

## What ball height unlocks — the plan (2026-08-19)

With `tools/ball_3d.py` working, several things recorded here as blocked are worth reopening.
Operator refinements to this plan are folded in below and change it materially.

### Between-point balls are DERIVED, not classified

The 18-of-23 remaining outdoor false positives (feeds, throws, balls rolling after a net hit,
shots after the point ended, pick-ups) were being treated as a classification problem, and it
resisted every attempt: 7 of 11 feeds are struck with a paddle, so they are physically real
shots that only game context excludes.

**Operator's point: if serves and rally-ends are accurate, between-point needs no classifier
at all — it is everything between a rally end and the next serve.** That is already the
design (`structure_points` sets `is_between_point` from rally boundaries); it was blocked
only because rally ends were undetectable. So this whole category collapses into the
rally-boundary work rather than needing its own solution.

It also disposes of a bad idea: a fed ball was going to be identified by its low, slow arc.
The operator points out a lob during a rally has a low slow arc too. Correct — and moot, if
the boundaries carry it.

### Rally end — the operator's taxonomy, now measurable

| rule | measurement | evidence |
|---|---|---|
| hit into the net → hitter loses | **sustained** z≈0 near the net | demonstrated at the 19.45s net hit: falls 4.33 ft → 0.00 and stays pinned 2+ seconds, 0.1-3 ft from the net line |
| bounces outside the court → hitter loses | bounce position (exact at z=0) | positions already trustworthy — 40 of 41 land inside the court |
| bounces in, not returned → hitter wins | bounce in court + no following contact | computable once bounces are reliable |

Note the discriminator for a net hit is z≈0 **sustained**, not z≈0 — ordinary play reaches
z=0 on every bounce. A bounce touches the floor for a frame or two; a dead ball stays for
seconds.

Bounce DETECTION also improves: currently a pixel y-flip heuristic, it becomes a local
minimum of z at z≈0. Bounce POSITION was never the weak part.

This reopens "Stage 7 - RALLY END is undetectable" (six routes failed). All six failed for
want of height.

### Serves

Behind-the-baseline and the time gap are already implemented and need no height (the
hitter's feet are at z=0, so their position is exact). What is new is **contact at or below
the hip** — measurable now for both the ball and the hip, since the same camera model applies
to any pose keypoint.

**Operator's caution: a drive can also be struck below the hip, so the hip rule is not
sufficient alone — it is the CONJUNCTION (behind the baseline + opening time gap + at or
below the hip + the ball rising from a toss beforehand) that makes a serve hard to confuse
with anything else.** Build it as a combined test, never as a single rule.

Height also attacks the toss bug the operator found: `serve_appearance` fires when the ball
first APPEARS after dead time, which is the toss, not the strike. A toss is unmistakable in
height — the ball rises with no impulse; the strike is the impulse at the bottom.

**Caveat:** contact frames are the WORST place to measure ball size (paddle occlusion, motion
blur, the ball against a bright body). Validate before trusting a hip comparison.

### Also unlocked

- **Shot type** — drive/dink/drop/lob is currently inferred from pixel velocities; apex height
  in feet IS the physical definition. Most likely fix for the 42% type-error rate.
- **Ball speed in real units** — 3-D positions plus frame rate give ft/s. Previously only
  px/frame, which conflates depth: the same shot reads slower at the far baseline purely
  because it is farther away. Needs fitting over a flight segment, not frame differences,
  since differentiating a noisy position amplifies noise.
- **Volleys** — becomes "no bounce since the last contact" rather than an inference.
- **Adjacent-court balls** — reconstruct to positions off our court.

### GATING ITEM: the bias constant

Everything absolute rests on `bias`, which ranges **1.59-2.03** across clips and windows
because it is estimated from only 5-20 bounces. Relative comparisons within a clip are sound;
heights in feet, speeds in mph and any cross-venue threshold are not, until this is fixed.
Do this first — it is cheap and everything else inherits its error.

## Ball-size calibration: the GATING ITEM is fixed (2026-08-20)

`bias` — the factor by which the measured ball blob over-reads its true size — ranged
1.59-2.03 across clips and windows, and every absolute number inherited that. Four attempts,
three of them wrong, and the wrong ones are worth recording:

1. **Motion blur.** Hypothesis: blur smears the ball along its travel, so a diameter from
   blob AREA grows with speed, and bounces (where the ball is slowest) under-correct fast
   balls. **REFUTED** — correlation with pixel speed is −0.08, and the speed bands show no
   trend (2.28, 2.74, 2.39, 2.82, 2.05). Using the blob's minor axis, which is
   blur-invariant, changed nothing.
2. **Fixed additive blur** (`measured = true + c`). **REFUTED** — fits far worse than
   multiplicative (CV 1.001 vs 0.200).
3. **A single constant, robustly estimated.** Rejecting blob merges (ratio ≥ 2.0, where the
   ball fuses with a line, shadow or player — 6 of 43 indoor, 17 of 69 outdoor) narrowed the
   spread from 1.59-2.03 to 1.44-1.68, and fitting over every bounce in the clip narrowed it
   again to 1.40 / 1.51. **But the bounce control then reconstructed to z = 1.19 ft instead
   of ~0.** No constant can be right, because the over-read is SIZE-DEPENDENT: measured
   outdoors at 1.74 at pred 6 px, 1.53 at 12 px, 1.30 at 21 px. A per-window constant only
   ever appeared to work because it matched that window's distances.
4. **Linear fit `measured = a*pred + c`** — a small multiplicative over-read plus a fixed
   blur width. This is what works.

Fitted over every bounce in the clip, with merges rejected:

| clip | fit | bounces |
|---|---|---|
| indoor | measured = **1.17**·pred + **2.94** px | 37 (6 rejected) |
| outdoor | measured = **1.18**·pred + **3.24** px | 52 (17 rejected) |

**Two different cameras, two resolutions, effectively the same fit** — it is capturing a real
imaging property (the detector's point spread), not per-clip noise.

Validated indoor, ONE calibration serving the whole clip:

| window | z at bounces (should be 0) | flight court y | within envelope |
|---|---|---|---|
| 3600-4400 | 0.30 ft | 19.5 ft | 89% |
| 1000-2000 | 0.77 ft | 19.1 ft | 93% |

**Residual error is ~0.3-0.8 ft (4-9 in) in absolute height.** Good enough for sustained-z≈0
net-hit detection and for crossing decisions (the net is 2.83 ft); NOT good enough to quote
absolute heights to the inch. Cached per clip in `ball_size_calib.json`, since the walk over
every bounce is expensive and the fit is a property of the camera, not the analysis window.

## Rally END detection — WORKING, precision is the open side (2026-08-20)

"Stage 7 - RALLY END is undetectable" records six failed routes. All six failed for want of
ball height. With `ball_3d.parquet` supplying it, `tools/detect_rally_ends.py` implements the
operator's taxonomy directly:

    hit into the net               -> hitter loses    (SUSTAINED low + near the net + not moving)
    ball bounces outside the court -> hitter loses    (bounce position, exact at z=0)
    bounces in and is not returned -> hitter WINS     (bounce in + no contact for 2s)

| | recall | precision |
|---|---|---|
| indoor point-ends (operator truth, 10) | **9/10** | 9/14 |
| outdoor net hits (operator-confirmed, 7) | **7/7** | 7/10 |

**Two mistakes on the way, both instructive.**

1. *A dead ball is a statistical state, not a clean one.* The first version required an
   unbroken run of low+near-net samples, which rejected the operator's 19.45s net hit
   outright — the ball is plainly dead on the floor for 4+ seconds, yet its reconstructed
   court_y wanders 18.6-30.2 ft and z crosses 1.0 ft repeatedly. Asking for a FRACTION of a
   window instead took outdoor net recall from 4/7 to 7/7.
2. *A window-scan bug that silently returned nothing.* The loop extended its window while
   `ts[b+1] - ts[a] < sustain_s`, guaranteeing the very next check `if ts[b] - ts[a] <
   sustain_s: break` always fired — so it broke at the first index every time and found
   nothing. Outdoor net detection read 0/7 and looked like a modelling failure; it was a
   two-line indexing error. **Symptoms that look like "the idea does not work" are worth one
   unit test before they are believed** — three lines of synthetic data caught it instantly.
3. *Low near the net is not sufficient.* Kitchen dinking puts the ball there repeatedly. A
   dead ball also stops TRAVELLING (`DEAD_TRAVEL_FT`), which lifted outdoor net precision
   from 7/12 to 7/10.

**Precision is the open side** — roughly 1.4 detected ends per real one. Indoor false calls
sit at 5s, 81s, 101s, 127s and 156s; 81s is very likely the 77s end (the only miss) detected
late, so the real over-firing is nearer 4.

**Treat these numbers as provisional.** They come from 10 points and 7 net hits, and several
thresholds were swept against them. A third labelled clip is needed before trusting the exact
figures.

## Between-point from measured ends: wired, NOT enabled — precision is the blocker (2026-08-20)

The operator's insight was that between-point balls need no classifier: with accurate serves
and rally-ends, between-point is simply everything between a point ending and the next serve.
That is right, and Stage 7 already drops `is_between_point` shots — the flag just never had
real boundaries (0 of 125 set on the acceptance clip).

`segment_rallies.apply_rally_ends` now derives it from `rally_ends.json`. **It is OFF by
default (`--use-rally-ends`), and the reason is measured, not cautious.**

Rally-end detection runs at 64-70% precision. Every FALSE end marks the live play behind it
as dead, and one missed serve (serve recall is 86%) means play never "resumes", so a single
false end can flag an entire real rally. Capping the dead interval limits the damage but does
not reverse the sign:

| cap | labelled junk excluded | real shots lost |
|---|---|---|
| no cap | 15/28 | **29/89** |
| 4s | 7/28 | 12/89 |
| 8s | 9/28 | 16/89 |
| 15s | 11/28 | 25/89 |

Every setting costs roughly **1.3-1.5 real shots per junk shot removed**. Enabling it would
make the analysis worse.

**What it needs:** end precision materially above 70%. Recall is already there (9/10 indoor
point-ends, 7/7 outdoor net hits), so this is entirely a false-positive problem, and the
false ends are identified — indoor 5s, 101s, 127s, 156s. Fixing those turns this on.

The default pipeline is unchanged and re-verified: 23/34 false positives, 94 real shots kept,
1 wrong-player, serves 86%/86%, 14 rallies.

## Rally-end restructure ATTEMPTED and REJECTED; and there is no third clip on disk (2026-08-20)

### The false ends are the ball still FLYING

Diagnosed the four indoor false ends against the reconstruction. Every one calls the point
over while the ball is airborne:

| false end | reason | ball height | travel |
|---|---|---|---|
| 4.69s | not-returned | **4.22 ft** | 26.4 ft |
| 127.24s | out | **3.41 ft** | 7.3 ft |
| 155.66s | out | **4.87 ft** | 28.6 ft |

against every correct `net` end at z ≈ 0.00-0.76 ft and travel under 6 ft.

### Anchoring on the dead ball instead — WORSE, measured

That suggested a cleaner rule: a point ends when the ball STOPS BEING IN PLAY, so find dead
intervals directly and read the reason off where the ball died (beside the net / outside the
lines / inside the court). It is more principled and it is empirically worse:

| | indoor point-ends | indoor precision | outdoor net hits |
|---|---|---|---|
| shipped (3 rules, shot-anchored) | **9/10** | **9/14 = 64%** | 7/7, 70% |
| dead-interval anchored | 7/10 | 7/16 = 44% | 7/7, 70% |

Swept the "how long after a contact" constraint over 2/3/4/6s; none reached the shipped
version. **Reverted.**

Why the tidier rule loses: the shipped `out` and `not-returned` rules read the BOUNCE stream,
which still catches point-ends where the ball never visibly settles — it rolls out of frame,
or is picked up straight away. Requiring the ball to come to rest on camera misses those. The
two mechanisms are complementary, and firing early on a flying ball costs less than missing
ends outright.

### No third clip exists without new footage

`videos/` holds three files: the indoor clip, the outdoor clip, and a 20-second excerpt. Every
processed folder in `data/` traces back to one of those two full videos, so there is no
independent third venue to validate against. Confirming these numbers needs NEW footage from
the operator — a different court or session — then calibration, a pipeline run, `build_ball_3d`
(~30-60 min of decode), and rally start/end times in the `_truth_worksheet.csv` format.

Until then, every threshold in the rally-end work rests on 10 indoor points and 7 outdoor net
hits, and several were swept against exactly those. Treat them as provisional.

## Rally ends: the side-change rule, from operator review of the false positives (2026-08-20)

The operator viewed the three flagged indoor false ends and returned a verdict on each:

| flagged end | operator's reading | verdict |
|---|---|---|
| 4.69s | *"the partner waiting for a return; the ball in frame is a bounce on the opponent's side right before he returns it"* | genuinely FALSE |
| 127.24s | *"a serve by partner"* | genuinely FALSE — this is a point START |
| 155.66s | *"the opponent hitting the ball out … either way, the end of rally you detected is correct"* | **actually CORRECT** |

**Measured precision understates the detector.** 155.66s is a real end scored as a false
positive only because the worksheet lists that point ending at 159s and the match window is
2.5s. Recorded in `truth.json` under `operator_corrections`.

### The fix: a point cannot end before the ball has crossed the net

127.24s and 155.66s are inseparable on timing — 1.96s vs 2.22s after a serve. They separate
completely on whether the ball was ever played across:

| end | shots since serve | sides seen |
|---|---|---|
| 127.24 (false) | 1 | `['near']` — only the serve itself |
| 155.66 (correct) | 2 | `['far', 'near']` — a real exchange |

So an `out` or `not-returned` end now requires at least one hitter-side change since the last
serve. That is a rule of the game, not a tuned threshold.

**`net` ends are exempt, and the exemption was measured, not assumed:** a serve INTO the net
is a legitimate point end with no side change at all, and requiring one cost a confirmed
outdoor net hit (7/7 → 6/7). A net end is self-evidencing — the ball demonstrably went dead
at the net — whereas `out` and `not-returned` rest on an inferred bounce position.

| | before | after |
|---|---|---|
| indoor point-ends | 9/10, precision 9/14 | 9/10, **precision 9/13** (10/13 counting 155.66) |
| outdoor net hits | 7/7, precision 7/10 | 7/7, precision 7/10 |

Remaining indoor false ends: 4.69s (caused by a MISSED return, not by the end rule — the
opponent did play the ball, we just did not detect it), 80.63s (very likely the 77s end
detected late) and 101.33s (during a between-point stretch).

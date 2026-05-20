# Known Issues and Deferred Decisions

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

## Stage 4.5 - TrackNetV2 approach abandoned after two failed attempts

**Observed:** May 2026, across two distinct deep-learning training attempts.

**Outcome:** Both attempts produced models that failed validation badly
enough to be unusable. Stage 4.5 was rewritten in v0.3.0 to use
classical computer vision (background subtraction + blob detection +
ROI filter + trajectory smoothing) instead of deep learning. The
TrackNetV2 architecture and weights are no longer used anywhere in
the pipeline.

### Attempt 1 (v1, contract v0.1.0): Fine-tune Dettor's PPA weights

- Setup: Adam lr=1e-4, weighted BCE with pos_weight=100, 10,701 user
  labels across 4 videos, T4 GPU on Colab free tier.
- Trained 6 epochs in 2.8h; best val loss was at epoch 1.
- Validation against held-out outdoor video:
  - detection_rate_at_10px: 0.323 (615/1904)
  - false_positive_rate: 1.000 (1117/1117)
  - mean confidence on visible: 0.99; on invisible: 0.98
- Diagnostic (tools/diag_heatmaps.py): on 20 visible-ball frames, the
  real ball was NOT in the model's top-5 candidate peaks in 13 cases
  (65%). Two fixed pixel coordinates - tree branches in the distance
  outside the court - appeared as top-5 peaks on nearly every frame.
- Root cause: BCE with pos_weight=100 made "confidently wrong"
  locally optimal. Static cameras + no spatial augmentation let the
  model memorize fixed background features. Dettor's PPA-broadcast
  features anchored the early layers on the wrong visual prior.

### Attempt 2 (v2, contract v0.2.0): Train from scratch with MSE + spatial aug

- Setup: random init, MSE loss, Adam lr=1e-3 with cosine decay,
  weight_decay=1e-5, rotation +/-5 deg, translation +/-10%, A100 GPU
  on Colab Pro, batch_size=8.
- Aborted at epoch 25 because training had visibly collapsed at epoch
  10 to the trivial "predict zero everywhere" solution.
- Val loss flatlined at ~0.000078 from epoch 10 onward. The math: the
  target heatmap is ~99.97% zeros; predicting 0 everywhere achieves
  MSE ~0.00007 (matching observed loss exactly).
- Root cause: MSE on sparse positive targets has a stable trivial
  local minimum at "predict nothing." Symmetric to v1's failure mode
  - v1 made wrong-confident optimal; v2 made nothing-confident
  optimal.

### Why TrackNet was the wrong tool

TrackNetV2 was designed for broadcast tennis and badminton: mast
cameras 15-30 feet up, 4K resolution where the ball spans 8-12 pixels,
clean uniform backgrounds, tens of thousands of training labels per
model, stable lighting and ball appearance.

User's footage is amateur phone-camera at ~6 ft height, 1080p with
4-6 pixel balls, busy backgrounds (trees, fences, light fixtures,
windows), thin per-environment label counts (~2,500 per visual
environment), and varying lighting across indoor/outdoor venues. All
five characteristics violate TrackNet's training distribution.

### Where fixed

v3 (contract v0.3.0) rewrites Stage 4.5 entirely around classical CV:
a per-video calibration tool (`tune_ball_cv.py`) produces a
`ball_cv_params.json` that Stage 4's CV pipeline consumes. Background
subtraction exploits the static-camera assumption directly - "free"
signal that deep learning cannot use. Low resolution becomes a non-
issue because a 4-pixel ball is a clear blob after background
subtraction. New venues are handled via a 2-3 minute calibration
step per video rather than via collecting more labels and
retraining.

Stage 4 will be rewritten to consume the new schema; ETA after Stage
4.5 v3 is complete.

### Artifacts retained but unused downstream

- `data/models/tracknet_v2_finetuned_v1.pt` - v1 failed weights.
- `data/training/validation_report.json` - v1 validate.py output.
- `data/training/diag_v1/*.png` - v1 diagnostic visualizations.
- `train_log_v1.json` on Drive - v2 training log up to epoch 25.
- `_tracknet_model.py` vendored in `stages/track_ball/` - will be
  deleted when Stage 4 is rewritten.
- v2 final weights were not saved (training was aborted before the
  save cell).

### Generalizable lessons

1. When fine-tuning, verify the source model's training data matches
   your problem's characteristics. Dettor's PPA-broadcast priors were
   actively misleading on amateur phone footage.
2. For sparse-positive heatmap detection, both raw weighted BCE
   (high pos_weight) and raw MSE have symmetric failure modes that
   produce equally useless models. Focal loss is the standard
   remedy IF deep learning is the right tool.
3. Match the tool to the data, not the problem to the tool. Anchoring
   on "TrackNet because Dettor trained on pickleball" cost two days
   and most of an A100 monthly budget. Static-camera amateur footage
   is closer to 1990s surveillance CV than to 2020s broadcast-sports
   CV; the tooling should match.

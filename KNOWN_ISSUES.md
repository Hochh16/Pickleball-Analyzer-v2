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
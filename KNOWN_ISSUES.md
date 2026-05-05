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
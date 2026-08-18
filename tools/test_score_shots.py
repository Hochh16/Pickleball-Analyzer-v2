"""Regression guard for shot-detection accuracy.

This exists because two filters were recorded as solved and silently went inert when the
ball detector changed underneath them: the Stage 5 ground-ball filter (median 0.25 on junk
vs median 0.25 on real shots) and the teleport-in gate (n_rejected_teleport_in = 0, it has
never once fired). Nothing re-measured either for weeks. The lesson is that an accuracy
claim without a standing score is only an assumption, so these bars are asserted, not
documented.

The bars are the measured state as of 2026-08-18, not aspirations. Tighten them whenever a
change improves the numbers; that is the point. If a change trades false positives for
recall, BOTH bars move and the trade becomes visible instead of silent.

Skips when data/pb_5_minute_outdoor-7 is absent -- data/ is gitignored, so this runs only
on a machine that has the acceptance clip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.score_shots import score

CLIP = Path("data/pb_5_minute_outdoor-7")

# Measured 2026-08-18 after the latch gate, the same-side strength rule, and the Stage 7
# serve return tie-break. Serve accuracy is scored separately by tools/score_serves.py --
# a Stage 5 change can move BOTH, and one earlier version of the strength rule bought
# 22/34 here by dropping serve recall to 50%. Never tighten these without re-running it.
MAX_FALSE_POSITIVES = 23   # 34 -> 29 (latch) -> 23 (strength rule at 8s)
MIN_REAL_SHOTS_KEPT = 94   # 91 -> 94; the strength rule recovers real shots, not just junk
MAX_WRONG_PLAYER = 1       # was 7; these were never attribution errors -- see KNOWN_ISSUES


needs_clip = pytest.mark.skipif(
    not (CLIP / "classified.json").exists() or not (CLIP / "shot_review.json").exists(),
    reason="acceptance clip not present (data/ is gitignored)")


@pytest.fixture(scope="module")
def scored():
    shots = json.loads((CLIP / "classified.json").read_text(encoding="utf-8"))["shots"]
    review = json.loads((CLIP / "shot_review.json").read_text(encoding="utf-8"))
    return score(shots, review)


@needs_clip
def test_false_positives_do_not_regress(scored):
    assert scored["false_positives"] <= MAX_FALSE_POSITIVES, (
        f"{scored['false_positives']} operator-labelled false positives still emitted, "
        f"bar is {MAX_FALSE_POSITIVES}. Breakdown: {dict(scored['fp_by_cause'])}")


@needs_clip
def test_real_shots_are_not_traded_away(scored):
    """A precision gain paid for with recall is not a gain. This is the other half."""
    assert scored["real_kept"] >= MIN_REAL_SHOTS_KEPT, (
        f"only {scored['real_kept']} real shots kept, bar is {MIN_REAL_SHOTS_KEPT} -- "
        f"a filter is eating real shots to improve precision")


@needs_clip
def test_attribution_does_not_regress(scored):
    assert scored["wrong_player"] <= MAX_WRONG_PLAYER, (
        f"{scored['wrong_player']} shots attributed to the wrong player, "
        f"bar is {MAX_WRONG_PLAYER}")

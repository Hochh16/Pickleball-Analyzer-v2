"""Tests for point-end detection.

The detector runs as a pipeline step, so its failure modes are the pipeline's failure modes.
"""
from __future__ import annotations

import json

from tools import detect_rally_ends as dre


def test_an_out_end_is_trusted_ONLY_when_the_bounce_was_projected_on_the_ground():
    """`out` was 29% and distrusted -- but it was reading the bounce POSITION off ball_3d,
    the absolute per-frame reconstruction this system does worst, which is where
    bounce_xy values like [-42.9, 117.2] came from.

    Projected instead from the bounce PIXEL through the ground homography -- exact at z=0,
    where a bounce is -- and scored on the operator's out-ends across three clips:

        margin 1 ft   7 fire, 3 right    43%
        margin 2 ft   4 fire, 3 right    75%
        margin 3 ft   2 fire, 2 right   100%

    So the reason alone does not earn trust; the SOURCE of the position does. A `not-
    returned` end stays untrusted either way (1 of 6).
    """
    assert dre.TRUSTED_REASONS == {"net", "out"}

    def trust(ends):
        dre._mark_trusted(ends)
        return [e.get("trusted") for e in ends]

    assert trust([{"reason": "out", "grounded": True}]) == [True]
    assert trust([{"reason": "out", "grounded": False}]) == [False]
    assert trust([{"reason": "out"}]) == [False], "no provenance is not a ground projection"
    assert trust([{"reason": "net"}]) == [True], "a net end never reads a position at all"
    assert trust([{"reason": "not-returned", "grounded": True}]) == [False]




def test_a_missing_input_does_not_take_the_pipeline_down(tmp_path):
    """This runs as a pipeline step now, and Stage 7 works without it. A synthetic-ball run
    has no ball_3d.parquet at all; that must cost the point-ends, not the whole analysis."""
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "court.json").write_text("{}", encoding="utf-8")
    assert dre.main([str(clip), "--force"]) == 0
    doc = json.loads((clip / "rally_ends.json").read_text(encoding="utf-8"))
    assert doc["ends"] == []
    assert "ball_3d.parquet" in doc["skipped_missing"]

"""Tests for point-end detection.

The detector runs as a pipeline step, so its failure modes are the pipeline's failure modes.
"""
from __future__ import annotations

import json

from tools import detect_rally_ends as dre


def test_only_net_ends_are_trusted():
    """Measured against the operator's 36 point-ends across three clips: net 17/20 (85%),
    out 10/35 (29%), not-returned 1/6. Downstream takes only the trusted ones."""
    assert dre.TRUSTED_REASONS == {"net"}




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

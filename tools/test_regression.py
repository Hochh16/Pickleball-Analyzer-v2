"""Tests for the regression harness itself.

The harness exists to catch numbers moving, so the thing that must not silently break is its
ability to notice. These cover the comparison and the exit code, not the scorers (each of
those has its own truth and its own tests).
"""
from __future__ import annotations

import json
from pathlib import Path

from tools import regression as reg


def test_render_flags_only_what_moved(capsys):
    base = {"clipA": {"shots": 100, "fp_emitted": 22, "rating": 3.99}}
    now = {"clipA": {"shots": 100, "fp_emitted": 25, "rating": 3.99}}
    moved = reg.render(now, base)
    out = capsys.readouterr().out
    assert moved == 1
    assert "was 22 (+3)" in out
    assert "shots" in out and "was 100" not in out   # unchanged rows carry no marker


def test_render_handles_a_key_absent_from_the_baseline(capsys):
    """A newly added measurement is not a regression -- it has nothing to move against."""
    moved = reg.render({"clipA": {"shots": 100, "serve_timing_median_s": 0.02}},
                       {"clipA": {"shots": 100}})
    assert moved == 0


def test_missing_truth_is_absent_not_zero(tmp_path):
    """A clip with no shot_review.json must omit the false-positive keys entirely. Reporting
    0 would read as 'no false positives' -- the opposite of 'not measured'."""
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "classified.json").write_text(json.dumps(
        {"shots": [{"shot_id": 1, "t_sec": 1.0, "shot_type": "drive", "is_volley": False}]}),
        encoding="utf-8")
    m = reg.measure(clip)
    assert m["shots"] == 1
    assert "fp_emitted" not in m
    assert "in_rally_shots" not in m


def test_identity_gap_is_reported(tmp_path):
    """shots = volleys + bounces is the one structural check that needs no operator truth,
    so it must be produced for any clip that has been analysed."""
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "classified.json").write_text(json.dumps({"shots": [
        {"shot_id": 1, "t_sec": 1.0, "shot_type": "drive", "is_volley": True},
        {"shot_id": 2, "t_sec": 2.0, "shot_type": "drive", "is_volley": False},
        {"shot_id": 3, "t_sec": 3.0, "shot_type": "dink", "is_volley": False}]}),
        encoding="utf-8")
    (clip / "bounces.json").write_text(json.dumps({"bounces": [{"t_sec": 2.5}]}),
                                       encoding="utf-8")
    m = reg.measure(clip)
    assert m["volleys"] == 1 and m["bounces"] == 1
    assert m["identity_gap"] == 3 - (1 + 1) == 1


def test_baseline_file_is_committed_and_parseable():
    """The baseline is version-controlled on purpose: a number changing should show up in a
    diff someone reads, not only in a terminal someone ran."""
    p = Path(reg.BASELINE)
    assert p.exists(), f"{p} is missing — run: python -m tools.regression --save"
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["clips"], "baseline has no clips"
    for name, m in doc["clips"].items():
        assert "shots" in m, f"{name} baseline has no shot count"

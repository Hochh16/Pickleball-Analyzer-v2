"""Tests for the shot-type labelling tool's pure logic.

The Tk loop and the video decode only run interactively; what must not break is the part that
decides WHAT is offered and the part that persists a label -- losing an operator's labels is
the one unrecoverable failure here, since their time is the scarce input.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from tools import label_shot_types as lst


def _clip(tmp_path, shots):
    c = tmp_path / "clip"
    c.mkdir()
    (c / "classified.json").write_text(json.dumps({"shots": shots}), encoding="utf-8")
    (c / "track_roles.json").write_text(json.dumps(
        {"roles": {"user": {"track_ids": [7]}, "opp_a": {"track_ids": [9]}}}),
        encoding="utf-8")
    return c


def test_todo_is_in_time_order_and_carries_the_hitter(tmp_path):
    """Time order on purpose: a least-confident-first sample measures the hard cases and
    cannot be compared against the accuracy we quote over an arbitrary set."""
    c = _clip(tmp_path, [
        {"shot_id": 5, "frame": 900, "t_sec": 15.0, "track_id": 9, "hitter_side": "far"},
        {"shot_id": 2, "frame": 100, "t_sec": 1.67, "track_id": 7, "hitter_side": "near"}])
    todo = lst.build_todo(c, 60.0, None)
    assert [s["shot_id"] for s in todo] == [2, 5]
    assert todo[0]["role"] == "user" and todo[0]["side"] == "near"
    assert todo[1]["role"] == "opp_a"


def test_limit_offers_a_prefix(tmp_path):
    c = _clip(tmp_path, [{"shot_id": i, "frame": i * 100, "t_sec": i * 1.5,
                          "track_id": 7, "hitter_side": "near"} for i in range(6)])
    assert len(lst.build_todo(c, 60.0, 3)) == 3


def test_labels_round_trip_and_resume(tmp_path):
    """Saved after every keypress, so a session can stop anywhere. A resumed run must skip
    what is already there rather than offering it again."""
    out = tmp_path / "_labeling" / lst.OUT_NAME
    done = {4: {"shot_no": 4, "frame": 240, "shot_id": 4, "time": "00:04.00",
                "hitter_role": "user", "hitter_side": "near", "true_type": "drop",
                "true_volley": "", "true_in": "", "notes": ""}}
    lst.save_done(out, done)
    again = lst.load_done(out)
    assert again[4]["true_type"] == "drop"
    assert set(again[4]) >= set(lst.FIELDS)


def test_saved_file_is_readable_by_the_scorer(tmp_path):
    """It lands in the same schema and folder as the existing label files, so
    tools/score_shot_types picks it up with no change -- it globs labels*.csv."""
    out = tmp_path / "_labeling" / lst.OUT_NAME
    assert out.name.startswith("labels") and out.name.endswith(".csv")
    lst.save_done(out, {1: {"shot_id": 1, "frame": 60, "true_type": "dink"}})
    with out.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["true_type"] == "dink" and rows[0]["frame"] == "60"
    from tools.score_shot_types import load_labels
    got = load_labels(Path(out).parent.parent, 60.0)
    assert got and got[0]["true_type"] == "dink" and got[0]["frame"] == 60


def test_every_key_maps_to_a_type_the_scorer_knows(tmp_path):
    """A key that produces a label the scorer discards is a wasted keypress and a silently
    wasted minute of the operator's time."""
    from tools.score_shot_types import REAL_TYPES
    for k, v in lst.KEYS.items():
        assert len(k) == 1, f"{k!r} is not a single key"
        assert v in REAL_TYPES or v == "not a shot", f"{v!r} is neither a real type nor junk"
    assert len(set(lst.KEYS.values())) == len(lst.KEYS), "two keys map to the same type"

"""Tests for the shot-review workbook round-trip.

The operator's time is the scarce input, so the failure that matters is a correction being
written and then not making it into the truth set -- silently, since a missing label looks
exactly like a shot nobody reviewed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools import shot_review_sheet as srs


def _clip(tmp_path):
    c = tmp_path / "clip"
    (c / "_labeling").mkdir(parents=True)
    (c / "classified.json").write_text(json.dumps({"shots": [
        {"shot_id": 0, "frame": 60, "t_sec": 1.0, "track_id": 7,
         "hitter_side": "near", "shot_type": "drive", "is_volley": False},
        {"shot_id": 1, "frame": 300, "t_sec": 5.0, "track_id": 9,
         "hitter_side": "far", "shot_type": "dink", "is_volley": True}]}), encoding="utf-8")
    (c / "track_roles.json").write_text(json.dumps(
        {"roles": {"user": {"track_ids": [7]}, "opp_a": {"track_ids": [9]}}}), encoding="utf-8")
    (c / "court.json").write_text(json.dumps({"video": {"fps": 60.0}}), encoding="utf-8")
    return c


def test_rows_are_numbered_as_the_video_numbers_them(tmp_path):
    """The '#' in the sheet and the '#N' burned into the annotated video must be the same
    number, or a correction lands on the wrong shot. Both number shots sorted by FRAME."""
    rows = srs.rows_for(_clip(tmp_path))
    assert [r["n"] for r in rows] == [1, 2]
    assert rows[0]["hitter"] == "user" and rows[0]["our_type"] == "drive"
    assert rows[1]["our_volley"] == "yes"


def test_build_refuses_to_clobber_a_filled_in_review(tmp_path, monkeypatch):
    """The operator's time is the scarce input. A rebuild on top of a completed review would
    destroy hours of it silently -- which nearly happened, caught only because Excel had the
    file locked."""
    import sys
    from openpyxl import load_workbook
    c = _clip(tmp_path)
    out = c / "_labeling" / srs.OUT_NAME
    srs.build(c, out)
    wb = load_workbook(out)
    ws = wb.active
    hdr = next(r for r in range(1, 20) if ws.cell(row=r, column=1).value == "#")
    ws.cell(row=hdr + 2, column=8, value="drop")
    wb.save(out)
    monkeypatch.setattr(sys, "argv", ["x", str(c)])
    with pytest.raises(SystemExit) as e:
        srs.main([str(c)])
    assert "Refusing to overwrite" in str(e.value)
    # --force gets through
    srs.main([str(c), "--force"])


def test_blank_correction_means_agree(tmp_path):
    """Silence is agreement. A reviewed shot we got right is still a label, and dropping it
    would quietly bias the truth set towards our own mistakes."""
    c = _clip(tmp_path)
    out = c / "_labeling" / srs.OUT_NAME
    srs.build(c, out)
    srs.score(c, out)
    with (c / "_labeling" / srs.CSV_NAME).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["true_type"] for r in rows] == ["drive", "dink"]


def test_a_correction_overrides_and_a_missed_shot_is_added(tmp_path):
    from openpyxl import load_workbook
    c = _clip(tmp_path)
    out = c / "_labeling" / srs.OUT_NAME
    srs.build(c, out)
    wb = load_workbook(out)
    ws = wb.active
    hdr = next(r for r in range(1, 20) if ws.cell(row=r, column=1).value == "#")
    # CORRECT_TYPE is column H: an ALREADY KNOWN column sits at G so a shot reviewed once is
    # not put in front of the operator again.
    ws.cell(row=hdr + 2, column=8, value="drop")          # correct the first real row
    blank = next(r for r in range(hdr, ws.max_row + 1)
                 if str(ws.cell(row=r, column=1).value or "").startswith("SHOTS WE MISSED"))
    ws.cell(row=blank + 1, column=2, value="0:09.50")     # a shot we never detected
    ws.cell(row=blank + 1, column=8, value="lob")
    wb.save(out)
    srs.score(c, out)
    with (c / "_labeling" / srs.CSV_NAME).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["true_type"] == "drop", "the correction did not override our type"
    missed = [r for r in rows if "MISSED" in r["notes"]]
    assert len(missed) == 1 and missed[0]["true_type"] == "lob"
    assert int(missed[0]["frame"]) == int(round(9.5 * 60)), "missed shot lost its timestamp"


def test_output_name_is_globbed_by_the_scorer():
    """score_shot_types reads every labels*.csv in _labeling; a name outside that pattern
    would make the whole review invisible."""
    assert srs.CSV_NAME.startswith("labels") and srs.CSV_NAME.endswith(".csv")


def test_clock_round_trips():
    for t in (0.0, 9.5, 63.82, 301.05):
        assert abs(srs.parse_clock(srs.clock(t)) - t) < 0.01

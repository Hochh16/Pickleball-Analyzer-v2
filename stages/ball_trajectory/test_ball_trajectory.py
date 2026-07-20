"""Smoke + unit tests for Stage 5.7 ball_trajectory (Phase 1)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stages.ball_trajectory import ball_trajectory as bt

ROOT = Path(__file__).resolve().parents[2]


# --- physical filters --------------------------------------------------------

def test_crosses_net_opposite_sides():
    assert bt.crosses_net(3.0, 30.0)      # near hitter -> far landing
    assert bt.crosses_net(30.0, 8.0)      # far hitter -> near landing


def test_crosses_net_same_side_rejected():
    assert not bt.crosses_net(3.0, 8.0)   # both on the near half = mis-assigned
    assert not bt.crosses_net(35.0, 30.0)


def test_crosses_net_at_net_allowed():
    assert bt.crosses_net(3.0, 21.5)      # lands right at the net (within tol)


def test_anchor_ok_range_cap():
    # range under court length + crosses net -> ok
    assert bt.anchor_ok((5.0, 3.0), (6.0, 30.0), 27.0, 44.0)
    # impossible range -> rejected even if it crosses the net
    assert not bt.anchor_ok((5.0, 3.0), (6.0, 30.0), 55.0, 44.0)


# --- landing index -----------------------------------------------------------

def test_build_landing_index_first_bounce_wins():
    bounces = [
        {"frame": 120, "between_shots": [4, 5], "court_xy_ft": [6.0, 30.0]},
        {"frame": 100, "between_shots": [4, 5], "court_xy_ft": [5.0, 29.0]},  # earlier
        {"frame": 140, "between_shots": [None, 0], "court_xy_ft": [1.0, 1.0]},  # no prev
    ]
    idx = bt.build_landing_index(bounces)
    assert idx[4]["frame"] == 100          # earliest bounce for shot 4
    assert 0 not in idx                     # between_shots[0] is None -> skipped


# --- compute -----------------------------------------------------------------

def _identity_M():
    return np.eye(3)  # image px == court ft for the synthetic case


def test_compute_bounce_and_volley():
    M = _identity_M()
    # shot 0 bounces (near hitter y=3 -> far bounce y=31); shot 1 volleyed (no
    # bounce, next contact on the far side).
    shots = [
        {"shot_id": 0, "frame": 0, "track_id": 1, "is_serve": False},
        {"shot_id": 1, "frame": 60, "track_id": 2, "is_serve": False},
        {"shot_id": 2, "frame": 90, "track_id": 1, "is_serve": False},
    ]
    landing = {0: {"frame": 30, "court_xy_ft": [5.0, 31.0]}}
    # players give ground foot positions directly (identity homography)
    players = {
        (0, 1): {"cx": 5.0, "cy": 3.0, "reliable": True},
        (60, 2): {"cx": 5.0, "cy": 35.0, "reliable": True},
        (90, 1): {"cx": 5.0, "cy": 5.0, "reliable": True},
    }
    params = {"min_airtime_s": 0.10, "max_range_ft": 44.0, "max_volley_gap_s": 1.5}
    import logging
    log = logging.getLogger("t"); log.addHandler(logging.NullHandler())
    res, stats = bt.compute(shots, landing, players, {}, M, 60.0, params, log)
    # shot 0: bounce anchor, range = 28 ft over 0.5 s = 56 ft/s
    assert res[0]["anchor_type"] == "bounce"
    assert res[0]["horizontal_speed_ftps"] == pytest.approx(56.0, abs=0.1)
    # shot 1: no bounce -> next_contact (shot 2 at frame 90). y 35 -> 5 crosses net.
    assert res[1]["anchor_type"] == "next_contact"
    # shot 2: last shot, no bounce, no next -> none
    assert res[2]["anchor_type"] == "none"
    assert res[2]["horizontal_speed_ftps"] is None


def test_compute_rejects_same_side_bounce():
    M = _identity_M()
    shots = [{"shot_id": 0, "frame": 0, "track_id": 1, "is_serve": False}]
    landing = {0: {"frame": 30, "court_xy_ft": [5.0, 6.0]}}  # same (near) side
    players = {(0, 1): {"cx": 5.0, "cy": 3.0, "reliable": True}}
    params = {"min_airtime_s": 0.10, "max_range_ft": 44.0, "max_volley_gap_s": 1.5}
    import logging
    log = logging.getLogger("t2"); log.addHandler(logging.NullHandler())
    res, _ = bt.compute(shots, landing, players, {}, M, 60.0, params, log)
    assert res[0]["anchor_type"] == "none"  # mis-assigned bounce rejected


# --- end-to-end on the drill fixture ----------------------------------------

@pytest.mark.skipif(not (ROOT / "data/pb_5min_test_20s-7/trajectory.json").exists()
                    and not (ROOT / "data/pb_5min_test_20s-7/shots.json").exists(),
                    reason="drill fixture not present")
def test_end_to_end_schema():
    clip = ROOT / "data/pb_5min_test_20s-7"
    r = subprocess.run([sys.executable, "-m",
                        "stages.ball_trajectory.ball_trajectory", str(clip), "--force"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    doc = json.load(open(clip / "trajectory.json"))
    assert doc["schema_version"] == 1 and doc["phase"] == 1
    shots = json.load(open(clip / "shots.json"))["shots"]
    assert len(doc["shots"]) == len(shots)
    for s in doc["shots"]:
        assert set(s) >= {"shot_id", "horizontal_speed_ftps", "anchor_type", "confidence"}
        if s["horizontal_speed_ftps"] is not None:
            assert 0 <= s["horizontal_speed_ftps"] < 200  # physical
            assert s["range_ft"] <= 44.0 + 1e-6

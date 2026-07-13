"""Smoke tests for the setup-wizard backend (Phase 1).

Self-contained: synthesizes a tiny video in a temp dir (no dependency on the
gitignored data/ folder), then exercises SessionStore end-to-end — the same
path the FastAPI routes call. Validates that the wizard produces the exact
input JSONs the pipeline consumes.

Run:  pytest app/test_app.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import video as video_mod
from app.sessions import SessionError, SessionStore, _default_name, slugify


def _make_video(path: Path, n_frames: int = 30, w: int = 320, h: int = 240, fps: int = 30) -> bool:
    """Write a tiny synthetic mp4. Returns False if no codec is available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return False
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 3 % 255, dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (280, 200), (0, 200, 0), 2)
        writer.write(frame)
    writer.release()
    return path.exists() and path.stat().st_size > 0


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "data")


@pytest.fixture
def video(tmp_path):
    vp = tmp_path / "clips" / "match.mp4"
    if not _make_video(vp):
        pytest.skip("No OpenCV video codec available to synthesize a fixture")
    return vp


# valid trapezoid (image-position order: BL, BR, TR, TL) + kitchen lines
VALID_MARKERS = {
    "court_corners_image": [[60, 200], [260, 200], [230, 60], [90, 60]],
    "kitchen_line_user_image": [[75, 150], [245, 150]],
    "kitchen_line_opponent_image": [[85, 90], [235, 90]],
    "user_baseline": "near",
    "dominant_hand": "right",
    "user_starting_corner": "left",
    "frame_used_for_calibration": 5,
}


def test_probe(video):
    meta = video_mod.probe(video)
    assert meta["frame_width"] == 320 and meta["frame_height"] == 240
    assert meta["frame_count"] >= 25
    assert meta["fps"] > 0 and meta["duration_sec"] > 0


def test_frame_serving(video):
    jpeg = video_mod.frame_server.frame_jpeg(video, 3, max_w=160)
    assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI
    # downscaled to <= max_w
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert arr.shape[1] <= 160
    video_mod.frame_server.release(video)


def test_name_derivation():
    assert _default_name(Path("data/pb_2min/video.mp4")) == "pb_2min"  # generic stem -> parent
    assert _default_name(Path("clips/my_match.mp4")) == "my_match"
    assert slugify("Sat AM game #3!") == "sat_am_game_3"


def test_full_setup_flow(store, video):
    session = store.create_from_path(video)
    sid = session["id"]
    assert session["video"]["frame_width"] == 320
    assert session["source"] == "local"

    # calibrate -> court.json / court_zones.json
    res = store.calibrate(sid, VALID_MARKERS)
    assert res["validation"]["homography_rmse_pixels"] < 5.0
    assert res["preview_jpeg_base64"]
    folder = store.folder(sid)
    court = json.loads((folder / "court.json").read_text())
    assert court["schema_version"] == 1
    assert court["user_inputs"]["dominant_hand"] == "right"
    assert court["video"]["frame_used_for_calibration"] == 5
    assert (folder / "court_zones.json").exists()

    # roster.json with current role vocabulary
    store.write_roster(sid, {"user": "right", "partner": "left", "opp_a": "unknown", "opp_b": "right"})
    roster = json.loads((folder / "roster.json").read_text())
    assert set(roster["handedness"]) == {"user", "partner", "opp_a", "opp_b"}
    assert roster["handedness"]["partner"] == "left"

    # user_clicks.json
    store.write_user_clicks(sid, [{"frame": 2, "x": 100, "y": 120}, {"frame": 20, "x": 80, "y": 90}])
    clicks = json.loads((folder / "user_clicks.json").read_text())
    assert len(clicks["clicks"]) == 2
    assert clicks["clicks"][0]["frame"] == 2  # sorted by frame

    # summary reflects all steps
    summ = store.summary(sid)
    assert summ["session"]["steps"] == {"calibration": True, "roster": True, "user_clicks": True}
    assert summ["user_clicks_count"] == 2


def test_user_clicks_skip_removes_file(store, video):
    sid = store.create_from_path(video)["id"]
    store.write_user_clicks(sid, [{"frame": 1, "x": 1, "y": 1}])
    assert (store.folder(sid) / "user_clicks.json").exists()
    # empty = skipped -> file removed, geometric seed used downstream
    store.write_user_clicks(sid, [])
    assert not (store.folder(sid) / "user_clicks.json").exists()
    assert store.get(sid)["steps"]["user_clicks"] is False


def test_bad_markers_rejected(store, video):
    sid = store.create_from_path(video)["id"]
    bad = dict(VALID_MARKERS, court_corners_image=[[0, 0], [1, 1]])  # only 2 corners
    with pytest.raises(SessionError):
        store.calibrate(sid, bad)


def test_bad_handedness_rejected(store, video):
    sid = store.create_from_path(video)["id"]
    with pytest.raises(SessionError):
        store.write_roster(sid, {"user": "sideways"})


def test_missing_video_rejected(store):
    with pytest.raises(SessionError):
        store.create_from_path(Path("does/not/exist.mp4"))

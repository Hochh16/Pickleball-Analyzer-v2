"""Smoke tests for the Phase 2 pipeline runner (state machine + plumbing).

The heavy real stages (torch/mediapipe subprocesses) are replaced with a fast
fake so the orchestration logic is exercised in ~a second. One test drives a
REAL subprocess (`python -m this`) to prove the stdout->log plumbing works.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from app import pipeline as pipe
from app.pipeline import PipelineRunner
from app.sessions import SessionStore
from app.test_app import VALID_MARKERS, _make_video


@pytest.fixture
def store_with_session(tmp_path):
    store = SessionStore(tmp_path / "data")
    vp = tmp_path / "clips" / "match.mp4"
    if not _make_video(vp):
        pytest.skip("No OpenCV video codec available")
    sid = store.create_from_path(vp)["id"]
    store.calibrate(sid, VALID_MARKERS)  # writes court.json
    return store, sid


def _wait(job, phases, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if job.phase in phases:
            return True
        time.sleep(0.05)
    return False


def _fake_module(record, fail_key=None):
    def run(job, module, folder):
        record.append(module)
        job.log.append(f"[fake] {module}")
        return 1 if (fail_key and fail_key in module) else 0
    return run


def test_pre_then_pause_then_resume(store_with_session):
    store, sid = store_with_session
    runner = PipelineRunner(store)
    calls = []
    runner._run_module = _fake_module(calls)

    job = runner.start(sid)
    assert _wait(job, ("ball",)), f"expected ball pause, got {job.phase}"

    # video.mp4 was materialized into the folder
    assert (store.folder(sid) / "video.mp4").exists()
    # pre steps ran and are done; ball is waiting
    statuses = {s["key"]: s["status"] for s in job.steps}
    assert statuses["track"] == "done" and statuses["pose"] == "done"
    assert statuses["ball"] == "waiting"
    assert any("track_players" in m for m in calls)

    # drop ball.parquet and resume
    (store.folder(sid) / "ball.parquet").write_bytes(b"PAR1")
    runner.resume_post(sid)
    assert _wait(job, ("done",)), f"expected done, got {job.phase} ({job.error})"
    statuses = {s["key"]: s["status"] for s in job.steps}
    assert statuses["ball"] == "done"
    assert statuses["report"] == "done"
    assert any("build_report" in m for m in calls)


def test_stage_failure_stops_run(store_with_session):
    store, sid = store_with_session
    runner = PipelineRunner(store)
    runner._run_module = _fake_module([], fail_key="classify_tracks")  # Stage 2.5 fails

    job = runner.start(sid)
    assert _wait(job, ("failed",)), f"expected failed, got {job.phase}"
    statuses = {s["key"]: s["status"] for s in job.steps}
    assert statuses["roles"] == "failed"
    assert statuses["pose"] == "pending"       # never reached
    assert job.error and "Identify players" in job.error


def test_real_subprocess_plumbing(store_with_session):
    store, sid = store_with_session
    runner = PipelineRunner(store)
    job = pipe.Job(sid)
    # `python -m this` prints the Zen of Python and exits 0 — proves stdout->log
    rc = runner._run_module(job, "this", store.folder(sid))
    assert rc == 0
    assert len(job.log) > 5
    assert any("Beautiful" in line for line in job.log)


def test_version_bumps_on_progress(store_with_session):
    store, sid = store_with_session
    runner = PipelineRunner(store)
    runner._run_module = _fake_module([])
    job = runner.start(sid)
    assert _wait(job, ("ball",))
    assert job.version > 3  # multiple state changes emitted for SSE

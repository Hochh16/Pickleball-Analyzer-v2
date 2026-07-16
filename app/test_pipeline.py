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
from app.pipeline import PipelineRunner, VISION_OUTPUTS
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
    def run(job, module, folder, extra_args=None):
        record.append((module, tuple(extra_args or ())))
        job.log.append(f"[fake] {module}")
        return 1 if (fail_key and fail_key in module) else 0
    return run


def _drop_vision_outputs(folder: Path):
    for name in VISION_OUTPUTS:
        (folder / name).write_bytes(b"PAR1" if name.endswith(".parquet") else b"{}")


def test_prepare_then_vision_handoff_then_resume(store_with_session, monkeypatch):
    store, sid = store_with_session
    monkeypatch.setattr(pipe, "_cuda_available", lambda: False)  # force the hand-off path
    runner = PipelineRunner(store)
    calls = []
    runner._run_module = _fake_module(calls)

    job = runner.start(sid)
    assert _wait(job, ("vision",)), f"expected vision hand-off, got {job.phase}"
    # video materialized; vision steps waiting for the GPU outputs
    assert (store.folder(sid) / "video.mp4").exists()
    statuses = {s["key"]: s["status"] for s in job.steps}
    assert statuses["video"] == "done"
    assert statuses["track"] == "waiting" and statuses["ball"] == "waiting"

    # drop the vision outputs (as if uploaded from Colab) and resume
    _drop_vision_outputs(store.folder(sid))
    runner.resume_post(sid)
    assert _wait(job, ("done",)), f"expected done, got {job.phase} ({job.error})"
    statuses = {s["key"]: s["status"] for s in job.steps}
    assert statuses["ball"] == "done" and statuses["report"] == "done"
    assert any("build_report" in m for m, _ in calls)


def test_local_gpu_runs_vision_locally(store_with_session, monkeypatch):
    store, sid = store_with_session
    monkeypatch.setattr(pipe, "_cuda_available", lambda: True)  # pretend a GPU is present
    runner = PipelineRunner(store)
    calls = []
    runner._run_module = _fake_module(calls)

    job = runner.start(sid)
    assert _wait(job, ("done",)), f"expected done, got {job.phase} ({job.error})"
    mods = [m for m, _ in calls]
    assert any("track_players" in m for m in mods)      # vision ran locally
    assert any("pose.pose" in m for m in mods)
    assert any("build_report" in m for m in mods)


def test_annotated_render_is_skipped(store_with_session, monkeypatch):
    """The Stage-11 overlay render + compress are intentionally omitted (the box
    overlay added little); the report links the original clip. Report still builds."""
    store, sid = store_with_session
    monkeypatch.setattr(pipe, "_cuda_available", lambda: True)
    runner = PipelineRunner(store)
    calls = []
    runner._run_module = _fake_module(calls)
    job = runner.start(sid)
    assert _wait(job, ("done",))
    mods = [m for m, _ in calls]
    assert not any("render.render" in m for m in mods)
    assert not any("compress_video" in m for m in mods)
    assert any("build_report" in m for m in mods)
    # the pipeline no longer defines render/compress steps at all
    assert {"render", "compress"}.isdisjoint(s["key"] for s in job.steps)


def test_stage_failure_stops_run(store_with_session, monkeypatch):
    store, sid = store_with_session
    monkeypatch.setattr(pipe, "_cuda_available", lambda: True)
    runner = PipelineRunner(store)
    runner._run_module = _fake_module([], fail_key="detect_shots")  # first post stage fails

    job = runner.start(sid)
    assert _wait(job, ("failed",)), f"expected failed, got {job.phase}"
    statuses = {s["key"]: s["status"] for s in job.steps}
    assert statuses["shots"] == "failed"
    assert statuses["report"] == "pending"
    assert job.error and "Detect shots" in job.error


def test_real_subprocess_plumbing(store_with_session):
    store, sid = store_with_session
    runner = PipelineRunner(store)
    job = pipe.Job(sid)
    rc = runner._run_module(job, "this", store.folder(sid))  # `python -m this` -> Zen
    assert rc == 0
    assert any("Beautiful" in line for line in job.log)

"""Tests for the Colab vision orchestration helpers (the non-Colab-only logic).

The subprocess/GPU/Drive-mount paths only run on Colab; here we cover the pure
logic that decides WHAT runs: clip auto-detection, resume/restore, and the stage
table's integrity.
"""
from __future__ import annotations

import json

import pytest

from tools import colab_vision as cv


# ---- derive_clip (B1) ----

def test_derive_clip_single(tmp_path):
    (tmp_path / "pb_5_minute_outdoor_vision_input.zip").write_bytes(b"z")
    assert cv.derive_clip(tmp_path) == "pb_5_minute_outdoor"


def test_derive_clip_forced_wins(tmp_path):
    (tmp_path / "whatever_vision_input.zip").write_bytes(b"z")
    assert cv.derive_clip(tmp_path, forced="myclip") == "myclip"


def test_derive_clip_none_present(tmp_path):
    with pytest.raises(SystemExit) as e:
        cv.derive_clip(tmp_path)
    assert "No" in str(e.value)


def test_derive_clip_multiple_errors(tmp_path):
    (tmp_path / "a_vision_input.zip").write_bytes(b"z")
    (tmp_path / "b_vision_input.zip").write_bytes(b"z")
    with pytest.raises(SystemExit) as e:
        cv.derive_clip(tmp_path)
    assert "Multiple" in str(e.value)


# ---- upload race (BadZipFile guard) ----

def test_wait_for_complete_zip(tmp_path):
    import zipfile as zf
    partial = tmp_path / "partial.zip"
    partial.write_bytes(b"still uploading, not a zip")
    with pytest.raises(SystemExit):                       # never completes -> clear error
        cv.wait_for_complete_zip(partial, tries=2, wait=0)
    good = tmp_path / "good.zip"
    with zf.ZipFile(good, "w") as z:
        z.writestr("video.mp4", "x")
    cv.wait_for_complete_zip(good, tries=1, wait=0)       # complete -> returns


# ---- restore / resume (B4) ----

def test_restore_outputs_copies_present(tmp_path):
    backup = tmp_path / "clip_outputs"
    backup.mkdir()
    (backup / "players.parquet").write_bytes(b"P")
    (backup / "track_roles.json").write_text("{}")
    clip_dir = tmp_path / "clip"
    restored = cv.restore_outputs(backup, clip_dir)
    assert set(restored) == {"players.parquet", "track_roles.json"}
    assert (clip_dir / "players.parquet").exists()
    assert not (clip_dir / "ball.parquet").exists()


def test_restore_outputs_no_backup_dir(tmp_path):
    assert cv.restore_outputs(tmp_path / "missing", tmp_path / "clip") == []


def test_have_all_required(tmp_path):
    clip = tmp_path / "clip"
    clip.mkdir()
    for f in cv.REQUIRED_OUTPUTS[:-1]:
        (clip / f).write_bytes(b"x")
    assert cv.have_all_required(clip) is False
    (clip / cv.REQUIRED_OUTPUTS[-1]).write_bytes(b"x")
    assert cv.have_all_required(clip) is True


# ---- stage table integrity ----

def test_stage_table_covers_required_outputs():
    # every required output is produced by exactly one stage, and each stage's
    # `required` is one of its `outputs`
    produced = {}
    for s in cv.STAGES:
        assert s["required"] in s["outputs"]
        for o in s["outputs"]:
            produced.setdefault(o, 0)
            produced[o] += 1
    for req in cv.REQUIRED_OUTPUTS:
        assert produced.get(req) == 1, f"{req} not produced by exactly one stage"
    # ball is the only GPU-batched stage
    assert [s["name"] for s in cv.STAGES if s.get("gpu_batch")] == ["ball"]


def test_notebook_builds_as_git_bootstrapper():
    from tools import build_vision_nb as gen
    cells = gen.build_cells()
    # valid JSON round-trip (nbformat sanity)
    nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 0}
    json.loads(json.dumps(nb))
    src = "".join("".join(c["source"]) for c in cells)
    assert "git" in src and "clone" in src            # pulls code from GitHub
    assert gen.REPO_URL in src
    assert "from tools.colab_vision import run_all" in src
    assert "run_all(REPO, clip=CLIP)" in src          # runs from the cloned repo
    assert "%%writefile" not in src                   # no embedded bundle anymore

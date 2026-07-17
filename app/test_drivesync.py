"""Tests for the Drive-for-Desktop auto-sync adapter (pure file logic)."""
from __future__ import annotations

from pathlib import Path

from app.drivesync import DriveSync, detect_drive_dir, INPUT_SUFFIX
from app.pipeline import VISION_OUTPUTS


def _touch(p: Path, data: bytes = b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_detect_drive_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PB_DRIVE_DIR", str(tmp_path))
    assert detect_drive_dir() == tmp_path
    monkeypatch.setenv("PB_DRIVE_DIR", str(tmp_path / "nope"))
    assert detect_drive_dir() is None


def test_disabled_when_no_dir():
    assert DriveSync(None).enabled() is False


def test_push_bundle_replaces_stale(tmp_path):
    drive = tmp_path / "MyDrive"
    drive.mkdir()
    # a stale bundle from a prior clip must be removed so the notebook sees exactly one
    _touch(drive / f"oldclip{INPUT_SUFFIX}")
    bundle = tmp_path / "src" / "new.zip"
    _touch(bundle, b"ZIP")
    ds = DriveSync(drive)
    dest = ds.push_bundle("newclip", bundle)
    assert dest.name == f"newclip{INPUT_SUFFIX}"
    present = sorted(p.name for p in drive.glob(f"*{INPUT_SUFFIX}"))
    assert present == [f"newclip{INPUT_SUFFIX}"]        # stale gone, new present
    assert dest.read_bytes() == b"ZIP"


def test_push_bundle_skips_when_already_synced(tmp_path):
    """Re-pushing the same-size bundle must NOT rewrite the synced file (that would
    make Drive re-upload multi-GB after every app restart)."""
    drive = tmp_path / "MyDrive"
    drive.mkdir()
    bundle = tmp_path / "b.zip"
    bundle.write_bytes(b"ZIPDATA")           # 7 bytes
    ds = DriveSync(drive)
    dest = ds.push_bundle("clip", bundle)
    dest.write_bytes(b"AAAAAAA")             # same size sentinel — proves skip below
    ds.push_bundle("clip", bundle)
    assert dest.read_bytes() == b"AAAAAAA"   # untouched: push was skipped


def test_outputs_ready_and_ingest(tmp_path):
    drive = tmp_path / "MyDrive"
    ds = DriveSync(drive)
    outs = ds.outputs_dir("clip")
    # not ready until ALL required outputs exist
    for f in VISION_OUTPUTS[:-1]:
        _touch(outs / f)
    assert ds.outputs_ready("clip") is False
    _touch(outs / VISION_OUTPUTS[-1])
    _touch(outs / "pose_summary.json")   # a sidecar too
    assert ds.outputs_ready("clip") is True

    session_folder = tmp_path / "data" / "clip"
    session_folder.mkdir(parents=True)
    got = ds.ingest_outputs("clip", session_folder)
    for f in VISION_OUTPUTS:
        assert (session_folder / f).exists()
    assert "pose_summary.json" in got   # sidecar carried across

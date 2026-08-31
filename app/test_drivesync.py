"""Tests for the Drive-for-Desktop auto-sync adapter (pure file logic)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app import drivesync as ds_mod
from app.drivesync import DriveSync, detect_drive_dir, INPUT_SUFFIX
from app.pipeline import VISION_OUTPUTS


def _touch(p: Path, data: bytes = b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _make_zip(p: Path, payload: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("video.mp4", payload)
    return p


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
    bundle = _make_zip(tmp_path / "src" / "new.zip")
    ds = DriveSync(drive)
    dest = ds.push_bundle("newclip", bundle)
    assert dest.name == f"newclip{INPUT_SUFFIX}"
    present = sorted(p.name for p in drive.glob(f"*{INPUT_SUFFIX}"))
    assert present == [f"newclip{INPUT_SUFFIX}"]        # stale gone, new present
    assert zipfile.is_zipfile(dest)                     # published copy verified


def test_push_bundle_skips_when_already_synced(tmp_path):
    """Re-pushing the same complete bundle must NOT rewrite the synced file (that
    would make Drive re-upload multi-GB after every app restart)."""
    import time
    drive = tmp_path / "MyDrive"
    drive.mkdir()
    bundle = _make_zip(tmp_path / "b.zip")
    ds = DriveSync(drive)
    dest = ds.push_bundle("clip", bundle)
    m1 = dest.stat().st_mtime_ns
    time.sleep(0.05)
    ds.push_bundle("clip", bundle)
    assert dest.stat().st_mtime_ns == m1     # untouched: push was skipped


def test_push_bundle_rejects_truncated_copy(tmp_path, monkeypatch):
    """DriveFS has been observed to drop write data without erroring — a truncated
    copy must never be published under the real bundle name."""
    drive = tmp_path / "MyDrive"
    drive.mkdir()
    bundle = _make_zip(tmp_path / "b.zip", payload="full content")

    def truncating_copy(src, dst):
        data = Path(src).read_bytes()
        Path(dst).write_bytes(data[: len(data) // 2])   # silently drop half

    monkeypatch.setattr(ds_mod.shutil, "copyfile", truncating_copy)
    ds = DriveSync(drive)
    with pytest.raises(RuntimeError, match="truncated|complete"):
        ds.push_bundle("clip", bundle)
    assert not (drive / f"clip{INPUT_SUFFIX}").exists()          # nothing published
    assert not (drive / f"clip{INPUT_SUFFIX}.part").exists()     # temp cleaned up


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


from app.drivesync import DriveSync, _readable_zip


def _zip(path: Path, payload: bytes = b"x" * 1024) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("video.mp4", payload)
    return path


def test_a_bundle_already_synced_is_not_recopied(tmp_path):
    """Re-running a clip must not push multi-GB again. The 5-minute bundles are 4.7 GB."""
    drive = tmp_path / "My Drive"
    drive.mkdir()
    src = _zip(tmp_path / "s_vision_input.zip")
    ds = DriveSync(drive)
    first = ds.push_bundle("s", src)
    stamp = first.stat().st_mtime_ns
    again = ds.push_bundle("s", src)
    assert again == first and again.stat().st_mtime_ns == stamp, "should have skipped the copy"


def test_pushing_one_bundle_clears_the_others(tmp_path):
    """The Colab notebook auto-detects the clip from the SINGLE bundle on Drive and refuses
    to guess when there are several."""
    drive = tmp_path / "My Drive"
    drive.mkdir()
    _zip(drive / "old_vision_input.zip")
    ds = DriveSync(drive)
    ds.push_bundle("new", _zip(tmp_path / "new_vision_input.zip"))
    assert [p.name for p in drive.glob("*_vision_input.zip")] == ["new_vision_input.zip"]


def test_a_failed_readback_is_reported_as_such_not_as_truncation(tmp_path, monkeypatch):
    """Observed on a 2.24 GB bundle: three back-to-back copies all reported
    "truncated/corrupt (2243191656/2243191656 bytes)" -- byte-identical sizes, so nothing was
    truncated. Only the zip READ-BACK failed, because Drive was still uploading the file.
    Saying "truncated" sent the diagnosis in exactly the wrong direction.
    """
    drive = tmp_path / "My Drive"
    drive.mkdir()
    src = _zip(tmp_path / "s_vision_input.zip")
    monkeypatch.setattr("app.drivesync._readable_zip", lambda p: False)
    monkeypatch.setattr("app.drivesync.VERIFY_BACKOFF_S", (0.0, 0.0))
    ds = DriveSync(drive)
    try:
        ds.push_bundle("s", src)
        raise AssertionError("should have raised")
    except RuntimeError as e:
        msg = str(e)
    assert "right size" in msg and "still be uploading" in msg, msg
    assert "truncated" not in msg, "the size matched, so nothing was truncated"


def test_readable_zip_swallows_a_read_error(tmp_path):
    """A read error on a Drive virtual filesystem means the upload is in flight, not damage."""
    assert _readable_zip(_zip(tmp_path / "ok.zip")) is True
    assert _readable_zip(tmp_path / "does_not_exist.zip") is False

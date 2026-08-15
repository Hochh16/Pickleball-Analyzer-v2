"""Google Drive for Desktop auto-sync for the vision hand-off.

The app can't push to the user's Drive (security boundary), but the user's OWN
Google Drive for Desktop client can. So when a synced `My Drive` folder is present,
the app:
  - **writes the clip bundle into it** (Drive Desktop uploads it), and
  - **watches `<clip>_outputs/`** for the vision results (Drive Desktop downloads
    them as individual files — no zip), ingesting + auto-resuming the moment
    they're complete.

The operator's only action becomes "Run All" on Colab — no manual download,
upload, or unzip. Falls back to the manual buttons when no synced folder is
configured (`detect_drive_dir()` returns None).
"""
from __future__ import annotations

import os
import shutil
import string
import zipfile
from pathlib import Path
from typing import List, Optional

from .pipeline import VISION_OUTPUTS  # the 5 required outputs (readiness gate)

INPUT_SUFFIX = "_vision_input.zip"
# Everything to pull back: the required set + optional sidecars Colab also writes.
OUTPUT_FILES = tuple(VISION_OUTPUTS) + ("players_pending.json", "pose_summary.json")


def detect_drive_dir() -> Optional[Path]:
    """Locate the synced `My Drive` root: the `PB_DRIVE_DIR` override if set, else a
    `<letter>:\\My Drive` mounted by Google Drive for Desktop (Windows). Returns
    None if nothing is found (auto-sync stays off, manual flow remains)."""
    env = os.environ.get("PB_DRIVE_DIR")
    if env:
        p = Path(env)
        return p if p.exists() else None
    for letter in string.ascii_uppercase:
        p = Path(f"{letter}:\\") / "My Drive"
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


class DriveSync:
    """Thin adapter over a synced My Drive folder. All methods are no-ops of the
    caller's making unless `enabled()`."""

    def __init__(self, drive_dir: Optional[Path]):
        self.drive_dir = Path(drive_dir) if drive_dir else None

    def enabled(self) -> bool:
        return self.drive_dir is not None and self.drive_dir.exists()

    def outputs_dir(self, session_id: str) -> Path:
        assert self.drive_dir is not None
        return self.drive_dir / f"{session_id}_outputs"

    def push_bundle(self, session_id: str, bundle_path: Path) -> Path:
        """Copy the clip bundle into the synced folder (Drive uploads it), removing
        any other `*_vision_input.zip` so the notebook auto-detects exactly one.

        Writes via a `.part` temp name, VERIFIES the copy (size + zip end record),
        then renames into place — observed in the wild: DriveFS can drop write data
        without erroring, leaving a silently truncated file that poisons the cloud
        copy. A crashed or failed copy can never appear under the real name."""
        assert self.enabled() and self.drive_dir is not None
        keep = f"{session_id}{INPUT_SUFFIX}"
        for stale in self.drive_dir.glob(f"*{INPUT_SUFFIX}"):
            if stale.name != keep:
                try:
                    stale.unlink()
                except OSError:
                    pass
        dest = self.drive_dir / keep
        src_size = Path(bundle_path).stat().st_size
        # Same complete bundle already synced (e.g. a restarted run): skip the copy
        # so Drive doesn't re-upload multi-GB for nothing.
        if dest.exists() and dest.stat().st_size == src_size and zipfile.is_zipfile(str(dest)):
            return dest
        part = self.drive_dir / (keep + ".part")
        last = "unknown"
        try:
            for _ in range(3):
                shutil.copyfile(bundle_path, part)
                if part.stat().st_size == src_size and zipfile.is_zipfile(str(part)):
                    os.replace(part, dest)
                    return dest
                last = f"copy landed truncated/corrupt ({part.stat().st_size}/{src_size} bytes)"
        finally:
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
        raise RuntimeError(f"could not write a complete bundle to {dest}: {last}")

    def sync_model(self, model_path: Path) -> str:
        """Put the deployed ball model on Drive, replacing an out-of-date copy.

        The Colab vision pass uses whatever `ball_model_v4.pt` is in My Drive. When the
        app's model is updated and Drive's is not, clips are analysed by the OLDER
        detector — no error, just quietly worse numbers. Asking the operator to notice
        that and re-upload by hand is asking them to track something invisible, so the
        same channel that already carries the clip carries the model.

        Returns one of "synced" / "up-to-date" / "missing" / "unavailable" (never raises:
        a model that fails to copy must not block the run, it degrades to the manual
        instructions).
        """
        if not self.enabled() or self.drive_dir is None:
            return "unavailable"
        src = Path(model_path)
        if not src.is_file():
            return "missing"
        dest = self.drive_dir / "ball_model_v4.pt"
        size = src.stat().st_size
        # Size alone is a weak identity check for two checkpoints of the same
        # architecture, so compare a content prefix too.
        def head(p: Path) -> bytes:
            with p.open("rb") as f:
                return f.read(1 << 20)
        try:
            if dest.exists() and dest.stat().st_size == size and head(dest) == head(src):
                return "up-to-date"
            # Verify then rename, and RETRY: DriveFS drops write data without raising —
            # measured here, a 45,508,290-byte model landed as 45,088,768 with a
            # successful return. push_bundle already guards the same way. A truncated
            # model that reached the real name would be loaded by Colab and fail there,
            # or worse, quietly behave differently.
            part = self.drive_dir / "ball_model_v4.pt.part"
            try:
                for _ in range(4):
                    shutil.copyfile(src, part)
                    if part.stat().st_size == size:
                        os.replace(part, dest)
                        return "synced"
            finally:
                if part.exists():
                    try:
                        part.unlink()
                    except OSError:
                        pass
            return "unavailable"
        except OSError:
            return "unavailable"

    def outputs_ready(self, session_id: str) -> bool:
        """True once all REQUIRED outputs are present in the synced outputs dir."""
        d = self.outputs_dir(session_id)
        return d.is_dir() and all((d / f).exists() for f in VISION_OUTPUTS)

    def ingest_outputs(self, session_id: str, dest_folder: Path) -> List[str]:
        """Copy the vision outputs from the synced folder into the session folder.
        Returns the names copied (required + any sidecars present)."""
        d = self.outputs_dir(session_id)
        dest_folder = Path(dest_folder)
        got: List[str] = []
        for name in OUTPUT_FILES:
            src = d / name
            if src.exists():
                shutil.copyfile(src, dest_folder / name)
                got.append(name)
        return got

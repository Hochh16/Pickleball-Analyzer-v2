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
        any other `*_vision_input.zip` so the notebook auto-detects exactly one."""
        assert self.enabled() and self.drive_dir is not None
        keep = f"{session_id}{INPUT_SUFFIX}"
        for stale in self.drive_dir.glob(f"*{INPUT_SUFFIX}"):
            if stale.name != keep:
                try:
                    stale.unlink()
                except OSError:
                    pass
        dest = self.drive_dir / keep
        # Same bundle already in the synced folder (e.g. a restarted run): skip the
        # copy so Drive doesn't re-upload multi-GB for nothing.
        if dest.exists() and dest.stat().st_size == Path(bundle_path).stat().st_size:
            return dest
        shutil.copyfile(bundle_path, dest)
        return dest

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

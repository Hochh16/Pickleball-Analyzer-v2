"""Build `data/pb_vision_upload.zip` — the code bundle for the combined vision
Colab pass (`stages/infer_vision.ipynb`).

Moving the heavy vision stages (2 track_players, 2.5 classify_tracks, 3 pose,
4 track_ball) off the local CPU onto a Colab GPU is what makes real-length clips
practical (local CPU is ~1 fps for tracking → hours per 5-min clip; KNOWN_ISSUES
"local CPU can't process real clips"). The notebook runs the REAL committed stage
modules from this bundle (no code duplication/drift) — `python -m stages.<x>.<x>
<clip_folder>` — with YOLO auto-using the GPU and pose on Colab's faster CPU.

The bundle is just the (small) stage source; both models auto-download on Colab
(`yolo11s.pt` via ultralytics, the MediaPipe pose model from Google), and the ball
weights (`ball_model_v4.pt`) already live in Drive root.

Usage:
    python tools/build_vision_bundle.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pb_vision_upload.zip"

# The exact stage code needed to run Stages 2 → 2.5 → 3 → 4 as modules on Colab.
# (Confirmed self-contained: only track_ball_v4 imports a sibling, _tracknet_model.)
REPO_FILES = [
    "stages/__init__.py",
    "stages/track_players/__init__.py",
    "stages/track_players/track.py",
    "stages/classify_tracks/__init__.py",
    "stages/classify_tracks/classify_tracks.py",
    "stages/pose/__init__.py",
    "stages/pose/pose.py",
    "stages/track_ball/__init__.py",
    "stages/track_ball/_tracknet_model.py",
    "stages/track_ball/track_ball_v4.py",
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    missing = [rf for rf in REPO_FILES if not (ROOT / rf).exists()]
    if missing:
        raise SystemExit(f"missing repo files: {missing}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for rf in REPO_FILES:
            z.write(ROOT / rf, rf)   # arcname keeps the stages/ layout at zip root
    kb = OUT.stat().st_size // 1024
    print(f"wrote {OUT} ({kb} KB, {len(REPO_FILES)} files)")
    print("Upload it to Drive root as MyDrive/pb_vision_upload.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

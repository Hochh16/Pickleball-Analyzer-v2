"""Build `stages/infer_vision.ipynb` — the combined GPU (Colab) vision pass.

Runs Stages **2 (track_players) → 2.5 (classify_tracks) → 3 (pose) →
4 (track_ball)** in one Colab GPU trip, so the heavy vision work is off the local
CPU (which is ~1 fps for tracking → hours per 5-min clip). It runs the REAL
committed stage modules from `pb_vision_upload.zip` (built by
`tools/build_vision_bundle.py`) — no code duplication — writing
`players.parquet` / `track_roles.json` / `poses.parquet` / `ball.parquet` (+ metas)
back to Drive. Local then only runs the light analytical stages (5–11 + report).

"Run All, set CLIP." Built by `tools/build_vision_nb.py`.

Usage:
    python tools/build_vision_nb.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stages" / "infer_vision.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(
"""# Combined vision pass — Stages 2 → 2.5 → 3 → 4 on GPU (Colab)

Runs player **tracking**, **role classification**, **pose**, and **ball detection**
in one GPU trip, so this heavy vision work is off the local CPU. Writes
`players.parquet`, `track_roles.json`, `poses.parquet`, `ball.parquet` (+ metas)
back to Drive; download them into the clip's `data/<clip>/` folder locally, then run
the light analytical stages (5–11 + report) on your machine.

**Setup on Drive (`MyDrive/`):**
- `pb_vision_upload.zip`  ← the code bundle (`python tools/build_vision_bundle.py`, then upload)
- `ball_model_v4.pt`  ← the trained ball model (already there)
- `pb_infer/<CLIP>/`  ← the clip folder, containing **`video.mp4` + the setup files**
  (`court.json`, `court_zones.json`, `roster.json`, and optionally `user_clicks.json`)

**Runtime → Change runtime type → GPU** (A100/L4/T4 all fine). Set `CLIP`, Run All.

Runs the real committed stage modules (`python -m stages.<x>.<x>`); nothing to edit
beyond `CLIP`. Built by `tools/build_vision_nb.py`.
"""))

CELLS.append(code(
"""import torch
print('CUDA:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (SLOW — set Runtime→GPU!)')
"""))

CELLS.append(code(
"""from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive')
"""))

CELLS.append(code(
"""# Deps: Colab has torch/opencv/numpy/pandas; the stages also need ultralytics + mediapipe.
!pip -q install ultralytics mediapipe pyarrow 2>/dev/null | tail -1
print('deps ready')
"""))

CELLS.append(code(
"""# Unpack the stage code bundle so `stages/` is importable at /content.
import zipfile, os
BUNDLE = DRIVE/'pb_vision_upload.zip'
assert BUNDLE.exists(), f'missing {BUNDLE} — run tools/build_vision_bundle.py and upload it to Drive root'
with zipfile.ZipFile(BUNDLE) as z:
    z.extractall('/content')
print('unpacked', BUNDLE, '->/content/stages')
assert Path('/content/stages/track_players/track.py').exists()
"""))

CELLS.append(code(
"""# ===== KNOBS =====
CLIP    = 'pb5test'                       # folder under MyDrive/pb_infer/<CLIP>/
WEIGHTS = DRIVE/'ball_model_v4.pt'        # ball model (Drive root)

CLIP_DIR = DRIVE/'pb_infer'/CLIP
assert (CLIP_DIR/'video.mp4').exists(), f'missing {CLIP_DIR}/video.mp4'
for req in ('court.json', 'court_zones.json'):
    assert (CLIP_DIR/req).exists(), f'missing {CLIP_DIR}/{req} — upload the setup files too'
print('clip folder:', CLIP_DIR)
print('inputs:', sorted(p.name for p in CLIP_DIR.iterdir()))
"""))

CELLS.append(code(
r"""# Run a stage module (cwd=/content so `stages` imports; live-stream its log).
import subprocess, sys, time
def run_stage(module, *args):
    cmd = [sys.executable, '-u', '-m', module, str(CLIP_DIR), *args]
    print(f"\n===== {module} =====", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd='/content', stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    p.wait()
    dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{module} failed (rc={p.returncode}) after {dt:.0f}s")
    print(f"  -> {module} done in {dt:.0f}s", flush=True)
"""))

CELLS.append(md("## Stage 2 — track players (YOLO on GPU)"))
CELLS.append(code("run_stage('stages.track_players.track')\n"))

CELLS.append(md("## Stage 2.5 — classify tracks into roles"))
CELLS.append(code("run_stage('stages.classify_tracks.classify_tracks', '--force')\n"))

CELLS.append(md("## Stage 3 — pose"))
CELLS.append(code("run_stage('stages.pose.pose')\n"))

CELLS.append(md(
"""## Stage 4 — ball detection (GPU)

Uses the committed `track_ball_v4` module (auto-uses CUDA). This is the per-frame
path; for the fastest ball inference the standalone **`infer_v4.ipynb`** uses a
batched GPU loop — a future optimization to fold in here."""))
CELLS.append(code("run_stage('stages.track_ball.track_ball_v4', '--weights', str(WEIGHTS), '--force')\n"))

CELLS.append(md("## Done — outputs written to the clip folder on Drive"))
CELLS.append(code(
r"""outs = ['players.parquet', 'players_pending.json', 'track_roles.json',
        'poses.parquet', 'pose_summary.json', 'ball.parquet', 'ball.meta.json']
print('Outputs in', CLIP_DIR, ':')
for o in outs:
    p = CLIP_DIR/o
    print(f"  {'OK ' if p.exists() else 'MISSING '} {o}" + (f"  ({p.stat().st_size//1024} KB)" if p.exists() else ''))
print('\nDownload these into your local data/<clip>/ folder, then run Stages 5-11 + report.')
"""))


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

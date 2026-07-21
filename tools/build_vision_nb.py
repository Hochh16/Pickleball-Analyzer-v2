"""Build `stages/infer_vision.ipynb` — the combined GPU (Colab) vision pass.

The notebook is a tiny, stable **git-clone bootstrapper**: it clones (or `git
pull`s) this repo on Colab and calls `tools.colab_vision.run_all`. So a code
change is just `git push` here -> **Run All** on Colab (it pulls the latest).
Nothing to rebuild or re-upload; the notebook itself almost never changes and can
be opened straight from GitHub (File -> Open notebook -> GitHub).

Runs Stages **2 -> 2.5 -> 3 -> 4** on the GPU, backing each stage's outputs up to
Drive as it finishes (reset-proof; see tools/colab_vision.py).

Usage:
    python tools/build_vision_nb.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stages" / "infer_vision.ipynb"

# Public repo -> Colab clones it anonymously (no token).
REPO_URL = "https://github.com/Hochh16/Pickleball-Analyzer-v2.git"
REPO_DIR = "/content/Pickleball-Analyzer-v2"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_cells() -> list:
    cells = []

    cells.append(md(
f"""# Vision pass — Stages 2 -> 2.5 -> 3 -> 4 on GPU (Colab)

Player **tracking**, **role classification**, **pose**, and **ball detection** in
one GPU trip. **Reset-proof:** each stage's outputs are backed up to Drive as it
finishes, so if the runtime disconnects you just **Run All** again and it resumes
where it left off.

The code is pulled fresh from GitHub each run — **there's no bundle to upload and
nothing to edit.** On Drive (`My Drive` root) you only need, per clip:
- `ball_model_v4.pt`  — the trained ball model (upload once)
- `<clip>_vision_input.zip`  — the clip bundle, downloaded from the app's run screen

**Runtime -> Change runtime type -> A100 GPU** (Colab Pro+; ~4-5x faster than a T4
and no ball OOM fallback — a T4 works too, just slower), then **Runtime -> Run
all**. The clip is auto-detected from the one `*_vision_input.zip` on Drive.
Outputs land in `content/<clip>/` and `My Drive/<clip>_outputs/`; upload them on the
app run screen to finish. Built by `tools/build_vision_nb.py`.
"""))

    cells.append(code(
'''import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("GPU:", gpu or "NONE — CPU (very slow)")
# Measured: the full pass is ~4-5x faster on an A100 than a T4 (and ball runs at
# full batch instead of the OOM fallback). A100 is included with Colab Pro+.
if gpu is None:
    raise SystemExit("No GPU. Set Runtime -> Change runtime type -> A100 GPU, then Run all.")
if not any(g in gpu for g in ("A100", "H100")):
    print(f"\\n>>> You are on {gpu}. For a big speedup switch to A100:\\n"
          ">>>   Runtime -> Change runtime type -> A100 GPU -> Save, then Runtime -> Run all.")
'''))

    cells.append(code(
"""from google.colab import drive
drive.mount('/content/drive')
"""))

    cells.append(code(
f"""# Pull the code fresh from GitHub (clone, or fast-forward if already cloned).
import os, subprocess
REPO_URL = {REPO_URL!r}
REPO = {REPO_DIR!r}
if os.path.isdir(os.path.join(REPO, '.git')):
    subprocess.run(['git', '-C', REPO, 'pull', '--ff-only'], check=True)
else:
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO], check=True)
print('repo ready at', REPO)
"""))

    cells.append(code(
"""# Deps: Colab has torch/opencv/numpy/pandas; the stages also need these.
!pip -q install ultralytics pyarrow 2>/dev/null | tail -1
print('deps ready')
"""))

    cells.append(code(
"""import sys
if REPO not in sys.path:
    sys.path.insert(0, REPO)
# A re-run in the SAME session must pick up freshly pulled code: Python caches
# imported modules, so drop any previously loaded repo modules first.
for name in [n for n in list(sys.modules) if n == 'tools' or n.startswith('tools.')]:
    del sys.modules[name]
from tools.colab_vision import run_all, STAGES
print('loaded — stages:', [s['name'] for s in STAGES])
"""))

    cells.append(md(
"""## Run

`CLIP = None` auto-detects the single clip on Drive. Only set it to a name if you
have more than one `*_vision_input.zip` uploaded.

`RERUN = None` normally. Set it to a stage name (e.g. `'ball'`) to FORCE that one
stage to run again after a code change — the other, slow stages (players / classify /
pose) are still skipped, so it only costs that stage's GPU time."""))
    cells.append(code(
"""CLIP  = None   # None = auto-detect the one uploaded clip; or set e.g. 'pb_5_minute_outdoor'
RERUN = None   # None = normal resume; or 'ball' to redo just the ball stage
run_all(REPO, clip=CLIP, rerun=RERUN)
"""))

    cells.append(md(
"""## Done

Download the 7 outputs from **`content/<clip>/`** (Files panel, left) **or** from
**`My Drive/<clip>_outputs/`** (survives a runtime reset), then upload them on the
app's run screen to resume Stages 5-11 and build the report."""))
    return cells


def main() -> int:
    nb = {
        "cells": build_cells(),
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
    print(f"wrote {OUT} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

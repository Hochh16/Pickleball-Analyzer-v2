"""Build `stages/infer_vision.ipynb` — the combined GPU (Colab) vision pass.

Runs Stages **2 (track_players) -> 2.5 (classify_tracks) -> 3 (pose) ->
4 (track_ball)** in one Colab GPU trip, so the heavy vision work is off the local
CPU. The orchestration lives in `tools/colab_vision.py` (a real, testable module);
this generator embeds that module into the notebook via a `%%writefile` cell, so
the notebook is a dead-simple **Run All** with nothing to edit:

  - auto-derives CLIP from the single `<clip>_vision_input.zip` on Drive,
  - pulls the clip + weights to local disk once (robustly),
  - backs up each stage's outputs to Drive and RESUMES from there (reset-proof),
  - self-manages the GPU (ball OOM auto-fallback).

Usage:
    python tools/build_vision_nb.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stages" / "infer_vision.ipynb"
MODULE = ROOT / "tools" / "colab_vision.py"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_cells() -> list:
    module_src = MODULE.read_text(encoding="utf-8")
    cells = []

    cells.append(md(
"""# Vision pass — Stages 2 -> 2.5 -> 3 -> 4 on GPU (Colab)

Player **tracking**, **role classification**, **pose**, and **ball detection** in
one GPU trip. **Reset-proof:** each stage's outputs are backed up to Drive as it
finishes, so if the runtime disconnects you just **Run All** again and it resumes
where it left off.

**On Drive (`My Drive` root), upload once:**
- `pb_vision_upload.zip`  — the stage code (`python tools/build_vision_bundle.py`, upload)
- `ball_model_v4.pt`  — the trained ball model
- `<clip>_vision_input.zip`  — the clip bundle, downloaded from the app's run screen

**Runtime -> Change runtime type -> GPU**, then **Runtime -> Run all**. Nothing to
edit — the clip is auto-detected from the one `*_vision_input.zip` on Drive.
Outputs land in `content/<clip>/` and in `My Drive/<clip>_outputs/`; upload them on
the app run screen to finish the analysis. Built by `tools/build_vision_nb.py`.
"""))

    cells.append(code(
"""import torch
print('CUDA:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (SLOW — set Runtime->GPU!)')
"""))

    cells.append(code(
"""from google.colab import drive
drive.mount('/content/drive')
"""))

    cells.append(code(
"""# Deps: Colab has torch/opencv/numpy/pandas; the stages also need these.
!pip -q install ultralytics mediapipe pyarrow 2>/dev/null | tail -1
print('deps ready')
"""))

    # Embed the orchestration module verbatim, then import it.
    cells.append(md("## Orchestration (auto-generated from `tools/colab_vision.py`)"))
    cells.append(code("%%writefile /content/colab_vision.py\n" + module_src))
    cells.append(code(
"""import sys
sys.path.insert(0, '/content')
import colab_vision
print('colab_vision loaded — stages:', [s['name'] for s in colab_vision.STAGES])
"""))

    cells.append(md(
"""## Run

`CLIP = None` auto-detects the single clip on Drive. Only set it to a name if you
have more than one `*_vision_input.zip` uploaded."""))
    cells.append(code(
"""CLIP = None   # None = auto-detect the one uploaded clip; or set e.g. 'pb_5_minute_outdoor'
colab_vision.run_all(clip=CLIP)
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

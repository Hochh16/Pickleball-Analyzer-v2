"""Score ball labels against an independent detector — the acceptance test for labels.

Motivation: an operator's click is trustworthy, but the FRAME it gets filed under is
not (see a98158b). A frame-indexing bug is invisible in the label file itself, and the
only symptom downstream is a model that mysteriously refuses to learn a venue. This
measures label quality directly by asking a detector that was never trained on this
clip where it sees the ball, and comparing that to where the label says it is.

Reported per clip:
    median px   distance from the model's heatmap peak to the label
    <=6px       fraction of labels the model agrees with tightly
    bad         fraction where the model is CONFIDENT (>=0.5) and >20px away --
                the labels that actively teach the wrong thing

Reference: pb_2min, the clip the 0.90-recall model was trained on, reads
1.7px / 97% / 0%. Anything far from that is a labeling problem, not a model problem.

Frames come from frames_720/ (written by one sequential decode), never from a seek.

Usage:
    python -m tools.audit_labels data/indoor_B1_3min data/pb_2min --n 110
    python -m tools.audit_labels data/indoor_B1_3min --save data/_audit_after.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.track_ball.track_ball_v4 import PROC_H, PROC_W, load_model, to_proc

DEFAULT_WEIGHTS = Path("data/models/ball_model_v4.pt")


def frame_720(folder: Path, idx: int):
    p = folder / "frames_720" / f"{idx}.jpg"
    return cv2.imread(str(p)) if p.exists() else None


@torch.no_grad()
def peak(model, device, folder: Path, idx: int):
    """Model peak for frame idx, in 720p coords, or None if the stack is incomplete."""
    buf = []
    for k in (-1, 0, 1):
        f = frame_720(folder, idx + k)
        if f is None:
            return None
        buf.append(to_proc(f))
    t = torch.from_numpy(np.concatenate(buf, axis=0)[None]).to(device)
    with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
        hm = model(t)[0, 0].float().cpu().numpy()
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    sx, sy = PROC_W / hm.shape[1], PROC_H / hm.shape[0]
    return ix * sx, iy * sy, float(hm[iy, ix])


def audit(folder: Path, model, device, n: int, seed: int) -> list[tuple[float, float]]:
    doc = json.loads((folder / "ball_labels.json").read_text(encoding="utf-8"))
    labels = [l for l in doc["labels"]
              if l.get("ball_visible") and l.get("pixel_x") is not None]
    rng = random.Random(seed)
    rng.shuffle(labels)
    sx = doc["video_width"] / float(PROC_W)
    sy = doc["video_height"] / float(PROC_H)

    out = []
    for l in labels:
        if len(out) >= n:
            break
        r = peak(model, device, folder, int(l["frame_idx"]))
        if r is None:
            continue
        px, py, conf = r
        out.append((float(np.hypot(px - l["pixel_x"] / sx, py - l["pixel_y"] / sy)), conf))
    return out


def report(name: str, rows: list[tuple[float, float]]) -> None:
    if not rows:
        print(f"{name:<18} no samples")
        return
    e = np.array([r[0] for r in rows])
    c = np.array([r[1] for r in rows])
    print(f"{name:<18} {np.median(e):6.1f}px  {100 * (e <= 6).mean():4.0f}%  "
          f"{100 * ((e > 20) & (c >= 0.5)).mean():4.0f}%   (n={len(e)})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--compare", type=Path, help="a previous --save, shown alongside")
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(a.weights, device)
    print(f"weights={a.weights.name} device={device}")
    print(f"{'clip':<18} {'median':>8} {'<=6px':>6} {'bad':>5}")

    prev = pickle.loads(a.compare.read_bytes()) if a.compare and a.compare.exists() else {}
    res = {}
    for f in a.folders:
        res[f.name] = audit(f, model, device, a.n, a.seed)
        if f.name in prev:
            report(f"{f.name} (was)", prev[f.name])
        report(f.name, res[f.name])
    if a.save:
        a.save.write_bytes(pickle.dumps(res))
        print(f"saved -> {a.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

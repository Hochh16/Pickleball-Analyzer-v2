"""Measure per-clip venue signals — and, where labels exist, the truth to calibrate them.

Purpose: contract D4 says a venue whose measurement quality is materially worse is not
merged into a collection. But a brand-new venue has no labels, so its recall — the thing
D4 gates on — cannot be computed. It has to be PREDICTED from signals available without
labels.

This tool emits both halves in ONE pass over the same frames and the same weights:

  UNLABELLED signals, available for any new clip
    det_rate      fraction of sampled frames with a peak >= CONF
    mean_conf     mean peak confidence
    p90_conf      90th percentile peak confidence
    contrast      ball-vs-background yellow contrast AT the detected peak
    continuity    fraction of consecutive-frame pairs whose peaks move a
                  ball-plausible distance -- a real ball flies, noise teleports

  LABELLED truth, only for clips we have labelled
    recall        fraction of visible labels the peak lands within TOL of

Measuring both with the SAME weights matters. The per-venue recalls quoted from Colab
(0.978 / 0.713 / 0.087) come from models that are not on this machine; calibrating local
signals against remote recalls would compare two different detectors. Recalibrate
whenever the deployed model changes — it is one command.

Sampling is in short CONSECUTIVE bursts, not scattered singles, because continuity is
only observable across adjacent frames.

Usage:
    python -m tools.venue_signals data/pb_2min data/pb_3min_indoor
    python -m tools.venue_signals data/* --out data/_venue_signals.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.track_ball.track_ball_v4 import PROC_H, PROC_W, load_model, to_proc

CONF = 0.30           # matches Stage 4's detection gate
TOL = 6.0             # px at 720p, matches the training eval
# Sized against measured CPU cost (~4.5 s per 9-channel 720p forward pass here): 140
# stacks/clip is ~10 min, ~80 min for the eight calibration clips. Consecutive frames in
# a burst share two thirds of their input, so decoding is cached and only the forward
# passes actually cost anything.
N_RECALL = 60         # labelled frames scored for truth
N_BURSTS, BURST = 16, 5   # 80 frames for the unlabelled signals
MAX_STEP_PX = 55.0    # 720p px/frame a real ball may move (4K 160px / 3)
BALL_R, BG_R = 6, 22


def frame(folder: Path, idx: int):
    p = folder / "frames_720" / f"{idx}.jpg"
    return cv2.imread(str(p)) if p.exists() else None


def proc_cached(folder: Path, idx: int, cache: dict):
    """to_proc() for a frame, memoised. A 5-frame burst needs 7 decodes, not 15."""
    if idx not in cache:
        f = frame(folder, idx)
        cache[idx] = None if f is None else to_proc(f)
    return cache[idx]


@torch.no_grad()
def peak(model, dev, folder: Path, idx: int, cache: dict | None = None):
    cache = {} if cache is None else cache
    buf = []
    for k in (-1, 0, 1):
        pr = proc_cached(folder, idx + k, cache)
        if pr is None:
            return None
        buf.append(pr)
    t = torch.from_numpy(np.concatenate(buf, axis=0)[None]).to(dev)
    with torch.amp.autocast("cuda", enabled=str(dev).startswith("cuda")):
        hm = model(t)[0, 0].float().cpu().numpy()
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    return (ix * PROC_W / hm.shape[1], iy * PROC_H / hm.shape[0], float(hm[iy, ix]))


def contrast_at(img, cx: float, cy: float):
    """Yellow standout of the ball against its immediate surroundings, or None at the
    frame edge. Same definition used to rank the venues."""
    h, w = img.shape[:2]
    cx, cy = int(cx), int(cy)
    if not (BG_R <= cx < w - BG_R and BG_R <= cy < h - BG_R):
        return None
    f = img.astype(np.float32)
    ball = f[cy-BALL_R:cy+BALL_R+1, cx-BALL_R:cx+BALL_R+1]
    ring = f[cy-BG_R:cy+BG_R+1, cx-BG_R:cx+BG_R+1].copy()
    ring[BG_R-BALL_R:BG_R+BALL_R+1, BG_R-BALL_R:BG_R+BALL_R+1] = np.nan
    yb = (ball[:, :, 2] + ball[:, :, 1]) / 2 - ball[:, :, 0]
    yr = (ring[:, :, 2] + ring[:, :, 1]) / 2 - ring[:, :, 0]
    return float(np.nanmax(yb) - np.nanmedian(yr))


def frame_range(folder: Path) -> tuple[int, int]:
    idx = sorted(int(p.stem) for p in (folder / "frames_720").glob("*.jpg"))
    return (idx[0], idx[-1]) if idx else (0, -1)


def measure(folder: Path, model, dev, seed: int) -> dict:
    lo, hi = frame_range(folder)
    if hi <= lo:
        return {"error": "no frames_720 cache"}
    rng = random.Random(seed)

    # ---- unlabelled signals, from consecutive bursts ----
    confs, contrasts, steps = [], [], []
    starts = [lo + 2 + int((hi - lo - BURST - 4) * i / max(1, N_BURSTS - 1))
              for i in range(N_BURSTS)]
    for s in starts:
        prev = None
        cache: dict = {}
        for f in range(s, s + BURST):
            r = peak(model, dev, folder, f, cache)
            if r is None:
                prev = None
                continue
            x, y, c = r
            confs.append(c)
            if c >= CONF:
                img = frame(folder, f)
                k = contrast_at(img, x, y) if img is not None else None
                if k is not None:
                    contrasts.append(k)
                if prev is not None:
                    steps.append(float(np.hypot(x - prev[0], y - prev[1])))
                prev = (x, y)
            else:
                prev = None

    confs = np.array(confs) if confs else np.array([0.0])
    steps = np.array(steps) if steps else np.array([])
    out = {
        "n_frames_scored": int(len(confs)),
        "det_rate": float((confs >= CONF).mean()),
        "mean_conf": float(confs.mean()),
        "p90_conf": float(np.percentile(confs, 90)),
        "contrast": float(np.median(contrasts)) if contrasts else None,
        "continuity": float((steps <= MAX_STEP_PX).mean()) if steps.size else None,
        "n_steps": int(steps.size),
    }

    # ---- truth, only where the operator has labelled ----
    lp = folder / "ball_labels.json"
    if lp.exists():
        doc = json.loads(lp.read_text(encoding="utf-8"))
        vis = [l for l in doc["labels"]
               if l.get("ball_visible") and l.get("pixel_x") is not None]
        rng.shuffle(vis)
        sx, sy = doc["video_width"] / float(PROC_W), doc["video_height"] / float(PROC_H)
        hit = tot = 0
        for l in vis:
            if tot >= N_RECALL:
                break
            r = peak(model, dev, folder, int(l["frame_idx"]))
            if r is None:
                continue
            x, y, c = r
            tot += 1
            if c >= CONF and np.hypot(x - l["pixel_x"]/sx, y - l["pixel_y"]/sy) <= TOL:
                hit += 1
        out["recall"] = round(hit / tot, 4) if tot else None
        out["n_recall"] = tot
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--weights", type=Path, default=Path("data/models/ball_model_v4.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/_venue_signals.json"))
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(a.weights, dev)
    print(f"weights={a.weights.name} device={dev}")
    hdr = f"{'clip':<20}{'det':>6}{'conf':>7}{'p90':>7}{'contr':>7}{'cont':>7}{'recall':>8}"
    print(hdr)

    res = {}
    for f in a.folders:
        if not (f / "frames_720").exists():
            continue
        r = measure(f, model, dev, a.seed)
        res[f.name] = r
        if "error" in r:
            print(f"{f.name:<20}{r['error']}")
            continue
        fmt = lambda v, w, p: (f"{v:>{w}.{p}f}" if v is not None else " " * (w - 1) + "-")
        print(f"{f.name:<20}{fmt(r['det_rate'],6,2)}{fmt(r['mean_conf'],7,3)}"
              f"{fmt(r['p90_conf'],7,3)}{fmt(r['contrast'],7,1)}"
              f"{fmt(r['continuity'],7,2)}{fmt(r.get('recall'),8,3)}")
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nsaved -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

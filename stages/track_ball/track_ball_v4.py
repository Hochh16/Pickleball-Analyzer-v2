"""Stage 4 (v4) — ball tracking inference + trajectory post-processing.

Runs the v4 TrackNet detector (trained by stages/finetune_ball_model) on a
video and produces a real ball.parquet + ball.meta.json (synthetic=false),
a drop-in replacement for the synthetic placeholder.

Pipeline per frame: 3-frame 720p stack -> heatmap -> peak (x,y,conf) -> map to
source pixels. Then court-agnostic trajectory post-processing: drop isolated
velocity outliers, interpolate short gaps (marked interpolated), leave long
gaps not-visible.

Output ball.parquet schema (matches synth_ball): schema_version, frame_idx,
pixel_x, pixel_y, visible, confidence, interpolated. Invariant: each frame is
exactly one of visible / interpolated / not-visible; known rows have non-NaN xy.

Usage:
    python -m stages.track_ball.track_ball_v4 data/pb_2min --force
    python -m stages.track_ball.track_ball_v4 data/pb_2min --start-frame 100 \
        --max-frames 400 --overlay data/pb_2min/_ball_check.mp4 --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stages.track_ball._tracknet_model import TrackNet

SCHEMA_VERSION = 1
PROC_H, PROC_W = 720, 1280
CONF_THRESH = 0.30          # heatmap peak >= this -> a detection (matches training eval)
MAX_GAP_FRAMES = 8          # interpolate confirmed-detection gaps up to this many frames
OUTLIER_MAX_STEP_PX = 250.0  # source px/frame; a det this far from BOTH neighbors is dropped


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("track_ball_v4")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


def load_model(weights: Path, device) -> Tuple[TrackNet, tuple]:
    ck = torch.load(str(weights), map_location=device)
    ishape = tuple(ck.get("input_shape", (PROC_H, PROC_W)))
    model = TrackNet(in_channels=ck.get("in_channels", 9),
                     out_channels=ck.get("out_channels", 1),
                     input_shape=ishape).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ishape


def to_proc(frame) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)


@torch.no_grad()
def detect(model, device, buf3: List[np.ndarray], sx: float, sy: float
           ) -> Tuple[float, float, float]:
    """buf3 = [t-1, t, t+1] proc CHW arrays. Returns (src_x, src_y, conf) for
    the CENTER frame."""
    stack = np.concatenate(buf3, axis=0)[None]  # (1,9,H,W)
    t = torch.from_numpy(stack).to(device)
    with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
        hm = model(t)[0, 0].float().cpu().numpy()
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    return ix * sx, iy * sy, float(hm[iy, ix])


def detect_batch(model, device, stacks: List[np.ndarray], centers: List[int],
                 sx: float, sy: float, conf_thresh: float,
                 dets: dict, raw_conf: list) -> None:
    """Run the model on a BATCH of (9,H,W) stacks and record a detection for each
    window's center frame. Batching is the real GPU speedup (per-window inference
    leaves the GPU mostly idle); results are identical to per-frame `detect()`."""
    if not stacks:
        return
    t = torch.from_numpy(np.stack(stacks)).to(device)   # (N,9,H,W)
    with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
        hm = model(t)[:, 0].float().cpu().numpy()        # (N,H,W)
    for k, center in enumerate(centers):
        h = hm[k]
        iy, ix = np.unravel_index(int(h.argmax()), h.shape)
        c = float(h[iy, ix])
        raw_conf.append(c)
        if c >= conf_thresh:
            dets[center] = (ix * sx, iy * sy, c)


def _batch_size(device, args) -> int:
    """Frames per forward pass. Scale to GPU memory (A100 OOMs at 16 @ 720x1280,
    fine at 8); CPU stays per-frame."""
    if getattr(args, "batch", None):
        return max(1, args.batch)
    if str(device).startswith("cuda"):
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return 8 if gb > 20 else 4
    return 1


# --- trajectory post-processing ---------------------------------------------

def postprocess(dets: dict, frames: List[int]) -> List[dict]:
    """dets: frame -> (x, y, conf) for frames whose peak >= CONF. frames: full
    ordered frame list to emit rows for. Returns per-frame row dicts."""
    conf_frames = sorted(dets.keys())
    # 1) drop isolated velocity outliers (far from BOTH neighbors)
    kept = set(conf_frames)
    for i, f in enumerate(conf_frames):
        x, y, _ = dets[f]
        bad = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(conf_frames):
                g = conf_frames[j]
                px, py, _ = dets[g]
                if np.hypot(x - px, y - py) > OUTLIER_MAX_STEP_PX * max(1, abs(f - g)):
                    bad.append(True)
                else:
                    bad.append(False)
        if bad and all(bad):  # impossible jump from every neighbor present
            kept.discard(f)
    conf = [f for f in conf_frames if f in kept]
    confset = set(conf)

    # 2) interpolate short gaps between consecutive confirmed detections
    interp = {}
    for a, b in zip(conf, conf[1:]):
        gap = b - a
        if 1 < gap <= MAX_GAP_FRAMES:
            xa, ya, _ = dets[a]
            xb, yb, _ = dets[b]
            for k in range(1, gap):
                t = k / gap
                interp[a + k] = (xa + t * (xb - xa), ya + t * (yb - ya))

    rows = []
    for f in frames:
        if f in confset:
            x, y, c = dets[f]
            rows.append({"frame_idx": f, "pixel_x": float(x), "pixel_y": float(y),
                         "visible": True, "confidence": float(c), "interpolated": False})
        elif f in interp:
            x, y = interp[f]
            rows.append({"frame_idx": f, "pixel_x": float(x), "pixel_y": float(y),
                         "visible": False, "confidence": np.nan, "interpolated": True})
        else:
            rows.append({"frame_idx": f, "pixel_x": np.nan, "pixel_y": np.nan,
                         "visible": False, "confidence": np.nan, "interpolated": False})
    return rows


# --- main --------------------------------------------------------------------

def run(folder: Path, args, log: logging.Logger) -> dict:
    video = folder / "video.mp4"
    if not video.exists():
        fail(f"video not found: {video}", FileNotFoundError)
    out_parquet = folder / "ball.parquet"
    out_meta = folder / "ball.meta.json"
    if out_parquet.exists() and not args.force:
        fail(f"output exists: {out_parquet}. Use --force.", FileExistsError)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ishape = load_model(Path(args.weights), device)
    log.info(f"model {args.weights} @ {ishape} on {device}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        fail(f"cannot open {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = sw / PROC_W, sh / PROC_H

    start = max(0, args.start_frame)
    end = n_total if args.max_frames is None else min(n_total, start + args.max_frames)
    frames = list(range(start, end))
    log.info(f"video {sw}x{sh}@{fps:.1f}, {n_total} frames; inferring [{start},{end})")

    # sliding 3-frame buffer; detection produced for the MIDDLE frame. Windows
    # are accumulated and run through the model in BATCHES (the GPU speedup).
    bsz = _batch_size(device, args)
    log.info(f"batch size {bsz}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    buf, dets, raw_conf = [], {}, []
    b_stacks, b_centers = [], []
    src_cache = {} if args.overlay else None
    fidx = start
    while fidx < end:
        ok, fr = cap.read()
        if not ok:
            break
        if src_cache is not None:
            src_cache[fidx] = fr
        buf.append(to_proc(fr))
        if len(buf) > 3:
            buf.pop(0)
        if len(buf) == 3:
            b_stacks.append(np.concatenate(buf, axis=0))   # (9,H,W)
            b_centers.append(fidx - 1)
            if len(b_stacks) >= bsz:
                detect_batch(model, device, b_stacks, b_centers, sx, sy, args.conf, dets, raw_conf)
                b_stacks, b_centers = [], []
        fidx += 1
        if (fidx - start) % 200 == 0 and fidx > start:
            log.info(f"  {fidx-start}/{len(frames)} frames")
    detect_batch(model, device, b_stacks, b_centers, sx, sy, args.conf, dets, raw_conf)  # flush
    cap.release()

    rows = postprocess(dets, frames)
    df = pd.DataFrame(rows)
    df.insert(0, "schema_version", SCHEMA_VERSION)
    df["visible"] = df["visible"].astype(bool)
    df["interpolated"] = df["interpolated"].astype(bool)
    df["confidence"] = df["confidence"].astype("float32")
    df.to_parquet(out_parquet, index=False)

    n_vis = int(df["visible"].sum())
    n_interp = int(df["interpolated"].sum())
    n_frames = len(df)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "synthetic": False,
        "video_path": str(video),
        "video_frame_count": n_total,
        "video_fps": float(fps),
        "video_width": sw,
        "video_height": sh,
        "detector": {"tool": "stages/track_ball/track_ball_v4.py",
                     "weights": str(args.weights), "proc_hw": [PROC_H, PROC_W],
                     "conf_thresh": args.conf, "max_gap_frames": MAX_GAP_FRAMES},
        "range": [start, end],
        "stats": {"frames": n_frames, "frames_visible": n_vis,
                  "frames_interpolated": n_interp,
                  "frames_not_visible": n_frames - n_vis - n_interp,
                  "visible_frac": round(n_vis / max(n_frames, 1), 4),
                  "detect_frac": round((n_vis + n_interp) / max(n_frames, 1), 4)},
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out_meta.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    log.info(f"wrote {out_parquet} + meta: {n_vis} visible, {n_interp} interp, "
             f"{n_frames-n_vis-n_interp} not-visible "
             f"(detect_frac {meta['stats']['detect_frac']})")

    if args.overlay:
        _render_overlay(Path(args.overlay), src_cache, df, fps, sw, sh, log)
    return meta


def _render_overlay(path: Path, src, df, fps, sw, sh, log):
    rows = {int(r.frame_idx): r for r in df.itertuples(index=False)}
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (sw, sh))
    for f in sorted(src.keys()):
        fr = src[f]
        r = rows.get(f)
        if r is not None and not (isinstance(r.pixel_x, float) and np.isnan(r.pixel_x)):
            col = (0, 255, 0) if r.visible else (0, 255, 255)  # green=det, yellow=interp
            cv2.circle(fr, (int(r.pixel_x), int(r.pixel_y)), 12, col, 3, cv2.LINE_AA)
        writer.write(fr)
    writer.release()
    log.info(f"wrote overlay {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 4 v4 — ball inference + trajectory")
    p.add_argument("folder", type=Path)
    p.add_argument("--weights", default="data/models/ball_model_v4.pt")
    p.add_argument("--start-frame", type=int, default=0, dest="start_frame")
    p.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    p.add_argument("--conf", type=float, default=CONF_THRESH)
    p.add_argument("--batch", type=int, default=None,
                   help="frames per GPU forward pass (default: auto from GPU memory; CPU=1)")
    p.add_argument("--overlay", default=None, help="write a debug overlay mp4 of the range")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log = setup_logging(args.log_level)
    try:
        run(args.folder, args, log)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        log.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

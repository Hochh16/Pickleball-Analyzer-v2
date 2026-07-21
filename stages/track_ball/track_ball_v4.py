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
import math
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

# --- candidate + continuity tracking (fixes adjacent-court flip-flop) ---------
# The model is a heatmap net; taking only its global argmax per frame meant a
# stronger peak on a NEIGHBOURING court silently stole the track, and a real but
# weak ball below CONF_THRESH was discarded with no alternative recorded. We now
# keep the top-k peaks down to CAND_CONF_FLOOR and let select_track() pick the
# path that actually moves like a ball.
CAND_TOPK = 3               # candidate peaks kept per frame
CAND_CONF_FLOOR = 0.15      # keep candidates down to here (accept gate stays CONF_THRESH)
PEAK_SUPPRESS_RADIUS = 6    # proc px zeroed around a peak before taking the next
TRACK_MAX_STEP_PX = 160.0   # source px/frame the ball may plausibly move (4K, fast drive ~55)
TRACK_LINK_GAP = 8          # frames a link may span (matches MAX_GAP_FRAMES)
TRACK_RESTART_COST = 2.5    # cost to re-acquire after losing the ball (~2.5 frames of conf)
TRACK_MOTION_W = 1.0        # weight on the normalised motion penalty
TRACK_GAP_PENALTY = 0.05    # small penalty per skipped frame (prefer continuous)
# A ball IN PLAY is never parked (measured median motion ~6.7 src px/frame). Without
# this, a stationary high-confidence object on a neighbouring court is the "smoothest"
# possible track and wins outright — the exact failure this rewrite targets.
TRACK_MIN_STEP_PX = 2.0     # below this src px/frame a link looks stationary, not ball-like
TRACK_STILL_W = 0.8         # penalty weight for a stationary link
WEAK_SUPPORT_GAP = 2        # a sub-threshold pick adjacent (<= this) to an accepted one is kept


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
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
        hm = model(t)[0, 0].float().cpu().numpy()
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    return ix * sx, iy * sy, float(hm[iy, ix])


def topk_peaks(h: np.ndarray, k: int, min_conf: float, radius: int):
    """Top-k LOCAL maxima of the heatmap (peak, then suppress its neighbourhood,
    repeat). The old code took only the single global argmax, so whenever an
    ADJACENT COURT's ball produced a stronger peak the track jumped to it — with
    no way to recover our ball, because the alternative was never recorded.
    Returns [(ix, iy, conf), ...] strongest first."""
    h = h.copy()
    out = []
    for _ in range(k):
        iy, ix = np.unravel_index(int(h.argmax()), h.shape)
        c = float(h[iy, ix])
        if c < min_conf:
            break
        out.append((int(ix), int(iy), c))
        y0, y1 = max(0, iy - radius), min(h.shape[0], iy + radius + 1)
        x0, x1 = max(0, ix - radius), min(h.shape[1], ix + radius + 1)
        h[y0:y1, x0:x1] = -1.0
    return out


def detect_batch(model, device, stacks: List[np.ndarray], centers: List[int],
                 sx: float, sy: float, conf_thresh: float,
                 cands: dict, raw_conf: list, topk: int = CAND_TOPK,
                 cand_floor: Optional[float] = None) -> None:
    """Run the model on a BATCH of (9,H,W) stacks and record CANDIDATE peaks for
    each window's center frame. Batching is the real GPU speedup (per-window
    inference leaves the GPU mostly idle).

    Records up to `topk` peaks down to `cand_floor` (BELOW the accept threshold):
    the winning detection is chosen later by `select_track`, which uses temporal
    continuity. That recovers (a) weak-but-real balls the single-argmax +
    conf_thresh path dropped, and (b) our ball on frames where a competing
    adjacent-court ball had the stronger peak."""
    if not stacks:
        return
    floor = conf_thresh if cand_floor is None else min(cand_floor, conf_thresh)
    t = torch.from_numpy(np.stack(stacks)).to(device)   # (N,9,H,W)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
        hm = model(t)[:, 0].float().cpu().numpy()        # (N,H,W)
    for k, center in enumerate(centers):
        h = hm[k]
        peaks = topk_peaks(h, topk, floor, PEAK_SUPPRESS_RADIUS)
        raw_conf.append(peaks[0][2] if peaks else 0.0)
        if peaks:
            cands[center] = [(ix * sx, iy * sy, c) for ix, iy, c in peaks]


def select_track(cands: dict, max_step_px: float, link_gap: int,
                 restart_cost: float, motion_w: float, gap_pen: float,
                 accept_conf: float) -> dict:
    """Choose ONE candidate per frame so the whole sequence forms the most
    plausible single trajectory (Viterbi-style DP over candidates).

    Score = summed candidate confidence, minus a motion penalty for how fast the
    ball would have to move between consecutive picks (hard-gated at
    `max_step_px` per frame), minus a small penalty for skipped frames. A
    "restart" (re-acquiring after a long gap) is allowed but costs `restart_cost`
    — enough that flip-flopping onto an adjacent court's ball never pays, while a
    genuine re-appearance still does.

    Returns {frame: (x, y, conf)} for the frames on the winning path."""
    frames = sorted(cands)
    nodes = []                      # (frame, x, y, conf)
    by_frame = {}
    for f in frames:
        by_frame[f] = []
        for (x, y, c) in cands[f]:
            by_frame[f].append(len(nodes))
            nodes.append((f, x, y, c))
    n = len(nodes)
    if n == 0:
        return {}
    score = [0.0] * n
    prev = [-1] * n
    run_best, run_arg = 0.0, -1     # best score among nodes at EARLIER frames

    for f in frames:
        cur = by_frame[f]
        # snapshot the running best BEFORE this frame (restart source)
        rb, ra = run_best, run_arg
        for ni in cur:
            _, x, y, c = nodes[ni]
            best = rb - restart_cost + c        # re-acquire from the best prior state
            bp = ra
            for pf in range(f - 1, max(frames[0] - 1, f - link_gap - 1), -1):
                if pf not in by_frame:
                    continue
                gap = f - pf
                lim = max_step_px * gap
                for pj in by_frame[pf]:
                    _, px, py, _ = nodes[pj]
                    d = math.hypot(x - px, y - py)
                    if d > lim:
                        continue
                    speed = d / gap
                    # penalise BOTH implausible speed and implausible stillness:
                    # a parked object would otherwise be the perfect "smooth track".
                    still = max(0.0, 1.0 - speed / TRACK_MIN_STEP_PX)
                    s = (score[pj] + c - motion_w * (d / lim)
                         - TRACK_STILL_W * still - gap_pen * (gap - 1))
                    if s > best:
                        best, bp = s, pj
            score[ni], prev[ni] = best, bp
        for ni in cur:                          # now they can serve as restart sources
            if score[ni] > run_best:
                run_best, run_arg = score[ni], ni

    path = {}
    i = max(range(n), key=lambda j: score[j])
    while i != -1:
        f, x, y, c = nodes[i]
        path[f] = (x, y, c)
        i = prev[i]

    # Acceptance. A pick clearing `accept_conf` is trusted outright. A WEAKER pick
    # is kept only when it sits on the track right next to an accepted one — that
    # temporal support is exactly what a single-frame threshold cannot see, and it
    # is what recovers the real-but-faint ball the old argmax+threshold discarded.
    pf_sorted = sorted(path)
    accepted = {f for f in pf_sorted if path[f][2] >= accept_conf}
    grew = True
    while grew:
        grew = False
        for f in pf_sorted:
            if f in accepted:
                continue
            for g in accepted:
                if abs(g - f) <= WEAK_SUPPORT_GAP:
                    accepted.add(f)
                    grew = True
                    break
    return {f: path[f] for f in pf_sorted if f in accepted}


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
    buf, cands, raw_conf = [], {}, []
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
                detect_batch(model, device, b_stacks, b_centers, sx, sy, args.conf, cands,
                             raw_conf, args.topk, args.cand_floor)
                b_stacks, b_centers = [], []
        fidx += 1
        if (fidx - start) % 200 == 0 and fidx > start:
            log.info(f"  {fidx-start}/{len(frames)} frames")
    detect_batch(model, device, b_stacks, b_centers, sx, sy, args.conf, cands,
                             raw_conf, args.topk, args.cand_floor)  # flush
    cap.release()

    n_cand_frames = len(cands)
    dets = select_track(cands, args.max_step_px, TRACK_LINK_GAP,
                        args.restart_cost, TRACK_MOTION_W, TRACK_GAP_PENALTY,
                        args.conf)
    log.info(f"candidates on {n_cand_frames} frames -> continuity track kept "
             f"{len(dets)} detections (topk={args.topk}, floor={args.cand_floor})")

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
                     "conf_thresh": args.conf, "max_gap_frames": MAX_GAP_FRAMES,
                     "topk": args.topk, "cand_floor": args.cand_floor,
                     "max_step_px": args.max_step_px,
                     "restart_cost": args.restart_cost,
                     "selection": "continuity_dp"},
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
    p.add_argument("--conf", type=float, default=CONF_THRESH,
                   help="accept threshold: a picked candidate must clear this to be VISIBLE")
    p.add_argument("--topk", type=int, default=CAND_TOPK,
                   help="candidate heatmap peaks kept per frame (1 = old argmax behaviour)")
    p.add_argument("--cand-floor", type=float, default=CAND_CONF_FLOOR,
                   dest="cand_floor",
                   help="keep candidates down to this confidence (below --conf); the "
                        "continuity tracker decides which is the ball")
    p.add_argument("--max-step-px", type=float, default=TRACK_MAX_STEP_PX,
                   dest="max_step_px",
                   help="max plausible ball motion in SOURCE px per frame (link gate)")
    p.add_argument("--restart-cost", type=float, default=TRACK_RESTART_COST,
                   dest="restart_cost",
                   help="cost to re-acquire the ball after losing it; higher = less "
                        "willing to jump to a neighbouring court's ball")
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

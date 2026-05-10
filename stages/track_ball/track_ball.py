"""
Stage 4 — track ball.

Detect the pickleball in every frame of a match video and emit a per-frame
record of its pixel-space position. See stages/track_ball/contract.md for
the full spec.

Usage:
    python -m stages.track_ball.track_ball \
        --video data/test_clip/video.mp4 \
        --court data/test_clip/court.json \
        --weights data/models/tracknet_v2_dettor.pt \
        --out data/test_clip/ball.parquet
"""
import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch

from stages.track_ball._tracknet_model import TrackNet

SCHEMA_VERSION = 1
STAGE_VERSION = "0.2.0"

# TrackNetV2 native input resolution
MODEL_H = 288
MODEL_W = 512


# ---------- helpers ----------

def fail(msg: str, exc_cls=RuntimeError):
    raise exc_cls(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("track_ball")
    log.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(handler)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


# ---------- court loading ----------

def _load_court(court_path: Path, log: logging.Logger) -> dict:
    """Read court.json (Stage 1's output) and return:
        {'court_to_pixel_H': 3x3 ndarray, 'court_corners_ft': (4,2) ndarray,
         'image_size': (w, h) or None}.

    Stage 1 schema (verified against test_clip/court.json):
      - homography.court_to_image : 3x3 court->pixel matrix
      - homography.image_to_court : 3x3 pixel->court matrix (unused here)
      - court_geometry_feet.width_ft, .length_ft : court dimensions
      - video.width, video.height : optional, for image_size

    Court corners are SYNTHESIZED from width_ft/length_ft using the
    convention from KNOWN_ISSUES.md: corners at (0,0), (W,0), (W,L), (0,L)
    in court-coord feet, with W=width_ft and L=length_ft. The four corners
    are emitted in this order so the resulting pixel-space polygon is
    consistently wound (matters for cv2.pointPolygonTest only insofar as
    the polygon must be simple and closed; winding direction does not).
    """
    if not court_path.exists():
        fail(f"court.json not found: {court_path}", FileNotFoundError)
    try:
        with court_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        fail(f"court.json is not valid JSON ({court_path}): {e}", ValueError)

    # Homography: prefer court_to_image directly, fall back to inverting
    # image_to_court if only that direction is present.
    homog = data.get("homography")
    if not isinstance(homog, dict):
        fail(f"court.json: 'homography' must be an object with "
             f"court_to_image and/or image_to_court keys; got {type(homog).__name__}",
             ValueError)

    H_court_to_pixel = None
    if "court_to_image" in homog:
        H_court_to_pixel = np.array(homog["court_to_image"], dtype=np.float64)
        log.debug("using homography.court_to_image directly")
    elif "image_to_court" in homog:
        log.info("homography.court_to_image not present; inverting image_to_court")
        H_inv = np.array(homog["image_to_court"], dtype=np.float64)
        if H_inv.shape != (3, 3):
            fail(f"homography.image_to_court must be 3x3, got {H_inv.shape}",
                 ValueError)
        try:
            H_court_to_pixel = np.linalg.inv(H_inv)
        except np.linalg.LinAlgError as e:
            fail(f"failed to invert image_to_court homography: {e}", ValueError)
    else:
        fail(f"court.json.homography missing both court_to_image and "
             f"image_to_court. Found keys: {list(homog.keys())}",
             ValueError)

    if H_court_to_pixel.shape != (3, 3):
        fail(f"court_to_image homography must be 3x3, got "
             f"{H_court_to_pixel.shape}", ValueError)

    # Court geometry: synthesize corners from width_ft/length_ft
    geom = data.get("court_geometry_feet")
    if not isinstance(geom, dict):
        fail(f"court.json: 'court_geometry_feet' must be an object with "
             f"width_ft and length_ft; got {type(geom).__name__}",
             ValueError)
    width_ft = geom.get("width_ft")
    length_ft = geom.get("length_ft")
    if width_ft is None or length_ft is None:
        fail(f"court.json.court_geometry_feet missing width_ft and/or "
             f"length_ft. Found keys: {list(geom.keys())}", ValueError)
    if not (width_ft > 0 and length_ft > 0):
        fail(f"court dimensions must be positive; got "
             f"width_ft={width_ft}, length_ft={length_ft}", ValueError)
    corners = np.array([
        [0.0, 0.0],
        [float(width_ft), 0.0],
        [float(width_ft), float(length_ft)],
        [0.0, float(length_ft)],
    ], dtype=np.float64)
    log.debug(f"court corners (ft): {corners.tolist()}")

    # Image size: optional, used by some downstream consumers
    image_size = None
    video = data.get("video")
    if isinstance(video, dict):
        w = video.get("width")
        h = video.get("height")
        if w is not None and h is not None:
            image_size = (int(w), int(h))

    return {
        "court_to_pixel_H": H_court_to_pixel,
        "court_corners_ft": corners,
        "image_size": image_size,
    }


def _build_roi_polygon(court: dict, roi_buffer_ft: float,
                      log: logging.Logger) -> np.ndarray:
    """Expand the four court corners outward by `roi_buffer_ft` in court-coord
    space, project to pixel space, return a (4,2) float32 array suitable
    for cv2.pointPolygonTest."""
    corners = court["court_corners_ft"]  # (4, 2)
    cx, cy = corners.mean(axis=0)
    # Expand each corner away from the centroid by roi_buffer_ft in each axis
    expanded = np.empty_like(corners)
    for i, (x, y) in enumerate(corners):
        dx = x - cx
        dy = y - cy
        # Extend along each axis by buffer; sign preserves outward direction
        nx = x + np.sign(dx) * roi_buffer_ft if dx != 0 else x
        ny = y + np.sign(dy) * roi_buffer_ft if dy != 0 else y
        expanded[i] = [nx, ny]

    # Project to pixel space via court_to_pixel homography
    H = court["court_to_pixel_H"]
    homo = np.hstack([expanded, np.ones((4, 1))])  # (4, 3)
    proj = (H @ homo.T).T  # (4, 3)
    if np.any(np.abs(proj[:, 2]) < 1e-9):
        fail("ROI polygon projection has near-zero w; homography is degenerate",
             ValueError)
    pixel_corners = proj[:, :2] / proj[:, 2:3]
    pixel_corners = pixel_corners.astype(np.float32)

    # Sanity: polygon must not be self-intersecting (signed area sign-flip
    # check between consecutive triangles fanning from corner 0).
    def signed_area(poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    area = signed_area(pixel_corners)
    if abs(area) < 1.0:
        fail(f"ROI polygon has near-zero area ({area:.2f}px²); "
             f"homography may be degenerate", ValueError)

    log.info(f"ROI polygon (pixel space): {pixel_corners.tolist()}")
    log.info(f"ROI polygon area: {abs(area):.0f} px²")
    return pixel_corners


# ---------- video loading ----------

def _open_video(video_path: Path, log: logging.Logger):
    if not video_path.exists():
        fail(f"video not found: {video_path}", FileNotFoundError)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"OpenCV could not open video: {video_path}", RuntimeError)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n_frames < 3:
        cap.release()
        fail(f"video has {n_frames} frames; TrackNetV2 needs at least 3",
             ValueError)
    log.info(f"video: {n_frames} frames, {fps:.2f} fps, {w}x{h}")
    return cap, n_frames, fps, w, h


# ---------- model ----------

def _load_model(weights_path: Path, device: str,
                log: logging.Logger) -> TrackNet:
    if not weights_path.exists():
        fail(f"weights file not found: {weights_path}", FileNotFoundError)
    if device == "cuda" and not torch.cuda.is_available():
        fail("--device cuda requested but torch.cuda.is_available() is False",
             RuntimeError)

    model = TrackNet(in_channels=9, out_channels=3)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception as e:
        fail(f"failed to load weights from {weights_path}: {e}", RuntimeError)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        fail(f"state_dict mismatch loading {weights_path}: {e}", RuntimeError)
    model.to(device)
    model.eval()
    log.info(f"loaded weights from {weights_path} onto {device}")
    return model


def _preprocess_triple(frames: list, model_h: int, model_w: int,
                       device: str) -> torch.Tensor:
    """Resize 3 BGR frames to (model_h, model_w), convert to RGB, scale to
    [0,1], stack into a (1, 9, H, W) tensor."""
    chans = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (model_w, model_h),
                             interpolation=cv2.INTER_LINEAR)
        # HWC -> CHW, normalize to [0,1]
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        chans.append(t)
    stacked = torch.cat(chans, dim=0)  # (9, H, W)
    return stacked.unsqueeze(0).to(device)  # (1, 9, H, W)


def _heatmap_to_xy(heatmap: np.ndarray) -> tuple:
    """Argmax over a 2D heatmap. Returns (x, y, peak_value) in heatmap coords."""
    # heatmap shape: (H, W)
    flat_idx = int(np.argmax(heatmap))
    y, x = divmod(flat_idx, heatmap.shape[1])
    return x, y, float(heatmap[y, x])


# ---------- main inference loop ----------

def run_stage(args, log: logging.Logger) -> dict:
    t0 = time.time()
    video_path = args.video
    court_path = args.court
    weights_path = args.weights
    out_path = args.out

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.",
             FileExistsError)
    if not (0.0 <= args.detection_threshold <= 1.0):
        fail(f"--detection-threshold must be in [0.0, 1.0], "
             f"got {args.detection_threshold}", ValueError)
    if args.max_gap_frames < 0:
        fail(f"--max-gap-frames must be >= 0, got {args.max_gap_frames}",
             ValueError)
    if args.roi_buffer_ft < 0:
        fail(f"--roi-buffer-ft must be >= 0, got {args.roi_buffer_ft}",
             ValueError)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load court + build ROI
    court = _load_court(court_path, log)
    roi_polygon = _build_roi_polygon(court, args.roi_buffer_ft, log)

    # Open video
    cap, n_frames, fps, vw, vh = _open_video(video_path, log)

    # Load model
    model = _load_model(weights_path, args.device, log)

    # Inference loop with rolling 3-frame buffer
    log.info(f"running inference on {n_frames} frames "
             f"(threshold={args.detection_threshold})")
    sx = vw / MODEL_W
    sy = vh / MODEL_H

    rows = []
    n_filtered_threshold = 0
    n_filtered_roi = 0
    buffer = []

    try:
        from tqdm import tqdm
        progress = tqdm(total=n_frames, unit="frame", file=sys.stderr)
    except ImportError:
        progress = None

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            log.warning(f"frame {frame_idx}: read failed; treating as missing")
            rows.append({
                "frame_idx": frame_idx,
                "pixel_x": np.nan, "pixel_y": np.nan,
                "visible": False, "confidence": np.nan,
                "interpolated": False,
            })
            buffer = []  # break the rolling window on read error
            if progress: progress.update(1)
            continue

        buffer.append(frame)
        if len(buffer) > 3:
            buffer.pop(0)

        if len(buffer) < 3:
            # First two frames have no detection (insufficient history)
            rows.append({
                "frame_idx": frame_idx,
                "pixel_x": np.nan, "pixel_y": np.nan,
                "visible": False, "confidence": np.nan,
                "interpolated": False,
            })
            if progress: progress.update(1)
            continue

        # Forward pass on the triple
        x = _preprocess_triple(buffer, MODEL_H, MODEL_W, args.device)
        with torch.no_grad():
            y = model(x)  # (1, 3, H, W) — last channel is the prediction for frame_idx
        heatmap = y[0, 2].cpu().numpy()  # third heatmap = current frame

        px, py, conf = _heatmap_to_xy(heatmap)

        if conf < args.detection_threshold:
            n_filtered_threshold += 1
            rows.append({
                "frame_idx": frame_idx,
                "pixel_x": np.nan, "pixel_y": np.nan,
                "visible": False, "confidence": np.nan,
                "interpolated": False,
            })
            if progress: progress.update(1)
            continue

        # Scale heatmap coords back to original video resolution
        orig_x = px * sx
        orig_y = py * sy

        # ROI filter
        inside = cv2.pointPolygonTest(roi_polygon, (float(orig_x), float(orig_y)),
                                      False)
        if inside < 0:
            n_filtered_roi += 1
            rows.append({
                "frame_idx": frame_idx,
                "pixel_x": np.nan, "pixel_y": np.nan,
                "visible": False, "confidence": np.nan,
                "interpolated": False,
            })
            if progress: progress.update(1)
            continue

        rows.append({
            "frame_idx": frame_idx,
            "pixel_x": float(orig_x), "pixel_y": float(orig_y),
            "visible": True, "confidence": float(conf),
            "interpolated": False,
        })
        if progress: progress.update(1)

    if progress: progress.close()
    cap.release()

    # Gap fill
    rows, n_interp, max_gap = _fill_gaps(rows, args.max_gap_frames)

    # Stats
    n_visible = sum(1 for r in rows if r["visible"])
    n_interp_actual = sum(1 for r in rows if r["interpolated"])
    n_missing = n_frames - n_visible - n_interp_actual
    detection_rate = (n_visible + n_interp_actual) / n_frames if n_frames else 0.0
    log.info(f"detection: {n_visible} visible, {n_interp_actual} interpolated, "
             f"{n_missing} missing -> rate={detection_rate:.3f}")
    log.info(f"filtered: {n_filtered_threshold} by threshold, "
             f"{n_filtered_roi} by ROI; max gap observed: {max_gap}")

    if n_visible == 0:
        log.warning("zero detections produced; ball.parquet will be all-missing")

    # Write parquet
    df = pd.DataFrame(rows)
    df["schema_version"] = SCHEMA_VERSION
    df["frame_idx"] = df["frame_idx"].astype("int64")
    df["pixel_x"] = df["pixel_x"].astype("float64")
    df["pixel_y"] = df["pixel_y"].astype("float64")
    df["visible"] = df["visible"].astype(bool)
    df["confidence"] = df["confidence"].astype("float32")
    df["interpolated"] = df["interpolated"].astype(bool)
    df = df[["schema_version", "frame_idx", "pixel_x", "pixel_y",
             "visible", "confidence", "interpolated"]]
    df.to_parquet(out_path, engine="pyarrow", index=False)
    log.info(f"wrote {out_path} ({len(df)} rows)")

    # Write meta
    wall = time.time() - t0
    meta = {
        "schema_version": SCHEMA_VERSION,
        "video_path": str(video_path),
        "video_frame_count": n_frames,
        "video_fps": fps,
        "video_width": vw,
        "video_height": vh,
        "court_path": str(court_path),
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "device": args.device,
        "detection_threshold": args.detection_threshold,
        "max_gap_frames": args.max_gap_frames,
        "roi_buffer_ft": args.roi_buffer_ft,
        "stats": {
            "frames_visible": n_visible,
            "frames_interpolated": n_interp_actual,
            "frames_missing": n_missing,
            "detection_rate": detection_rate,
            "detections_filtered_by_threshold": n_filtered_threshold,
            "detections_filtered_by_roi": n_filtered_roi,
            "max_gap_observed_frames": max_gap,
        },
        "wall_time_seconds": wall,
        "stage_version": STAGE_VERSION,
        "completed_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    log.info(f"wrote {meta_path}")
    return meta


def _fill_gaps(rows: list, max_gap: int) -> tuple:
    """Walk rows, find runs of visible=False between two visible=True rows,
    and linearly interpolate pixel_x/pixel_y across runs of length <= max_gap.
    Returns (rows, n_interpolated, max_gap_observed)."""
    if max_gap == 0:
        max_gap_observed = 0
        in_gap = 0
        for r in rows:
            if not r["visible"]:
                in_gap += 1
                if in_gap > max_gap_observed:
                    max_gap_observed = in_gap
            else:
                in_gap = 0
        return rows, 0, max_gap_observed

    n_interp = 0
    max_gap_observed = 0
    i = 0
    n = len(rows)
    while i < n:
        if rows[i]["visible"]:
            i += 1
            continue
        # find end of this gap
        j = i
        while j < n and not rows[j]["visible"]:
            j += 1
        gap_len = j - i
        if gap_len > max_gap_observed:
            max_gap_observed = gap_len
        # bounded gap?
        left = i - 1
        right = j
        if (gap_len <= max_gap and left >= 0 and right < n
                and rows[left]["visible"] and rows[right]["visible"]):
            x0, y0 = rows[left]["pixel_x"], rows[left]["pixel_y"]
            x1, y1 = rows[right]["pixel_x"], rows[right]["pixel_y"]
            for k in range(i, j):
                t = (k - left) / (right - left)
                rows[k]["pixel_x"] = x0 + t * (x1 - x0)
                rows[k]["pixel_y"] = y0 + t * (y1 - y0)
                rows[k]["visible"] = False
                rows[k]["interpolated"] = True
                rows[k]["confidence"] = np.nan
                n_interp += 1
        i = j

    return rows, n_interp, max_gap_observed


# ---------- CLI ----------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 — track ball")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--court", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--detection-threshold", type=float, default=0.5,
                   dest="detection_threshold")
    p.add_argument("--max-gap-frames", type=int, default=5,
                   dest="max_gap_frames")
    p.add_argument("--roi-buffer-ft", type=float, default=8.0,
                   dest="roi_buffer_ft")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   dest="log_level")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    log = setup_logging(args.log_level)
    try:
        run_stage(args, log)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        log.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
Stage 4.5 — validate fine-tuned weights.

Run the Stage 4 inference path against a held-out validation video and
compare predictions against ground-truth ball_labels.json. Output a
validation_report.json with detection rate, false-positive rate, and
pixel-error statistics.

Inference logic mirrors stages/track_ball/track_ball.py exactly:
same frame triple construction, same RGB conversion, same resize, same
normalization, same channel 2 readout. Court-ROI filtering and gap
interpolation from Stage 4 are intentionally skipped here — we want to
measure raw model behavior on labeled frames, not the post-processed
output.

Usage:
    python -m stages.finetune_ball_model.validate \
        --weights data/models/tracknet_v2_finetuned_v1.pt \
        --video   data/outdoor/video.mp4 \
        --court   data/outdoor/court.json \
        --labels  data/outdoor/ball_labels.json \
        --out     data/training/validation_report.json
"""
import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from stages.track_ball._tracknet_model import TrackNet

SCHEMA_VERSION = 1
STAGE_VERSION = "0.1.0"

LABELS_SCHEMA_VERSION = 1

MODEL_H = 288
MODEL_W = 512

DETECTION_PX_THRESHOLDS = (10.0, 25.0)
FP_CONFIDENCE_THRESHOLD = 0.5


def fail(msg, exc_cls=RuntimeError):
    raise exc_cls(msg)


def setup_logging(level):
    log = logging.getLogger("validate")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


def _load_labels(path, log):
    if not path.exists():
        fail(f"ball_labels.json not found: {path}", FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    sv = data.get("schema_version")
    if sv != LABELS_SCHEMA_VERSION:
        fail(f"{path}: schema_version={sv}, expected {LABELS_SCHEMA_VERSION}",
             ValueError)
    for k in ("video_frame_count", "video_width", "video_height", "labels"):
        if k not in data:
            fail(f"{path}: missing field '{k}'", ValueError)
    log.info(f"loaded {len(data['labels'])} labels from {path}")
    return data


def _preprocess_triple(frames, device):
    """Match stages.track_ball._preprocess_triple exactly."""
    chans = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (MODEL_W, MODEL_H),
                             interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        chans.append(t)
    stacked = torch.cat(chans, dim=0)
    return stacked.unsqueeze(0).to(device)


def _heatmap_to_xy(heatmap):
    """Match stages.track_ball._heatmap_to_xy."""
    flat_idx = int(np.argmax(heatmap))
    y, x = divmod(flat_idx, heatmap.shape[1])
    return x, y, float(heatmap[y, x])


def _load_model(weights_path, device, log):
    if not weights_path.exists():
        fail(f"weights not found: {weights_path}", FileNotFoundError)
    if device == "cuda" and not torch.cuda.is_available():
        fail("--device cuda requested but no CUDA available", RuntimeError)
    model = TrackNet(in_channels=9, out_channels=3)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    log.info(f"loaded weights from {weights_path} onto {device}")
    return model


def run_stage(args, log):
    t0 = time.time()
    video_path = args.video
    court_path = args.court
    labels_path = args.labels
    weights_path = args.weights
    out_path = args.out

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.",
             FileExistsError)

    if not court_path.exists():
        fail(f"court.json not found: {court_path}", FileNotFoundError)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels_data = _load_labels(labels_path, log)
    label_by_idx = {lab["frame_idx"]: lab for lab in labels_data["labels"]}
    max_label_idx = max(label_by_idx.keys()) if label_by_idx else -1

    if not video_path.exists():
        fail(f"video not found: {video_path}", FileNotFoundError)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"OpenCV could not open video: {video_path}", RuntimeError)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"video: {n_frames} frames, {vw}x{vh}")

    if vw != labels_data["video_width"] or vh != labels_data["video_height"]:
        fail(f"video dimensions ({vw}x{vh}) do not match labels "
             f"({labels_data['video_width']}x{labels_data['video_height']})",
             ValueError)

    model = _load_model(weights_path, args.device, log)

    sx = vw / MODEL_W
    sy = vh / MODEL_H
    last_frame = min(n_frames - 1, max_label_idx)
    log.info(f"will scan frames 0..{last_frame} "
             f"({len(label_by_idx)} labeled)")

    try:
        from tqdm import tqdm
        progress = tqdm(total=last_frame + 1, unit="frame", file=sys.stderr)
    except ImportError:
        progress = None

    buffer = []
    predictions = {}  # frame_idx -> (pixel_x, pixel_y, confidence)

    for frame_idx in range(last_frame + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            log.warning(f"frame {frame_idx}: read failed; clearing buffer")
            buffer = []
            if progress: progress.update(1)
            continue
        buffer.append(frame)
        if len(buffer) > 3:
            buffer.pop(0)

        if frame_idx in label_by_idx and len(buffer) == 3:
            x = _preprocess_triple(buffer, args.device)
            with torch.no_grad():
                y = model(x)
            heatmap = y[0, 2].cpu().numpy()
            px, py, conf = _heatmap_to_xy(heatmap)
            orig_x = px * sx
            orig_y = py * sy
            predictions[frame_idx] = (orig_x, orig_y, conf)
        if progress: progress.update(1)
    if progress: progress.close()
    cap.release()

    # Compute metrics
    n_visible_total = 0
    n_invisible_total = 0
    n_visible_skipped = 0  # labeled visible but frame_idx < 2

    hits_by_threshold = {t: 0 for t in DETECTION_PX_THRESHOLDS}
    pixel_errors = []
    false_positives = 0
    confidences_visible = []
    confidences_invisible = []

    for fi, lab in label_by_idx.items():
        if lab["ball_visible"]:
            n_visible_total += 1
            if fi not in predictions:
                n_visible_skipped += 1
                continue
            orig_x, orig_y, conf = predictions[fi]
            gt_x = float(lab["pixel_x"])
            gt_y = float(lab["pixel_y"])
            err = float(np.hypot(orig_x - gt_x, orig_y - gt_y))
            pixel_errors.append(err)
            confidences_visible.append(conf)
            for thr in DETECTION_PX_THRESHOLDS:
                if err <= thr:
                    hits_by_threshold[thr] += 1
        else:
            n_invisible_total += 1
            if fi not in predictions:
                continue  # not counted; we can't say true/false without a prediction
            _, _, conf = predictions[fi]
            confidences_invisible.append(conf)
            if conf > FP_CONFIDENCE_THRESHOLD:
                false_positives += 1

    # Denominator for detection rate excludes labels we couldn't evaluate
    n_visible_evaluated = n_visible_total - n_visible_skipped
    n_invisible_evaluated = len(confidences_invisible)

    detection_rates = {}
    for thr in DETECTION_PX_THRESHOLDS:
        if n_visible_evaluated > 0:
            detection_rates[thr] = hits_by_threshold[thr] / n_visible_evaluated
        else:
            detection_rates[thr] = 0.0

    fp_rate = (false_positives / n_invisible_evaluated
               if n_invisible_evaluated > 0 else 0.0)

    median_err = float(np.median(pixel_errors)) if pixel_errors else float("nan")
    p95_err = (float(np.percentile(pixel_errors, 95))
               if pixel_errors else float("nan"))
    mean_conf_visible = (float(np.mean(confidences_visible))
                         if confidences_visible else float("nan"))
    mean_conf_invisible = (float(np.mean(confidences_invisible))
                           if confidences_invisible else float("nan"))

    wall = time.time() - t0

    report = {
        "schema_version": SCHEMA_VERSION,
        "stage_version": STAGE_VERSION,
        "weights_path": str(weights_path),
        "video_path": str(video_path),
        "labels_path": str(labels_path),
        "court_path": str(court_path),
        "n_labeled_frames": len(label_by_idx),
        "n_ball_visible": n_visible_total,
        "n_ball_invisible": n_invisible_total,
        "n_ball_visible_evaluated": n_visible_evaluated,
        "n_ball_invisible_evaluated": n_invisible_evaluated,
        "n_skipped_insufficient_history": n_visible_skipped,
        "detection_rate_at_10px": detection_rates[10.0],
        "detection_rate_at_25px": detection_rates[25.0],
        "false_positive_rate": fp_rate,
        "fp_confidence_threshold": FP_CONFIDENCE_THRESHOLD,
        "median_pixel_error": median_err,
        "p95_pixel_error": p95_err,
        "mean_confidence_on_visible": mean_conf_visible,
        "mean_confidence_on_invisible": mean_conf_invisible,
        "wall_time_seconds": wall,
        "evaluated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace(
            "+00:00", "Z"),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    log.info("=" * 60)
    log.info(f"detection_rate_at_10px : {detection_rates[10.0]:.3f}  "
             f"({hits_by_threshold[10.0]}/{n_visible_evaluated})")
    log.info(f"detection_rate_at_25px : {detection_rates[25.0]:.3f}  "
             f"({hits_by_threshold[25.0]}/{n_visible_evaluated})")
    log.info(f"false_positive_rate    : {fp_rate:.3f}  "
             f"({false_positives}/{n_invisible_evaluated})")
    log.info(f"median pixel error     : {median_err:.2f} px")
    log.info(f"p95 pixel error        : {p95_err:.2f} px")
    log.info(f"mean conf on visible   : {mean_conf_visible:.4f}")
    log.info(f"mean conf on invisible : {mean_conf_invisible:.4f}")
    log.info("=" * 60)
    log.info(f"wrote {out_path}")
    log.info(f"wall time: {wall:.1f}s")

    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 4.5 - validate")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--court", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   dest="log_level")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log = setup_logging(args.log_level)
    try:
        run_stage(args, log)
    except (FileNotFoundError, FileExistsError, ValueError,
            RuntimeError) as e:
        log.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
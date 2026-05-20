"""
Stage 4.5 diagnostic: visualize predicted heatmaps vs ground truth.

For each picked frame, runs the fine-tuned model and reports the top-K
heatmap peaks (with non-maximum suppression so we get distinct peaks,
not 5 pixels of the same blob). On ball_visible=true frames, also
reports the rank of the GT location among those K peaks — i.e. is the
real ball peak #1, #2, or not in the top K at all?

This is not part of the pipeline; it exists to diagnose Stage 4.5 v1
behavior before deciding what to change for v2 training.

Usage:
    python tools/diag_heatmaps.py \
        --weights data/models/tracknet_v2_finetuned_v1.pt \
        --video   data/outdoor/video.mp4 \
        --labels  data/outdoor/ball_labels.json \
        --out-dir data/training/diag_v1 \
        --n-visible 5 --n-invisible 5
"""
import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to sys.path so 'from stages.track_ball...' resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from stages.track_ball._tracknet_model import TrackNet

MODEL_H = 288
MODEL_W = 512

# Top-K analysis
TOPK = 5
# Minimum pixel separation between peaks in model-resolution space
NMS_RADIUS_MODEL = 8
# How close (in original-resolution pixels) a peak must be to GT to count
# as "the GT peak." Same threshold validate.py uses for "10px hit."
GT_MATCH_PX_ORIG = 10.0


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _preprocess_triple(frames, device):
    chans = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (MODEL_W, MODEL_H),
                             interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        chans.append(t)
    return torch.cat(chans, dim=0).unsqueeze(0).to(device)


def _load_model(weights_path, device):
    model = TrackNet(in_channels=9, out_channels=3)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _topk_peaks(heatmap, k, nms_radius):
    """Return up to k peaks as list of (x, y, value) in heatmap coords,
    sorted by value descending. Non-maximum suppression: after picking a
    peak, zero out all pixels within `nms_radius` before picking next."""
    h, w = heatmap.shape
    work = heatmap.copy()
    peaks = []
    for _ in range(k):
        flat = int(np.argmax(work))
        y, x = divmod(flat, w)
        val = float(work[y, x])
        if val <= 0:
            break
        peaks.append((x, y, val))
        # Zero out NMS neighborhood
        y0, y1 = max(0, y - nms_radius), min(h, y + nms_radius + 1)
        x0, x1 = max(0, x - nms_radius), min(w, x + nms_radius + 1)
        work[y0:y1, x0:x1] = 0
    return peaks


def _render_sample(frame_bgr, heatmap, lab, peaks, gt_rank, vw, vh, out_path):
    """Left panel: original frame with GT (green circle) and top-K predictions
    numbered 1..K (red). Right panel: heatmap with same numbered crosses."""
    left = frame_bgr.copy()
    sx = vw / MODEL_W
    sy = vh / MODEL_H

    # GT (visible only)
    if lab["ball_visible"]:
        gt_x = int(lab["pixel_x"])
        gt_y = int(lab["pixel_y"])
        cv2.circle(left, (gt_x, gt_y), 25, (0, 255, 0), 3)
        cv2.putText(left, "GT", (gt_x + 30, gt_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Top-K predictions, numbered
    for rank, (px, py, conf) in enumerate(peaks, start=1):
        orig_x = int(px * sx)
        orig_y = int(py * sy)
        color = (0, 0, 255) if rank == 1 else (0, 165, 255)  # red, orange
        cv2.drawMarker(left, (orig_x, orig_y), color,
                       markerType=cv2.MARKER_CROSS,
                       markerSize=40, thickness=3)
        cv2.putText(left, f"{rank}({conf:.2f})",
                    (orig_x + 25, orig_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    tag = (f"frame {lab['frame_idx']}  visible={lab['ball_visible']}  "
           f"top1={peaks[0][2]:.3f}")
    if lab["ball_visible"]:
        rank_str = ("not in top-K" if gt_rank is None
                    else f"GT is peak #{gt_rank}")
        tag += f"  ({rank_str})"
    cv2.rectangle(left, (0, 0), (left.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(left, tag, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Right: heatmap as jet colormap
    hm_norm = (heatmap / max(heatmap.max(), 1e-6) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    hm_color = cv2.resize(hm_color, (vw, vh),
                          interpolation=cv2.INTER_NEAREST)
    for rank, (px, py, conf) in enumerate(peaks, start=1):
        orig_x = int(px * sx)
        orig_y = int(py * sy)
        cv2.drawMarker(hm_color, (orig_x, orig_y), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS,
                       markerSize=40, thickness=3)
        cv2.putText(hm_color, str(rank),
                    (orig_x + 25, orig_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    combined = np.hstack([left, hm_color])
    out_h = 720
    out_w = int(combined.shape[1] * out_h / combined.shape[0])
    combined = cv2.resize(combined, (out_w, out_h),
                          interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), combined)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True, dest="out_dir")
    p.add_argument("--n-visible", type=int, default=5, dest="n_visible")
    p.add_argument("--n-invisible", type=int, default=5, dest="n_invisible")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    if not args.weights.exists():
        fail(f"weights not found: {args.weights}")
    if not args.video.exists():
        fail(f"video not found: {args.video}")
    if not args.labels.exists():
        fail(f"labels not found: {args.labels}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.labels.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)
    all_labels = labels_data["labels"]
    visible = [l for l in all_labels if l["ball_visible"] and l["frame_idx"] >= 2]
    invisible = [l for l in all_labels if not l["ball_visible"] and l["frame_idx"] >= 2]
    print(f"available: {len(visible)} visible, {len(invisible)} invisible")

    if len(visible) < args.n_visible or len(invisible) < args.n_invisible:
        fail(f"not enough labels")

    rng = random.Random(args.seed)
    picks = sorted(
        rng.sample(visible, args.n_visible) +
        rng.sample(invisible, args.n_invisible),
        key=lambda l: l["frame_idx"],
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        fail(f"OpenCV could not open video: {args.video}")
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx = vw / MODEL_W
    sy = vh / MODEL_H

    model = _load_model(args.weights, args.device)
    print(f"model loaded onto {args.device}")
    print(f"top-K = {TOPK}, NMS radius = {NMS_RADIUS_MODEL} px (model res)")
    print(f"GT match threshold = {GT_MATCH_PX_ORIG} px (original res)")
    print()

    max_target = max(l["frame_idx"] for l in picks)
    pick_idxs = {l["frame_idx"]: l for l in picks}

    buffer = []
    rank_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, None: 0}
    n_visible_processed = 0

    for frame_idx in range(max_target + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            buffer = []
            continue
        buffer.append(frame)
        if len(buffer) > 3:
            buffer.pop(0)
        if frame_idx in pick_idxs and len(buffer) == 3:
            lab = pick_idxs[frame_idx]
            x = _preprocess_triple(buffer, args.device)
            with torch.no_grad():
                y = model(x)
            heatmap = y[0, 2].cpu().numpy()
            peaks = _topk_peaks(heatmap, TOPK, NMS_RADIUS_MODEL)

            # Where does GT fall in the peak ranking? (visible frames only)
            gt_rank = None
            if lab["ball_visible"]:
                n_visible_processed += 1
                gt_x = float(lab["pixel_x"])
                gt_y = float(lab["pixel_y"])
                for rank, (px, py, _) in enumerate(peaks, start=1):
                    peak_orig_x = px * sx
                    peak_orig_y = py * sy
                    err = float(np.hypot(peak_orig_x - gt_x,
                                         peak_orig_y - gt_y))
                    if err <= GT_MATCH_PX_ORIG:
                        gt_rank = rank
                        break
                rank_counts[gt_rank] += 1

            tag = "VIS" if lab["ball_visible"] else "INV"
            out_path = args.out_dir / f"frame_{frame_idx:06d}_{tag}.png"
            _render_sample(frame, heatmap, lab, peaks, gt_rank,
                           vw, vh, out_path)

            # Console line
            top_str = " | ".join(
                f"#{r}=({int(px*sx)},{int(py*sy)},c={conf:.2f})"
                for r, (px, py, conf) in enumerate(peaks, start=1)
            )
            extra = ""
            if lab["ball_visible"]:
                extra = (f"  GT=({int(lab['pixel_x'])},{int(lab['pixel_y'])})"
                         f" rank={gt_rank if gt_rank else 'MISS'}")
            print(f"  frame {frame_idx} ({tag}): {top_str}{extra}")
    cap.release()

    print()
    print("=" * 60)
    print(f"GT-rank summary across {n_visible_processed} visible frames:")
    for r in (1, 2, 3, 4, 5):
        print(f"  GT was peak #{r}: {rank_counts[r]}")
    print(f"  GT not in top-{TOPK}: {rank_counts[None]}")
    print("=" * 60)
    print(f"wrote {len(picks)} diagnostic PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
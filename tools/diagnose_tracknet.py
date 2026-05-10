"""
Diagnostic: load TrackNet + weights, run inference on a single frame triple,
dump source frame + raw heatmap + overlay so you can see whether the
model is detecting the ball at all.

Usage:
    python tools\diagnose_tracknet.py `
        --video data\test_clip\video.mp4 `
        --weights data\models\tracknet_v2_dettor.pt `
        --frame 250 `
        --out data\test_clip\diagnostic

Writes (into --out folder):
    frame_<N>_source.jpg     — the input frame, full resolution
    frame_<N>_heatmap.jpg    — heatmap for the current frame, grayscale
    frame_<N>_overlay.jpg    — heatmap heatmap overlaid on source
    frame_<N>_stats.txt      — peak value, peak location, percentiles

If the heatmap shows a clear peak near the ball, weights are working
and the issue is downstream (threshold too high, ROI too small, etc.).
If the heatmap is uniform low values everywhere or peaks in nonsensical
places, the weights are not working as expected.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stages.track_ball._tracknet_model import TrackNet


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--frame", type=int, required=True,
                    help="Frame index to diagnose (0-based). Will read frames "
                         "[N-2, N-1, N] for the 3-frame TrackNet input.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output folder for diagnostic images")
    args = ap.parse_args()

    if not args.video.exists():
        fail(f"video not found: {args.video}")
    if not args.weights.exists():
        fail(f"weights not found: {args.weights}")
    if args.frame < 2:
        fail(f"--frame must be >= 2 (need 2 prior frames for 3-frame input)")

    args.out.mkdir(parents=True, exist_ok=True)

    # Load model
    model = TrackNet(in_channels=9, out_channels=3)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"loaded model from {args.weights}")

    # Read 3 consecutive frames
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        fail(f"could not open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total:
        fail(f"--frame {args.frame} >= total frames {total}")

    frames = []
    for idx in (args.frame - 2, args.frame - 1, args.frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok or f is None:
            fail(f"could not read frame {idx}")
        frames.append(f)
    cap.release()

    src_h, src_w = frames[2].shape[:2]
    print(f"source frame: {src_w}x{src_h}")

    # Resize each to 512x288, convert to RGB, scale to [0,1], stack to (1,9,288,512)
    MODEL_W, MODEL_H = 512, 288
    chans = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        chans.append(t)
    x = torch.cat(chans, dim=0).unsqueeze(0)
    print(f"model input shape: {tuple(x.shape)}")

    with torch.no_grad():
        y = model(x)
    print(f"model output shape: {tuple(y.shape)}")

    # Take the third heatmap (current frame's prediction)
    heatmap = y[0, 2].cpu().numpy()  # (288, 512)

    # Top-K peaks (non-maximum suppression with a small radius)
    def find_top_peaks(hm, k=5, suppress_radius=10):
        hm_work = hm.copy()
        peaks = []
        for _ in range(k):
            idx = int(np.argmax(hm_work))
            py, px = divmod(idx, hm_work.shape[1])
            val = float(hm_work[py, px])
            peaks.append((px, py, val))
            # Suppress a radius around this peak
            y0 = max(0, py - suppress_radius)
            y1 = min(hm_work.shape[0], py + suppress_radius + 1)
            x0 = max(0, px - suppress_radius)
            x1 = min(hm_work.shape[1], px + suppress_radius + 1)
            hm_work[y0:y1, x0:x1] = 0.0
        return peaks

    top_peaks = find_top_peaks(heatmap, k=5, suppress_radius=10)
    print(f"top 5 peaks (after NMS, suppress radius 10):")
    for i, (px, py, val) in enumerate(top_peaks):
        sx_pk = src_w / MODEL_W
        sy_pk = src_h / MODEL_H
        print(f"  peak {i+1}: heatmap ({px}, {py})  -> "
              f"source ({px*sx_pk:.0f}, {py*sy_pk:.0f})  "
              f"value={val:.4f}")

    # Stats
    peak_val = float(heatmap.max())
    peak_idx = int(np.argmax(heatmap))
    peak_y, peak_x = divmod(peak_idx, heatmap.shape[1])
    # Scale peak to original frame coords
    sx = src_w / MODEL_W
    sy = src_h / MODEL_H
    src_peak_x = peak_x * sx
    src_peak_y = peak_y * sy

    pct = {f"p{p}": float(np.percentile(heatmap, p))
           for p in (50, 75, 90, 95, 99, 99.9)}

    print(f"heatmap stats:")
    print(f"  shape: {heatmap.shape}")
    print(f"  min={heatmap.min():.6f} max={heatmap.max():.6f} mean={heatmap.mean():.6f}")
    print(f"  percentiles: {pct}")
    print(f"  peak at heatmap ({peak_x}, {peak_y}) — "
          f"source frame ({src_peak_x:.0f}, {src_peak_y:.0f})")

    # Save outputs
    base = args.out / f"frame_{args.frame:05d}"

    # 1. Source frame
    cv2.imwrite(f"{base}_source.jpg", frames[2], [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 2. Heatmap as grayscale, upscaled to original resolution
    hm_255 = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
    hm_full = cv2.resize(hm_255, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"{base}_heatmap.jpg", hm_full, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 3. Overlay (50/50 blend of color heatmap + source)
    hm_color = cv2.applyColorMap(hm_full, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frames[2], 0.5, hm_color, 0.5, 0)
    # Mark peak with a circle
    cv2.circle(overlay, (int(src_peak_x), int(src_peak_y)), 20, (0, 255, 255), 3)
    cv2.imwrite(f"{base}_overlay.jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 4. Stats text
    with open(f"{base}_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"frame: {args.frame}\n")
        f.write(f"video: {args.video}\n")
        f.write(f"weights: {args.weights}\n")
        f.write(f"\n")
        f.write(f"heatmap shape: {heatmap.shape}\n")
        f.write(f"min: {heatmap.min():.6f}\n")
        f.write(f"max: {heatmap.max():.6f}\n")
        f.write(f"mean: {heatmap.mean():.6f}\n")
        for k, v in pct.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write(f"\n")
        f.write(f"peak heatmap coords: ({peak_x}, {peak_y})\n")
        f.write(f"peak source coords: ({src_peak_x:.0f}, {src_peak_y:.0f})\n")
        f.write(f"peak value: {peak_val:.6f}\n")

    print(f"\nwrote diagnostic outputs to {args.out}")
    print(f"  inspect {base}_overlay.jpg — yellow circle marks the predicted peak.")
    print(f"  if the circle lands on or near the ball, weights are working.")


if __name__ == "__main__":
    main()
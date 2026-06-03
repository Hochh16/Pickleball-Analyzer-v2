"""Stage 4.5 v4 — prepare a training frame-cache for one clip.

Pre-extracts the frames needed for training (each labeled center frame +/- the
3-frame-stack neighbors) at the 720p processing resolution as JPEGs, plus a
v4_manifest.json describing each sample. Training then reads small JPEGs
(fast, fork-safe DataLoader, ~1-2 GB to upload to Colab) instead of seeking
through 15 GB of 4K video.

Reads frames SEQUENTIALLY over the needed range (4K random seeks are slow).

Usage:
    python -m stages.finetune_ball_model.prepare_v4 data/pb_2min --clip pb_2min
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from stages.finetune_ball_model._v4_data import (
    densify_labels, PROC_H, PROC_W, FRAME_STRIDE)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage 4.5 v4 — frame-cache prep")
    p.add_argument("folder", type=Path)
    p.add_argument("--clip", default=None, help="clip name (default folder name)")
    p.add_argument("--stride", type=int, default=FRAME_STRIDE)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    clip = args.clip or args.folder.name
    label_path = args.folder / "ball_labels.json"
    if not label_path.exists():
        print(f"no labels: {label_path}")
        return 1
    d = json.loads(label_path.read_text(encoding="utf-8"))
    video_path = d.get("video_path") or str(args.folder / "video.mp4")
    if not Path(video_path).exists():
        # fall back to the local copy if the recorded path isn't reachable
        local = args.folder / "video.mp4"
        if local.exists():
            video_path = str(local)
        else:
            print(f"video not found: {video_path}")
            return 1
    src_w = int(d.get("video_width", 0)) or 3840
    src_h = int(d.get("video_height", 0)) or 2160

    dens = densify_labels(d["labels"])
    # build samples: keep those whose full 3-frame window is in-range
    samples = []
    needed = set()
    for l in dens:
        c = int(l["frame_idx"])
        frames = [c - args.stride, c, c + args.stride]
        if frames[0] < 0:
            continue
        vis = bool(l.get("ball_visible")) and l.get("pixel_x") is not None
        samples.append({
            "center": c, "frames": frames, "visible": vis,
            "x_proc": (float(l["pixel_x"]) * PROC_W / src_w) if vis else None,
            "y_proc": (float(l["pixel_y"]) * PROC_H / src_h) if vis else None,
        })
        needed.update(frames)
    if not needed:
        print("no usable samples")
        return 1

    frames_dir = args.folder / "frames_720"
    frames_dir.mkdir(exist_ok=True)
    lo, hi = min(needed), max(needed)
    print(f"{clip}: {len(samples)} samples, {len(needed)} unique frames "
          f"to extract over [{lo},{hi}] from {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open {video_path}")
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    written = 0
    for idx in range(lo, hi + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if idx in needed:
            out = frames_dir / f"{idx}.jpg"
            if args.force or not out.exists():
                small = cv2.resize(fr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out), small,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            written += 1
            if written % 200 == 0:
                print(f"  {written}/{len(needed)} frames...")
    cap.release()

    manifest = {
        "schema_version": 1, "clip": clip, "proc_h": PROC_H, "proc_w": PROC_W,
        "stride": args.stride, "frames_dir": "frames_720",
        "src_w": src_w, "src_h": src_h, "n_samples": len(samples),
        "n_visible": sum(1 for s in samples if s["visible"]),
        "samples": samples,
    }
    (args.folder / "v4_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.folder/'v4_manifest.json'} "
          f"({len(samples)} samples, {manifest['n_visible']} visible); "
          f"extracted {written} JPEGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

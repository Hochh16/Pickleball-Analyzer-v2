"""Compress an annotated render into a web-friendly clip for the report.

Downscales annotated.mp4 (source-resolution, large) to 720p and, for 60fps
sources, halves the frame rate, writing annotated_web.mp4 next to it. build_report
prefers annotated_web.mp4 when present. Uses OpenCV only (no ffmpeg dependency);
mp4v is not a highly efficient codec, so the win comes mostly from the resolution
and frame-rate reduction (pb_2min: ~460 MB 4K/60 -> ~50 MB 720p/30).

Usage:
    python -m tools.compress_video data/pb_2min
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

TARGET_H = 720


def compress(src: Path, dst: Path) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"cannot open {src}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip = 2 if fps >= 50 else 1                       # 60 -> 30 fps
    out_h = min(TARGET_H, h)
    out_w = int(round(w * out_h / h)) // 2 * 2         # keep aspect, even width
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / skip, (out_w, out_h))
    i = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % skip == 0:
            writer.write(cv2.resize(frame, (out_w, out_h),
                                    interpolation=cv2.INTER_AREA))
            written += 1
        i += 1
        if i % 1200 == 0:
            print(f"  {i}/{n} frames ({written} written)", flush=True)
    cap.release()
    writer.release()
    mb = dst.stat().st_size // (1024 * 1024)
    print(f"wrote {dst} — {written} frames @ {out_w}x{out_h} {fps/skip:.0f}fps ({mb} MB)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compress annotated.mp4 -> annotated_web.mp4")
    p.add_argument("folder", type=Path)
    p.add_argument("--src", default="annotated.mp4")
    p.add_argument("--out", default="annotated_web.mp4")
    args = p.parse_args(argv)
    src = args.folder / args.src
    if not src.exists():
        print(f"{src} not found (run Stage 11 render first)", file=sys.stderr)
        return 1
    return compress(src, args.folder / args.out)


if __name__ == "__main__":
    sys.exit(main())

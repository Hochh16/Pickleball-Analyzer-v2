"""Compress an annotated render into a web-friendly clip for the report.

Downscales annotated.mp4 (source-resolution, large) to 720p and, for 60fps
sources, halves the frame rate, writing annotated_web.mp4 next to it. build_report
prefers annotated_web.mp4 when present.

Encodes **H.264 (yuv420p) + faststart** via ffmpeg so the clip plays inline in a
browser `<video>` tag. ffmpeg comes from the `imageio-ffmpeg` wheel (a portable
static build with libx264) — no system ffmpeg needed. If ffmpeg can't be found,
falls back to OpenCV `mp4v`, which is smaller effort but **does NOT play in most
browsers** (blank video) — a warning is printed in that case.

Usage:
    python -m tools.compress_video data/pb_2min
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

TARGET_H = 720


def _probe(src: Path):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n, height = h
    cap.release()
    return fps, n, height


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _compress_ffmpeg(exe: str, src: Path, dst: Path, fps: float) -> int:
    out_fps = fps / 2 if fps >= 50 else fps           # 60 -> 30
    # scale to <=720p keeping aspect (even dims); H.264 yuv420p + faststart for web.
    vf = f"scale=-2:'min({TARGET_H},ih)',fps={out_fps:.4f}"
    cmd = [
        exe, "-y", "-loglevel", "error", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}", file=sys.stderr)
        return proc.returncode
    mb = dst.stat().st_size // (1024 * 1024)
    print(f"wrote {dst} — H.264 720p@{out_fps:.0f}fps ({mb} MB, browser-playable)")
    return 0


def _compress_opencv(src: Path, dst: Path) -> int:
    """Fallback: OpenCV mp4v. Smaller effort but usually will NOT play in a browser."""
    print("WARNING: ffmpeg unavailable — writing mp4v, which may not play in a browser.",
          file=sys.stderr)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"cannot open {src}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip = 2 if fps >= 50 else 1
    out_h = min(TARGET_H, h)
    out_w = int(round(w * out_h / h)) // 2 * 2
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / skip, (out_w, out_h))
    i = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % skip == 0:
            writer.write(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA))
            written += 1
        i += 1
    cap.release()
    writer.release()
    mb = dst.stat().st_size // (1024 * 1024)
    print(f"wrote {dst} — {written} frames @ {out_w}x{out_h} {fps/skip:.0f}fps ({mb} MB, mp4v)")
    return 0


def compress(src: Path, dst: Path) -> int:
    probe = _probe(src)
    if probe is None:
        print(f"cannot open {src}", file=sys.stderr)
        return 1
    fps, _n, _h = probe
    exe = _ffmpeg_exe()
    if exe:
        rc = _compress_ffmpeg(exe, src, dst, fps)
        if rc == 0:
            return 0
        print("falling back to OpenCV mp4v…", file=sys.stderr)
    return _compress_opencv(src, dst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compress annotated.mp4 -> annotated_web.mp4 (H.264)")
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

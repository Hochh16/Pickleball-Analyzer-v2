"""Video access for the setup wizard.

Serves EXACT source frames by index (via OpenCV, the same indexing the pipeline
uses) as downscaled JPEGs for in-browser marking. The frontend maps display
clicks back to source pixels using the source dimensions carried in the session
meta, so every marked coordinate is in original-video pixel space — exactly what
markers.json / user_clicks.json require.

A per-path OpenCV VideoCapture is reused under a lock (cv2 capture objects are
not thread-safe and FastAPI runs sync endpoints in a threadpool). A small
LRU-ish cache of encoded JPEGs keeps scrubbing snappy.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

# Extensions we treat as videos in the file browser.
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpg", ".mpeg", ".webm"}

_JPEG_QUALITY = 82
_FRAME_CACHE_MAX = 24  # encoded frames kept in memory (per process)


class VideoError(RuntimeError):
    """Raised when a video can't be opened or a frame can't be read."""


def probe(video_path: Path) -> Dict:
    """Read dimensions, fps, frame count and duration from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise VideoError(f"Video has invalid dimensions: {width}x{height}")
    if fps <= 0:
        fps = 30.0  # some containers don't report fps; downstream default
    duration_sec = (frame_count / fps) if (frame_count > 0 and fps > 0) else 0.0
    return {
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


class _CaptureHandle:
    """A lock-guarded, reusable VideoCapture for one video path."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        # (frame_idx, max_w) -> jpeg bytes
        self._cache: "OrderedDict[tuple, bytes]" = OrderedDict()

    def _cap_open(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise VideoError(f"Could not open video: {self.path}")
        return self._cap

    def frame_jpeg(self, frame_idx: int, max_w: int) -> bytes:
        key = (int(frame_idx), int(max_w))
        with self.lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit

            cap = self._cap_open()
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            idx = frame_idx
            if n > 0:
                idx = max(0, min(n - 1, idx))
            else:
                idx = max(0, idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise VideoError(f"Could not read frame {idx} of {self.path}")

            h, w = frame.shape[:2]
            if max_w > 0 and w > max_w:
                scale = max_w / float(w)
                frame = cv2.resize(
                    frame,
                    (int(round(w * scale)), int(round(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
            )
            if not ok:
                raise VideoError("Failed to JPEG-encode frame")
            data = buf.tobytes()

            self._cache[key] = data
            self._cache.move_to_end(key)
            while len(self._cache) > _FRAME_CACHE_MAX:
                self._cache.popitem(last=False)
            return data

    def release(self) -> None:
        with self.lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._cache.clear()


class FrameServer:
    """Process-wide registry of open video handles keyed by resolved path."""

    def __init__(self) -> None:
        self._handles: Dict[str, _CaptureHandle] = {}
        self._lock = threading.Lock()

    def _handle(self, video_path: Path) -> _CaptureHandle:
        key = str(video_path.resolve())
        with self._lock:
            h = self._handles.get(key)
            if h is None:
                h = _CaptureHandle(video_path)
                self._handles[key] = h
            return h

    def frame_jpeg(self, video_path: Path, frame_idx: int, max_w: int = 1600) -> bytes:
        return self._handle(video_path).frame_jpeg(frame_idx, max_w)

    def release(self, video_path: Path) -> None:
        key = str(video_path.resolve())
        with self._lock:
            h = self._handles.pop(key, None)
        if h is not None:
            h.release()

    def release_all(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for h in handles:
            h.release()


# Single shared instance used by the server.
frame_server = FrameServer()

"""Stage 4.5 v4 — data pipeline for the TrackNet ball detector.

Pure, importable, CPU-testable. Builds per-frame training samples from the
sampled ball_labels.json files:
  - densify: interpolate ball position between consecutive visible labels so
    every frame in a rally gets a label (clicks understate effective labels)
  - 3-frame channel-stacked input (9ch) at the processing resolution
  - Gaussian heatmap target (peak exactly 1.0 at the ball) for the CENTER frame
  - clip-based train/val/test split (hold out a whole clip)

The Colab training notebook imports this; the local CPU sanity check exercises
it on data/pb_2min/ without a GPU.

See stages/finetune_ball_model/contract_v4.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Processing resolution (decision: start 720p; escalate to 1080p if recall low)
PROC_H, PROC_W = 720, 1280
HEATMAP_SIGMA_PX = 3.0      # Gaussian std at processing res (~ball radius)
FRAME_STRIDE = 1            # gap between the 3 stacked frames (tunable)


@dataclass
class Sample:
    """One training sample: a labeled center frame + its clip."""
    clip: str
    video_path: str
    center_frame: int
    visible: bool
    # ball position in SOURCE pixels (None if not visible)
    x: Optional[float]
    y: Optional[float]
    src_w: int
    src_h: int


# --- Densification -----------------------------------------------------------

def densify_labels(labels: List[dict], max_gap: int = 4) -> List[dict]:
    """Interpolate ball position between consecutive VISIBLE labels whose frame
    gap is <= max_gap (the path is locally smooth). Returns per-frame labels:
    the originals plus interpolated in-betweens. Not-visible labels pass through
    unchanged (and block interpolation across them)."""
    out: List[dict] = []
    prev = None
    for lab in labels:
        out.append(dict(lab))
        if lab.get("ball_visible") and lab.get("pixel_x") is not None:
            if prev is not None:
                fg = lab["frame_idx"] - prev["frame_idx"]
                if 1 < fg <= max_gap:
                    for k in range(1, fg):
                        t = k / fg
                        out.append({
                            "frame_idx": prev["frame_idx"] + k,
                            "ball_visible": True,
                            "pixel_x": prev["pixel_x"] + t * (lab["pixel_x"] - prev["pixel_x"]),
                            "pixel_y": prev["pixel_y"] + t * (lab["pixel_y"] - prev["pixel_y"]),
                            "interpolated": True,
                        })
            prev = lab
        else:
            prev = None
    out.sort(key=lambda l: l["frame_idx"])
    return out


def load_samples(label_path: Path, clip: str, densify: bool = True) -> List[Sample]:
    d = json.loads(Path(label_path).read_text(encoding="utf-8"))
    src_w = int(d.get("video_width", 0)) or 3840
    src_h = int(d.get("video_height", 0)) or 2160
    vpath = d.get("video_path", "")
    labs = d["labels"]
    if densify:
        labs = densify_labels(labs)
    samples = []
    for l in labs:
        vis = bool(l.get("ball_visible")) and l.get("pixel_x") is not None
        samples.append(Sample(clip, vpath, int(l["frame_idx"]), vis,
                              (float(l["pixel_x"]) if vis else None),
                              (float(l["pixel_y"]) if vis else None),
                              src_w, src_h))
    return samples


# --- Heatmap target ----------------------------------------------------------

def make_heatmap(x_proc: float, y_proc: float, h: int = PROC_H, w: int = PROC_W,
                 sigma: float = HEATMAP_SIGMA_PX) -> np.ndarray:
    """2D Gaussian heatmap, peak EXACTLY 1.0 at (x_proc, y_proc). Empty (zeros)
    if the position is None/out of frame."""
    hm = np.zeros((h, w), dtype=np.float32)
    if x_proc is None or y_proc is None:
        return hm
    ix, iy = int(round(x_proc)), int(round(y_proc))
    if not (0 <= ix < w and 0 <= iy < h):
        return hm
    r = int(3 * sigma)
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    xs = np.arange(x0, x1)[None, :]
    ys = np.arange(y0, y1)[:, None]
    g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma ** 2))
    hm[y0:y1, x0:x1] = g.astype(np.float32)
    hm[iy, ix] = 1.0  # exact peak (focal loss treats ==1 as positive)
    return hm


# --- Frame reading + sample tensor ------------------------------------------

class VideoFrameReader:
    """Caches a single VideoCapture per video path for sequential-ish reads."""
    def __init__(self):
        self._caps: Dict[str, cv2.VideoCapture] = {}

    def get(self, path: str) -> cv2.VideoCapture:
        if path not in self._caps:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video: {path}")
            self._caps[path] = cap
        return self._caps[path]

    def read_proc(self, path: str, frame_idx: int) -> Optional[np.ndarray]:
        cap = self.get(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, fr = cap.read()
        if not ok:
            return None
        return cv2.resize(fr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

    def release(self):
        for c in self._caps.values():
            c.release()
        self._caps.clear()


def build_input_stack(reader: VideoFrameReader, s: Sample,
                      stride: int = FRAME_STRIDE) -> Optional[np.ndarray]:
    """Read frames [t-stride, t, t+stride], resize to proc res, return a
    (9, H, W) float32 array in [0,1] (RGB, channel-stacked). None on read fail."""
    frames = []
    for df in (-stride, 0, stride):
        fr = reader.read_proc(s.video_path, s.center_frame + df)
        if fr is None:
            return None
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(rgb)
    stack = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
    return stack.astype(np.float32)


def sample_target(s: Sample) -> np.ndarray:
    """Center-frame heatmap target, ball mapped SOURCE->proc resolution."""
    if not s.visible:
        return make_heatmap(None, None)
    xp = s.x * (PROC_W / s.src_w)
    yp = s.y * (PROC_H / s.src_h)
    return make_heatmap(xp, yp)


# --- Clip split --------------------------------------------------------------

def clip_split(samples: List[Sample], holdout_clip: str
               ) -> Tuple[List[Sample], List[Sample]]:
    """Split into (train, holdout) by clip name. The held-out clip is the
    cross-background generalization test."""
    train = [s for s in samples if s.clip != holdout_clip]
    held = [s for s in samples if s.clip == holdout_clip]
    return train, held

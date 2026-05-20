"""
Shared CV ball-detection pipeline for Stage 4.5 (calibration/validation)
and the rewritten Stage 4 (per-video inference).

Algorithm:
  1. Compute a video-level median background image from N sampled frames.
     Cached in BackgroundModel; computed once per video.
  2. For each input frame:
     a. abs-diff against background, gaussian blur, binarize.
     b. find connected components inside the court-ROI polygon.
     c. filter components by area and circularity.
     d. score each surviving candidate via weighted sum of:
        - motion: distance from previous prediction within plausible range
        - circularity: how round the blob is
        - color: HSV closeness to expected ball color (median +- tolerance)
     e. return the highest-scoring candidate, or None.

First-frame handling: no motion component (no previous prediction).
Score collapses to circularity + color only. Confidence is reduced
to reflect missing motion signal. Stage 4's gap-fill is responsible
for stitching across early-frame misses.

Failures are loud: degenerate court geometry, missing parameters, or
malformed video raise RuntimeError. No silent fallbacks.

Schema version: ball_cv_params.json must have schema_version=1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


PARAMS_SCHEMA_VERSION = 1
COURT_SCHEMA_REQUIRED_KEYS = ("homography", "court_geometry_feet")


# --------------------------------------------------------------------- params

@dataclass
class BallCVParams:
    """Tuned parameters from ball_cv_params.json, decoded into a struct."""
    schema_version: int
    video_path: str
    video_width: int
    video_height: int
    video_fps: float
    background_method: str
    background_n_frames: int
    bg_subtraction_threshold: int
    blob_area_px_min: float
    blob_area_px_max: float
    blob_circularity_min: float
    ball_color_hsv_median: list  # [H, S, V]
    ball_color_hsv_tolerance: list  # [H_tol, S_tol, V_tol]
    motion_displacement_px_per_frame_min: float
    motion_displacement_px_per_frame_max: float
    calibration_method: str
    n_calibration_frames_used: int
    calibration_completed_at_utc: str
    stage_version: str

    @classmethod
    def load(cls, path: Path) -> "BallCVParams":
        if not path.exists():
            raise FileNotFoundError(f"ball_cv_params.json not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sv = data.get("schema_version")
        if sv != PARAMS_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version={sv}, expected "
                f"{PARAMS_SCHEMA_VERSION}"
            )
        try:
            return cls(**{k: data[k] for k in cls.__dataclass_fields__})
        except KeyError as e:
            raise ValueError(
                f"{path}: missing required field {e}"
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: getattr(self, k) for k in self.__dataclass_fields__}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


# ----------------------------------------------------------------- court ROI

def load_court_roi(court_path: Path) -> np.ndarray:
    """Return the 4-corner court polygon in pixel space as a (4,2) float32
    array. Same convention as Stage 4 / Stage 1: corners derived from
    width_ft x length_ft, projected via homography.court_to_image."""
    if not court_path.exists():
        raise FileNotFoundError(f"court.json not found: {court_path}")
    with court_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for k in COURT_SCHEMA_REQUIRED_KEYS:
        if k not in data:
            raise ValueError(f"{court_path}: missing '{k}'")

    homog = data["homography"]
    if "court_to_image" in homog:
        H = np.array(homog["court_to_image"], dtype=np.float64)
    elif "image_to_court" in homog:
        H_inv = np.array(homog["image_to_court"], dtype=np.float64)
        H = np.linalg.inv(H_inv)
    else:
        raise ValueError(
            f"{court_path}: homography needs court_to_image or image_to_court"
        )
    if H.shape != (3, 3):
        raise ValueError(f"{court_path}: homography not 3x3")

    geom = data["court_geometry_feet"]
    w_ft = float(geom["width_ft"])
    l_ft = float(geom["length_ft"])
    if w_ft <= 0 or l_ft <= 0:
        raise ValueError(f"{court_path}: non-positive court dimensions")

    corners_ft = np.array([
        [0.0, 0.0],
        [w_ft, 0.0],
        [w_ft, l_ft],
        [0.0, l_ft],
    ])
    homo = np.hstack([corners_ft, np.ones((4, 1))])
    proj = (H @ homo.T).T
    if np.any(np.abs(proj[:, 2]) < 1e-9):
        raise ValueError(f"{court_path}: degenerate projection")
    pixel_corners = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
    return pixel_corners


def expand_roi(corners: np.ndarray, buffer_px: float) -> np.ndarray:
    """Expand the 4-corner polygon outward by buffer_px from its centroid.
    Used to give the detector a little margin beyond the strict court
    edges (the ball can briefly fly over a sideline mid-rally)."""
    cx, cy = corners.mean(axis=0)
    expanded = corners.copy()
    for i, (x, y) in enumerate(corners):
        dx, dy = x - cx, y - cy
        norm = np.hypot(dx, dy)
        if norm > 0:
            expanded[i, 0] = x + (dx / norm) * buffer_px
            expanded[i, 1] = y + (dy / norm) * buffer_px
    return expanded.astype(np.float32)


# ---------------------------------------------------------- background model

class BackgroundModel:
    """Holds the median-background image for a video. Computed once at
    startup from N sampled frames. The frame is BGR uint8, same shape
    as the source video."""

    def __init__(self, background_bgr: np.ndarray):
        if background_bgr.ndim != 3 or background_bgr.shape[2] != 3:
            raise ValueError(
                f"background must be (H, W, 3); got {background_bgr.shape}"
            )
        if background_bgr.dtype != np.uint8:
            raise ValueError(
                f"background must be uint8; got {background_bgr.dtype}"
            )
        self.background = background_bgr

    @classmethod
    def from_video(cls, video_path: Path, n_frames: int = 100,
                   method: str = "median") -> "BackgroundModel":
        if method != "median":
            raise ValueError(f"unsupported background method: {method}")
        if n_frames < 5:
            raise ValueError(f"need at least 5 sample frames; got {n_frames}")
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < n_frames:
            raise RuntimeError(
                f"video has {total} frames; need >= {n_frames} for background"
            )

        # Sample frames at roughly even intervals
        sample_indices = np.linspace(
            0, total - 1, num=n_frames, dtype=int
        )
        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(frame)
        cap.release()

        if len(frames) < n_frames * 0.8:
            raise RuntimeError(
                f"could only read {len(frames)} of {n_frames} sample "
                f"frames from {video_path}"
            )

        stack = np.stack(frames, axis=0).astype(np.uint8)
        background = np.median(stack, axis=0).astype(np.uint8)
        return cls(background)


# -------------------------------------------------------- detection per frame

@dataclass
class BallDetection:
    pixel_x: float
    pixel_y: float
    confidence: float
    blob_area_px: float
    blob_circularity: float


def _make_roi_mask(image_shape: tuple, polygon: np.ndarray) -> np.ndarray:
    """Return uint8 mask (H, W) with 255 inside the polygon, 0 outside."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def _candidate_blobs(fg_bin: np.ndarray, roi_mask: np.ndarray,
                     params: BallCVParams) -> list:
    """Find connected components in fg_bin that pass area and circularity
    filters, restricted to the ROI mask. Returns list of dicts."""
    # Restrict foreground to ROI
    fg_in_roi = cv2.bitwise_and(fg_bin, roi_mask)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_in_roi, connectivity=8
    )

    candidates = []
    for label in range(1, n_labels):  # 0 is background
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < params.blob_area_px_min or area > params.blob_area_px_max:
            continue
        # Circularity = 4*pi*area / perimeter^2
        # Approximate perimeter by counting edge pixels of this component
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], closed=True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < params.blob_circularity_min:
            continue
        cx, cy = centroids[label]
        candidates.append({
            "pixel_x": float(cx),
            "pixel_y": float(cy),
            "area": area,
            "circularity": float(circularity),
        })
    return candidates


def _color_score(frame_bgr: np.ndarray, x: float, y: float,
                 params: BallCVParams) -> float:
    """Sample HSV at the candidate point (3x3 median to reduce noise) and
    score by how close it is to params.ball_color_hsv_median, with hue
    treated cyclically. Returns score in [0, 1]."""
    h, w = frame_bgr.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - 1), min(w, xi + 2)
    y0, y1 = max(0, yi - 1), min(h, yi + 2)
    patch_bgr = frame_bgr[y0:y1, x0:x1]
    if patch_bgr.size == 0:
        return 0.0
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h_med = float(np.median(patch_hsv[:, :, 0]))
    s_med = float(np.median(patch_hsv[:, :, 1]))
    v_med = float(np.median(patch_hsv[:, :, 2]))

    target_h, target_s, target_v = params.ball_color_hsv_median
    tol_h, tol_s, tol_v = params.ball_color_hsv_tolerance

    # Hue is cyclic on [0, 180] for OpenCV
    dh = abs(h_med - target_h)
    if dh > 90:
        dh = 180 - dh
    ds = abs(s_med - target_s)
    dv = abs(v_med - target_v)

    # Score components in [0, 1]: 1.0 if within tolerance, decay outside
    def comp(d, tol):
        if tol <= 0:
            return 1.0 if d == 0 else 0.0
        if d <= tol:
            return 1.0
        return max(0.0, 1.0 - (d - tol) / (tol * 2))

    return (comp(dh, tol_h) + comp(ds, tol_s) + comp(dv, tol_v)) / 3.0


def _motion_score(x: float, y: float, prev_xy: Optional[tuple],
                  params: BallCVParams) -> float:
    """Score motion plausibility. If prev_xy is None, returns 1.0 (neutral
    -- we don't penalize the first frame for lacking a prior). Otherwise
    rewards displacements within the configured min/max range."""
    if prev_xy is None:
        return 1.0
    dx = x - prev_xy[0]
    dy = y - prev_xy[1]
    d = float(np.hypot(dx, dy))
    lo = params.motion_displacement_px_per_frame_min
    hi = params.motion_displacement_px_per_frame_max
    if d < lo:
        # Too still -- probably a static feature, not the ball
        if lo <= 0:
            return 1.0
        return max(0.0, d / lo)
    if d > hi:
        # Too fast -- probably an ID swap or noise
        return max(0.0, 1.0 - (d - hi) / hi)
    return 1.0


# Weights for the final score. Sum to 1.0. Motion is heaviest because it
# is the strongest discriminator once we have any history.
W_MOTION = 0.45
W_CIRCULARITY = 0.25
W_COLOR = 0.30
# First-frame penalty: without motion info, confidence is multiplied by this
# so downstream can recognize first-frame detections as lower-quality.
FIRST_FRAME_CONF_FACTOR = 0.7


def detect_in_frame(
    frame_bgr: np.ndarray,
    background: BackgroundModel,
    roi_polygon: np.ndarray,
    params: BallCVParams,
    prev_xy: Optional[tuple] = None,
) -> Optional[BallDetection]:
    """Run the CV pipeline on one frame. Returns the best BallDetection or
    None if no candidate survived filtering. prev_xy is the previous
    frame's detection (pixel_x, pixel_y) for motion scoring; None on
    first frame or after a gap."""
    bg = background.background
    if frame_bgr.shape != bg.shape:
        raise ValueError(
            f"frame shape {frame_bgr.shape} != background shape {bg.shape}"
        )

    # Foreground = abs diff vs background, grayscale, blur, threshold
    diff = cv2.absdiff(frame_bgr, bg)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), sigmaX=1.0)
    _, fg_bin = cv2.threshold(
        diff_blur, params.bg_subtraction_threshold, 255, cv2.THRESH_BINARY
    )

    # Morphological closing to merge ball pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_CLOSE, kernel)

    roi_mask = _make_roi_mask(frame_bgr.shape, roi_polygon)
    candidates = _candidate_blobs(fg_bin, roi_mask, params)

    if not candidates:
        return None

    best = None
    best_score = -1.0
    for cand in candidates:
        circ_score = min(1.0, cand["circularity"] / 0.8)  # normalize at 0.8
        color = _color_score(frame_bgr, cand["pixel_x"], cand["pixel_y"],
                             params)
        motion = _motion_score(cand["pixel_x"], cand["pixel_y"], prev_xy,
                               params)
        score = (
            W_MOTION * motion +
            W_CIRCULARITY * circ_score +
            W_COLOR * color
        )
        if score > best_score:
            best_score = score
            best = cand

    if best is None:
        return None

    conf = best_score
    if prev_xy is None:
        conf *= FIRST_FRAME_CONF_FACTOR

    return BallDetection(
        pixel_x=best["pixel_x"],
        pixel_y=best["pixel_y"],
        confidence=float(conf),
        blob_area_px=best["area"],
        blob_circularity=best["circularity"],
    )
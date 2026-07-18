"""Stage 3 — Pose estimation.

CLI tool that takes a per-video folder containing video.mp4, court.json, and
players.parquet, runs a GPU pose model (Ultralytics YOLO-pose) on the bbox crops
of in-scope player detections, and produces poses.parquet plus pose_summary.json
in the same folder. YOLO-pose emits COCO-17 keypoints mapped onto the same
BlazePose-33 column schema the pipeline already expects (unused points -> NaN),
so it's a drop-in for every downstream stage.

Usage:
    python -m stages.pose.pose data/match_001/

The folder must contain:
    video.mp4              - source video
    court.json             - from Stage 1
    players.parquet        - from Stage 2

Outputs (in the same folder):
    poses.parquet          - per-(frame, track_id) pose with 33 landmarks
    pose_summary.json      - per-track diagnostic and warnings

In-scope detections (which tracks get posed):
    is_user=True (Stage 2.5 role 'user')
    OR role in {partner, opp_a, opp_b}               (track_roles.json present)
    OR — fallback, no track_roles.json — the geometric gate:
       (transient=False AND in_court_frac >= 0.50
        AND court_y_ft.max() <= 44 AND court_y_ft.min() >= -8
        AND lifetime > 5 seconds)

Per-detection processing: for each in-scope detection, the bbox crop is
masked to grey out the regions of OTHER detections on the same frame
(including non-in-scope detections like adjacent-court players), then we take
the highest-confidence person in the crop. This prevents the pose model from
locking onto the wrong person when multiple people overlap in the crop.

This module has no side effects on import. All work happens in main().
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -- Tunable constants ---------------------------------------------------------

# GPU pose: Ultralytics YOLO-pose (COCO-17). The small model keeps good keypoint
# accuracy on player-sized crops while running fast on GPU; auto-downloads
# (~20 MB) on first use. Override with PB_POSE_MODEL (e.g. yolo11n-pose.pt for a
# faster/lighter run, or yolo11m-pose.pt for more accuracy).
POSE_MODEL = os.environ.get("PB_POSE_MODEL", "yolo11s-pose.pt")
POSE_MIN_CONFIDENCE = 0.25      # min person-detection confidence per crop
BBOX_PAD_FRAC = 0.10
USER_DETECTION_RATE_WARNING = 0.5

# Scope: roles Stage 2.5 assigns to the non-user players we pose. The geometric
# court_y gate (below) is only the fallback when track_roles.json is absent — it
# cannot survive far-side projection jitter, so roles are the primary scope.
# See SYSTEM_DESIGN.md Stage 2/3.
PLAYER_SCOPE_ROLES = frozenset({"partner", "opp_a", "opp_b"})

# Geometric-fallback scope constants (no track_roles.json).
SCOPE_MIN_IN_COURT_FRAC = 0.50
SCOPE_MAX_Y_FT = 44.0
SCOPE_MIN_Y_FT = -8.0
SCOPE_MIN_LIFETIME_SEC = 5.0

# Mask color used to grey-out other detections within a crop. Mid-grey BGR.
# Chosen to be neutral relative to a green pickleball court; a deep red would
# be likelier to fool MediaPipe into seeing skin.
OTHER_PERSON_MASK_COLOR = (128, 128, 128)
# Shrink the masking rectangles slightly so we don't overlap-and-mask the
# subject's own bbox edges by accident. Fraction of width/height shrunk on
# each side.
OTHER_PERSON_MASK_SHRINK_FRAC = 0.05

POSE_SUMMARY_SCHEMA_VERSION = 1


# -- Landmark order (matches MediaPipe's PoseLandmark enum) --------------------

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

assert len(LANDMARK_NAMES) == 33, "MediaPipe Pose has exactly 33 landmarks"

METADATA_COLUMNS = ["frame", "t_sec", "track_id", "is_user", "pose_detected"]

LANDMARK_COLUMNS: List[str] = []
for name in LANDMARK_NAMES:
    LANDMARK_COLUMNS.extend([
        f"{name}_x_px",
        f"{name}_y_px",
        f"{name}_z",
        f"{name}_visibility",
    ])

PARQUET_COLUMNS = METADATA_COLUMNS + LANDMARK_COLUMNS

assert len(PARQUET_COLUMNS) == 5 + 33 * 4, "Expected 137 columns total"

# YOLO-pose emits COCO-17 keypoints; map them onto the BlazePose-33 column layout
# by name. The 16 MediaPipe-only points (fine face detail, hands, heel/foot-index)
# are consumed by NO downstream stage, so they stay NaN — every landmark the
# pipeline reads (wrists, shoulders, hips, ankles, elbows, knees) is in COCO-17.
COCO17_TO_LANDMARK = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}
assert all(n in LANDMARK_NAMES for n in COCO17_TO_LANDMARK.values())


# -- Exceptions ----------------------------------------------------------------


class InputError(ValueError):
    """Raised when an input file is missing or malformed."""


class VideoError(RuntimeError):
    """Raised when the video can't be opened or read."""


class PoseImportError(RuntimeError):
    """Raised when the pose backend (ultralytics) is not installed."""


class ModelDownloadError(RuntimeError):
    """Raised when the MediaPipe model file can't be downloaded."""


# -- Loaders -------------------------------------------------------------------


def load_court_fps(path: Path) -> float:
    if not path.exists():
        raise InputError(f"court.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return float(data["video"]["fps"])
    except KeyError as e:
        raise InputError(f"court.json missing field: {e}")


def load_players(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise InputError(f"players.parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {
        "frame", "t_sec", "track_id", "is_user", "transient", "in_court",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "court_x_ft", "court_y_ft",
    }
    missing = required - set(df.columns)
    if missing:
        raise InputError(
            f"players.parquet missing required columns: {sorted(missing)}"
        )
    return df


# -- Scope filter --------------------------------------------------------------


def load_track_roles(path: Path) -> Optional[Dict[int, str]]:
    """Load Stage 2.5 roles as {track_id: role}, or None if absent/unreadable."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {int(t): info["role"] for t, info in d.get("track_roles", {}).items()}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"could not read {path}: {e}; falling back to geometric scope")
        return None


def filter_to_scope(
    df: pd.DataFrame, fps: float, roles: Optional[Dict[int, str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Apply the per-track scope filter (which tracks get posed).

    `is_user` rows are always in scope (set by the caller from the Stage 2.5 role
    'user'). For the OTHER players:
      * **roles given (track_roles.json present): scope by ROLE** — pose every
        track Stage 2.5 classified as partner / opponent, and exclude the noise
        tracks. This is robust where a geometric court_y gate is NOT: far-side
        foot points jitter past the baseline (the homography is hypersensitive
        near the horizon, ~4 px/ft), so a court_y threshold either deletes real
        opponents (max-based) or admits in-court noise (median-based). The role
        classification is the right discriminator. (See SYSTEM_DESIGN.md §3.)
      * **no roles (fallback): the conservative geometric gate** (max court_y).
    """
    total_player_detections = int(len(df))
    non_transient = df[~df["transient"]]
    non_transient_detections = int(len(non_transient))

    user_df = df[df["is_user"]]
    user_track_ids = set(df.loc[df["is_user"], "track_id"].unique().tolist())

    candidates = df[(~df["transient"]) & (~df["is_user"])]
    candidate_tids = set(candidates["track_id"].unique().tolist())

    if roles is not None:
        # Role-based scope: the classified non-user players; noise excluded.
        scope_basis = "role"
        in_scope_track_ids: set = {
            tid for tid in candidate_tids
            if roles.get(tid) in PLAYER_SCOPE_ROLES
        }
    elif len(candidates) == 0:
        scope_basis = "geometric"
        in_scope_track_ids = set()
    else:
        # Geometric fallback (no Stage 2.5): conservative max-based gate. Strict
        # (may drop a jittery far player) but won't admit noise when we have no
        # role info to discriminate.
        scope_basis = "geometric"
        g = candidates.groupby("track_id")
        per_track = pd.DataFrame({
            "in_court_frac":  g["in_court"].mean(),
            "y_max":          g["court_y_ft"].max(),
            "y_min":          g["court_y_ft"].min(),
            "frame_min":      g["frame"].min(),
            "frame_max":      g["frame"].max(),
        })
        per_track["lifetime_sec"] = (
            per_track["frame_max"] - per_track["frame_min"] + 1
        ) / fps
        passes = (
            (per_track["in_court_frac"] >= SCOPE_MIN_IN_COURT_FRAC)
            & (per_track["y_max"] <= SCOPE_MAX_Y_FT)
            & (per_track["y_min"] >= SCOPE_MIN_Y_FT)
            & (per_track["lifetime_sec"] > SCOPE_MIN_LIFETIME_SEC)
        )
        in_scope_track_ids = set(per_track.index[passes].tolist())

    in_scope_other = df[
        (~df["is_user"])
        & (df["track_id"].isin(in_scope_track_ids))
    ]
    in_scope_df = pd.concat([user_df, in_scope_other], ignore_index=False)
    in_scope_df = in_scope_df.sort_values(["frame", "track_id"]).reset_index(drop=True)

    all_in_scope_tids = sorted(in_scope_track_ids | user_track_ids)

    stats = {
        "total_player_detections":  total_player_detections,
        "non_transient_detections": non_transient_detections,
        "in_scope_detections":      int(len(in_scope_df)),
        "in_scope_tracks":          len(all_in_scope_tids),
        "in_scope_track_ids":       all_in_scope_tids,
        "scope_basis":              scope_basis,
    }
    return in_scope_df, stats


# -- MediaPipe -----------------------------------------------------------------


def load_pose_model():
    """Load the YOLO-pose model on GPU if available (else CPU).
    Returns (model, device_str)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise PoseImportError(
            "ultralytics not installed. Run: pip install ultralytics"
        ) from e
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"
    try:
        model = YOLO(POSE_MODEL)
        model.to(device)
    except Exception as e:  # noqa: BLE001  (download or load failure)
        raise ModelDownloadError(
            f"Failed to load YOLO-pose model {POSE_MODEL!r}: {e}"
        ) from e
    return model, device


# -- Bbox padding and cropping -------------------------------------------------


def pad_and_clip_bbox(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
    pad_frac: float,
) -> Tuple[int, int, int, int]:
    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_frac
    pad_y = h * pad_frac
    px1 = int(max(0, np.floor(x1 - pad_x)))
    py1 = int(max(0, np.floor(y1 - pad_y)))
    px2 = int(min(img_w, np.ceil(x2 + pad_x)))
    py2 = int(min(img_h, np.ceil(y2 + pad_y)))
    return px1, py1, px2, py2


def shrunk_bbox(
    x1: float, y1: float, x2: float, y2: float,
    shrink_frac: float,
) -> Tuple[float, float, float, float]:
    """Shrink a bbox by shrink_frac of its width/height on each side.
    Used when masking other people's bboxes to avoid overlapping the subject's
    own bbox at the edges."""
    w = x2 - x1
    h = y2 - y1
    sx = w * shrink_frac
    sy = h * shrink_frac
    return (x1 + sx, y1 + sy, x2 - sx, y2 - sy)


def mask_other_detections(
    crop: np.ndarray,
    crop_origin: Tuple[int, int],     # (px1, py1) of the crop in image coords
    subject_bbox: Tuple[float, float, float, float],  # x1,y1,x2,y2 image coords
    other_bboxes: List[Tuple[float, float, float, float]],  # all OTHER bboxes on this frame
) -> np.ndarray:
    """In-place modify a copy of the crop, painting over regions belonging to
    other detected people. Returns the modified crop.

    Each other bbox is shrunk slightly before masking (to be safe near the
    subject's own bbox edges), then intersected with the crop, then filled
    with OTHER_PERSON_MASK_COLOR.

    The subject's bbox is never masked, even if it overlaps another bbox.
    """
    if not other_bboxes:
        return crop
    out = crop.copy()
    cx, cy = crop_origin
    crop_h, crop_w = out.shape[:2]
    sx1, sy1, sx2, sy2 = subject_bbox

    for ox1, oy1, ox2, oy2 in other_bboxes:
        # Shrink the other bbox a bit so we don't accidentally mask the
        # subject's own edges when bboxes touch.
        ox1, oy1, ox2, oy2 = shrunk_bbox(
            ox1, oy1, ox2, oy2, OTHER_PERSON_MASK_SHRINK_FRAC,
        )
        # Convert to crop-local coords.
        local_x1 = int(round(max(0, ox1 - cx)))
        local_y1 = int(round(max(0, oy1 - cy)))
        local_x2 = int(round(min(crop_w, ox2 - cx)))
        local_y2 = int(round(min(crop_h, oy2 - cy)))
        if local_x2 <= local_x1 or local_y2 <= local_y1:
            continue
        # Don't mask any region that lies inside the subject's bbox.
        # We do this by painting the rectangle, then re-painting the subject's
        # rectangle back from the original crop.
        out[local_y1:local_y2, local_x1:local_x2] = OTHER_PERSON_MASK_COLOR

    # Restore the subject's region from the original (in case any other-bbox
    # overlap painted over it).
    sub_local_x1 = int(round(max(0, sx1 - cx)))
    sub_local_y1 = int(round(max(0, sy1 - cy)))
    sub_local_x2 = int(round(min(crop_w, sx2 - cx)))
    sub_local_y2 = int(round(min(crop_h, sy2 - cy)))
    if sub_local_x2 > sub_local_x1 and sub_local_y2 > sub_local_y1:
        out[sub_local_y1:sub_local_y2, sub_local_x1:sub_local_x2] = (
            crop[sub_local_y1:sub_local_y2, sub_local_x1:sub_local_x2]
        )
    return out


# -- Pose extraction -----------------------------------------------------------


def empty_landmark_values() -> List[float]:
    return [float("nan")] * (33 * 4)


def landmarks_from_result(result, crop_x: int, crop_y: int) -> Optional[List[float]]:
    """Map the top-confidence person's COCO-17 keypoints from one YOLO result into
    the BlazePose-33 flat landmark row (absolute image px; per-keypoint conf ->
    visibility). Returns None if the crop had no detected person. Non-COCO
    landmarks and all `_z` columns stay NaN (read by no consumer)."""
    kps = getattr(result, "keypoints", None)
    if kps is None or kps.xy is None or len(kps.xy) == 0:
        return None
    xy = kps.xy.cpu().numpy()          # (n, 17, 2) in crop-local pixels
    conf = (kps.conf.cpu().numpy() if kps.conf is not None
            else np.ones(xy.shape[:2], dtype=float))
    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.conf is not None and len(boxes.conf) == len(xy):
        bi = int(boxes.conf.cpu().numpy().argmax())   # the subject = top-conf person
    else:
        bi = 0
    kp_xy, kp_conf = xy[bi], conf[bi]
    vals = dict.fromkeys(LANDMARK_COLUMNS, float("nan"))
    for ci, name in COCO17_TO_LANDMARK.items():
        vals[f"{name}_x_px"] = float(crop_x + kp_xy[ci][0])
        vals[f"{name}_y_px"] = float(crop_y + kp_xy[ci][1])
        vals[f"{name}_visibility"] = float(kp_conf[ci])
    return [vals[c] for c in LANDMARK_COLUMNS]


# -- Main processing -----------------------------------------------------------


def process_video(
    video_path: Path,
    scope_df: pd.DataFrame,
    all_players_df: pd.DataFrame,
) -> List[Dict]:
    """Walk the video, run pose on each in-scope detection, return rows.

    For each in-scope detection, the crop has all OTHER detections on the same
    frame (in or out of scope) masked out before pose runs, and the highest-
    confidence person in the crop is taken. This prevents the pose model from
    locking onto the wrong person when bboxes overlap.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video: {video_path}")
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector, device = load_pose_model()

    rows: List[Dict] = []

    by_frame_scope: Dict[int, pd.DataFrame] = dict(tuple(scope_df.groupby("frame")))
    if not by_frame_scope:
        cap.release()
        return rows
    last_frame = max(by_frame_scope.keys())

    # All detections (not just in-scope) by frame, used for masking. Includes
    # transient and out-of-scope tracks because they still represent real
    # people whose bodies could confuse the (single-subject-per-crop) pose model.
    by_frame_all: Dict[int, pd.DataFrame] = dict(
        tuple(all_players_df.groupby("frame"))
    )

    logger.info(
        f"Running YOLO-pose ({POSE_MODEL}, {device}) on {len(scope_df)} in-scope "
        f"detections across {len(by_frame_scope)} frames (max frame {last_frame})..."
    )

    current_frame = -1
    processed_dets = 0
    pose_detected_count = 0

    try:
        while current_frame < last_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise VideoError(
                    f"Frame read failed at frame index {current_frame + 1}; "
                    f"video may be truncated."
                )
            current_frame += 1

            scope_dets = by_frame_scope.get(current_frame)
            if scope_dets is None or len(scope_dets) == 0:
                continue

            # All bboxes on this frame, indexed by track_id, for masking.
            all_dets = by_frame_all.get(current_frame)
            all_bboxes: Dict[int, Tuple[float, float, float, float]] = {}
            if all_dets is not None:
                for _, d in all_dets.iterrows():
                    all_bboxes[int(d["track_id"])] = (
                        float(d["bbox_x1"]), float(d["bbox_y1"]),
                        float(d["bbox_x2"]), float(d["bbox_y2"]),
                    )

            # Build one masked crop per in-scope detection, then run them as a
            # single batched GPU call (the real speedup vs per-crop CPU).
            crops: List[np.ndarray] = []
            meta: List[Tuple[pd.Series, int, int, Optional[int]]] = []
            for _, det in scope_dets.iterrows():
                tid = int(det["track_id"])
                x1, y1, x2, y2 = (
                    float(det["bbox_x1"]), float(det["bbox_y1"]),
                    float(det["bbox_x2"]), float(det["bbox_y2"]),
                )
                px1, py1, px2, py2 = pad_and_clip_bbox(
                    x1, y1, x2, y2, img_w, img_h, BBOX_PAD_FRAC,
                )
                if px2 - px1 <= 0 or py2 - py1 <= 0:
                    meta.append((det, px1, py1, None))
                    continue
                crop = frame[py1:py2, px1:px2]
                other_bboxes = [
                    bb for other_tid, bb in all_bboxes.items() if other_tid != tid
                ]
                if other_bboxes:
                    crop = mask_other_detections(
                        crop, (px1, py1), (x1, y1, x2, y2), other_bboxes,
                    )
                meta.append((det, px1, py1, len(crops)))
                crops.append(crop)

            results = (
                detector.predict(crops, device=device, conf=POSE_MIN_CONFIDENCE,
                                 verbose=False)
                if crops else []
            )

            for det, px1, py1, ridx in meta:
                landmarks = None
                if ridx is not None:
                    landmarks = landmarks_from_result(results[ridx], px1, py1)
                pose_detected = landmarks is not None
                if pose_detected:
                    pose_detected_count += 1
                else:
                    landmarks = empty_landmark_values()

                row = {
                    "frame":         int(det["frame"]),
                    "t_sec":         float(det["t_sec"]),
                    "track_id":      int(det["track_id"]),
                    "is_user":       bool(det["is_user"]),
                    "pose_detected": bool(pose_detected),
                }
                for col_name, val in zip(LANDMARK_COLUMNS, landmarks):
                    row[col_name] = val
                rows.append(row)
                processed_dets += 1

            if current_frame > 0 and current_frame % 200 == 0:
                rate = (
                    pose_detected_count / processed_dets
                    if processed_dets else 0.0
                )
                logger.info(
                    f"  frame {current_frame}: {processed_dets} detections "
                    f"processed, {pose_detected_count} poses ({rate:.1%})"
                )
    finally:
        cap.release()

    rate = pose_detected_count / processed_dets if processed_dets else 0.0
    logger.info(
        f"Pose extraction complete: {processed_dets} detections processed, "
        f"{pose_detected_count} poses detected ({rate:.1%})"
    )
    return rows


# -- Output --------------------------------------------------------------------


def rows_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=PARQUET_COLUMNS)


def build_summary(df: pd.DataFrame, scope_stats: Dict) -> Tuple[Dict, List[str]]:
    per_track: List[Dict] = []
    warnings: List[str] = []

    if len(df) == 0:
        return ({
            "schema_version":            POSE_SUMMARY_SCHEMA_VERSION,
            "scope_filter":              {
                "total_player_detections":  scope_stats["total_player_detections"],
                "non_transient_detections": scope_stats["non_transient_detections"],
                "in_scope_detections":      scope_stats["in_scope_detections"],
                "in_scope_tracks":          scope_stats["in_scope_tracks"],
            },
            "total_pose_detected":       0,
            "overall_detection_rate":    0.0,
            "per_track":                 [],
            "warnings":                  ["No in-scope detections; poses.parquet is empty."],
        }, ["No in-scope detections; poses.parquet is empty."])

    grouped = df.groupby("track_id")
    for tid, group in grouped:
        n_det = int(len(group))
        n_pose = int(group["pose_detected"].sum())
        rate = n_pose / n_det if n_det else 0.0
        is_user = bool(group["is_user"].any())
        per_track.append({
            "track_id":        int(tid),
            "is_user":         is_user,
            "n_detections":    n_det,
            "n_pose_detected": n_pose,
            "rate":            float(round(rate, 4)),
        })
        if is_user and rate < USER_DETECTION_RATE_WARNING:
            warnings.append(
                f"User track {tid}: pose detection rate is {rate:.1%} "
                f"(< {USER_DETECTION_RATE_WARNING:.0%}). "
                f"Bbox crops may be too small or noisy."
            )

    per_track.sort(key=lambda r: (not r["is_user"], r["track_id"]))

    total_det = int(len(df))
    total_pose = int(df["pose_detected"].sum())
    overall_rate = total_pose / total_det if total_det else 0.0

    summary = {
        "schema_version":            POSE_SUMMARY_SCHEMA_VERSION,
        "scope_filter": {
            "total_player_detections":  scope_stats["total_player_detections"],
            "non_transient_detections": scope_stats["non_transient_detections"],
            "in_scope_detections":      scope_stats["in_scope_detections"],
            "in_scope_tracks":          scope_stats["in_scope_tracks"],
        },
        "total_pose_detected":       total_pose,
        "overall_detection_rate":    float(round(overall_rate, 4)),
        "per_track":                 per_track,
        "warnings":                  warnings,
    }
    return summary, warnings


# -- Main ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: pose estimation. Reads video.mp4, court.json, and "
            "players.parquet from a per-video folder; writes poses.parquet "
            "and pose_summary.json."
        )
    )
    parser.add_argument(
        "video_folder", type=Path,
        help=(
            "Folder containing video.mp4, court.json, players.parquet. "
            "Outputs are written to the same folder."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    folder = args.video_folder
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        return 2

    video_path = folder / "video.mp4"
    court_path = folder / "court.json"
    players_path = folder / "players.parquet"
    out_parquet = folder / "poses.parquet"
    out_summary = folder / "pose_summary.json"

    if not video_path.exists():
        print(f"Error: video.mp4 not found in {folder}", file=sys.stderr)
        return 2

    try:
        fps = load_court_fps(court_path)
        players_df = load_players(players_path)
    except InputError as e:
        print(f"Input error: {e}", file=sys.stderr)
        return 2

    # Stage 2.5 track_roles.json is the authority on player identity. When
    # present: mark is_user from the role 'user' (not the click-only flag in
    # players.parquet, empty in the no-clicks flow) so every user segment is
    # always in scope, AND scope the non-user players (partner/opponents) by
    # role too — robust where the geometric court_y gate is not (far-side foot
    # jitter past the baseline used to delete every opponent from pose).
    roles = load_track_roles(folder / "track_roles.json")
    if roles is not None:
        user_tids = {tid for tid, r in roles.items() if r == "user"}
        players_df["is_user"] = players_df["track_id"].isin(user_tids)
        logger.info(f"using track_roles.json: {len(user_tids)} user track(s) "
                    "drive is_user; partner/opponents scoped by role")
    else:
        logger.info("no track_roles.json; using players.parquet is_user + "
                    "geometric scope filter only")

    scope_df, scope_stats = filter_to_scope(players_df, fps, roles)
    n_user = int(scope_df["is_user"].sum())
    n_other = int((~scope_df["is_user"]).sum())
    logger.info(
        f"Loaded {scope_stats['total_player_detections']} player rows; "
        f"{scope_stats['non_transient_detections']} non-transient; "
        f"{scope_stats['in_scope_detections']} in-scope after strict filter "
        f"({n_user} user, {n_other} non-user); "
        f"{scope_stats['in_scope_tracks']} in-scope track(s); fps={fps:.2f}"
    )
    logger.info(
        f"In-scope track IDs: {scope_stats['in_scope_track_ids']}"
    )

    if len(scope_df) == 0:
        print(
            "Error: no in-scope detections after filtering. "
            "The scope filter may be too strict for this footage. "
            "Check pose_summary.json for the breakdown.",
            file=sys.stderr,
        )
        summary, _ = build_summary(scope_df, scope_stats)
        with out_summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return 1

    try:
        rows = process_video(video_path, scope_df, players_df)
    except VideoError as e:
        print(f"Video error: {e}", file=sys.stderr)
        return 1
    except (PoseImportError, ModelDownloadError) as e:
        print(f"Dependency error: {e}", file=sys.stderr)
        return 2

    df = rows_to_dataframe(rows)
    df.to_parquet(out_parquet, index=False)
    logger.info(f"Wrote {out_parquet} ({len(df)} rows, {len(df.columns)} columns)")

    summary, warnings = build_summary(df, scope_stats)
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        f"Wrote {out_summary} ({len(summary['per_track'])} track(s), "
        f"{len(warnings)} warning(s))"
    )

    for w in warnings:
        logger.warning(w)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
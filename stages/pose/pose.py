"""Stage 3 — Pose estimation.

CLI tool that takes a per-video folder containing video.mp4, court.json, and
players.parquet, runs MediaPipe Pose on the bbox crops of in-scope player
detections, and produces poses.parquet plus pose_summary.json in the same
folder.

Usage:
    python -m stages.pose.pose data/match_001/

The folder must contain:
    video.mp4              - source video
    court.json             - from Stage 1
    players.parquet        - from Stage 2

Outputs (in the same folder):
    poses.parquet          - per-(frame, track_id) pose with 33 landmarks
    pose_summary.json      - per-track diagnostic and warnings

In-scope detections pass a strict per-track filter (see contract.md):
    is_user=True
    OR (transient=False
        AND in_court_frac >= 0.50
        AND court_y_ft.max() <= 44.0
        AND court_y_ft.min() >= -8.0
        AND lifetime > 5 seconds)

Per-detection processing: for each in-scope detection, the bbox crop is
masked to grey out the regions of OTHER detections on the same frame
(including non-in-scope detections like adjacent-court players). This
prevents MediaPipe (which is single-person) from picking the wrong person
when multiple people overlap in the crop.

This module has no side effects on import. All work happens in main().
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -- Tunable constants ---------------------------------------------------------

MODEL_COMPLEXITY = 1            # 0=lite, 1=full, 2=heavy
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
BBOX_PAD_FRAC = 0.10
USER_DETECTION_RATE_WARNING = 0.5

# Scope filter constants
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

# MediaPipe Tasks API model files. The model is auto-downloaded on first run
# and cached at ~/.cache/mediapipe_models/.
MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
MODEL_FILENAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}

MODEL_CACHE_DIR = Path.home() / ".cache" / "mediapipe_models"


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


# -- Exceptions ----------------------------------------------------------------


class InputError(ValueError):
    """Raised when an input file is missing or malformed."""


class VideoError(RuntimeError):
    """Raised when the video can't be opened or read."""


class MediaPipeImportError(RuntimeError):
    """Raised when MediaPipe is not installed."""


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


def filter_to_scope(df: pd.DataFrame, fps: float) -> Tuple[pd.DataFrame, Dict]:
    """Apply the per-track scope filter described in the contract."""
    total_player_detections = int(len(df))
    non_transient = df[~df["transient"]]
    non_transient_detections = int(len(non_transient))

    user_df = df[df["is_user"]]

    candidates = df[(~df["transient"]) & (~df["is_user"])]
    if len(candidates) == 0:
        in_scope_track_ids: set = set()
    else:
        g = candidates.groupby("track_id")
        per_track = pd.DataFrame({
            "n_rows":         g.size(),
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

    user_track_ids = set(df.loc[df["is_user"], "track_id"].unique().tolist())
    all_in_scope_tids = sorted(in_scope_track_ids | user_track_ids)

    stats = {
        "total_player_detections":  total_player_detections,
        "non_transient_detections": non_transient_detections,
        "in_scope_detections":      int(len(in_scope_df)),
        "in_scope_tracks":          len(all_in_scope_tids),
        "in_scope_track_ids":       all_in_scope_tids,
    }
    return in_scope_df, stats


# -- MediaPipe -----------------------------------------------------------------


def ensure_model(complexity: int) -> Path:
    """Return the local path to the .task model file, downloading if needed."""
    if complexity not in MODEL_URLS:
        raise ValueError(f"MODEL_COMPLEXITY must be 0, 1, or 2; got {complexity}")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_CACHE_DIR / MODEL_FILENAMES[complexity]
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    url = MODEL_URLS[complexity]
    logger.info(f"Downloading MediaPipe model: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        model_path.write_bytes(data)
        logger.info(
            f"Wrote model to {model_path} ({len(data) / 1024 / 1024:.1f} MB)"
        )
    except Exception as e:
        raise ModelDownloadError(
            f"Failed to download MediaPipe model from {url}: {e}\n"
            f"You can download it manually and place it at: {model_path}"
        ) from e
    return model_path


def make_pose_detector():
    """Create a MediaPipe PoseLandmarker via the Tasks API."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as e:
        raise MediaPipeImportError(
            "MediaPipe not installed or too old. Run: pip install --upgrade mediapipe"
        ) from e

    model_path = ensure_model(MODEL_COMPLEXITY)

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        output_segmentation_masks=False,
    )
    detector = mp_vision.PoseLandmarker.create_from_options(options)
    return detector, mp


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


def extract_landmarks_from_result(
    result, crop_x: int, crop_y: int, crop_w: int, crop_h: int,
) -> List[float]:
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return empty_landmark_values()
    lms = result.pose_landmarks[0]
    if len(lms) != 33:
        return empty_landmark_values()
    out: List[float] = []
    for lm in lms:
        out.append(float(crop_x + lm.x * crop_w))
        out.append(float(crop_y + lm.y * crop_h))
        out.append(float(lm.z))
        out.append(float(lm.visibility))
    return out


# -- Main processing -----------------------------------------------------------


def process_video(
    video_path: Path,
    scope_df: pd.DataFrame,
    all_players_df: pd.DataFrame,
) -> List[Dict]:
    """Walk the video, run pose on each in-scope detection, return rows.

    For each in-scope detection, the crop has all OTHER detections on the same
    frame (in or out of scope) masked out before being passed to MediaPipe.
    This prevents single-person MediaPipe from picking the wrong person when
    bboxes overlap.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video: {video_path}")
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector, mp = make_pose_detector()

    rows: List[Dict] = []

    by_frame_scope: Dict[int, pd.DataFrame] = dict(tuple(scope_df.groupby("frame")))
    if not by_frame_scope:
        cap.release()
        return rows
    last_frame = max(by_frame_scope.keys())

    # All detections (not just in-scope) by frame, used for masking. Includes
    # transient and out-of-scope tracks because they still represent real
    # people whose bodies could confuse MediaPipe.
    by_frame_all: Dict[int, pd.DataFrame] = dict(
        tuple(all_players_df.groupby("frame"))
    )

    logger.info(
        f"Running pose on {len(scope_df)} in-scope detections across "
        f"{len(by_frame_scope)} frames (max frame {last_frame})..."
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

            for _, det in scope_dets.iterrows():
                tid = int(det["track_id"])
                x1, y1, x2, y2 = (
                    float(det["bbox_x1"]), float(det["bbox_y1"]),
                    float(det["bbox_x2"]), float(det["bbox_y2"]),
                )
                px1, py1, px2, py2 = pad_and_clip_bbox(
                    x1, y1, x2, y2, img_w, img_h, BBOX_PAD_FRAC,
                )
                crop_w = px2 - px1
                crop_h = py2 - py1

                if crop_w <= 0 or crop_h <= 0:
                    landmarks = empty_landmark_values()
                    pose_detected = False
                else:
                    crop = frame[py1:py2, px1:px2]
                    other_bboxes = [
                        bb for other_tid, bb in all_bboxes.items()
                        if other_tid != tid
                    ]
                    if other_bboxes:
                        crop = mask_other_detections(
                            crop, (px1, py1),
                            (x1, y1, x2, y2),
                            other_bboxes,
                        )
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=crop_rgb,
                    )
                    result = detector.detect(mp_image)
                    if not result.pose_landmarks:
                        landmarks = empty_landmark_values()
                        pose_detected = False
                    else:
                        landmarks = extract_landmarks_from_result(
                            result, px1, py1, crop_w, crop_h,
                        )
                        pose_detected = not all(
                            np.isnan(v) for v in landmarks
                        )
                        if pose_detected:
                            pose_detected_count += 1

                row = {
                    "frame":         int(det["frame"]),
                    "t_sec":         float(det["t_sec"]),
                    "track_id":      tid,
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
        try:
            detector.close()
        except Exception:
            pass

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

    scope_df, scope_stats = filter_to_scope(players_df, fps)
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
    except (MediaPipeImportError, ModelDownloadError) as e:
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
"""Stage 1 — Court calibration.

CLI tool that takes a video and a markers JSON file (8 user clicks + 3 form
answers) and produces court.json + court_zones.json in the per-video data
folder.

Usage:
    python -m stages.calibrate.calibrate \
        --video data/match_001/video.mp4 \
        --markers data/match_001/markers.json \
        --out-dir data/match_001/

The markers.json file is produced by the frontend (or hand-crafted for
testing). Schema:

    {
        "court_corners_image": [[x,y], [x,y], [x,y], [x,y]],
        "kitchen_line_user_image":     [[x,y], [x,y]],
        "kitchen_line_opponent_image": [[x,y], [x,y]],
        "user_baseline":         "near" | "far",
        "dominant_hand":         "right" | "left",
        "user_starting_corner":  "left" | "right",
        "frame_used_for_calibration": 0
    }

This module has no side effects on import. All work happens in main().
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COURT_JSON_SCHEMA_VERSION = 1
COURT_ZONES_JSON_SCHEMA_VERSION = 1
COURT_ZONES_POLICY_VERSION = 1

COURT_WIDTH_FT = 20.0
COURT_LENGTH_FT = 44.0
KITCHEN_DEPTH_FT = 7.0  # depth of the non-volley zone, measured from the NET

# Y-coordinate (in court-feet) of each kitchen line, given user_baseline.
# Net is at y = COURT_LENGTH_FT / 2 = 22 ft.
# User-side kitchen is 7 ft from the net on the user's side.
# Opponent-side kitchen is 7 ft from the net on the opponent's side.
NET_LINE_FT = COURT_LENGTH_FT / 2.0  # = 22.0

HOMOGRAPHY_RMSE_WARNING_PX = 5.0
KITCHEN_PROJECTION_WARNING_PX = 10.0
KITCHEN_LINE_TOO_SHORT_PX = 50.0


class MarkersError(ValueError):
    """Raised when markers.json is malformed or missing required fields."""


def load_markers(path: Path) -> Dict:
    """Load and validate the markers JSON file."""
    if not path.exists():
        raise MarkersError(f"Markers file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_lists = {
        "court_corners_image": 4,
        "kitchen_line_user_image": 2,
        "kitchen_line_opponent_image": 2,
    }
    for key, expected_len in required_lists.items():
        if key not in data:
            raise MarkersError(f"Missing required field: {key}")
        if not isinstance(data[key], list) or len(data[key]) != expected_len:
            raise MarkersError(
                f"{key} must be a list of {expected_len} [x, y] points; "
                f"got {len(data.get(key, []))}"
            )
        for i, pt in enumerate(data[key]):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise MarkersError(
                    f"{key}[{i}] must be a 2-element [x, y] list"
                )

    enums = {
        "user_baseline": ("near", "far"),
        "dominant_hand": ("right", "left"),
        "user_starting_corner": ("left", "right"),
    }
    for key, allowed in enums.items():
        if key not in data:
            raise MarkersError(f"Missing required field: {key}")
        if data[key] not in allowed:
            raise MarkersError(
                f"{key} must be one of {allowed}; got {data[key]!r}"
            )

    if "frame_used_for_calibration" not in data:
        data["frame_used_for_calibration"] = 0

    return data


class VideoError(RuntimeError):
    """Raised when the video can't be opened or read."""


def probe_video(video_path: Path) -> Dict:
    """Read frame dimensions, fps, and frame count from a video file."""
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
    if width <= 0 or height <= 0 or fps <= 0:
        raise VideoError(
            f"Video has invalid metadata: {width}x{height} @ {fps}fps"
        )
    return {
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def compute_homography(
    image_corners: List[List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute image-to-court and court-to-image 3x3 homography matrices.

    image_corners: 4 points in image-position order
        bottom-left, bottom-right, top-right, top-left of the visible court.

    Court coordinate system (in feet):
        (0, 0)              = user's near-LEFT baseline corner
        (COURT_WIDTH_FT, 0) = user's near-RIGHT baseline corner
        (COURT_WIDTH_FT, COURT_LENGTH_FT) = far-RIGHT baseline corner
        (0, COURT_LENGTH_FT) = far-LEFT baseline corner
    """
    src = np.asarray(image_corners, dtype=np.float32)
    dst = np.asarray(
        [
            [0.0,             0.0],
            [COURT_WIDTH_FT,  0.0],
            [COURT_WIDTH_FT,  COURT_LENGTH_FT],
            [0.0,             COURT_LENGTH_FT],
        ],
        dtype=np.float32,
    )
    image_to_court, _ = cv2.findHomography(src, dst)
    court_to_image, _ = cv2.findHomography(dst, src)
    if image_to_court is None or court_to_image is None:
        raise ValueError(
            "findHomography returned None - corners likely don't form a valid quadrilateral"
        )
    return image_to_court, court_to_image


def project_point(matrix: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
    """Project a 2D point through a 3x3 homography matrix."""
    p = np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    out = matrix @ p
    return (float(out[0] / out[2]), float(out[1] / out[2]))


def compute_homography_rmse(
    image_corners: List[List[float]],
    image_to_court: np.ndarray,
    court_to_image: np.ndarray,
) -> float:
    """Round-trip the corners through the homography and report mean error."""
    src = np.asarray(image_corners, dtype=np.float64)
    errors = []
    for i, court_pt_target in enumerate([
        (0.0, 0.0),
        (COURT_WIDTH_FT, 0.0),
        (COURT_WIDTH_FT, COURT_LENGTH_FT),
        (0.0, COURT_LENGTH_FT),
    ]):
        projected = project_point(court_to_image, court_pt_target)
        errors.append(np.hypot(projected[0] - src[i][0], projected[1] - src[i][1]))
    return float(np.sqrt(np.mean(np.square(errors))))


def _user_kitchen_y_ft(user_baseline: str) -> float:
    """Y-coordinate of the user's kitchen line, in court-feet."""
    if user_baseline == "near":
        return NET_LINE_FT - KITCHEN_DEPTH_FT  # 22 - 7 = 15
    else:
        return NET_LINE_FT + KITCHEN_DEPTH_FT  # 22 + 7 = 29


def _opponent_kitchen_y_ft(user_baseline: str) -> float:
    """Y-coordinate of the opponent's kitchen line, in court-feet."""
    if user_baseline == "near":
        return NET_LINE_FT + KITCHEN_DEPTH_FT  # 22 + 7 = 29
    else:
        return NET_LINE_FT - KITCHEN_DEPTH_FT  # 22 - 7 = 15


def compute_kitchen_projection_error(
    user_kitchen_image: List[List[float]],
    user_baseline: str,
    court_to_image: np.ndarray,
) -> float:
    """How far is the user's clicked kitchen line from the homography-projected
    kitchen line? In pixels."""
    y_ft = _user_kitchen_y_ft(user_baseline)
    expected_left  = project_point(court_to_image, (0.0, y_ft))
    expected_right = project_point(court_to_image, (COURT_WIDTH_FT, y_ft))

    clicked_left, clicked_right = user_kitchen_image
    err_left  = np.hypot(expected_left[0] - clicked_left[0],   expected_left[1] - clicked_left[1])
    err_right = np.hypot(expected_right[0] - clicked_right[0], expected_right[1] - clicked_right[1])
    return float((err_left + err_right) / 2.0)


def compute_opponent_kitchen_projection_error(
    opponent_kitchen_image: List[List[float]],
    user_baseline: str,
    court_to_image: np.ndarray,
) -> float:
    """Same as compute_kitchen_projection_error but for the opponent's side."""
    y_ft = _opponent_kitchen_y_ft(user_baseline)
    expected_left  = project_point(court_to_image, (0.0, y_ft))
    expected_right = project_point(court_to_image, (COURT_WIDTH_FT, y_ft))

    clicked_left, clicked_right = opponent_kitchen_image
    err_left  = np.hypot(expected_left[0] - clicked_left[0],   expected_left[1] - clicked_left[1])
    err_right = np.hypot(expected_right[0] - clicked_right[0], expected_right[1] - clicked_right[1])
    return float((err_left + err_right) / 2.0)


def derive_polygons(
    court_to_image: np.ndarray,
    user_baseline: str,
) -> Dict[str, List[List[float]]]:
    """Compute pixel-coord polygons for the four key court regions.

    The kitchen polygon spans from the kitchen line to the net (7 ft deep).
    The half-court polygon spans from the baseline to the net (22 ft deep).
    """
    user_kitchen_y = _user_kitchen_y_ft(user_baseline)
    opp_kitchen_y = _opponent_kitchen_y_ft(user_baseline)

    if user_baseline == "near":
        user_half_court = [
            (0.0,             0.0),
            (COURT_WIDTH_FT,  0.0),
            (COURT_WIDTH_FT,  NET_LINE_FT),
            (0.0,             NET_LINE_FT),
        ]
        user_kitchen_court = [
            (0.0,             user_kitchen_y),
            (COURT_WIDTH_FT,  user_kitchen_y),
            (COURT_WIDTH_FT,  NET_LINE_FT),
            (0.0,             NET_LINE_FT),
        ]
        opp_half_court = [
            (0.0,             NET_LINE_FT),
            (COURT_WIDTH_FT,  NET_LINE_FT),
            (COURT_WIDTH_FT,  COURT_LENGTH_FT),
            (0.0,             COURT_LENGTH_FT),
        ]
        opp_kitchen_court = [
            (0.0,             NET_LINE_FT),
            (COURT_WIDTH_FT,  NET_LINE_FT),
            (COURT_WIDTH_FT,  opp_kitchen_y),
            (0.0,             opp_kitchen_y),
        ]
    else:
        user_half_court = [
            (0.0,             NET_LINE_FT),
            (COURT_WIDTH_FT,  NET_LINE_FT),
            (COURT_WIDTH_FT,  COURT_LENGTH_FT),
            (0.0,             COURT_LENGTH_FT),
        ]
        user_kitchen_court = [
            (0.0,             NET_LINE_FT),
            (COURT_WIDTH_FT,  NET_LINE_FT),
            (COURT_WIDTH_FT,  user_kitchen_y),
            (0.0,             user_kitchen_y),
        ]
        opp_half_court = [
            (0.0,             0.0),
            (COURT_WIDTH_FT,  0.0),
            (COURT_WIDTH_FT,  NET_LINE_FT),
            (0.0,             NET_LINE_FT),
        ]
        opp_kitchen_court = [
            (0.0,             opp_kitchen_y),
            (COURT_WIDTH_FT,  opp_kitchen_y),
            (COURT_WIDTH_FT,  NET_LINE_FT),
            (0.0,             NET_LINE_FT),
        ]

    def project_polygon(court_polygon):
        return [list(project_point(court_to_image, p)) for p in court_polygon]

    return {
        "user_half_polygon_image":         project_polygon(user_half_court),
        "opponent_half_polygon_image":     project_polygon(opp_half_court),
        "user_kitchen_polygon_image":      project_polygon(user_kitchen_court),
        "opponent_kitchen_polygon_image":  project_polygon(opp_kitchen_court),
    }


def compute_pixels_per_foot(court_to_image: np.ndarray) -> Tuple[float, float]:
    """Return pixels-per-foot at the near baseline and far baseline."""
    near_left  = project_point(court_to_image, (0.0,            0.0))
    near_right = project_point(court_to_image, (COURT_WIDTH_FT, 0.0))
    far_left   = project_point(court_to_image, (0.0,            COURT_LENGTH_FT))
    far_right  = project_point(court_to_image, (COURT_WIDTH_FT, COURT_LENGTH_FT))

    near_px = np.hypot(near_right[0] - near_left[0], near_right[1] - near_left[1])
    far_px  = np.hypot(far_right[0]  - far_left[0],  far_right[1]  - far_left[1])

    return float(near_px / COURT_WIDTH_FT), float(far_px / COURT_WIDTH_FT)


TOP_DOWN_PX_PER_FT = 30


def render_top_down_preview(
    video_path: Path,
    frame_index: int,
    image_to_court: np.ndarray,
) -> np.ndarray:
    """Warp the calibration frame into a synthetic bird's-eye view of the court."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video for preview: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError(f"Could not read frame {frame_index} for preview")
    finally:
        cap.release()

    out_w = int(round(COURT_WIDTH_FT  * TOP_DOWN_PX_PER_FT))
    out_h = int(round(COURT_LENGTH_FT * TOP_DOWN_PX_PER_FT))
    scale = np.array(
        [[TOP_DOWN_PX_PER_FT, 0.0,                 0.0],
         [0.0,                 TOP_DOWN_PX_PER_FT, 0.0],
         [0.0,                 0.0,                 1.0]],
        dtype=np.float64,
    )
    image_to_topdown = scale @ image_to_court
    warped = cv2.warpPerspective(frame, image_to_topdown, (out_w, out_h))

    cyan = (255, 255, 0)
    for y_ft in (NET_LINE_FT - KITCHEN_DEPTH_FT, NET_LINE_FT + KITCHEN_DEPTH_FT, NET_LINE_FT):
        y_px = int(round(y_ft * TOP_DOWN_PX_PER_FT))
        cv2.line(warped, (0, y_px), (out_w, y_px), cyan, 2)

    return warped


def calibrate(
    video_path: Path,
    markers: Dict,
) -> Tuple[Dict, Dict]:
    """Compute court.json and court_zones.json content from inputs."""
    video_meta = probe_video(video_path)

    image_to_court, court_to_image = compute_homography(
        markers["court_corners_image"]
    )

    rmse = compute_homography_rmse(
        markers["court_corners_image"], image_to_court, court_to_image,
    )
    kitchen_user_err = compute_kitchen_projection_error(
        markers["kitchen_line_user_image"], markers["user_baseline"], court_to_image,
    )
    kitchen_opp_err = compute_opponent_kitchen_projection_error(
        markers["kitchen_line_opponent_image"], markers["user_baseline"], court_to_image,
    )

    warnings: List[str] = []
    if rmse > HOMOGRAPHY_RMSE_WARNING_PX:
        warnings.append(
            f"Homography RMSE is {rmse:.1f}px (>{HOMOGRAPHY_RMSE_WARNING_PX}px) - "
            f"corners may have been clicked imprecisely"
        )
    if kitchen_user_err > KITCHEN_PROJECTION_WARNING_PX:
        warnings.append(
            f"User kitchen line projection error is {kitchen_user_err:.1f}px "
            f"(>{KITCHEN_PROJECTION_WARNING_PX}px) - kitchen line and corners disagree"
        )
    if kitchen_opp_err > KITCHEN_PROJECTION_WARNING_PX:
        warnings.append(
            f"Opponent kitchen line projection error is {kitchen_opp_err:.1f}px "
            f"(>{KITCHEN_PROJECTION_WARNING_PX}px)"
        )

    user_kitchen_left  = markers["kitchen_line_user_image"][0]
    user_kitchen_right = markers["kitchen_line_user_image"][1]
    user_kitchen_len = float(np.hypot(
        user_kitchen_right[0] - user_kitchen_left[0],
        user_kitchen_right[1] - user_kitchen_left[1],
    ))
    if user_kitchen_len < KITCHEN_LINE_TOO_SHORT_PX:
        warnings.append(
            f"User kitchen line is {user_kitchen_len:.0f}px in the image "
            f"(<{KITCHEN_LINE_TOO_SHORT_PX}px) - extreme camera angle, accuracy may be reduced"
        )

    polygons = derive_polygons(court_to_image, markers["user_baseline"])
    px_per_ft_near, px_per_ft_far = compute_pixels_per_foot(court_to_image)

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    court_json = {
        "schema_version": COURT_JSON_SCHEMA_VERSION,
        "video": {
            "path": str(video_path),
            "frame_width":  video_meta["frame_width"],
            "frame_height": video_meta["frame_height"],
            "fps":          video_meta["fps"],
            "frame_used_for_calibration": int(markers["frame_used_for_calibration"]),
        },
        "user_inputs": {
            "court_corners_image":           markers["court_corners_image"],
            "kitchen_line_user_image":       markers["kitchen_line_user_image"],
            "kitchen_line_opponent_image":   markers["kitchen_line_opponent_image"],
            "user_baseline":         markers["user_baseline"],
            "dominant_hand":         markers["dominant_hand"],
            "user_starting_corner":  markers["user_starting_corner"],
        },
        "court_geometry_feet": {
            "width_ft":         COURT_WIDTH_FT,
            "length_ft":        COURT_LENGTH_FT,
            "kitchen_depth_ft": KITCHEN_DEPTH_FT,
        },
        "homography": {
            "image_to_court": image_to_court.tolist(),
            "court_to_image": court_to_image.tolist(),
        },
        "derived": {
            **polygons,
            "pixels_per_foot_at_near_baseline": px_per_ft_near,
            "pixels_per_foot_at_far_baseline":  px_per_ft_far,
        },
        "validation": {
            "homography_rmse_pixels":              float(rmse),
            "kitchen_projection_error_user_px":    float(kitchen_user_err),
            "kitchen_projection_error_opponent_px": float(kitchen_opp_err),
            "warnings": warnings,
        },
        "created_at": now_iso,
    }

    court_zones_json = {
        "schema_version": COURT_ZONES_JSON_SCHEMA_VERSION,
        "policy_version": COURT_ZONES_POLICY_VERSION,
        "zones": {
            "kitchen_strict": {
                "depth_ft": KITCHEN_DEPTH_FT,
                "description": "The actual non-volley zone, exactly as marked on the court",
            },
            "kitchen_effective": {
                "depth_ft": 9.0,
                "buffer_ft": 2.0,
                "description": "Kitchen + 2ft buffer counts as 'at the kitchen line' for stat purposes",
                "priority_rule": "If a player is in the buffer zone, count as kitchen, NOT as transition",
            },
            "transition": {
                "near_ft": 9.0,
                "far_ft":  32.0,
                "description": "Between effective kitchen and the rear of the court",
            },
            "baseline_zone": {
                "near_ft": 32.0,
                "far_ft":  COURT_LENGTH_FT,
                "description": "Last 12 feet near the baseline",
            },
        },
        "tracking_zone": {
            "behind_baseline_ft":  8.0,
            "beyond_sideline_ft":  6.0,
            "description": (
                "How far beyond the court lines the player tracker should look for the user. "
                "Players serve from behind the baseline and chase wide shots beyond sidelines."
            ),
        },
        "in_play_polygon_source": "court.json:derived.user_half_polygon_image and opponent_half_polygon_image",
        "in_play_description": (
            "Ball-bounce IN/OUT determination uses the STRICT court polygons from court.json. "
            "A ball that bounces outside these is OUT. A ball hit BEFORE bouncing (player "
            "contacts mid-air) can happen anywhere - shot-impact location is not constrained "
            "by any court polygon."
        ),
        "created_at": now_iso,
    }

    return court_json, court_zones_json


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1: court calibration. Produces court.json and court_zones.json."
    )
    parser.add_argument("--video",   required=True, type=Path, help="Path to video file")
    parser.add_argument("--markers", required=True, type=Path, help="Path to markers JSON file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to write outputs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        markers = load_markers(args.markers)
    except MarkersError as e:
        print(f"Markers error: {e}", file=sys.stderr)
        return 2

    try:
        court_json, court_zones_json = calibrate(args.video, markers)
    except (VideoError, ValueError) as e:
        print(f"Calibration error: {e}", file=sys.stderr)
        return 1

    court_path = args.out_dir / "court.json"
    zones_path = args.out_dir / "court_zones.json"
    with court_path.open("w", encoding="utf-8") as f:
        json.dump(court_json, f, indent=2)
    with zones_path.open("w", encoding="utf-8") as f:
        json.dump(court_zones_json, f, indent=2)

    logger.info(f"Wrote {court_path}")
    logger.info(f"Wrote {zones_path}")
    if court_json["validation"]["warnings"]:
        logger.warning(
            "Calibration completed with %d warning(s):",
            len(court_json["validation"]["warnings"]),
        )
        for w in court_json["validation"]["warnings"]:
            logger.warning("  - %s", w)
    else:
        logger.info("Calibration successful with no warnings")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
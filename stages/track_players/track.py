"""Stage 2 — Player tracking.

CLI tool that takes a per-video folder containing video.mp4, court.json,
court_zones.json, and user_clicks.json and produces players.parquet plus
players_pending.json in the same folder.

Usage:
    python -m stages.track_players.track data/match_001/

The folder must contain:
    video.mp4              - source video
    court.json             - from Stage 1
    court_zones.json       - from Stage 1
    user_clicks.json       - {"clicks": [{"frame": N, "x": X, "y": Y}, ...]}

Outputs (in the same folder):
    players.parquet        - per-(frame, track_id) detections with court coords
    players_pending.json   - any user-track gaps requiring re-identification

This module has no side effects on import. All work happens in main().
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# -- Tunable constants ---------------------------------------------------------

MODEL = "yolo11s.pt"            # auto-downloads via ultralytics on first run
TRACK_LOSS_TOLERANCE_FRAMES = 30
TRANSIENT_LIFETIME_FRAMES = 30
DOUBLES_PERSIST_SECONDS = 5.0
DOUBLES_IN_COURT_FRAC = 0.80
CLICK_MAX_DISTANCE_PX = 150     # max distance from click to nearest detection

PARQUET_COLUMNS = [
    "frame", "t_sec", "track_id",
    "is_user", "user_segment_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "foot_x", "foot_y",
    "court_x_ft", "court_y_ft",
    "in_court", "transient",
]

# -- Exceptions ----------------------------------------------------------------


class InputError(ValueError):
    """Raised when an input file is missing or malformed."""


class VideoError(RuntimeError):
    """Raised when the video can't be opened or read."""


class ClickResolutionError(RuntimeError):
    """Raised when a click can't be resolved to any detected person."""


# -- Loaders -------------------------------------------------------------------


def load_court_json(path: Path) -> Dict:
    if not path.exists():
        raise InputError(f"court.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        h = data["homography"]["image_to_court"]
        w = data["court_geometry_feet"]["width_ft"]
        length = data["court_geometry_feet"]["length_ft"]
        fps = data["video"]["fps"]
    except KeyError as e:
        raise InputError(f"court.json missing field: {e}")
    return {
        "image_to_court": np.asarray(h, dtype=np.float64),
        "width_ft": float(w),
        "length_ft": float(length),
        "fps": float(fps),
    }


def load_court_zones_json(path: Path) -> Dict:
    if not path.exists():
        raise InputError(f"court_zones.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        tz = data["tracking_zone"]
        behind = tz["behind_baseline_ft"]
        beyond = tz["beyond_sideline_ft"]
    except KeyError as e:
        raise InputError(f"court_zones.json missing field: {e}")
    return {
        "behind_baseline_ft": float(behind),
        "beyond_sideline_ft": float(beyond),
    }


def load_user_clicks(path: Path) -> List[Dict]:
    """User clicks are OPTIONAL. If user_clicks.json is missing or has an empty
    clicks list, return [] — every detection keeps is_user=False and the user is
    identified downstream in Stage 2.5 from court.json's user_starting_corner.
    Clicks, when present, are an override that seeds the user directly."""
    if not path.exists():
        logger.info(
            f"no user_clicks.json at {path}; proceeding with no clicks "
            "(user identified in Stage 2.5 from court.json user_starting_corner)"
        )
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "clicks" not in data:
        raise InputError('user_clicks.json must be {"clicks": [...]}.')
    clicks = data["clicks"]
    if not isinstance(clicks, list):
        raise InputError('user_clicks.json "clicks" must be a list.')
    if len(clicks) == 0:
        logger.info("user_clicks.json has no clicks; proceeding with no clicks "
                    "(user identified in Stage 2.5 from user_starting_corner)")
        return []
    out = []
    for i, c in enumerate(clicks):
        try:
            out.append({
                "frame": int(c["frame"]),
                "x": float(c["x"]),
                "y": float(c["y"]),
            })
        except (KeyError, TypeError, ValueError):
            raise InputError(
                f"user_clicks.json[{i}] must have integer 'frame' and numeric 'x','y'"
            )
    out.sort(key=lambda c: c["frame"])
    return out


# -- Projection ----------------------------------------------------------------


def project_to_court(
    foot_xy: Tuple[float, float],
    image_to_court: np.ndarray,
) -> Tuple[float, float]:
    """Apply 3x3 homography to image-space foot point. Returns (NaN, NaN) on
    non-finite result. No interpolation, no fallback."""
    p = np.asarray([foot_xy[0], foot_xy[1], 1.0], dtype=np.float64)
    out = image_to_court @ p
    if not np.isfinite(out[2]) or abs(out[2]) < 1e-9:
        return (float("nan"), float("nan"))
    cx = out[0] / out[2]
    cy = out[1] / out[2]
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return (float("nan"), float("nan"))
    return (float(cx), float(cy))


def in_court_rectangle(
    court_x_ft: float, court_y_ft: float,
    width_ft: float, length_ft: float,
) -> bool:
    if not (np.isfinite(court_x_ft) and np.isfinite(court_y_ft)):
        return False
    return 0.0 <= court_x_ft <= width_ft and 0.0 <= court_y_ft <= length_ft


def in_tracking_zone(
    court_x_ft: float, court_y_ft: float,
    width_ft: float, length_ft: float,
    behind_baseline_ft: float, beyond_sideline_ft: float,
) -> bool:
    if not (np.isfinite(court_x_ft) and np.isfinite(court_y_ft)):
        return False
    return (
        -beyond_sideline_ft <= court_x_ft <= width_ft + beyond_sideline_ft
        and -behind_baseline_ft <= court_y_ft <= length_ft + behind_baseline_ft
    )


# -- Detection + tracking ------------------------------------------------------


def run_detection_and_tracking(
    video_path: Path,
    court: Dict,
    zones: Dict,
) -> Tuple[List[Dict], int]:
    """Run YOLO+ByteTrack over the video. Returns (detections, last_frame_idx).

    Each detection dict has keys:
        frame, t_sec, track_id, bbox, foot, court, in_court, in_tracking_zone
    """
    fps = court["fps"]
    width_ft = court["width_ft"]
    length_ft = court["length_ft"]
    image_to_court = court["image_to_court"]

    # Probe video for total frame count (used only for progress + sanity).
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video: {video_path}")
    expected_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    logger.info(f"Loading YOLO model: {MODEL}")
    model = YOLO(MODEL)

    detections: List[Dict] = []
    frame_idx = -1

    logger.info(
        f"Running tracking on {video_path} "
        f"(metadata reports ~{expected_count} frames)..."
    )
    results_stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        classes=[0],            # COCO class 0 = person
        tracker="bytetrack.yaml",
        verbose=False,
    )

    for i, r in enumerate(results_stream):
        frame_idx = i
        t_sec = frame_idx / fps if fps > 0 else 0.0

        if r.boxes is not None and r.boxes.id is not None:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            for bbox, tid in zip(boxes_xyxy, track_ids):
                x1, y1, x2, y2 = (float(v) for v in bbox)
                foot_x = (x1 + x2) / 2.0
                foot_y = y2
                cx, cy = project_to_court((foot_x, foot_y), image_to_court)
                ic = in_court_rectangle(cx, cy, width_ft, length_ft)
                itz = in_tracking_zone(
                    cx, cy, width_ft, length_ft,
                    zones["behind_baseline_ft"], zones["beyond_sideline_ft"],
                )
                detections.append({
                    "frame": frame_idx,
                    "t_sec": t_sec,
                    "track_id": int(tid),
                    "bbox": (x1, y1, x2, y2),
                    "foot": (foot_x, foot_y),
                    "court": (cx, cy),
                    "in_court": ic,
                    "in_tracking_zone": itz,
                })

        if frame_idx > 0 and frame_idx % 100 == 0:
            logger.info(
                f"  frame {frame_idx}: {len(detections)} detections so far"
            )

    if frame_idx < 0:
        raise VideoError(
            f"No frames produced from {video_path}; video may be empty or unreadable."
        )

    total_frames = frame_idx + 1
    if expected_count > 0 and total_frames < expected_count - 10:
        logger.warning(
            f"Processed {total_frames} frames; metadata reported {expected_count}. "
            f"Likely stale metadata, but worth verifying the video isn't truncated."
        )
    logger.info(
        f"Tracking complete: {total_frames} frames, {len(detections)} detections"
    )
    return detections, frame_idx


# -- Click resolution ----------------------------------------------------------


def resolve_click_to_track_id(
    click: Dict,
    detections: List[Dict],
) -> int:
    """Find the closest detection on click["frame"] to the click point.

    Distance = 0 if the click lies inside a bbox; otherwise Euclidean distance
    from the click to the bbox's nearest edge. Raises ClickResolutionError if
    no detections exist on that frame, or if the closest distance exceeds
    CLICK_MAX_DISTANCE_PX.
    """
    cx_click, cy_click = click["x"], click["y"]
    frame_dets = [d for d in detections if d["frame"] == click["frame"]]
    if not frame_dets:
        raise ClickResolutionError(
            f"Click at frame {click['frame']} ({cx_click:.0f}, {cy_click:.0f}) "
            f"has no detected persons on that frame. Re-click on a frame where "
            f"the user is clearly visible."
        )

    def bbox_distance(bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        dx = max(x1 - cx_click, 0.0, cx_click - x2)
        dy = max(y1 - cy_click, 0.0, cy_click - y2)
        return float(np.hypot(dx, dy))

    best = min(frame_dets, key=lambda d: bbox_distance(d["bbox"]))
    dist = bbox_distance(best["bbox"])
    if dist > CLICK_MAX_DISTANCE_PX:
        raise ClickResolutionError(
            f"Click at frame {click['frame']} ({cx_click:.0f}, {cy_click:.0f}) "
            f"has no detected person within {CLICK_MAX_DISTANCE_PX} px "
            f"(closest is {dist:.0f} px away). Re-click."
        )
    logger.info(
        f"Click at frame {click['frame']} resolved to track_id={best['track_id']} "
        f"(distance={dist:.0f} px)"
    )
    return best["track_id"]


# -- User segment assignment ---------------------------------------------------


def assign_user_segments(
    detections: List[Dict],
    clicks: List[Dict],
    last_processed_frame: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Walk through clicks in order. For each click, identify the user's
    track_id and walk forward marking is_user=True until the track is lost
    for more than TRACK_LOSS_TOLERANCE_FRAMES consecutive frames or the
    next click takes over.

    Returns (annotated_detections, gaps).
    """
    by_frame: Dict[int, List[Dict]] = {}
    for d in detections:
        by_frame.setdefault(d["frame"], []).append(d)

    # Copy detections with default is_user=False, segment=None.
    annotated: List[Dict] = []
    for d in detections:
        d2 = dict(d)
        d2["is_user"] = False
        d2["user_segment_id"] = None
        annotated.append(d2)

    # Index for fast (frame, track_id) -> dict mutation.
    ann_index: Dict[Tuple[int, int], Dict] = {
        (d["frame"], d["track_id"]): d for d in annotated
    }

    gaps: List[Dict] = []

    for seg_id, click in enumerate(clicks):
        user_tid = resolve_click_to_track_id(click, detections)
        next_click_frame = (
            clicks[seg_id + 1]["frame"]
            if seg_id + 1 < len(clicks)
            else last_processed_frame + 1
        )

        last_seen = click["frame"]
        for f in range(click["frame"], next_click_frame):
            frame_dets = by_frame.get(f, [])
            user_present = any(d["track_id"] == user_tid for d in frame_dets)
            if user_present:
                key = (f, user_tid)
                if key in ann_index:
                    ann_index[key]["is_user"] = True
                    ann_index[key]["user_segment_id"] = seg_id
                last_seen = f
            elif (f - last_seen) > TRACK_LOSS_TOLERANCE_FRAMES:
                gaps.append({
                    "gap_id": len(gaps),
                    "last_user_frame": int(last_seen),
                    "resumes_at_or_after": int(f),
                    "reason": "track_lost",
                })
                logger.info(
                    f"Segment {seg_id} closed: user track lost after frame "
                    f"{last_seen}; gap declared at frame {f}"
                )
                break

    return annotated, gaps


# -- Post-processing -----------------------------------------------------------


def flag_transient_tracks(annotated: List[Dict]) -> None:
    """Mutate annotated detections in place to add the 'transient' field.

    A track is transient if its lifetime in frames is less than
    TRANSIENT_LIFETIME_FRAMES, OR if none of its foot points were inside the
    tracking zone (i.e., it lived entirely off-court).
    """
    by_track: Dict[int, List[Dict]] = {}
    for d in annotated:
        by_track.setdefault(d["track_id"], []).append(d)

    transient_tracks = set()
    for tid, ds in by_track.items():
        frames = [d["frame"] for d in ds]
        lifetime = (max(frames) - min(frames) + 1) if frames else 0
        any_in_zone = any(d["in_tracking_zone"] for d in ds)
        if lifetime < TRANSIENT_LIFETIME_FRAMES or not any_in_zone:
            transient_tracks.add(tid)

    for d in annotated:
        d["transient"] = d["track_id"] in transient_tracks


def doubles_sanity_warning(
    annotated: List[Dict],
    fps: float,
) -> Optional[str]:
    """Return a warning string if more than 4 tracks persist >5s AND have
    >=80% in-court foot points; else None."""
    by_track: Dict[int, List[Dict]] = {}
    for d in annotated:
        by_track.setdefault(d["track_id"], []).append(d)

    persist_threshold_frames = DOUBLES_PERSIST_SECONDS * fps
    offending: List[int] = []
    for tid, ds in by_track.items():
        frames = [d["frame"] for d in ds]
        lifetime = (max(frames) - min(frames) + 1) if frames else 0
        if lifetime <= persist_threshold_frames:
            continue
        in_court_count = sum(1 for d in ds if d["in_court"])
        if len(ds) > 0 and in_court_count / len(ds) >= DOUBLES_IN_COURT_FRAC:
            offending.append(tid)

    if len(offending) > 4:
        return (
            f"Doubles sanity check: {len(offending)} tracks persist >"
            f"{DOUBLES_PERSIST_SECONDS}s with >="
            f"{int(DOUBLES_IN_COURT_FRAC * 100)}% in-court foot points "
            f"(track_ids: {sorted(offending)}). Likely cause: misconfigured "
            f"tracking_zone or adjacent-court contamination."
        )
    return None


# -- Output --------------------------------------------------------------------


def annotated_to_dataframe(annotated: List[Dict]) -> pd.DataFrame:
    rows = []
    for d in annotated:
        bx1, by1, bx2, by2 = d["bbox"]
        fx, fy = d["foot"]
        cx, cy = d["court"]
        rows.append({
            "frame":           int(d["frame"]),
            "t_sec":           float(d["t_sec"]),
            "track_id":        int(d["track_id"]),
            "is_user":         bool(d["is_user"]),
            "user_segment_id": d["user_segment_id"],     # int or None
            "bbox_x1":         float(bx1),
            "bbox_y1":         float(by1),
            "bbox_x2":         float(bx2),
            "bbox_y2":         float(by2),
            "foot_x":          float(fx),
            "foot_y":          float(fy),
            "court_x_ft":      float(cx) if np.isfinite(cx) else float("nan"),
            "court_y_ft":      float(cy) if np.isfinite(cy) else float("nan"),
            "in_court":        bool(d["in_court"]),
            "transient":       bool(d.get("transient", False)),
        })
    df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    # Nullable integer dtype so None becomes <NA> in parquet, not NaN-as-float.
    df["user_segment_id"] = df["user_segment_id"].astype("Int64")
    return df


# -- Main ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: track players. Reads video.mp4, court.json, "
            "court_zones.json, user_clicks.json from a per-video folder; "
            "writes players.parquet and players_pending.json."
        )
    )
    parser.add_argument(
        "video_folder", type=Path,
        help=(
            "Folder containing video.mp4, court.json, court_zones.json, "
            "user_clicks.json. Outputs are written to the same folder."
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
    zones_path = folder / "court_zones.json"
    clicks_path = folder / "user_clicks.json"
    out_parquet = folder / "players.parquet"
    out_pending = folder / "players_pending.json"

    if not video_path.exists():
        print(f"Error: video.mp4 not found in {folder}", file=sys.stderr)
        return 2

    try:
        court = load_court_json(court_path)
        zones = load_court_zones_json(zones_path)
        clicks = load_user_clicks(clicks_path)
    except InputError as e:
        print(f"Input error: {e}", file=sys.stderr)
        return 2

    logger.info(
        f"Loaded court ({court['width_ft']:.1f} x {court['length_ft']:.1f} ft, "
        f"fps={court['fps']:.2f}); {len(clicks)} click(s) loaded"
    )

    try:
        detections, last_frame = run_detection_and_tracking(
            video_path, court, zones,
        )
    except VideoError as e:
        print(f"Video error: {e}", file=sys.stderr)
        return 1

    try:
        annotated, gaps = assign_user_segments(detections, clicks, last_frame)
    except ClickResolutionError as e:
        print(f"Click resolution error: {e}", file=sys.stderr)
        return 1

    flag_transient_tracks(annotated)
    warning = doubles_sanity_warning(annotated, court["fps"])

    df = annotated_to_dataframe(annotated)
    df.to_parquet(out_parquet, index=False)
    logger.info(f"Wrote {out_parquet} ({len(df)} rows)")

    pending = {
        "gaps": gaps,
        "warnings": [warning] if warning else [],
    }
    with out_pending.open("w", encoding="utf-8") as f:
        json.dump(pending, f, indent=2)
    logger.info(
        f"Wrote {out_pending} ({len(gaps)} gap(s), "
        f"{len(pending['warnings'])} warning(s))"
    )

    if gaps:
        logger.warning(
            "Run completed with %d unresolved user-track gap(s). "
            "Add clicks to user_clicks.json and rerun.", len(gaps),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
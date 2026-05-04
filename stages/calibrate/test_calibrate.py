"""Smoke test for Stage 1 calibration.

Run with:
    pytest stages/calibrate/test_calibrate.py -v

Or directly:
    python stages/calibrate/test_calibrate.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from stages.calibrate.calibrate import (
    COURT_LENGTH_FT,
    calibrate,
    project_point,
    render_top_down_preview,
)


def _make_fake_video(path: Path, frame: np.ndarray, fps: float = 30.0) -> None:
    """Write a 5-frame video file at `path` containing the given frame."""
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter for fake video")
    try:
        for _ in range(5):
            writer.write(frame)
    finally:
        writer.release()


def test_round_trip_calibration() -> None:
    """Verify all 6 smoke-test conditions from the contract."""
    # Synthetic frame: green court rectangle from (200, 200) to (1700, 900).
    frame = np.full((1080, 1920, 3), (40, 80, 30), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (1700, 900), (50, 130, 80), -1)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        video_path = tmp / "test.mp4"
        _make_fake_video(video_path, frame)

        markers = {
            "court_corners_image": [
                [200,  900],
                [1700, 900],
                [1700, 200],
                [200,  200],
            ],
            "kitchen_line_user_image": [
                [200,  200 + (700 * 7 / 44)],
                [1700, 200 + (700 * 7 / 44)],
            ],
            "kitchen_line_opponent_image": [
                [200,  200 + (700 * 37 / 44)],
                [1700, 200 + (700 * 37 / 44)],
            ],
            "user_baseline":        "near",
            "dominant_hand":        "right",
            "user_starting_corner": "left",
            "frame_used_for_calibration": 0,
        }

        court_json, zones_json = calibrate(video_path, markers)

        # 1. Both files produced.
        assert "schema_version" in court_json
        assert "schema_version" in zones_json

        # 2. Project (0, 0) court -> image lands within 2 px of clicked corner.
        court_to_image = np.asarray(court_json["homography"]["court_to_image"])
        proj = project_point(court_to_image, (0.0, 0.0))
        assert abs(proj[0] - 200) < 2, f"x off: {proj[0]}"
        assert abs(proj[1] - 900) < 2, f"y off: {proj[1]}"

        # 3. Project (10, 7) court (center of user kitchen) lands inside the kitchen polygon.
        kitchen_poly = np.asarray(
            court_json["derived"]["user_kitchen_polygon_image"], dtype=np.float32
        )
        proj = project_point(court_to_image, (10.0, 7.0))
        inside = cv2.pointPolygonTest(kitchen_poly, (float(proj[0]), float(proj[1])), False)
        assert inside >= 0, (
            f"point not inside kitchen polygon: {proj}, polygon: {kitchen_poly.tolist()}"
        )

        # 4. Project (0, 44) court -> image lands within 2 px of far-left clicked corner.
        proj = project_point(court_to_image, (0.0, COURT_LENGTH_FT))
        assert abs(proj[0] - 200) < 2, f"far-left x off: {proj[0]}"
        assert abs(proj[1] - 200) < 2, f"far-left y off: {proj[1]}"

        # 5. RMSE is < 5 px.
        assert court_json["validation"]["homography_rmse_pixels"] < 5

        # 6. Top-down preview renders without crashing.
        image_to_court = np.asarray(court_json["homography"]["image_to_court"])
        preview = render_top_down_preview(video_path, 0, image_to_court)
        assert preview.ndim == 3
        assert preview.shape[2] == 3

    print("All 6 smoke-test conditions PASSED.")


if __name__ == "__main__":
    test_round_trip_calibration()
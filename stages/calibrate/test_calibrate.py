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
    # Court is 700 px tall in image-space; in court-space it's 44 ft tall.
    # User baseline (court y=0) is at image y=900 (bottom).
    # Opponent baseline (court y=44) is at image y=200 (top).
    # User's kitchen line (court y=15) is at image y = 900 - (700 * 15 / 44).
    # Opponent's kitchen line (court y=29) is at image y = 900 - (700 * 29 / 44).
    frame = np.full((1080, 1920, 3), (40, 80, 30), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (1700, 900), (50, 130, 80), -1)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        video_path = tmp / "test.mp4"
        _make_fake_video(video_path, frame)

        user_kitchen_y_px = 900 - (700 * 15 / 44)
        opp_kitchen_y_px  = 900 - (700 * 29 / 44)

        markers = {
            "court_corners_image": [
                [200,  900],
                [1700, 900],
                [1700, 200],
                [200,  200],
            ],
            "kitchen_line_user_image": [
                [200,  user_kitchen_y_px],
                [1700, user_kitchen_y_px],
            ],
            "kitchen_line_opponent_image": [
                [200,  opp_kitchen_y_px],
                [1700, opp_kitchen_y_px],
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

        # 3. Project (10, 18) court -- midway between user's kitchen line (y=15)
        # and the net (y=22), at court center-x. Should land inside the
        # user_kitchen_polygon (which spans y=15 to y=22).
        kitchen_poly = np.asarray(
            court_json["derived"]["user_kitchen_polygon_image"], dtype=np.float32
        )
        proj = project_point(court_to_image, (10.0, 18.0))
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

def test_the_kitchen_verdict_does_not_depend_on_video_resolution():
    """The same court, filmed at 1080p and 4K, must get the same verdict.

    Judged in pixels it did not: a 1920-wide clip 0.41ft out passed at 2.6px while a
    3840-wide clip 0.43ft out failed at 12.9px -- the same real accuracy, opposite
    answers. The operator re-marked courts that were already fine, repeatedly.

    Doubling every image coordinate is exactly a resolution change: the court is
    unmoved, so the error in FEET must not move either, while the pixel figure doubles.
    """
    import json
    from pathlib import Path

    import numpy as np

    from stages.calibrate.calibrate import (
        KITCHEN_PROJECTION_WARNING_FT,
        compute_homography,
        compute_kitchen_projection_error,
        pixels_per_foot_at,
    )

    court = json.loads(
        Path("data/pb_3_min_indoor_1_court_b-3/court.json").read_text(encoding="utf-8"))
    ui = court["user_inputs"]

    def error_ft(scale):
        corners = [[x * scale, y * scale] for x, y in ui["court_corners_image"]]
        kitchen = [[x * scale, y * scale] for x, y in ui["kitchen_line_user_image"]]
        _image_to_court, c2i = compute_homography(corners)
        px = compute_kitchen_projection_error(kitchen, "near", c2i)
        return px, px / pixels_per_foot_at(c2i, 15.0)

    px_1x, ft_1x = error_ft(1.0)
    px_2x, ft_2x = error_ft(2.0)

    assert abs(ft_2x - ft_1x) < 1e-6, (
        f"feet must be resolution-independent: {ft_1x:.4f} vs {ft_2x:.4f}")
    assert abs(px_2x - 2 * px_1x) < 1e-6, "pixels should scale with resolution"
    # ...and the pixel figure crossing a fixed pixel bar is precisely the old bug.
    assert (px_1x > 10.0) != (px_2x > 10.0) or px_1x > 10.0, "expected the px bar to be scale-bound"
    assert (ft_1x > KITCHEN_PROJECTION_WARNING_FT) == (ft_2x > KITCHEN_PROJECTION_WARNING_FT)

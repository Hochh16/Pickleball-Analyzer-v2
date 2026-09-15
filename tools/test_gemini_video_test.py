"""Pure parts of tools/gemini_video_test: matching, crop, windows, merge, scoring. No API calls."""
import pytest

from tools.gemini_video_test import (HELD_OUT, completeness, court_crop, main, merge_windows,
                                     one_to_one, rally_spans, score, window_starts)


def test_one_to_one_uses_each_side_once_shortest_first():
    # 10.0 is closest to 10.1, so 10.25 must not also claim it
    assert sorted(one_to_one([10.1, 10.25], [10.0], 0.35)) == [(0, 0)]
    assert sorted(one_to_one([1.0, 2.0], [1.1, 2.2], 0.35)) == [(0, 0), (1, 1)]
    assert one_to_one([1.0], [2.0], 0.35) == []


def test_court_crop_from_court_json_is_even_and_inside_frame():
    court = {"user_inputs": {"court_corners_image": [[287, 1266], [2299, 1747], [3104, 984], [2196, 914]]},
             "derived": {"pixels_per_foot_at_near_baseline": 103.43, "pixels_per_foot_at_far_baseline": 45.53}}
    x, y, w, h = court_crop(court, 3840, 2160)
    assert x == 0 and w % 2 == 0 and h % 2 == 0
    assert x + w <= 3840 and y + h <= 2160
    assert y < 914 - 7 * 45.53          # room above the far baseline for a standing player
    assert y + h > 1747 + 2 * 103.43    # and behind the near baseline


def test_window_starts_cover_the_clip_with_overlap():
    assert window_starts(63.0) == [0.0, 16.0, 32.0, 48.0]
    assert window_starts(10.0) == [0.0]


def test_rally_spans_pad_clip_and_use_clip_time():
    rallies = [{"start_t_sec": 40.0, "end_t_sec": 52.0}, {"start_t_sec": 57.5, "end_t_sec": 81.2},
               {"start_t_sec": 113.8, "end_t_sec": 117.0}, {"start_t_sec": 133.0, "end_t_sec": 142.0}]
    assert rally_spans(rallies, 56.0, 119.0) == [(0.0, 28.2), (54.8, 63.0)]


def test_completeness_flags_errors_short_lists_and_early_stops():
    def w(start, times):
        return {"start": start, "end": start + 60, "result": {"shots": [{"time_s": t} for t in times]}}
    assert completeness([w(0, [1, 20, 40, 55])], 60.0, 4) is None
    assert "failed" in completeness([{"start": 0, "end": 60, "error": "x"}], 60.0, 4)
    assert "listed 2" in completeness([w(0, [1, 55])], 60.0, 4)
    assert "stopped" in completeness([w(0, [1, 5, 9, 12])], 60.0, 4)


def test_merge_keeps_the_more_central_copy_and_shifts_to_video_time():
    windows = [
        {"start": 0.0, "end": 20.0, "result": {"shots": [{"time_s": 18.0, "side": "near", "type": "dink", "volley": False}],
                                             "points": []}},
        {"start": 16.0, "end": 36.0, "result": {"shots": [{"time_s": 2.1, "side": "far", "type": "drive", "volley": False}],
                                              "points": []}},
        {"start": 32.0, "end": 52.0, "error": "boom"},
    ]
    shots, points = merge_windows(windows, clip_start=56.0)
    assert len(shots) == 1 and points == []
    # 18.0 sits 2.0 s from its window's edge; 18.1 sits 2.1 s from its window's edge -> the second wins
    assert shots[0]["type"] == "drive" and shots[0]["t"] == pytest.approx(56.0 + 18.1)


def test_score_counts_junk_type_side_serves_and_ends():
    truth = {"shots": [{"t_sec": 10.0, "type": "serve", "side": "near"},
                       {"t_sec": 12.0, "type": "return", "side": "far"},
                       {"t_sec": 14.0, "type": None, "side": None}],
             "ends": [{"t_sec": 14.0, "notes": "", "reason": "net"}]}
    shots = [dict(t=10.1, type="serve", side="near"), dict(t=12.2, type="drive", side="far"),
             dict(t=20.0, type="dink", side="near")]
    points = [dict(t=10.1, last=14.5, end_reason="net")]
    r = score(shots, [10.1, 30.0], truth, points)
    assert (r["found"], r["junk"], r["type_right"], r["typed"], r["side_right"]) == (2, 1, 1, 2, 2)
    assert (r["serves_right"], r["serves_false"], r["ends_right"], r["end_reason_right"]) == (1, 1, 1, 1)


def test_held_out_clip_is_refused_without_the_flag(tmp_path):
    d = tmp_path / next(iter(HELD_OUT))
    d.mkdir()
    with pytest.raises(SystemExit, match="held-out"):
        main([str(d)])

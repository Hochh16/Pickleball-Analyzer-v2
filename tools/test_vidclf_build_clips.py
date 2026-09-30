"""Pure parts of tools/vidclf_build_clips: matching, court scale, crop geometry."""
from tools.vidclf_build_clips import (crop_box, label_candidates, one_to_one,
                                      pixels_per_foot_at)

COURT = {"user_inputs": {"court_corners_image": [[287, 1266], [2299, 1747], [3104, 984],
                                                 [2196, 914]]},
         "derived": {"pixels_per_foot_at_near_baseline": 103.43,
                     "pixels_per_foot_at_far_baseline": 45.53}}


def test_labels_are_one_to_one_so_two_candidates_cannot_claim_one_shot():
    """Loose many-to-one matching inflated two findings in September; one real shot must make
    at most one candidate 'real'."""
    truth = [{"t_sec": 10.0, "type": "drive"}]
    got = label_candidates([9.9, 10.05, 10.3], truth)
    assert [g is not None for g in got] == [False, True, False]
    assert one_to_one([1.0, 2.0], [1.1, 2.2], 0.35) == {0: 0, 1: 1}


def test_court_scale_grows_toward_the_camera():
    far_y = (914 + 984) / 2
    near_y = (1266 + 1747) / 2
    assert abs(pixels_per_foot_at(far_y, COURT) - 45.53) < 1e-6
    assert abs(pixels_per_foot_at(near_y, COURT) - 103.43) < 1e-6
    assert pixels_per_foot_at(1300, COURT) > pixels_per_foot_at(1000, COURT)


def test_crop_is_square_inside_the_frame_and_scaled_by_depth():
    l, t, s = crop_box(3800, 2100, COURT, 3840, 2160)          # a near candidate at the corner
    assert l + s <= 3840 and t + s <= 2160 and l >= 0 and t >= 0
    _, _, s_far = crop_box(2000, 950, COURT, 3840, 2160)
    _, _, s_near = crop_box(2000, 1500, COURT, 3840, 2160)
    assert s_near > s_far >= 320


def test_probe_auc_and_emit_behave():
    import numpy as np
    from tools.vidclf_probe import auc, emit, score
    y = np.array([1, 1, 0, 0])
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    # two candidates 0.1 s apart are one contact: keep the more confident
    t = np.array([10.0, 10.1, 12.0])
    assert emit(t, np.array([0.6, 0.9, 0.3]), 0.5) == [10.1]
    assert score([10.1, 15.0], [10.0]) == (1, 1)

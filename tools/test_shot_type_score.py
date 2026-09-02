"""The pairing has to be one-to-one and side-constrained, or the score measures the window.

Consecutive shots here are often under a second apart, so nearest-within-a-window silently
pairs a labelled shot with its NEIGHBOUR and the answer moves with the window rather than
with the code. That is the same fault that made the serve count read anywhere from 7 to 0
missing depending on the tolerance, and made a bounce shortfall read 148 or 63.
"""
from tools.shot_type_score import pair_up


def _t(t, side, kind):
    return {"t_sec": t, "side": side, "type": kind}


def _s(frame, side, kind):
    return {"frame": frame, "hitter_side": side, "shot_type": kind, "shot_id": frame}


def test_each_detection_is_claimed_once():
    """Two labelled shots close together must not both claim the same detection."""
    truth = [_t(10.00, "near", "drive"), _t(10.30, "far", "dink")]
    shots = [_s(600, "near", "drive")]          # 10.00s at 60fps
    got = pair_up(truth, shots, 60.0)
    claimed = [s for _, s in got if s is not None]
    assert len(claimed) == 1, "one detection was handed to two labelled shots"
    assert got[0][1] is not None and got[1][1] is None, "the closer pair should win"


def test_a_detection_on_the_wrong_side_is_not_a_match():
    """The same physical contact cannot be struck from both ends of the court."""
    truth = [_t(10.0, "near", "drive")]
    shots = [_s(600, "far", "drive")]
    assert pair_up(truth, shots, 60.0)[0][1] is None


def test_the_shortest_pair_wins_not_the_first():
    truth = [_t(10.0, "near", "drive")]
    shots = [_s(618, "near", "dink"), _s(602, "near", "drive")]
    tr, s = pair_up(truth, shots, 60.0)[0]
    assert s["frame"] == 602


def test_a_missing_side_label_still_matches():
    """Older labels carry no side; refusing them would silently shrink the sample."""
    truth = [{"t_sec": 10.0, "type": "drive"}]
    assert pair_up(truth, [_s(600, "near", "drive")], 60.0)[0][1] is not None

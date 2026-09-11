"""Smoke tests for tools/rating_leverage.py.

The one that matters is `test_probes_do_not_compound`: `_unwrap_user` returns views that
share structure with the metrics dict, so probing without a deep copy mutates the source
and every later probe compounds the last. That bug shipped once into a leverage table
where three unrelated drivers all read exactly +0.418.
"""
from __future__ import annotations

import copy

from tools import rating_leverage as RL


def _metrics() -> dict:
    """Minimal metrics.json shaped like Stage 8 v2 output, wrapped values and all."""
    def w(v):
        return {"value": v, "confidence": 1.0, "n": 10, "limited_by": "measurement"}
    return {
        "match": {
            "n_rallies": w(20),
            "rally_length_shots": w({"mean": 6.5, "median": 5.0}),
        },
        "team": {"near": w({"both_at_kitchen_frac": 0.25})},
        "players": {"user": {
            "position": w({"zone_time_frac": {"kitchen": 0.37, "transition": 0.20,
                                              "baseline": 0.43},
                           "movement": {"distance_ft_per_min": 221.0}}),
            "n_shots": w(80),
            "shot_mix": {
                "by_shot_type": w({"dink": 23, "drive": 30, "drop": 8}),
                "by_stroke_side": w({"forehand": 45, "backhand": 30}),
                "volley": w({"volley_rate": 0.225, "n_volley": 18}),
                "technique": w({
                    "by_stroke_side": {"forehand": {"n": 44, "mean": 2.63},
                                       "backhand": {"n": 26, "mean": 1.41}},
                    "knee_bend_dink": {"n": 23, "pct_good": 0.26},
                    "knee_bend_drive_by_side": {"forehand": {"n": 12, "pct_good": 0.33},
                                                "backhand": {"n": 10, "pct_good": 0.40}},
                }),
            },
            "serve": w({"n_serves": 10, "serve_fault_rate": 0.0,
                        "in_play": {"n": 10, "n_measured": 9, "in_frac": 1.0},
                        "depth": {"n": 10, "n_measured": 9, "deep_frac": 0.444}}),
            "return_in_play": w({"n": 14, "n_measured": 7, "in_frac": 0.571}),
            "return_depth": w({"n": 14, "n_measured": 7, "deep_frac": 0.714}),
            "n_returns": w(14),
            "third_shot": w({"n_third_shots": 8, "n_third_decisions": 2, "drop_rate": 0.25}),
            "transition": w({"n": 8, "n_measured": 7, "arrived_frac": 0.429}),
            "dink_control": w({"n": 23, "n_measured": 14, "in_kitchen_frac": 0.643}),
            "popup": w({"n": 23, "n_measured": 13, "popup_frac": 0.308}),
            "errors_committed": w(14),
            "ready_position": w({"n_frames": 100, "by_zone": {}, "trend_ok": False}),
        }},
    }


def test_probes_do_not_compound():
    """Every probe starts from the same baseline -- the source dict is never mutated."""
    m = _metrics()
    before = copy.deepcopy(m)
    out = RL.leverage(m)
    assert m == before, "leverage() mutated the metrics it was given"
    # Two unrelated strategy drivers must not produce an identical delta, which is the
    # fingerprint of the compounding bug.
    strat = {r["driver"]: r["d_subscore"] for r in out["rows"]
             if r["category"] == "strategy" and r["d_subscore"] > 0}
    assert len(set(strat.values())) == len(strat), f"suspiciously equal deltas: {strat}"


def test_kitchen_time_is_the_top_strategy_lever():
    out = RL.leverage(_metrics())
    strat = [r for r in out["rows"] if r["category"] == "strategy"]
    top = max(strat, key=lambda r: r["d_rating"])
    assert top["driver"] == "Time at the kitchen line"
    assert top["d_rating"] > 0


def test_ceiling_reads_inert_and_floor_reads_below_floor():
    """The two cases that look identical at one step must be told apart."""
    rows = {(r["category"], r["driver"]): r for r in RL.leverage(_metrics())["rows"]}
    ceiling = rows[("serve_return", "Serves landing in")]      # already 100%
    floor = rows[("serve_return", "Returns landing in")]       # 57%, scale starts at 70%
    assert ceiling["verdict"] == "inert"
    assert floor["verdict"] == "below_floor"
    assert floor["d_rating"] == 0.0
    assert floor["d_rating_max"] > 0.0


def test_unscored_drivers_are_inert():
    """Deliberately excluded from score_strategy -- see docs/REPORT_REDESIGN.md section 4."""
    rows = {(r["category"], r["driver"]): r for r in RL.leverage(_metrics())["rows"]}
    for name in ("Court covered", "Ready position (paddle up)", "Unforced errors"):
        assert rows[("strategy", name)]["verdict"] == "inert", name


def test_drop_rate_below_the_sample_floor_is_inert():
    """2 typed third shots, and MIN_THIRD_DECISIONS is 4."""
    rows = {(r["category"], r["driver"]): r for r in RL.leverage(_metrics())["rows"]}
    assert rows[("third_shot", "Third-shot drop rate")]["verdict"] == "inert"
    assert rows[("third_shot", "Getting to the kitchen after the 3rd")]["d_rating"] > 0


def test_category_shares_renormalise_to_one():
    rating = {"dimensions": [
        {"name": "strategy", "weight": 0.20, "confidence": 0.98, "subscore_level": 4.126},
        {"name": "volley", "weight": 0.13, "confidence": 0.47, "subscore_level": 3.587},
    ]}
    shares = RL.category_shares(rating)
    assert abs(sum(s["actual_share"] for s in shares) - 1.0) < 1e-6
    # High confidence buys a bigger share than the static weight alone implies.
    strategy = next(s for s in shares if s["category"] == "strategy")
    assert strategy["actual_share"] > strategy["weight"]

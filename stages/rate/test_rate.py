"""Stage 9 — Smoke test.

There is NO ground-truth rating (and the ball is synthetic), so accuracy cannot
be graded. The test gates on schema + internal consistency + DIRECTIONAL
MONOTONICITY (the engine must move the right way) + reliability machinery —
mirroring how Stage 6 unit-tested rule logic and Stage 8 gated on reconciliation.

End-to-end chain (synth -> S5 -> S5.5 -> S6 -> S7 -> S2.5 -> S8 -> S9) plus
pure-function checks on compute_rating / scorers (no pipeline needed).

Requires data/test_clip/ with video.mp4, court.json, court_zones.json,
players.parquet, poses.parquet, roster.json, user_clicks.json.

Usage:
    python -m stages.rate.test_rate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from stages.detect_shots.detect_shots import main as detect_main
from stages.detect_bounces.detect_bounces import main as bounces_main
from stages.classify_shots.classify_shots import main as classify_main
from stages.segment_rallies.segment_rallies import main as rallies_main
from stages.classify_tracks.classify_tracks import main as roles_main
from stages.compute_metrics.compute_metrics import main as metrics_main
from stages.rate.rate import (
    main as rate_main, compute_rating, range_of, band_of,
    score_net_play, score_error_control, score_serve, score_shot_skill,
    score_rally_consistency, WEIGHTS, USAPA_BANDS,
)
import subprocess

TEST_FOLDER = Path("data/test_clip")
SEED = 1234
GAP_FRAC = 0.20

REQUIRED_TOP_KEYS = {
    "schema_version", "source_metrics", "ball_source", "rated_role", "rating",
    "dimensions", "reliability", "skill_coverage", "usapa_anchor_version",
    "params", "warnings", "stage_version", "completed_at_utc",
}


def _fail(m): print(f"  FAIL: {m}")
def _pass(m): print(f"  PASS: {m}")


def check_fixtures() -> bool:
    needed = ["video.mp4", "court.json", "court_zones.json", "players.parquet",
              "poses.parquet", "roster.json", "user_clicks.json"]
    missing = [f for f in needed if not (TEST_FOLDER / f).exists()]
    if missing:
        print(f"Missing fixtures in {TEST_FOLDER}: {missing}")
        return False
    return True


def gen_ball(gap_frac: float) -> bool:
    cmd = [sys.executable, "tools/synth_ball.py", str(TEST_FOLDER),
           "--seed", str(SEED), "--force"]
    if gap_frac > 0:
        cmd += ["--gap-frac", str(gap_frac)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  synth_ball failed:\n{r.stderr}")
        return False
    return True


def run_chain() -> bool:
    for stage_main in (detect_main, bounces_main, classify_main, rallies_main,
                       metrics_main, rate_main):
        if stage_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
            _fail(f"stage {stage_main.__module__} crashed")
            return False
    return True


def load(name): return json.load((TEST_FOLDER / name).open(encoding="utf-8"))


# --- synthetic metrics builders (for pure-function checks) -------------------

def make_metrics(strong: bool) -> dict:
    if strong:
        return {
            "schema_version": 1, "ball_source": "synthetic",
            "players": {"user": {
                "n_shots": 120, "errors_committed": 4,
                "shot_mix": {"by_shot_type": {"dink": 40, "drop": 30, "reset": 10,
                                              "drive": 20, "lob": 10, "overhead": 5,
                                              "unknown": 5}, "volley_rate": 0.25},
                "serve": {"n_serves": 40, "serve_fault_rate": 0.0},
                "position": {"n_frames": 8000,
                             "zone_time_frac": {"kitchen": 0.5, "transition": 0.1,
                                                "baseline": 0.4},
                             "court_coverage_frac": 0.65,
                             "movement": {"distance_ft_per_min": 150.0}}}},
            "match": {"n_rallies": 40, "rally_length_shots": {"mean": 9.0},
                      "third_shot": {"drop_rate": 0.6}},
            "team": {"near": {"both_at_kitchen_frac": 0.6}},
        }
    return {
        "schema_version": 1, "ball_source": "synthetic",
        "players": {"user": {
            "n_shots": 60, "errors_committed": 30,
            "shot_mix": {"by_shot_type": {"drive": 10, "unknown": 50},
                         "volley_rate": 0.0},
            "serve": {"n_serves": 40, "serve_fault_rate": 0.3},
            "position": {"n_frames": 8000,
                         "zone_time_frac": {"kitchen": 0.05, "transition": 0.5,
                                            "baseline": 0.45},
                         "court_coverage_frac": 0.25,
                         "movement": {"distance_ft_per_min": 40.0}}}},
        "match": {"n_rallies": 40, "rally_length_shots": {"mean": 2.5},
                  "third_shot": {"drop_rate": 0.05}},
        "team": {"near": {"both_at_kitchen_frac": 0.0}},
    }


# --- Conditions --------------------------------------------------------------

def cond_schema(r) -> bool:
    if r.get("schema_version") != 1:
        _fail(f"bad schema_version {r.get('schema_version')}")
        return False
    if not REQUIRED_TOP_KEYS <= set(r.keys()):
        _fail(f"missing top keys {REQUIRED_TOP_KEYS - set(r.keys())}")
        return False
    rt = r["rating"]
    if not (1.0 <= rt["estimate"] <= 5.5):
        _fail(f"estimate {rt['estimate']} out of [1.0,5.5]")
        return False
    if rt["band"] not in [f"{b:.1f}" for b in USAPA_BANDS]:
        _fail(f"band {rt['band']} not a USAPA half-step")
        return False
    lo, hi = rt["range"]
    if not (lo <= rt["estimate"] <= hi):
        _fail(f"range {rt['range']} does not bracket estimate {rt['estimate']}")
        return False
    if not (0.0 <= rt["confidence"] <= 1.0):
        _fail(f"confidence {rt['confidence']} out of [0,1]")
        return False
    dims = r["dimensions"]
    if len(dims) != 6 or abs(sum(d["weight"] for d in dims) - 1.0) > 1e-6:
        _fail(f"dimensions: {len(dims)} dims, weights sum "
              f"{sum(d['weight'] for d in dims)}")
        return False
    for d in dims:
        if not (1.0 <= d["subscore_level"] <= 5.5):
            _fail(f"{d['name']} subscore out of range")
            return False
        if not (0.0 <= d["confidence"] <= 1.0):
            _fail(f"{d['name']} confidence out of [0,1]")
            return False
    _pass(f"rating.json valid: estimate={rt['estimate']} band={rt['band']} "
          f"range={rt['range']} conf={rt['confidence']}, 6 dims, weights=1.0")
    return True


def cond_banding(r) -> bool:
    est = r["rating"]["estimate"]
    ok = r["rating"]["band"] == band_of(est)
    (_pass if ok else _fail)(
        f"banding: band {r['rating']['band']} == nearest half-step to {est}"
        if ok else f"banding wrong: band={r['rating']['band']} est={est}")
    return ok


def cond_range_monotonic() -> bool:
    """Range half-width must shrink as confidence rises."""
    widths = []
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        lo, hi = range_of(3.0, c)
        widths.append(hi - lo)
    ok = all(widths[i] >= widths[i + 1] for i in range(len(widths) - 1)) \
        and widths[0] > widths[-1]
    (_pass if ok else _fail)(
        f"range half-width monotonic-decreasing in confidence: {widths}"
        if ok else f"range NOT monotonic in confidence: {widths}")
    return ok


def cond_reliability(r) -> bool:
    rel = r["reliability"]
    ok = rel.get("synthetic_ball") is True
    ok = ok and abs(rel["real_weight"] + rel["synthetic_weight"] - 1.0) < 1e-6
    ok = ok and any("synthetic" in w.lower() or "placeholder" in w.lower()
                    for w in r["warnings"])
    ok = ok and any("uncalibrated" in w.lower() for w in r["warnings"])
    # synthetic dims lower confidence than real dims (equal sample sufficiency
    # on test_clip: user has ample frames/shots/rallies)
    real_confs = [d["confidence"] for d in r["dimensions"] if d["data_source"] == "real"]
    synth_confs = [d["confidence"] for d in r["dimensions"] if d["data_source"] == "synthetic"]
    if real_confs and synth_confs:
        ok = ok and min(real_confs) > max(synth_confs)
    (_pass if ok else _fail)(
        "reliability: synthetic_ball=true, weights sum to 1, placeholder + "
        "uncalibrated warnings present, synthetic dims < real dims confidence"
        if ok else "reliability machinery inconsistent")
    return ok


def cond_dimension_monotonic() -> bool:
    """Each scorer must move the right way for a clearly-stronger driver."""
    failures = []
    # net_play: more kitchen -> higher
    hi_k, _ = score_net_play(
        {"position": {"zone_time_frac": {"kitchen": 0.5, "transition": 0.1}}},
        {"both_at_kitchen_frac": 0.5})
    lo_k, _ = score_net_play(
        {"position": {"zone_time_frac": {"kitchen": 0.05, "transition": 0.1}}},
        {"both_at_kitchen_frac": 0.5})
    if not hi_k > lo_k:
        failures.append(f"net_play not increasing in kitchen ({lo_k}->{hi_k})")
    # error_control: fewer errors -> higher
    hi_e, _ = score_error_control({"errors_committed": 4}, 40)
    lo_e, _ = score_error_control({"errors_committed": 30}, 40)
    if not hi_e > lo_e:
        failures.append(f"error_control not decreasing in errors ({lo_e}<-{hi_e})")
    # serve: lower fault -> higher
    hi_s, _ = score_serve({"serve": {"n_serves": 40, "serve_fault_rate": 0.0}})
    lo_s, _ = score_serve({"serve": {"n_serves": 40, "serve_fault_rate": 0.3}})
    if not hi_s > lo_s:
        failures.append(f"serve not decreasing in fault rate ({lo_s}<-{hi_s})")
    # shot_skill: higher drop_rate -> higher
    user_sk = {"n_shots": 100, "shot_mix": {"by_shot_type": {"dink": 30, "drop": 30,
               "drive": 20, "unknown": 20}}}
    hi_sk, _ = score_shot_skill(user_sk, {"third_shot": {"drop_rate": 0.6}})
    lo_sk, _ = score_shot_skill(user_sk, {"third_shot": {"drop_rate": 0.05}})
    if not hi_sk > lo_sk:
        failures.append(f"shot_skill not increasing in drop_rate ({lo_sk}->{hi_sk})")
    # rally_consistency: longer rallies -> higher
    hi_r, _ = score_rally_consistency({"shot_mix": {"volley_rate": 0.1}},
                                      {"rally_length_shots": {"mean": 9.0}})
    lo_r, _ = score_rally_consistency({"shot_mix": {"volley_rate": 0.1}},
                                      {"rally_length_shots": {"mean": 2.5}})
    if not hi_r > lo_r:
        failures.append(f"rally not increasing in length ({lo_r}->{hi_r})")
    # end-to-end: strong metrics >= weak metrics
    strong, _ = compute_rating(make_metrics(True), "synthetic")
    weak, _ = compute_rating(make_metrics(False), "synthetic")
    if not strong["estimate"] > weak["estimate"]:
        failures.append(f"end-to-end strong {strong['estimate']} not > weak "
                        f"{weak['estimate']}")
    if failures:
        _fail(f"monotonicity: {failures}")
        return False
    _pass(f"directional monotonicity: all 5 scorers + end-to-end move correctly "
          f"(strong {strong['estimate']} > weak {weak['estimate']})")
    return True


def cond_confidence_drops_with_synth() -> bool:
    """Same metrics rated as real -> higher confidence + narrower range than
    rated as synthetic. Proves the honesty machinery engages."""
    m = make_metrics(True)
    r_real, _ = compute_rating(m, "real")
    r_synth, _ = compute_rating(m, "synthetic")
    real_w = r_real["range"][1] - r_real["range"][0]
    synth_w = r_synth["range"][1] - r_synth["range"][0]
    ok = (r_real["confidence"] > r_synth["confidence"]) and (real_w < synth_w)
    (_pass if ok else _fail)(
        f"confidence drops with synthetic ball: real conf={r_real['confidence']} "
        f"(width {real_w}) > synth conf={r_synth['confidence']} (width {synth_w})"
        if ok else "synthetic penalty did not engage")
    return ok


def cond_degradation() -> bool:
    """Empty user block -> valid rating, confidence ~0, max range, no crash."""
    degraded = {"schema_version": 1, "ball_source": "synthetic",
                "players": {"user": {"n_shots": 0, "errors_committed": 0,
                                     "shot_mix": {"by_shot_type": {}},
                                     "serve": {"n_serves": 0},
                                     "position": {"n_frames": 0}}},
                "match": {"n_rallies": 0}, "team": {"near": {}}}
    rt, dims = compute_rating(degraded, "synthetic")
    lo, hi = rt["range"]
    ok = (rt["confidence"] < 0.05 and 1.0 <= rt["estimate"] <= 5.5
          and lo <= rt["estimate"] <= hi and (hi - lo) >= 2.0)
    (_pass if ok else _fail)(
        f"degradation: empty user -> conf={rt['confidence']}, range={rt['range']}, "
        f"valid, no crash" if ok else f"degradation handling wrong: {rt}")
    return ok


def cond_skill_coverage(r) -> bool:
    sc = r["skill_coverage"]
    buckets = ["covered", "proxy_or_pending", "not_captured_yet", "out_of_scope"]
    ok = all(b in sc for b in buckets)
    ok = ok and sc["covered"] == list(WEIGHTS.keys())
    # no skill in two buckets
    seen, dup = set(), []
    for b in buckets:
        for s in sc[b]:
            (dup.append(s) if s in seen else seen.add(s))
    ok = ok and not dup
    (_pass if ok else _fail)(
        f"skill_coverage: 4 buckets, covered==dimensions, no skill double-listed "
        f"({len(seen)} skills mapped)" if ok else
        f"skill_coverage invalid (dups={dup})")
    return ok


# --- Runner ------------------------------------------------------------------

def run_smoke_test() -> int:
    print(f"Stage 9 smoke test - fixture: {TEST_FOLDER}")
    print()
    if not check_fixtures():
        return 1
    for stale in ("rating.json",):
        p = TEST_FOLDER / stale
        if p.exists():
            p.unlink()

    if roles_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
        _fail("classify_tracks (Stage 2.5) crashed")
        return 1

    results = []

    # Phase A: gap variant must not crash
    print(f"Phase A: gap variant (--gap-frac {GAP_FRAC})")
    if not gen_ball(GAP_FRAC):
        return 1
    ok_gap = run_chain() and (TEST_FOLDER / "rating.json").exists()
    (_pass if ok_gap else _fail)("gap variant completed without crash")
    results.append(ok_gap)
    print()

    # Phase B: clean variant graded
    print("Phase B: clean variant")
    if not gen_ball(0.0):
        return 1
    if not run_chain():
        return 1
    r = load("rating.json")

    print("Checking conditions:")
    results.append(cond_schema(r))
    results.append(cond_banding(r))
    results.append(cond_range_monotonic())
    results.append(cond_reliability(r))
    results.append(cond_dimension_monotonic())
    results.append(cond_confidence_drops_with_synth())
    results.append(cond_degradation())
    results.append(cond_skill_coverage(r))

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())

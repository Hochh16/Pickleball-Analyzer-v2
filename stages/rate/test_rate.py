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
    main as rate_main, compute_rating, range_of, band_of, coverage_status,
    score_strategy, score_third_shot, score_dink, score_volley,
    score_serve_return, score_forehand, score_backhand, WEIGHTS, USAPA_BANDS,
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
# Stage 8 schema_version 2: every metric is a {value, confidence, n, limited_by}
# wrapper. compute_rating unwraps for the scorers and reads .confidence for the
# per-dimension confidence, so the fixtures wrap each consumed metric.

VALID_LIMITERS = {"sample_size", "measurement", "known_limit", "detection_floor"}


def W(value, confidence=0.7, n=100, limited_by="measurement") -> dict:
    return {"value": value, "confidence": confidence, "n": n,
            "limited_by": limited_by}


def make_metrics(strong: bool) -> dict:
    if strong:
        user = {
            "n_shots": W(120, 1.0, 120, "detection_floor"),
            "errors_committed": W(4, 0.7, 4, "sample_size"),
            "shot_mix": {
                "by_shot_type": W({"dink": 40, "drop": 30, "reset": 10,
                                   "drive": 20, "lob": 10, "overhead": 5,
                                   "unknown": 5}, 0.7, 120, "measurement"),
                "by_stroke_side": W({"forehand": 70, "backhand": 50}, 0.7, 120),
                "volley": W({"n_volley": 30, "volley_rate": 0.25}, 0.7, 120),
            },
            "serve": W({"n_serves": 40, "serve_fault_rate": 0.0}, 0.7, 40,
                       "sample_size"),
            "position": W({"n_frames": 8000,
                           "zone_time_frac": {"kitchen": 0.5, "transition": 0.1,
                                              "baseline": 0.4},
                           "court_coverage_frac": 0.65,
                           "movement": {"distance_ft_per_min": 150.0}},
                          0.95, 8000, "measurement"),
        }
        match = {"n_rallies": W(40, 1.0, 40, "detection_floor"),
                 "rally_length_shots": W({"mean": 9.0}, 0.84, 40, "sample_size"),
                 "third_shot": W({"drop_rate": 0.6}, 0.7, 30)}
        team = {"near": W({"both_at_kitchen_frac": 0.6}, 0.95, 8000)}
        return {"schema_version": 2, "ball_source": "synthetic",
                "players": {"user": user}, "match": match, "team": team}
    user = {
        "n_shots": W(60, 1.0, 60, "detection_floor"),
        "errors_committed": W(30, 0.7, 30, "sample_size"),
        "shot_mix": {
            "by_shot_type": W({"drive": 10, "unknown": 50}, 0.7, 60, "measurement"),
            "by_stroke_side": W({"forehand": 30, "backhand": 30}, 0.7, 60),
            "volley": W({"n_volley": 0, "volley_rate": 0.0}, 0.7, 60),
        },
        "serve": W({"n_serves": 40, "serve_fault_rate": 0.3}, 0.7, 40, "sample_size"),
        "position": W({"n_frames": 8000,
                       "zone_time_frac": {"kitchen": 0.05, "transition": 0.5,
                                          "baseline": 0.45},
                       "court_coverage_frac": 0.25,
                       "movement": {"distance_ft_per_min": 40.0}},
                      0.95, 8000, "measurement"),
    }
    match = {"n_rallies": W(40, 1.0, 40, "detection_floor"),
             "rally_length_shots": W({"mean": 2.5}, 0.84, 40, "sample_size"),
             "third_shot": W({"drop_rate": 0.05}, 0.7, 30)}
    team = {"near": W({"both_at_kitchen_frac": 0.0}, 0.95, 8000)}
    return {"schema_version": 2, "ball_source": "synthetic",
            "players": {"user": user}, "match": match, "team": team}


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
    if len(dims) != 7 or abs(sum(d["weight"] for d in dims) - 1.0) > 1e-6:
        _fail(f"dimensions: {len(dims)} dims, weights sum "
              f"{sum(d['weight'] for d in dims)}")
        return False
    if [d["name"] for d in dims] != list(WEIGHTS.keys()):
        _fail(f"dimension names {[d['name'] for d in dims]} != 7 USAPA categories")
        return False
    for d in dims:
        if not (1.0 <= d["subscore_level"] <= 5.5):
            _fail(f"{d['name']} subscore out of range")
            return False
        if not (0.0 <= d["confidence"] <= 1.0):
            _fail(f"{d['name']} confidence out of [0,1]")
            return False
        if d.get("limited_by") not in VALID_LIMITERS:
            _fail(f"{d['name']} limited_by invalid: {d.get('limited_by')}")
            return False
        if d.get("coverage_status") not in {"measured", "partial", "not_assessable"}:
            _fail(f"{d['name']} coverage_status invalid: {d.get('coverage_status')}")
            return False
    _pass(f"rating.json valid: estimate={rt['estimate']} band={rt['band']} "
          f"range={rt['range']} conf={rt['confidence']}, 7 USAPA categories, "
          f"weights=1.0, limited_by + coverage_status present")
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
    """Each scorer with a live driver must move the right way; count-only strokes
    return a valid neutral level."""
    failures = []
    # strategy: more kitchen -> higher
    hi_k, _ = score_strategy(
        {"position": {"zone_time_frac": {"kitchen": 0.5, "transition": 0.1}}},
        {"both_at_kitchen_frac": 0.5}, 40)
    lo_k, _ = score_strategy(
        {"position": {"zone_time_frac": {"kitchen": 0.05, "transition": 0.1}}},
        {"both_at_kitchen_frac": 0.5}, 40)
    if not hi_k > lo_k:
        failures.append(f"strategy not increasing in kitchen ({lo_k}->{hi_k})")
    # third_shot: higher drop_rate -> higher
    hi_t, _ = score_third_shot({"third_shot": {"drop_rate": 0.6}})
    lo_t, _ = score_third_shot({"third_shot": {"drop_rate": 0.05}})
    if not hi_t > lo_t:
        failures.append(f"third_shot not increasing in drop_rate ({lo_t}->{hi_t})")
    # dink: higher dink fraction -> higher
    hi_d, _ = score_dink({"n_shots": 100, "shot_mix": {"by_shot_type": {"dink": 60}}},
                         {"rally_length_shots": {"mean": 6.0}})
    lo_d, _ = score_dink({"n_shots": 100, "shot_mix": {"by_shot_type": {"dink": 5}}},
                         {"rally_length_shots": {"mean": 6.0}})
    if not hi_d > lo_d:
        failures.append(f"dink not increasing in dink_frac ({lo_d}->{hi_d})")
    # volley: higher volley_rate -> higher (scorer reads the flattened shot_mix)
    hi_v, _ = score_volley({"shot_mix": {"volley_rate": 0.5, "n_volley": 50}})
    lo_v, _ = score_volley({"shot_mix": {"volley_rate": 0.05, "n_volley": 5}})
    if not hi_v > lo_v:
        failures.append(f"volley not increasing in volley_rate ({lo_v}->{hi_v})")
    # serve_return: lower fault -> higher
    hi_s, _ = score_serve_return({"serve": {"n_serves": 40, "serve_fault_rate": 0.0}})
    lo_s, _ = score_serve_return({"serve": {"n_serves": 40, "serve_fault_rate": 0.3}})
    if not hi_s > lo_s:
        failures.append(f"serve_return not decreasing in fault rate ({lo_s}<-{hi_s})")
    # forehand/backhand: count-only -> valid neutral level, count surfaced
    fh, fhd = score_forehand({"n_shots": 100,
                              "shot_mix": {"by_stroke_side": {"forehand": 60}}})
    if not (1.0 <= fh <= 5.5 and fhd["forehand_count"] == 60):
        failures.append(f"forehand count-only scorer wrong ({fh}, {fhd})")
    # end-to-end: strong metrics > weak metrics
    strong, _ = compute_rating(make_metrics(True), "synthetic")
    weak, _ = compute_rating(make_metrics(False), "synthetic")
    if not strong["estimate"] > weak["estimate"]:
        failures.append(f"end-to-end strong {strong['estimate']} not > weak "
                        f"{weak['estimate']}")
    if failures:
        _fail(f"monotonicity: {failures}")
        return False
    _pass(f"directional monotonicity: 5 live scorers + count-only strokes + "
          f"end-to-end move correctly (strong {strong['estimate']} > weak "
          f"{weak['estimate']})")
    return True


def cond_confidence_drops_with_synth() -> bool:
    """Same metrics rated as real -> higher confidence than rated as synthetic
    (the 6 ball-derived categories are down-weighted; strategy stays real). The
    range WIDENS as confidence drops, but that property is proven directly by
    cond_range_monotonic on range_of; here the half-step rounding + 5.0 cap can
    make the rounded widths equal when the estimate is high, so we only require the
    range not to NARROW."""
    m = make_metrics(True)
    r_real, _ = compute_rating(m, "real")
    r_synth, _ = compute_rating(m, "synthetic")
    real_w = r_real["range"][1] - r_real["range"][0]
    synth_w = r_synth["range"][1] - r_synth["range"][0]
    ok = (r_real["confidence"] > r_synth["confidence"]) and (real_w <= synth_w)
    (_pass if ok else _fail)(
        f"confidence drops with synthetic ball: real conf={r_real['confidence']} "
        f"> synth conf={r_synth['confidence']} (range {real_w} <= {synth_w})"
        if ok else "synthetic penalty did not engage")
    return ok


def cond_degradation() -> bool:
    """Empty user block -> valid rating, confidence ~0, max range, no crash."""
    degraded = {"schema_version": 2, "ball_source": "synthetic",
                "players": {"user": {
                    "n_shots": W(0, 0.0, 0, "detection_floor"),
                    "errors_committed": W(0, 0.0, 0, "sample_size"),
                    "shot_mix": {"by_shot_type": W({}, 0.0, 0),
                                 "by_stroke_side": W({}, 0.0, 0),
                                 "volley": W({"n_volley": 0, "volley_rate": None},
                                             0.0, 0)},
                    "serve": W({"n_serves": 0}, 0.0, 0, "sample_size"),
                    "position": W({"n_frames": 0}, 0.0, 0)}},
                "match": {"n_rallies": W(0, 0.0, 0, "detection_floor"),
                          "rally_length_shots": W({}, 0.0, 0, "sample_size"),
                          "third_shot": W({}, 0.0, 0)},
                "team": {"near": W({}, 0.0, 0)}}
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

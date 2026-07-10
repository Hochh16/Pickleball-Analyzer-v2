"""Stage 10 — Smoke test.

No ground-truth plan exists, so the test gates on schema + internal consistency
+ directional behavior (mirrors Stage 9), via the end-to-end chain
(synth -> S5 -> S5.5 -> S6 -> S7 -> S2.5 -> S8 -> S9 -> S10) plus pure-function
checks on compute_plan with synthesized ratings.

Requires data/test_clip/ with video.mp4, court.json, court_zones.json,
players.parquet, poses.parquet, roster.json, user_clicks.json.

Usage:
    python -m stages.plan_improvement.test_plan_improvement
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from stages.detect_shots.detect_shots import main as detect_main
from stages.detect_bounces.detect_bounces import main as bounces_main
from stages.classify_shots.classify_shots import main as classify_main
from stages.segment_rallies.segment_rallies import main as rallies_main
from stages.classify_tracks.classify_tracks import main as roles_main
from stages.compute_metrics.compute_metrics import main as metrics_main
from stages.rate.rate import main as rate_main, WEIGHTS
from stages.plan_improvement.plan_improvement import (
    main as plan_main, compute_plan, next_half_step, OPERATOR_ACTION,
)

VALID_LIMITERS = {"sample_size", "measurement", "known_limit", "detection_floor"}
# Mirror Stage 9's real behavior for the 7 USAPA categories: serve_return is
# sample-size-limited, the rest measurement-limited (used to wrap test ratings).
DIM_LIMITERS = {"strategy": "measurement", "third_shot": "measurement",
                "dink": "measurement", "volley": "measurement",
                "serve_return": "sample_size", "forehand": "measurement",
                "backhand": "measurement"}
# Confidence per category mirroring the data reality: strategy is measured, the
# soft/net categories partial, serve_return + count-only strokes not-assessable.
DIM_CONF = {"strategy": 0.95, "third_shot": 0.6, "dink": 0.6, "volley": 0.6,
            "serve_return": 0.05, "forehand": 0.05, "backhand": 0.05}

TEST_FOLDER = Path("data/test_clip")
SEED = 1234
GAP_FRAC = 0.20

REQUIRED_TOP_KEYS = {
    "schema_version", "source_rating", "ball_source", "rated_role", "current",
    "target", "focus_areas", "strengths", "developing_capability",
    "reliability", "operator_considerations", "warnings", "params",
    "stage_version", "completed_at_utc",
}
DIM_NAMES = ["strategy", "third_shot", "dink", "volley", "serve_return",
             "forehand", "backhand"]
REAL_DIMS = {"strategy"}


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
                       metrics_main, rate_main, plan_main):
        if stage_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
            _fail(f"stage {stage_main.__module__} crashed")
            return False
    return True


def load(name): return json.load((TEST_FOLDER / name).open(encoding="utf-8"))


SKILL_COVERAGE = {
    "covered": DIM_NAMES,
    "proxy_or_pending": ["serve_depth_placement", "third_shot_drop_outcome",
                         "dink_tolerance", "forced_vs_unforced",
                         "shot_placement_targeting", "pace_power_control"],
    "not_captured_yet": ["return_of_serve", "volleys_hands_battles",
                         "attack_conversion", "reset_under_pressure",
                         "defense_scrambling", "partner_stacking_poaching",
                         "footwork_split_step", "shot_selection_iq"],
    "out_of_scope": ["spin", "score_situational_decisions"],
}

DRIVERS = {
    "strategy": {"user_kitchen_time_frac": 0.13, "both_at_kitchen_frac": 0.02,
                 "user_transition_time_frac": 0.32, "distance_ft_per_min": 120.0,
                 "unforced_error_rate": 0.36},
    "third_shot": {"third_shot_drop_rate": 0.38,
                   "third_shot_by_type": {"drop": 10, "drive": 15},
                   "per_user": False},
    "dink": {"dink_count": 20, "dink_frac": 0.2, "mean_rally_length": 5.7},
    "volley": {"volley_rate": 0.25, "n_volley": 15},
    "serve_return": {"serve_fault_rate": 0.0, "n_serves": 12, "return_metric": None},
    "forehand": {"forehand_count": 40, "forehand_frac": 0.33, "pace_mph": None,
                 "depth": None, "consistency": None},
    "backhand": {"backhand_count": 30, "backhand_frac": 0.25, "pace_mph": None,
                 "depth": None, "consistency": None},
}


def make_rating(subscores: dict, band="3.5", ball_source="synthetic") -> dict:
    dims = []
    for n in DIM_NAMES:
        is_real = n in REAL_DIMS or ball_source == "real"
        # mirror Stage 9: ball-derived categories are synth-gated x0.35 on the
        # synthetic ball; count-only strokes are capped low regardless.
        conf = DIM_CONF[n] if is_real else round(DIM_CONF[n] * 0.35, 4)
        dims.append({
            "name": n, "subscore_level": subscores[n], "weight": WEIGHTS[n],
            "confidence": conf,
            "data_source": "real" if is_real else "synthetic",
            "coverage_status": ("measured" if conf >= 0.5 else
                                "partial" if conf >= 0.1 else "not_assessable"),
            "limited_by": DIM_LIMITERS[n],
            "driver_metrics": DRIVERS[n],
        })
    return {"schema_version": 1, "ball_source": ball_source,
            "rating": {"estimate": 3.69, "band": band, "confidence": 0.55},
            "dimensions": dims, "skill_coverage": SKILL_COVERAGE}


# --- Conditions --------------------------------------------------------------

def cond_schema(p) -> bool:
    if p.get("schema_version") != 1:
        _fail(f"bad schema_version {p.get('schema_version')}")
        return False
    if not REQUIRED_TOP_KEYS <= set(p.keys()):
        _fail(f"missing top keys {REQUIRED_TOP_KEYS - set(p.keys())}")
        return False
    cur_band = float(p["current"]["band"])
    exp_target = min(5.0, cur_band + 0.5)
    if abs(p["target"]["level"] - exp_target) > 1e-6:
        _fail(f"target.level {p['target']['level']} != next half-step {exp_target}")
        return False
    _pass(f"plan valid: current {p['current']['band']} -> target "
          f"{p['target']['band']}, {len(p['focus_areas'])} focus areas")
    return True


def cond_focus_correctness(p) -> bool:
    target = p["target"]["level"]
    fa = p["focus_areas"]
    failures = []
    for f in fa:
        if f["current_subscore"] >= target:
            failures.append(f"{f['dimension']} in focus but >= target")
        if not (1 <= len(f["drills"]) <= 3):
            failures.append(f"{f['dimension']} has {len(f['drills'])} drills")
    scores = [f["priority_score"] for f in fa]
    if scores != sorted(scores, reverse=True):
        failures.append("focus areas not sorted by priority_score desc")
    if [f["priority"] for f in fa] != list(range(1, len(fa) + 1)):
        failures.append("priority not contiguous 1..n")
    if len(fa) > p["params"]["max_focus_areas"]:
        failures.append("exceeds max_focus_areas")
    # Every category lands in exactly one bucket: focus (below target, assessable),
    # strength (>= target), or not_assessable_now (data gap: near-zero confidence or
    # zero detected events). focus+strengths+not_assessable cover all dims UNLESS the
    # focus list is capped at max_focus_areas (low-priority below-target dims dropped).
    strong_dims = {s["dimension"] for s in p["strengths"]}
    focus_dims = {f["dimension"] for f in fa}
    na_dims = {e["dimension"]
               for e in p["developing_capability"]["not_assessable_now"]}
    if strong_dims & focus_dims:
        failures.append("dimension in both focus and strengths")
    covered = len(strong_dims) + len(focus_dims) + len(na_dims)
    capped = len(fa) >= p["params"]["max_focus_areas"]
    if covered != len(DIM_NAMES) and not capped:
        failures.append(f"focus+strengths+not_assessable cover {covered} != "
                        f"{len(DIM_NAMES)} dims (focus not capped)")
    if failures:
        _fail(f"focus correctness: {failures[:3]}")
        return False
    _pass(f"focus correctness: all below-target, sorted, 1-3 drills, "
          f"strengths={sorted(strong_dims)}")
    return True


def cond_provisional_flags(p) -> bool:
    failures = []
    for f in p["focus_areas"]:
        if f["data_source"] == "synthetic":
            if f["confidence"] != "provisional" or f["provisional_note"] is None:
                failures.append(f"{f['dimension']} synthetic but not flagged")
        else:
            if f["confidence"] != "high" or f["provisional_note"] is not None:
                failures.append(f"{f['dimension']} real but flagged")
    if failures:
        _fail(f"provisional flags: {failures}")
        return False
    _pass("provisional flags: synthetic->provisional+note, real->high+null")
    return True


def cond_developing(p, rating) -> bool:
    dc = p["developing_capability"]
    sc = rating["skill_coverage"]
    failures = []
    for bucket in ("proxy_or_pending", "not_captured_yet"):
        names = [e["skill"] for e in dc[bucket]]
        if names != sc[bucket]:
            failures.append(f"{bucket} skills {names} != skill_coverage {sc[bucket]}")
        for e in dc[bucket]:
            if not all(k in e for k in ("unlocked_by", "will_assess", "will_recommend")):
                failures.append(f"{e['skill']} missing descriptor fields")
    if dc["out_of_scope"] != sc["out_of_scope"]:
        failures.append("out_of_scope mismatch")
    # no developing skill is also a focus dimension
    dev_skills = {e["skill"] for e in dc["proxy_or_pending"] + dc["not_captured_yet"]}
    focus_dims = {f["dimension"] for f in p["focus_areas"]}
    if dev_skills & focus_dims:
        failures.append(f"overlap focus/developing: {dev_skills & focus_dims}")
    if failures:
        _fail(f"developing capability: {failures[:3]}")
        return False
    _pass(f"developing capability: matches skill_coverage exactly "
          f"({len(dc['proxy_or_pending'])} proxy + {len(dc['not_captured_yet'])} "
          f"not-captured + {len(dc['out_of_scope'])} oos)")
    return True


def make_real_lowconf_rating() -> dict:
    """A real-ball rating with genuinely low-confidence categories (so an operator
    limiter bites): serve_return sample_size-limited, third_shot measurement-
    limited, strategy high-confidence (must NOT trigger)."""
    dims = [
        {"name": "strategy", "subscore_level": 2.5, "weight": 0.2,
         "confidence": 0.95, "data_source": "real", "limited_by": "measurement",
         "coverage_status": "measured", "driver_metrics": DRIVERS["strategy"]},
        {"name": "third_shot", "subscore_level": 2.8, "weight": 0.18,
         "confidence": 0.45, "data_source": "real", "limited_by": "measurement",
         "coverage_status": "partial", "driver_metrics": DRIVERS["third_shot"]},
        {"name": "serve_return", "subscore_level": 2.6, "weight": 0.12,
         "confidence": 0.40, "data_source": "real", "limited_by": "sample_size",
         "coverage_status": "partial", "driver_metrics": DRIVERS["serve_return"]},
    ]
    return {"schema_version": 1, "ball_source": "real",
            "rating": {"estimate": 3.0, "band": "3.0", "confidence": 0.5},
            "dimensions": dims, "skill_coverage": SKILL_COVERAGE}


def cond_operator_considerations(p) -> bool:
    """OPERATOR section is separate from player coaching, surfaced only when a
    real-data limiter bites. (a) Player focus areas carry NO operator fields.
    (b) On the synthetic-ball pipeline it is SUPPRESSED (empty). (c) On a
    real-ball low-confidence rating it fires both categories with correct
    actions; a high-confidence real dim does NOT trigger."""
    failures = []
    for f in p["focus_areas"]:
        if "limited_by" in f or "remedy" in f:
            failures.append(f"{f['dimension']} leaks operator fields into coaching")
    oc = p.get("operator_considerations") or {}
    if "items" not in oc:
        _fail("operator_considerations missing items")
        return False
    if oc["items"]:
        failures.append(f"synthetic-ball plan should suppress operator items, "
                        f"got {len(oc['items'])}")
    plan = compute_plan(make_real_lowconf_rating(), None)
    items = plan["operator_considerations"]["items"]
    cats = {it["category"] for it in items}
    if cats != {"more_data", "capture_quality"}:
        failures.append(f"real-ball low-conf categories {cats} != both")
    affected = {a for it in items for a in it["affects"]}
    if "strategy" in affected:
        failures.append("high-confidence strategy wrongly flagged for operator")
    for it in items:
        if it["action"] != OPERATOR_ACTION[it["category"]]:
            failures.append(f"action mismatch for {it['category']}")
        if not it["affects"] or any(l not in VALID_LIMITERS for l in it["limiters"]):
            failures.append(f"bad affects/limiters in {it['category']}")
    if failures:
        _fail(f"operator considerations: {failures[:3]}")
        return False
    _pass("operator considerations: no leak into coaching; suppressed on "
          "synthetic; fires both categories on real-ball low-conf; high-conf "
          "dim not flagged")
    return True


def cond_reliability(p) -> bool:
    rel = p["reliability"]
    ok = rel.get("synthetic_ball") is True
    ok = ok and (rel["n_focus_real"] + rel["n_focus_provisional"]
                 == len(p["focus_areas"]))
    ok = ok and any("synthetic" in w.lower() or "placeholder" in w.lower()
                    for w in p["warnings"])
    ok = ok and any("uncalibrated" in w.lower() for w in p["warnings"])
    (_pass if ok else _fail)(
        "reliability: synthetic_ball=true, focus counts reconcile, placeholder + "
        "uncalibrated warnings present" if ok else "reliability inconsistent")
    return ok


def cond_directional() -> bool:
    base = {n: 3.8 for n in DIM_NAMES}
    failures = []
    # (a) lowering strategy (the measured category) below target raises its priority
    plan_hi = compute_plan(make_rating({**base, "strategy": 3.8}), None)
    plan_lo = compute_plan(make_rating({**base, "strategy": 2.5}), None)
    hi_fa = {f["dimension"]: f for f in plan_hi["focus_areas"]}
    lo_fa = {f["dimension"]: f for f in plan_lo["focus_areas"]}
    if "strategy" not in lo_fa:
        failures.append("lowered strategy not in focus areas")
    elif "strategy" in hi_fa and not (lo_fa["strategy"]["priority_score"]
                                      > hi_fa["strategy"]["priority_score"]):
        failures.append("lowering strategy did not raise priority_score")
    # (b) real ranks above synthetic at equal gap*weight: strategy(0.20,real) vs
    #     third_shot(0.18,synthetic). Equal leverage: 0.36*0.20 == 0.40*0.18 = 0.072.
    r = make_rating({**{n: 5.0 for n in DIM_NAMES},
                     "strategy": 4.0 - 0.36, "third_shot": 4.0 - 0.40})
    plan = compute_plan(r, None)
    ps = {f["dimension"]: f["priority_score"] for f in plan["focus_areas"]}
    if "strategy" in ps and "third_shot" in ps:
        if not ps["strategy"] > ps["third_shot"]:
            failures.append(f"real strategy {ps['strategy']} not > synthetic "
                            f"third_shot {ps['third_shot']} at equal leverage")
    # (c) ball_source real -> no provisional flags
    plan_real = compute_plan(make_rating(base, ball_source="real"), None)
    if any(f["confidence"] == "provisional" for f in plan_real["focus_areas"]):
        failures.append("ball_source=real still has provisional focus areas")
    if failures:
        _fail(f"directional: {failures}")
        return False
    _pass("directional: lowering subscore raises priority + adds focus; real > "
          "synthetic at equal leverage; ball_source=real clears provisional")
    return True


def cond_degradation() -> bool:
    """All categories at/above target -> empty focus; each category is either a
    strength or (if not assessable: near-zero confidence / zero events) routed to
    developing. strengths + not_assessable cover all 7; no crash."""
    allstrong = {n: 5.0 for n in DIM_NAMES}
    plan = compute_plan(make_rating(allstrong, band="2.0"), None)
    na = {e["dimension"] for e in plan["developing_capability"]["not_assessable_now"]}
    strong = {s["dimension"] for s in plan["strengths"]}
    ok = (plan["focus_areas"] == []
          and not (strong & na)
          and len(strong) + len(na) == len(DIM_NAMES)
          and plan["reliability"]["n_focus_real"] == 0
          and plan["reliability"]["n_focus_provisional"] == 0)
    (_pass if ok else _fail)(
        f"degradation: all-strong -> empty focus; {len(strong)} strengths + "
        f"{len(na)} not-assessable cover all {len(DIM_NAMES)}, valid"
        if ok else "degradation handling wrong")
    return ok


# --- Runner ------------------------------------------------------------------

def run_smoke_test() -> int:
    print(f"Stage 10 smoke test - fixture: {TEST_FOLDER}")
    print()
    if not check_fixtures():
        return 1
    for stale in ("improvement_plan.json",):
        pth = TEST_FOLDER / stale
        if pth.exists():
            pth.unlink()

    if roles_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
        _fail("classify_tracks (Stage 2.5) crashed")
        return 1

    results = []

    print(f"Phase A: gap variant (--gap-frac {GAP_FRAC})")
    if not gen_ball(GAP_FRAC):
        return 1
    ok_gap = run_chain() and (TEST_FOLDER / "improvement_plan.json").exists()
    (_pass if ok_gap else _fail)("gap variant completed without crash")
    results.append(ok_gap)
    print()

    print("Phase B: clean variant")
    if not gen_ball(0.0):
        return 1
    if not run_chain():
        return 1
    p = load("improvement_plan.json")
    rating = load("rating.json")

    print("Checking conditions:")
    results.append(cond_schema(p))
    results.append(cond_focus_correctness(p))
    results.append(cond_provisional_flags(p))
    results.append(cond_operator_considerations(p))
    results.append(cond_developing(p, rating))
    results.append(cond_reliability(p))
    results.append(cond_directional())
    results.append(cond_degradation())

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())

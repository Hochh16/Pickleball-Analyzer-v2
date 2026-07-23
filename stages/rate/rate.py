"""Stage 9 — rate (USAPA skill rating).

Map the aggregated metrics.json (Stage 8) to a USA Pickleball (USAPA) skill
rating for the USER: a continuous estimate, the nearest official half-step band,
a confidence range, and per-dimension evidence. Rule-based, anchored in the
published USA Pickleball Player Skill Rating Definitions.

ACCEPTANCE BAR: validated for LOGICAL CORRECTNESS assuming trustworthy inputs
(if the metrics were real, is the rating computed + combined correctly?), NOT
for real-world accuracy. Nothing here (nor Stages 5-8's ball-derived output) is
useful until Stage 4/4.5 (real ball) is complete. The full rating is computed
from all dimensions per the operator's explicit choice; honesty is carried by a
loud placeholder warning, lowered confidence, and a wide range.

See stages/rate/contract.md for the full spec.

Usage:
    python -m stages.rate.rate data/test_clip
    python -m stages.rate.rate data/test_clip --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 1               # rating.json output schema (additive: + limited_by, + coverage_status)
METRICS_SCHEMA_VERSION = 2       # required input metrics.json schema (Stage 8 v2)
STAGE_VERSION = "0.4.0"          # USAPA REALIGN: 6 homegrown dims -> 7 official categories
USAPA_ANCHOR_VERSION = "2024-self-rating"

# --- Config (matches contract) ----------------------------------------------
# The 7 OFFICIAL USA Pickleball rating categories (replaces the 6 homegrown dims).
# Weights are UNCALIBRATED heuristics for rough skill importance; the confidence-
# weighted estimate already prevents low-confidence categories from inflating the
# number, so weights mainly shape reported coverage. See docs/USAPA_REALIGN_DESIGN.md.
WEIGHTS = {"strategy": 0.20, "third_shot": 0.18, "dink": 0.15, "volley": 0.13,
           "serve_return": 0.12, "forehand": 0.12, "backhand": 0.10}
SYNTH_CONFIDENCE_FACTOR = 0.35   # data_conf for synthetic-ball-derived categories
NEUTRAL_PRIOR_LEVEL = 3.0        # subscore when a driver is missing / quality unmeasured
ASSESS_CONF_FLOOR = 0.10         # below this a category is 'not_assessable'
MEASURED_CONF_FLOOR = 0.50       # at/above this a category is 'measured' (else 'partial')
QUALITY_UNMEASURED_CONF = 0.05   # count-only categories (forehand/backhand): the stroke is
                                 # counted but its QUALITY (pace/depth/consistency) can't be
                                 # judged yet -> capped below the floor -> not_assessable
RANGE_MIN_HALF = 0.25            # min half-width of the confidence range
RANGE_SPAN = 1.25                # extra half-width at confidence 0
USAPA_BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

LEVEL_MIN, LEVEL_MAX = 1.0, 5.5
# Strategy is court-position-derived -> real regardless of ball_source. The other
# six are shot/ball-derived -> become 'real' once the ball is real.
REAL_DIMS = {"strategy"}
# Categories we can currently only COUNT (no quality metric) -> confidence capped.
COUNT_ONLY_DIMS = {"forehand", "backhand"}
# Category -> its count driver. A non-zero count makes the category at least
# 'partial' coverage (we can show volume even when quality is unmeasured).
COUNT_DRIVER = {"forehand": "forehand_count", "backhand": "backhand_count",
                "dink": "dink_count", "volley": "n_volley",
                "serve_return": "n_serves"}
SOFT_SHOT_TYPES = {"dink", "drop", "reset"}
VARIETY_SHOT_TYPES = {"dink", "drop", "reset", "drive", "lob", "overhead"}

EPS = 1e-9


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("rate")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"required input not found: {path}", FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --- Inline metric-wrapper helpers (Stage 8 schema_version 2) ----------------
# Stage 8 emits every metric as {value, confidence, n, limited_by}. The scorers
# operate on raw values (so the monotonicity unit tests can call them directly),
# so compute_rating unwraps to .value for scoring and reads .confidence /
# .limited_by separately for the per-dimension confidence.

def _is_wrapped(x) -> bool:
    return isinstance(x, dict) and "value" in x and "limited_by" in x


def _v(x):
    """Unwrap to .value (pass through if already raw — keeps scorers/tests simple)."""
    return x["value"] if _is_wrapped(x) else x


def _c(x, default: float = 1.0) -> float:
    """Inline .confidence (default 1.0 for an unwrapped/raw input)."""
    return float(x["confidence"]) if _is_wrapped(x) else default


def _lim(x, default: str = "measurement") -> str:
    return x["limited_by"] if _is_wrapped(x) else default


def _unwrap_user(uw: dict) -> dict:
    """Raw-value view of a wrapped user block for the scorers (restores the
    pre-v2 flat shot_mix shape: by_shot_type + by_stroke_side + n_volley +
    volley_rate)."""
    sm = uw.get("shot_mix", {}) or {}
    volley = _v(sm.get("volley")) or {}
    return {
        "n_shots": _v(uw.get("n_shots")) or 0,
        "errors_committed": _v(uw.get("errors_committed")),
        "shot_mix": {
            "by_shot_type": _v(sm.get("by_shot_type")) or {},
            "by_stroke_side": _v(sm.get("by_stroke_side")) or {},
            "n_volley": volley.get("n_volley", 0),
            "volley_rate": volley.get("volley_rate"),
        },
        "serve": _v(uw.get("serve")) or {},
        "position": _v(uw.get("position")) or {},
        "mean_post_speed_ftps": _v(uw.get("mean_post_speed_ftps")),
    }


def _unwrap_match(mw: dict) -> dict:
    return {
        "n_rallies": _v(mw.get("n_rallies")) or 0,
        "rally_length_shots": _v(mw.get("rally_length_shots")) or {},
        "third_shot": _v(mw.get("third_shot")) or {},
    }


# --- Scoring primitives ------------------------------------------------------

def lin(v: Optional[float], x0: float, y0: float, x1: float, y1: float
        ) -> Optional[float]:
    """Clamped linear map: value in [x0,x1] -> [y0,y1]. Monotonic (increasing if
    y1>y0, decreasing if y1<y0). Returns None if v is None."""
    if v is None:
        return None
    if abs(x1 - x0) < EPS:
        return y0
    t = (v - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def clamp_level(x: float) -> float:
    return max(LEVEL_MIN, min(LEVEL_MAX, x))


def band_of(estimate: float) -> str:
    e = max(1.0, min(5.0, estimate))           # band tops out at 5.0
    b = round(e * 2.0) / 2.0                    # nearest half-step
    return f"{b:.1f}"


def range_of(estimate: float, confidence: float) -> List[float]:
    h = RANGE_MIN_HALF + RANGE_SPAN * (1.0 - confidence)
    lo = round((estimate - h) * 2.0) / 2.0
    hi = round((estimate + h) * 2.0) / 2.0
    lo = max(1.0, min(5.0, lo))
    hi = max(1.0, min(5.0, hi))
    if lo > hi:
        lo = hi
    return [lo, hi]


# --- Per-category scorers (7 USAPA categories; monotonic in the primary driver) -
# Each returns (subscore_level, driver_metrics) on the 1.0-5.5 USAPA scale.
# Missing/absent drivers -> NEUTRAL_PRIOR_LEVEL (the category's confidence
# independently falls toward 0, routing it to 'not_assessable' downstream).

def coverage_status(conf: float) -> str:
    """Map a category confidence to how well we can actually assess it."""
    if conf >= MEASURED_CONF_FLOOR:
        return "measured"
    return "partial" if conf >= ASSESS_CONF_FLOOR else "not_assessable"


def score_strategy(user: dict, team_near: dict, n_rallies: int) -> Tuple[float, dict]:
    """Court positioning / NVZ approach / move-as-a-team + effort. The anchor
    category — the only one on high-confidence real position data today. Unforced
    errors are exposed as a driver but NOT scored/confidence-gated here (they're
    undetectable until ball recall improves; folding them in would either zero the
    confidence or read 'no errors = perfect')."""
    pos = user.get("position", {}) or {}
    zone = pos.get("zone_time_frac", {}) or {}
    kitchen = zone.get("kitchen")
    transition = zone.get("transition")
    both = team_near.get("both_at_kitchen_frac")
    per_min = (pos.get("movement", {}) or {}).get("distance_ft_per_min")
    errs = user.get("errors_committed")
    epr = (errs / n_rallies) if (errs is not None and n_rallies > 0) else None
    drivers = {"user_kitchen_time_frac": kitchen,
               "both_at_kitchen_frac": both,
               "user_transition_time_frac": transition,
               "distance_ft_per_min": per_min,
               "unforced_error_rate": round(epr, 4) if epr is not None else None}
    if kitchen is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    base = lin(kitchen, 0.0, 2.5, 0.6, 4.5)
    bonus = lin(both, 0.0, 0.0, 0.5, 0.4) or 0.0
    penalty = lin(transition, 0.2, 0.0, 0.6, 0.6) or 0.0
    mv = lin(per_min, 40.0, 0.0, 200.0, 0.2) or 0.0   # small footwork/effort bonus
    return clamp_level(base + bonus - penalty + mv), drivers


def score_third_shot(match: dict) -> Tuple[float, dict]:
    """Third-shot drop-to-net (soft/power mix). drop_rate is the live signal.
    NOTE: match-level (all players' 3rd shots) until per-user attribution lands."""
    ts = match.get("third_shot", {}) or {}
    drop = ts.get("drop_rate")
    drivers = {"third_shot_drop_rate": drop,
               "third_shot_by_type": ts.get("by_shot_type") or None,
               "per_user": False}
    if drop is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    return clamp_level(lin(drop, 0.1, 2.8, 0.6, 4.3)), drivers


def score_dink(user: dict, match: dict) -> Tuple[float, dict]:
    """Soft game at the kitchen: how much of the shot mix is dinks + rally
    sustain (proxy). Low confidence on a small detected-shot sample."""
    n_shots = user.get("n_shots", 0) or 0
    by_type = (user.get("shot_mix", {}) or {}).get("by_shot_type", {}) or {}
    dink_n = by_type.get("dink", 0)
    dink_frac = (dink_n / n_shots) if n_shots > 0 else None
    mean_len = (match.get("rally_length_shots", {}) or {}).get("mean")
    drivers = {"dink_count": dink_n,
               "dink_frac": round(dink_frac, 4) if dink_frac is not None else None,
               "mean_rally_length": mean_len}
    if dink_frac is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    base = lin(dink_frac, 0.0, 2.8, 0.4, 4.3)          # more dinking -> softer game
    sustain = lin(mean_len, 3.0, 0.0, 10.0, 0.3) or 0.0
    return clamp_level(base + sustain), drivers


def score_volley(user: dict) -> Tuple[float, dict]:
    """Net volleys. volley_rate is the live signal (block/reset/put-away sub-skills
    are not detected yet). Low confidence (heuristic volley flag, small sample).
    NOTE: the scorer sees the UNWRAPPED user, where Stage 8's shot_mix.volley is
    flattened to shot_mix.volley_rate / shot_mix.n_volley (see _unwrap_user)."""
    sm = user.get("shot_mix", {}) or {}
    rate = sm.get("volley_rate")
    drivers = {"volley_rate": rate, "n_volley": sm.get("n_volley", 0)}
    if rate is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    return clamp_level(lin(rate, 0.0, 2.8, 0.4, 4.2)), drivers


def score_serve_return(user: dict) -> Tuple[float, dict]:
    """Serve + return. serve_fault_rate is the only live signal, and only when
    serves are detected (n_serves>0). Return quality is not separately detected."""
    serve = user.get("serve", {}) or {}
    n_serves = serve.get("n_serves", 0) or 0
    rate = serve.get("serve_fault_rate") if n_serves > 0 else None
    drivers = {"serve_fault_rate": rate, "n_serves": n_serves,
               "return_metric": None}
    if rate is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    return clamp_level(lin(rate, 0.0, 4.2, 0.3, 2.5)), drivers


def _score_stroke(user: dict, side: str) -> Tuple[float, dict]:
    """Forehand / backhand. We can COUNT the stroke (by_stroke_side) but cannot yet
    judge its QUALITY (pace/depth/directional control/consistency) — those need
    court-plane speed (F7), landing depth (C4), and reliable stroke-side (F16). So
    the subscore is neutral and the category is confidence-capped to
    not_assessable; the count is surfaced for context only."""
    by_side = (user.get("shot_mix", {}) or {}).get("by_stroke_side", {}) or {}
    n_shots = user.get("n_shots", 0) or 0
    cnt = by_side.get(side, 0)
    frac = (cnt / n_shots) if n_shots > 0 else None
    drivers = {f"{side}_count": cnt,
               f"{side}_frac": round(frac, 4) if frac is not None else None,
               "pace_mph": None, "depth": None, "consistency": None}
    return NEUTRAL_PRIOR_LEVEL, drivers


def score_forehand(user: dict) -> Tuple[float, dict]:
    return _score_stroke(user, "forehand")


def score_backhand(user: dict) -> Tuple[float, dict]:
    return _score_stroke(user, "backhand")


# --- Skill coverage map (static; surfaced so nothing implies full coverage) --

def skill_coverage_block() -> dict:
    return {
        "covered": list(WEIGHTS.keys()),
        "proxy_or_pending": ["serve_depth_placement", "third_shot_drop_outcome",
                             "dink_tolerance", "forced_vs_unforced",
                             "shot_placement_targeting", "pace_power_control"],
        "not_captured_yet": ["return_of_serve", "volleys_hands_battles",
                             "attack_conversion", "reset_under_pressure",
                             "defense_scrambling", "partner_stacking_poaching",
                             "footwork_split_step", "shot_selection_iq"],
        "out_of_scope": ["spin", "score_situational_decisions"],
        "note": ("not_captured_yet skills are NOT reflected in the rating; they "
                 "need new metrics/stages. Surfaced so Stage 10 / the UI don't "
                 "imply full coverage."),
    }


# --- Core rating (pure function over metrics + ball_source) ------------------

def compute_rating(metrics: dict, ball_source: str,
                   synth_factor: float = SYNTH_CONFIDENCE_FACTOR
                   ) -> Tuple[dict, List[dict]]:
    """Returns (rating_dict, dimensions_list). Pure — no I/O — so the smoke test
    can call it on synthesized metrics for monotonicity checks."""
    players = metrics.get("players", {}) or {}
    user_w = players.get("user", {}) or {}
    match_w = metrics.get("match", {}) or {}
    team_near_w = (metrics.get("team", {}) or {}).get("near", {}) or {}

    # Raw-value views for the scorers (which operate on unwrapped values).
    user = _unwrap_user(user_w)
    match = _unwrap_match(match_w)
    team_near = _v(team_near_w) or {}
    n_rallies = match["n_rallies"]

    ball_is_synth = (ball_source == "synthetic")

    # Inline (confidence, limited_by) of each driving Stage 8 metric. A
    # dimension's confidence = the MIN over its drivers (weakest evidence caps
    # it); the binding driver's limited_by names the remedy.
    sm_w = user_w.get("shot_mix", {}) or {}
    drv = {
        "position":   (_c(user_w.get("position")),   _lim(user_w.get("position"))),
        "team_near":  (_c(team_near_w),               _lim(team_near_w)),
        "shot_type":  (_c(sm_w.get("by_shot_type")),  _lim(sm_w.get("by_shot_type"))),
        "stroke_side": (_c(sm_w.get("by_stroke_side")), _lim(sm_w.get("by_stroke_side"))),
        "third_shot": (_c(match_w.get("third_shot")), _lim(match_w.get("third_shot"))),
        "serve":      (_c(user_w.get("serve")),       _lim(user_w.get("serve"))),
        "rally_len":  (_c(match_w.get("rally_length_shots")),
                       _lim(match_w.get("rally_length_shots"))),
        "volley":     (_c(sm_w.get("volley")),        _lim(sm_w.get("volley"))),
    }
    # A category's confidence = the MIN over its driving Stage 8 metrics (weakest
    # evidence caps it); the binding driver's limited_by names the remedy.
    DIM_DRIVERS = {
        "strategy": ["position", "team_near"],
        "third_shot": ["third_shot"],
        "dink": ["shot_type", "rally_len"],
        "volley": ["volley"],
        "serve_return": ["serve"],
        "forehand": ["stroke_side"],
        "backhand": ["stroke_side"],
    }

    # (name, subscore, drivers)
    raw = [
        ("strategy", *score_strategy(user, team_near, n_rallies)),
        ("third_shot", *score_third_shot(match)),
        ("dink", *score_dink(user, match)),
        ("volley", *score_volley(user)),
        ("serve_return", *score_serve_return(user)),
        ("forehand", *score_forehand(user)),
        ("backhand", *score_backhand(user)),
    ]

    dimensions: List[dict] = []
    for name, subscore, drivers in raw:
        base_conf, binding_lim = min((drv[k] for k in DIM_DRIVERS[name]),
                                     key=lambda t: t[0])
        inherent_real = name in REAL_DIMS
        # ball/shot-derived categories become 'real' once the ball is real.
        is_real_source = inherent_real or (not ball_is_synth)
        # Synthetic gate: inline confidence is artificially clean on the synthetic
        # ball, so ball-derived categories are down-weighted until the ball is real
        # (Stage 8 contract § Synthetic-ball interaction). Inactive on real ball.
        gate = 1.0 if is_real_source else synth_factor
        conf = base_conf * gate
        # Count-only categories (forehand/backhand): the stroke is counted but its
        # quality is unmeasured, so cap confidence below the floor -> not_assessable.
        if name in COUNT_ONLY_DIMS:
            conf = min(conf, QUALITY_UNMEASURED_CONF)
        conf = round(conf, 4)
        # Coverage vs the RATING is quality-driven (a count-only category can't move
        # the number confidently). But the report's coverage BADGE should tell the
        # truth about what we can show: a category with a reliable COUNT is at least
        # 'partial' (we know how MUCH, just not how WELL). Only a category with no
        # count stays 'not_assessable'. This keeps the headline honest while the
        # chart reflects the validated volumes (dink/serve/volley counts checked
        # against operator truth 2026-07-22).
        cov = coverage_status(conf)
        cdk = COUNT_DRIVER.get(name)
        if cov == "not_assessable" and cdk and (drivers.get(cdk) or 0) > 0:
            cov = "partial"
        dimensions.append({
            "name": name,
            "subscore_level": round(subscore, 3),
            "weight": WEIGHTS[name],
            "confidence": conf,
            "limited_by": binding_lim,
            "data_source": "real" if is_real_source else "synthetic",
            "coverage_status": cov,
            "driver_metrics": drivers,
        })

    # Confidence-weight the estimate: each category's influence is its static
    # weight x its confidence, renormalized. A category we can't trust (e.g.
    # serve_return at confidence 0 because no serves are detected, or forehand/
    # backhand where the stroke is counted but its quality is unmeasured) no longer
    # inflates the headline number — the estimate leans on the categories we can
    # actually measure (today, mostly Strategy). Falls back to the plain
    # static-weight sum only if NO category carries any confidence (avoids /0).
    eff = [(d, d["weight"] * d["confidence"]) for d in dimensions]
    eff_sum = sum(w for _, w in eff)
    if eff_sum > 0:
        estimate = clamp_level(sum(d["subscore_level"] * w for d, w in eff) / eff_sum)
    else:
        estimate = clamp_level(sum(d["subscore_level"] * d["weight"] for d in dimensions))
    # Reported confidence stays the static-weight-average of dim confidences: it
    # measures how much of the INTENDED skill picture is trustworthy (still low
    # when half the weight rests on zero-confidence dims), which keeps the range
    # honestly wide — the estimate is a confident read of INCOMPLETE coverage.
    confidence = round(sum(d["confidence"] * d["weight"] for d in dimensions), 4)
    rating = {
        "estimate": round(estimate, 2),
        "band": band_of(estimate),
        "range": range_of(estimate, confidence),
        "confidence": confidence,
    }
    return rating, dimensions


# --- Main pipeline -----------------------------------------------------------

def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    metrics_path = folder / "metrics.json"
    out_path = folder / "rating.json"

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.",
             FileExistsError)

    metrics = load_json(metrics_path)
    if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
        fail(f"metrics.json schema_version={metrics.get('schema_version')} "
             f"unexpected (expects {METRICS_SCHEMA_VERSION}; run Stage 8 v0.2.0+)",
             ValueError)

    ball_source = metrics.get("ball_source") or "real"
    ball_is_synth = (ball_source == "synthetic")
    if ball_is_synth:
        log.warning("ball_source is SYNTHETIC: the rating point estimate is "
                    "PLACEHOLDER (most weight is synthetic-derived).")

    rating, dimensions = compute_rating(metrics, ball_source, args.synth_confidence_factor)

    real_weight = round(sum(d["weight"] for d in dimensions
                            if d["data_source"] == "real"), 4)
    synthetic_weight = round(1.0 - real_weight, 4)

    measured = [d["name"] for d in dimensions if d["coverage_status"] == "measured"]
    assessable = [d["name"] for d in dimensions
                  if d["coverage_status"] != "not_assessable"]
    not_assessable = [d["name"] for d in dimensions
                      if d["coverage_status"] == "not_assessable"]

    user = (metrics.get("players", {}) or {}).get("user", {}) or {}
    n_shots = _v(user.get("n_shots")) or 0

    warnings: List[str] = []
    if ball_is_synth:
        warnings.append(
            f"ball_source is 'synthetic': the rating point estimate is "
            f"PLACEHOLDER ({synthetic_weight:.2f} of its weight is "
            f"synthetic-ball-derived). Treat as a scaffold until ball detection "
            f"v4; confidence is reduced and range widened accordingly.")
    warnings.append(
        "Rating thresholds are UNCALIBRATED heuristics anchored to USAPA "
        "descriptions — no corpus of rated footage exists yet (see KNOWN_ISSUES.md).")
    if len(measured) <= 2 and len(dimensions) == 7:
        lean = ", ".join(measured) if measured else "no category"
        warnings.append(
            f"USAPA COVERAGE: this estimate currently rests almost entirely on "
            f"{lean} — {len(not_assessable)} of 7 categories are not yet reliably "
            f"measured ({', '.join(not_assessable)}). The 6 shot-based categories "
            f"need ball recall / serve detection / stroke-side / shot speed before "
            f"they carry signal; they are confidence-gated, not guessed.")
    if n_shots == 0:
        warnings.append("user has zero shots in metrics.json: rating is "
                        "near-zero-confidence (degraded input).")
    if rating["confidence"] < 0.25:
        warnings.append(f"overall confidence {rating['confidence']} is very low: "
                        f"rating range is wide; interpret as a rough scaffold only.")

    log.info(f"rating: estimate={rating['estimate']} band={rating['band']} "
             f"range={rating['range']} confidence={rating['confidence']} "
             f"(real_weight={real_weight})")

    out = {
        "schema_version": SCHEMA_VERSION,
        "source_metrics": str(metrics_path),
        "ball_source": ball_source,
        "rated_role": "user",
        "rating": rating,
        "dimensions": dimensions,
        "reliability": {
            "synthetic_ball": ball_is_synth,
            "real_weight": real_weight,
            "synthetic_weight": synthetic_weight,
            "measured_categories": measured,
            "assessable_categories": assessable,
            "not_assessable_categories": not_assessable,
            "note": (f"{synthetic_weight:.2f} of the rating weight comes from "
                     f"synthetic-ball-derived dimensions; estimate is "
                     f"PLACEHOLDER until ball v4. confidence + range reflect this."
                     if ball_is_synth else
                     "ball_source is real; all dimensions count as real data."),
        },
        "skill_coverage": skill_coverage_block(),
        "usapa_anchor_version": USAPA_ANCHOR_VERSION,
        "params": {
            "synth_confidence_factor": args.synth_confidence_factor,
            "weights": WEIGHTS,
        },
        "warnings": warnings,
        "stage_version": STAGE_VERSION,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    log.info(f"wrote {out_path}")
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 9 — rate (USAPA skill rating)")
    p.add_argument("folder", type=Path, help="per-video folder with metrics.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--synth-confidence-factor", type=float,
                   default=SYNTH_CONFIDENCE_FACTOR, dest="synth_confidence_factor")
    p.add_argument("--role", default="user", choices=["user"],
                   help="reserved for future multi-role rating; v1 = user only")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    log = setup_logging(args.log_level)
    try:
        run(args.folder, args, log)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        log.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

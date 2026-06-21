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

SCHEMA_VERSION = 1               # rating.json output schema (additive: + limited_by)
METRICS_SCHEMA_VERSION = 2       # required input metrics.json schema (Stage 8 v2)
STAGE_VERSION = "0.2.0"          # Foundation #3 — consume inline metric confidence
USAPA_ANCHOR_VERSION = "2024-self-rating"

# --- Config (matches contract) ----------------------------------------------
WEIGHTS = {"net_play": 0.20, "movement": 0.10, "error_control": 0.25,
           "shot_skill": 0.25, "serve": 0.10, "rally_consistency": 0.10}
SYNTH_CONFIDENCE_FACTOR = 0.35   # data_conf for synthetic-ball-derived dimensions
NEUTRAL_PRIOR_LEVEL = 3.0        # subscore when a driver is missing
RANGE_MIN_HALF = 0.25            # min half-width of the confidence range
RANGE_SPAN = 1.25                # extra half-width at confidence 0
USAPA_BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

LEVEL_MIN, LEVEL_MAX = 1.0, 5.5
# Inherent data source per dimension (before ball_source is considered).
REAL_DIMS = {"net_play", "movement"}
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


# --- Per-dimension scorers (documented thresholds; monotonic in each driver) -
# Each returns (subscore_level, driver_metrics). subscore is on the 1.0-5.5
# scale, USAPA-anchored. Missing drivers -> NEUTRAL_PRIOR_LEVEL (sample_conf
# will independently drive dim confidence toward 0).

def score_net_play(user: dict, team_near: dict) -> Tuple[float, dict]:
    pos = user.get("position", {}) or {}
    zone = pos.get("zone_time_frac", {}) or {}
    kitchen = zone.get("kitchen")
    transition = zone.get("transition")
    both = team_near.get("both_at_kitchen_frac")
    drivers = {"user_kitchen_time_frac": kitchen,
               "both_at_kitchen_frac": both,
               "user_transition_time_frac": transition}
    if kitchen is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    base = lin(kitchen, 0.0, 2.5, 0.6, 4.5)
    bonus = lin(both, 0.0, 0.0, 0.5, 0.4) or 0.0
    penalty = lin(transition, 0.2, 0.0, 0.6, 0.6) or 0.0
    return clamp_level(base + bonus - penalty), drivers


def score_movement(user: dict) -> Tuple[float, dict]:
    pos = user.get("position", {}) or {}
    coverage = pos.get("court_coverage_frac")
    per_min = (pos.get("movement", {}) or {}).get("distance_ft_per_min")
    drivers = {"court_coverage_frac": coverage, "distance_ft_per_min": per_min}
    if coverage is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    base = lin(coverage, 0.2, 2.5, 0.7, 4.5)
    # per_min: non-decreasing capped contribution (frantic-overmovement penalty
    # is a future refinement; kept monotonic for v1).
    mv = lin(per_min, 40.0, 0.0, 200.0, 0.3) or 0.0
    return clamp_level(base + mv), drivers


def score_error_control(user: dict, n_rallies: int) -> Tuple[float, dict]:
    errs = user.get("errors_committed")
    epr = (errs / n_rallies) if (errs is not None and n_rallies > 0) else None
    drivers = {"errors_per_rally": round(epr, 4) if epr is not None else None,
               "unforced_rate": None}   # pending_real_ball
    if epr is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    # fewer errors per rally -> higher level (decreasing map)
    return clamp_level(lin(epr, 0.1, 4.5, 0.7, 2.5)), drivers


def score_shot_skill(user: dict, match: dict) -> Tuple[float, dict]:
    n_shots = user.get("n_shots", 0) or 0
    by_type = (user.get("shot_mix", {}) or {}).get("by_shot_type", {}) or {}
    drop_rate = (match.get("third_shot", {}) or {}).get("drop_rate")
    variety = sum(1 for t in VARIETY_SHOT_TYPES if by_type.get(t, 0) > 0)
    soft = sum(by_type.get(t, 0) for t in SOFT_SHOT_TYPES)
    soft_frac = (soft / n_shots) if n_shots > 0 else None
    unknown_frac = (by_type.get("unknown", 0) / n_shots) if n_shots > 0 else None
    drivers = {"third_shot_drop_rate": drop_rate, "shot_variety": variety,
               "soft_game_frac": round(soft_frac, 4) if soft_frac is not None else None,
               "unknown_type_frac": round(unknown_frac, 4) if unknown_frac is not None else None}
    if n_shots == 0:
        return NEUTRAL_PRIOR_LEVEL, drivers
    base = lin(drop_rate, 0.1, 2.8, 0.6, 4.3)
    if base is None:
        base = NEUTRAL_PRIOR_LEVEL
    variety_bonus = lin(float(variety), 2.0, 0.0, 6.0, 0.4) or 0.0
    soft_bonus = lin(soft_frac, 0.2, 0.0, 0.6, 0.3) or 0.0
    unknown_penalty = lin(unknown_frac, 0.2, 0.0, 0.6, 0.5) or 0.0
    return clamp_level(base + variety_bonus + soft_bonus - unknown_penalty), drivers


def score_serve(user: dict) -> Tuple[float, dict]:
    serve = user.get("serve", {}) or {}
    n_serves = serve.get("n_serves", 0) or 0
    rate = serve.get("serve_fault_rate") if n_serves > 0 else None
    drivers = {"serve_fault_rate": rate, "n_serves": n_serves}
    if rate is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    # lower fault rate -> higher level (decreasing map)
    return clamp_level(lin(rate, 0.0, 4.2, 0.3, 2.5)), drivers


def score_rally_consistency(user: dict, match: dict) -> Tuple[float, dict]:
    mean_len = (match.get("rally_length_shots", {}) or {}).get("mean")
    volley_rate = (user.get("shot_mix", {}) or {}).get("volley_rate")
    drivers = {"mean_rally_length": mean_len, "volley_rate": volley_rate}
    if mean_len is None:
        return NEUTRAL_PRIOR_LEVEL, drivers
    base = lin(mean_len, 2.0, 2.5, 10.0, 4.3)
    volley_bonus = lin(volley_rate, 0.0, 0.0, 0.3, 0.4) or 0.0
    return clamp_level(base + volley_bonus), drivers


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
        "errors":     (_c(user_w.get("errors_committed")),
                       _lim(user_w.get("errors_committed"))),
        "shot_type":  (_c(sm_w.get("by_shot_type")),  _lim(sm_w.get("by_shot_type"))),
        "third_shot": (_c(match_w.get("third_shot")), _lim(match_w.get("third_shot"))),
        "serve":      (_c(user_w.get("serve")),       _lim(user_w.get("serve"))),
        "rally_len":  (_c(match_w.get("rally_length_shots")),
                       _lim(match_w.get("rally_length_shots"))),
        "volley":     (_c(sm_w.get("volley")),        _lim(sm_w.get("volley"))),
    }
    DIM_DRIVERS = {
        "net_play": ["position", "team_near"],
        "movement": ["position"],
        "error_control": ["errors"],
        "shot_skill": ["shot_type", "third_shot"],
        "serve": ["serve"],
        "rally_consistency": ["rally_len", "volley"],
    }

    # (name, subscore, drivers)
    raw = [
        ("net_play", *score_net_play(user, team_near)),
        ("movement", *score_movement(user)),
        ("error_control", *score_error_control(user, n_rallies)),
        ("shot_skill", *score_shot_skill(user, match)),
        ("serve", *score_serve(user)),
        ("rally_consistency", *score_rally_consistency(user, match)),
    ]

    dimensions: List[dict] = []
    for name, subscore, drivers in raw:
        base_conf, binding_lim = min((drv[k] for k in DIM_DRIVERS[name]),
                                     key=lambda t: t[0])
        inherent_real = name in REAL_DIMS
        # ball-derived dims become 'real' once the ball is real.
        is_real_source = inherent_real or (not ball_is_synth)
        # Synthetic gate: inline confidence is artificially clean on the synthetic
        # ball, so ball-derived dims are still down-weighted until the ball is real
        # (Stage 8 contract § Synthetic-ball interaction). Inactive on real ball.
        gate = 1.0 if is_real_source else synth_factor
        conf = round(base_conf * gate, 4)
        dimensions.append({
            "name": name,
            "subscore_level": round(subscore, 3),
            "weight": WEIGHTS[name],
            "confidence": conf,
            "limited_by": binding_lim,
            "data_source": "real" if is_real_source else "synthetic",
            "driver_metrics": drivers,
        })

    estimate = clamp_level(sum(d["subscore_level"] * d["weight"] for d in dimensions))
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

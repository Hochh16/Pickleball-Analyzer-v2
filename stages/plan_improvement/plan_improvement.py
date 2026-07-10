"""Stage 10 — plan improvement.

Turn the Stage 9 rating.json (+ Stage 8 metrics.json) into an
improvement_plan.json for the USER: the gap to the next USAPA half-step, a
prioritized set of focus areas (each with a data-grounded finding + 1-3 drills
from a built-in library), and a forward-looking developing_capability block
that scaffolds in the skills not yet measurable — so the plan reaches full
capability once real ball detection (Stage 4/4.5) + the missing-skill metrics
land.

Rule-based. Synthetic-ball-derived focus areas are flagged 'provisional';
honesty is carried by per-area flags, reliability counts, and loud warnings.

See stages/plan_improvement/contract.md for the full spec.

Usage:
    python -m stages.plan_improvement.plan_improvement data/test_clip
    python -m stages.plan_improvement.plan_improvement data/test_clip --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 1               # improvement_plan.json schema (additive: +limited_by/remedy)
METRICS_SCHEMA_VERSION = 2       # Stage 8 metrics.json schema (read-but-not-consumed)
STAGE_VERSION = "0.4.0"          # plain-English findings with good/bad verdicts (0.3.0: conf gating)

# --- Config (matches contract) ----------------------------------------------
MAX_FOCUS_AREAS = 4
CONFIDENCE_WEIGHT_FLOOR = 0.5    # min multiplier applied to a dim's leverage
TARGET_CAP = 5.0                 # USAPA top band
USAPA_BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Foundation #3: OPERATOR-side analysis-reliability notes — kept SEPARATE from
# (and lower-priority than) player coaching. A dimension's limited_by (from
# Stage 9) maps to an operator action category, surfaced only when a real-data
# limiter materially bites (confidence < OPERATOR_CONF_FLOOR). The operator (who
# records/configures the capture) may be a different person than the player.
OPERATOR_CONF_FLOOR = 0.6        # real-data dim confidence below this = limiter bites
ASSESS_CONF_FLOOR = 0.1          # real-data dim confidence below this = a DATA GAP, not a
                                 # coaching signal -> routed to developing_capability
UNMEASURED_REASON = {
    "serve": "Serves aren't reliably detected yet (needs serve detection, C3) — "
             "can't assess serve skill.",
    "error_control": "Errors aren't attributable yet (rally end_reasons are "
                     "bounce-recall-gated, mostly unknown) — can't assess error control.",
}
LIMITER_CATEGORY = {"sample_size": "more_data",
                    "measurement": "capture_quality",
                    "known_limit": "capture_quality",
                    "detection_floor": "capture_quality"}
OPERATOR_ACTION = {
    "more_data": ("Record longer sessions, or combine clips across sessions - "
                  "these assessments currently rest on few rallies."),
    "capture_quality": ("Capped by single-camera 2D (no ball height/depth) - a "
                        "higher-mounted or second camera would improve precision."),
}

EPS = 1e-9


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("plan_improvement")
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


# --- Static knowledge: why-it-matters, drills, developing descriptors -------

WHY = {
    "net_play": ("USAPA 3.5-4.0 players win the net: they get up to the kitchen "
                 "line, hold it together as a team, and avoid getting stuck back "
                 "in the middle of the court."),
    "movement": ("Good footwork and quick recovery keep you in position to play "
                 "the next ball instead of reaching or scrambling."),
    "error_control": ("Cutting unforced errors is the fastest way up the rating "
                      "ladder — higher levels simply miss less."),
    "shot_skill": ("Reliable third-shot drops and a varied, controlled shot mix "
                   "are what separate 3.5 from 4.0+ players."),
    "serve": ("A consistent, deep serve denies the returner easy offense and "
              "starts the point on your terms."),
    "rally_consistency": ("Sustaining longer rallies and being comfortable in "
                          "net exchanges reflects control under pressure."),
}

DRILLS = {
    "get_to_line": {"name": "Get-to-the-line",
                    "cue": "After every return, sprint to the NVZ line and freeze "
                           "before the next ball — 'get to the line, then play.'"},
    "move_as_unit": {"name": "Move as a unit",
                     "cue": "Shadow your partner across the kitchen keeping ~8-10 ft "
                            "spacing; close the middle together."},
    "transition_resets": {"name": "Transition resets",
                          "cue": "From mid-court, reset a hard feed softly into the "
                                 "kitchen, then advance to the line."},
    "split_step_recover": {"name": "Split-step + recover",
                           "cue": "Split-step as your opponent contacts the ball, then "
                                  "recover to a paddle-up ready position after each shot."},
    "coverage_ladder": {"name": "Court-coverage ladder",
                        "cue": "Cone drill covering your half — touch each cone and "
                               "reset to center between reps."},
    "coop_dink_count": {"name": "Cooperative dink count",
                        "cue": "Cross-court dink rally to 20 without an error before "
                               "adding pace."},
    "reset_under_pressure": {"name": "Reset under pressure",
                            "cue": "Have a partner feed hard balls; block them softly "
                                   "into the kitchen instead of swinging."},
    "third_shot_drop_reps": {"name": "Third-shot-drop reps",
                            "cue": "From the baseline, drop the 3rd ball into the "
                                   "kitchen — target 7/10 before moving up."},
    "shot_variety_ladder": {"name": "Shot-variety ladder",
                           "cue": "Cycle drive / drop / dink on command so you can "
                                  "hit the right shot for the situation."},
    "soft_game_targets": {"name": "Soft-game targets",
                         "cue": "Dink and drop to floor targets in the kitchen to "
                                "build touch and placement."},
    "deep_serve_targets": {"name": "Deep-serve targets",
                          "cue": "Serve to the deep third of the box — 8/10 in and "
                                 "deep before adding spin or pace."},
    "pre_serve_routine": {"name": "Pre-serve routine",
                         "cue": "Same toss and contact point every serve; groove a "
                                "repeatable routine."},
    "sustained_rally_game": {"name": "Sustained-rally game",
                            "cue": "Cooperative rally to 15 shots before anyone tries "
                                   "to win the point."},
    "volley_exchanges": {"name": "Volley exchanges",
                        "cue": "Net-height volley-to-volley exchanges with a partner; "
                               "soft hands, paddle in front."},
}

# Developing-capability descriptors (skills not yet fully measured).
DEVELOPING = {
    # proxy_or_pending (mostly unlocked by real ball v4)
    "serve_depth_placement": {
        "unlocked_by": "real ball detection v4 (serve landing point)",
        "will_assess": "Serve depth and placement within the service box.",
        "will_recommend": "Deep / wide serve targeting drills."},
    "third_shot_drop_outcome": {
        "unlocked_by": "real ball detection v4 (Stage 8 pending_real_ball)",
        "will_assess": "Whether third-shot drops actually win the kitchen approach.",
        "will_recommend": "Drop-and-advance pattern drills, not just drop reps."},
    "dink_tolerance": {
        "unlocked_by": "real ball detection v4 (Stage 8 pending_real_ball)",
        "will_assess": "How many dinks you sustain before erroring or attacking.",
        "will_recommend": "Extended dink-rally patience drills."},
    "forced_vs_unforced": {
        "unlocked_by": "real ball detection v4 (Stage 8 pending_real_ball)",
        "will_assess": "Split your errors into forced vs unforced.",
        "will_recommend": "Consistency vs shot-selection drills based on which dominates."},
    "shot_placement_targeting": {
        "unlocked_by": "real ball v4 + Stage 2.5 v2 (reliable opponent roles)",
        "will_assess": "How often you target the opponent's backhand / open court.",
        "will_recommend": "Directional targeting and opponent-backhand drills."},
    "pace_power_control": {
        "unlocked_by": "real ball detection v4 (reliable shot speeds)",
        "will_assess": "Whether you apply and absorb pace at the right moments.",
        "will_recommend": "Pace-change and speed-up timing drills."},
    # not_captured_yet (need a new metric or stage)
    "return_of_serve": {
        "unlocked_by": "new return-quality metric (real ball + positioning)",
        "will_assess": "Return depth and whether you get to the net behind it.",
        "will_recommend": "Deep-return + split-step-and-advance drills."},
    "volleys_hands_battles": {
        "unlocked_by": "new net-exchange/reaction metric (fast real ball)",
        "will_assess": "Speed-up and counter ability in hands battles at the net.",
        "will_recommend": "Reaction-volley and counter-attack drills."},
    "attack_conversion": {
        "unlocked_by": "new pop-up / put-away metric (real ball)",
        "will_assess": "How often you attack pop-ups and finish points.",
        "will_recommend": "Put-away and overhead finishing drills."},
    "reset_under_pressure": {
        "unlocked_by": "new reset-quality metric (real ball + transition position)",
        "will_assess": "Quality of resets from the transition zone under attack.",
        "will_recommend": "Pressure-reset drills from mid-court."},
    "defense_scrambling": {
        "unlocked_by": "new defensive metric (real ball + movement)",
        "will_assess": "Defensive coverage and re-resetting when out of position.",
        "will_recommend": "Defensive lob/scramble recovery drills."},
    "partner_stacking_poaching": {
        "unlocked_by": "new team-formation metric (positions; partly real now)",
        "will_assess": "Stacking, poaching, and switching coordination.",
        "will_recommend": "Stacking and poach-timing drills with your partner."},
    "footwork_split_step": {
        "unlocked_by": "pose-technique stage (Tier-C; mostly real pose data)",
        "will_assess": "Split-step timing and ready-position recovery.",
        "will_recommend": "Split-step-on-contact timing drills."},
    "shot_selection_iq": {
        "unlocked_by": "new strategic metric (shot selection vs situation)",
        "will_assess": "Whether you pick the right shot for the situation.",
        "will_recommend": "Decision-pattern and shot-selection drills."},
}


def _pct(v: Optional[float]) -> Optional[str]:
    return f"{v * 100:.0f}%" if isinstance(v, (int, float)) else None


def _verdict(v: Optional[float], lo: float, hi: float,
             low: str, mid: str, high: str) -> Optional[str]:
    """Plain good/bad phrase for a value's qualitative band (<lo / [lo,hi) / >=hi).
    None when the value is missing (caller supplies a generic fallback)."""
    if not isinstance(v, (int, float)):
        return None
    return low if v < lo else (high if v >= hi else mid)


# --- Per-dimension finding + drill selection --------------------------------

def finding_and_drills(dim: str, dr: dict) -> Tuple[str, List[dict]]:
    """Build a data-grounded finding sentence + 1-3 drill keys for a dimension
    from its rating.json driver_metrics. Falls back to generic phrasing when a
    driver value is missing (never fabricates a number)."""
    keys: List[str] = []
    if dim == "net_play":
        k = dr.get("user_kitchen_time_frac")
        t = dr.get("user_transition_time_frac")
        b = dr.get("both_at_kitchen_frac")
        kv = _verdict(k, 0.25, 0.45,
                      "so you're not getting up to the kitchen line often enough",
                      "so you get there but don't hold it for the whole point",
                      "so you get up there and hold it well")
        s = (f"You're at the kitchen line about {_pct(k)} of each rally, {kv}"
             if kv else "You're spending very little time up at the kitchen line")
        bv = _verdict(b, 0.25, 0.50,
                      "you and your partner are rarely at the line together at once",
                      "you and your partner get to the line together only part of "
                      "the time",
                      "and you get to the line together as a team well")
        if bv:
            s += f"; {bv} ({_pct(b)} of the rally)"
        finding = s + "."
        keys.append("get_to_line")
        if (b or 0) < 0.30:
            keys.append("move_as_unit")
        if (t or 0) > 0.25:
            keys.append("transition_resets")
    elif dim == "movement":
        cov = _pct(dr.get("court_coverage_frac"))
        dpm = dr.get("distance_ft_per_min")
        obs = []
        if cov:
            obs.append(f"you range across about {cov} of your side of the court")
        if isinstance(dpm, (int, float)):
            obs.append(f"and move roughly {int(round(dpm))} ft per minute of play")
        if obs:
            finding = ("During points " + " ".join(obs)
                       + ". On its own that isn't good or bad (strong players often "
                       "move less but get to better spots), so the lever here is "
                       "footwork: split-stepping as your opponent hits and recovering "
                       "to a ready position between shots.")
        else:
            finding = ("The lever for your movement is footwork — split-stepping as "
                       "your opponent hits and recovering to a ready position between "
                       "shots — rather than simply covering more ground.")
        keys.append("split_step_recover")
        if (dr.get("court_coverage_frac") or 0) < 0.5:
            keys.append("coverage_ladder")
    elif dim == "error_control":
        epr = dr.get("errors_per_rally")
        finding = ((f"You're giving away roughly {epr:.1f} point(s) per rally on "
                    "mistakes — the single biggest lever at your level, since "
                    "higher-rated players simply miss less. (We can't split forced "
                    "vs unforced errors yet; that needs the real-ball upgrade.)")
                   if isinstance(epr, (int, float))
                   else ("Cutting unforced mistakes is the quickest way up — "
                         "higher-rated players simply miss less."))
        keys += ["coop_dink_count", "reset_under_pressure"]
    elif dim == "shot_skill":
        drop = dr.get("third_shot_drop_rate")
        var = dr.get("shot_variety")
        dv = _verdict(drop, 0.35, 0.65,
                      "so you drive far more third shots than you drop them",
                      "so it's roughly a coin-flip rather than a drop you rely on",
                      "so you're favouring the drop, which is what higher levels do")
        finding = ((f"On the third shot you drop the ball about {_pct(drop)} of the "
                    f"time, {dv}.")
                   if dv else
                   ("Your third-shot and shot selection are still developing — a "
                    "drop you can trust is the skill that moves 3.5 players to 4.0."))
        if (drop or 0) < 0.4:
            keys.append("third_shot_drop_reps")
        if (var or 0) < 4:
            keys.append("shot_variety_ladder")
        keys.append("soft_game_targets")
    elif dim == "serve":
        sf = dr.get("serve_fault_rate")
        sv = _verdict(sf, 0.05, 0.15,
                      "so your serve is reliable",
                      "so it mostly goes in but faults more than it should",
                      "so you're faulting too many serves and giving away free points")
        finding = (f"About {_pct(sf)} of your serves are faults, {sv}."
                   if sv else
                   "Getting a consistent, deep serve in play starts the point on "
                   "your terms.")
        if (sf or 0) > 0.1:
            keys.append("deep_serve_targets")
        keys.append("pre_serve_routine")
    elif dim == "rally_consistency":
        ml = dr.get("mean_rally_length")
        rv = _verdict(ml, 4.0, 8.0,
                      "on the short side, so points end quickly",
                      "a reasonable length but short of the sustained exchanges the "
                      "next level rallies through",
                      "already the sustained length the next level rallies through")
        finding = (f"Your rallies last about {ml:.0f} shots on average, which is {rv}."
                   if rv else
                   "Your rallies are ending quickly; the next level keeps the ball "
                   "in play longer, especially in the dinking exchanges at the net.")
        keys.append("sustained_rally_game")
        if (dr.get("volley_rate") or 0) < 0.15:
            keys.append("volley_exchanges")
    else:
        finding = "This area is below your target level."
    # de-dup preserve order, cap at 3, guarantee >= 1
    seen, ordered = set(), []
    for k in keys:
        if k not in seen and k in DRILLS:
            seen.add(k)
            ordered.append(k)
    ordered = ordered[:3] or ["split_step_recover"]
    return finding, [DRILLS[k] for k in ordered]


# --- Banding helpers ---------------------------------------------------------

def band_float(band: str) -> float:
    try:
        return float(band)
    except (TypeError, ValueError):
        return 3.0


def next_half_step(current_band: str) -> float:
    return min(TARGET_CAP, band_float(current_band) + 0.5)


# --- Core plan (pure function) ----------------------------------------------

def conf_label(c: float) -> str:
    """Map a real per-dimension confidence to an honest coaching label."""
    return "high" if c >= 0.6 else ("moderate" if c >= 0.3 else "low")


def compute_plan(rating: dict, metrics: Optional[dict],
                 max_focus: int = MAX_FOCUS_AREAS,
                 conf_floor: float = CONFIDENCE_WEIGHT_FLOOR,
                 target_override: Optional[float] = None
                 ) -> dict:
    """Pure — builds the plan body from rating(.json) (+ optional metrics).
    Returns a dict with current/target/focus_areas/strengths/
    developing_capability/reliability."""
    rt = rating.get("rating", {}) or {}
    current_band = rt.get("band", "3.0")
    target_level = (target_override if target_override is not None
                    else next_half_step(current_band))
    dims = rating.get("dimensions", []) or []
    ball_source = rating.get("ball_source", "real")

    focus, strengths, unmeasured = [], [], []
    op_hits: Dict[str, dict] = {}   # operator category -> {limiters, affects}
    for d in dims:
        name = d.get("name")
        sub = d.get("subscore_level")
        weight = d.get("weight", 0.0)
        dconf = d.get("confidence", 0.0)
        data_source = d.get("data_source", "synthetic")
        limited_by = d.get("limited_by", "measurement")
        drivers = d.get("driver_metrics", {}) or {}
        if sub is None:
            continue
        # Operator note collection (SEPARATE from player coaching): a limiter
        # "bites" only on REAL-data dimensions whose confidence is materially low.
        # Synthetic-ball dims are excluded — their gated-low confidence reflects
        # the placeholder ball (already warned), not an operator-fixable limiter.
        if data_source == "real" and dconf < OPERATOR_CONF_FLOOR:
            cat = LIMITER_CATEGORY.get(limited_by, "capture_quality")
            hit = op_hits.setdefault(cat, {"limiters": set(), "affects": []})
            hit["limiters"].add(limited_by)
            hit["affects"].append(name)
        # A REAL dimension at near-zero confidence is a DATA GAP, not a signal —
        # e.g. serve (0 serves detected) or error_control (errors undetectable:
        # end_reasons all unknown). Presenting it as a weakness to fix OR a
        # strength to celebrate would coach off missing data, so route it to
        # developing_capability with the reason and skip focus/strength.
        if data_source == "real" and dconf < ASSESS_CONF_FLOOR:
            unmeasured.append({
                "dimension": name,
                "confidence": round(dconf, 3),
                "limited_by": limited_by,
                "reason": UNMEASURED_REASON.get(
                    name, f"{name} not reliably measured yet (limited_by {limited_by})."),
            })
            continue
        if sub >= target_level:
            strengths.append({
                "dimension": name,
                "current_subscore": sub,
                "data_source": data_source,
                "confidence": ("provisional" if data_source == "synthetic"
                               else conf_label(dconf)),
                "note": ("At/above the target"
                         + (" (provisional — synthetic)."
                            if data_source == "synthetic" else ".")),
            })
            continue
        gap = max(0.0, target_level - sub)
        conf_term = conf_floor + (1.0 - conf_floor) * dconf
        priority_score = round(gap * weight * conf_term, 4)
        provisional = (data_source == "synthetic")
        finding, drills = finding_and_drills(name, drivers)
        focus.append({
            "dimension": name,
            "data_source": data_source,
            "confidence": "provisional" if provisional else conf_label(dconf),
            "current_subscore": sub,
            "gap_to_target": round(gap, 3),
            "priority_score": priority_score,
            "finding": finding,
            "why_it_matters": WHY.get(name, ""),
            "drills": drills,
            "provisional_note": ("Derived from the synthetic ball; revisit when "
                                 "real ball detection (v4) lands."
                                 if provisional else None),
        })

    focus.sort(key=lambda f: f["priority_score"], reverse=True)
    focus = focus[:max_focus]
    for i, f in enumerate(focus):
        f["priority"] = i + 1
    # reorder keys so priority leads
    focus = [{"priority": f.pop("priority"), **f} for f in focus]

    # developing capability from rating.json skill_coverage
    sc = rating.get("skill_coverage", {}) or {}

    def descriptor_list(bucket: str) -> List[dict]:
        out = []
        for skill in sc.get(bucket, []) or []:
            desc = DEVELOPING.get(skill, {
                "unlocked_by": "future metric/stage",
                "will_assess": "(to be defined)",
                "will_recommend": "(to be defined)",
            })
            out.append({"skill": skill, **desc})
        return out

    developing = {
        "_comment": ("Skills not yet fully measured. v1 emits NO recommendations "
                     "for these; they scaffold in once their data source lands, "
                     "giving the plan full capability post Stage 4/4.5 + the "
                     "listed new stages."),
        "proxy_or_pending": descriptor_list("proxy_or_pending"),
        "not_captured_yet": descriptor_list("not_captured_yet"),
        "not_assessable_now": unmeasured,   # real dims at ~0 confidence (data gaps)
        "out_of_scope": list(sc.get("out_of_scope", []) or []),
    }

    n_real = sum(1 for f in focus if f["data_source"] == "real")
    n_prov = sum(1 for f in focus if f["data_source"] == "synthetic")
    reliability = {
        "synthetic_ball": ball_source == "synthetic",
        "n_focus_real": n_real,
        "n_focus_provisional": n_prov,
        "note": (f"{n_prov} of {len(focus)} focus areas are provisional "
                 f"(synthetic-ball-derived). Real-data focus areas "
                 f"(positioning/movement) are trustworthy now; the rest firm up "
                 f"at ball v4." if focus else
                 "No focus areas: user is at/above target on every measured "
                 "dimension (or input degraded)."),
    }

    target = {
        "band": f"{target_level:.1f}",
        "level": target_level,
        "rationale": (f"Next USAPA half-step. Closing the focus-area gaps below "
                      f"moves the user toward {target_level:.1f}."
                      if band_float(current_band) < TARGET_CAP else
                      "Already at the top band; maintain and refine."),
    }

    # Operator considerations (analysis reliability) — built last, kept separate
    # from and lower-priority than the player coaching above. Empty `items` when
    # no real-data limiter bites (the UI surfaces the section only when non-empty).
    operator_items = []
    for cat in ("more_data", "capture_quality"):   # stable order
        if cat in op_hits:
            h = op_hits[cat]
            operator_items.append({
                "category": cat,
                "limiters": sorted(h["limiters"]),
                "affects": h["affects"],
                "action": OPERATOR_ACTION[cat],
            })
    operator_considerations = {
        "_comment": ("Analysis-reliability notes for the OPERATOR (who records / "
                     "configures the capture) — a possibly-different audience than "
                     "the player coaching above, and lower priority. Surfaced only "
                     "when a real-data limiter materially reduces confidence; "
                     "`items` is empty otherwise (UI hides the section)."),
        "items": operator_items,
    }

    return {
        "current": {"estimate": rt.get("estimate"), "band": current_band,
                    "confidence": rt.get("confidence")},
        "target": target,
        "focus_areas": focus,
        "strengths": strengths,
        "developing_capability": developing,
        "reliability": reliability,
        "operator_considerations": operator_considerations,
    }


# --- Main pipeline -----------------------------------------------------------

def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    rating_path = folder / "rating.json"
    metrics_path = folder / "metrics.json"
    out_path = folder / "improvement_plan.json"

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.",
             FileExistsError)

    rating = load_json(rating_path)
    if rating.get("schema_version") != 1:
        fail(f"rating.json schema_version={rating.get('schema_version')} "
             f"unexpected (expects 1)", ValueError)

    metrics = None
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
            fail(f"metrics.json schema_version={metrics.get('schema_version')} "
                 f"unexpected (expects {METRICS_SCHEMA_VERSION}; run Stage 8 "
                 f"v0.2.0+)", ValueError)

    ball_source = rating.get("ball_source") or "real"
    ball_is_synth = (ball_source == "synthetic")
    if ball_is_synth:
        log.warning("ball_source is SYNTHETIC: provisional focus areas are "
                    "placeholder-derived (scaffold until ball v4).")

    target_override = band_float(args.target_band) if args.target_band else None
    body = compute_plan(rating, metrics, args.max_focus_areas,
                        CONFIDENCE_WEIGHT_FLOOR, target_override)
    if target_override is not None:
        body["target"]["rationale"] = "Operator-specified target band."

    warnings: List[str] = []
    if ball_is_synth:
        warnings.append("ball_source is 'synthetic': provisional focus areas are "
                        "derived from PLACEHOLDER ball data — treat as a scaffold "
                        "until ball detection v4.")
    warnings.append("Rating + plan thresholds are UNCALIBRATED heuristics (no "
                    "rated-footage corpus); see KNOWN_ISSUES.md.")
    if body["focus_areas"] and body["focus_areas"][0]["confidence"] == "provisional":
        warnings.append("The #1 focus area is provisional (synthetic-ball-derived); "
                        "the top real-data priority may be a lower-listed item.")
    if not body["focus_areas"]:
        warnings.append("No focus areas below target — plan lists strengths + "
                        "developing capability only.")

    log.info(f"plan: target band {body['target']['band']}, "
             f"{len(body['focus_areas'])} focus areas "
             f"({body['reliability']['n_focus_real']} real, "
             f"{body['reliability']['n_focus_provisional']} provisional)")

    out = {
        "schema_version": SCHEMA_VERSION,
        "source_rating": str(rating_path),
        "source_metrics": str(metrics_path) if metrics is not None else None,
        "ball_source": ball_source,
        "rated_role": "user",
        **body,
        "warnings": warnings,
        "params": {"max_focus_areas": args.max_focus_areas,
                   "confidence_weight_floor": CONFIDENCE_WEIGHT_FLOOR},
        "stage_version": STAGE_VERSION,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    log.info(f"wrote {out_path}")
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 10 — plan improvement")
    p.add_argument("folder", type=Path,
                   help="per-video folder with rating.json (+ metrics.json)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-focus-areas", type=int, default=MAX_FOCUS_AREAS,
                   dest="max_focus_areas")
    p.add_argument("--target-band", default=None, dest="target_band",
                   help="override the auto next-half-step target (e.g. 4.0)")
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

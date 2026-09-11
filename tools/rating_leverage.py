"""What is each measurement actually WORTH to the rating?

The report shows a number for every driver metric. A player reading it reasonably asks
"so which of these should I work on?" -- and the honest answer is not the one with the
worst-looking percentage. It is the one where moving the needle moves the RATING, which
depends on the scoring function's slope, the category's weight, the category's confidence,
and where the player already sits on that scale.

This measures it rather than guessing: nudge ONE driver by a fixed step, re-run the REAL
scorer from stages/rate/rate.py, and read how far the overall estimate moves. Nothing is
re-implemented here -- an earlier attempt to replay the scoring by hand produced numbers
that agreed by coincidence, so this calls the shipped functions.

Four things it surfaces that the raw numbers hide:

  at the ceiling      the driver is maxed (serves in play at 100%); no gain is possible
  below the floor     the driver is under where its scale STARTS, so a small gain reads
                      as zero -- but crossing the threshold is worth a lot (returns in
                      play at 57% against a scale that begins at 70%: +0.06 at 100%)
  too few to judge    a sample floor gates the driver out entirely (third-shot drop rate
                      needs 4 typed shots)
  not scored at all   the scorer computes it and deliberately does not use it (court
                      coverage, ready position, unforced errors)

The last is NOT a judgement that the thing does not matter. Unforced errors are one of
the clearest markers between 3.0, 3.5 and 5.0 in the USAPA ladder; they are excluded
because rally-end reason is right 8 times in 23, and the dominant confusion (a ball into
the net read as a ball nobody returned) is exactly the winner/error flip the metric needs.
See docs/REPORT_REDESIGN.md.

Usage:
    python -m tools.rating_leverage data/_collections/david2
    python -m tools.rating_leverage data/_collections/david2 --json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from stages.rate import rate as R

# (category, label, which dict, path into it, step description, step function)
#
# The step is TEN POINTS on the driver's own scale, because that is a unit a player can
# picture -- "ten percent more of your rallies spent at the kitchen line". Counts step by
# a comparable amount. Keep the unit uniform: a column mixing units is unreadable.
PROBES: List[Tuple[str, str, str, List[str], str, Callable]] = [
    ("strategy", "Time at the kitchen line", "u",
     ["position", "zone_time_frac", "kitchen"], "+10pp", lambda v: v + 0.10),
    ("strategy", "You and your partner up together", "t",
     ["both_at_kitchen_frac"], "+10pp", lambda v: v + 0.10),
    ("strategy", "Time stuck in mid-court", "u",
     ["position", "zone_time_frac", "transition"], "-10pp", lambda v: max(0.0, v - 0.10)),
    ("strategy", "Court covered", "u",
     ["position", "movement", "distance_ft_per_min"], "+20%", lambda v: v * 1.2),
    ("strategy", "Ready position (paddle up)", "u",
     ["ready_position", "trend_ok"], "-> good", lambda v: True),
    ("strategy", "Unforced errors", "u",
     ["errors_committed"], "halved", lambda v: (v / 2) if v else v),

    ("third_shot", "Getting to the kitchen after the 3rd", "u",
     ["transition", "arrived_frac"], "+10pp", lambda v: v + 0.10),
    ("third_shot", "Third-shot drop rate", "u",
     ["third_shot", "drop_rate"], "+10pp", lambda v: (v or 0.0) + 0.10),

    ("dink", "Pop-ups given up", "u", ["popup", "popup_frac"], "-10pp",
     lambda v: max(0.0, v - 0.10)),
    ("dink", "Share of your shots that are dinks", "u",
     ["shot_mix", "by_shot_type", "dink"], "+10 shots", lambda v: v + 10),
    ("dink", "Average rally length", "m", ["rally_length_shots", "mean"], "+2 shots",
     lambda v: v + 2),
    ("dink", "Dinks landing in the kitchen", "u",
     ["dink_control", "in_kitchen_frac"], "+10pp", lambda v: min(1.0, v + 0.10)),
    ("dink", "Knee bend on dinks", "u",
     ["shot_mix", "technique", "knee_bend_dink", "pct_good"], "+10pp",
     lambda v: min(1.0, v + 0.10)),

    ("volley", "Share of your shots that are volleys", "u",
     ["shot_mix", "volley_rate"], "+10pp", lambda v: min(1.0, v + 0.10)),

    ("serve_return", "Serves landing in", "u", ["serve", "in_play", "in_frac"], "+10pp",
     lambda v: min(1.0, v + 0.10)),
    ("serve_return", "Returns landing in", "u", ["return_in_play", "in_frac"], "+10pp",
     lambda v: min(1.0, v + 0.10)),
    ("serve_return", "Serves landing deep", "u", ["serve", "depth", "deep_frac"], "+10pp",
     lambda v: min(1.0, v + 0.10)),
    ("serve_return", "Returns landing deep", "u", ["return_depth", "deep_frac"], "+10pp",
     lambda v: min(1.0, v + 0.10)),

    ("forehand", "Contact point in front of the hip", "u",
     ["shot_mix", "technique", "by_stroke_side", "forehand", "mean"], "+0.5",
     lambda v: v + 0.5),
    ("forehand", "Knee bend on drives", "u",
     ["shot_mix", "technique", "knee_bend_drive_by_side", "forehand", "pct_good"],
     "+10pp", lambda v: min(1.0, v + 0.10)),

    ("backhand", "Contact point in front of the hip", "u",
     ["shot_mix", "technique", "by_stroke_side", "backhand", "mean"], "+0.5",
     lambda v: v + 0.5),
    ("backhand", "Knee bend on drives", "u",
     ["shot_mix", "technique", "knee_bend_drive_by_side", "backhand", "pct_good"],
     "+10pp", lambda v: min(1.0, v + 0.10)),
]

# How far to push a driver when the 10-point step reads zero, to tell "at the ceiling"
# from "below the floor". Both read +0.000 at one step; only one of them is a dead end.
FULL_PROBE = [0.25, 0.50, 1.00]

SCORERS = {
    "strategy": lambda u, m, t: R.score_strategy(u, t, m["n_rallies"])[0],
    "third_shot": lambda u, m, t: R.score_third_shot(u, m)[0],
    "dink": lambda u, m, t: R.score_dink(u, m)[0],
    "volley": lambda u, m, t: R.score_volley(u)[0],
    "serve_return": lambda u, m, t: R.score_serve_return(u)[0],
    "forehand": lambda u, m, t: R.score_forehand(u)[0],
    "backhand": lambda u, m, t: R.score_backhand(u)[0],
}


def _views(metrics: dict) -> Tuple[dict, dict, dict]:
    """Unwrapped (user, match, team_near) -- a DEEP COPY every time.

    `_unwrap_user` returns views that share structure with the metrics dict, so probing
    without copying mutates the source and every later probe compounds the last one. That
    bug produced a first leverage table where three unrelated drivers all read +0.418.
    """
    m = copy.deepcopy(metrics)
    players = m.get("players", {}) or {}
    return (R._unwrap_user(players.get("user", {}) or {}),
            R._unwrap_match(m.get("match", {}) or {}),
            R._v((m.get("team", {}) or {}).get("near", {}) or {}) or {})


def _poke(target: dict, path: List[str], fn: Callable) -> bool:
    cur = target
    try:
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = fn(cur[path[-1]])
        return True
    except (KeyError, TypeError, IndexError):
        return False


def leverage(metrics: dict) -> dict:
    """{'baseline': {...}, 'rows': [...]} -- every driver's worth, measured."""
    u0, m0, t0 = _views(metrics)
    base = {c: round(f(u0, m0, t0), 4) for c, f in SCORERS.items()}
    rows = []
    for cat, label, which, path, step, fn in PROBES:
        u, m, t = _views(metrics)
        if not _poke({"u": u, "m": m, "t": t}[which], path, fn):
            rows.append({"category": cat, "driver": label, "step": step,
                         "present": False, "d_subscore": 0.0, "d_rating": 0.0,
                         "verdict": "absent"})
            continue
        d = round(round(SCORERS[cat](u, m, t), 4) - base[cat], 4)
        row = {"category": cat, "driver": label, "step": step, "present": True,
               "d_subscore": d, "d_rating": round(d * R.WEIGHTS[cat], 4)}
        if d > 0:
            row["verdict"] = "scores"
        else:
            # Zero at one step means one of three very different things. Push harder to
            # find out which -- a driver below its scale's floor looks identical to one
            # at the ceiling until you step far enough to cross the threshold.
            best = 0.0
            for big in FULL_PROBE:
                u2, m2, t2 = _views(metrics)
                if not _poke({"u": u2, "m": m2, "t": t2}[which], path,
                             lambda v: min(1.0, v + big) if isinstance(v, float) and v <= 1.0
                             else fn(v)):
                    break
                best = max(best, round(round(SCORERS[cat](u2, m2, t2), 4) - base[cat], 4))
            row["d_subscore_max"] = best
            row["d_rating_max"] = round(best * R.WEIGHTS[cat], 4)
            row["verdict"] = "below_floor" if best > 0 else "inert"
        rows.append(row)
    rows.sort(key=lambda r: (-r["d_rating"], -r.get("d_rating_max", 0.0)))
    return {"baseline": base, "weights": dict(R.WEIGHTS), "rows": rows}


def category_shares(rating: dict) -> List[dict]:
    """Intended weight vs the share a category ACTUALLY carries.

    The estimate is confidence-weighted (weight x confidence, renormalised), so the static
    weight is not what a category contributes. On David2 that gap is large: strategy is
    meant to carry 20% and carries 39%, because positioning is the one thing measured near
    completely. A report that prints only the static weight overstates how balanced the
    rating is.
    """
    dims = rating.get("dimensions", []) or []
    eff = [(d["name"], d.get("weight", 0.0), d.get("confidence", 0.0),
            d.get("subscore_level"), d.get("weight", 0.0) * d.get("confidence", 0.0))
           for d in dims]
    total = sum(e[4] for e in eff) or 1.0
    return [{"category": n, "weight": w, "confidence": c, "subscore": s,
             "effective_weight": round(e, 4), "actual_share": round(e / total, 4)}
            for n, w, c, s, e in sorted(eff, key=lambda x: -x[4])]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, help="session or collection folder")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    a = ap.parse_args(argv)

    mp, rp = a.folder / "metrics.json", a.folder / "rating.json"
    if not mp.exists():
        print(f"metrics.json not found in {a.folder}", file=sys.stderr)
        return 1
    metrics = json.loads(mp.read_text(encoding="utf-8"))
    rating = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    out = leverage(metrics)
    out["category_shares"] = category_shares(rating) if rating else []
    out["estimate"] = (rating.get("rating", {}) or {}).get("estimate")

    if a.json:
        print(json.dumps(out, indent=1))
        return 0

    if out["category_shares"]:
        print(f"ESTIMATE {out['estimate']}\n")
        print(f"{'category':14s} {'meant':>6s} {'conf':>6s} {'carrying':>9s}  subscore")
        print("-" * 52)
        for c in out["category_shares"]:
            print(f"{c['category']:14s} {c['weight']:6.0%} {c['confidence']:6.2f} "
                  f"{c['actual_share']:9.1%}  {c['subscore']}")
        print()

    NOTE = {"inert": "not scored / at the ceiling", "absent": "driver not present"}
    for cat in SCORERS:
        rs = [r for r in out["rows"] if r["category"] == cat]
        if not rs:
            continue
        print(f"--- {cat}  (weight {R.WEIGHTS[cat]:.0%}, subscore {out['baseline'][cat]})")
        for r in rs:
            if r["verdict"] == "scores":
                tail = f"rating {r['d_rating']:+.4f}"
            elif r["verdict"] == "below_floor":
                tail = (f"rating  0.0000   <- BELOW THE FLOOR: worth up to "
                        f"{r['d_rating_max']:+.4f} once it crosses")
            else:
                tail = f"rating  0.0000   <- {NOTE.get(r['verdict'], r['verdict'])}"
            print(f"    {r['driver']:38s} {r['step']:>10s}  {tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

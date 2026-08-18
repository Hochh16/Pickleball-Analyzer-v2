"""Build the consumer report — per-video JSONs -> a self-contained report.html.

Reads a per-video folder's Stage 8-11 outputs (rating.json, improvement_plan.json,
metrics.json, timeline.json, bounces.json, heatmap PNGs, annotated.mp4) and emits a
single self-contained `report.html` (images inline as data URIs) that a player can
open in any browser. Aligned to the 7 official USAPA categories; honest about
coverage (measured / partial / not-yet-measured) rather than overclaiming.

Usage:
    python -m tools.build_report data/pb_2min --force
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# --- Static reference content (USAPA-anchored) -------------------------------

CATEGORY_ORDER = ["strategy", "third_shot", "dink", "volley", "serve_return",
                  "forehand", "backhand"]
CATEGORY_LABEL = {
    "strategy": "Strategy", "third_shot": "Third Shot", "dink": "Dink",
    "volley": "Volley", "serve_return": "Serve / Return",
    "forehand": "Forehand", "backhand": "Backhand",
}

# What USA Pickleball rates in each category, broken into elements with our current
# coverage: "live" (measured now), "partial" (early/low-confidence), "planned".
CATEGORY_ELEMENTS = {
    "strategy": [("Kitchen-line positioning", "live"), ("Moving as a team", "live"),
                 ("Court coverage & movement", "live"),
                 ("Ready position (paddle up)", "live"), ("Stacking", "planned"),
                 ("Targeting weakness", "planned"), ("Resets under pressure", "planned"),
                 ("Unforced errors", "planned")],
    "third_shot": [("How often you play the 3rd shot", "live"),
                   ("Drop-vs-drive choice", "partial"),
                   ("Drop landing depth", "planned"), ("Transition success", "planned")],
    "dink": [("How much you dink", "partial"), ("Dink-rally length", "partial"),
             ("Knee bend (staying low)", "live"),
             ("Pop-up rate", "planned"), ("Height & depth control", "planned")],
    "volley": [("How often you volley at the net", "partial"),
               ("Block / reset", "planned"), ("Put-aways", "planned"),
               ("Speed-ups & counters", "planned")],
    "serve_return": [("Serve / return count", "live"),
                     ("In-play rate & faults", "partial"),
                     ("Depth", "planned"), ("Pace & spin", "planned")],
    "forehand": [("How many forehands you hit", "live"),
                 ("Contact point (in front of hip)", "live"),
                 ("Knee bend on drives", "live"),
                 ("Pace", "planned"), ("Placement & depth", "planned")],
    "backhand": [("How many backhands you hit", "live"),
                 ("Contact point (in front of hip)", "live"),
                 ("Knee bend on drives", "live"),
                 ("Pace & depth", "planned")],
}

# Driver-metric key -> (plain-English label, format). Keys not listed are hidden
# from the player report (internal flags / not-yet-available inputs).
METRIC_DISPLAY = {
    "user_kitchen_time_frac": ("Time at the kitchen line", "pct_rally"),
    "both_at_kitchen_frac": ("You and your partner at the kitchen together", "pct_rally"),
    "user_transition_time_frac": ("Time caught in mid-court (transition)", "pct_rally"),
    "distance_ft_per_min": ("Court covered during play", "ftmin"),
    "third_shot_drop_rate": ("Third shots played as a soft drop", "pct"),
    "dink_count": ("Dinks detected", "int"),
    "volley_rate": ("Your shots that were volleys", "pct"),
    "n_volley": ("Net volleys detected", "int"),
    "serve_fault_rate": ("Serves that faulted", "pct"),
    "n_serves": ("Serves detected", "int"),
    "n_returns": ("Returns of serve detected", "int"),
    "forehand_count": ("Forehands detected", "int"),
    "backhand_count": ("Backhands detected", "int"),
    "ready_position": ("Ready position (paddle up)", "ready"),
    "contact_front": ("Contact point", "contact"),
    "drive_knee_bend": ("Knee bend on drives", "knee"),
    "dink_knee_bend": ("Knee bend on dinks", "knee"),
    "mean_rally_length": ("Average rally length", "shots"),
}

# Condensed USAPA level criteria across all 7 categories (my synthesis of the
# published definitions; uncalibrated — see footnote).
USAPA_LADDER = [
    ("2.0", "New to the game. Struggles to serve in play or direct shots; rallies rarely sustain."),
    ("2.5", "Sustains short rallies and serves in. Reaches the kitchen but late; dinks pop up; drives most third shots."),
    ("3.0", "Keeps the ball in play. Knows the third-shot drop but it's inconsistent; sometimes chooses drop vs drive; dinks with some control; still frequent unforced errors."),
    ("3.5", "More consistent dinks; third-shot drops with a plan to get to the net; holds the kitchen line as a team; basic stacking; fewer unforced errors; developing volleys."),
    ("4.0", "Controlled, consistent dinks; reliable third-shot drops with a clean transition to the net; resets from mid-court; deeper serves and returns; reads attackable balls; directs both forehand and backhand."),
    ("4.5", "Absorbs pace with blocks and resets; disciplined dinks; speeds up at the right targets; dependable on both wings; sound shot selection."),
    ("5.0", "Selects drop / drive / hybrid correctly; resets under stress; precise speed-ups; very few unforced errors."),
    ("5.5+", "Tournament-level dominance (results-based)."),
]

COVERAGE = {
    "measured": ("Measured", "b-measured"),
    "partial": ("Partial", "b-partial"),
    "not_assessable": ("Not yet measured", "b-na"),
}
SYMBOL = {"live": "●", "partial": "◐", "planned": "○"}

# Seconds of run-up before a serve when jumping to a point. Landing exactly on the serve
# frame drops the viewer in mid-motion with no idea how the players were set.
JUMP_LEAD_S = 3.0


SHOT_LEAD_S = 2.0       # run-up when jumping to one shot rather than a whole rally
LONG_RALLY_S = 25.0     # beyond this a "rally" is usually several points run together


def user_shot_groups(classified: dict, rallies: list, track_roles: dict) -> dict:
    """{label: [t_sec, ...]} for the operator's OWN shots, by kind.

    Only the user's shots: the point of this list is reviewing your own play, and a
    mixed list would need a name against every row to be readable.

    "Third shots" is positional (the 3rd shot of a rally), not a shot type — that is what
    the third-shot category rates, and it is the one the drop-vs-drive choice belongs to.
    """
    tids = {int(t) for t, i in (track_roles.get("track_roles", {}) or {}).items()
            if i.get("role") == "user"}
    by_id = {int(s["shot_id"]): s for s in classified.get("shots", [])}
    mine = lambda s: s.get("track_id") is not None and int(s["track_id"]) in tids

    third = []
    for r in rallies:
        ids = [i for i in r.get("shot_ids", []) if i in by_id]
        if len(ids) >= 3 and mine(by_id[ids[2]]):
            third.append(float(by_id[ids[2]]["t_sec"]))

    def of_type(*types):
        return sorted(float(s["t_sec"]) for s in by_id.values()
                      if mine(s) and s.get("shot_type") in types)

    return {
        "Serves & returns": of_type("serve", "return"),
        "Third shots": sorted(third),
        "Dinks": of_type("dink"),
        "Drops": of_type("drop"),
        "Volleys": sorted(float(s["t_sec"]) for s in by_id.values()
                          if mine(s) and s.get("is_volley")),
    }


def clock(sec) -> str:
    """m:ss for the point index."""
    try:
        sec = max(0, int(float(sec)))
    except (TypeError, ValueError):
        return "0:00"
    return f"{sec // 60}:{sec % 60:02d}"


# --- Data helpers ------------------------------------------------------------

def load_json(folder: Path, name: str) -> Optional[dict]:
    p = folder / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def v(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def data_uri_png(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def band_of(estimate: float) -> str:
    e = max(1.0, min(5.0, estimate))
    return f"{round(e * 2.0) / 2.0:.1f}"


def fmt_metric(fmt: str, val) -> Optional[str]:
    if val is None:
        return None
    try:
        if fmt == "pct":
            return f"{val * 100:.0f}%"
        if fmt == "pct_rally":
            return f"{val * 100:.0f}% of each rally"
        if fmt == "ftmin":
            return f"{val:.0f} ft per minute of play"
        if fmt == "int":
            return f"{int(val)}"
        if fmt == "shots":
            return f"{val:.1f} shots"
        if fmt == "contact":
            # val = {n, n_in_front, pct_in_front, mean}. Report the count + share
            # hit in front of the body (the coachable technique read).
            if not isinstance(val, dict) or not val.get("n"):
                return None
            n, nf = int(val["n"]), int(val.get("n_in_front", 0))
            pct = int(round(val.get("pct_in_front", 0) * 100))
            return f"{nf} of {n} hit in front of your hip ({pct}%)"
        if fmt == "ready":
            # val = {n_frames, by_zone:{kitchen/transition/baseline:{median_height}},
            # trend_ok}. Height should be HIGHER at the net, dropping toward the back.
            # (Hand height, a proxy for the paddle -- we can't see the tip.)
            if not isinstance(val, dict) or not val.get("by_zone"):
                return None
            def _label(h):
                return ("up near chest" if h >= 0.35 else
                        "around the waist" if h >= 0.1 else "low (hip or below)")
            bz = val["by_zone"]
            parts = []
            for z, name in [("kitchen", "at the kitchen"), ("baseline", "at the baseline")]:
                if z in bz:
                    parts.append(f"{name}: hands {_label(bz[z]['median_height'])}")
            tail = ("" if val.get("trend_ok") is None else
                    " — good: lower as you move back" if val["trend_ok"] else
                    " — try carrying it higher at the net and lower at the back")
            return ("; ".join(parts) + tail) if parts else None
        if fmt == "knee":
            # val = {n, mean_bend_deg, n_good, pct_good}. Good = knee bend within the
            # operator's per-shot-type band (soft shots want a deeper bend).
            if not isinstance(val, dict) or not val.get("n"):
                return None
            n, ng = int(val["n"]), int(val.get("n_good", 0))
            pct = int(round(val.get("pct_good", 0) * 100))
            mb = int(val.get("mean_bend_deg", 0))
            return f"{ng} of {n} with the right knee bend ({pct}%; avg {mb}°)"
    except (TypeError, ValueError):
        return None
    return str(val)


# --- Ball-landing sequence diagram (drawn with OpenCV) -----------------------

def landing_diagram_uri(bounces: list, rally_windows: list,
                        user_hits: Optional[dict] = None) -> Optional[str]:
    """Top-down court showing WHERE the ball bounced during rallies, with a line from
    each of YOUR shots to where that ball landed.

    Still not a shot-by-shot sequence: bounces are not numbered and unlinked dots keep no
    implied order, because volleys never bounce and some bounces are missed. The lines are
    safe where the numbering was not, because each one joins a specific bounce to the
    specific shot the bounce stage already attributed to it (`between_shots`), rather than
    inferring order from time.

    user_hits: {bounce_id: (hit_x_ft, hit_y_ft)} for bounces produced by the user's shots.
    Volleys are absent by construction — a volley has no bounce to link to.

    Returns a PNG data URI."""
    import cv2
    def rally_of(f):
        for i, (a, b) in enumerate(rally_windows):
            if a <= f <= b:
                return i
        return None
    pts = []
    for b in bounces:
        xy = b.get("court_xy_ft")
        if not xy or xy[0] is None:
            continue
        ri = rally_of(int(b["frame"]))
        if ri is None:           # between-point / out-of-play bounce -> skip
            continue
        pts.append((xy, bool(b.get("is_in_court")), ri, b.get("bounce_id")))
    if not pts:
        return None
    W_FT, L_FT, KIT = 20.0, 44.0, 7.0
    sx = 20          # px per foot (x) -> 400 wide
    sy = 13          # px per foot (y) -> 572 long
    pad = 26
    W, L = int(W_FT * sx) + 2 * pad, int(L_FT * sy) + 2 * pad
    img = np.full((L, W, 3), 248, np.uint8)          # near-white ground
    court = (70, 120, 60)                            # BGR muted court green
    line = (120, 150, 130)
    def to_px(x, y):
        # near baseline (y=0) at BOTTOM, far (y=44) at top; clamp x for drawing
        px = int(pad + max(-2, min(W_FT + 2, x)) * sx)
        py = int(pad + (L_FT - max(-4, min(L_FT + 4, y))) * sy)
        return px, py
    # court fill + lines
    cv2.rectangle(img, to_px(0, L_FT), to_px(W_FT, 0), (232, 240, 233), -1)
    cv2.rectangle(img, to_px(0, L_FT), to_px(W_FT, 0), court, 2)
    for yy, w in [(L_FT / 2, 2), (L_FT / 2 - KIT, 1), (L_FT / 2 + KIT, 1)]:
        cv2.line(img, to_px(0, yy), to_px(W_FT, yy), line, w)
    cv2.line(img, to_px(W_FT / 2, 0), to_px(W_FT / 2, L_FT), line, 1)
    teal, red = (110, 118, 15), (60, 60, 210)
    mine = (150, 90, 40)          # BGR muted blue — the operator's own shots
    # Your shot -> where that ball landed. Drawn first so the dots sit on top.
    uh = user_hits or {}
    n_lines = 0
    for (xy, ok, ri, bid) in pts:
        hit = uh.get(bid)
        if not hit:
            continue
        cv2.line(img, to_px(hit[0], hit[1]), to_px(xy[0], xy[1]), mine, 2, cv2.LINE_AA)
        cv2.circle(img, to_px(hit[0], hit[1]), 4, mine, -1, cv2.LINE_AA)
        n_lines += 1
    for (xy, ok, ri, bid) in pts:
        p = to_px(xy[0], xy[1])
        cv2.circle(img, p, 8, teal if ok else red, -1, cv2.LINE_AA)
        cv2.circle(img, p, 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "net", (pad + 4, to_px(0, L_FT / 2)[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, line, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# --- HTML --------------------------------------------------------------------

CSS = """
:root{
  --ground:#f3f6f5; --card:#ffffff; --ink:#182421; --muted:#5f6d68; --line:#e2e8e5;
  --court:#0f766e; --court-deep:#0b5a54; --ball:#8a962f;
  --measured:#2f8f5b; --measured-bg:#e6f3ec;
  --partial:#a9772a; --partial-bg:#f7efdd;
  --na:#6f7a83; --na-bg:#eceff1;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#101614; --card:#182320; --ink:#e7ede9; --muted:#93a09a; --line:#26332e;
  --court:#3fb3a3; --court-deep:#57c3b3; --ball:#c3d05a;
  --measured:#5fc88e; --measured-bg:#12291f; --partial:#dcae54; --partial-bg:#2c2513;
  --na:#8b959c; --na-bg:#20282b; }}
:root[data-theme="dark"]{
  --ground:#101614; --card:#182320; --ink:#e7ede9; --muted:#93a09a; --line:#26332e;
  --court:#3fb3a3; --court-deep:#57c3b3; --ball:#c3d05a;
  --measured:#5fc88e; --measured-bg:#12291f; --partial:#dcae54; --partial-bg:#2c2513;
  --na:#8b959c; --na-bg:#20282b; }
:root[data-theme="light"]{
  --ground:#f3f6f5; --card:#ffffff; --ink:#182421; --muted:#5f6d68; --line:#e2e8e5;
  --court:#0f766e; --court-deep:#0b5a54; --ball:#8a962f;
  --measured:#2f8f5b; --measured-bg:#e6f3ec; --partial:#a9772a; --partial-bg:#f7efdd;
  --na:#6f7a83; --na-bg:#eceff1; }

*{box-sizing:border-box;}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;}
.wrap{max-width:840px;margin:0 auto;padding:30px 20px 90px;}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--court);
  font-weight:600;margin:0 0 6px;}
h1{font-family:var(--serif);font-size:34px;line-height:1.1;margin:0;text-wrap:balance;font-weight:600;}
h2{font-family:var(--serif);font-size:23px;margin:44px 0 6px;font-weight:600;text-wrap:balance;}
.rule{height:2px;background:linear-gradient(90deg,var(--court),transparent);border:0;margin:6px 0 16px;}
h3{font-size:15px;margin:0 0 4px;}
p{margin:9px 0;} .muted{color:var(--muted);} .small{font-size:13px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:14px 0;}
.warn-card{border-left:5px solid #d98200;background:#fff8ec;}
.hero{display:flex;flex-wrap:wrap;gap:24px;align-items:center;
  background:linear-gradient(135deg,var(--card),var(--measured-bg));}
.score{font-family:var(--serif);font-size:72px;font-weight:600;line-height:.9;color:var(--court);
  font-variant-numeric:tabular-nums;}
.score sup{font-size:22px;color:var(--muted);font-weight:500;}
.hero-meta{flex:1;min-width:240px;}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:3px 9px;border-radius:20px;
  letter-spacing:.04em;text-transform:uppercase;white-space:nowrap;}
.b-measured{color:var(--measured);background:var(--measured-bg);}
.b-partial{color:var(--partial);background:var(--partial-bg);}
.b-na{color:var(--na);background:var(--na-bg);}
table{width:100%;border-collapse:collapse;font-size:14px;}
th,td{text-align:left;padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top;}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;}
tbody tr:last-child td{border-bottom:0;}
.num{font-variant-numeric:tabular-nums;}
.stats{display:flex;flex-wrap:wrap;gap:10px 28px;justify-content:space-between;}
.stat{text-align:center;flex:1;min-width:90px;}
.stat-n{font-family:var(--serif);font-size:28px;font-weight:600;color:var(--court);line-height:1;}
.stat-l{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;margin-top:4px;}
.lvl{font-family:var(--serif);font-size:19px;font-weight:600;color:var(--court);}
.bar{height:6px;border-radius:4px;background:var(--na-bg);overflow:hidden;margin-top:6px;max-width:120px;}
.bar>i{display:block;height:100%;background:var(--court);}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--muted);margin:10px 2px 0;}
.legend span{display:inline-flex;align-items:center;gap:6px;}
.sym{color:var(--court);font-size:14px;}
.metric{margin:3px 0;} .metric b{font-variant-numeric:tabular-nums;}
.el{display:inline-block;margin:2px 10px 2px 0;font-size:13px;white-space:nowrap;}
.el .planned{color:var(--muted);}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media (max-width:640px){.grid2{grid-template-columns:1fr;}}
.hm{text-align:center;} .hm img{max-width:100%;border-radius:10px;border:1px solid var(--line);}
.ramp{height:12px;border-radius:6px;margin:6px auto 4px;max-width:220px;
  background:linear-gradient(90deg,#000,#420a68,#932667,#dd513a,#f3a712,#fcffa4);}
.focus{border-left:4px solid var(--court);}
.drill{font-size:13.5px;margin:5px 0;padding-left:16px;position:relative;}
.drill::before{content:"▸";position:absolute;left:0;color:var(--court);}
.here{background:var(--measured-bg);}
.scrollx{overflow-x:auto;}
sup a{color:var(--court);text-decoration:none;font-size:11px;padding:0 1px;}
.foot{color:var(--muted);font-size:12.5px;margin-top:48px;border-top:1px solid var(--line);padding-top:14px;}
.foot li{margin:5px 0;}
a{color:var(--court-deep);} .vid video{width:100%;border-radius:10px;border:1px solid var(--line);}
.points{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px;margin:10px 0;}
.pt{display:flex;align-items:center;gap:7px;padding:7px 9px;border:1px solid var(--line);
background:#fff;border-radius:9px;cursor:pointer;font:inherit;text-align:left;}
.pt:hover{border-color:var(--court-deep);}
.pt-n{font-weight:700;min-width:1.4em;color:var(--court-deep);}
.pt-t{font-variant-numeric:tabular-nums;font-weight:600;}
.pt-d{font-size:11.5px;color:var(--muted);margin-left:auto;white-space:nowrap;}
.pt-long{border-color:#e0b23c;background:#fffaf0;}
.pt.pt-s{justify-content:center;padding:5px 8px;}
.shotrow{display:flex;gap:10px;align-items:flex-start;margin:8px 0;flex-wrap:wrap;}
.shotrow-l{min-width:130px;padding-top:6px;font-size:13px;}
.shotrow .points{flex:1;grid-template-columns:repeat(auto-fill,minmax(72px,1fr));margin:0;}
"""


def badge(cov: str) -> str:
    label, cls = COVERAGE.get(cov, ("—", "b-na"))
    return f'<span class="badge {cls}">{esc(label)}</span>'


def fn(n: int) -> str:
    return f'<sup><a href="#fn{n}">{n}</a></sup>'


def user_seed_basis(track_roles: dict) -> tuple:
    """How the "user" role was decided, as (basis, confidence).

    Stage 2.5 seeds the user from a CLICK at confidence 0.95, or geometrically from
    court.json's `user_starting_corner` at 0.5 when no click exists. The report is entirely
    per-player, so a wrong seed makes every number belong to the partner with nothing else
    looking wrong -- which is exactly what makes it worth a banner. Returns (None, None)
    when no user role is present at all.
    """
    for info in (track_roles.get("track_roles", {}) or {}).values():
        if (info or {}).get("role") == "user":
            return info.get("basis"), info.get("confidence")
    return None, None


def build_html(folder: Path) -> str:
    # Present only for a cumulative report: it is the collection's membership record,
    # and it carries the player's name (needed in the hero, far above the role labels).
    collection = load_json(folder, "collection.json")
    # Needed to tell the user's shots from everyone else's when linking bounces back to
    # the shot that produced them.
    track_roles = load_json(folder, "track_roles.json") or {}
    rating = load_json(folder, "rating.json") or {}
    plan = load_json(folder, "improvement_plan.json") or {}
    bounces_doc = load_json(folder, "bounces.json") or {}
    timeline = load_json(folder, "timeline.json") or {}
    classified = load_json(folder, "classified.json") or {}
    rallies_doc = load_json(folder, "rallies.json") or {}
    metrics = load_json(folder, "metrics.json") or {}
    # Match-level totals (all four players) for the count drivers, so each count row
    # can show BOTH the match total and the user's share ("22 in the match, 6 by
    # you") — operator directive: it puts the user's numbers in perspective.
    _mm = (metrics.get("match", {}) or {}).get("shot_mix", {}) or {}
    _bt = ((_mm.get("by_shot_type", {}) or {}).get("value", {}) or {})
    _bs = ((_mm.get("by_stroke_side", {}) or {}).get("value", {}) or {})
    match_counts = {
        "dink_count": _bt.get("dink"),
        "n_volley": ((_mm.get("volley", {}) or {}).get("value", {}) or {}).get("n_volley"),
        "n_serves": _bt.get("serve"),
        "n_returns": ((metrics.get("match", {}) or {}).get("returns", {}) or {}).get("value"),
        "forehand_count": _bs.get("forehand"),
        "backhand_count": _bs.get("backhand"),
    }

    rt = rating.get("rating", {}) or {}
    dims = {d["name"]: d for d in rating.get("dimensions", [])}
    rel = rating.get("reliability", {}) or {}
    not_assessable = {e["dimension"]: e for e in
                      (plan.get("developing_capability", {}) or {}).get("not_assessable_now", [])}

    def cov_of(c):
        # The coverage BADGE reflects MEASUREMENT (what we can show), which is the
        # rating's coverage_status. It is decoupled from the plan's COACHING gate
        # (`not_assessable`): a category can have a validated count we display as
        # 'partial' while still not being coached off a tiny sample. Only fall back
        # to the coaching gate when the rating has no status at all.
        return dims.get(c, {}).get(
            "coverage_status", "not_assessable" if c in not_assessable else "not_assessable")

    O = []
    A = O.append
    A('<div class="wrap">')

    # ---- Hero ----
    est = rt.get("estimate")
    # Whose report this is, stated up front. Without it, a library of reports is
    # indistinguishable one from another, and the operator has no way to tell which
    # cumulative report a new video belongs to.
    who = ""
    if collection and collection.get("name"):
        who = str(collection["name"])
    else:
        sess = load_json(folder, "session.json") or {}
        who = str(sess.get("player_name") or "")
    n_vids = len((collection or {}).get("members", []) or [])
    A('<p class="eyebrow">USA Pickleball–aligned skill report</p>')
    A(f'<h1>{esc(who)} — Player Report</h1>' if who else '<h1>Your Player Report</h1>')
    if collection:
        A(f'<p class="muted small">Cumulative report across {n_vids} '
          f'video{"" if n_vids == 1 else "s"}, combined as one longer match.</p>')
    # IDENTITY WARNING. Everything below is per-player, so if the wrong near-side player
    # was picked as "you", every number here is the partner's and nothing else in the
    # report would look wrong. Stage 2.5 seeds the user from a CLICK at confidence 0.95,
    # or from the starting corner at 0.5 when no click exists. The 0.5 case is a coin flip
    # nothing verifies, so say so where it cannot be missed rather than in a log line.
    _basis, _conf = user_seed_basis(track_roles)
    if _basis and _basis != "click":
        A('<div class="card warn-card">'
          '<p style="margin:0"><b>Check this is you.</b> Nobody marked which player to '
          'analyse, so we guessed from the starting side you chose during setup '
          f'(confidence {esc(_conf if _conf is not None else "?")} out of 1.0). If the '
          'guess is wrong, every number in this report belongs to your partner. Re-run '
          'setup and click on yourself in the frame to remove the guess.</p></div>')
    A('<div class="card hero">')
    A(f'<div><div class="score">{est if est is not None else "—"}</div>'
      f'<div class="muted small">USAPA band {esc(rt.get("band","—"))}</div></div>')
    # Score plus one sentence. The old hero also printed a likely RANGE, claimed the
    # rating "rests mostly on court strategy", and counted categories not yet measured —
    # all three are stale now that every category contributes real measurements, and the
    # count read "0 of 7 aren't measured yet", which says nothing.
    A('<div class="hero-meta">')
    A(f'<p style="margin-top:0;margin-bottom:0"><b>Your estimated rating is {est}</b>.'
      f'{fn(1)}</p>')
    A('</div></div>')

    # ---- Session at a glance ----
    rallies = rallies_doc.get("rallies", [])
    # Header counts use IN-RALLY shots only, matching the category counts (metrics
    # scopes to rallies). Otherwise the headline "Shots" (all detections incl.
    # between-point) disagrees with everything below it.
    _rsids = {int(i) for r in rallies for i in r.get("shot_ids", [])}
    _all_shots = classified.get("shots", [])
    shots = ([s for s in _all_shots if int(s["shot_id"]) in _rsids]
             if _rsids else _all_shots)
    n_volley = sum(1 for s in shots if s.get("is_volley"))
    # "Minutes analyzed" was blank whenever timeline.json was missing or carried no
    # duration — which is always true for a cumulative report, since a union has no single
    # video to render a timeline from. match_span_sec is what every other number here is
    # computed against, so take it first.
    dur = v((metrics.get("match") or {}).get("match_span_sec")) if metrics else None
    if not isinstance(dur, (int, float)) or dur <= 0:
        dur = timeline.get("duration_sec")
    mins = f"{dur/60:.1f}" if isinstance(dur, (int, float)) and dur > 0 else "—"
    stats = [("Minutes analyzed", mins), ("Rallies", len(rallies)),
             ("Shots", len(shots)), ("Volleys (hit in the air)", n_volley),
             ("Ball bounces", len(bounces_doc.get("bounces", [])))]
    A('<div class="card"><div class="stats">')
    for label, val in stats:
        ref = fn(5) if label == "Ball bounces" else ""
        A(f'<div class="stat"><div class="stat-n num">{esc(val)}</div>'
          f'<div class="stat-l">{esc(label)}{ref}</div></div>')
    A('</div></div>')

    # ---- 7-category table (level + numbers + what USAPA rates, in one) ----
    # Previously two tables listing the same seven categories, which made the reader
    # cross-reference to answer one question. The coverage column and its badges are gone
    # too: the "what USA Pickleball rates" column already shows which elements are
    # measured, so the badge restated it in vaguer words.
    A('<h2>Your 7 categories</h2><hr class="rule">')
    A('<p class="muted small">USA Pickleball rates players across these seven '
      'categories. Here\'s your level in each, the numbers behind it, and what the '
      'category covers.</p>')
    A('<div class="card scrollx"><table><thead><tr>'
      '<th>Category</th><th>Your level</th><th>Your numbers now</th>'
      '<th>What USA Pickleball rates</th></tr></thead><tbody>')
    for c in CATEGORY_ORDER:
        d = dims.get(c, {})
        sub = d.get("subscore_level")
        if cov_of(c) == "not_assessable" or not isinstance(sub, (int, float)):
            lvl = '<span class="muted">—</span>'
        else:
            barpct = int(max(0, min(100, ((sub - 1.0) / 4.5) * 100)))
            lvl = (f'<span class="lvl num">{band_of(sub)}</span>'
                   f'<div class="bar"><i style="width:{barpct}%"></i></div>')
        drivers = d.get("driver_metrics", {}) or {}
        nums = []
        for k, (label, fmt) in METRIC_DISPLAY.items():
            if k in drivers:
                s = fmt_metric(fmt, drivers[k])
                if s is None:
                    continue
                ref = fn(4) if k == "distance_ft_per_min" else ""
                # Count drivers: show the match total AND the user's share, so the
                # user's numbers sit in perspective (a 5-min clip has ~4 players).
                mt = match_counts.get(k)
                if k in match_counts and mt is not None:
                    nums.append(f'<div class="metric">{esc(label)}: '
                                f'<b>{esc(str(mt))}</b> in the match, '
                                f'<b>{esc(s)}</b> by you{ref}</div>')
                else:
                    nums.append(f'<div class="metric">{esc(label)}: '
                                f'<b>{esc(s)}</b>{ref}</div>')
        numhtml = "".join(nums) if nums else '<span class="muted small">—</span>'
        els = "".join(
            f'<span class="el"><span class="sym">{SYMBOL[st]}</span> '
            f'<span class="{ "planned" if st=="planned" else "" }">{esc(lbl)}</span></span>'
            for lbl, st in CATEGORY_ELEMENTS[c])
        A(f'<tr><td><b>{esc(CATEGORY_LABEL[c])}</b></td><td>{lvl}</td>'
          f'<td>{numhtml}</td><td class="small">{els}</td></tr>')
    A('</tbody></table></div>')
    A('<div class="legend"><span><span class="sym">●</span> measured now</span>'
      '<span><span class="sym">◐</span> partial / early signal</span>'
      '<span><span class="sym">○</span> coming soon</span></div>')
    A('<p class="small muted" style="margin-top:8px">About knee bend: '
      '&ldquo;the right knee bend&rdquo; means your knees were flexed into the range '
      'good technique calls for on that shot &mdash; you get lower on soft, control '
      'shots and less on power shots. Target bend (how far the knees flex from '
      'straight): serve &amp; return 10&ndash;30&deg;, drive 20&ndash;35&deg;, '
      'third-shot drop 30&ndash;45&deg;, dink &amp; reset 35&ndash;50&deg;.</p>')

    # ---- Improvement plan ----
    A('<h2>Your improvement plan</h2><hr class="rule">')
    tgt = plan.get("target", {}) or {}
    A(f'<p class="muted small">Toward USAPA {esc(tgt.get("band","—"))}: '
      f'{esc(tgt.get("rationale",""))}</p>')
    for f in plan.get("focus_areas", []):
        A('<div class="card focus">')
        A(f'<h3>{esc(CATEGORY_LABEL.get(f["dimension"], f["dimension"]))}</h3>')
        A(f'<p>{esc(f.get("finding",""))}</p>')
        if f.get("why_it_matters"):
            A(f'<p class="small muted">{esc(f["why_it_matters"])}</p>')
        for dr in f.get("drills", []):
            A(f'<div class="drill"><b>{esc(dr.get("name",""))}:</b> {esc(dr.get("cue",""))}</div>')
        A('</div>')
    if not_assessable:
        A('<div class="card"><h3>Not coached yet</h3>'
          '<p class="small muted">These need upcoming detection work before we can '
          'coach them honestly:</p>')
        for name, e in not_assessable.items():
            A(f'<p class="small metric"><b>{esc(CATEGORY_LABEL.get(name,name))}:</b> '
              f'{esc(e.get("reason",""))}</p>')
        A('</div>')

    # ---- USAPA ratings ladder ----
    A('<h2>USAPA ratings</h2><hr class="rule">')
    A(f'<p class="muted small">The official skill levels and what each looks like '
      f'across the seven categories.{fn(2)} You\'re highlighted.</p>')
    A('<div class="card scrollx"><table><thead><tr><th>Level</th>'
      '<th>What it looks like</th></tr></thead><tbody>')
    for lvl, desc in USAPA_LADDER:
        here = (lvl == rt.get("band"))
        mark = ' &nbsp;<span class="badge b-measured">You</span>' if here else ''
        A(f'<tr class="{ "here" if here else "" }"><td class="lvl">{esc(lvl)}{mark}</td>'
          f'<td class="small">{esc(desc)}</td></tr>')
    A('</tbody></table></div>')

    # ---- Court positioning ----
    A('<h2>Court positioning</h2><hr class="rule">')
    A('<p class="muted small">Where each player spent time during points.'
      + fn(3) + '</p>')
    A('<div class="grid2">')
    # A CUMULATIVE report covers several videos, where "Opponent A" is not one person and
    # neither is the partner. Stage 7.9 already pools every opponent into one bucket
    # (contract D1), so the labels have to say so rather than implying a named individual.
    role_labels = ([("user", "You"), ("partner", "Partners"), ("opp_a", "Opponents")]
                   if collection else
                   [("user", "You"), ("partner", "Partner"),
                    ("opp_a", "Opponent A"), ("opp_b", "Opponent B")])
    for role, label in role_labels:
        uri = data_uri_png(folder / f"heatmap_position_{role}.png")
        if uri:
            A(f'<div class="card hm"><h3>{esc(label)}</h3>'
              f'<img alt="{esc(label)} position" src="{uri}"></div>')
    A('</div>')
    A('<div class="card hm"><div class="ramp"></div>'
      '<p class="small muted" style="margin:2px 0 0">Dark = little time there · '
      'bright yellow = where you spent the most time. The white line is the net.</p></div>')

    # ball landings
    all_bounces = bounces_doc.get("bounces", [])
    rally_windows = [(int(r["start_frame"]), int(r["end_frame"]))
                     for r in rallies]
    def _inr(f):
        return any(a <= f <= b for a, b in rally_windows)
    n_inr = sum(1 for b in all_bounces
                if b.get("court_xy_ft") and _inr(int(b["frame"])))
    # Link each bounce to the shot that produced it, but only for the user's own shots.
    # between_shots[0] is the shot the bounce stage already attributed the bounce to, so
    # this needs no timing guesswork; a volley simply has no bounce pointing at it.
    _user_tids = {int(t) for t, i in (track_roles.get("track_roles", {}) or {}).items()
                  if i.get("role") == "user"}
    _shot_by_id = {int(x["shot_id"]): x for x in classified.get("shots", [])}
    user_hits = {}
    for b in all_bounces:
        prev = (b.get("between_shots") or [None, None])[0]
        sh = _shot_by_id.get(int(prev)) if prev is not None else None
        if sh is None or sh.get("track_id") is None:
            continue
        if int(sh["track_id"]) not in _user_tids:
            continue
        hxy = sh.get("hitter_court_xy_ft")
        if hxy and hxy[0] is not None:
            user_hits[b.get("bounce_id")] = (float(hxy[0]), float(hxy[1]))
    land = landing_diagram_uri(all_bounces, rally_windows, user_hits)
    if land:
        A('<div class="grid2"><div class="card hm"><h3>Where the ball bounced</h3>'
          f'<img alt="ball landing sequence" src="{land}"></div>'
          '<div class="card"><h3>Reading it</h3>'
          f'<p class="small muted">Lines join <b>your</b> shots to where that ball '
          f'landed ({len(user_hits)} of them) — the dot at the start is where you hit '
          f'from. Your volleys have no line: a volley never bounces.</p>'
          f'<p class="small muted">Each dot is where the ball bounced during a rally — '
          f'<span style="color:var(--court)">●</span> in bounds · '
          f'<span style="color:#d23">●</span> out. Near baseline at the bottom, far '
          f'court at the top, net across the middle.</p>'
          f'<p class="small muted">Showing {n_inr} of {len(all_bounces)} detected '
          f'bounces (the rest were between points). Volleys never bounce, and a '
          f'ball hit into the net doesn\'t bounce either, so those aren\'t shown. '
          f'A few real bounces are also still missed by detection{fn(5)}.</p></div></div>')

    # ---- Match video + point index ----
    # Deliberately NOT a re-rendered video with overlays. Those overlays (boxes, ball
    # trails, track ids) exist to verify DETECTION; a player does not need a box drawn
    # around themselves. What they need is "watch these moments" — and the rally
    # boundaries that answer it already exist, so this costs no rendering, no extra
    # storage and no waiting, on a source video that is several GB.
    A('<h2>Watch the points</h2><hr class="rule">')
    if (folder / "video.mp4").exists():
        A('<div class="card vid">')
        A('<video id="matchvid" controls preload="metadata" src="video.mp4"></video>')
        A(f'<p class="small muted">Click any time below to jump straight there — '
          f'{JUMP_LEAD_S:g}s before a rally starts, {SHOT_LEAD_S:g}s before a single '
          f'shot, so you see the setup rather than landing mid-motion.</p>')
        A('<h3>Rallies</h3>')
        A('<div class="points">')
        for i, r in enumerate(rallies, 1):
            t0 = max(0.0, float(r.get("start_t_sec", 0.0)) - JUMP_LEAD_S)
            dur = float(r.get("duration_sec") or 0.0)
            # A real point runs ~5-15s. A much longer one is usually several points run
            # together, because rally-END detection is a known open limitation — so flag
            # it rather than presenting "31 shots" as one rally the player did not play.
            long = dur > LONG_RALLY_S
            A(f'<button class="pt{" pt-long" if long else ""}" data-t="{t0:.2f}"'
              f'{" title=\'Unusually long - this may be several points run together\'" if long else ""}>'
              f'<span class="pt-n">{i}</span>'
              f'<span class="pt-t">{clock(r.get("start_t_sec", 0))}</span>'
              f'<span class="pt-d">{dur:.0f}s{" &middot; ?" if long else ""}</span>'
              f'</button>')
        A('</div>')
        n_long = sum(1 for r in rallies
                     if float(r.get("duration_sec") or 0) > LONG_RALLY_S)
        if n_long:
            A(f'<p class="small muted">{n_long} rall{"y is" if n_long == 1 else "ies are"} '
              f'longer than {LONG_RALLY_S:g}s and marked <b>?</b> — a point rarely runs '
              f'that long, so those are probably several points run together. Detecting '
              f'where a point ENDS is a known open limitation.</p>')

        # Jump straight to the operator's own shots by type. Same zero-cost mechanism as
        # the rally list, just a different filter over shots we already classified.
        groups = user_shot_groups(classified, rallies, track_roles)
        if any(groups.values()):
            A('<h3>Your shots</h3>')
            for label, items in groups.items():
                if not items:
                    continue
                A(f'<div class="shotrow"><span class="shotrow-l">{esc(label)} '
                  f'<b>({len(items)})</b></span><div class="points">')
                for t in items:
                    A(f'<button class="pt pt-s" data-t="{max(0.0, t - SHOT_LEAD_S):.2f}">'
                      f'<span class="pt-t">{clock(t)}</span></button>')
                A('</div></div>')
        A('<p class="small muted">Plays when you open this report from the same folder '
          'as the video; a shared copy will not include it. '
          '<a href="video.mp4" download>Download the video</a>.</p>')
        A('</div>')
        A('<script>document.querySelectorAll(".pt").forEach(function(b){'
          'b.addEventListener("click",function(){var v=document.getElementById("matchvid");'
          'v.currentTime=parseFloat(b.dataset.t);v.play();'
          'v.scrollIntoView({behavior:"smooth",block:"center"});});});</script>')
    elif collection:
        # A cumulative report has no single video — it has several. Rather than send the
        # operator off to open each per-video report, give one player per member with that
        # member's own rallies underneath it.
        members = collection.get("members", []) or []
        member_groups: dict = {}
        shown = 0
        for m in members:
            sid = m.get("session_id")
            mf = Path(m.get("path") or "")
            if not sid or not (mf / "video.mp4").exists():
                continue
            try:
                mr = json.loads((mf / "rallies.json").read_text(encoding="utf-8"))["rallies"]
            except (OSError, json.JSONDecodeError, KeyError):
                mr = []
            shown += 1
            vid = f"vid{shown}"
            # Each member's own shots, keyed to ITS player. Computed from the member
            # folder rather than the union: the union's times are offset and its track ids
            # renumbered, so member-local times are what seek this member's video.
            try:
                mcl = json.loads((mf / "classified.json").read_text(encoding="utf-8"))
                mtr = json.loads((mf / "track_roles.json").read_text(encoding="utf-8"))
                for lbl, ts in user_shot_groups(mcl, mr, mtr).items():
                    member_groups.setdefault(lbl, []).extend((vid, shown, t) for t in ts)
            except (OSError, json.JSONDecodeError, KeyError):
                pass
            # Rallies join the same cross-video index as the shot kinds, rather than
            # sitting in a per-video grid. Two layouts for the same idea made the reader
            # learn the page twice.
            for r in mr:
                member_groups.setdefault("Rallies", []).append(
                    (vid, shown, float(r.get("start_t_sec", 0.0)),
                     float(r.get("duration_sec") or 0.0)))
            # Served through the app's per-session file route: the member videos live
            # outside this folder, and the collection file route (rightly) refuses to
            # serve anything outside it.
            src = f"/api/sessions/{sid}/files/video.mp4"
            A('<div class="card vid">')
            A(f'<h3>Video {shown} &mdash; {esc(m.get("session_id"))}</h3>')
            A(f'<video id="{vid}" controls preload="none" src="{esc(src)}"></video>')
            A('</div>')
        if shown and any(member_groups.values()):
            # The point of a cumulative report: every rally, third shot, dink or volley
            # across ALL videos in one place, each button seeking its own video.
            A('<h3>Jump to any moment, across all videos</h3>')
            # Rallies first: they are the coarse index, the shot kinds refine it.
            order = ["Rallies"] + [k for k in member_groups if k != "Rallies"]
            n_long = 0
            for label in order:
                items = member_groups.get(label) or []
                if not items:
                    continue
                A(f'<div class="shotrow"><span class="shotrow-l">{esc(label)} '
                  f'<b>({len(items)})</b></span><div class="points">')
                for it in items:
                    vid, n, t = it[0], it[1], it[2]
                    dur = it[3] if len(it) > 3 else None
                    lead = JUMP_LEAD_S if dur is not None else SHOT_LEAD_S
                    long = dur is not None and dur > LONG_RALLY_S
                    n_long += 1 if long else 0
                    extra = (f'<span class="pt-d">{dur:.0f}s{" &middot; ?" if long else ""}'
                             f'</span>') if dur is not None else ''
                    A(f'<button class="pt{"" if dur is not None else " pt-s"}'
                      f'{" pt-long" if long else ""}" data-v="{vid}" '
                      f'data-t="{max(0.0, t - lead):.2f}">'
                      f'<span class="pt-n">{n}</span>'
                      f'<span class="pt-t">{clock(t)}</span>{extra}</button>')
                A('</div></div>')
            A('<p class="small muted">The small number is which video above.</p>')
            if n_long:
                A(f'<p class="small muted">{n_long} rall{"y is" if n_long == 1 else "ies are"} '
                  f'longer than {LONG_RALLY_S:g}s and marked <b>?</b> — a point rarely runs '
                  f'that long, so those are probably several points run together.</p>')
        if shown:
            A('<p class="small muted">One player per video in this report. Videos load '
              'from the app, so this section needs the app open (a downloaded copy of '
              'this page will not include them).</p>')
            A('<script>document.querySelectorAll(".pt[data-v]").forEach(function(b){'
              'b.addEventListener("click",function(){'
              'var v=document.getElementById(b.dataset.v);'
              'v.currentTime=parseFloat(b.dataset.t);v.play();'
              'v.scrollIntoView({behavior:"smooth",block:"center"});});});</script>')
        else:
            A('<p class="muted small">The videos for this report are not on this '
              'computer, so there is nothing to scrub.</p>')
    else:
        A('<p class="muted small">The match video isn\'t in this folder.</p>')

    # The "Coming soon" section advertised two things that now exist: body mechanics is
    # measured and feeds the categories above (ready position, knee bend, contact point),
    # and cross-session trends are cumulative reports. Promising delivered features as
    # future ones makes the whole report look out of date.

    # ---- Footnotes ----
    A('<div class="foot"><h3>Notes</h3><ol>')
    A(f'<li id="fn1">Measurement coverage is {int(round((rt.get("confidence") or 0)*100))}%. '
      f'This is how much of the full 7-category skill picture we can measure from one '
      f'camera yet — <b>not</b> how sure we are of your rating. It\'s low mainly '
      f'because shot <i>quality</i> (pace, dink height, return depth) isn\'t measured '
      f'yet; the counts we do report are validated. Thresholds are uncalibrated '
      f'heuristics anchored to the USAPA definitions, not an official rating.</li>')
    A('<li id="fn2">Level descriptions are a condensed synthesis of the published '
      'USA Pickleball definitions across the seven categories.</li>')
    A('<li id="fn3">Positioning is measured from the player\'s front foot, during '
      'live rallies only (between-point standing is excluded).</li>')
    A('<li id="fn4">Court covered is a work-rate figure (feet per minute of play). '
      'On its own it isn\'t good or bad — strong players often move <i>less</i> but '
      'get to better spots — so we don\'t rate it. The coachable lever is footwork '
      'and positioning, which lives under Strategy above.</li>')
    A(f'<li id="fn5">Every ground shot bounces once, so bounces should equal your '
      f'non-volley shots: about {max(0, len(shots) - n_volley)} expected here, '
      f'{len(all_bounces)} detected. The ~{max(0, len(shots) - n_volley - len(all_bounces))} '
      f'gap is real bounces missed by detection — a known ball-detection limit we\'re '
      f'improving. It thins the landing map and depth stats, but doesn\'t affect your '
      f'positioning or rating.</li>')
    A('</ol>')
    A('</div></div>')
    return _PAGE.replace("__CSS__", CSS).replace("__BODY__", "\n".join(O))


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your Pickleball Player Report</title><style>__CSS__</style></head>
<body>__BODY__</body></html>"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the consumer report HTML")
    p.add_argument("folder", type=Path)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default="report.html")
    args = p.parse_args(argv)
    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 1
    if not (args.folder / "rating.json").exists():
        print(f"rating.json not found in {args.folder} (run Stages 8-11 first)",
              file=sys.stderr)
        return 1
    out_path = args.folder / args.out
    if out_path.exists() and not args.force:
        print(f"output exists: {out_path}. Use --force.", file=sys.stderr)
        return 1
    s = build_html(args.folder)
    out_path.write_text(s, encoding="utf-8")
    print(f"wrote {out_path} ({len(s)//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

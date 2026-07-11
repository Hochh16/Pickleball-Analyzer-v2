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
                 ("Court coverage & movement", "live"), ("Stacking", "planned"),
                 ("Targeting weakness", "planned"), ("Resets under pressure", "planned"),
                 ("Unforced errors", "planned")],
    "third_shot": [("How often you play the 3rd shot", "live"),
                   ("Drop-vs-drive choice", "partial"),
                   ("Drop landing depth", "planned"), ("Transition success", "planned")],
    "dink": [("How much you dink", "partial"), ("Dink-rally length", "partial"),
             ("Pop-up rate", "planned"), ("Height & depth control", "planned")],
    "volley": [("How often you volley at the net", "partial"),
               ("Block / reset", "planned"), ("Put-aways", "planned"),
               ("Speed-ups & counters", "planned")],
    "serve_return": [("Serve / return count", "live"),
                     ("In-play rate & faults", "partial"),
                     ("Depth", "planned"), ("Pace & spin", "planned")],
    "forehand": [("How many forehands you hit", "live"), ("Consistency", "partial"),
                 ("Pace", "planned"), ("Placement & depth", "planned")],
    "backhand": [("How many backhands you hit", "live"),
                 ("Running around it", "partial"), ("Error rate", "partial"),
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
    "volley_rate": ("Shots played as a net volley", "pct"),
    "n_volley": ("Net volleys detected", "int"),
    "serve_fault_rate": ("Serves that faulted", "pct"),
    "n_serves": ("Serves detected", "int"),
    "forehand_count": ("Forehands detected", "int"),
    "backhand_count": ("Backhands detected", "int"),
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
    except (TypeError, ValueError):
        return None
    return str(val)


# --- Ball-landing sequence diagram (drawn with OpenCV) -----------------------

def landing_diagram_uri(bounces: list) -> Optional[str]:
    """Top-down court with in-bounds ball landings numbered in play order and
    connected, so the reader sees the sequence. Returns a PNG data URI."""
    import cv2
    pts = [(b["court_xy_ft"], bool(b.get("is_in_court")))
           for b in bounces
           if b.get("court_xy_ft") and b["court_xy_ft"][0] is not None]
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
    # landings, numbered + connected in order
    teal, red, grey = (110, 118, 15), (60, 60, 210), (150, 150, 150)
    prev = None
    for i, (xy, ok) in enumerate(pts, 1):
        p = to_px(xy[0], xy[1])
        if prev is not None:
            cv2.line(img, prev, p, grey, 1, cv2.LINE_AA)
        prev = p
    for i, (xy, ok) in enumerate(pts, 1):
        p = to_px(xy[0], xy[1])
        cv2.circle(img, p, 9, teal if ok else red, -1, cv2.LINE_AA)
        cv2.circle(img, p, 9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(i), (p[0] - (4 if i < 10 else 7), p[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
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
"""


def badge(cov: str) -> str:
    label, cls = COVERAGE.get(cov, ("—", "b-na"))
    return f'<span class="badge {cls}">{esc(label)}</span>'


def fn(n: int) -> str:
    return f'<sup><a href="#fn{n}">{n}</a></sup>'


def build_html(folder: Path) -> str:
    rating = load_json(folder, "rating.json") or {}
    plan = load_json(folder, "improvement_plan.json") or {}
    bounces_doc = load_json(folder, "bounces.json") or {}
    timeline = load_json(folder, "timeline.json") or {}

    rt = rating.get("rating", {}) or {}
    dims = {d["name"]: d for d in rating.get("dimensions", [])}
    rel = rating.get("reliability", {}) or {}
    not_assessable = {e["dimension"]: e for e in
                      (plan.get("developing_capability", {}) or {}).get("not_assessable_now", [])}

    def cov_of(c):
        # reconcile: plan's zero-event guard overrides the rating's confidence status
        return "not_assessable" if c in not_assessable else dims.get(c, {}).get(
            "coverage_status", "not_assessable")

    O = []
    A = O.append
    A('<div class="wrap">')

    # ---- Hero ----
    est = rt.get("estimate")
    A('<p class="eyebrow">USA Pickleball–aligned skill report</p>')
    A('<h1>Your Player Report</h1>')
    A('<div class="card hero">')
    A(f'<div><div class="score">{est if est is not None else "—"}</div>'
      f'<div class="muted small">USAPA band {esc(rt.get("band","—"))}</div></div>')
    rng = rt.get("range") or [None, None]
    A('<div class="hero-meta">')
    A(f'<p style="margin-top:0"><b>Your estimated rating is {est}</b>, most likely '
      f'between {rng[0]} and {rng[1]}.{fn(1)}</p>')
    measured = rel.get("measured_categories", [])
    na_cats = [c for c in CATEGORY_ORDER if cov_of(c) == "not_assessable"]
    A(f'<p class="small muted" style="margin-bottom:0">Right now this rests mostly on '
      f'your <b>court strategy &amp; positioning</b> — the part we can measure well '
      f'from one camera. {len(na_cats)} of the 7 categories aren\'t measured yet and '
      f'are marked below.{fn(2)}</p>')
    A('</div></div>')

    # ---- 7-category table ----
    A('<h2>Your 7 categories</h2><hr class="rule">')
    A('<p class="muted small">USA Pickleball rates players across these seven '
      'categories. Here\'s your level in each, and how well we can measure it today.</p>')
    A('<div class="card scrollx"><table><thead><tr>'
      '<th>Category</th><th>Your level</th><th>Coverage</th></tr></thead><tbody>')
    for c in CATEGORY_ORDER:
        d = dims.get(c, {})
        sub = d.get("subscore_level")
        cov = cov_of(c)
        if cov == "not_assessable" or not isinstance(sub, (int, float)):
            lvl = '<span class="muted">—</span>'
        else:
            barpct = int(max(0, min(100, ((sub - 1.0) / 4.5) * 100)))
            lvl = (f'<span class="lvl num">{band_of(sub)}</span>'
                   f'<div class="bar"><i style="width:{barpct}%"></i></div>')
        A(f'<tr><td><b>{esc(CATEGORY_LABEL[c])}</b></td><td>{lvl}</td>'
          f'<td>{badge(cov)}</td></tr>')
    A('</tbody></table></div>')
    A('<div class="legend">'
      '<span><span class="badge b-measured">Measured</span> real data, high confidence</span>'
      '<span><span class="badge b-partial">Partial</span> early signal — read as a hint</span>'
      '<span><span class="badge b-na">Not yet measured</span> needs upcoming detection</span>'
      '</div>')

    # ---- Category detail table ----
    A('<h2>What\'s behind each category</h2><hr class="rule">')
    A('<div class="card scrollx"><table><thead><tr>'
      '<th>Category</th><th>Your numbers now</th>'
      '<th>What USA Pickleball rates</th></tr></thead><tbody>')
    for c in CATEGORY_ORDER:
        d = dims.get(c, {})
        drivers = d.get("driver_metrics", {}) or {}
        nums = []
        for k, (label, fmt) in METRIC_DISPLAY.items():
            if k in drivers:
                s = fmt_metric(fmt, drivers[k])
                if s is not None:
                    nums.append(f'<div class="metric">{esc(label)}: <b>{esc(s)}</b></div>')
        numhtml = "".join(nums) if nums else '<span class="muted small">—</span>'
        els = "".join(
            f'<span class="el"><span class="sym">{SYMBOL[st]}</span> '
            f'<span class="{ "planned" if st=="planned" else "" }">{esc(lbl)}</span></span>'
            for lbl, st in CATEGORY_ELEMENTS[c])
        A(f'<tr><td><b>{esc(CATEGORY_LABEL[c])}</b><br>{badge(cov_of(c))}</td>'
          f'<td>{numhtml}</td><td class="small">{els}</td></tr>')
    A('</tbody></table></div>')
    A('<div class="legend"><span><span class="sym">●</span> measured now</span>'
      '<span><span class="sym">◐</span> partial / early signal</span>'
      '<span><span class="sym">○</span> coming soon</span></div>')

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
      f'across the seven categories.{fn(3)} You\'re highlighted.</p>')
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
      + fn(4) + '</p>')
    A('<div class="grid2">')
    for role, label in [("user", "You"), ("partner", "Partner"),
                        ("opp_a", "Opponent A"), ("opp_b", "Opponent B")]:
        uri = data_uri_png(folder / f"heatmap_position_{role}.png")
        if uri:
            A(f'<div class="card hm"><h3>{esc(label)}</h3>'
              f'<img alt="{esc(label)} position" src="{uri}"></div>')
    A('</div>')
    A('<div class="card hm"><div class="ramp"></div>'
      '<p class="small muted" style="margin:2px 0 0">Dark = little time there · '
      'bright yellow = where you spent the most time. The white line is the net.</p></div>')

    # ball landings
    land = landing_diagram_uri(bounces_doc.get("bounces", []))
    if land:
        A('<div class="grid2"><div class="card hm"><h3>Ball landings, in order</h3>'
          f'<img alt="ball landing sequence" src="{land}"></div>'
          '<div class="card"><h3>Reading it</h3>'
          '<p class="small muted">Each detected ball bounce is numbered in the order '
          'it happened and connected by a line, so you can follow the flow of play. '
          '<span style="color:var(--court)">●</span> landed in bounds · '
          '<span style="color:#d23">●</span> landed out. Near baseline is at the '
          'bottom, the far court at the top, net across the middle.</p></div></div>')

    # ---- Annotated video ----
    A('<h2>Annotated video &amp; timeline</h2><hr class="rule">')
    if (folder / "annotated.mp4").exists():
        A('<div class="card vid">'
          '<video controls preload="metadata" src="annotated.mp4"></video>'
          '<p class="small muted">Per-shot labels, the ball trail, and a live court '
          'mini-map. <a href="annotated.mp4" download>Download the video</a>. '
          '(Plays when you open this report from the same folder as the video; a '
          'shared/online copy won\'t include it.)</p></div>')
    else:
        A('<p class="muted small">No annotated video yet — run the render step to '
          'produce <code>annotated.mp4</code> next to this report.</p>')
    n_events = len(timeline.get("events", []))
    A(f'<p class="small muted">Timeline: {n_events} shot &amp; bounce events over '
      f'{timeline.get("duration_sec","—")} seconds, each carrying its own confidence.</p>')

    # ---- Technique & trends ----
    A('<h2>Coming soon</h2><hr class="rule">')
    A('<div class="grid2">'
      '<div class="card"><h3>Technique (body mechanics)</h3>'
      '<p class="small muted">A supporting pose layer — split-step timing, '
      'knees-bent dinks, contact-point consistency, ready-position recovery — that '
      'feeds the categories above. Not measured yet.</p></div>'
      '<div class="card"><h3>Trends across sessions</h3>'
      '<p class="small muted">Your rating and key stats over time, once multiple '
      'sessions are linked. This is a single-session report.</p></div></div>')

    # ---- Footnotes ----
    A('<div class="foot"><h3>Notes</h3><ol>')
    A(f'<li id="fn1">Overall confidence is {int(round((rt.get("confidence") or 0)*100))}%. '
      f'That reflects how much of the full skill picture we can measure yet — a '
      f'confident read of <i>incomplete</i> coverage, not an uncertain rating. '
      f'Thresholds are uncalibrated heuristics anchored to the USAPA definitions, '
      f'not an official rating.</li>')
    A('<li id="fn2">"Not yet measured" categories need upcoming shot-speed, serve, '
      'and stroke detection. We flag them rather than guess a number.</li>')
    A('<li id="fn3">Level descriptions are a condensed synthesis of the published '
      'USA Pickleball definitions across the seven categories.</li>')
    A('<li id="fn4">Positioning is measured from the player\'s front foot, during '
      'live rallies only (between-point standing is excluded).</li>')
    A('</ol>')
    for w in (rating.get("warnings", []) or []):
        A(f'<p>⚠ {esc(w)}</p>')
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

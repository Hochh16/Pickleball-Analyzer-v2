"""Build the consumer report — per-video JSONs -> a self-contained report.html.

Reads a per-video folder's Stage 8-11 outputs (rating.json, improvement_plan.json,
metrics.json, timeline.json, heatmap PNGs, annotated.mp4) and emits a single
self-contained `report.html` (images inline as data URIs) that a player can open
in any browser. Aligned to the 7 official USAPA categories; honest about coverage
(measured / partial / not-yet-measured) rather than overclaiming.

Usage:
    python -m tools.build_report data/pb_2min
    python -m tools.build_report data/pb_2min --force
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import sys
from pathlib import Path
from typing import Optional

# --- Static reference content (USAPA-anchored) -------------------------------

CATEGORY_ORDER = ["strategy", "third_shot", "dink", "volley", "serve_return",
                  "forehand", "backhand"]

CATEGORY_LABEL = {
    "strategy": "Strategy", "third_shot": "Third Shot", "dink": "Dink",
    "volley": "Volley", "serve_return": "Serve / Return",
    "forehand": "Forehand", "backhand": "Backhand",
}

# What USA Pickleball actually rates in each category (official descriptive
# elements, condensed from the published level definitions).
USAPA_RATES = {
    "strategy": "Court positioning & NVZ approach, hard-vs-soft game, moving as a "
                "team + coverage, stacking, targeting weakness, poaching, resets, "
                "and unforced errors.",
    "third_shot": "The third-shot drop to the net, soft-vs-power selection, "
                  "direction, and placement 'not easily returned.'",
    "dink": "Dink-rally sustain, height/depth control, consistency, pace variation, "
            "recognizing attackable balls, and offensive intent.",
    "volley": "Handling pace, control, block/reset volleys, swinging volleys, and "
              "overhead put-aways.",
    "serve_return": "In-play consistency, depth, direction, pace, and spin variation "
                    "on both the serve and the return.",
    "forehand": "Pace, directional control, consistency, and depth on the forehand.",
    "backhand": "Consistency, direction, tendency to avoid it, and depth/pace on "
                "the backhand.",
}

# Per-category driver metrics we currently map to it (● live · ◐ partial · ○ planned).
CATEGORY_METRIC_MAP = {
    "strategy": "zone times (kitchen/transition/baseline) ● · move-as-a-team ● · "
                "court coverage/movement ● · stacking ○ · targeting ○ · resets ○ · "
                "unforced errors ○",
    "third_shot": "3rd-shot count ● · drop-vs-drive mix ◐ · drop landing depth ○ · "
                  "transition success ○",
    "dink": "dink count ◐ · dink-rally length ◐ · pop-up rate ○ · height/depth ○",
    "volley": "volley count ◐ · block/reset ○ · put-away ○ · speed-up/counter ○",
    "serve_return": "serve/return count ● · in-play % / faults ◐ · depth ○ · "
                    "pace/spin ○",
    "forehand": "FH count ● · consistency ◐ · pace mph ○ · placement/depth ○",
    "backhand": "BH count ● · avoids-BH ratio ◐ · error rate ◐ · pace/depth ○",
}

SKILL_LADDER = [
    ("2.0", "True beginner; can't reliably direct the ball."),
    ("2.5", "Sustains a short rally; serves in; moves to the kitchen late; pops up dinks."),
    ("3.0", "Keeps the ball in play; knows the 3rd-shot-drop idea (inconsistent); chooses drop vs drive."),
    ("3.5", "Dinks moderately consistent; 3rd-shot drop with a kitchen plan; basic stacking."),
    ("4.0", "Deeper returns; drops with a clean transition; resets from transition; reads attackable balls."),
    ("4.5", "Absorbs pace (blocks/resets); speeds up at the right targets; disciplined dinks."),
    ("5.0", "Drop/drive/hybrid selected correctly; resets under stress; precise speed-ups."),
    ("5.5+", "Outcomes tier — dominance + tournament results."),
]

COVERAGE_LABEL = {
    "measured": ("Measured", "Real data, high confidence."),
    "partial": ("Partial", "Some signal, low confidence — read as a hint."),
    "not_assessable": ("Not yet measured", "Needs upcoming detection — not rated yet."),
}


# --- Data helpers ------------------------------------------------------------

def load_json(folder: Path, name: str) -> Optional[dict]:
    p = folder / name
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def v(x):
    """Unwrap a Stage 8 {value, confidence, ...} wrapper to its value."""
    return x["value"] if isinstance(x, dict) and "value" in x else x


def conf_of(x) -> Optional[float]:
    return x.get("confidence") if isinstance(x, dict) and "confidence" in x else None


def data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b}"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(x, digits=0) -> str:
    return f"{x * 100:.{digits}f}%" if isinstance(x, (int, float)) else "—"


def band_of(estimate: float) -> str:
    e = max(1.0, min(5.0, estimate))
    return f"{round(e * 2.0) / 2.0:.1f}"


# --- HTML building -----------------------------------------------------------

CSS = """
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#1a1d21; --muted:#5c6773; --line:#e4e8ec;
  --accent:#2f6f4f; --accent2:#3a6ea5;
  --measured:#2f8f5b; --measured-bg:#e7f4ec;
  --partial:#b8860b; --partial-bg:#faf2dd;
  --na:#7a828b; --na-bg:#eef1f3;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#14171a; --card:#1d2126; --ink:#e8ebee; --muted:#9aa4ae; --line:#2c333a;
    --accent:#57b98a; --accent2:#6fa8dc;
    --measured:#5fc88e; --measured-bg:#173226; --partial:#e0b64b; --partial-bg:#332a12;
    --na:#8b95a0; --na-bg:#232830; }
}
:root[data-theme="dark"]{ --bg:#14171a; --card:#1d2126; --ink:#e8ebee; --muted:#9aa4ae; --line:#2c333a;
  --accent:#57b98a; --accent2:#6fa8dc; --measured:#5fc88e; --measured-bg:#173226;
  --partial:#e0b64b; --partial-bg:#332a12; --na:#8b95a0; --na-bg:#232830; }
:root[data-theme="light"]{ --bg:#f6f7f9; --card:#ffffff; --ink:#1a1d21; --muted:#5c6773; --line:#e4e8ec;
  --accent:#2f6f4f; --accent2:#3a6ea5; --measured:#2f8f5b; --measured-bg:#e7f4ec;
  --partial:#b8860b; --partial-bg:#faf2dd; --na:#7a828b; --na-bg:#eef1f3; }

*{ box-sizing:border-box; }
body{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap{ max-width:900px; margin:0 auto; padding:24px 18px 80px; }
h1{ font-size:26px; margin:0 0 2px; }
h2{ font-size:19px; margin:34px 0 12px; padding-bottom:6px; border-bottom:2px solid var(--line); }
h3{ font-size:15px; margin:0 0 4px; }
p{ margin:8px 0; } .muted{ color:var(--muted); } .small{ font-size:13px; }
.card{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px; margin:12px 0; }
.hero{ display:flex; flex-wrap:wrap; gap:18px; align-items:center; }
.score{ font-size:54px; font-weight:700; line-height:1; color:var(--accent); }
.band{ font-size:15px; color:var(--muted); }
.hero-meta{ flex:1; min-width:220px; }
.badge{ display:inline-block; font-size:11px; font-weight:600; padding:2px 8px;
  border-radius:20px; letter-spacing:.02em; text-transform:uppercase; }
.b-measured{ color:var(--measured); background:var(--measured-bg); }
.b-partial{ color:var(--partial); background:var(--partial-bg); }
.b-na{ color:var(--na); background:var(--na-bg); }
table{ width:100%; border-collapse:collapse; font-size:14px; }
th,td{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
.cat-row td:first-child{ font-weight:600; }
.bar{ height:7px; border-radius:4px; background:var(--na-bg); overflow:hidden; margin-top:5px; }
.bar > i{ display:block; height:100%; background:var(--accent); }
.banner{ border-left:4px solid var(--partial); background:var(--partial-bg);
  padding:10px 14px; border-radius:8px; margin:10px 0; font-size:14px; }
.grid2{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }
@media (max-width:620px){ .grid2{ grid-template-columns:1fr; } }
.hm{ text-align:center; } .hm img{ max-width:100%; border-radius:8px; border:1px solid var(--line); }
.drill{ font-size:13.5px; margin:4px 0; padding-left:14px; position:relative; }
.drill::before{ content:"▸"; position:absolute; left:0; color:var(--accent); }
.focus{ border-left:4px solid var(--accent); }
.tag{ font-size:11px; color:var(--muted); }
.scrollx{ overflow-x:auto; } .foot{ color:var(--muted); font-size:12px; margin-top:40px;
  border-top:1px solid var(--line); padding-top:12px; }
a{ color:var(--accent2); }
"""


def badge(coverage: str) -> str:
    label = COVERAGE_LABEL.get(coverage, (coverage, ""))[0]
    cls = {"measured": "b-measured", "partial": "b-partial",
           "not_assessable": "b-na"}.get(coverage, "b-na")
    return f'<span class="badge {cls}">{esc(label)}</span>'


def build_html(folder: Path) -> str:
    rating = load_json(folder, "rating.json") or {}
    plan = load_json(folder, "improvement_plan.json") or {}
    metrics = load_json(folder, "metrics.json") or {}
    timeline = load_json(folder, "timeline.json") or {}

    rt = rating.get("rating", {}) or {}
    dims = {d["name"]: d for d in rating.get("dimensions", [])}
    rel = rating.get("reliability", {}) or {}
    ball_source = rating.get("ball_source", "real")
    focus_by_dim = {f["dimension"]: f for f in plan.get("focus_areas", [])}
    not_assessable = {e["dimension"]: e
                      for e in (plan.get("developing_capability", {}) or {})
                      .get("not_assessable_now", [])}

    out = []
    A = out.append

    # ---- Header / rating hero ----
    est = rt.get("estimate")
    A('<div class="wrap">')
    A(f'<h1>Your Pickleball Report</h1>')
    A(f'<p class="muted small">USA Pickleball–aligned analysis · '
      f'source: {esc(folder.name)} · ball source: {esc(ball_source)}</p>')
    A('<div class="card hero">')
    A(f'<div><div class="score">{est if est is not None else "—"}</div>'
      f'<div class="band">USAPA band {esc(rt.get("band","—"))}</div></div>')
    rng = rt.get("range") or [None, None]
    A('<div class="hero-meta">')
    A(f'<p><b>Estimated rating {est}</b> (likely range {rng[0]}–{rng[1]}).</p>')
    A(f'<p class="muted small">Overall confidence {pct(rt.get("confidence"))} — '
      f'a confident read of <i>incomplete</i> coverage: most categories aren\'t '
      f'fully measured yet (see below). Thresholds are uncalibrated heuristics '
      f'anchored to the USAPA definitions, not an official rating.</p>')
    A('</div></div>')

    measured = rel.get("measured_categories", [])
    na_cats = rel.get("not_assessable_categories", [])
    if na_cats:
        A(f'<div class="banner">This estimate rests mainly on '
          f'<b>{esc(", ".join(CATEGORY_LABEL.get(c,c) for c in measured) or "—")}</b> '
          f'(the categories we can measure well today). '
          f'{len(na_cats)} of 7 categories are not yet reliably measured '
          f'({esc(", ".join(CATEGORY_LABEL.get(c,c) for c in na_cats))}) — they '
          f'need upcoming shot-speed / serve / stroke detection and are shown here '
          f'as "not yet measured," never guessed.</div>')

    # ---- 7-category assessment table ----
    A('<h2>The 7 USA Pickleball categories</h2>')
    A('<p class="muted small">What USA Pickleball rates in each category, your '
      'assessment, and how well we can measure it today.</p>')
    A('<div class="card scrollx"><table>')
    A('<tr><th>Category</th><th>Your level</th><th>Coverage</th>'
      '<th>What USA Pickleball rates</th></tr>')
    for c in CATEGORY_ORDER:
        d = dims.get(c, {})
        sub = d.get("subscore_level")
        # Reconcile the two coverage lenses: if the plan routed a category to
        # not-assessable (its zero-event guard, e.g. 0 dinks), show it that way in
        # the table too even when the rating's confidence-based status was "partial".
        cov = ("not_assessable" if c in not_assessable
               else d.get("coverage_status", "not_assessable"))
        subtxt = (f'{band_of(sub)} <span class="tag">({sub:.1f})</span>'
                  if isinstance(sub, (int, float)) and cov != "not_assessable"
                  else '<span class="tag">—</span>')
        barpct = int(max(0, min(100, ((sub - 1.0) / 4.5) * 100))) if isinstance(sub, (int, float)) else 0
        bar = (f'<div class="bar"><i style="width:{barpct}%"></i></div>'
               if cov != "not_assessable" else '')
        A(f'<tr class="cat-row"><td>{esc(CATEGORY_LABEL[c])}</td>'
          f'<td>{subtxt}{bar}</td><td>{badge(cov)}</td>'
          f'<td class="small muted">{esc(USAPA_RATES[c])}</td></tr>')
    A('</table></div>')

    # ---- Per-category detail (metrics aligned to the category) ----
    A('<h2>Category detail — the numbers behind each</h2>')
    for c in CATEGORY_ORDER:
        d = dims.get(c, {})
        cov = ("not_assessable" if c in not_assessable
               else d.get("coverage_status", "not_assessable"))
        drivers = d.get("driver_metrics", {}) or {}
        A('<div class="card">')
        A(f'<h3>{esc(CATEGORY_LABEL[c])} {badge(cov)}</h3>')
        A(f'<p class="small muted">Metrics mapped to this category: '
          f'{esc(CATEGORY_METRIC_MAP[c])}</p>')
        # driver metric values
        rows = [f'{esc(k)}: <b>{esc(_fmt(val))}</b>' for k, val in drivers.items()
                if val is not None]
        if rows:
            A('<p class="small">' + ' &nbsp;·&nbsp; '.join(rows) + '</p>')
        # finding or not-assessable reason
        if c in focus_by_dim:
            A(f'<p><b>Finding:</b> {esc(focus_by_dim[c]["finding"])}</p>')
        elif c in not_assessable:
            A(f'<p class="muted small"><b>Not yet measured:</b> '
              f'{esc(not_assessable[c]["reason"])}</p>')
        A('</div>')

    # ---- Coaching plan ----
    A('<h2>Your improvement plan</h2>')
    tgt = plan.get("target", {}) or {}
    A(f'<p class="muted small">Toward USAPA {esc(tgt.get("band","—"))}: '
      f'{esc(tgt.get("rationale",""))}</p>')
    for f in plan.get("focus_areas", []):
        A('<div class="card focus">')
        A(f'<h3>#{f.get("priority","")} {esc(CATEGORY_LABEL.get(f["dimension"], f["dimension"]))} '
          f'<span class="tag">({esc(f.get("confidence",""))} confidence)</span></h3>')
        A(f'<p>{esc(f.get("finding",""))}</p>')
        if f.get("why_it_matters"):
            A(f'<p class="small muted">{esc(f["why_it_matters"])}</p>')
        for dr in f.get("drills", []):
            A(f'<div class="drill"><b>{esc(dr.get("name",""))}:</b> {esc(dr.get("cue",""))}</div>')
        A('</div>')
    if not_assessable:
        A('<div class="card"><h3>Not yet measured</h3>'
          '<p class="small muted">These need upcoming detection work before we can '
          'coach them honestly:</p>')
        for name, e in not_assessable.items():
            A(f'<p class="small"><b>{esc(CATEGORY_LABEL.get(name,name))}:</b> '
              f'{esc(e.get("reason",""))}</p>')
        A('</div>')

    # ---- Skill ladder ----
    A('<h2>Where you sit on the ladder</h2>')
    A('<div class="card scrollx"><table>')
    A('<tr><th>Level</th><th>What it looks like</th></tr>')
    cur_band = rt.get("band")
    for lvl, desc in SKILL_LADDER:
        here = (lvl == cur_band)
        mark = ' &nbsp;<span class="badge b-measured">you</span>' if here else ''
        style = ' style="background:var(--measured-bg)"' if here else ''
        A(f'<tr{style}><td><b>{esc(lvl)}</b>{mark}</td><td class="small">{esc(desc)}</td></tr>')
    A('</table></div>')

    # ---- Court positioning heatmaps ----
    A('<h2>Court positioning</h2>')
    A('<p class="muted small">Where each player stood during points (in-rally '
      'frames only, from the front foot).</p>')
    A('<div class="grid2">')
    for role, label in [("user", "You"), ("partner", "Partner"),
                        ("opp_a", "Opponent A"), ("opp_b", "Opponent B")]:
        uri = data_uri(folder / f"heatmap_position_{role}.png")
        if uri:
            A(f'<div class="card hm"><h3>{esc(label)}</h3>'
              f'<img alt="{esc(label)} position heatmap" src="{uri}"></div>')
    A('</div>')
    ball_hm = data_uri(folder / "heatmap_ball_landing.png")
    if ball_hm:
        A(f'<div class="card hm"><h3>Ball landings</h3>'
          f'<img alt="ball landing heatmap" src="{ball_hm}"></div>')

    # ---- Annotated video + timeline ----
    A('<h2>Annotated video & timeline</h2>')
    if (folder / "annotated.mp4").exists():
        A('<p>An annotated video (<code>annotated.mp4</code>) was rendered next to '
          'this report — open it to see per-shot labels, the ball trail, and the '
          'court mini-map.</p>')
    else:
        A('<p class="muted small">No annotated video rendered yet '
          '(<code>annotated.mp4</code> absent). Run Stage 11 render to produce it.</p>')
    summ = timeline.get("summary", {}) or {}
    n_events = len(timeline.get("events", []))
    A(f'<p class="small muted">Timeline: {n_events} events over '
      f'{timeline.get("duration_sec","—")}s. Each shot/bounce carries its own '
      f'confidence so the scrubbable view can gate what it shows.</p>')

    # ---- Technique (mechanics) — honest placeholder ----
    A('<h2>Technique (body mechanics)</h2>')
    A('<div class="card"><p class="muted small">A supporting pose layer '
      '(split-step timing, knees-bent-on-dinks, contact-point consistency, '
      'ready-position recovery, paddle-up) is planned — it feeds the categories '
      'above rather than being its own rating. <b>Not measured in this report yet.</b></p></div>')

    # ---- Trends — honest placeholder ----
    A('<h2>Trends across sessions</h2>')
    A('<div class="card"><p class="muted small">Single-session report. '
      'Rating + key stats over time will appear here once multiple sessions are '
      'linked by player identity.</p></div>')

    # ---- Footer / honesty ----
    A('<div class="foot">')
    for w in (rating.get("warnings", []) or []):
        A(f'<p>⚠ {esc(w)}</p>')
    A(f'<p>Generated by tools/build_report.py · Stage 9 {esc(rating.get("stage_version",""))} '
      f'· Stage 10 {esc(plan.get("stage_version",""))}.</p>')
    A('</div></div>')

    return _PAGE.replace("__CSS__", CSS).replace("__BODY__", "\n".join(out))


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.2f}".rstrip("0").rstrip(".")
    if isinstance(val, dict):
        return ", ".join(f"{k} {v}" for k, v in val.items())
    return str(val)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pickleball Report</title>
<style>__CSS__</style></head>
<body>__BODY__</body></html>
"""


# --- Main --------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the consumer report HTML")
    p.add_argument("folder", type=Path, help="per-video folder with the stage JSONs")
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
        print(f"output exists: {out_path}. Use --force to overwrite.", file=sys.stderr)
        return 1

    html_str = build_html(args.folder)
    out_path.write_text(html_str, encoding="utf-8")
    print(f"wrote {out_path} ({len(html_str)//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

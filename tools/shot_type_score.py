"""Score shot TYPE against the operator's labels, and say which confusions dominate.

    python -m tools.shot_type_score [clip ...]

Shot type is the weakest thing we measure -- 60% against the operator's labels, where
volleys run 82% and server side 95% -- and it is the output the USAPA categories are
actually built from: third-shot drops, dinks, drives. A single accuracy number cannot say
where to aim, because "drive read as drop" and "dink read as drop" have nothing in common
except the word wrong. So this prints the confusion matrix.

MATCHING. Same side, one-to-one, shortest pair first. Nearest-within-a-window is what the
older scorer does and it is not stable here: consecutive shots are often under a second
apart, so a widened window silently pairs a labelled shot with its neighbour, and the
answer moves with the window rather than with the code. The same fix settled the serve
count and the missing-bounce count.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.truth_store import known

MATCH_S = 0.6
LANDING_S = 2.5     # how long after a shot its landing bounce may be
# The types the operator uses. "serve" is included because it is a type they label, but it
# is reported separately: it is decided structurally, not by the type classifier.
TYPES = ("serve", "return", "drive", "dink", "drop", "lob")


def clock(t: float) -> str:
    return f"{int(t)//60}:{t % 60:05.2f}"


def pair_up(truth, shots, fps: float, window: float = MATCH_S):
    """One-to-one, side-constrained. Returns [(truth_row, shot_or_None)]."""
    cand = []
    for i, tr in enumerate(truth):
        t = float(tr["t_sec"])
        side = tr.get("side")
        for j, s in enumerate(shots):
            d = abs(int(s["frame"]) / fps - t)
            if d > window:
                continue
            if side and s.get("hitter_side") and s["hitter_side"] != side:
                continue
            cand.append((d, i, j))
    cand.sort()
    ti, sj = {}, set()
    for d, i, j in cand:
        if i in ti or j in sj:
            continue
        ti[i] = j
        sj.add(j)
    return [(tr, shots[ti[i]] if i in ti else None) for i, tr in enumerate(truth)]


def score_clip(clip: Path) -> dict:
    cl = json.loads((clip / "classified.json").read_text(encoding="utf-8"))
    fps = cl.get("fps") or 60.0
    shots = sorted(cl["shots"], key=lambda s: int(s["frame"]))
    truth = sorted((s for s in known(clip)["shots"]
                    if not s.get("not_a_shot") and s.get("type") in TYPES),
                   key=lambda s: float(s["t_sec"]))
    # Where the shot LANDED, taken from the bounce list rather than impact_court_xy_ft.
    # That field is the ball's position at the PADDLE projected through the ground
    # homography, and the ball is in the air there -- only 35% of those values land within
    # 5 ft of the court at all, against 99% for a bounce, which really is on the ground.
    # The bounce must also be on the FAR side of the net: a shot that crossed has to land
    # there, and "the next bounce" alone picks up the hitter's own side and reads negative.
    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    net = float(court["court_geometry_feet"]["length_ft"]) / 2.0
    bpath = clip / "bounces.json"
    bounces = (sorted(json.loads(bpath.read_text(encoding="utf-8"))["bounces"],
                      key=lambda b: int(b["frame"])) if bpath.exists() else [])

    def landing(shot):
        f, side = int(shot["frame"]), shot.get("hitter_side")
        if side not in ("near", "far"):
            return None
        for b in bounces:
            bf = int(b["frame"])
            if not (0 < bf - f <= LANDING_S * fps):
                continue
            xy = b.get("court_xy_ft")
            if not xy or xy[1] is None:
                continue
            y = float(xy[1])
            if (y > net) if side == "near" else (y < net):
                return (y - net) if side == "near" else (net - y)
        return None

    rows = []
    for tr, s in pair_up(truth, shots, fps):
        got = None if s is None else (s.get("shot_type") or "").strip().lower()
        rows.append({"t": float(tr["t_sec"]), "want": tr["type"], "got": got,
                     "is_volley": None if s is None else bool(s.get("is_volley")),
                     "depth": None if s is None else landing(s),
                     "clip": clip.name})
    return {"clip": clip.name, "fps": fps, "rows": rows}


def main(clips=None) -> int:
    from tools.serve_score import CLIPS as DEFAULT
    rows = []
    for name in (clips or DEFAULT):
        clip = Path("data") / name
        if not (clip / "classified.json").exists():
            print(f"  {name}: not analysed, skipping")
            continue
        rows += score_clip(clip)["rows"]
    if not rows:
        print("no analysed clips with shot-type truth")
        return 1

    matched = [r for r in rows if r["got"] is not None]
    unmatched = len(rows) - len(matched)
    typed = [r for r in matched if r["got"]]
    right = sum(1 for r in typed if r["got"] == r["want"])

    print(f"{len(rows)} labelled shots; {unmatched} had no detection to compare against")
    print(f"  type correct: {right}/{len(typed)}  ({right/len(typed):.0%})")

    # non-serve only: the serve type is decided structurally, so it flatters the classifier
    ns = [r for r in typed if r["want"] != "serve"]
    ns_right = sum(1 for r in ns if r["got"] == r["want"])
    print(f"  excluding serves: {ns_right}/{len(ns)}  ({ns_right/len(ns):.0%})")

    print("\n  confusion — rows are what the operator said, columns what we said:")
    got_types = sorted({r["got"] for r in typed})
    w = max(8, *(len(g) + 1 for g in got_types))
    print(f"    {'':<9}" + "".join(f"{g:>{w}}" for g in got_types) + f"{'total':>8}")
    for want in TYPES:
        sel = [r for r in typed if r["want"] == want]
        if not sel:
            continue
        c = Counter(r["got"] for r in sel)
        line = "".join(f"{c.get(g, 0) or '':>{w}}" for g in got_types)
        ok = c.get(want, 0)
        print(f"    {want:<9}" + line + f"{len(sel):>8}   recall {ok/len(sel):>4.0%}")

    print("\n  and the other way — when we SAY a type, how often is it right:")
    for g in got_types:
        sel = [r for r in typed if r["got"] == g]
        ok = sum(1 for r in sel if r["want"] == g)
        print(f"    we say {g:<9} {ok:>3}/{len(sel):<4} {ok/len(sel):>4.0%}")

    print("\n  where the ball LANDED, by what the operator called the shot:")
    print("     (feet past the net; the kitchen line is 7 ft, the far baseline 22 ft)")
    for want in ("dink", "drop", "drive", "lob", "return", "serve"):
        sel = [r for r in typed if r["want"] == want]
        if not sel:
            continue
        v = sorted(r["depth"] for r in sel if r.get("depth") is not None)
        if v:
            print(f"    {want:<8} landing found {len(v):>3}/{len(sel):<3} "
                  f"({len(v)/len(sel):>3.0%})   median {v[len(v)//2]:>5.1f} ft   "
                  f"p25 {v[len(v)//4]:>5.1f}  p75 {v[3*len(v)//4]:>5.1f}")
        else:
            print(f"    {want:<8} landing found   0/{len(sel):<3}")
    print("    Landing depth separates where it exists -- drop and dink land at the kitchen,")
    print("    drive and return land deep -- while speed does not. COVERAGE is the limit,")
    print("    and it is worst exactly where it is needed most.")

    print("\n  the confusions that cost the most:")
    conf = Counter(f'{r["want"]} -> {r["got"]}' for r in typed if r["got"] != r["want"])
    for k, n in conf.most_common(8):
        print(f"    {n:>3}   {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))

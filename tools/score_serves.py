"""Score SERVE detection against operator truth, on every clip that has it.

Rally segmentation is the fault behind the wrong per-player counts (see the indoor truth:
shots 88 vs 82 is fine, but 5 of 8 rallies span 2-3 real points). Rally END detection is
recorded in KNOWN_ISSUES as attempted and failed six ways, so the productive inversion is
to anchor rallies on their START: a rally is serve -> next serve.

That makes serve detection the thing to measure, and both labelled clips have serve truth:

    pb_5_minute_outdoor-2   14 serve times, docs/ACCURACY_LEDGER.md (operator-verified)
    pb_3_min_indoor_1_court_b   10 point starts, data/<clip>/truth.json

Scored as a matching problem, not a count: a detected serve counts only if it lands within
TOL seconds of a truth serve, and each truth serve can be claimed once. Counting alone
hides the case that actually bites -- the right NUMBER of serves in the wrong PLACES,
which is what produces mid-point "serves" and the wrong server.

Usage:
    python -m tools.score_serves data/pb_5_minute_outdoor-7 data/pb_3_min_indoor_1_court_b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL_S = 3.0

# Operator-verified serve times for the acceptance clip (docs/ACCURACY_LEDGER.md).
# Any session whose video is "PB 5 minute outdoor" shares them.
OUTDOOR_SERVES = [3, 33, 47, 64, 76, 93, 128, 152, 165, 186, 219, 244, 283, 301]


def truth_serves(clip: Path) -> list[float] | None:
    t = clip / "truth.json"
    if t.exists():
        doc = json.loads(t.read_text(encoding="utf-8"))
        return [float(p["start_t_sec"]) for p in doc.get("points", [])]
    try:
        sess = json.loads((clip / "session.json").read_text(encoding="utf-8"))
        if "5 minute outdoor" in str(sess.get("video_path", "")).lower():
            return [float(x) for x in OUTDOOR_SERVES]
    except (OSError, json.JSONDecodeError):
        pass
    return None


def detected_serves(clip: Path) -> list[float]:
    """Rally starts — what the pipeline currently treats as a serve."""
    ra = json.loads((clip / "rallies.json").read_text(encoding="utf-8"))["rallies"]
    return [float(r["start_t_sec"]) for r in ra]


def match(truth: list[float], got: list[float], tol: float = TOL_S) -> dict:
    """Greedy nearest matching, each truth serve claimable once."""
    pairs = sorted(((abs(g - t), i, j) for i, t in enumerate(truth)
                    for j, g in enumerate(got) if abs(g - t) <= tol))
    used_t, used_g, hits = set(), set(), []
    for d, i, j in pairs:
        if i in used_t or j in used_g:
            continue
        used_t.add(i); used_g.add(j); hits.append((truth[i], got[j], d))
    return {"hits": hits,
            "missed": [t for i, t in enumerate(truth) if i not in used_t],
            "false": [g for j, g in enumerate(got) if j not in used_g]}


def clock(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def strike_times(clip: Path) -> list[float] | None:
    """Operator-marked PADDLE-STRIKE times, if any (tools/mark_serve_strikes.py).

    Recall and precision above are matched against the serve times the operator typed while
    watching, and those are not the strike -- measured, they sit a median 1.06s early on
    court B and 0.32s early outdoors, and the bias differs per clip because the convention
    varied. A +/-3s match window absorbs that, so recall/precision are unaffected. TIMING is
    not: it can only be scored against a marked strike, which is what this reads.
    """
    p = clip / "serve_strikes.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    out = [float(s["t_sec"]) for s in doc.get("strikes", []) if not s.get("skipped")]
    return sorted(out) or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--tol", type=float, default=TOL_S)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    print(f"Serve detection vs operator truth (match window +/-{a.tol:g}s)")
    print()
    print(f"{'clip':<28}{'truth':>6}{'found':>6}{'hit':>5}{'miss':>6}{'false':>6}"
          f"{'recall':>8}{'prec':>7}{'|t-strike|':>12}{'n':>4}")
    rc = 0
    for c in a.clips:
        t = truth_serves(c)
        if t is None:
            print(f"{c.name:<28} no truth available")
            continue
        g = detected_serves(c)
        m = match(t, g, a.tol)
        h, mi, fa = len(m["hits"]), len(m["missed"]), len(m["false"])
        rec = h / len(t) if t else 0.0
        pre = h / len(g) if g else 0.0
        st = strike_times(c)
        errs = []
        if st:
            sm = match(st, g, a.tol)
            errs = sorted(abs(got - tr) for tr, got, _ in sm["hits"])
        med = f"{errs[len(errs) // 2]:.2f}s" if errs else "-"
        print(f"{c.name:<28}{len(t):>6}{len(g):>6}{h:>5}{mi:>6}{fa:>6}{rec:>7.0%}{pre:>7.0%}"
              f"{med:>12}{(len(errs) if errs else '-'):>4}")
        if a.verbose:
            if m["missed"]:
                print(f"    missed serves : {', '.join(clock(x) for x in m['missed'])}")
            if m["false"]:
                print(f"    false starts  : {', '.join(clock(x) for x in m['false'])}")
            if errs:
                print(f"    vs marked strikes: median {errs[len(errs) // 2]:.2f}s, "
                      f"worst {errs[-1]:.2f}s, "
                      f"{sum(1 for e in errs if e <= 0.5)}/{len(errs)} within 0.5s")
        if mi or fa:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

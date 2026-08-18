"""Score shots against per-RALLY operator truth, without needing per-shot labels.

The outdoor clip has a per-shot review; the indoor clip has something different and cheaper
to produce — the operator listed every point with its start, end, server and shot count
(data/<clip>/truth.json, from _truth_worksheet.csv). That is enough to measure two things
directly, with no further labelling:

  * BETWEEN-POINT false positives. A detected shot outside every rally window is, by the
    operator's own account of when points were live, not a shot in play. No per-shot review
    can add anything to that judgement.
  * WHERE the shot count is wrong. A per-rally delta localises over- and under-detection to
    a 10-second window, which a total count cannot: the outdoor clip held 34 non-shots AND
    17 missed shots at 125 detected, so a total near truth can hide large errors in both
    directions at once.

TOL_S widens each window because the operator's start/end times are hand-typed off a
stopwatch, and a serve contact a beat before their noted start is still that rally's serve.

Usage:
    python -m tools.score_rally_shots data/pb_3_min_indoor_1_court_b
    python -m tools.score_rally_shots data/pb_3_min_indoor_1_court_b --verbose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL_S = 1.0


def load(clip: Path) -> tuple[list[dict], list[dict]]:
    shots = json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"]
    truth = json.loads((clip / "truth.json").read_text(encoding="utf-8"))
    return shots, truth.get("points", [])


def score(shots: list[dict], points: list[dict], tol: float = TOL_S) -> dict:
    rows = []
    claimed: set[int] = set()
    for p in points:
        lo, hi = float(p["start_t_sec"]) - tol, float(p["end_t_sec"]) + tol
        inside = [i for i, s in enumerate(shots) if lo <= s["t_sec"] <= hi]
        claimed.update(inside)
        rows.append({"point": p.get("point"), "start": p["start_t_sec"],
                     "end": p["end_t_sec"], "server": p.get("server"),
                     "truth": int(p["n_shots"]), "got": len(inside),
                     "delta": len(inside) - int(p["n_shots"])})
    outside = [i for i in range(len(shots)) if i not in claimed]
    return {"rows": rows, "outside": outside,
            "n_truth": sum(r["truth"] for r in rows),
            "n_in_rallies": sum(r["got"] for r in rows),
            "n_detected": len(shots)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--tol", type=float, default=TOL_S)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    shots, points = load(a.clip)
    if not points:
        print("no per-point truth in truth.json")
        return 1
    r = score(shots, points, a.tol)

    print(f"{a.clip.name}: {r['n_detected']} shots detected, "
          f"operator truth {r['n_truth']} in {len(points)} rallies")
    print()
    print(f"  {'rally':>6}{'window':>18}{'server':>9}{'truth':>7}{'got':>6}{'delta':>7}")
    for row in r["rows"]:
        mark = "" if row["delta"] == 0 else ("  <-- over" if row["delta"] > 0 else "  <-- UNDER")
        print(f"  {str(row['point']):>6}{f'{row['start']:.0f}-{row['end']:.0f}s':>18}"
              f"{str(row['server']):>9}{row['truth']:>7}{row['got']:>6}{row['delta']:>+7}{mark}")
    miss = sum(-x["delta"] for x in r["rows"] if x["delta"] < 0)
    over = sum(x["delta"] for x in r["rows"] if x["delta"] > 0)
    print(f"  {'':>6}{'':>18}{'TOTAL':>9}{r['n_truth']:>7}{r['n_in_rallies']:>6}"
          f"{r['n_in_rallies'] - r['n_truth']:>+7}")
    print()
    print(f"  shots inside rallies : {r['n_in_rallies']}  "
          f"({over} more than truth in some rallies, {miss} fewer in others)")
    print(f"  shots OUTSIDE every rally window: {len(r['outside'])}"
          f"   <-- between-point false positives, by the operator's own point boundaries")
    if a.verbose and r["outside"]:
        for i in r["outside"]:
            s = shots[i]
            print(f"      {s['t_sec']:8.2f}s  {s['shot_type']:<7} "
                  f"serve={str(s.get('is_serve')):<5} between={s.get('is_between_point')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

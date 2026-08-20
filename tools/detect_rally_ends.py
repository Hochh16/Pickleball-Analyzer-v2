"""Detect how each point ENDED, using the operator's own taxonomy.

Recorded in KNOWN_ISSUES as "RALLY END is undetectable" after six failed routes. All six
failed for want of the ball's height, which `tools/build_ball_3d.py` now supplies. The
operator stated the rules:

    hit into the net              -> the hitter loses the point
    ball bounces outside the court -> the hitter loses the point
    bounces in and is not returned -> the hitter WINS the point

Each is a different measurement, and two of the three were already within reach — what was
missing is the first.

**Net hit = SUSTAINED z≈0 near the net, not z≈0.** Ordinary play reaches z=0 on every bounce,
so an instantaneous test fires constantly. A bounce touches the floor for a frame or two; a
ball that hit the net drops and STAYS on the floor beside it. Measured at the operator's
19.45s net hit: the ball falls from 4.33 ft and holds z=0.00 for over two seconds, 0.1-3 ft
from the net line.

`LOW_Z_FT` is 1.0 ft rather than something tighter because the reconstruction's absolute
height error is 0.3-0.8 ft (see the calibration note in KNOWN_ISSUES). The signature survives
that slack precisely because it is about DURATION, not a precise height.

Usage:
    python -m tools.detect_rally_ends data/pb_3_min_indoor_1_court_b
    python -m tools.detect_rally_ends data/pb_3_min_indoor_1_court_b --score
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_LOOK_S = 6.0        # how far past a contact to look for its outcome
LOW_Z_FT = 1.0          # at or below this height counts as "on the floor"
SUSTAIN_S = 0.5         # ...for at least this long = the ball is DEAD, not bouncing
NET_BAND_FT = 5.0       # within this of the net line = "at the net"
OUT_MARGIN_FT = 1.0     # tolerance outside the lines before calling a bounce OUT
NOT_RETURNED_S = 2.0    # no contact this long after a bounce = nobody played it
SCORE_TOL_S = 2.5       # match window when scoring against operator truth


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] index pairs of consecutive True runs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def detect(clip: Path) -> list[dict]:
    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    geom = court["court_geometry_feet"]
    W, L = float(geom["width_ft"]), float(geom["length_ft"])
    net_y = L / 2.0

    shots = sorted(json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"],
                   key=lambda s: s["t_sec"])
    b3 = pd.read_parquet(clip / "ball_3d.parquet").sort_values("t_sec")

    bdoc = json.loads((clip / "bounces.json").read_text(encoding="utf-8"))
    bkey = next(k for k in bdoc if isinstance(bdoc[k], list))
    bounce_t = sorted(float(b.get("t_sec", 0)) for b in bdoc[bkey])

    ends = []
    for i, s in enumerate(shots):
        t0 = float(s["t_sec"])
        t_next = float(shots[i + 1]["t_sec"]) if i + 1 < len(shots) else np.inf
        w = b3[(b3.t_sec > t0) & (b3.t_sec <= t0 + MAX_LOOK_S)]
        if w.empty:
            continue

        # --- 1. NET HIT: the ball goes to the floor beside the net and stays there -----
        near_net = (w.z_ft <= LOW_Z_FT) & ((w.court_y_ft - net_y).abs() <= NET_BAND_FT)
        m = near_net.to_numpy()
        ts = w.t_sec.to_numpy()
        hit_net = None
        for a, z in _runs(m):
            if ts[z] - ts[a] >= SUSTAIN_S:
                hit_net = float(ts[a])
                break
        if hit_net is not None and hit_net < t_next + SUSTAIN_S:
            ends.append({"t_sec": round(hit_net, 2), "reason": "net",
                         "by_shot_t": round(t0, 2),
                         "hitter_side": s.get("hitter_side"),
                         "hitter_is_user": bool(s.get("is_user")),
                         "outcome": "hitter_loses"})
            continue

        # --- 2/3. the next BOUNCE after this contact ---------------------------------
        nb = next((b for b in bounce_t if t0 < b <= t0 + MAX_LOOK_S), None)
        if nb is None:
            continue
        row = b3.iloc[(b3.t_sec - nb).abs().argsort()[:1]]
        if row.empty:
            continue
        bx = float(row.court_x_ft.iloc[0])
        by = float(row.court_y_ft.iloc[0])
        out = (bx < -OUT_MARGIN_FT or bx > W + OUT_MARGIN_FT
               or by < -OUT_MARGIN_FT or by > L + OUT_MARGIN_FT)
        if out:
            ends.append({"t_sec": round(nb, 2), "reason": "out",
                         "by_shot_t": round(t0, 2),
                         "hitter_side": s.get("hitter_side"),
                         "hitter_is_user": bool(s.get("is_user")),
                         "bounce_xy_ft": [round(bx, 1), round(by, 1)],
                         "outcome": "hitter_loses"})
            continue
        # bounced IN — did anyone play it?
        if t_next - nb >= NOT_RETURNED_S:
            ends.append({"t_sec": round(nb, 2), "reason": "not-returned",
                         "by_shot_t": round(t0, 2),
                         "hitter_side": s.get("hitter_side"),
                         "hitter_is_user": bool(s.get("is_user")),
                         "bounce_xy_ft": [round(bx, 1), round(by, 1)],
                         "outcome": "hitter_wins"})

    # one END per point: collapse anything within NOT_RETURNED_S of the previous one
    ends.sort(key=lambda e: e["t_sec"])
    merged: list[dict] = []
    for e in ends:
        if merged and e["t_sec"] - merged[-1]["t_sec"] < NOT_RETURNED_S:
            continue
        merged.append(e)
    return merged


def score(clip: Path, ends: list[dict], tol: float = SCORE_TOL_S) -> dict | None:
    tp = clip / "truth.json"
    if not tp.exists():
        return None
    points = json.loads(tp.read_text(encoding="utf-8")).get("points", [])
    truth = [float(p["end_t_sec"]) for p in points]
    got = [e["t_sec"] for e in ends]
    pairs = sorted((abs(g - t), i, j) for i, t in enumerate(truth)
                   for j, g in enumerate(got) if abs(g - t) <= tol)
    ut, ug = set(), set()
    for _, i, j in pairs:
        if i in ut or j in ug:
            continue
        ut.add(i)
        ug.add(j)
    return {"truth": len(truth), "found": len(got), "hit": len(ut),
            "missed": [t for i, t in enumerate(truth) if i not in ut],
            "false": [g for j, g in enumerate(got) if j not in ug]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    ends = detect(a.clip)
    from collections import Counter
    print(f"{a.clip.name}: {len(ends)} point-ends detected")
    for r, n in Counter(e["reason"] for e in ends).most_common():
        print(f"    {r:<14}{n}")
    for e in ends:
        print(f"  {e['t_sec']:8.2f}s  {e['reason']:<13} after a {e['hitter_side']}-side "
              f"shot at {e['by_shot_t']:.2f}s  -> {e['outcome']}")
    if a.write:
        (a.clip / "rally_ends.json").write_text(
            json.dumps({"schema_version": 1, "ends": ends}, indent=1), encoding="utf-8")
        print(f"  wrote {a.clip / 'rally_ends.json'}")
    if a.score:
        s = score(a.clip, ends)
        if s is None:
            print("  no truth.json to score against")
        else:
            rec = s["hit"] / s["truth"] if s["truth"] else 0
            pre = s["hit"] / s["found"] if s["found"] else 0
            print(f"\n  vs operator truth: {s['hit']}/{s['truth']} ends found "
                  f"(recall {rec:.0%}, precision {pre:.0%})")
            if s["missed"]:
                print(f"    missed: {', '.join(f'{t:.0f}s' for t in s['missed'])}")
            if s["false"]:
                print(f"    false : {', '.join(f'{t:.0f}s' for t in s['false'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

Two things the first version got wrong, both worth keeping in mind:

* **A dead ball is a statistical state, not a clean one.** Requiring an unbroken run of
  in-band samples rejected the operator's 19.45s net hit outright — the ball is plainly dead
  on the floor, yet its reconstructed court_y wanders 18.6-30.2 ft and z crosses 1.0 ft
  repeatedly. `_dead_start` asks for a FRACTION of the window instead.
* **Low near the net is not sufficient** — a kitchen dink exchange puts the ball there over
  and over. A dead ball also stops TRAVELLING, hence `DEAD_TRAVEL_FT`.

MEASURED 2026-08-20:

| | recall | precision |
|---|---|---|
| indoor point-ends (operator truth, 10) | **9/10** | 9/14 |
| outdoor net hits (operator-confirmed, 7) | **7/7** | 7/10 |

Recall is strong; precision is the open side — it still calls roughly 1.4 ends for every real
one. Note both figures come from small samples (10 points, 7 net hits) and several thresholds
were swept against them, so treat the exact numbers as provisional until a third clip is
labelled.

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
NET_BAND_FT = 8.0       # within this of the net line = "at the net"
DEAD_FRAC = 0.65        # share of samples in the window that must be low+near-net
DEAD_TRAVEL_FT = 6.0    # a DEAD ball barely moves; a dink crosses the net at speed
OUT_MARGIN_FT = 1.0     # tolerance outside the lines before calling a bounce OUT
NOT_RETURNED_S = 2.0    # no contact this long after a bounce = nobody played it
SCORE_TOL_S = 2.5       # match window when scoring against operator truth


def _dead_start(mask: np.ndarray, ts: np.ndarray, sustain_s: float, frac: float,
                xs: np.ndarray | None = None, ys: np.ndarray | None = None,
                max_travel_ft: float | None = None) -> float | None:
    """First time at which the ball is DEAD: mostly low and near the net, and stays that way.

    Deliberately NOT a contiguous run. The reconstruction flickers — at the operator's
    19.45s net hit the ball is unmistakably dead on the floor beside the net, yet its
    reconstructed court_y wanders between 18.6 and 30.2 ft and z crosses 1.0 ft repeatedly.
    Requiring an unbroken run of in-band samples rejected that outright, and with it the
    three earliest net hits on the clip.

    A dead ball is a statistical state, not a clean one: over any window of `sustain_s`,
    at least `frac` of the samples sit low and near the net.
    """
    n = len(mask)
    if n == 0:
        return None
    for a in range(n):
        end_t = ts[a] + sustain_s
        if ts[-1] < end_t:
            break                                  # not enough data left to judge
        b = int(np.searchsorted(ts, end_t, side="right"))
        w = mask[a:b]
        if not len(w) or w.mean() < frac:
            continue
        if max_travel_ft is not None and xs is not None and ys is not None:
            span = max(float(np.ptp(xs[a:b])), float(np.ptp(ys[a:b])))
            if span > max_travel_ft:
                continue                           # still travelling: a rally, not a dead ball
        return float(ts[a])
    return None


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

        # --- 1. NET HIT: the ball goes to the floor beside the net and STAYS there -----
        # Being low near the net is not enough on its own: a kitchen dink exchange puts the
        # ball there over and over. A dead ball also stops TRAVELLING, so require the court
        # position to be near-stationary over the window as well.
        near_net = (w.z_ft <= LOW_Z_FT) & ((w.court_y_ft - net_y).abs() <= NET_BAND_FT)
        hit_net = _dead_start(near_net.to_numpy(), w.t_sec.to_numpy(),
                              SUSTAIN_S, DEAD_FRAC,
                              w.court_x_ft.to_numpy(), w.court_y_ft.to_numpy(),
                              DEAD_TRAVEL_FT)
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

    # A point cannot end before the ball has been played ACROSS the net. Operator review
    # of the flagged false ends settled this: the call at 127.24s lands on a serve by the
    # partner ("1 shot since the serve, sides ['near']") — the ball had not yet crossed, so
    # nothing could have ended. The call at 155.66s looks identical on timing (2.2s after a
    # serve vs 1.96s) but has sides ['far', 'near'], a real exchange, and the operator
    # confirms it IS the end of that point. Timing cannot separate those two; a side change
    # can, and it is a rule of the game rather than a tuned threshold.
    def _crossed_since(anchor_t: float, end_t: float) -> bool:
        sides = [s.get("hitter_side") for s in shots
                 if anchor_t <= s["t_sec"] <= end_t and s.get("hitter_side")]
        return len(set(sides)) > 1

    serve_t = [float(s["t_sec"]) for s in shots if s.get("is_serve")]
    kept_ends = []
    for e in ends:
        # A NET end is self-evidencing — the ball demonstrably died at the net — and a serve
        # INTO the net is a legitimate point end with no side change at all. Requiring one
        # here cost a confirmed outdoor net hit. The rule belongs only on `out` and
        # `not-returned`, which rest on an inferred bounce position rather than on the ball
        # visibly going dead.
        if e["reason"] != "net":
            prior = [x for x in serve_t if x <= e["t_sec"]]
            if prior and not _crossed_since(prior[-1], e["t_sec"]):
                continue                # ball never left the serving side: not an end
        kept_ends.append(e)
    ends = kept_ends

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

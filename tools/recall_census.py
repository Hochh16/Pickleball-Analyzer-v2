"""Measure the CEILING: is the ball detectable where we know real events happened?

The decisive question before any redesign. Every count in the report is derived
from the ball track, so the ball's detectability at contact frames bounds what ANY
architecture can achieve. If the ball is present at real events but we produce no
shot, that is an ALGORITHM problem and restructuring the pipeline can fix it. If the
ball is absent, that is a DETECTOR/CAPTURE limit and restructuring cannot.

The 2026-08-01 session lost a day to fixes that each broke another stage; the design
review that followed needs this number as its input, or it is speculation.

    python tools/recall_census.py --clip data/pb_5_minute_outdoor-2

Ground truth used: the operator's 14 confirmed serve timestamps (docs/ACCURACY_LEDGER,
corrected 2026-08-01). Serves are the ideal probe — the operator has confirmed every
one, they are unambiguous moments, and a serve is the single most important frame in a
point (it anchors the rally).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Operator-confirmed serve times (seconds). ACCURACY_LEDGER, corrected 2026-08-01:
# all 14 are real serves; 1:29 is really 1:33 and 4:58 is really 5:01.
OPERATOR_SERVES_S = [3, 33, 47, 64, 76, 93, 128, 152, 165, 186, 219, 245, 283, 301]
TRUTH_CLIP = "pb_5_minute_outdoor-2"


def fmt(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--window", type=float, default=1.0,
                    help="+/- seconds around a known event to search")
    args = ap.parse_args()
    D = Path(args.clip)

    ball = pd.read_parquet(D / "ball.parquet").set_index("frame_idx")
    shots = json.load(open(D / "shots.json", encoding="utf-8"))
    fps = float(shots.get("fps") or 60.0)
    det = sorted(s["frame"] / fps for s in shots["shots"])
    det_serves = sorted(s["frame"] / fps for s in shots["shots"] if s.get("is_serve"))
    W = int(round(args.window * fps))

    vis = ball["visible"].to_numpy()
    interp = ball["interpolated"].to_numpy()
    n = len(ball)

    print(f"\nRECALL CENSUS — {D.name}")
    print(f"  {n} frames @ {fps:g}fps | ball VISIBLE {vis.mean():.1%} | "
          f"interpolated {interp.mean():.1%} | no signal {(~vis & ~interp).mean():.1%}")

    if D.name != TRUTH_CLIP:
        print("  (no operator serve truth for this clip — coverage stats only)\n")
        return

    print(f"\nAt the operator's {len(OPERATOR_SERVES_S)} confirmed serves "
          f"(+/-{args.window:g}s window):\n")
    print(f"  {'serve':>7s}  {'ball visible':>12s}  {'best conf':>9s}  "
          f"{'shot detected':>13s}  {'flagged serve':>13s}   verdict")
    cats = {"ok": 0, "algo": 0, "detector": 0}
    for t in OPERATOR_SERVES_S:
        f0 = int(round(t * fps))
        lo, hi = max(0, f0 - W), min(n - 1, f0 + W)
        seg = ball.loc[lo:hi]
        nvis = int(seg["visible"].sum())
        conf = seg.loc[seg["visible"], "confidence"]
        best = float(conf.max()) if len(conf) else float("nan")
        has_shot = any(abs(d - t) <= args.window for d in det)
        has_serve = any(abs(d - t) <= args.window for d in det_serves)

        if not nvis:
            verdict, key = "NO BALL SIGNAL -> detector/capture limit", "detector"
        elif not has_shot:
            verdict, key = "ball present, NO SHOT -> algorithm", "algo"
        elif not has_serve:
            verdict, key = "shot found, not flagged serve -> algorithm", "algo"
        else:
            verdict, key = "ok", "ok"
        cats[key] += 1
        print(f"  {fmt(t):>7s}  {nvis:>4d}/{hi - lo + 1:<7d}  "
              f"{best:>9.2f}  {str(has_shot):>13s}  {str(has_serve):>13s}   {verdict}")

    tot = sum(cats.values())
    print(f"\n  correct                       {cats['ok']:2d}/{tot}")
    print(f"  ALGORITHM-limited (ball there) {cats['algo']:2d}/{tot}  <- a redesign can fix these")
    print(f"  DETECTOR-limited (no ball)     {cats['detector']:2d}/{tot}  <- no architecture fixes these")

    # Ball continuity: long gaps are where shots vanish regardless of architecture.
    runs, cur = [], 0
    for v in vis:
        if v:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    runs = np.array(runs) if runs else np.array([0])
    print(f"\nInvisible-ball gaps: {len(runs)} gaps | median {np.median(runs):.0f}f "
          f"({np.median(runs) / fps:.2f}s) | p90 {np.percentile(runs, 90):.0f}f | "
          f"max {runs.max():.0f}f ({runs.max() / fps:.1f}s)")
    print(f"  gaps >= 0.5s: {(runs >= fps * 0.5).sum()}  (a contact inside one of "
          f"these is unrecoverable from the ball track alone)\n")


if __name__ == "__main__":
    main()

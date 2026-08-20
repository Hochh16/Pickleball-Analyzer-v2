"""Detect floor contacts from the reconstructed ball HEIGHT.

The existing Stage 5.5 detector works in pixel space (a y-flip heuristic), which predates
having any height at all. With `ball_3d.parquet` a floor contact has a direct definition: a
local minimum of z at the floor, which the ball ARRIVES at from height and LEAVES to height.

**A correction worth recording.** This was first judged a failure because it produced 107-133
events on a 3-minute clip against "59 shots". That comparison was wrong, as the operator
pointed out: 59 is the count of IN-PLAY shots, and most floor contacts are not in play at all
— between points the ball is dropped, bounced before serving, and rolled back repeatedly. Nor
does every shot bounce (volleys do not).

The correct test is the physical bound. A pickleball may bounce **at most once between
shots**, so inside a rally the number of floor contacts cannot exceed the number of shots:

| clip | in-play shots | height contacts in play | current detector |
|---|---|---|---|
| court C | 59 | **45** | 28 |
| court B | 82 | **62** | 34 |

Both sit under the bound while finding ~1.8x more in-play contacts than the pixel-space
detector, and each clip yields another 54-87 between-point contacts that the operator never
counted.

CAVEAT, measured: the height track fits projectile motion only loosely — over single flight
arcs the recovered gravity has a median of 12.3 ft/s² against a true 32.2, with 24% of arcs
within a factor of two, and an RMS residual of 0.59 ft. So these are contacts located by a
noisy signal, and their TIMING should be treated as approximate.

Usage:
    python -m tools.bounces_from_height data/pb_3_min_indoor_1_court_c
    python -m tools.bounces_from_height data/pb_3_min_indoor_1_court_c --write
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FLOOR_Z_FT = 0.35       # at or below this counts as touching the floor
ARRIVE_Z_FT = 1.0       # must have been this high shortly before
DEPART_Z_FT = 0.5       # ...and reach this after (a bounce loses energy, so lower)
LOOK_FRAMES = 15        # how far either side to look for those heights
REFRACTORY_S = 0.30     # minimum spacing between contacts


def detect(b3: pd.DataFrame) -> list[dict]:
    b3 = b3.sort_values("t_sec").reset_index(drop=True)
    t = b3.t_sec.to_numpy()
    z = b3.z_ft.to_numpy()
    x = b3.court_x_ft.to_numpy()
    y = b3.court_y_ft.to_numpy()
    out: list[dict] = []
    for i in range(LOOK_FRAMES, len(z) - LOOK_FRAMES):
        if z[i] > FLOOR_Z_FT or not (z[i] <= z[i - 1] and z[i] <= z[i + 1]):
            continue
        if z[max(0, i - LOOK_FRAMES):i].max() < ARRIVE_Z_FT:
            continue
        if z[i + 1:i + 1 + LOOK_FRAMES].max() < DEPART_Z_FT:
            continue
        if out and t[i] - out[-1]["t_sec"] <= REFRACTORY_S:
            continue
        out.append({"t_sec": round(float(t[i]), 3),
                    "frame": int(b3.frame.iloc[i]),
                    "court_xy_ft": [round(float(x[i]), 1), round(float(y[i]), 1)],
                    "z_ft": round(float(z[i]), 2),
                    "source": "height"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--write", action="store_true",
                    help="write bounces_height.json (does NOT overwrite bounces.json, which "
                         "the ball-3D calibration depends on)")
    a = ap.parse_args(argv)

    b3 = pd.read_parquet(a.clip / "ball_3d.parquet")
    ev = detect(b3)
    print(f"{a.clip.name}: {len(ev)} floor contacts from height")

    tp = a.clip / "truth.json"
    if tp.exists():
        pts = json.loads(tp.read_text(encoding="utf-8")).get("points", [])
        if pts:
            def in_play(x):
                return any(float(p["start_t_sec"]) <= x <= float(p["end_t_sec"]) for p in pts)
            n_in = sum(1 for e in ev if in_play(e["t_sec"]))
            shots = sum(int(p["n_shots"]) for p in pts)
            print(f"  in play {n_in} vs {shots} shots  "
                  f"({'within' if n_in <= shots else 'EXCEEDS'} the one-bounce-per-shot bound)")
            print(f"  between points {len(ev) - n_in}")
    if a.write:
        out = a.clip / "bounces_height.json"
        out.write_text(json.dumps({"schema_version": 1, "bounces": ev}, indent=1),
                       encoding="utf-8")
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

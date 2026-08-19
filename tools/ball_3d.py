"""Recover the ball's COURT POSITION and HEIGHT from a single camera.

The pipeline's oldest limitation is that a pixel is a ray, not a point: a ball position has
3 unknowns and a pixel gives 2 equations. The ground homography supplies the third by
assuming z=0 — exact at a bounce, nonsense in the air. Three separate rules have died on
this (net-line crossing, opposing-player reach, the ground-ball filter), so this attacks the
cause.

Two measurements close the gap, and neither needs anything from the operator beyond the
court calibration already on disk.

1. THE CAMERA'S GROUND POSITION, from player boxes. For a camera at height H and a person
   of height h at ground point P_feet, the head projects through the homography to
   P_head_proj = C + k*(P_feet - C) with k = H/(H-h) > 1. So C, P_feet and P_head_proj are
   COLLINEAR: every player in every frame is a line through the camera, and many lines
   intersect at C. Measured median line residual is 0.0 ft on both clips — the model is
   exact, not approximate. The same k gives H once a person height is assumed.

2. THE BALL'S RANGE, from apparent size. A pickleball is a known 2.9 inches, so its pixel
   diameter fixes distance. Comparing it against the size predicted at the ray's ground
   intersection gives the fraction of the way along the ray the ball actually sits:

       P_true = C + (P_ground - C) / k_ball,     k_ball = measured_px / predicted_px / bias
       z      = H * (1 - 1/k_ball)

   `bias` corrects the blob measurement over-reading (motion blur and halo inflate FWHM) and
   is taken from THIS CLIP'S OWN BOUNCES, where the ball is genuinely at z=0. Nothing is
   tuned per court: px/ft comes from the clip's calibration, C and H from its players, bias
   from its bounces.

NOTE: the reconstructed x,y do NOT depend on the assumed person height — only z in feet
does. Position rests on C and the size ratio alone.

VALIDATED 2026-08-19 (indoor, frames 3600-4400):

    camera height solves to 6.7 ft            (the rig is a ~6 ft camera)
    at bounces, reconstructed z = 0.14 ft     (should be 0)
    in flight, court y: 80.4 ft raw -> 23.0 ft reconstructed (the net is at 22)
    in flight, within the play envelope: 26% raw -> 89% reconstructed

KNOWN WEAKNESS: `bias` is estimated from only 5-20 bounces and moves between 1.59 and 2.03
across clips and windows. It is the least stable part and deserves a wider estimation
window, or a bounce-independent calibration, before anything depends on absolute heights.

Usage:
    python -m tools.ball_3d data/pb_3_min_indoor_1_court_b 3600 4400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tools.measure_ball_size import BALL_FT, measure_diameter, scale_map

PERSON_FT = 5.6          # assumed standing height; scales z only, never x/y
DEFAULT_BIAS = 1.7       # fallback when a clip has too few bounces to calibrate


def camera_ground(clip: Path, stride: int = 4):
    """(C, H, k) — camera ground position in court ft, its height, median player k."""
    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    M = np.array(court["homography"]["image_to_court"], float)

    def to_court(u, v):
        p = M @ np.stack([u, v, np.ones_like(u)])
        return p[0] / p[2], p[1] / p[2]

    pl = pd.read_parquet(clip / "players.parquet")
    pl = pl[(~pl.transient) & pl.court_pos_reliable].iloc[::stride]
    HX, HY = to_court(((pl.bbox_x1 + pl.bbox_x2) / 2).to_numpy(), pl.bbox_y1.to_numpy())
    FX, FY = pl.court_x_ft.to_numpy(), pl.court_y_ft.to_numpy()

    d = np.stack([HX - FX, HY - FY], 1)
    n = np.linalg.norm(d, axis=1)
    ok = np.isfinite(n) & (n > 1.0) & np.isfinite(FX) & np.isfinite(FY)
    if ok.sum() < 20:
        raise ValueError("not enough usable player observations to locate the camera")
    d, P = d[ok] / n[ok, None], np.stack([FX[ok], FY[ok]], 1)

    A, b = np.zeros((2, 2)), np.zeros(2)
    for di, Pi in zip(d, P):
        Q = np.eye(2) - np.outer(di, di)     # project onto the line's normal
        A += Q
        b += Q @ Pi
    C = np.linalg.solve(A, b)

    k = (np.linalg.norm(np.stack([HX, HY], 1)[ok] - C, axis=1)
         / np.linalg.norm(P - C, axis=1))
    k = k[np.isfinite(k) & (k > 1.01) & (k < 10)]
    kk = float(np.median(k)) if len(k) else 1.5
    return C, PERSON_FT / (1.0 - 1.0 / kk), kk


def reconstruct(clip: Path, f_lo: int, f_hi: int, bias: float | None = None):
    """Per-frame ball court position and height over [f_lo, f_hi]."""
    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    to_court, px_per_ft = scale_map(court)
    C, H, _ = camera_ground(clip)

    ball = pd.read_parquet(clip / "ball.parquet")
    ball = ball[ball.visible].set_index("frame_idx")
    bdoc = json.loads((clip / "bounces.json").read_text(encoding="utf-8"))
    bkey = next(k for k in bdoc if isinstance(bdoc[k], list))
    fps = float(court["video"]["fps"])
    bounce_frames = {int(round(float(b.get("t_sec", 0)) * fps)) for b in bdoc[bkey]}

    cap = cv2.VideoCapture(str(clip / "video.mp4"))
    rows, f = [], 0
    while f <= f_hi:
        ok, img = cap.read()
        if not ok:
            break
        if f >= f_lo and f in ball.index:
            u, v = float(ball.loc[f, "pixel_x"]), float(ball.loc[f, "pixel_y"])
            s = px_per_ft(u, v)
            if s:
                pred = BALL_FT * s
                meas = measure_diameter(img, u, v, pred)
                if meas:
                    gx, gy = to_court(u, v)
                    rows.append({"frame": f, "t_sec": f / fps, "ratio": meas / pred,
                                 "ground_x": gx, "ground_y": gy,
                                 "is_bounce": min((abs(f - b) for b in bounce_frames),
                                                  default=999) <= 2})
        f += 1
    cap.release()

    df = pd.DataFrame(rows)
    if df.empty:
        return df, C, H, DEFAULT_BIAS
    if bias is None:
        b = df[df.is_bounce]
        bias = float(np.median(b.ratio)) if len(b) >= 5 else DEFAULT_BIAS
    # k < 1 would put the ball BEYOND its own ground intersection, which is impossible;
    # clamping saturates those readings at z = 0 rather than inventing negative heights.
    df["k"] = (df.ratio / bias).clip(lower=1.0)
    g = np.stack([df.ground_x, df.ground_y], 1)
    t = (g - C) / df.k.to_numpy()[:, None]
    df["court_x_ft"] = C[0] + t[:, 0]
    df["court_y_ft"] = C[1] + t[:, 1]
    df["z_ft"] = H * (1.0 - 1.0 / df.k)
    return df, C, H, bias


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("f_lo", type=int)
    ap.add_argument("f_hi", type=int)
    a = ap.parse_args(argv)

    df, C, H, bias = reconstruct(a.clip, a.f_lo, a.f_hi)
    if df.empty:
        print("no measurements")
        return 1
    L = float(json.loads((a.clip / "court.json").read_text(encoding="utf-8"))
              ["court_geometry_feet"]["length_ft"])
    print(f"{a.clip.name}: camera ({C[0]:.1f}, {C[1]:.1f}) ft at {H:.1f} ft high; "
          f"bias {bias:.2f}; n={len(df)}")
    b, fl = df[df.is_bounce], df[~df.is_bounce]
    inside = lambda y: float(((y >= -15) & (y <= L + 15)).mean())
    if len(b):
        print(f"  CONTROL at bounces : z median {b.z_ft.median():.2f} ft (should be ~0)")
    print(f"  in flight          : court y {fl.ground_y.median():7.1f} ft raw "
          f"-> {fl.court_y_ft.median():.1f} ft reconstructed   (net at {L/2:.0f})")
    print(f"  within play envelope: {inside(fl.ground_y):.0%} raw "
          f"-> {inside(fl.court_y_ft):.0%} reconstructed")
    print(f"  height             : median {fl.z_ft.median():.2f} ft, "
          f"p90 {fl.z_ft.quantile(.9):.2f} ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

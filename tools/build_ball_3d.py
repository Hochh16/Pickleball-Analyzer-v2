"""Reconstruct the ball in 3-D for a WHOLE clip and persist it to ball_3d.parquet.

Everything downstream of ball height — rally-end detection, net hits, shot type from apex,
speed in real units — needs the same reconstruction. Recomputing it means decoding the video
again at roughly ten minutes a pass, which makes each experiment expensive enough to
discourage running it. So it is computed once per clip and written to disk.

ONE PASS. `tools/ball_3d.reconstruct` walked the video twice: once to fit the blob
calibration from bounce frames, once to reconstruct. Both need the same measurement — the
blob size at every visible ball frame — so this measures once, fits the calibration from the
rows that happen to be bounces, and applies it to all of them.

Output columns (one row per frame where the ball was visible AND measurable):

    frame, t_sec        source frame index and time
    pred_px, meas_px    ball diameter predicted at the ray's ground intersection, and measured
    ground_x, ground_y  the z=0 projection — correct only at a bounce, kept for comparison
    k                   how much nearer the ball is than its ground intersection (1.0 = on it)
    court_x_ft, court_y_ft, z_ft    the reconstruction
    is_bounce           within 2 frames of a detected bounce

Usage:
    python -m tools.build_ball_3d data/pb_3_min_indoor_1_court_b
    python -m tools.build_ball_3d data/pb_3_min_indoor_1_court_b --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tools.ball_3d import (CALIB_FILE, DEFAULT_A, DEFAULT_C, MERGE_REJECT_RATIO,
                           camera_ground)
from tools.measure_ball_size import BALL_FT, measure_diameter, scale_map

OUT_NAME = "ball_3d.parquet"


def measure_clip(clip: Path, log_every: int = 3000) -> pd.DataFrame:
    """Blob measurement at every visible ball frame — the expensive pass, done once."""
    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    to_court, px_per_ft = scale_map(court)
    fps = float(court["video"]["fps"])

    ball = pd.read_parquet(clip / "ball.parquet")
    ball = ball[ball.visible].set_index("frame_idx")
    bdoc = json.loads((clip / "bounces.json").read_text(encoding="utf-8"))
    bkey = next(k for k in bdoc if isinstance(bdoc[k], list))
    bounce_frames = {int(round(float(b.get("t_sec", 0)) * fps)) for b in bdoc[bkey]}

    cap = cv2.VideoCapture(str(clip / "video.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip / 'video.mp4'}")
    want = set(ball.index.tolist())
    hi = max(want) if want else 0

    rows, f = [], 0
    while f <= hi:
        if f in want:
            ok, img = cap.read()
            if not ok:
                break
            u, v = float(ball.loc[f, "pixel_x"]), float(ball.loc[f, "pixel_y"])
            s = px_per_ft(u, v)
            if s:
                pred = BALL_FT * s
                meas = measure_diameter(img, u, v, pred)
                if meas and pred > 0:
                    gx, gy = to_court(u, v)
                    rows.append({"frame": f, "t_sec": f / fps, "pred_px": pred,
                                 "meas_px": meas, "ground_x": gx, "ground_y": gy,
                                 "is_bounce": min((abs(f - b) for b in bounce_frames),
                                                  default=999) <= 2})
        elif not cap.grab():
            break
        f += 1
        if log_every and f % log_every == 0:
            print(f"  {f}/{hi} frames ({f / max(hi, 1):.0%}), {len(rows)} measured",
                  flush=True)
    cap.release()
    return pd.DataFrame(rows)


def fit_calibration(df: pd.DataFrame) -> tuple[float, float, int, int]:
    """Fit measured = a*pred + c on the bounce rows, rejecting blob merges.

    At a bounce the ball is genuinely at z=0, so the predicted size IS the true size and any
    difference is pure measurement error. Merges (the ball fusing with a line, shadow or
    player) read 2-3x large and must go first — they were the single largest source of
    apparent instability in this constant.
    """
    b = df[df.is_bounce]
    b = b[(b.meas_px / b.pred_px) < MERGE_REJECT_RATIO]
    n_rej = int(df.is_bounce.sum()) - len(b)
    if len(b) < 8:
        return DEFAULT_A, DEFAULT_C, len(b), n_rej
    A = np.vstack([b.pred_px.to_numpy(), np.ones(len(b))]).T
    a, c = np.linalg.lstsq(A, b.meas_px.to_numpy(), rcond=None)[0]
    if not (np.isfinite(a) and np.isfinite(c)) or a <= 0.2:
        return DEFAULT_A, DEFAULT_C, len(b), n_rej
    return float(a), float(c), len(b), n_rej


def build(clip: Path) -> tuple[pd.DataFrame, dict]:
    df = measure_clip(clip)
    if df.empty:
        return df, {}
    a, c, n_fit, n_rej = fit_calibration(df)
    C, H, _ = camera_ground(clip)

    true_px = (df.meas_px - c) / a
    # k < 1 puts the ball BEYOND its own ground intersection, which is impossible; clamping
    # saturates at z = 0 rather than inventing negative heights.
    df["k"] = (true_px / df.pred_px).clip(lower=1.0)
    g = np.stack([df.ground_x, df.ground_y], 1)
    t = (g - C) / df.k.to_numpy()[:, None]
    df["court_x_ft"] = C[0] + t[:, 0]
    df["court_y_ft"] = C[1] + t[:, 1]
    df["z_ft"] = H * (1.0 - 1.0 / df.k)

    meta = {"a": round(a, 4), "c": round(c, 4), "n_fit": n_fit, "n_rejected": n_rej,
            "camera_x_ft": round(float(C[0]), 2), "camera_y_ft": round(float(C[1]), 2),
            "camera_height_ft": round(float(H), 2), "n_rows": len(df),
            "merge_reject_ratio": MERGE_REJECT_RATIO}
    return df, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out = a.clip / OUT_NAME
    if out.exists() and not a.force:
        print(f"output exists: {out}. Use --force.")
        return 0

    df, meta = build(a.clip)
    if df.empty:
        print("no measurable ball frames")
        return 1
    df.to_parquet(out, index=False)
    (a.clip / CALIB_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")

    b = df[df.is_bounce]
    L = float(json.loads((a.clip / "court.json").read_text(encoding="utf-8"))
              ["court_geometry_feet"]["length_ft"])
    print(f"wrote {out}  ({len(df)} rows)")
    print(f"  blob fit  measured = {meta['a']:.2f}*pred + {meta['c']:.2f} px "
          f"({meta['n_fit']} bounces, {meta['n_rejected']} merges rejected)")
    print(f"  camera    ({meta['camera_x_ft']}, {meta['camera_y_ft']}) ft "
          f"at {meta['camera_height_ft']} ft")
    if len(b):
        print(f"  CONTROL   z at bounces median {b.z_ft.median():.2f} ft (should be ~0)")
    fl = df[~df.is_bounce]
    inside = float(((fl.court_y_ft >= -15) & (fl.court_y_ft <= L + 15)).mean())
    raw = float(((fl.ground_y >= -15) & (fl.ground_y <= L + 15)).mean())
    print(f"  in flight within the play envelope: {raw:.0%} raw -> {inside:.0%} reconstructed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Can the ball's APPARENT SIZE recover its height? (feasibility probe)

The pipeline's oldest limitation is that a pixel is a *ray*, not a point. The ground
homography picks the far end of that ray by assuming z=0, which is exact for a bounce and
nonsense for a ball in the air — during one indoor volley exchange the ball projected to
court y = 67, 100, 179, 70 and 120 ft on a 44 ft court, every one of them reading "far side".
Three separate rules have now died on this (net-line crossing, opposing-player reach, and
the ground-ball filter), so it is worth attacking the cause rather than working around it.

Apparent size is the missing third equation, and the only one that helps a VOLLEY. A
pickleball is a known 2.9 inches, so its pixel diameter fixes its range directly, with no
assumption about how it is moving — unlike a ballistic fit, which needs curvature and is
therefore weakest during the short flat exchanges where the problem bites.

NOTHING HERE IS TUNED PER COURT — the operator's constraint, since apparent size depends
entirely on camera placement:

    predicted_px = BALL_FT * px_per_ft(where the ray meets the ground)
    ratio        = measured_px / predicted_px      1.0 = on the ground, >1 = airborne

`px_per_ft` is read from that clip's own calibration at that image location, so it
re-derives itself for any court and any camera position.

BOUNCES ARE THE CONTROL, and they also calibrate the measurement. At a bounce the ball is
genuinely at z=0, so the ratio must come out near 1.0. It comes out near 1.6 instead, because
FWHM over-reads a small bright blob (motion blur and halo). That bias is stable, and taking
it from each clip's own bounces keeps the whole method self-calibrating.

MEASURED 2026-08-19:

| clip | bounce (control) | in flight | separability AUC |
|---|---|---|---|
| pb_3_min_indoor_1_court_b | 1.52 | 4.66 | 0.85 (frames 3600-4700, n=13 bounces) |
| pb_5_minute_outdoor-7 | 1.72 | 2.42 | 0.84 (frames 3600-4400, n=20 bounces) |
| pb_3_min_indoor_1_court_b | 1.59 | 4.50 | 0.78 (frames 3600-4400, n=8 bounces) |

**The AUC estimate is noisy** — the control set is only 8-20 bounces, and the same clip
moves between 0.78 and 0.85 depending on the window. Treat the honest range as ~0.8, and
widen the window before drawing a firmer number.

What matters more than the exact figure: the two courts agree despite the outdoor ball being
half the size in pixels (15 px near / 7 px far, against 24 / 11 indoors). The signal is not
a property of one venue. The residual overlap sits at the LOW end of flight, which is correct
physics — the ball really is near the ground just before and after a bounce.

That AUC is also per-FRAME. The question this is meant to answer ("did the ball reach the far
side between these two impacts?") has 30-60 frames to aggregate over, so the usable
reliability should be well above the per-frame figure — but that is an expectation, not yet
a measurement.

Usage:
    python -m tools.measure_ball_size data/pb_3_min_indoor_1_court_b 3600 4400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BALL_FT = 2.9 / 12.0
MIN_CONTRAST = 8.0      # peak-minus-background below this = no blob worth measuring


def scale_map(court: dict):
    """(to_court, px_per_ft) for this clip's calibration."""
    M = np.array(court["homography"]["image_to_court"], float)
    Minv = np.linalg.inv(M)

    def to_court(u, v):
        p = M @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def to_img(x, y):
        p = Minv @ np.array([x, y, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def px_per_ft(u, v):
        """Local image scale where the ray through (u,v) meets the ground.

        Sampled from the calibration itself rather than assumed, so it is correct for any
        camera placement and varies across the frame as perspective demands.
        """
        x, y = to_court(u, v)
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        a, b = np.array(to_img(x - 0.5, y)), np.array(to_img(x + 0.5, y))
        c, d = np.array(to_img(x, y - 0.5)), np.array(to_img(x, y + 0.5))
        sx, sy = np.linalg.norm(b - a), np.linalg.norm(d - c)
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
            return None
        return float(np.sqrt(sx * sy))      # geometric mean = isotropic scale

    return to_court, px_per_ft


def measure_diameter(img, u: float, v: float, expect_px: float):
    """Full-width-at-half-max diameter of the blob at (u,v), in pixels.

    FWHM rather than an absolute threshold: half-max is defined relative to this crop's own
    peak and background, so it carries across venues, exposures and lighting without a
    brightness constant.
    """
    h, w = img.shape[:2]
    r = int(max(8, min(60, expect_px * 3)))
    x0, x1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    y0, y1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    cu, cv_ = int(u) - x0, int(v) - y0
    k = max(2, int(expect_px))
    cen = crop[max(0, cv_ - k):cv_ + k + 1, max(0, cu - k):cu + k + 1]
    if cen.size == 0:
        return None
    peak = float(cen.max())
    bg = float(np.median(np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])))
    if peak - bg < MIN_CONTRAST:
        return None
    mask = (crop >= (peak + bg) / 2.0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    cu_c = min(max(cu, 0), crop.shape[1] - 1)
    cv_c = min(max(cv_, 0), crop.shape[0] - 1)
    lid = int(lab[cv_c, cu_c])
    if lid == 0:                                    # centre unlit: take the nearest blob
        best, bd = 0, 1e9
        for i in range(1, n):
            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
            cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
            dd = (cx - cu) ** 2 + (cy - cv_) ** 2
            if dd < bd:
                best, bd = i, dd
        if best == 0 or bd > (expect_px * 2) ** 2:
            return None
        lid = best
    area = int(stats[lid, cv2.CC_STAT_AREA])
    if area < 2 or area > 4 * r * r:
        return None
    return float(2.0 * np.sqrt(area / np.pi))


def run(clip: Path, f_lo: int, f_hi: int) -> pd.DataFrame:
    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    _, px_per_ft = scale_map(court)
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
                    rows.append({"frame": f, "pred": pred, "meas": meas,
                                 "ratio": meas / pred,
                                 "is_bounce": min((abs(f - b) for b in bounce_frames),
                                                  default=999) <= 2})
        f += 1
    cap.release()
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("f_lo", type=int)
    ap.add_argument("f_hi", type=int)
    a = ap.parse_args(argv)

    df = run(a.clip, a.f_lo, a.f_hi)
    if df.empty:
        print("no measurements")
        return 1
    print(f"{a.clip.name}: {len(df)} ball measurements, frames {a.f_lo}-{a.f_hi}")
    print(f"  predicted diameter where the ray meets the ground: "
          f"{df.pred.min():.1f}-{df.pred.max():.1f} px (median {df.pred.median():.1f})")
    print(f"  measured  diameter                               : "
          f"{df.meas.min():.1f}-{df.meas.max():.1f} px (median {df.meas.median():.1f})")
    b, fl = df[df.is_bounce], df[~df.is_bounce]

    def q(s, lbl):
        if not len(s):
            print(f"  {lbl:<28} (none)")
            return
        print(f"  {lbl:<28} n={len(s):<4} p10 {np.percentile(s,10):5.2f}  "
              f"p25 {np.percentile(s,25):5.2f}  median {np.median(s):5.2f}  "
              f"p75 {np.percentile(s,75):5.2f}  p90 {np.percentile(s,90):5.2f}")

    print()
    q(b.ratio, "on the ground (bounces)")
    q(fl.ratio, "in flight")
    if len(b) and len(fl):
        bb, aa = b.ratio.to_numpy(), fl.ratio.to_numpy()
        auc = float((aa[:, None] > bb[None, :]).mean()
                    + 0.5 * (aa[:, None] == bb[None, :]).mean())
        print()
        print(f"  separability AUC = {auc:.2f}   (0.50 = no information, 1.00 = perfect)")
        print(f"  measurement bias from this clip's bounces = {np.median(bb):.2f}  "
              f"(divide ratios by this before use)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

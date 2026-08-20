"""THE TEST: does reconstructed HEIGHT tell "the ball went over" from "genuine handling"?

reject_same_side_runs collapses consecutive same-net-side impacts. That is right when the
ball never left the side, and catastrophic when a shot between them was MISSED -- it then
deletes a second real shot, which is 9 of the operator's 21 confirmed misses.

The ball must clear the NET (34 in = 2.83 ft) to have gone over and come back. So for every
pair of same-side impacts the filter would collapse, take the maximum reconstructed height
between them and compare it against the operator's ground truth:

    CROSSED  - the operator lists a real shot strictly between the two impacts
    HANDLING - they do not

Labels are only valid inside the four rallies the operator reviewed, so the test is
restricted to those.
"""
import json, sys, subprocess
import numpy as np, pandas as pd, cv2
from pathlib import Path
sys.path.insert(0, ".")
from tools.measure_ball_size import BALL_FT, measure_diameter, scale_map
from tools.ball_3d import camera_ground, DEFAULT_BIAS

CLIP = Path("data/pb_3_min_indoor_1_court_b")
REVIEWED = {0, 3, 5, 9}
RESET_S = 3.0
NET_FT = 34.0 / 12.0

court = json.loads((CLIP / "court.json").read_text())
FPS = float(court["video"]["fps"])
truth = json.loads((CLIP / "truth.json").read_text())["points"]
missed = json.loads((CLIP / "missed_review.json").read_text())["missed"]
windows = [(float(p["start_t_sec"]) - 1, float(p["end_t_sec"]) + 1)
           for p in truth if p["point"] in REVIEWED]

# shots WITH the handling filter disabled -- these include the impacts it deletes
off = json.loads(Path(sys.argv[1]).read_text())["shots"]
off = sorted(off, key=lambda s: s["frame"])

pairs = []
for a, z in zip(off, off[1:]):
    if a.get("hitter_side") is None or a.get("hitter_side") != z.get("hitter_side"):
        continue
    if (z["frame"] - a["frame"]) > RESET_S * FPS:
        continue
    if not any(lo <= a["t_sec"] <= hi for lo, hi in windows):
        continue
    crossed = any(a["t_sec"] < m["t_sec"] < z["t_sec"] for m in missed)
    pairs.append({"f0": int(a["frame"]), "f1": int(z["frame"]),
                  "t0": a["t_sec"], "t1": z["t_sec"], "crossed": crossed})
print(f"{len(pairs)} same-side pairs inside the reviewed rallies "
      f"({sum(p['crossed'] for p in pairs)} with a missed shot between them)")

need = set()
for p in pairs:
    need.update(range(p["f0"], p["f1"] + 1))
print(f"decoding {len(need)} frames")

to_court, px_per_ft = scale_map(court)
C, H, _ = camera_ground(CLIP)
ball = pd.read_parquet(CLIP / "ball.parquet"); ball = ball[ball.visible].set_index("frame_idx")
bdoc = json.loads((CLIP / "bounces.json").read_text())
bkey = next(k for k in bdoc if isinstance(bdoc[k], list))
bfr = {int(round(float(b.get("t_sec", 0)) * FPS)) for b in bdoc[bkey]}

cap = cv2.VideoCapture(str(CLIP / "video.mp4"))
rec, f = {}, 0
hi_f = max(need)
while f <= hi_f:
    if f in need:
        ok, img = cap.read()
        if not ok: break
        if f in ball.index:
            u, v = float(ball.loc[f, "pixel_x"]), float(ball.loc[f, "pixel_y"])
            s = px_per_ft(u, v)
            if s:
                pred = BALL_FT * s
                meas = measure_diameter(img, u, v, pred)
                if meas:
                    rec[f] = {"ratio": meas / pred,
                              "bounce": min((abs(f - b) for b in bfr), default=999) <= 2}
    else:
        if not cap.grab(): break
    f += 1
cap.release()

bb = [r["ratio"] for r in rec.values() if r["bounce"]]
bias = float(np.median(bb)) if len(bb) >= 5 else DEFAULT_BIAS
print(f"bias from {len(bb)} bounce frames = {bias:.2f}\n")

for p in pairs:
    zs = []
    for g in range(p["f0"], p["f1"] + 1):
        r = rec.get(g)
        if not r: continue
        k = max(1.0, r["ratio"] / bias)
        zs.append(H * (1.0 - 1.0 / k))
    p["max_z"] = max(zs) if zs else np.nan
    p["n"] = len(zs)

df = pd.DataFrame(pairs).dropna(subset=["max_z"])
print(f"{'t0':>8}{'t1':>8}{'label':>10}{'max height ft':>15}{'over the net?':>15}")
for r in df.sort_values("t0").itertuples():
    print(f"{r.t0:>8.2f}{r.t1:>8.2f}{('CROSSED' if r.crossed else 'handling'):>10}"
          f"{r.max_z:>15.2f}{('yes' if r.max_z > NET_FT else 'no'):>15}")
c, h = df[df.crossed], df[~df.crossed]
print()
for lbl, s in (("CROSSED (shot missed between)", c), ("handling (no shot between)", h)):
    if len(s):
        print(f"  {lbl:<32} n={len(s):<3} median max height {s.max_z.median():5.2f} ft   "
              f"{(s.max_z > NET_FT).mean():.0%} clear the net")
if len(c) and len(h):
    cc, hh = c.max_z.to_numpy(), h.max_z.to_numpy()
    auc = float((cc[:, None] > hh[None, :]).mean() + 0.5 * (cc[:, None] == hh[None, :]).mean())
    print(f"\n  separability AUC = {auc:.2f}")

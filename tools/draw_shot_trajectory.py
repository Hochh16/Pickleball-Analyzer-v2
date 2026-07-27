"""Prototype/validation: draw the CLEAN post-shot ball trajectory in image space.

For each rally-ending shot, walk the ball forward from the contact point accepting
only visible detections within a continuity radius (rejects the parked-background
jumps the tracker makes when the real ball is lost), and draw that path as a fading
polyline over a mid-flight frame, with the net line. Lets the operator SEE whether
the shot's OUTCOME (crossed to the far court vs died near-side = net/short) is
readable from direction alone, without a landing bounce.

    python tools/draw_shot_trajectory.py --clip data/pb_5_minute_outdoor-2
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

STEP = 280          # px continuity gate between consecutive accepted points
MAXF = 90           # look up to 1.5s (60fps) after contact
MISS_STOP = 10      # stop after this many consecutive rejected/missing frames


def load(clip, name):
    return json.load(open(Path(clip) / name, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--reasons", default="net-or-short,ball-out,ball-not-returned")
    args = ap.parse_args()
    D = Path(args.clip)
    court = load(D, "court.json")
    Hc2i = np.array(court["homography"]["court_to_image"])

    def c2i(cx, cy):
        v = Hc2i @ np.array([cx, cy, 1.0])
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    net_pts = [c2i(x, 22.0) for x in np.linspace(0, 20, 9)]
    ball = pd.read_parquet(D / "ball.parquet").set_index("frame_idx")
    shots = {s["shot_id"]: s for s in load(D, "shots.json")["shots"]}
    cls = {s["shot_id"]: s for s in load(D, "classified.json")["shots"]}
    rallies = load(D, "rallies.json")["rallies"]
    wanted = set(args.reasons.split(","))

    def clean_traj(f0, start):
        path = [(f0, float(start[0]), float(start[1]))]
        miss = 0
        for f in range(f0 + 1, f0 + MAXF + 1):
            if f not in ball.index:
                continue
            r = ball.loc[f]
            if not bool(r["visible"]) or np.isnan(r["pixel_x"]):
                miss += 1
                if miss >= MISS_STOP:
                    break
                continue
            px, py = float(r["pixel_x"]), float(r["pixel_y"])
            if np.hypot(px - path[-1][1], py - path[-1][2]) <= STEP:
                path.append((f, px, py))
                miss = 0
            else:
                miss += 1
                if miss >= MISS_STOP:
                    break
        return path

    out = D / "_traj_check"
    out.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(D / "video.mp4"))
    todo = [r for r in rallies if r["end_reason"] in wanted]
    print(f"{len(todo)} rally-enders")
    for r in todo:
        sid = r["shot_ids"][-1]
        s = shots[sid]
        ip = s.get("impact_pixel_xy")
        if not ip:
            continue
        f0 = int(s["frame"])
        path = clean_traj(f0, ip)
        # draw over a frame ~1/3 into the trajectory so the ball is mid-flight
        fmid = path[min(len(path) // 3, len(path) - 1)][0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fmid)
        ok, img = cap.read()
        if not ok:
            continue
        for a, b in zip(net_pts, net_pts[1:]):
            cv2.line(img, a, b, (0, 200, 255), 3)
        # fading polyline: yellow (contact) -> red (end)
        for i in range(1, len(path)):
            t = i / max(1, len(path) - 1)
            col = (0, int(255 * (1 - t)), 255)     # BGR: yellow->red
            cv2.line(img, (int(path[i - 1][1]), int(path[i - 1][2])),
                     (int(path[i][1]), int(path[i][2])), col, 3)
            cv2.circle(img, (int(path[i][1]), int(path[i][2])), 5, col, -1)
        cv2.drawMarker(img, (int(ip[0]), int(ip[1])), (255, 255, 0),
                       cv2.MARKER_CROSS, 46, 4)                      # contact
        cv2.circle(img, (int(path[-1][1]), int(path[-1][2])), 16, (0, 0, 255), 3)  # end
        ct = cls.get(sid, {})
        who = "YOU" if s.get("is_user") else "opp"
        banner = (f"rally {r['rally_id']}  end={r['end_reason']}  "
                  f"{ct.get('shot_type','?')}/{ct.get('stroke_side','?')} by {who}  "
                  f"traj_pts={len(path)} (lost after {len(path)} of {MAXF})")
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(img, banner, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        name = f"r{r['rally_id']:02d}_{r['end_reason']}_{who}.png"
        cv2.imwrite(str(out / name), img)
        print("  wrote", name, "pts=", len(path))
    cap.release()
    print("->", out)


if __name__ == "__main__":
    main()

"""Validation render for NET-HIT / short-ball detection.

For every rally the pipeline flagged `net-or-short` (the ball landed in-court on
the HITTER's own side, i.e. it never cleared the net / crossed), tile a handful
of frames from the last shot through the ball dying, with the ball position, the
net line, and the call stamped on. Also renders the `ball-out` enders for
contrast, so the operator can see the elimination working (out vs net vs in).

    python tools/verify_net_hits.py --clip data/pb_5_minute_outdoor-2

Writes one PNG per rally-ending shot into <clip>/_net_check/.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

NET_Y_FT = 22.0
COURT_W_FT = 20.0


def load(clip: Path, name: str):
    return json.load(open(clip / name, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--reasons", default="net-or-short,ball-out",
                    help="comma list of end_reasons to render")
    ap.add_argument("--cols", type=int, default=6, help="frames per contact sheet")
    ap.add_argument("--crop", type=int, default=900,
                    help="half-size of the square crop around the ball/net (px)")
    args = ap.parse_args()
    D = Path(args.clip)
    wanted = set(args.reasons.split(","))

    court = load(D, "court.json")
    H = np.array(court["homography"]["court_to_image"])

    def c2i(cx, cy):
        v = H @ np.array([cx, cy, 1.0])
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    net_pts = [c2i(x, NET_Y_FT) for x in np.linspace(0, COURT_W_FT, 9)]

    rallies = load(D, "rallies.json")["rallies"]
    shots = {s["shot_id"]: s for s in load(D, "shots.json")["shots"]}
    cls = {s["shot_id"]: s for s in load(D, "classified.json")["shots"]}
    bounces = {b["bounce_id"]: b for b in load(D, "bounces.json")["bounces"]}
    ball = pd.read_parquet(D / "ball.parquet").set_index("frame_idx")

    out_dir = D / "_net_check"
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(D / "video.mp4"))

    todo = [r for r in rallies if r["end_reason"] in wanted]
    print(f"{len(todo)} rally-enders to render ({sorted(wanted)})")

    for r in todo:
        sid = r["shot_ids"][-1]
        s = shots[sid]
        f0 = int(s["frame"])
        # The DECISION input is the ending bounce (where the ball landed), NOT the
        # airborne track. Centre + time the sheet on that bounce so the operator sees
        # what the call rests on: a landing on the hitter's own side = net/short.
        eb = bounces.get(r.get("ending_bounce_id"))
        bpx = None
        if eb and eb.get("court_xy_ft") and eb["court_xy_ft"][0] is not None:
            bpx = c2i(eb["court_xy_ft"][0], eb["court_xy_ft"][1])
            fb = int(eb["frame"])
        else:
            fb = min(int(r["end_frame"]), f0 + 60)
        # window from the shot to just past the bounce
        lo, hi = min(f0, fb - 20), fb + 20
        frames = np.linspace(lo, hi, args.cols).astype(int)

        # centre the crop on the bounce (fallback: net at mid-court)
        if bpx is not None:
            cxc, cyc = bpx
        else:
            cxc = int(np.median([c2i(x, NET_Y_FT)[0] for x in (5, 10, 15)]))
            cyc = int(np.median([p[1] for p in net_pts]))
        cr = args.crop

        tiles = []
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if not ok:
                continue
            # draw net line
            for a, b in zip(net_pts, net_pts[1:]):
                cv2.line(img, a, b, (0, 200, 255), 3)
            # ball
            if int(f) in ball.index:
                row = ball.loc[int(f)]
                if bool(row["visible"]) and not np.isnan(row["pixel_x"]):
                    cv2.circle(img, (int(row["pixel_x"]), int(row["pixel_y"])),
                               14, (0, 0, 255), 3)
            # projected landing bounce (the decision input) -- green target,
            # brightest on the actual bounce frame
            if bpx is not None:
                thick = 4 if abs(int(f) - fb) <= 3 else 2
                cv2.drawMarker(img, bpx, (0, 255, 0), cv2.MARKER_TILTED_CROSS,
                               46, thick)
                cv2.circle(img, bpx, 26, (0, 255, 0), thick)
            # contact marker on/near the shot frame
            if abs(int(f) - f0) <= 3 and s.get("impact_pixel_xy"):
                ip = s["impact_pixel_xy"]
                cv2.drawMarker(img, (int(ip[0]), int(ip[1])), (255, 255, 0),
                               cv2.MARKER_CROSS, 40, 3)
            y0, y1 = max(0, cyc - cr), cyc + cr
            x0, x1 = max(0, cxc - cr), cxc + cr
            crop = img[y0:y1, x0:x1]
            crop = cv2.resize(crop, (360, 360))
            cv2.putText(crop, f"f{int(f)}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            tiles.append(crop)

        if not tiles:
            continue
        sheet = np.hstack(tiles)
        who = ("YOU" if s.get("is_user") else "opp")
        typ = cls.get(sid, {}).get("shot_type", "?")
        side = cls.get(sid, {}).get("stroke_side", "?")
        es = r.get("end_signals", {})
        banner = (f"rally {r['rally_id']}  end={r['end_reason']}  "
                  f"shot={typ}/{side} by {who}  conf={r['end_reason_confidence']}  "
                  f"hitter_side={es.get('hitter_side')} "
                  f"last_bounce(in_court={es.get('last_bounce_in_court')},"
                  f"side={es.get('last_bounce_side')})")
        bar = np.zeros((40, sheet.shape[1], 3), np.uint8)
        cv2.putText(bar, banner, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        sheet = np.vstack([bar, sheet])
        name = f"r{r['rally_id']:02d}_{r['end_reason']}_{who}.png"
        cv2.imwrite(str(out_dir / name), sheet)
        print("  wrote", name)

    cap.release()
    print("done ->", out_dir)


if __name__ == "__main__":
    main()

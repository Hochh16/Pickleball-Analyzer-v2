"""Render the frames where the operator says we named the wrong hitter.

Attribution disagreements cannot be settled by argument -- the operator can see who swung
and the pipeline cannot. This draws every NEAR-SIDE player box at each disputed contact,
labelled with the role we assigned and the track id, plus the ball, so the operator can say
directly whether the box we called "user" is on the user.

Usage:
    python -m tools.show_attribution data/pb_5_minute_outdoor-7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

PAD = 260


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    clip = a.clip
    out = a.out or clip / "_labeling" / "attribution"
    out.mkdir(parents=True, exist_ok=True)
    shots = json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"]
    review = json.loads((clip / "shot_review.json").read_text(encoding="utf-8"))
    roles = json.loads((clip / "track_roles.json").read_text(encoding="utf-8"))["roles"]
    role_of = {t: r for r, d in roles.items() for t in d["track_ids"]}
    pl = pd.read_parquet(clip / "players.parquet")
    ball = pd.read_parquet(clip / "ball.parquet")

    cap = cv2.VideoCapture(str(clip / "video.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip / 'video.mp4'}")

    written = []
    for d in sorted(review.get("wrong_player", []), key=lambda x: x["t_sec"]):
        near = [s for s in shots if abs(s["t_sec"] - d["t_sec"]) <= 1.0]
        if not near:
            continue
        s = min(near, key=lambda x: abs(x["t_sec"] - d["t_sec"]))
        f = int(s["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue

        rows = pl[pl.frame == f]
        xs, ys = [], []
        for r in rows.itertuples():
            role = role_of.get(r.track_id)
            if role not in ("user", "partner"):
                continue
            chosen = r.track_id == s["track_id"]
            # the box we picked is drawn thick; the other near-side player thin
            colour = (0, 0, 255) if chosen else (0, 200, 0)
            p1 = (int(r.bbox_x1), int(r.bbox_y1))
            p2 = (int(r.bbox_x2), int(r.bbox_y2))
            cv2.rectangle(img, p1, p2, colour, 6 if chosen else 3)
            tag = f"{role} #{r.track_id}" + ("  <- WE SAID THIS ONE HIT IT" if chosen else "")
            cv2.putText(img, tag, (p1[0], max(40, p1[1] - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
            xs += [p1[0], p2[0]]
            ys += [p1[1], p2[1]]

        bq = ball[(ball.frame_idx == f) & ball.visible]
        if not bq.empty:
            bx, by = int(bq.pixel_x.iloc[0]), int(bq.pixel_y.iloc[0])
            cv2.circle(img, (bx, by), 22, (0, 255, 255), 5)
            cv2.putText(img, "ball", (bx + 26, by), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 3)
            xs.append(bx)
            ys.append(by)

        if xs:
            h, w = img.shape[:2]
            x1, x2 = max(0, min(xs) - PAD), min(w, max(xs) + PAD)
            y1, y2 = max(0, min(ys) - PAD), min(h, max(ys) + PAD)
            img = img[y1:y2, x1:x2]

        mm, ss = divmod(s["t_sec"], 60)
        cv2.putText(img, f"{int(mm)}:{ss:05.2f}   operator: {d['actually']}",
                    (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        name = out / f"{s['t_sec']:07.2f}.jpg".replace(".jpg", "") 
        p = out / f"t{s['t_sec']:07.2f}.jpg"
        cv2.imwrite(str(p), img)
        written.append(p)

    cap.release()
    print(f"wrote {len(written)} frames to {out}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

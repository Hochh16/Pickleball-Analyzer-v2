"""Render ONE rally as an annotated video for operator validation.

Single sequential decode pass (4K random seeks are too slow). Draws the ball
(with a short trail), each player's bbox + role + front-foot court_y, and a HUD
listing every shot in the rally with its type/side/zone, highlighting the active
shot. A contact flash marks each paddle strike.

Usage:
    python tools/render_rally.py --clip data/pb_5_minute_outdoor-2 --rally 10
    python tools/render_rally.py --clip <clip> --frames 17884-18744 --out out.mp4
"""
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROLE_COLOR = {  # BGR
    "user": (0, 220, 0), "partner": (0, 200, 200),
    "opp_near": (0, 120, 255), "opp_far": (0, 120, 255),
    "opponent": (0, 120, 255), "opponent_1": (0, 120, 255),
    "opponent_2": (60, 60, 255),
}
TYPE_COLOR = {
    "serve": (255, 200, 0), "drive": (60, 60, 255), "dink": (0, 220, 0),
    "drop": (0, 200, 255), "lob": (255, 0, 200), "reset": (200, 200, 200),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--rally", type=int, default=None)
    ap.add_argument("--frames", default=None, help="a-b override")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=1280, help="output width (downscale)")
    ap.add_argument("--pad", type=int, default=30, help="frames of lead-in/out")
    args = ap.parse_args()

    D = Path(args.clip)
    shots = json.load(open(D / "classified.json"))["shots"]
    by_id = {s["shot_id"]: s for s in shots}
    roles_raw = json.load(open(D / "track_roles.json")).get("track_roles", {})
    roles = {int(k): v.get("role", "?") for k, v in roles_raw.items()}
    players = pd.read_parquet(D / "players.parquet")
    ball = pd.read_parquet(D / "ball.parquet").sort_values("frame_idx")
    bx = dict(zip(ball["frame_idx"], ball["pixel_x"]))
    by = dict(zip(ball["frame_idx"], ball["pixel_y"]))
    bvis = dict(zip(ball["frame_idx"], ball["visible"] | ball["interpolated"]))

    # resolve the frame window + which shots belong to it
    if args.frames:
        f0, f1 = (int(x) for x in args.frames.split("-"))
        rally_shots = [s for s in shots if f0 <= s["frame"] <= f1]
    else:
        rallies = json.load(open(D / "rallies.json"))
        rallies = rallies.get("rallies", rallies)
        r = rallies[args.rally]
        ids = r.get("shot_ids", r.get("shots", []))
        rally_shots = [by_id[i] for i in ids if i in by_id]
        f0, f1 = rally_shots[0]["frame"], rally_shots[-1]["frame"]
    f0 = max(0, f0 - args.pad)
    f1 = f1 + args.pad
    contacts = {s["frame"]: s for s in rally_shots}
    label = args.rally if args.rally is not None else f"{f0}-{f1}"
    out_path = Path(args.out) if args.out else D / f"_rally_{label}_check.mp4"

    # per-frame player rows, indexed for the window only (memory-lean)
    win = players[(players["frame"] >= f0) & (players["frame"] <= f1)]
    pf = {}
    for r in win.itertuples(index=False):
        pf.setdefault(int(r.frame), []).append(r)

    cap = cv2.VideoCapture(str(D / "video.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    scale = args.width / W
    outW, outH = args.width, int(H * scale)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (outW, outH))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    trail = deque(maxlen=12)
    active_idx = -1
    flash = 0
    for f in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        # ball + trail
        if bvis.get(f) and not np.isnan(bx.get(f, np.nan)):
            p = (int(bx[f]), int(by[f]))
            trail.append(p)
            for j, tp in enumerate(trail):
                cv2.circle(frame, tp, 4, (255, 255, 255), -1)
            cv2.circle(frame, p, 12, (0, 255, 255), 3)
        # players
        for r in pf.get(f, []):
            role = roles.get(int(r.track_id), "?")
            col = ROLE_COLOR.get(role, (200, 200, 200))
            cv2.rectangle(frame, (int(r.bbox_x1), int(r.bbox_y1)),
                          (int(r.bbox_x2), int(r.bbox_y2)), col, 2)
            cv2.putText(frame, f"{role} y={r.court_y_ft:.0f}",
                        (int(r.bbox_x1), int(r.bbox_y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        # contact flash
        if f in contacts:
            s = contacts[f]
            active_idx = rally_shots.index(s)
            flash = 8
        if flash > 0:
            s = rally_shots[active_idx]
            ix, iy = s["impact_pixel_xy"]
            cv2.circle(frame, (int(ix), int(iy)), 34, (0, 0, 255), 4)
            flash -= 1
        frame = cv2.resize(frame, (outW, outH))
        # HUD: shot list
        y = 26
        cv2.putText(frame, f"Rally {label}  ({len(rally_shots)} shots)",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, s in enumerate(rally_shots):
            y += 22
            fe = s["features"]
            txt = (f"{i}: {s['shot_type']:5} {s.get('hitter_side','?'):4} "
                   f"{s.get('stroke_side','?'):8} z={fe.get('contact_zone','?')[:4]} "
                   f"{'V' if s.get('is_volley') else ' '}")
            col = TYPE_COLOR.get(s["shot_type"], (200, 200, 200))
            if i == active_idx:
                cv2.rectangle(frame, (8, y - 15), (430, y + 5), (40, 40, 40), -1)
            cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 2 if i == active_idx else 1)
        vw.write(frame)
    cap.release()
    vw.release()

    # Transcode mp4v -> H.264 (yuv420p, faststart) so it plays in any viewer /
    # browser. The raw cv2 mp4v output is blank in many players.
    try:
        import subprocess
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp = out_path.with_suffix(".raw.mp4")
        out_path.replace(tmp)
        r = subprocess.run(
            [exe, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt",
             "yuv420p", "-movflags", "+faststart", "-preset", "fast",
             str(out_path)], capture_output=True, text=True)
        if r.returncode == 0:
            tmp.unlink()
        else:
            tmp.replace(out_path)
            print("WARN: H.264 transcode failed; left mp4v output")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: transcode skipped ({e}); output may not play everywhere")
    print(f"wrote {out_path}  ({f1 - f0 + 1} frames, {outW}x{outH})")


if __name__ == "__main__":
    main()

"""Annotate the WHOLE video in place — no cutting, no snippet reels.

Built after the operator rejected the reel format outright: *"showing only snippets of a
shot vs continuous feed is making it difficult to view … I can't tell if you are referring
to shot 9 being my dink or the opponent's dink since videos overlap … show the whole video
with the labeling and indicate start/stop time."* Cutting is what made the earlier review
painful, so this cuts nothing. Every frame of the source appears once, in order.

Burned into each frame:
  * a running clock, so any observation can be quoted as a timestamp
  * every detected shot, flashed at its contact frame and held briefly afterwards
  * a live per-rally tally — "rally 3 · 4 detected / 14 expected" — when the clip has
    operator truth. That is the whole point: the operator can SEE the counter falling
    behind while the rally is in progress, instead of reconciling numbers afterwards.
  * the rally window and its server, also from truth

Output plays at half speed by default (`--slow`), because the operator asked for that too:
a 60fps contact is a single frame, and real-time playback makes it impossible to stop on.

Frames are streamed straight to the writer. An earlier reel builder buffered annotated
frames and needed ~66 GB for 125 shots; nothing here holds more than one frame.

Usage:
    python -m tools.annotate_full data/pb_3_min_indoor_1_court_b
    python -m tools.annotate_full data/pb_3_min_indoor_1_court_b --slow 1 --width 1920
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

HOLD_S = 0.6          # how long a shot marker stays on screen after its contact
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, YELLOW, RED, GREEN, DIM = ((255, 255, 255), (0, 255, 255), (0, 0, 255),
                                  (0, 220, 0), (170, 170, 170))


def clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--slow", type=float, default=2.0,
                    help="playback slowdown factor (2 = half speed)")
    a = ap.parse_args(argv)

    clip = a.clip
    shots = json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"]
    roles = json.loads((clip / "track_roles.json").read_text(encoding="utf-8"))["roles"]
    role_of = {t: r for r, d in roles.items() for t in d["track_ids"]}
    tp = clip / "truth.json"
    points = json.loads(tp.read_text(encoding="utf-8")).get("points", []) if tp.exists() else []
    ballp = clip / "ball.parquet"
    ball = pd.read_parquet(ballp) if ballp.exists() else None
    if ball is not None:
        ball = ball[ball.visible].set_index("frame_idx")

    cap = cv2.VideoCapture(str(clip / "video.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip / 'video.mp4'}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, a.width / max(src_w, 1))
    out_w, out_h = int(src_w * scale), int(src_h * scale)

    out = a.out or clip / "_labeling" / f"{clip.name}_annotated.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, fps / max(a.slow, 0.01)), (out_w, out_h))

    # shot lookup by frame, and a running index so the tally is O(1) per frame
    by_frame: dict[int, list[dict]] = {}
    for i, s in enumerate(sorted(shots, key=lambda x: x["frame"])):
        by_frame.setdefault(int(s["frame"]), []).append({**s, "n": i + 1})
    hold = int(HOLD_S * fps)

    def point_at(t: float):
        for p in points:
            if float(p["start_t_sec"]) - 1.0 <= t <= float(p["end_t_sec"]) + 1.0:
                return p
        return None

    f = 0
    last = []          # [(frame_shown, shot)] still within the hold window
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if scale < 1.0:
            img = cv2.resize(img, (out_w, out_h))
        t = f / fps

        for s in by_frame.get(f, []):
            last.append((f, s))
        last = [(g, s) for g, s in last if f - g <= hold]

        # ball
        if ball is not None and f in ball.index:
            bx = int(float(ball.loc[f, "pixel_x"]) * scale)
            by = int(float(ball.loc[f, "pixel_y"]) * scale)
            cv2.circle(img, (bx, by), 13, YELLOW, 2)

        # shot markers, brightest at the contact frame
        for g, s in last:
            px, py = s.get("impact_pixel_xy") or (None, None)
            if px is None:
                continue
            x, y = int(px * scale), int(py * scale)
            fresh = (f - g) < int(0.12 * fps)
            cv2.circle(img, (x, y), 34 if fresh else 24, RED, 4 if fresh else 2)
            who = role_of.get(s["track_id"], "?")
            cv2.putText(img, f"#{s['n']} {who}{' SERVE' if s.get('is_serve') else ''}",
                        (x + 38, y - 6), FONT, 0.72, RED, 2)

        # header: clock, rally window, live tally
        cv2.rectangle(img, (0, 0), (out_w, 78), (0, 0, 0), -1)
        cv2.putText(img, clock(t), (14, 52), FONT, 1.25, WHITE, 3)

        p = point_at(t)
        if p is not None:
            got = sum(1 for s in shots
                      if float(p["start_t_sec"]) - 1.0 <= s["t_sec"] <= t)
            exp = int(p["n_shots"])
            behind = got < exp and t >= float(p["end_t_sec"])
            cv2.putText(img, f"rally {p['point']}  ({p['server']} serves)  "
                             f"{clock(float(p['start_t_sec']))}-{clock(float(p['end_t_sec']))}",
                        (210, 32), FONT, 0.78, GREEN, 2)
            cv2.putText(img, f"detected {got} / {exp} expected",
                        (210, 64), FONT, 0.85, RED if behind else YELLOW, 2)
        else:
            cv2.putText(img, "BETWEEN POINTS  (operator truth: no rally live here)",
                        (210, 50), FONT, 0.8, DIM, 2)

        writer.write(img)
        f += 1
        if n_frames and f % 600 == 0:
            print(f"  {f}/{n_frames} frames ({f / n_frames:.0%})", flush=True)

    cap.release()
    writer.release()
    mb = out.stat().st_size / 1e6 if out.exists() else 0
    print(f"wrote {out}  ({mb:.0f} MB, {f} frames, plays at {a.slow:g}x slower)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

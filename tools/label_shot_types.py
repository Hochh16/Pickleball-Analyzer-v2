r"""Label what each shot ACTUALLY WAS — one keypress per shot.

Why this exists. Shot typing scores **12 of 32 = 37.5%**, and that 32 is the entire truth set
this project has. Three of them are drops. A number built on 32 labels cannot separate a real
improvement from noise in either direction — a change of two shots moves it six points — and
twice now a shot-typing change has been accepted or rejected on exactly that much evidence.
Labels are the binding constraint, and they are the one thing only the operator can supply.

`tools/label_shots.py` already does this by building a reel and a CSV to fill in alongside it.
The operator's verdict on that workflow was that snippets are hard to judge and switching
between a video and a spreadsheet is worse. So this is the same job done the way
`mark_serve_strikes.py` was: the shot plays in a window, you press one key, it saves and moves
on.

It deliberately does NOT show what the classifier guessed. Seeing the guess turns labelling
into agreeing, and an independent label is the entire point.

Output goes to `<clip>/_labeling/labels_interactive.csv` in the same schema as the existing
label files, so `tools/score_shot_types.py` picks it up with no change — it reads every
`labels*.csv` in the folder.

Usage:
    python tools/label_shot_types.py data/pb_5_minute_outdoor-7
    python tools/label_shot_types.py data/pb_3_min_indoor_1_court_b --limit 40
    python tools/label_shot_types.py data/pb_5_minute_outdoor-7 --report

Keys — one per shot type, shown on screen the whole time:

    d  drive        k  dink         r  drop         l  lob
    s  serve        t  return       e  reset
    n  NOT a shot (between points, a feed, junk)
    ?  unsure — skip it; a skip is worth more than a guess

    SPACE  replay this shot          BACKSPACE  go back one shot
    Esc    save and quit  (it also saves after every single label)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_NAME = "labels_interactive.csv"
FIELDS = ["shot_no", "frame", "shot_id", "time", "hitter_role", "hitter_side",
          "true_type", "true_volley", "true_in", "notes"]

PRE_S = 1.2          # show this much before the contact...
POST_S = 1.8         # ...and this much after, so the flight and the landing are both visible
JPEG_QUALITY = 82
DISPLAY_FRACTION = 0.85
PLAY_FPS = 20
TRAIL_FRAMES = 12

# One key per type. Chosen so the common ones are on the home row and nothing collides:
# 'r' is drop (not return) because drops are the class we are shortest of and the one the
# classifier gets wrong 2 times in 3.
KEYS = {"d": "drive", "k": "dink", "r": "drop", "l": "lob",
        "s": "serve", "t": "return", "e": "reset", "n": "not a shot"}


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_done(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["shot_id"])] = r
            except (KeyError, TypeError, ValueError):
                continue
    return out


def save_done(path: Path, done: Dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(done):
            w.writerow({c: done[k].get(c, "") for c in FIELDS})
    tmp.replace(path)


def decode_windows(video: Path, windows, scale_to: int, log=print) -> Dict[int, bytes]:
    """One forward pass, JPEG in memory. Never seeks backwards: cap.set is not frame-accurate
    on long-GOP H.264 and has silently misfiled a labelling session before."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        fail(f"cannot open {video}")
    want = set()
    for w in windows:
        want.update(w)
    hi = max(want)
    got: Dict[int, bytes] = {}
    t0 = time.time()
    f = 0
    while f <= hi:
        if f in want:
            ok, img = cap.read()
            if not ok:
                break
            h, w_ = img.shape[:2]
            if w_ > scale_to:
                img = cv2.resize(img, (scale_to, int(h * scale_to / w_)),
                                 interpolation=cv2.INTER_AREA)
            ok2, buf = cv2.imencode(".jpg", img,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok2:
                got[f] = buf.tobytes()
        else:
            if not cap.grab():
                break
        f += 1
        if f % 2000 == 0:
            rate = f / max(time.time() - t0, 1e-6)
            log(f"   {f}/{hi} ({f / hi:.0%}), ~{(hi - f) / max(rate, 1e-6):.0f}s left",
                flush=True)
    cap.release()
    log(f"   decoded {len(got)} frames in {time.time() - t0:.0f}s "
        f"({sum(len(v) for v in got.values()) / 1e6:.0f} MB held)")
    return got


class Labeller:
    def __init__(self, root, clip: Path, fps: float, src_w: int, todo: List[dict],
                 frames: Dict[int, bytes], jpeg_w: int, balls, done, out_path: Path):
        self.root, self.clip, self.fps = root, clip, fps
        self.todo, self.frames, self.done, self.out_path = todo, frames, done, out_path
        self.balls = balls
        self.scale = jpeg_w / float(src_w)
        self.i = 0
        self.cur = 0
        self.playing = False
        self.image_id = None
        self.tk_image = None

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.max_w = int(sw * DISPLAY_FRACTION)
        self.max_h = int(sh * DISPLAY_FRACTION) - 110
        self.canvas = tk.Canvas(root, width=self.max_w, height=self.max_h, bg="black",
                                highlightthickness=0)
        self.canvas.pack()
        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, anchor="w",
                 font=("Consolas", 12)).pack(fill="x")
        keys = "   ".join(f"{k} {v}" for k, v in KEYS.items())
        tk.Label(root, text=keys, anchor="w", font=("Consolas", 11),
                 fg="#1a1a1a").pack(fill="x")
        tk.Label(root, anchor="w", font=("Consolas", 10), fg="#555",
                 text="  ? unsure (skip)    SPACE replay    BACKSPACE previous shot"
                      "    Esc save + quit").pack(fill="x")

        for k, v in KEYS.items():
            root.bind(k, lambda e, tv=v: self.label(tv))
            root.bind(k.upper(), lambda e, tv=v: self.label(tv))
        root.bind("<question>", lambda e: self.label(""))
        root.bind("<slash>", lambda e: self.label(""))
        root.bind("<space>", lambda e: self.replay())
        root.bind("<BackSpace>", lambda e: self.prev_shot())
        root.bind("<Left>", lambda e: self.prev_shot())
        root.bind("<Escape>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.enter()

    # -- navigation -------------------------------------------------------
    def enter(self):
        while self.i < len(self.todo) and self.todo[self.i]["shot_id"] in self.done:
            self.i += 1
        if self.i >= len(self.todo):
            self.quit()
            return
        self.cur = self.todo[self.i]["lo"]
        self.replay()

    def label(self, type_name: str):
        a = self.todo[self.i]
        self.done[a["shot_id"]] = {
            "shot_no": a["shot_id"], "frame": a["frame"], "shot_id": a["shot_id"],
            "time": f"{int(a['t'] // 60):02d}:{a['t'] % 60:05.2f}",
            "hitter_role": a.get("role", ""), "hitter_side": a.get("side", ""),
            "true_type": type_name, "true_volley": "", "true_in": "",
            "notes": "skipped as unsure" if not type_name else "",
        }
        save_done(self.out_path, self.done)
        n = sum(1 for v in self.done.values() if v.get("true_type"))
        print(f"  shot {a['shot_id']} @ {a['t']:.2f}s -> "
              f"{type_name or 'UNSURE'}   ({n} labelled)")
        self.playing = False
        self.i += 1
        self.enter()

    def prev_shot(self):
        if self.i == 0:
            return
        if self.todo[self.i - 1]["lo"] not in self.frames:
            return
        self.i -= 1
        self.done.pop(self.todo[self.i]["shot_id"], None)
        save_done(self.out_path, self.done)
        self.cur = self.todo[self.i]["lo"]
        self.replay()

    def replay(self):
        self.cur = self.todo[self.i]["lo"]
        self.playing = True
        self.tick()

    def tick(self):
        if not self.playing:
            return
        a = self.todo[self.i]
        self.render()
        if self.cur >= a["hi"]:
            # Settle back on the CONTACT frame rather than the aftermath. The still image
            # the operator decides from should be the informative one -- who hit it and
            # where the ball was -- not wherever the clip happened to end 1.8 s later.
            self.playing = False
            self.cur = a["frame"]
            self.render()
            return
        self.cur += 1
        self.root.after(int(1000 / PLAY_FPS), self.tick)

    # -- drawing ----------------------------------------------------------
    def render(self):
        buf = self.frames.get(self.cur)
        if buf is None:
            return
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        s = self.scale
        a = self.todo[self.i]
        for k in range(TRAIL_FRAMES, 0, -1):
            q = self.balls.get(self.cur - k)
            if q:
                sh = int(80 + 140 * (1 - k / TRAIL_FRAMES))
                cv2.circle(img, (int(q[0] * s), int(q[1] * s)), 4, (0, sh, sh), -1)
        bp = self.balls.get(self.cur)
        if bp:
            cv2.circle(img, (int(bp[0] * s), int(bp[1] * s)), 15, (0, 255, 255), 2)
        # the contact frame gets a flash, so the shot being labelled is unambiguous
        if abs(self.cur - a["frame"]) <= 1:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                          (0, 200, 255), 6)
            cv2.putText(img, "CONTACT", (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 0), 6)
            cv2.putText(img, "CONTACT", (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 200, 255), 2)

        h, w = img.shape[:2]
        k = min(self.max_w / w, self.max_h / h)
        disp = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        hh, ww, _ = rgb.shape
        self.tk_image = tk.PhotoImage(data=f"P6 {ww} {hh} 255 ".encode() + rgb.tobytes(),
                                      format="PPM")
        if self.image_id is None:
            self.image_id = self.canvas.create_image(self.max_w // 2, self.max_h // 2,
                                                     anchor="center", image=self.tk_image)
        else:
            self.canvas.itemconfig(self.image_id, image=self.tk_image)
        n = sum(1 for v in self.done.values() if v.get("true_type"))
        self.status.set(f"  shot {self.i + 1}/{len(self.todo)}   t={a['t']:7.2f}s"
                        f"   hit by {a.get('role') or '?'} ({a.get('side') or '?'} side)"
                        f"   |   {n} labelled")

    def quit(self):
        save_done(self.out_path, self.done)
        print(f"\nsaved {self.out_path}")
        self.root.destroy()


def build_todo(clip: Path, fps: float, limit: Optional[int]) -> List[dict]:
    """Shots to offer, in time order.

    Time order rather than "least confident first" on purpose. A biased sample measures the
    hard cases and cannot be compared with the 37.5% we quote today, which is over an
    arbitrary set. An unbiased sequence also lets the operator follow the game, which is what
    makes each judgement quick.
    """
    cls = json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"]
    roles = {}
    rp = clip / "track_roles.json"
    if rp.exists():
        for role, info in json.loads(rp.read_text(encoding="utf-8"))["roles"].items():
            for tid in info.get("track_ids", []):
                roles[int(tid)] = role
    out = []
    for s in sorted(cls, key=lambda x: x["frame"]):
        out.append({"shot_id": int(s["shot_id"]), "frame": int(s["frame"]),
                    "t": float(s["t_sec"]),
                    "role": roles.get(int(s.get("track_id", -1)), ""),
                    "side": s.get("hitter_side") or ""})
    return out[:limit] if limit else out


def report(clip: Path) -> int:
    from collections import Counter
    rows = []
    for p in sorted((clip / "_labeling").glob("labels*.csv")):
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows += [(p.name, r) for r in csv.DictReader(f)]
    typed = [(n, r) for n, r in rows if (r.get("true_type") or "").strip()]
    print(f"{clip.name}: {len(typed)} labelled shots across "
          f"{len({n for n, _ in typed})} file(s)")
    c = Counter((r["true_type"] or "").strip().lower() for _, r in typed)
    for k, v in c.most_common():
        print(f"  {k:<14}{v:>4}")
    real = sum(v for k, v in c.items()
               if k in ("drive", "dink", "drop", "lob", "serve", "return", "reset"))
    print(f"\n  {real} real-type labels (what tools/score_shot_types scores against)")
    if c.get("drop", 0) < 10:
        print(f"  drops: {c.get('drop', 0)} — the class the classifier gets wrong 2 in 3, "
              f"and the one worth the most")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="only offer the first N shots (resume picks up where you stopped)")
    ap.add_argument("--report", action="store_true", help="show what is labelled and stop")
    a = ap.parse_args(argv)
    clip = a.clip
    if not clip.is_dir():
        fail(f"not a folder: {clip}")
    if a.report:
        return report(clip)
    if not (clip / "classified.json").exists():
        fail(f"{clip}/classified.json missing — analyse the clip first")
    video = clip / "video.mp4"
    if not video.exists():
        fail(f"no video: {video}")

    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    fps = float(court["video"]["fps"])
    cap = cv2.VideoCapture(str(video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    todo = build_todo(clip, fps, a.limit)
    for s in todo:
        s["lo"] = max(0, int(s["frame"] - PRE_S * fps))
        s["hi"] = min(n_frames - 1, int(s["frame"] + POST_S * fps))

    out_path = clip / "_labeling" / OUT_NAME
    done = load_done(out_path)
    left = [s for s in todo if s["shot_id"] not in done]
    print(f"{clip.name}: {len(todo)} shots, {len(done)} already labelled, {len(left)} to go")
    if not left:
        print("nothing left. --report shows what is there.")
        return 0

    jpeg_w = min(src_w, 1600)
    windows = [range(s["lo"], s["hi"] + 1) for s in left]
    last = max(w.stop for w in windows)
    print(f"decoding {sum(len(w) for w in windows)} frames, walking to frame {last} "
          f"({last / fps:.0f}s in). One forward pass — this is the slow part.")
    frames = decode_windows(video, windows, jpeg_w)

    ball = pd.read_parquet(clip / "ball.parquet")
    ball = ball[ball.visible]
    balls = {int(f): (float(x), float(y))
             for f, x, y in zip(ball.frame_idx, ball.pixel_x, ball.pixel_y)}

    root = tk.Tk()
    root.title(f"Label shot types — {clip.name}")
    Labeller(root, clip, fps, src_w, todo, frames, jpeg_w, balls, done, out_path)
    root.mainloop()
    return report(clip)


if __name__ == "__main__":
    raise SystemExit(main())

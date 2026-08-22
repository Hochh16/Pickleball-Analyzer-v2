r"""Mark the frame where the SERVER'S PADDLE HITS THE BALL, for a handful of serves.

Why this exists. Serve detections read ~0.8-1.2s later than `truth.json`'s `start_t_sec` on
every clip, which looked like a timing defect until the frames were pulled: on the court B
serve at truth 29.00s the ball is still at the server's body until 30.05s and only leaves at
~30.10s. So truth is about a second BEFORE the strike too, and every offset measured against
it is measured against a moving target. Nothing about serve timing can be scored until a real
strike time exists, and only a person watching the frames can supply one.

That is all this asks for: step to the frame where the paddle meets the ball, press ENTER.
Six serves is enough to settle the convention; more is better.

Frames are decoded ONCE, in a single forward pass, and held in memory as JPEG. The reader
never seeks backwards — `cap.set(CAP_PROP_POS_FRAMES)` is not frame-accurate on long-GOP
H.264 and has silently misfiled a whole labelling session before. Stepping in the window is
therefore instant, and the wait is all up front, on the console, where a wait is legible.

Usage:
    python tools/mark_serve_strikes.py data/pb_3_min_indoor_1_court_b
    python tools/mark_serve_strikes.py data/pb_5_minute_outdoor-7 --limit 6
    python tools/mark_serve_strikes.py data/pb_3_min_indoor_1_court_b --report

Keys:
    Right / d       next frame            Left / a    previous frame
    Up / w          +10 frames            Down / s    -10 frames
    Space           play / pause, slowly (a quarter speed)
    z               zoom in (follows the BALL when it is detected), or back out
    ENTER / m       MARK this frame as the paddle strike, go to the next serve
    n               skip this serve (server off-screen, or you cannot tell)
    b               back to the previous serve to redo it
    Esc             save and quit  (progress is saved after every mark anyway)

Writes serve_strikes.json in the clip folder. Re-running resumes: serves already marked are
skipped, so it is safe to do a few at a time.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
from tools.score_serves import truth_serves, detected_serves

SCHEMA_VERSION = 1
OUT_NAME = "serve_strikes.json"
PRE_S = 1.0          # window starts this far before the anchor
POST_S = 3.0         # ...and ends this far after (the strike ran ~1.1s late on the one case)
JPEG_QUALITY = 82
DISPLAY_FRACTION = 0.85
PLAY_FPS = 15        # deliberately slow; the operator asked for slower playback
ZOOM_BOX_PX = 1200   # source-pixel crop when zoomed
TRAIL_FRAMES = 8     # how many past ball positions to draw behind it


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


# ---------- what to mark ----------

def anchors(clip: Path, limit: Optional[int]) -> List[dict]:
    """Serve moments worth marking, each with whatever times we already believe.

    The operator's own serve times come first because those are the ones the scorers use;
    the detected time rides along so the marked frame can be compared against both at once.
    """
    truth = truth_serves(clip)
    det = []
    try:
        det = detected_serves(clip)
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    if truth:
        src = "truth.json / ledger"
        times = list(truth)
    elif det:
        src = "detected rallies (no operator serve times for this clip)"
        times = list(det)
    else:
        fail(f"no serve times for {clip.name}: need truth.json or rallies.json")
    out = []
    for i, t in enumerate(sorted(times)):
        near = min(det, key=lambda g: abs(g - t)) if det else None
        out.append({"serve_index": i, "anchor_t_sec": round(float(t), 3),
                    "anchor_source": src,
                    "detected_t_sec": (round(float(near), 3)
                                       if near is not None and abs(near - t) <= 3.0 else None)})
    return out[:limit] if limit else out


def load_marks(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {int(s["serve_index"]): s for s in doc.get("strikes", [])}


def save_marks(path: Path, clip: Path, fps: float, marks: Dict[int, dict]) -> None:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "clip": clip.name,
        "fps": float(fps),
        "source": "operator marked the frame where the paddle meets the ball",
        "note": ("t_sec is the STRIKE. truth.json's start_t_sec is a different moment -- "
                 "measured about a second earlier -- so do not compare the two as if they "
                 "were the same quantity."),
        "strikes": [marks[k] for k in sorted(marks)],
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------- decoding ----------

def decode_windows(video: Path, windows: List[range], scale_to: int, log=print) -> Dict[int, bytes]:
    """One forward pass; keep the frames inside the windows, JPEG-encoded, skip the rest."""
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
            log(f"   {f}/{hi} frames ({f / hi:.0%}), ~{(hi - f) / max(rate, 1e-6):.0f}s left",
                flush=True)
    cap.release()
    log(f"   decoded {len(got)} frames in {time.time() - t0:.0f}s "
        f"({sum(len(v) for v in got.values()) / 1e6:.0f} MB held)")
    return got


# ---------- overlays ----------

def player_boxes(clip: Path, net_y: float) -> Dict[int, List[dict]]:
    """In-court player boxes per frame, tagged with role and distance from the net.

    Used only to HINT at who is serving -- see compose(). Nothing here decides anything:
    picking the server automatically is what made two earlier attempts at cropping these
    serves land on the wrong person, so the operator gets the whole frame and a suggestion.
    """
    p = clip / "players.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    roles = {}
    rp = clip / "track_roles.json"
    if rp.exists():
        doc = json.loads(rp.read_text(encoding="utf-8")).get("roles", {})
        for role, info in doc.items():
            for tid in info.get("track_ids", []):
                roles[int(tid)] = role
    df = df[df.get("in_court", True) & ~df.get("transient", False)]
    out: Dict[int, List[dict]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(int(r.frame), []).append({
            "bbox": (float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2)),
            "role": roles.get(int(r.track_id), f"track {int(r.track_id)}"),
            "from_net": (abs(float(r.court_y_ft) - net_y)
                         if getattr(r, "court_pos_reliable", False)
                         and not np.isnan(r.court_y_ft) else -1.0)})
    return out


def ball_positions(clip: Path) -> Dict[int, tuple]:
    b = pd.read_parquet(clip / "ball.parquet")
    b = b[b.visible]
    return {int(f): (float(x), float(y))
            for f, x, y in zip(b.frame_idx, b.pixel_x, b.pixel_y)}


# ---------- UI ----------

class Marker:
    def __init__(self, root, clip: Path, fps: float, src_w: int,
                 todo: List[dict], frames: Dict[int, bytes], jpeg_w: int,
                 boxes, balls, marks: Dict[int, dict], out_path: Path):
        self.root, self.clip, self.fps = root, clip, fps
        self.todo, self.frames, self.marks, self.out_path = todo, frames, marks, out_path
        self.boxes, self.balls = boxes, balls
        self.jpeg_scale = jpeg_w / float(src_w)     # source px -> stored px
        self.i = 0
        self.zoom = False
        self.playing = False
        self.tk_image = None
        self.image_id = None

        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        self.max_w = int(screen_w * DISPLAY_FRACTION)
        self.max_h = int(screen_h * DISPLAY_FRACTION) - 90
        self.canvas = tk.Canvas(root, width=self.max_w, height=self.max_h, bg="black",
                                highlightthickness=0)
        self.canvas.pack()
        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, anchor="w", justify="left",
                 font=("Consolas", 11)).pack(fill="x")
        self.help = tk.StringVar()
        self.help.set("ENTER mark strike   n skip   b previous serve   "
                      "arrows/wasd step   SPACE play   z zoom   Esc save+quit")
        tk.Label(root, textvariable=self.help, anchor="w",
                 font=("Consolas", 10), fg="#555").pack(fill="x")

        for k in ("<Right>", "<d>", "<D>"):
            root.bind(k, lambda e: self.step(1))
        for k in ("<Left>", "<a>", "<A>"):
            root.bind(k, lambda e: self.step(-1))
        for k in ("<Up>", "<w>", "<W>"):
            root.bind(k, lambda e: self.step(10))
        for k in ("<Down>", "<s>", "<S>"):
            root.bind(k, lambda e: self.step(-10))
        for k in ("<Return>", "<m>", "<M>"):
            root.bind(k, lambda e: self.mark())
        root.bind("<n>", lambda e: self.skip())
        root.bind("<N>", lambda e: self.skip())
        root.bind("<b>", lambda e: self.prev_serve())
        root.bind("<B>", lambda e: self.prev_serve())
        root.bind("<z>", lambda e: self.toggle_zoom())
        root.bind("<Z>", lambda e: self.toggle_zoom())
        root.bind("<space>", lambda e: self.toggle_play())
        root.bind("<Escape>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.enter_serve()

    # -- serve navigation ------------------------------------------------
    def enter_serve(self):
        while self.i < len(self.todo) and self.todo[self.i]["serve_index"] in self.marks:
            self.i += 1
        if self.i >= len(self.todo):
            self.quit()
            return
        a = self.todo[self.i]
        self.cur = a["frame_lo"]
        self.render()

    def prev_serve(self):
        if self.i == 0:
            return
        # Only serves whose window was decoded this run can be revisited -- a resumed session
        # decodes just what is left to do, so an earlier serve may have no frames in memory.
        if self.todo[self.i - 1]["frame_lo"] not in self.frames:
            self.help.set("that serve was marked in an earlier run and its frames are not "
                          "loaded -- delete its entry from serve_strikes.json to redo it")
            return
        self.i -= 1
        self.marks.pop(self.todo[self.i]["serve_index"], None)
        save_marks(self.out_path, self.clip, self.fps, self.marks)
        self.cur = self.todo[self.i]["frame_lo"]
        self.render()

    def mark(self):
        a = self.todo[self.i]
        self.marks[a["serve_index"]] = {
            "serve_index": a["serve_index"],
            "anchor_t_sec": a["anchor_t_sec"],
            "anchor_source": a["anchor_source"],
            "detected_t_sec": a["detected_t_sec"],
            "frame": int(self.cur),
            "t_sec": round(self.cur / self.fps, 3),
            "skipped": False,
        }
        save_marks(self.out_path, self.clip, self.fps, self.marks)
        print(f"  serve {a['serve_index']}: strike at {self.cur / self.fps:.3f}s "
              f"(anchor {a['anchor_t_sec']:.2f}s, "
              f"{self.cur / self.fps - a['anchor_t_sec']:+.2f}s)")
        self.playing = False
        self.i += 1
        self.enter_serve()

    def skip(self):
        a = self.todo[self.i]
        self.marks[a["serve_index"]] = {
            "serve_index": a["serve_index"], "anchor_t_sec": a["anchor_t_sec"],
            "anchor_source": a["anchor_source"], "detected_t_sec": a["detected_t_sec"],
            "frame": None, "t_sec": None, "skipped": True}
        save_marks(self.out_path, self.clip, self.fps, self.marks)
        print(f"  serve {a['serve_index']}: skipped")
        self.playing = False
        self.i += 1
        self.enter_serve()

    # -- frame navigation ------------------------------------------------
    def step(self, n):
        a = self.todo[self.i]
        self.cur = int(np.clip(self.cur + n, a["frame_lo"], a["frame_hi"]))
        self.render()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.tick()

    def tick(self):
        if not self.playing:
            return
        a = self.todo[self.i]
        if self.cur >= a["frame_hi"]:
            self.playing = False
            return
        self.cur += 1
        self.render()
        self.root.after(int(1000 / PLAY_FPS), self.tick)

    def toggle_zoom(self):
        self.zoom = not self.zoom
        self.render()

    # -- drawing ---------------------------------------------------------
    def compose(self, frame_idx: int) -> Optional[np.ndarray]:
        """The annotated frame, before it is scaled to the window. Split out so a preview
        can be written to a file with exactly what the operator will be looking at."""
        buf = self.frames.get(frame_idx)
        if buf is None:
            return None
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        s = self.jpeg_scale

        here = self.boxes.get(frame_idx, [])
        bp = self.balls.get(frame_idx)
        # Who is serving. The ball starts in the server's hand, so when it is detected the
        # nearest player is a far better hint than "furthest from the net" -- that rule put
        # the box on the wrong person on the first serve tried. Fall back to it only before
        # the ball appears.
        if bp and here:
            server = min(here, key=lambda q: (abs((q["bbox"][0] + q["bbox"][2]) / 2 - bp[0])
                                              + abs((q["bbox"][1] + q["bbox"][3]) / 2 - bp[1])))
            why = "nearest the ball"
        elif here:
            server = max(here, key=lambda q: q["from_net"])
            why = "furthest from net"
        else:
            server, why = None, ""
        for p in here:
            x1, y1, x2, y2 = [int(v * s) for v in p["bbox"]]
            hit = p is server
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (0, 220, 255) if hit else (120, 120, 120), 2 if hit else 1)
            tag = p["role"] + (f" <- {why}" if hit else "")
            cv2.putText(img, tag, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3)
            cv2.putText(img, tag, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 220, 255) if hit else (200, 200, 200), 1)
        # A short trail: at the strike the ball's direction reverses, and that is far easier
        # to see as a shape than as one dot moving between key presses.
        for k in range(TRAIL_FRAMES, 0, -1):
            q = self.balls.get(frame_idx - k)
            if q:
                shade = int(90 + 120 * (1 - k / TRAIL_FRAMES))
                cv2.circle(img, (int(q[0] * s), int(q[1] * s)), 4, (0, shade, shade), -1)
        if bp:
            cv2.circle(img, (int(bp[0] * s), int(bp[1] * s)), 16, (0, 255, 255), 2)
            cv2.circle(img, (int(bp[0] * s), int(bp[1] * s)), 2, (0, 255, 255), -1)

        if self.zoom and (bp or server is not None):
            if bp:                      # the ball is the thing being watched
                cx, cy = bp[0] * s, bp[1] * s
            else:
                x1, y1, x2, y2 = [v * s for v in server["bbox"]]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = ZOOM_BOX_PX * s / 2
            h, w = img.shape[:2]
            a0 = int(np.clip(cx - half, 0, max(0, w - 2 * half)))
            b0 = int(np.clip(cy - half, 0, max(0, h - 2 * half)))
            img = img[b0:b0 + int(2 * half), a0:a0 + int(2 * half)]
        return img

    def render(self):
        img = self.compose(self.cur)
        if img is None:
            return

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

        a = self.todo[self.i]
        t = self.cur / self.fps
        det = (f"{a['detected_t_sec']:.2f}s ({t - a['detected_t_sec']:+.2f})"
               if a["detected_t_sec"] is not None else "none")
        done = sum(1 for m in self.marks.values() if not m["skipped"])
        self.status.set(
            f" serve {self.i + 1}/{len(self.todo)}   frame {self.cur}   t={t:7.3f}s"
            f"   operator anchor {a['anchor_t_sec']:.2f}s ({t - a['anchor_t_sec']:+.2f})"
            f"   detected {det}"
            f"   |   marked {done}   {'ZOOM' if self.zoom else 'full frame'}"
            f"{'   PLAYING' if self.playing else ''}")

    def quit(self):
        save_marks(self.out_path, self.clip, self.fps, self.marks)
        print(f"\nsaved {self.out_path}")
        self.root.destroy()


# ---------- report ----------

def report(clip: Path) -> int:
    path = clip / OUT_NAME
    if not path.exists():
        fail(f"{path} does not exist yet -- run without --report first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = [s for s in doc["strikes"] if not s["skipped"]]
    if not rows:
        print("no strikes marked yet")
        return 0
    print(f"{clip.name}: {len(rows)} marked strike(s)\n")
    print(f"  {'strike':>9}{'operator anchor':>17}{'anchor - strike':>17}"
          f"{'detected':>10}{'detected - strike':>19}")
    da, dd = [], []
    for s in rows:
        a = s["anchor_t_sec"] - s["t_sec"]
        da.append(a)
        d = (s["detected_t_sec"] - s["t_sec"]) if s["detected_t_sec"] is not None else None
        if d is not None:
            dd.append(d)
        det_s = "-" if s["detected_t_sec"] is None else f"{s['detected_t_sec']:.2f}s"
        d_s = "-" if d is None else f"{d:+.2f}s"
        print(f"  {s['t_sec']:>8.2f}s{s['anchor_t_sec']:>16.2f}s{a:>+16.2f}s"
              f"{det_s:>10}{d_s:>19}")
    print()
    print(f"  operator anchor sits {np.median(da):+.2f}s from the strike (median of {len(da)})")
    if dd:
        print(f"  detection sits       {np.median(dd):+.2f}s from the strike "
              f"(median of {len(dd)})")
        print(f"  |detection - strike| median {np.median(np.abs(dd)):.2f}s, "
              f"worst {np.max(np.abs(dd)):.2f}s")
    return 0


# ---------- main ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="only offer the first N serves (six is enough to settle it)")
    ap.add_argument("--report", action="store_true",
                    help="print what has been marked so far and stop")
    a = ap.parse_args(argv)
    clip = a.clip
    if not clip.is_dir():
        fail(f"not a folder: {clip}")
    if a.report:
        return report(clip)

    video = clip / "video.mp4"
    if not video.exists():
        fail(f"no video: {video}")
    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    fps = float(court["video"]["fps"])
    net_y = float(court.get("net_y_ft", 22.0))

    cap = cv2.VideoCapture(str(video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    todo = anchors(clip, a.limit)
    for t in todo:
        c = t["anchor_t_sec"]
        t["frame_lo"] = max(0, int((c - PRE_S) * fps))
        t["frame_hi"] = min(n_frames - 1, int((c + POST_S) * fps))

    out_path = clip / OUT_NAME
    marks = load_marks(out_path)
    left = [t for t in todo if t["serve_index"] not in marks]
    print(f"{clip.name}: {len(todo)} serve(s) to offer, {len(marks)} already marked, "
          f"{len(left)} to go")
    if not left:
        print("nothing left to mark. --report shows what is there.")
        return 0

    jpeg_w = min(src_w, 1920)
    windows = [range(t["frame_lo"], t["frame_hi"] + 1) for t in left]
    total = sum(len(w) for w in windows)
    last = max(w.stop for w in windows)
    print(f"decoding {total} frames across {len(windows)} window(s), walking to frame {last} "
          f"({last / fps:.0f}s in). One forward pass, no backward seeks -- this is the slow "
          f"part, a few minutes on a 4K clip. The window opens when it finishes.")
    frames = decode_windows(video, windows, jpeg_w)

    boxes = player_boxes(clip, net_y)
    balls = ball_positions(clip)

    root = tk.Tk()
    root.title(f"Mark serve strikes - {clip.name}")
    Marker(root, clip, fps, src_w, todo, frames, jpeg_w, boxes, balls, marks, out_path)
    root.mainloop()
    return report(clip)


if __name__ == "__main__":
    raise SystemExit(main())

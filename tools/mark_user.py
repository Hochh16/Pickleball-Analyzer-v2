r"""
Stage 2.5 user-clicker — desktop Tkinter UI for producing user_clicks.json
files that Stage 2.5 (classify_tracks) consumes to seed the `user` role.

Sibling of tools/mark_court.py (same frame-loading / scaling / atomic-save
pattern). Where mark_court.py captures 8 fixed court points on ONE frame, this
captures one click PER chosen frame marking where YOU (the user) are, across a
handful of well-separated frames.

Usage:
    python tools\mark_user.py --video data\pb_2min\video.mp4 --out data\pb_2min\user_clicks.json

UX:
    1. Window opens on the video's first frame.
    2. Scrub to a frame where YOU are clearly visible and well-separated from
       your partner (so the click maps unambiguously to your track).
    3. Left-click on your body (torso/center). That records {frame, x, y}.
    4. Scrub to another spread-out frame, click again. Aim for ~5 clicks across
       the start / middle / end of the clip, ideally with you on both sides of
       the court if you switch ends.
    5. Backspace = undo last click. "Save" writes user_clicks.json.

If the file at --out exists, it loads existing clicks and lets you add/undo.
Coordinates are stored in original-video pixel space regardless of display
downscale, matching the format Stage 2.5 expects.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import ttk, messagebox


DEFAULT_MAX_DISPLAY_FRACTION = 0.8
SUGGESTED_CLICKS = 5


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_or_init_clicks(out_path: Path) -> list:
    """Load existing clicks if present, else return an empty list."""
    if not out_path.exists():
        return []
    try:
        with out_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        fail(f"existing {out_path} is malformed JSON: {e}")
    clicks = data.get("clicks", [])
    out = []
    for c in clicks:
        try:
            out.append({"frame": int(c["frame"]), "x": int(c["x"]), "y": int(c["y"])})
        except (KeyError, TypeError, ValueError):
            fail(f"existing {out_path} has a malformed click entry: {c!r}")
    return out


def save_clicks(out_path: Path, clicks: list) -> None:
    """Atomic save: write to .tmp then rename. Matches the compact one-line-per-
    click style of the existing test_clip/user_clicks.json."""
    clicks = sorted(clicks, key=lambda c: c["frame"])
    lines = ["{", '  "clicks": [']
    body = [
        f'    {{"frame": {c["frame"]}, "x": {c["x"]}, "y": {c["y"]}}}'
        for c in clicks
    ]
    lines.append(",\n".join(body))
    lines.append("  ]")
    lines.append("}")
    text = "\n".join(lines) + "\n"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
    tmp.replace(out_path)


# ---------- Video access (mirrors mark_court.py) ----------

class VideoReader:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            fail(f"could not open {path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self, idx: int):
        idx = max(0, min(self.n_frames - 1, idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


# ---------- App ----------

class UserClickApp:
    def __init__(self, root, video, out_path, clicks, max_display_fraction):
        self.root = root
        self.video = video
        self.out_path = out_path
        self.clicks = clicks  # list of {"frame", "x", "y"}

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        avail_h = int(screen_h * max_display_fraction) - 160
        avail_w = int(screen_w * max_display_fraction)
        self.display_scale = min(1.0, avail_w / video.width, avail_h / video.height)
        self.display_w = int(video.width * self.display_scale)
        self.display_h = int(video.height * self.display_scale)
        print(f"Display scale: {self.display_scale:.3f} "
              f"({self.display_w}x{self.display_h}) from {video.width}x{video.height}")

        self.current_frame_idx = 0

        top = ttk.Frame(root, padding=6)
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Click on YOURSELF; scrub to ~5 spread-out frames."
                  ).pack(side="left")
        ttk.Button(top, text="Save", command=self.on_save).pack(side="right")
        ttk.Button(top, text="Undo last click", command=self.on_undo
                   ).pack(side="right", padx=(0, 6))
        ttk.Button(top, text="Clear all", command=self.on_clear
                   ).pack(side="right", padx=(0, 6))

        self.canvas = tk.Canvas(root, width=self.display_w, height=self.display_h,
                                bg="black", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(side="top")

        scrub = ttk.Frame(root, padding=4)
        scrub.pack(side="top", fill="x")
        ttk.Label(scrub, text="Frame:").pack(side="left")
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(scrub, from_=0, to=video.n_frames - 1,
                                      orient="horizontal", variable=self.frame_var,
                                      command=self.on_scrub)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.frame_label_var = tk.StringVar()
        ttk.Label(scrub, textvariable=self.frame_label_var, width=18).pack(side="right")

        self.status_var = tk.StringVar()
        ttk.Label(root, textvariable=self.status_var, anchor="w", padding=4
                  ).pack(side="top", fill="x")

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Motion>", self.on_motion)
        root.bind("<BackSpace>", lambda e: self.on_undo())
        root.bind("<Escape>", lambda e: self.on_quit())
        root.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.tk_image = None
        self.canvas_image_id = None
        self.artifacts = []
        self.reticle_v = self.reticle_h = None
        self.render_frame()

    def render_frame(self):
        frame = self.video.read_frame(self.current_frame_idx)
        if frame is None:
            self.status_var.set(f"ERROR reading frame {self.current_frame_idx}")
            return
        if self.display_scale < 1.0:
            disp = cv2.resize(frame, (self.display_w, self.display_h),
                              interpolation=cv2.INTER_AREA)
        else:
            disp = frame
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        ppm = f"P6 {w} {h} 255 ".encode("ascii") + rgb.tobytes()
        self.tk_image = tk.PhotoImage(data=ppm, format="PPM")
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw",
                                                            image=self.tk_image)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)
        self.redraw_clicks()
        self.update_status()

    def redraw_clicks(self):
        for item in self.artifacts:
            self.canvas.delete(item)
        self.artifacts.clear()
        for i, c in enumerate(self.clicks):
            on_this = (c["frame"] == self.current_frame_idx)
            color = "#30ff30" if on_this else "#ffaa30"
            x = c["x"] * self.display_scale
            y = c["y"] * self.display_scale
            r = 7
            oid = self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                          outline=color, width=2, fill="")
            self.artifacts.append(oid)
            tid = self.canvas.create_text(x + r + 3, y - r - 3,
                                          text=f"{i+1}@f{c['frame']}", fill=color,
                                          anchor="sw", font=("Consolas", 9, "bold"))
            self.artifacts.append(tid)

    def update_status(self):
        n = len(self.clicks)
        self.frame_label_var.set(f"{self.current_frame_idx} / {self.video.n_frames - 1}")
        msg = f"{n} click(s) placed (aim for ~{SUGGESTED_CLICKS}). "
        if n < 2:
            msg += "Click on yourself, then scrub to another frame and click again."
        else:
            msg += "Green = click on THIS frame, orange = other frames. Save when done."
        self.status_var.set(msg)

    def on_left_click(self, event):
        src_x = int(round(event.x / self.display_scale))
        src_y = int(round(event.y / self.display_scale))
        self.clicks.append({"frame": int(self.current_frame_idx),
                            "x": src_x, "y": src_y})
        self.render_frame()

    def on_motion(self, event):
        for item in (self.reticle_v, self.reticle_h):
            if item is not None:
                self.canvas.delete(item)
        self.reticle_v = self.canvas.create_line(event.x, 0, event.x, self.display_h,
                                                 fill="#ff00ff", width=1)
        self.reticle_h = self.canvas.create_line(0, event.y, self.display_w, event.y,
                                                 fill="#ff00ff", width=1)

    def on_undo(self):
        if self.clicks:
            self.clicks.pop()
            self.render_frame()

    def on_clear(self):
        if self.clicks and messagebox.askyesno("Clear all", "Remove all clicks?"):
            self.clicks = []
            self.render_frame()

    def on_scrub(self, val):
        new_idx = int(float(val))
        if new_idx != self.current_frame_idx:
            self.current_frame_idx = new_idx
            self.render_frame()

    def on_save(self):
        if len(self.clicks) < 2:
            if not messagebox.askyesno(
                    "Few clicks",
                    f"You have {len(self.clicks)} click(s). ~{SUGGESTED_CLICKS} "
                    f"spread-out clicks give Stage 2.5 the best chance. Save anyway?"):
                return
        save_clicks(self.out_path, self.clicks)
        self.status_var.set(f"Saved {len(self.clicks)} clicks to {self.out_path.name}. "
                            f"You can close this window.")
        print(f"saved {len(self.clicks)} clicks -> {self.out_path}")
        messagebox.showinfo("Saved",
                            f"Wrote {self.out_path}\n\n{len(self.clicks)} clicks.\n"
                            f"You can close this window.")

    def on_quit(self):
        if self.clicks and messagebox.askyesno("Save before quitting?",
                                               f"Save {len(self.clicks)} clicks before quitting?"):
            self.on_save()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="Stage 2.5 user-clicker")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output user_clicks.json path.")
    ap.add_argument("--max-display-fraction", type=float,
                    default=DEFAULT_MAX_DISPLAY_FRACTION, dest="max_display_fraction")
    args = ap.parse_args()

    if not args.video.exists():
        fail(f"video not found: {args.video}")
    if not (0.1 <= args.max_display_fraction <= 1.0):
        fail("--max-display-fraction must be in [0.1, 1.0]")

    print(f"opening {args.video}")
    video = VideoReader(args.video)
    print(f"video: {video.n_frames} frames, {video.fps:.2f} fps, "
          f"{video.width}x{video.height}")
    clicks = load_or_init_clicks(args.out)

    root = tk.Tk()
    root.title(f"mark_user — {args.out.name}")
    UserClickApp(root, video, args.out, clicks, args.max_display_fraction)
    print()
    print("UI controls:")
    print("  Left-click    Mark yourself on the current frame.")
    print("  Backspace     Undo last click.")
    print("  Frame slider  Scrub to another frame.")
    print("  Esc           Quit (prompts to save).")
    print()
    root.mainloop()
    video.release()


if __name__ == "__main__":
    main()

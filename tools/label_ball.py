"""
Stage 4.5 labeling tool — desktop UI for clicking ball locations
in video frames.

Usage:
    python tools\label_ball.py --video data\test_clip\video.mp4 --out data\test_clip\ball_labels.json

Optional args:
    --sample-every N      Label every Nth frame (default 3).
    --start-frame N       Start at frame N (default 0).
    --end-frame N         Stop at frame N inclusive (default end of video).
    --max-display-fraction F   Max fraction of screen to use (default 0.8).

UX:
    Left-click       Mark ball at click position; advance to next sampled frame.
    Spacebar         Mark "ball not visible"; advance.
    Right-click      Same as Spacebar (mark not visible; advance).
    Backspace        Go back one labeled frame (to fix a misclick).
    Left-arrow       Same as Backspace.
    Esc              Save and quit.
    Window-close     Save and quit.

Auto-saves every 25 labels and on quit. Crash-safe — if you re-launch
with the same --out, the tool resumes from the last labeled frame.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

SCHEMA_VERSION = 1
AUTOSAVE_EVERY_N_LABELS = 25
DEFAULT_SAMPLE_EVERY = 3
DEFAULT_MAX_DISPLAY_FRACTION = 0.8
PER_VIDEO_MIN_LABELS = 200
PER_VIDEO_TARGET_LABELS = 250


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- Persistence ----------

def load_or_init_labels(out_path: Path, video_path: Path,
                        n_frames: int, fps: float, w: int, h: int,
                        sample_every: int) -> dict:
    """Load existing labels file, or initialize a new one."""
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            fail(f"existing {out_path} is malformed JSON: {e}\n"
                 f"Delete it or rename it to start fresh.")

        if data.get("schema_version") != SCHEMA_VERSION:
            fail(f"{out_path} schema_version is {data.get('schema_version')!r}, "
                 f"expected {SCHEMA_VERSION}")
        # Sanity: video matches
        if data.get("video_frame_count") != n_frames:
            print(f"WARNING: existing labels file says video has "
                  f"{data.get('video_frame_count')} frames; current video "
                  f"has {n_frames}. Continuing but check that you opened "
                  f"the right video.")
        if data.get("sample_every") != sample_every:
            print(f"WARNING: existing labels used sample_every="
                  f"{data.get('sample_every')}; current run uses {sample_every}. "
                  f"You may end up with mixed sample rates.")
        n_existing = len(data.get("labels", []))
        print(f"Loaded {n_existing} existing labels from {out_path}")
        return data

    # New file
    return {
        "schema_version": SCHEMA_VERSION,
        "video_path": str(video_path),
        "video_frame_count": n_frames,
        "video_fps": fps,
        "video_width": w,
        "video_height": h,
        "sample_every": sample_every,
        "labels": [],
        "started_at_utc": utc_now_iso(),
        "last_saved_at_utc": utc_now_iso(),
    }


def save_labels(out_path: Path, data: dict) -> None:
    """Atomic save: write to .tmp then rename."""
    data["last_saved_at_utc"] = utc_now_iso()
    # Sort by frame_idx ascending; deduplicate keeping the last entry
    seen = {}
    for entry in data["labels"]:
        seen[entry["frame_idx"]] = entry
    data["labels"] = sorted(seen.values(), key=lambda e: e["frame_idx"])

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(out_path)


# ---------- Video access ----------

class VideoReader:
    """Wraps cv2.VideoCapture with a simple frame-index API."""
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            fail(f"could not open {path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self, idx: int):
        if idx < 0 or idx >= self.n_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()


# ---------- Main UI app ----------

class LabelApp:
    def __init__(self, root: tk.Tk, video: VideoReader, out_path: Path,
                 data: dict, frame_indices: list, max_display_fraction: float):
        self.root = root
        self.video = video
        self.out_path = out_path
        self.data = data
        self.frame_indices = frame_indices  # list of frame_idx values to label, in order
        self.unsaved_count = 0

        # Build a lookup: frame_idx -> position in frame_indices list
        self.idx_to_pos = {f: i for i, f in enumerate(frame_indices)}

        # Find starting position — the first unlabeled frame in the sample list
        labeled_frames = {entry["frame_idx"] for entry in data["labels"]}
        self.position = 0
        for i, fi in enumerate(frame_indices):
            if fi not in labeled_frames:
                self.position = i
                break
        else:
            # All frames already labeled
            self.position = len(frame_indices) - 1

        # Compute display scaling
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        max_w = int(screen_w * max_display_fraction)
        max_h = int(screen_h * max_display_fraction)
        scale_w = max_w / video.width
        scale_h = max_h / video.height
        self.display_scale = min(1.0, scale_w, scale_h)
        self.display_w = int(video.width * self.display_scale)
        self.display_h = int(video.height * self.display_scale)
        print(f"Display scale: {self.display_scale:.3f} "
              f"({self.display_w}x{self.display_h}) "
              f"from source ({video.width}x{video.height})")

        # Build UI
        self.canvas = tk.Canvas(root, width=self.display_w,
                                height=self.display_h, bg="black",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack()
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(root, textvariable=self.status_var,
                                     anchor="w", font=("Consolas", 10))
        self.status_label.pack(fill="x")

        # Bindings
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_motion)
        root.bind("<space>", self.on_space)
        root.bind("<BackSpace>", self.on_back)
        root.bind("<Left>", self.on_back)
        root.bind("<Escape>", self.on_quit)
        root.protocol("WM_DELETE_WINDOW", self.on_quit)

        # State
        self.current_image_id = None  # canvas item id
        self.tk_image = None  # keep reference so it doesn't get GC'd
        self.reticle_v = None
        self.reticle_h = None
        self.last_known_label_circle = None

        # Initial render
        self.render_current_frame()

    # ----- Rendering -----

    def render_current_frame(self):
        if self.position >= len(self.frame_indices):
            self.show_finished()
            return

        fi = self.frame_indices[self.position]
        frame = self.video.read_frame(fi)
        if frame is None:
            self.set_status(f"ERROR reading frame {fi}; press right arrow or "
                            f"backspace to go elsewhere")
            return

        # Resize for display
        if self.display_scale < 1.0:
            disp = cv2.resize(frame,
                              (self.display_w, self.display_h),
                              interpolation=cv2.INTER_AREA)
        else:
            disp = frame
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        # Convert to PhotoImage via PPM bytes (no PIL dependency)
        h, w, _ = rgb.shape
        ppm_header = f"P6 {w} {h} 255 ".encode("ascii")
        self.tk_image = tk.PhotoImage(data=ppm_header + rgb.tobytes(),
                                      format="PPM")
        if self.current_image_id is None:
            self.current_image_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self.tk_image)
        else:
            self.canvas.itemconfig(self.current_image_id, image=self.tk_image)

        # Clear reticle (will be redrawn on next mouse move)
        for item in (self.reticle_v, self.reticle_h):
            if item is not None:
                self.canvas.delete(item)
        self.reticle_v = None
        self.reticle_h = None

        # If this frame already has a label, draw a marker circle
        if self.last_known_label_circle is not None:
            self.canvas.delete(self.last_known_label_circle)
            self.last_known_label_circle = None
        existing = self.lookup_label(fi)
        if existing is not None and existing.get("ball_visible"):
            disp_x = existing["pixel_x"] * self.display_scale
            disp_y = existing["pixel_y"] * self.display_scale
            r = 8
            self.last_known_label_circle = self.canvas.create_oval(
                disp_x - r, disp_y - r, disp_x + r, disp_y + r,
                outline="#00ff00", width=2)

        self.update_status()

    def update_status(self):
        if self.position >= len(self.frame_indices):
            return
        fi = self.frame_indices[self.position]
        n_labeled = len(self.data["labels"])
        n_total = len(self.frame_indices)
        pct = 100.0 * (self.position + 1) / n_total
        existing = self.lookup_label(fi)
        existing_str = ""
        if existing is not None:
            if existing["ball_visible"]:
                existing_str = (f"  [already labeled at "
                                f"({existing['pixel_x']:.0f}, "
                                f"{existing['pixel_y']:.0f}) — "
                                f"green circle]")
            else:
                existing_str = "  [already labeled as not visible]"
        self.status_var.set(
            f"frame {fi}  |  position {self.position + 1}/{n_total} ({pct:.1f}%)  |  "
            f"labels saved: {n_labeled}{existing_str}"
        )
        self.root.title(f"label_ball — {self.out_path.name} — "
                        f"frame {fi} ({self.position + 1}/{n_total})")

    def show_finished(self):
        n_labeled = len(self.data["labels"])
        self.set_status(
            f"FINISHED — all {len(self.frame_indices)} sampled frames seen. "
            f"Labels saved: {n_labeled}. Esc to quit."
        )

    def set_status(self, msg: str):
        self.status_var.set(msg)

    # ----- Lookup helpers -----

    def lookup_label(self, frame_idx: int):
        for entry in self.data["labels"]:
            if entry["frame_idx"] == frame_idx:
                return entry
        return None

    def upsert_label(self, frame_idx: int, ball_visible: bool,
                     pixel_x, pixel_y):
        # Remove any existing entry for this frame
        self.data["labels"] = [e for e in self.data["labels"]
                               if e["frame_idx"] != frame_idx]
        self.data["labels"].append({
            "frame_idx": frame_idx,
            "ball_visible": ball_visible,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
        })
        self.unsaved_count += 1
        if self.unsaved_count >= AUTOSAVE_EVERY_N_LABELS:
            save_labels(self.out_path, self.data)
            self.unsaved_count = 0
            print(f"  auto-saved ({len(self.data['labels'])} labels total)")

    # ----- Event handlers -----

    def on_left_click(self, event):
        if self.position >= len(self.frame_indices):
            return
        fi = self.frame_indices[self.position]
        # Convert display coords to source coords
        src_x = event.x / self.display_scale
        src_y = event.y / self.display_scale
        self.upsert_label(fi, True, float(src_x), float(src_y))
        self.position += 1
        self.render_current_frame()

    def on_right_click(self, event):
        self.mark_not_visible()

    def on_space(self, event):
        self.mark_not_visible()

    def mark_not_visible(self):
        if self.position >= len(self.frame_indices):
            return
        fi = self.frame_indices[self.position]
        self.upsert_label(fi, False, None, None)
        self.position += 1
        self.render_current_frame()

    def on_back(self, event):
        if self.position > 0:
            self.position -= 1
            self.render_current_frame()

    def on_motion(self, event):
        # Draw reticle (cross at cursor)
        for item in (self.reticle_v, self.reticle_h):
            if item is not None:
                self.canvas.delete(item)
        self.reticle_v = self.canvas.create_line(
            event.x, 0, event.x, self.display_h,
            fill="#ff00ff", width=1)
        self.reticle_h = self.canvas.create_line(
            0, event.y, self.display_w, event.y,
            fill="#ff00ff", width=1)

    def on_quit(self, event=None):
        save_labels(self.out_path, self.data)
        n = len(self.data["labels"])
        print(f"Saved {n} labels to {self.out_path}")
        if n < PER_VIDEO_MIN_LABELS:
            print(f"WARNING: only {n} labels — Stage 4.5 contract requires "
                  f">= {PER_VIDEO_MIN_LABELS} per video.")
        elif n < PER_VIDEO_TARGET_LABELS:
            print(f"Note: {n} labels meets minimum but is below the "
                  f"target of {PER_VIDEO_TARGET_LABELS} per video.")
        self.root.destroy()


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Stage 4.5 ball-labeling tool")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY,
                    dest="sample_every")
    ap.add_argument("--start-frame", type=int, default=0, dest="start_frame")
    ap.add_argument("--end-frame", type=int, default=None, dest="end_frame")
    ap.add_argument("--max-display-fraction", type=float,
                    default=DEFAULT_MAX_DISPLAY_FRACTION,
                    dest="max_display_fraction")
    args = ap.parse_args()

    if not args.video.exists():
        fail(f"video not found: {args.video}")
    if args.sample_every < 1:
        fail("--sample-every must be >= 1")
    if args.start_frame < 0:
        fail("--start-frame must be >= 0")
    if not (0.1 <= args.max_display_fraction <= 1.0):
        fail("--max-display-fraction must be in [0.1, 1.0]")

    print(f"opening {args.video}")
    video = VideoReader(args.video)
    print(f"video: {video.n_frames} frames, {video.fps:.2f} fps, "
          f"{video.width}x{video.height}")

    end_frame = args.end_frame if args.end_frame is not None else video.n_frames - 1
    if end_frame >= video.n_frames:
        end_frame = video.n_frames - 1
    if end_frame < args.start_frame:
        fail(f"--end-frame ({end_frame}) is before --start-frame "
             f"({args.start_frame})")

    # Frame triple needs frame_idx >= 2
    effective_start = max(args.start_frame, 2)
    frame_indices = list(range(effective_start, end_frame + 1,
                               args.sample_every))
    print(f"will sample {len(frame_indices)} frames "
          f"(every {args.sample_every}th from {effective_start} to {end_frame})")

    if len(frame_indices) == 0:
        fail("no frames to label given the start/end/sample-every settings")

    data = load_or_init_labels(args.out, args.video, video.n_frames,
                               video.fps, video.width, video.height,
                               args.sample_every)

    root = tk.Tk()
    root.title(f"label_ball — {args.out.name}")
    app = LabelApp(root, video, args.out, data, frame_indices,
                   args.max_display_fraction)

    print()
    print("UI controls:")
    print("  Left-click       Mark ball at click position; advance.")
    print("  Spacebar         Mark 'ball not visible'; advance.")
    print("  Right-click      Same as Spacebar.")
    print("  Backspace / <-   Go back one frame (to fix a misclick).")
    print("  Esc              Save and quit.")
    print()

    root.mainloop()
    video.release()


if __name__ == "__main__":
    main()
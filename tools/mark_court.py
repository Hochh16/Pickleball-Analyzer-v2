r"""
Stage 1 markers-clicker — desktop Tkinter UI for producing markers.json
files that Stage 1's CLI consumes.

This tool replaces the React frontend originally specified in Stage 1's
contract.md. It produces a markers.json with the exact same schema that
the FastAPI backend expects, then user runs the existing Stage 1 CLI:

    python -m stages.calibrate.calibrate \
        --video <video> --markers <markers.json> --out-dir <dir>

Usage:
    python tools\mark_court.py --video data\indoor_b\video.mp4 --out data\indoor_b\markers.json

UX:
    1. Window opens showing the video's first frame.
    2. Three dropdowns at the top: dominant_hand, user_baseline, user_starting_corner.
    3. Frame scrubber (slider) to find a frame with all corners + kitchen lines visible.
    4. Click 8 points in order:
       1. Court bottom-left corner (image position)
       2. Court bottom-right corner
       3. Court top-right corner
       4. Court top-left corner
       5. User-kitchen line LEFT endpoint
       6. User-kitchen line RIGHT endpoint
       7. Opponent-kitchen line LEFT endpoint
       8. Opponent-kitchen line RIGHT endpoint
    5. As points are added, dots + connecting lines render on the canvas.
    6. Backspace = undo last point.
    7. "Save" button writes markers.json.

If the file at --out exists, it loads existing markers and lets you edit
them. To start fresh, delete the file first.

Coordinates are stored in original-video pixel space, regardless of how
much the frame was downscaled for display.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox


DEFAULT_MAX_DISPLAY_FRACTION = 0.8
N_REQUIRED_POINTS = 8

# Ordered list of (point_label, hint_text, color)
POINT_SPEC = [
    ("Court corner: bottom-LEFT (in image)",
     "First click: bottom-left corner of court as it appears in the image.",
     "#ff3030"),
    ("Court corner: bottom-RIGHT",
     "Second click: bottom-right corner.",
     "#ff3030"),
    ("Court corner: top-RIGHT",
     "Third click: top-right corner.",
     "#ff3030"),
    ("Court corner: top-LEFT",
     "Fourth click: top-left corner.",
     "#ff3030"),
    ("User-kitchen line: LEFT endpoint",
     "Fifth click: where user's kitchen line meets the LEFT sideline.",
     "#30c0ff"),
    ("User-kitchen line: RIGHT endpoint",
     "Sixth click: where user's kitchen line meets the RIGHT sideline.",
     "#30c0ff"),
    ("Opponent-kitchen line: LEFT endpoint",
     "Seventh click: where opponent's kitchen line meets the LEFT sideline.",
     "#ffaa30"),
    ("Opponent-kitchen line: RIGHT endpoint",
     "Eighth click: where opponent's kitchen line meets the RIGHT sideline.",
     "#ffaa30"),
]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_or_init_markers(out_path: Path) -> dict:
    """Load existing markers if present, else return empty template."""
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            fail(f"existing {out_path} is malformed JSON: {e}")
        # Validate the schema we expect
        for k in ("court_corners_image", "kitchen_line_user_image",
                  "kitchen_line_opponent_image"):
            if k not in data:
                fail(f"existing {out_path} is missing required field '{k}'")
        return data
    # Empty template
    return {
        "court_corners_image": [],
        "kitchen_line_user_image": [],
        "kitchen_line_opponent_image": [],
        "user_baseline": "near",
        "dominant_hand": "right",
        "user_starting_corner": "left",
        "frame_used_for_calibration": 0,
    }


def markers_to_flat_points(data: dict) -> list:
    """Flatten the 8 markers into a single ordered list of (x, y) ints."""
    pts = []
    pts.extend(data.get("court_corners_image", []))
    pts.extend(data.get("kitchen_line_user_image", []))
    pts.extend(data.get("kitchen_line_opponent_image", []))
    return [tuple(p) for p in pts]


def flat_points_to_markers(pts: list, base_data: dict) -> dict:
    """Inverse of markers_to_flat_points. Pads with empties as needed."""
    pts = list(pts)
    while len(pts) < 8:
        pts.append(None)
    return {
        "court_corners_image": [list(p) for p in pts[0:4] if p is not None],
        "kitchen_line_user_image": [list(p) for p in pts[4:6] if p is not None],
        "kitchen_line_opponent_image": [list(p) for p in pts[6:8] if p is not None],
        "user_baseline": base_data.get("user_baseline", "near"),
        "dominant_hand": base_data.get("dominant_hand", "right"),
        "user_starting_corner": base_data.get("user_starting_corner", "left"),
        "frame_used_for_calibration": base_data.get("frame_used_for_calibration", 0),
    }


def save_markers(out_path: Path, data: dict) -> None:
    """Atomic save: write to .tmp then rename."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        # Match the formatting style used in the existing test_clip markers.json
        # (compact arrays for point pairs, each top-level key on its own line)
        json_text = format_markers_compact(data)
        f.write(json_text)
        f.write("\n")
    tmp.replace(out_path)


def format_markers_compact(data: dict) -> str:
    """Produce JSON with compact inner arrays — matches the hand-edited
    style of the existing markers.json file in test_clip."""
    def fmt_pt_list(pts):
        if not pts:
            return "[]"
        return "[" + ", ".join(f"[{int(p[0])}, {int(p[1])}]" for p in pts) + "]"
    parts = []
    parts.append(f'  "court_corners_image": {fmt_pt_list(data["court_corners_image"])}')
    parts.append(f'  "kitchen_line_user_image": {fmt_pt_list(data["kitchen_line_user_image"])}')
    parts.append(f'  "kitchen_line_opponent_image": {fmt_pt_list(data["kitchen_line_opponent_image"])}')
    parts.append(f'  "user_baseline": "{data["user_baseline"]}"')
    parts.append(f'  "dominant_hand": "{data["dominant_hand"]}"')
    parts.append(f'  "user_starting_corner": "{data["user_starting_corner"]}"')
    parts.append(f'  "frame_used_for_calibration": {int(data["frame_used_for_calibration"])}')
    return "{\n" + ",\n".join(parts) + "\n}"


# ---------- Video access ----------

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
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()


# ---------- App ----------

class MarkerApp:
    def __init__(self, root: tk.Tk, video: VideoReader, out_path: Path,
                 data: dict, max_display_fraction: float):
        self.root = root
        self.video = video
        self.out_path = out_path
        self.data = data

        # Display scaling
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        # Reserve some space for toolbar + dropdowns + status bar
        avail_h = int(screen_h * max_display_fraction) - 200
        avail_w = int(screen_w * max_display_fraction)
        scale_w = avail_w / video.width
        scale_h = avail_h / video.height
        self.display_scale = min(1.0, scale_w, scale_h)
        self.display_w = int(video.width * self.display_scale)
        self.display_h = int(video.height * self.display_scale)
        print(f"Display scale: {self.display_scale:.3f} "
              f"({self.display_w}x{self.display_h}) "
              f"from source ({video.width}x{video.height})")

        # State: list of exactly N_REQUIRED_POINTS (x, y) slots in
        # original-video coordinates. Unfilled slots are None.
        loaded = markers_to_flat_points(data)
        self.points = list(loaded[:N_REQUIRED_POINTS])
        while len(self.points) < N_REQUIRED_POINTS:
            self.points.append(None)
        self.current_frame_idx = data.get("frame_used_for_calibration", 0)

        # ---- UI layout ----
        # Top row: dropdowns
        top = ttk.Frame(root, padding=6)
        top.pack(side="top", fill="x")

        ttk.Label(top, text="Dominant hand:").pack(side="left")
        self.var_hand = tk.StringVar(value=data.get("dominant_hand", "right"))
        ttk.Combobox(top, textvariable=self.var_hand,
                     values=["right", "left"], width=6, state="readonly"
                     ).pack(side="left", padx=(2, 12))

        ttk.Label(top, text="User baseline:").pack(side="left")
        self.var_baseline = tk.StringVar(value=data.get("user_baseline", "near"))
        ttk.Combobox(top, textvariable=self.var_baseline,
                     values=["near", "far"], width=6, state="readonly"
                     ).pack(side="left", padx=(2, 12))

        ttk.Label(top, text="Starting corner:").pack(side="left")
        self.var_corner = tk.StringVar(value=data.get("user_starting_corner", "left"))
        ttk.Combobox(top, textvariable=self.var_corner,
                     values=["left", "right"], width=6, state="readonly"
                     ).pack(side="left", padx=(2, 12))

        ttk.Button(top, text="Save", command=self.on_save).pack(side="right")
        ttk.Button(top, text="Undo last point", command=self.on_undo
                   ).pack(side="right", padx=(0, 6))
        ttk.Button(top, text="Clear all points", command=self.on_clear
                   ).pack(side="right", padx=(0, 6))

        # Middle: canvas
        self.canvas = tk.Canvas(root, width=self.display_w,
                                height=self.display_h, bg="black",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack(side="top")

        # Frame scrubber
        scrub = ttk.Frame(root, padding=4)
        scrub.pack(side="top", fill="x")
        ttk.Label(scrub, text="Frame:").pack(side="left")
        self.frame_var = tk.IntVar(value=self.current_frame_idx)
        self.frame_slider = ttk.Scale(scrub, from_=0, to=video.n_frames - 1,
                                      orient="horizontal",
                                      variable=self.frame_var,
                                      command=self.on_scrub)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.frame_label_var = tk.StringVar()
        ttk.Label(scrub, textvariable=self.frame_label_var, width=18
                  ).pack(side="right")

        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(root, textvariable=self.status_var, anchor="w",
                  padding=4).pack(side="top", fill="x")

        # Bindings
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Motion>", self.on_motion)
        root.bind("<BackSpace>", lambda e: self.on_undo())
        root.bind("<Escape>", lambda e: self.on_quit())
        root.protocol("WM_DELETE_WINDOW", self.on_quit)

        # Render state
        self.tk_image = None
        self.canvas_image_id = None
        self.point_artifacts = []  # canvas item ids for points/lines
        self.reticle_v = None
        self.reticle_h = None

        self.render_frame()

    # ---- Rendering ----

    def render_frame(self):
        frame = self.video.read_frame(self.current_frame_idx)
        if frame is None:
            self.set_status(f"ERROR reading frame {self.current_frame_idx}")
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
            self.canvas_image_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self.tk_image)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)

        self.redraw_points()
        self.update_status()

    def redraw_points(self):
        # Clear old artifacts
        for item in self.point_artifacts:
            self.canvas.delete(item)
        self.point_artifacts.clear()

        # Draw points as small colored circles + connecting lines
        # Connecting lines: corners 0->1->2->3->0 (rectangle); kitchen pairs 4-5, 6-7
        def to_disp(p):
            return (p[0] * self.display_scale, p[1] * self.display_scale)

        for i, p in enumerate(self.points):
            if p is None:
                continue
            x, y = to_disp(p)
            color = POINT_SPEC[i][2]
            r = 5
            oid = self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline=color, width=2, fill="")
            self.point_artifacts.append(oid)
            tid = self.canvas.create_text(
                x + r + 3, y - r - 3,
                text=str(i + 1), fill=color,
                anchor="sw", font=("Consolas", 9, "bold"))
            self.point_artifacts.append(tid)

        # Court rectangle (4 corners)
        if all(self.points[i] is not None for i in (0, 1, 2, 3)):
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                xa, ya = to_disp(self.points[a])
                xb, yb = to_disp(self.points[b])
                lid = self.canvas.create_line(xa, ya, xb, yb,
                                              fill="#ff3030", width=2)
                self.point_artifacts.append(lid)

        # User kitchen line (5-6)
        if all(self.points[i] is not None for i in (4, 5)):
            xa, ya = to_disp(self.points[4])
            xb, yb = to_disp(self.points[5])
            lid = self.canvas.create_line(xa, ya, xb, yb,
                                          fill="#30c0ff", width=2)
            self.point_artifacts.append(lid)

        # Opponent kitchen line (6-7)
        if all(self.points[i] is not None for i in (6, 7)):
            xa, ya = to_disp(self.points[6])
            xb, yb = to_disp(self.points[7])
            lid = self.canvas.create_line(xa, ya, xb, yb,
                                          fill="#ffaa30", width=2)
            self.point_artifacts.append(lid)

    def update_status(self):
        n = sum(1 for p in self.points if p is not None)
        self.frame_label_var.set(f"{self.current_frame_idx} / {self.video.n_frames - 1}")
        if n < N_REQUIRED_POINTS:
            spec = POINT_SPEC[n]
            self.status_var.set(f"Next: {spec[0]}  —  {spec[1]}  "
                                f"(point {n+1}/{N_REQUIRED_POINTS})")
        else:
            self.status_var.set(
                f"All 8 points marked. Click 'Save' to write markers.json. "
                f"Use 'Undo last point' or scrub to a different frame to redo.")

    def set_status(self, msg):
        self.status_var.set(msg)

    # ---- Event handlers ----

    def on_left_click(self, event):
        n = sum(1 for p in self.points if p is not None)
        if n >= N_REQUIRED_POINTS:
            messagebox.showinfo(
                "All points marked",
                "All 8 points already marked. Use 'Undo last point' or "
                "'Clear all points' to redo.")
            return
        # Convert display coords to source coords
        src_x = int(round(event.x / self.display_scale))
        src_y = int(round(event.y / self.display_scale))
        # Find first None slot
        for i, p in enumerate(self.points):
            if p is None:
                self.points[i] = (src_x, src_y)
                break
        else:
            self.points.append((src_x, src_y))
        # Persist frame index when first point of this session is placed
        self.data["frame_used_for_calibration"] = self.current_frame_idx
        self.render_frame()

    def on_motion(self, event):
        # Reticle
        for item in (self.reticle_v, self.reticle_h):
            if item is not None:
                self.canvas.delete(item)
        self.reticle_v = self.canvas.create_line(
            event.x, 0, event.x, self.display_h,
            fill="#ff00ff", width=1)
        self.reticle_h = self.canvas.create_line(
            0, event.y, self.display_w, event.y,
            fill="#ff00ff", width=1)

    def on_undo(self):
        # Remove last placed point (rightmost non-None)
        for i in range(len(self.points) - 1, -1, -1):
            if self.points[i] is not None:
                self.points[i] = None
                break
        self.points = [p for p in self.points if p is not None]
        while len(self.points) < N_REQUIRED_POINTS:
            self.points.append(None)
        self.render_frame()

    def on_clear(self):
        if messagebox.askyesno("Clear all points",
                               "Remove all 8 points and start over?"):
            self.points = [None] * N_REQUIRED_POINTS
            self.render_frame()

    def on_scrub(self, val):
        new_idx = int(float(val))
        if new_idx == self.current_frame_idx:
            return
        self.current_frame_idx = new_idx
        self.render_frame()

    def on_save(self):
        """Save markers.json AND run Stage 1 calibrate to produce court.json."""
        n = sum(1 for p in self.points if p is not None)
        if n < N_REQUIRED_POINTS:
            messagebox.showwarning(
                "Not enough points",
                f"You have {n} of {N_REQUIRED_POINTS} required points. "
                f"Mark all 8 before saving.")
            return
        self.data["dominant_hand"] = self.var_hand.get()
        self.data["user_baseline"] = self.var_baseline.get()
        self.data["user_starting_corner"] = self.var_corner.get()
        self.data["frame_used_for_calibration"] = self.current_frame_idx
        out_data = flat_points_to_markers(self.points, self.data)
        save_markers(self.out_path, out_data)
        self.set_status(f"Saved {self.out_path.name}; running calibrate...")
        self.root.update_idletasks()
        # Run Stage 1 calibrate as a subprocess
        result = run_calibrate(
            video_path=self.data.get("_video_path_for_display", ""),
            markers_path=self.out_path,
            out_dir=self.out_path.parent,
        )
        show_calibrate_result(self.out_path, result)
        if result["ok"]:
            self.set_status(
                f"Saved markers.json and produced court.json. "
                f"You can close this window.")
        else:
            self.set_status(
                f"Saved markers.json but calibrate FAILED. "
                f"Re-mark points and click Save again.")

    def video_path(self):
        # Best-effort: derive from the VideoReader's underlying source.
        # cv2.VideoCapture doesn't expose the path; rely on the data dict.
        return self.data.get("_video_path_for_display", "<video>")

    def on_quit(self):
        # Don't auto-save on quit (markers are all-or-nothing)
        n = sum(1 for p in self.points if p is not None)
        if n > 0 and n < N_REQUIRED_POINTS:
            if not messagebox.askyesno(
                    "Quit without saving?",
                    f"You have {n} unsaved points. Quit anyway?"):
                return
        elif n == N_REQUIRED_POINTS:
            if messagebox.askyesno("Save before quitting?",
                                   "All 8 points are placed. Save before "
                                   "quitting?"):
                self.on_save()
        self.root.destroy()


# ---------- Calibrate-runner helpers ----------

def run_calibrate(video_path: str, markers_path: Path, out_dir: Path) -> dict:
    """Run Stage 1 calibrate as a subprocess. Capture stdout, stderr, exit
    code. Returns a dict with the captured output."""
    if not video_path or video_path == "<video>":
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "internal error: video path was not recorded",
            "command": "",
        }
    cmd = [
        sys.executable, "-m", "stages.calibrate.calibrate",
        "--video", str(video_path),
        "--markers", str(markers_path),
        "--out-dir", str(out_dir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
            timeout=120,
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "command": " ".join(cmd),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": -2,
            "stdout": "",
            "stderr": "calibrate subprocess timed out after 120s",
            "command": " ".join(cmd),
        }
    except Exception as e:
        return {
            "ok": False,
            "exit_code": -3,
            "stdout": "",
            "stderr": f"failed to run calibrate: {type(e).__name__}: {e}",
            "command": " ".join(cmd),
        }


def show_calibrate_result(markers_path: Path, result: dict) -> None:
    """Show a message box summarizing the calibrate run."""
    out_dir = markers_path.parent
    court_json = out_dir / "court.json"
    court_zones = out_dir / "court_zones.json"
    court_exists = court_json.exists()
    zones_exists = court_zones.exists()

    summary_lines = []
    summary_lines.append(f"markers.json: {markers_path}")
    summary_lines.append(f"command: {result['command']}")
    summary_lines.append(f"exit code: {result['exit_code']}")
    summary_lines.append(f"court.json: {'present' if court_exists else 'MISSING'}")
    summary_lines.append(f"court_zones.json: {'present' if zones_exists else 'MISSING'}")
    summary_lines.append("")

    # Trim long stdout/stderr to 2KB each so the dialog is readable
    def trim(s):
        if not s:
            return "(empty)"
        if len(s) > 2000:
            return s[:2000] + "\n...[truncated, full output in your terminal]..."
        return s

    if result["stdout"]:
        summary_lines.append("--- stdout ---")
        summary_lines.append(trim(result["stdout"]))
    if result["stderr"]:
        summary_lines.append("--- stderr ---")
        summary_lines.append(trim(result["stderr"]))

    text = "\n".join(summary_lines)
    # Print to console too — terminal is the durable record
    print()
    print("=== calibrate result ===")
    print(text)
    print()

    if result["ok"] and court_exists:
        messagebox.showinfo(
            "Calibrate succeeded",
            f"court.json was produced.\n\n"
            f"Files:\n  {court_json}\n  {court_zones}\n\n"
            f"See the terminal for the full calibrate output. "
            f"Check for any 'warnings' field in court.json — projection "
            f"errors over 10px may need re-clicking.")
    else:
        messagebox.showerror(
            "Calibrate FAILED",
            f"Exit code: {result['exit_code']}\n"
            f"court.json present: {court_exists}\n\n"
            f"Re-mark the 8 points (try a clearer frame) and click Save again.\n\n"
            f"See the terminal for full stdout/stderr.")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Stage 1 markers-clicker for Pickleball-Analyzer-v2")
    ap.add_argument("--video", type=Path, required=True,
                    help="Video file to calibrate.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output markers.json path.")
    ap.add_argument("--max-display-fraction", type=float,
                    default=DEFAULT_MAX_DISPLAY_FRACTION,
                    dest="max_display_fraction")
    args = ap.parse_args()

    if not args.video.exists():
        fail(f"video not found: {args.video}")
    if not (0.1 <= args.max_display_fraction <= 1.0):
        fail("--max-display-fraction must be in [0.1, 1.0]")

    print(f"opening {args.video}")
    video = VideoReader(args.video)
    print(f"video: {video.n_frames} frames, {video.fps:.2f} fps, "
          f"{video.width}x{video.height}")

    data = load_or_init_markers(args.out)
    data["_video_path_for_display"] = str(args.video)

    root = tk.Tk()
    root.title(f"mark_court — {args.out.name}")
    app = MarkerApp(root, video, args.out, data, args.max_display_fraction)

    print()
    print("UI controls:")
    print("  Left-click       Place next point.")
    print("  Backspace        Undo last point.")
    print("  Frame slider     Scrub to a clearer frame.")
    print("  Esc              Quit (prompts to save if all 8 placed).")
    print()

    root.mainloop()
    video.release()


if __name__ == "__main__":
    main()
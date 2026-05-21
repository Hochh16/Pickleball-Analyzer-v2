"""
Stage 4.5 v3 Sub-piece 2: tune CV ball-detection parameters per video.

Reads a video, its court.json (Stage 1 output), and optionally a
ball_labels.json (from tools/label_ball.py). Derives a tuned
ball_cv_params.json by:

1. Computing a median background reference (via shared _ball_cv_pipeline).
2. Loading labeled-visible frames as the "where is the ball?" oracle
   (interactive click phase is available but in practice we have labels
    for our 4 videos; clicks remain as a fallback for new videos with no
    labels yet).
3. Measuring ball appearance on labeled frames:
   - Foreground-diff intensity at the ball location (sets bg threshold)
   - Connected-component area and circularity at the ball location
     (sets blob area/circularity bounds)
   - HSV color at the ball location (sets ball color median + tolerance)
   - Frame-to-frame displacement between consecutive labeled-visible
     ball positions (sets motion bounds)
4. Running the resulting parameters back through the CV pipeline on a
   held-back set of labeled frames, reporting "tune accuracy" (% of
   labeled-visible frames where detection lands within 10 px of GT).
5. Showing a 4x5 grid PNG of the tool's predictions overlaid on the
   sample frames for operator review.
6. Operator approves or aborts based on the printed metrics + PNG.
7. On approval, writes ball_cv_params.json.

CLI:
    python -m stages.finetune_ball_model.tune_ball_cv \
        --video data/test_clip/video.mp4 \
        --court data/test_clip/court.json \
        --labels data/test_clip/ball_labels.json \
        --out   data/test_clip/ball_cv_params.json

Without --labels, the tool drops into an interactive click loop on N
sampled frames (same UX pattern as label_ball.py, ~3 min per video).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tkinter as tk
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Local-style import: this module sits in the same package
from stages.finetune_ball_model._ball_cv_pipeline import (
    BackgroundModel,
    BallCVParams,
    PARAMS_SCHEMA_VERSION,
    detect_in_frame,
    expand_roi,
    load_court_roi,
)

STAGE_VERSION = "0.3.0"

# Tuning constants
DEFAULT_BG_N_FRAMES = 100
DEFAULT_N_INTERACTIVE_FRAMES = 20
DEFAULT_ROI_BUFFER_FT = 8.0
DEFAULT_ROI_BUFFER_PX = 80  # crude default for now; refined to ft once we
                            # have the homography in hand
N_VAL_FRAMES_MAX = 100  # cap on labeled frames used for tune accuracy
GT_MATCH_PX = 10.0     # detection within this counts as correct
GRID_COLS = 5
GRID_ROWS = 4


# ----------------------------------------------------------------- utilities

def fail(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def percentile(values, pct):
    return float(np.percentile(values, pct))


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"cannot open video: {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, n_frames, fps, w, h


def read_frame(cap, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


# --------------------------------------------------------- label-derived measurements

def _measure_ball_fg_intensity(frame_bgr: np.ndarray, background: np.ndarray,
                               x: float, y: float, patch: int = 5) -> float:
    """Median of the abs-diff intensity in a small patch around (x, y).
    Used to set bg_subtraction_threshold below this value with margin."""
    diff = cv2.absdiff(frame_bgr, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    xi, yi = int(round(x)), int(round(y))
    r = patch // 2
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(np.median(region))


def _measure_ball_hsv(frame_bgr: np.ndarray, x: float, y: float,
                      patch: int = 5) -> tuple:
    """Median HSV (h, s, v) in a small patch around (x, y). Hue is in [0,180]
    (OpenCV convention)."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    xi, yi = int(round(x)), int(round(y))
    r = patch // 2
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    region = hsv[y0:y1, x0:x1]
    if region.size == 0:
        return (0.0, 0.0, 0.0)
    return (
        float(np.median(region[:, :, 0])),
        float(np.median(region[:, :, 1])),
        float(np.median(region[:, :, 2])),
    )


# Hard caps reflecting pickleball physics at this camera distance.
# A pickleball at ~6ft camera height in 1080p is typically 4-6 px in
# diameter, so blob area is in the 10-30 px^2 range. Anything > 80 is
# definitely not a ball; anything > 6 px away from the click is a
# different object (player's hand, court line, etc.).
MAX_REASONABLE_BALL_AREA_PX = 80.0
MAX_BLOB_SNAP_DISTANCE_PX = 12.0  # widened: motion-blur streak centroid can be 5-10 px from labeled tip


def _is_isolated_blob(frame_bgr: np.ndarray, background: np.ndarray,
                      x: float, y: float,
                      permissive_threshold: int = 12,
                      surround_radius: int = 30,
                      max_surround_foreground_frac: float = 0.20) -> bool:
    """Is the labeled position (x, y) sitting on an isolated small foreground
    blob, vs being inside a large player blob? Returns True iff:
      - the foreground component containing/nearest (x, y) has area <= 80 px^2
      - within a (2*surround_radius+1) square centered on (x, y), no more
        than max_surround_foreground_frac of the *non-ball* pixels are
        foreground.

    The point: a held ball is inside a player's foreground blob; even if we
    can locate a small component nearby, the surrounding pixels are part of
    the player and the measurement is contaminated. A struck (or dinked or
    lobbed) ball is surrounded by background pixels regardless of its speed.
    """
    diff = cv2.absdiff(frame_bgr, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    _, fg = cv2.threshold(blur, permissive_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    h, w = fg.shape
    xi, yi = int(round(x)), int(round(y))
    # Extract surround square (clipped to image)
    x0 = max(0, xi - surround_radius)
    x1 = min(w, xi + surround_radius + 1)
    y0 = max(0, yi - surround_radius)
    y1 = min(h, yi + surround_radius + 1)
    surround = fg[y0:y1, x0:x1]
    if surround.size == 0:
        return False
    n_foreground = int((surround > 0).sum())
    n_total = surround.size
    # Subtract a generous estimate of the ball itself (up to 80 px^2)
    n_non_ball_fg = max(0, n_foreground - 80)
    n_non_ball_total = max(1, n_total - 80)
    frac = n_non_ball_fg / n_non_ball_total
    return frac <= max_surround_foreground_frac


def _measure_ball_blob(frame_bgr: np.ndarray, background: np.ndarray,
                       x: float, y: float,
                       permissive_threshold: int = 12) -> Optional[dict]:
    """Run a permissive foreground-blob detection and find the component
    that contains (or is closest to) (x, y). Returns area + circularity,
    or None if no component is within MAX_BLOB_SNAP_DISTANCE_PX or if the
    nearest component is larger than MAX_REASONABLE_BALL_AREA_PX."""
    diff = cv2.absdiff(frame_bgr, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    _, fg = cv2.threshold(blur, permissive_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg, connectivity=8
    )
    best = None
    best_dist = 1e18
    for label in range(1, n_labels):
        # Filter by area BEFORE distance: huge blobs are not the ball
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area > MAX_REASONABLE_BALL_AREA_PX:
            continue
        if area < 2.0:
            continue
        cx, cy = centroids[label]
        d = float(np.hypot(cx - x, cy - y))
        if d < best_dist:
            best_dist = d
            best = label
    if best is None or best_dist > MAX_BLOB_SNAP_DISTANCE_PX:
        return None
    area = float(stats[best, cv2.CC_STAT_AREA])
    component = (labels == best).astype(np.uint8)
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    perim = cv2.arcLength(contours[0], closed=True)
    if perim <= 0:
        return None
    circ = float(4 * np.pi * area / (perim * perim))
    return {"area": area, "circularity": circ}


# ------------------------------------------------------------- threshold derivation

def derive_thresholds(measurements: dict, video_fps: float) -> dict:
    """Given the per-frame measurements dict (filled by the collection loop),
    derive the 6 cv_params numeric fields. Returns a dict suitable for
    BallCVParams instantiation alongside metadata."""

    intensities = measurements["fg_intensities"]
    if not intensities:
        fail("no foreground-intensity samples collected; cannot tune")
    # We want the threshold below the typical ball signal.
    # 10th percentile of observed intensities is the dimmer end; go 60%.
    bg_thr = max(8, int(percentile(intensities, 10) * 0.6))
    bg_thr = min(bg_thr, 60)  # cap to avoid pathological cases

    areas = measurements["areas"]
    circs = measurements["circularities"]
    if not areas or not circs:
        fail("no valid blob measurements collected; cannot tune")
    area_min = max(2.0, percentile(areas, 5) * 0.7)
    area_max = max(area_min + 5, percentile(areas, 95) * 1.5)
    # Hard ceiling: a pickleball at this camera distance is never > 80 px^2.
    # Without this cap, a single contaminated measurement (e.g., a player's
    # hand near the labeled position) can blow up area_max and make the
    # filter useless.
    area_max = min(area_max, MAX_REASONABLE_BALL_AREA_PX)
    circ_min = max(0.10, percentile(circs, 5) - 0.05)

    hsvs = measurements["hsvs"]  # list of (h, s, v)
    if not hsvs:
        fail("no HSV samples collected")
    hs = [v[0] for v in hsvs]
    ss = [v[1] for v in hsvs]
    vs = [v[2] for v in hsvs]
    h_med = float(np.median(hs))
    s_med = float(np.median(ss))
    v_med = float(np.median(vs))
    # IQR-based tolerance
    h_tol = max(5.0, 1.5 * (percentile(hs, 75) - percentile(hs, 25)))
    s_tol = max(20.0, 1.5 * (percentile(ss, 75) - percentile(ss, 25)))
    v_tol = max(20.0, 1.5 * (percentile(vs, 75) - percentile(vs, 25)))

    disps = measurements["per_frame_displacements"]
    if not disps:
        # Fallback: no consecutive ball motion observed
        m_min = 1.0
        m_max = 80.0
    else:
        m_min = max(0.5, percentile(disps, 5) * 0.5)
        m_max = max(m_min + 5, percentile(disps, 95) * 1.5)

    return {
        "bg_subtraction_threshold": int(bg_thr),
        "blob_area_px_min": float(area_min),
        "blob_area_px_max": float(area_max),
        "blob_circularity_min": float(circ_min),
        "ball_color_hsv_median": [h_med, s_med, v_med],
        "ball_color_hsv_tolerance": [h_tol, s_tol, v_tol],
        "motion_displacement_px_per_frame_min": float(m_min),
        "motion_displacement_px_per_frame_max": float(m_max),
    }


# -------------------------------------------------------------- validation pass

def validate_with_params(
    cap, params: BallCVParams,
    background: BackgroundModel,
    roi_polygon: np.ndarray,
    eval_samples: list,  # list of (frame_idx, gt_x, gt_y) for VISIBLE frames
) -> dict:
    """Re-run the CV pipeline on `eval_samples` with the derived params.
    Returns dict with hits, total, accuracy, per-sample (frame_idx, gt, pred, hit).
    """
    results = []
    hits = 0
    prev_xy = None
    eval_samples = sorted(eval_samples, key=lambda r: r[0])
    last_frame_idx = -1
    for frame_idx, gt_x, gt_y in eval_samples:
        # Reset prev_xy if there's a gap (we're not doing dense per-frame
        # evaluation; the motion-history assumption between sparse samples
        # would be wrong).
        if frame_idx - last_frame_idx > 5:
            prev_xy = None
        last_frame_idx = frame_idx
        frame = read_frame(cap, frame_idx)
        if frame is None:
            results.append({
                "frame_idx": frame_idx, "gt": (gt_x, gt_y),
                "pred": None, "hit": False, "reason": "read_failed",
            })
            continue
        det = detect_in_frame(frame, background, roi_polygon, params, prev_xy)
        if det is None:
            results.append({
                "frame_idx": frame_idx, "gt": (gt_x, gt_y),
                "pred": None, "hit": False, "reason": "no_detection",
            })
            prev_xy = None
            continue
        err = float(np.hypot(det.pixel_x - gt_x, det.pixel_y - gt_y))
        hit = err <= GT_MATCH_PX
        if hit:
            hits += 1
        results.append({
            "frame_idx": frame_idx,
            "gt": (gt_x, gt_y),
            "pred": (det.pixel_x, det.pixel_y, det.confidence),
            "error_px": err,
            "hit": hit,
        })
        prev_xy = (det.pixel_x, det.pixel_y)
    total = len(eval_samples)
    accuracy = hits / total if total else 0.0
    return {"hits": hits, "total": total, "accuracy": accuracy,
            "per_sample": results}


# -------------------------------------------------------- approval grid render

def render_approval_grid(cap, results: list, video_w: int, video_h: int,
                         out_png: Path,
                         tile_w: int = 480, tile_h: int = 270):
    """Build a GRID_ROWS x GRID_COLS image with annotated tiles."""
    n_tiles = GRID_ROWS * GRID_COLS
    selected = results[:n_tiles]
    while len(selected) < n_tiles:
        selected.append(None)  # pad if fewer results

    grid = np.zeros((tile_h * GRID_ROWS, tile_w * GRID_COLS, 3),
                    dtype=np.uint8)
    for i, r in enumerate(selected):
        row = i // GRID_COLS
        col = i % GRID_COLS
        y0 = row * tile_h
        x0 = col * tile_w
        if r is None:
            continue
        frame = read_frame(cap, r["frame_idx"])
        if frame is None:
            continue
        canvas = frame.copy()
        gt = r["gt"]
        cv2.circle(canvas, (int(gt[0]), int(gt[1])), 18,
                   (0, 255, 0), 2)  # GT green
        if r["pred"] is not None:
            px, py, _conf = r["pred"]
            color = (0, 255, 0) if r["hit"] else (0, 0, 255)
            cv2.drawMarker(canvas, (int(px), int(py)), color,
                           markerType=cv2.MARKER_CROSS,
                           markerSize=24, thickness=2)
        tag = f"#{r['frame_idx']} "
        if r["pred"] is None:
            tag += "NO_DET"
        else:
            tag += f"err={r.get('error_px', -1):.1f}px"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(canvas, tag, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tile = cv2.resize(canvas, (tile_w, tile_h),
                          interpolation=cv2.INTER_AREA)
        grid[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    cv2.imwrite(str(out_png), grid)


# --------------------------------------------------------- interactive click UI

def interactive_click_session(cap, n_frames_video: int, n_samples: int) -> list:
    """Open a Tkinter window cycling through n_samples evenly-spaced frames.
    Left-click marks ball location; right-click or space marks not-visible.
    Returns list of (frame_idx, ball_visible, pixel_x, pixel_y) tuples."""
    sample_indices = np.linspace(
        0, n_frames_video - 1, num=n_samples, dtype=int
    ).tolist()

    labels = []
    state = {"i": 0, "done": False, "last_frame": None}

    root = tk.Tk()
    root.title("tune_ball_cv: click ball location, right-click or space if not visible")

    # Make canvas roughly fit a 1080p frame scaled to fit screen
    canvas = tk.Canvas(root, width=1280, height=720, bg="black")
    canvas.pack()
    info = tk.Label(root, text="", anchor="w", justify="left")
    info.pack(fill="x")

    photo_ref = {"img": None}

    def load_frame():
        if state["i"] >= len(sample_indices):
            state["done"] = True
            root.quit()
            return
        idx = sample_indices[state["i"]]
        frame = read_frame(cap, idx)
        if frame is None:
            state["i"] += 1
            load_frame()
            return
        state["last_frame"] = (idx, frame)
        # Resize to canvas
        from PIL import Image, ImageTk
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        scale = min(1280 / pil.width, 720 / pil.height)
        new_w = int(pil.width * scale)
        new_h = int(pil.height * scale)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        photo_ref["img"] = photo
        canvas.delete("all")
        canvas.create_image(640, 360, image=photo, anchor="center")
        state["scale"] = scale
        state["offset_x"] = 640 - new_w // 2
        state["offset_y"] = 360 - new_h // 2
        info.config(text=f"frame {idx}  ({state['i']+1} of {len(sample_indices)})")

    def on_left_click(event):
        if state["last_frame"] is None:
            return
        idx, _ = state["last_frame"]
        # Map canvas coords back to original frame coords
        x_canvas = event.x - state["offset_x"]
        y_canvas = event.y - state["offset_y"]
        x_orig = x_canvas / state["scale"]
        y_orig = y_canvas / state["scale"]
        labels.append((idx, True, float(x_orig), float(y_orig)))
        state["i"] += 1
        load_frame()

    def on_right_or_space(event=None):
        if state["last_frame"] is None:
            return
        idx, _ = state["last_frame"]
        labels.append((idx, False, None, None))
        state["i"] += 1
        load_frame()

    def on_esc(event=None):
        state["done"] = True
        root.quit()

    canvas.bind("<Button-1>", on_left_click)
    canvas.bind("<Button-3>", on_right_or_space)
    root.bind("<space>", on_right_or_space)
    root.bind("<Escape>", on_esc)

    load_frame()
    root.mainloop()
    root.destroy()
    return labels


# ----------------------------------------------------------------- main flow

def run(args) -> int:
    video_path = Path(args.video)
    court_path = Path(args.court)
    labels_path = Path(args.labels) if args.labels else None
    out_path = Path(args.out)

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        fail(f"video not found: {video_path}")
    if not court_path.exists():
        fail(f"court.json not found: {court_path}")

    # 1) Background
    print(f"computing background ({DEFAULT_BG_N_FRAMES} frames)...",
          flush=True)
    background = BackgroundModel.from_video(
        video_path, n_frames=DEFAULT_BG_N_FRAMES)
    print(f"  background shape: {background.background.shape}")

    # 2) Court ROI (expanded)
    base_roi = load_court_roi(court_path)
    roi_polygon = expand_roi(base_roi, DEFAULT_ROI_BUFFER_PX)

    # 3) Collect (frame_idx, ball_visible, x, y) tuples
    cap, n_frames_video, fps_video, vw, vh = open_video(video_path)

    if labels_path is not None and labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            labels_data = json.load(f)
        all_labels = labels_data["labels"]
        ball_labels = [
            (lab["frame_idx"], lab["ball_visible"],
             lab.get("pixel_x"), lab.get("pixel_y"))
            for lab in all_labels
        ]
        print(f"using {len(ball_labels)} labels from {labels_path}")
        calibration_method = "labels"
    else:
        print(f"no labels provided; running interactive click session "
              f"on {DEFAULT_N_INTERACTIVE_FRAMES} frames")
        ball_labels = interactive_click_session(
            cap, n_frames_video, DEFAULT_N_INTERACTIVE_FRAMES)
        calibration_method = "click"

    visible_labels = [l for l in ball_labels if l[1] and l[2] is not None]
    if len(visible_labels) < 10:
        cap.release()
        fail(f"only {len(visible_labels)} visible-ball samples; need >= 10 "
             f"to tune")

    # Filter to "isolated-blob" labels: the ball is a small foreground
    # component sitting in a mostly-background neighborhood. This rejects
    # held-ball labels (where the ball is embedded in a player blob)
    # without requiring the ball to be moving fast. Works for dinks, lobs,
    # drives, anything where the ball is visually separated from a body.
    #
    # We pre-screen a generous random sample of visible labels (cap at 500
    # for speed; this is a O(N) loop with per-frame video reads).
    print(f"  pre-screening visible labels for isolated-blob eligibility "
          f"(this involves ~500 video reads, ~30s)...")
    bg_img = background.background
    # Stride through visible labels to cap reads at ~500
    stride = max(1, len(visible_labels) // 500)
    screen_set = visible_labels[::stride]
    isolated = []
    for (fi, vis, x, y) in screen_set:
        frame = read_frame(cap, fi)
        if frame is None:
            continue
        if _is_isolated_blob(frame, bg_img, x, y):
            isolated.append((fi, vis, x, y))
    print(f"  isolated-blob labels: {len(isolated)} of "
          f"{len(screen_set)} screened "
          f"(from {len(visible_labels)} total visible)")
    if len(isolated) < 10:
        cap.release()
        fail(f"only {len(isolated)} isolated-blob visible-ball samples; "
             f"need >= 10 to tune. Most labels are likely partially "
             f"occluded by players. This video may need additional labels "
             f"of fully-visible in-flight balls.")

    # 4) Take measurements on the isolated-blob set
    sample_for_measurement = isolated[
        :: max(1, len(isolated) // 200)
    ]
    print(f"measuring on {len(sample_for_measurement)} visible labels...")
    fg_intensities = []
    areas = []
    circs = []
    hsvs = []
    bg_img = background.background
    for (frame_idx, _vis, x, y) in sample_for_measurement:
        frame = read_frame(cap, frame_idx)
        if frame is None:
            continue
        fg_intensities.append(
            _measure_ball_fg_intensity(frame, bg_img, x, y))
        blob = _measure_ball_blob(frame, bg_img, x, y)
        if blob is not None:
            areas.append(blob["area"])
            circs.append(blob["circularity"])
        hsvs.append(_measure_ball_hsv(frame, x, y))

    # Per-frame displacements: only between truly consecutive visible labels.
    # ball_labels are sorted by frame_idx in the input.
    disps = []
    prev = None
    for (frame_idx, vis, x, y) in ball_labels:
        if not vis or x is None:
            prev = None
            continue
        if prev is not None:
            pf, px, py = prev
            df = frame_idx - pf
            if 1 <= df <= 6:
                d = float(np.hypot(x - px, y - py)) / df
                disps.append(d)
        prev = (frame_idx, x, y)

    measurements = {
        "fg_intensities": fg_intensities,
        "areas": areas,
        "circularities": circs,
        "hsvs": hsvs,
        "per_frame_displacements": disps,
    }
    print(f"  fg intensities samples: {len(fg_intensities)}")
    print(f"  area samples:           {len(areas)}")
    print(f"  hsv samples:            {len(hsvs)}")
    print(f"  displacement samples:   {len(disps)}")

    derived = derive_thresholds(measurements, fps_video)
    print()
    print("derived parameters:")
    for k, v in derived.items():
        print(f"  {k} = {v}")

    # 5) Validate with derived parameters on held-back labeled samples
    # Validate on the same isolated-blob pool we measured from, so we test
    # the CV pipeline on frames where the ball IS detectable in principle.
    # Held-ball / occluded-ball frames are out of scope for CV detection by
    # design (the ball is inside a player blob and can't be separated).
    eval_samples = [(fi, x, y) for (fi, vis, x, y) in isolated
                    if x is not None][:N_VAL_FRAMES_MAX]
    params = BallCVParams(
        schema_version=PARAMS_SCHEMA_VERSION,
        video_path=str(video_path),
        video_width=vw,
        video_height=vh,
        video_fps=fps_video,
        background_method="median",
        background_n_frames=DEFAULT_BG_N_FRAMES,
        bg_subtraction_threshold=derived["bg_subtraction_threshold"],
        blob_area_px_min=derived["blob_area_px_min"],
        blob_area_px_max=derived["blob_area_px_max"],
        blob_circularity_min=derived["blob_circularity_min"],
        ball_color_hsv_median=derived["ball_color_hsv_median"],
        ball_color_hsv_tolerance=derived["ball_color_hsv_tolerance"],
        motion_displacement_px_per_frame_min=derived[
            "motion_displacement_px_per_frame_min"],
        motion_displacement_px_per_frame_max=derived[
            "motion_displacement_px_per_frame_max"],
        calibration_method=calibration_method,
        n_calibration_frames_used=len(sample_for_measurement),
        calibration_completed_at_utc=dt.datetime.now(dt.UTC).isoformat().replace(
            "+00:00", "Z"),
        stage_version=STAGE_VERSION,
    )

    print()
    print(f"validating on {len(eval_samples)} labeled-visible frames...")
    val_report = validate_with_params(
        cap, params, background, roi_polygon, eval_samples)
    print(f"  tune accuracy: {val_report['accuracy']:.3f} "
          f"({val_report['hits']}/{val_report['total']})")

    # 6) Render approval grid PNG
    grid_path = out_path.with_suffix(".tune_preview.png")
    render_approval_grid(cap, val_report["per_sample"], vw, vh, grid_path)
    print(f"  wrote approval grid: {grid_path}")
    cap.release()

    # 7) Operator approval
    print()
    print("review the approval grid PNG, then approve or abort.")
    print(f"  open: {grid_path}")
    response = input("save these parameters to ball_cv_params.json? [y/N] ")
    if response.strip().lower() not in ("y", "yes"):
        print("aborted; ball_cv_params.json NOT written")
        return 1

    params.save(out_path)
    print(f"wrote {out_path}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 4.5 v3: tune CV params")
    p.add_argument("--video", required=True)
    p.add_argument("--court", required=True)
    p.add_argument("--labels", required=False, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
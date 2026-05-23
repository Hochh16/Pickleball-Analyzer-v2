"""Stage 5 — detect shots.

Find every moment a player strikes the ball and emit shots.json. A shot is an
*impulsive* change in the ball's pixel-space trajectory (a single-frame turn-
rate spike and/or a sudden speed jump — the paddle-strike signature) that
coincides spatially with a tracked player. Free-flight gravity arcs (e.g. a
lob's apex over a player's head) bend the path gradually and are NOT shots;
ground bounces are impulsive but happen away from players and are rejected by
the player-proximity filter.

See stages/detect_shots/contract.md for the full spec.

Usage:
    python -m stages.detect_shots.detect_shots data/test_clip
    python -m stages.detect_shots.detect_shots data/test_clip --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
STAGE_VERSION = "0.1.0"

# --- Detection defaults (see contract "Configuration") ----------------------
MIN_TURN_RATE_DEG = 45.0
MIN_SPEED_CHANGE_RATIO = 0.35
MIN_DIRECTION_CHANGE_DEG = 45.0
IMPACT_WINDOW_FRAMES = 6
VELOCITY_WINDOW_FRAMES = 3
ASSOC_BBOX_HEIGHT_FRAC = 0.5
ASSOC_MAX_PX = 120.0
ASSOC_MAX_PX_MIN = 30.0
MIN_BALL_SPEED_PX_PER_FRAME = 1.5
MAX_BALL_SPEED_PX_PER_FRAME = 400.0
BALL_COVERAGE_WARN_FRAC = 0.30
FPS_TOLERANCE = 0.5
WRIST_VISIBILITY_FLOOR = 0.5
MIN_SERVE_GAP_S = 0.7  # not-visible gap before a serve (dead time vs detection gap)

EPS = 1e-9


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("detect_shots")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


# --- Loaders -----------------------------------------------------------------

def load_court(path: Path) -> dict:
    if not path.exists():
        fail(f"court.json not found: {path}", FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        c = json.load(f)
    homog = c.get("homography", {})
    if "image_to_court" not in homog:
        fail("court.json.homography missing image_to_court", ValueError)
    M = np.array(homog["image_to_court"], dtype=np.float64)
    if M.shape != (3, 3):
        fail(f"image_to_court must be 3x3, got {M.shape}", ValueError)
    video = c.get("video", {}) or {}
    return {"image_to_court": M, "fps": video.get("fps"),
            "frame_width": video.get("frame_width"),
            "frame_height": video.get("frame_height")}


def load_ball_meta(path: Path) -> dict:
    if not path.exists():
        fail(f"ball.meta.json not found: {path}. Stage 5 requires the ball "
             f"metadata sidecar (carries fps and the synthetic flag).",
             FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ball(path: Path, log: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        fail(f"ball.parquet not found: {path}", FileNotFoundError)
    df = pd.read_parquet(path)
    need = {"frame_idx", "pixel_x", "pixel_y", "visible", "interpolated"}
    missing = need - set(df.columns)
    if missing:
        fail(f"ball.parquet missing columns: {sorted(missing)}", ValueError)
    df = df.sort_values("frame_idx").reset_index(drop=True)
    # Re-assert Stage 4 schema invariants (defense against bad/placeholder data)
    vis = df["visible"].to_numpy()
    interp = df["interpolated"].to_numpy()
    if np.any(vis & interp):
        fail("ball.parquet invariant violated: visible AND interpolated on the "
             "same row", ValueError)
    known = vis | interp
    xy_nan = df["pixel_x"].isna().to_numpy() | df["pixel_y"].isna().to_numpy()
    if np.any(known & xy_nan):
        fail("ball.parquet invariant violated: known row (visible/interpolated) "
             "with NaN pixel_x/pixel_y", ValueError)
    if np.any(~known & ~xy_nan):
        fail("ball.parquet invariant violated: not-known row with non-NaN "
             "pixel coords", ValueError)
    return df


def index_players(path: Path) -> Tuple[Dict[int, List[dict]], int]:
    if not path.exists():
        fail(f"players.parquet not found: {path}", FileNotFoundError)
    df = pd.read_parquet(path)
    need = {"frame", "track_id", "is_user", "transient",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "foot_x", "foot_y"}
    missing = need - set(df.columns)
    if missing:
        fail(f"players.parquet missing columns: {sorted(missing)}", ValueError)
    df = df[~df["transient"]]
    by_frame: Dict[int, List[dict]] = {}
    for r in df.itertuples(index=False):
        by_frame.setdefault(int(r.frame), []).append({
            "track_id": int(r.track_id), "is_user": bool(r.is_user),
            "bbox": (float(r.bbox_x1), float(r.bbox_y1),
                     float(r.bbox_x2), float(r.bbox_y2)),
            "foot": (float(r.foot_x), float(r.foot_y)),
        })
    return by_frame, len(df)


def index_poses(path: Path) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """(frame, track_id) -> list of visible wrist (x, y) pixel points."""
    if not path.exists():
        return {}
    cols = ["frame", "track_id", "pose_detected",
            "left_wrist_x_px", "left_wrist_y_px", "left_wrist_visibility",
            "right_wrist_x_px", "right_wrist_y_px", "right_wrist_visibility"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["pose_detected"]]
    out: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for r in df.itertuples(index=False):
        wrists = []
        if r.left_wrist_visibility >= WRIST_VISIBILITY_FLOOR and not math.isnan(r.left_wrist_x_px):
            wrists.append((float(r.left_wrist_x_px), float(r.left_wrist_y_px)))
        if r.right_wrist_visibility >= WRIST_VISIBILITY_FLOOR and not math.isnan(r.right_wrist_x_px):
            wrists.append((float(r.right_wrist_x_px), float(r.right_wrist_y_px)))
        if wrists:
            out[(int(r.frame), int(r.track_id))] = wrists
    return out


# --- Geometry helpers --------------------------------------------------------

def angle_between(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def bbox_distance(bbox, px, py) -> float:
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(math.hypot(dx, dy))


def project_to_court(M: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    v = M @ np.array([px, py, 1.0])
    if abs(v[2]) < EPS or not np.all(np.isfinite(v)):
        return float("nan"), float("nan")
    return float(v[0] / v[2]), float(v[1] / v[2])


# --- Core detection ----------------------------------------------------------

def detect(df_ball: pd.DataFrame, players_by_frame, poses, court_M,
           log: logging.Logger, params: dict) -> Tuple[List[dict], dict, List[str]]:
    n = len(df_ball)
    fx = df_ball["pixel_x"].to_numpy()
    fy = df_ball["pixel_y"].to_numpy()
    vis = df_ball["visible"].to_numpy()
    interp = df_ball["interpolated"].to_numpy()
    known = vis | interp
    warnings: List[str] = []

    known_idx = np.where(known)[0]
    if len(known_idx) == 0:
        return [], {"analyzed_frame_range": [0, 0]}, ["ball has no usable positions; zero shots."]
    f_lo, f_hi = int(known_idx[0]), int(known_idx[-1])

    # --- Defense: teleport / impossible-motion check on contiguous known pairs
    max_speed = params["max_ball_speed_px_per_frame"]
    for i in range(f_lo + 1, f_hi + 1):
        if known[i] and known[i - 1]:
            d = math.hypot(fx[i] - fx[i - 1], fy[i] - fy[i - 1])
            if d > max_speed:
                fail(f"ball.parquet has impossible motion: {d:.0f} px between "
                     f"frames {i-1} and {i} (> {max_speed:.0f} px/frame cap). "
                     f"Ball data is corrupt.", ValueError)

    # --- Per-frame single-frame velocity, turn rate, speed-change ratio
    def vel(i):  # single-frame velocity into frame i (requires i-1, i known & contiguous)
        if i - 1 < 0 or not (known[i] and known[i - 1]):
            return None
        return np.array([fx[i] - fx[i - 1], fy[i] - fy[i - 1]])

    turn = np.full(n, np.nan)
    sratio = np.full(n, np.nan)
    for i in range(1, n - 1):
        v_in = vel(i)
        v_out = vel(i + 1)
        if v_in is None or v_out is None:
            continue
        ang = angle_between(v_in, v_out)
        if ang is None:
            continue
        turn[i] = ang
        s_in, s_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
        sratio[i] = abs(s_out - s_in) / max(s_in, s_out, EPS)

    # --- Windowed velocity (reported; nearest known neighbor within window)
    k = params["velocity_window_frames"]

    def nearest_known(i, lo, hi):
        for j in range(1, k + 1):
            t = i + j if hi else i - j
            if lo <= t < n and known[t]:
                return t
        return None

    # --- Candidate impacts: impulse signature + speed floor
    min_turn = params["min_turn_rate_deg"]
    min_sr = params["min_speed_change_ratio"]
    min_speed = params["min_ball_speed_px_per_frame"]
    min_dir = params["min_direction_change_deg"]

    n_candidates = 0
    n_low_speed = 0
    n_gap_rejected = 0
    cand: List[Tuple[int, float]] = []  # (frame, score)
    for i in range(1, n - 1):
        if math.isnan(turn[i]):
            # Could a likely impact be hiding in a gap here? windowed dir change
            fb = nearest_known(i, 0, False)
            ff = nearest_known(i, 0, True)
            if known[i] and fb is not None and ff is not None:
                v_in = np.array([fx[i] - fx[fb], fy[i] - fy[fb]]) / (i - fb)
                v_out = np.array([fx[ff] - fx[i], fy[ff] - fy[i]]) / (ff - i)
                dchg = angle_between(v_in, v_out)
                if dchg is not None and dchg >= min_dir:
                    n_gap_rejected += 1
            continue
        impulse = (turn[i] >= min_turn) or (sratio[i] >= min_sr)
        if not impulse:
            continue
        v_in = vel(i)
        v_out = vel(i + 1)
        s_max = max(np.linalg.norm(v_in), np.linalg.norm(v_out))
        if s_max < min_speed:
            n_low_speed += 1
            continue
        n_candidates += 1
        score = max(turn[i] / 180.0, min(1.0, sratio[i]))
        cand.append((i, score))

    # --- Non-maximum suppression within IMPACT_WINDOW_FRAMES
    W = params["impact_window_frames"]
    cand.sort(key=lambda c: c[1], reverse=True)
    accepted: List[int] = []
    suppressed = 0
    taken = np.zeros(n, dtype=bool)
    for f, _ in cand:
        if any(abs(f - a) <= W for a in accepted):
            suppressed += 1
            continue
        accepted.append(f)
        taken[f] = True
    accepted.sort()

    # --- Shared helpers (used by impulse shots and serves) ------------------
    bbox_frac = params["assoc_bbox_height_frac"]
    amax = params["assoc_max_px"]
    amin = params["assoc_max_px_min"]
    shots: List[dict] = []
    n_no_player = 0

    def associate(f, bx, by):
        """Closest in-range player to (bx, by) on frame f -> (player, dist,
        basis, radius), or None. Wrist first, then bbox, then foot."""
        best = None
        for p in players_by_frame.get(f, []):
            _, y1, _, y2 = p["bbox"]
            radius = min(max(bbox_frac * max(1.0, y2 - y1), amin), amax)
            ws = poses.get((f, p["track_id"]))
            if ws:
                d = min(math.hypot(bx - wx, by - wy) for wx, wy in ws)
                basis = "wrist"
            else:
                d = bbox_distance(p["bbox"], bx, by)
                basis = "bbox"
                if d > radius:
                    fdx, fdy = p["foot"]
                    fd = math.hypot(bx - fdx, by - fdy)
                    if fd < d:
                        d, basis = fd, "foot"
            if d <= radius:
                key = (d, 0 if p["is_user"] else 1, p["track_id"])
                if best is None or key < best[0]:
                    best = (key, basis, p, radius)
        return None if best is None else (best[2], best[0][0], best[1], best[3])

    def windowed(f):
        fb = nearest_known(f, 0, False)
        ff = nearest_known(f, 0, True)
        pre = [float("nan"), float("nan")]
        post = [float("nan"), float("nan")]
        dchg = float("nan")
        if fb is not None:
            pre = [(fx[f] - fx[fb]) / (f - fb), (fy[f] - fy[fb]) / (f - fb)]
        if ff is not None:
            post = [(fx[ff] - fx[f]) / (ff - f), (fy[ff] - fy[f]) / (ff - f)]
        if fb is not None and ff is not None:
            a = angle_between(np.array(pre), np.array(post))
            dchg = a if a is not None else float("nan")
        return pre, post, dchg

    def vfield(v):
        return [round(v[0], 3), round(v[1], 3)] if not math.isnan(v[0]) else [None, None]

    def court_xy(f, bx, by):
        cx, cy = project_to_court(court_M, float(bx), float(by))
        if math.isnan(cx):
            warnings.append(f"shot at frame {f}: court projection non-finite "
                            f"(degenerate homography); impact_court_xy_ft=NaN")
        return [round(cx, 2), round(cy, 2)] if not math.isnan(cx) else [None, None]

    def quality(f):
        w0, w1 = max(0, f - W), min(n, f + W + 1)
        return float(vis[w0:w1].mean()) if w1 > w0 else 0.0

    # --- Impulse shots (rally hits) ----------------------------------------
    for f in accepted:
        bx, by = float(fx[f]), float(fy[f])
        a = associate(f, bx, by)
        if a is None:
            n_no_player += 1
            continue
        p, dist, basis, radius = a
        pre, post, dchg = windowed(f)
        s_pre = math.hypot(*pre) if not math.isnan(pre[0]) else float("nan")
        s_post = math.hypot(*post) if not math.isnan(post[0]) else float("nan")
        impulse_term = max(min(1.0, turn[f] / 120.0), min(1.0, sratio[f]))
        prox_term = 1.0 - min(1.0, dist / radius)
        conf = float(np.clip(0.5 * impulse_term + 0.3 * prox_term
                             + 0.2 * quality(f), 0.0, 1.0))
        shots.append({
            "shot_id": 0, "frame": int(f), "t_sec": round(f / params["fps"], 3),
            "track_id": int(p["track_id"]), "is_user": bool(p["is_user"]),
            "is_serve": False, "detection_method": "impulse",
            "impact_pixel_xy": [round(bx, 2), round(by, 2)],
            "impact_court_xy_ft": court_xy(f, bx, by),
            "player_distance_px": round(float(dist), 2), "assoc_basis": basis,
            "pre_velocity_px_per_frame": vfield(pre),
            "post_velocity_px_per_frame": vfield(post),
            "speed_pre_px_per_frame": round(s_pre, 3) if not math.isnan(s_pre) else None,
            "speed_post_px_per_frame": round(s_post, 3) if not math.isnan(s_post) else None,
            "direction_change_deg": round(dchg, 1) if not math.isnan(dchg) else None,
            "turn_rate_deg": round(float(turn[f]), 1),
            "speed_change_ratio": round(float(sratio[f]), 3),
            "confidence": round(conf, 3),
        })
    impulse_frames = sorted(s["frame"] for s in shots)

    # --- Serves (ball appears near a player after dead time) ----------------
    # A serve has no incoming ball trajectory, so the impulse detector is blind
    # to it. Detect the START of a ball-visible run that follows a not-visible
    # gap longer than serve_gap_frames (dead time between rallies, distinct from
    # a short mid-rally detection gap), with an outgoing launch trajectory, near
    # a player. Flagged is_serve=True for downstream (Stage 6 classify, Stage 7
    # rally segmentation).
    serve_gap = params["serve_gap_frames"]
    n_serves = 0
    gap_run = 0
    for f in range(f_lo, f_hi + 1):
        if not known[f]:
            gap_run += 1
            continue
        run_start = (f == f_lo) or (not known[f - 1])
        preceding = (serve_gap + 1) if f == f_lo else gap_run
        gap_run = 0
        if not run_start or preceding < serve_gap:
            continue
        ff = nearest_known(f, 0, True)
        if ff is None:
            continue
        launch = [(fx[ff] - fx[f]) / (ff - f), (fy[ff] - fy[f]) / (ff - f)]
        if math.hypot(*launch) < min_speed:
            continue
        if any(abs(f - sf) <= W for sf in impulse_frames):
            continue  # already captured as an impulse shot
        bx, by = float(fx[f]), float(fy[f])
        a = associate(f, bx, by)
        if a is None:
            n_no_player += 1
            continue
        p, dist, basis, radius = a
        prox_term = 1.0 - min(1.0, dist / radius)
        conf = float(np.clip(0.4 + 0.4 * prox_term + 0.2 * quality(f), 0.0, 1.0))
        shots.append({
            "shot_id": 0, "frame": int(f), "t_sec": round(f / params["fps"], 3),
            "track_id": int(p["track_id"]), "is_user": bool(p["is_user"]),
            "is_serve": True, "detection_method": "serve_appearance",
            "impact_pixel_xy": [round(bx, 2), round(by, 2)],
            "impact_court_xy_ft": court_xy(f, bx, by),
            "player_distance_px": round(float(dist), 2), "assoc_basis": basis,
            "pre_velocity_px_per_frame": [None, None],
            "post_velocity_px_per_frame": vfield(launch),
            "speed_pre_px_per_frame": None,
            "speed_post_px_per_frame": round(math.hypot(*launch), 3),
            "direction_change_deg": None, "turn_rate_deg": None,
            "speed_change_ratio": None, "confidence": round(conf, 3),
        })
        n_serves += 1

    shots.sort(key=lambda s: s["frame"])
    for i, s in enumerate(shots):
        s["shot_id"] = i

    ball_visible_frac = float(vis.sum()) / n if n else 0.0
    if ball_visible_frac < params["ball_coverage_warn_frac"]:
        warnings.append(f"ball_visible_frac={ball_visible_frac:.2f} below "
                        f"{params['ball_coverage_warn_frac']:.2f}: shot recall "
                        f"will be poor (ball seldom detected).")

    stats = {
        "n_shots": len(shots),
        "n_serves": n_serves,
        "n_candidate_inflections": n_candidates,
        "n_rejected_no_player": n_no_player,
        "n_rejected_ball_gap": n_gap_rejected,
        "n_rejected_low_speed": n_low_speed,
        "n_merged_duplicates": suppressed,
        "ball_visible_frac": round(ball_visible_frac, 4),
        "analyzed_frame_range": [f_lo, f_hi],
    }
    return shots, stats, warnings


def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    court_path = folder / "court.json"
    ball_path = folder / "ball.parquet"
    ball_meta_path = folder / "ball.meta.json"
    players_path = folder / "players.parquet"
    poses_path = folder / "poses.parquet"
    out_path = folder / "shots.json"

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.", FileExistsError)

    court = load_court(court_path)
    ball_meta = load_ball_meta(ball_meta_path)
    df_ball = load_ball(ball_path, log)
    players_by_frame, n_player_rows = index_players(players_path)
    poses = index_poses(poses_path)

    # fps consistency
    fps = court["fps"] or ball_meta.get("video_fps")
    if fps is None or fps <= 0:
        fail("could not determine fps from court.json or ball.meta.json", ValueError)
    bfps = ball_meta.get("video_fps")
    if bfps and abs(float(bfps) - float(fps)) > FPS_TOLERANCE:
        fail(f"fps mismatch: court.json={fps}, ball.meta.json={bfps} "
             f"(> {FPS_TOLERANCE}). Refusing to run.", ValueError)

    ball_source = "synthetic" if ball_meta.get("synthetic") else "real"
    if ball_source == "synthetic":
        log.warning("ball_source is SYNTHETIC: shots are placeholder-derived, "
                    "not real detections.")

    fw = court["frame_width"] or ball_meta.get("video_width")
    fh = court["frame_height"] or ball_meta.get("video_height")

    params = {
        "fps": float(fps),
        "min_turn_rate_deg": args.min_turn_rate_deg,
        "min_speed_change_ratio": args.min_speed_change_ratio,
        "min_direction_change_deg": MIN_DIRECTION_CHANGE_DEG,
        "impact_window_frames": args.impact_window_frames,
        "velocity_window_frames": args.velocity_window_frames,
        "assoc_bbox_height_frac": ASSOC_BBOX_HEIGHT_FRAC,
        "assoc_max_px": args.assoc_max_px,
        "assoc_max_px_min": ASSOC_MAX_PX_MIN,
        "min_ball_speed_px_per_frame": MIN_BALL_SPEED_PX_PER_FRAME,
        "max_ball_speed_px_per_frame": args.max_ball_speed_px_per_frame,
        "ball_coverage_warn_frac": BALL_COVERAGE_WARN_FRAC,
        "serve_gap_frames": int(round(MIN_SERVE_GAP_S * float(fps))),
    }

    log.info(f"ball={len(df_ball)} frames ({ball_source}); "
             f"players={n_player_rows} non-transient rows; "
             f"poses indexed for {len(poses)} (frame,track) pairs")

    shots, stats, warnings = detect(df_ball, players_by_frame, poses,
                                    court["image_to_court"], log, params)

    if ball_source == "synthetic":
        warnings.insert(0, "ball_source is 'synthetic': shots are derived from "
                        "PLACEHOLDER ball data and are not real detections.")

    log.info(f"detected {stats['n_shots']} shots "
             f"({stats['n_serves']} serves; "
             f"{stats['n_candidate_inflections']} impulse candidates, "
             f"{stats['n_merged_duplicates']} merged, "
             f"{stats['n_rejected_no_player']} no-player, "
             f"{stats['n_rejected_ball_gap']} gap-limited, "
             f"{stats['n_rejected_low_speed']} low-speed)")

    out = {
        "schema_version": SCHEMA_VERSION,
        "video_path": ball_meta.get("video_path"),
        "fps": float(fps),
        "frame_width": int(fw) if fw else None,
        "frame_height": int(fh) if fh else None,
        "ball_source": ball_source,
        "params": params,
        "shots": shots,
        "stats": stats,
        "warnings": warnings,
        "stage_version": STAGE_VERSION,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    log.info(f"wrote {out_path}")
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 5 — detect shots")
    p.add_argument("folder", type=Path,
                   help="per-video folder with court.json, ball.parquet, "
                        "ball.meta.json, players.parquet, poses.parquet")
    p.add_argument("--force", action="store_true")
    p.add_argument("--min-turn-rate-deg", type=float, default=MIN_TURN_RATE_DEG,
                   dest="min_turn_rate_deg")
    p.add_argument("--min-speed-change-ratio", type=float,
                   default=MIN_SPEED_CHANGE_RATIO, dest="min_speed_change_ratio")
    p.add_argument("--impact-window-frames", type=int, default=IMPACT_WINDOW_FRAMES,
                   dest="impact_window_frames")
    p.add_argument("--velocity-window-frames", type=int,
                   default=VELOCITY_WINDOW_FRAMES, dest="velocity_window_frames")
    p.add_argument("--assoc-max-px", type=float, default=ASSOC_MAX_PX,
                   dest="assoc_max_px")
    p.add_argument("--max-ball-speed-px-per-frame", type=float,
                   default=MAX_BALL_SPEED_PX_PER_FRAME,
                   dest="max_ball_speed_px_per_frame")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    log = setup_logging(args.log_level)
    try:
        run(args.folder, args, log)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        log.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

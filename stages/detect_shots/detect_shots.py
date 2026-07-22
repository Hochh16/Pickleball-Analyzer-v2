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
STAGE_VERSION = "0.3.0"  # 0.2.0: real-ball adaptations (see contract "Real-ball
                         # adaptations"). 0.3.0: adjacent-court contamination gates
                         # (serve-run-length + impulse teleport-in), real ball only.

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
HANDLING_RESET_S = 3.0  # consecutive same-net-side impacts within this window = ball-handling
# Adjacent-court contamination gates (real ball only). On a multi-court venue the
# single-ball detector grabs a NEIGHBORING court's ball when ours is occluded,
# producing phantom shots/serves. Two trajectory-coherence gates reject them:
MIN_SERVE_RUN_S = 0.13  # a real serve launches a SUSTAINED run; a blip serve
                        # (other-court ball appearing briefly) does not. (8f @60fps)
TELEPORT_IN_PX_PER_FRAME = 40.0  # ref px/frame @1920 (scaled by frame_width/1920):
                        # an impulse impact whose ball run TELEPORTED in (jumped
                        # from where our ball actually was) is the other court's ball.
SERVE_DEDUP_S = 2.0     # two serve detections this close with no rally shot between
                        # = a pre-serve artifact + the real serve; keep the longer run.
REFERENCE_WIDTH_PX = 1920.0  # resolution the px defaults were tuned at; thresholds scale by frame_width/this
REFERENCE_FPS = 30.0  # fps the frame-count windows were tuned at; they scale by fps/this

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
    geom = c.get("court_geometry_feet", {}) or {}
    length_ft = float(geom.get("length_ft", 44.0))
    return {"image_to_court": M, "fps": video.get("fps"),
            "frame_width": video.get("frame_width"),
            "frame_height": video.get("frame_height"),
            "net_y_ft": length_ft / 2.0}


def load_track_roles(path: Path) -> Optional[Dict[int, str]]:
    """Stage 2.5 roles as {track_id: role}, or None if absent/unreadable. The
    authority on who the user is — players.parquet's is_user is click-only and
    empty in the no-clicks flow."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {int(t): info["role"] for t, info in d.get("track_roles", {}).items()}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


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


def index_players(path: Path, net_y_ft: float,
                  user_tids: Optional[set] = None,
                  participant_tids: Optional[set] = None
                  ) -> Tuple[Dict[int, List[dict]], int, Dict[int, str]]:
    """Index non-transient players by frame. `is_user` comes from `user_tids`
    (the Stage 2.5 role 'user') when provided, else from players.parquet's
    click-only flag (empty in the no-clicks flow). Also returns each track's
    net side ('near'/'far') from its median court_y — robust for every track
    (role-independent), used by the ball-handling alternation filter."""
    if not path.exists():
        fail(f"players.parquet not found: {path}", FileNotFoundError)
    df = pd.read_parquet(path)
    need = {"frame", "track_id", "is_user", "transient", "court_x_ft", "court_y_ft",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "foot_x", "foot_y"}
    missing = need - set(df.columns)
    if missing:
        fail(f"players.parquet missing columns: {sorted(missing)}", ValueError)
    df = df[~df["transient"]]
    # Keep ONLY the four match participants. On a multi-court venue the frame is
    # full of people on ADJACENT courts (Stage 2.5 role 'noise'); associating a
    # ball impulse with one of them manufactures a phantom shot. Measured on
    # pb_5_minute_outdoor-2: 38 of 155 detected shots (25%) were attributed to
    # noise tracks. Roles come from Stage 2.5, which runs before this stage.
    if participant_tids is not None:
        df = df[df["track_id"].isin(list(participant_tids))]
    side_by_track: Dict[int, str] = {}
    for tid, med_y in df.groupby("track_id")["court_y_ft"].median().items():
        if not np.isnan(med_y):
            side_by_track[int(tid)] = "near" if med_y < net_y_ft else "far"
    by_frame: Dict[int, List[dict]] = {}
    for r in df.itertuples(index=False):
        tid = int(r.track_id)
        is_user = (tid in user_tids) if user_tids is not None else bool(r.is_user)
        by_frame.setdefault(int(r.frame), []).append({
            "track_id": tid, "is_user": is_user,
            "bbox": (float(r.bbox_x1), float(r.bbox_y1),
                     float(r.bbox_x2), float(r.bbox_y2)),
            "foot": (float(r.foot_x), float(r.foot_y)),
            "court_xy": (float(r.court_x_ft), float(r.court_y_ft)),
        })
    return by_frame, len(df), side_by_track


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

def reject_same_side_runs(shots: List[dict], side_by_track: Dict[int, str],
                          reset_frames: int) -> Tuple[List[dict], int]:
    """Net-side alternation / ball-handling rejection. Every rally shot crosses
    the net, so the striker's net side must alternate; a run of consecutive
    same-side impacts means the ball stayed on one side — a player catching /
    holding / bouncing the ball between points, not rally shots (you can't
    legally hit twice in a row).

    Within each same-side run, keep the LAST impact and drop the earlier ones:
    handling precedes the real shot (you catch/bounce, THEN serve/hit), so the
    last same-side impact before the ball finally crosses is the real one. Runs
    are split by a side change or a gap longer than reset_frames (a new rally).
    Returns (kept, n_dropped)."""
    shots_sorted = sorted(shots, key=lambda x: x["frame"])
    kept: List[dict] = []
    n_dropped = 0
    run: List[dict] = []
    prev_side: Optional[str] = None
    prev_frame: Optional[int] = None

    def flush():
        nonlocal n_dropped
        if run:
            kept.append(run[-1])        # keep the LAST of the same-side run
            n_dropped += len(run) - 1

    for s in shots_sorted:
        side = side_by_track.get(s["track_id"])
        same_run = (run and side is not None and side == prev_side
                    and prev_frame is not None
                    and (s["frame"] - prev_frame) <= reset_frames)
        if same_run:
            run.append(s)
        else:
            flush()
            run = [s]
        prev_side, prev_frame = side, s["frame"]
    flush()
    kept.sort(key=lambda x: x["frame"])
    return kept, n_dropped


def detect(df_ball: pd.DataFrame, players_by_frame, poses, court_M,
           log: logging.Logger, params: dict,
           side_by_track: Optional[Dict[int, str]] = None) -> Tuple[List[dict], dict, List[str]]:
    n = len(df_ball)
    fx = df_ball["pixel_x"].to_numpy(copy=True)
    fy = df_ball["pixel_y"].to_numpy(copy=True)
    vis = df_ball["visible"].to_numpy(copy=True)
    interp = df_ball["interpolated"].to_numpy(copy=True)
    known = vis | interp
    warnings: List[str] = []

    known_idx = np.where(known)[0]
    if len(known_idx) == 0:
        return [], {"analyzed_frame_range": [0, 0]}, ["ball has no usable positions; zero shots."]
    f_lo, f_hi = int(known_idx[0]), int(known_idx[-1])

    # --- Defense: drop teleport / impossible-motion outliers (don't crash).
    #     Real ball detection leaves a few residual bad detections that survive
    #     Stage 4's postprocess; crashing the whole stage on one is wrong. Drop
    #     the later frame of each impossible pair (-> a gap). Left-to-right, this
    #     removes isolated spikes (both of a spike's pairs resolve from one drop).
    max_speed = params["max_ball_speed_px_per_frame"]
    n_teleport_dropped = 0
    for i in range(f_lo + 1, f_hi + 1):
        if known[i] and known[i - 1]:
            d = math.hypot(fx[i] - fx[i - 1], fy[i] - fy[i - 1])
            if d > max_speed:
                vis[i] = False
                interp[i] = False
                known[i] = False
                fx[i] = np.nan
                fy[i] = np.nan
                n_teleport_dropped += 1
    if n_teleport_dropped:
        msg = (f"dropped {n_teleport_dropped} ball detection(s) with impossible "
               f"motion (> {max_speed:.0f} px/frame); treated as gaps.")
        warnings.append(msg)
        log.warning(msg)

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

    net_y = params["net_y_ft"]

    def hitter_fields(p):
        """Reliable shot court-position from the HITTING PLAYER's GROUND position
        (court_xy from players.parquet), NOT the airborne ball-contact projection
        (impact_court_xy_ft), which explodes through the ground homography for an
        elevated contact. Downstream side logic (Stage 7) must use these."""
        cx, cy = p.get("court_xy", (float("nan"), float("nan")))
        side = None
        if not math.isnan(cy):
            side = "near" if cy < net_y else "far"
        xy = [round(cx, 2), round(cy, 2)] if not math.isnan(cx) else [None, None]
        return xy, side

    # --- Adjacent-court contamination gates (real ball only) ----------------
    def run_bounds(f):
        """[start, end] of the contiguous known-ball run containing frame f."""
        a = f
        while a - 1 >= 0 and known[a - 1]:
            a -= 1
        z = f
        while z + 1 < n and known[z + 1]:
            z += 1
        return a, z

    def teleport_in_pxpf(f):
        """How far (px/frame) the ball jumped from its last known position
        BEFORE the run containing f. A real rally ball is continuous; a
        neighbouring-court ball picked up mid-gap jumps in implausibly."""
        a, _ = run_bounds(f)
        p = a - 1
        while p >= 0 and not known[p]:
            p -= 1
        if p < 0:
            return 0.0
        d = math.hypot(fx[a] - fx[p], fy[a] - fy[p])
        return d / max(a - p, 1)

    contam_filter = bool(params.get("contamination_filter"))
    min_serve_run = params["min_serve_run_frames"]
    teleport_thresh = params["teleport_in_px_per_frame"]
    serve_dedup_frames = params["serve_dedup_frames"]
    n_rejected_serve_blip = 0
    n_rejected_teleport = 0

    # --- Impulse shots (rally hits) ----------------------------------------
    for f in accepted:
        # Adjacent-court gate: reject an impact whose ball run teleported in AND is
        # only a short blip. A real rally shot is usually occluded at the paddle
        # strike, so it too reappears "teleported" after the contact gap -- but it
        # then launches a SUSTAINED run to the next contact, whereas a neighbouring-
        # court ball flashes in for only a few frames. Requiring the blip length
        # keeps the contamination defense without eating real (gap-occluded) shots
        # (teleport-alone rejected ~80% of real shots on a multi-court venue).
        if contam_filter and teleport_in_pxpf(f) > teleport_thresh:
            a_run, z_run = run_bounds(f)
            if (z_run - a_run + 1) < min_serve_run:
                n_rejected_teleport += 1
                continue
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
            "hitter_court_xy_ft": hitter_fields(p)[0],
            "hitter_side": hitter_fields(p)[1],
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

    # --- Net-side alternation filter (real ball only). Rejects ball-handling
    #     (catch / hold / bounce between points) that the synthetic placeholder
    #     never produces. Gated to real ball because the synthetic generator does
    #     not model strict net-crossing alternation.
    if params.get("handling_filter"):
        shots, n_handling = reject_same_side_runs(
            shots, side_by_track or {}, params["handling_reset_frames"])
    else:
        n_handling = 0
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
        # Adjacent-court gate: a real serve launches a SUSTAINED ball run; a
        # neighbouring-court ball appearing briefly after dead time does not.
        if contam_filter and (run_bounds(f)[1] - f + 1) < min_serve_run:
            n_rejected_serve_blip += 1
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
            "hitter_court_xy_ft": hitter_fields(p)[0],
            "hitter_side": hitter_fields(p)[1],
            "player_distance_px": round(float(dist), 2), "assoc_basis": basis,
            "pre_velocity_px_per_frame": [None, None],
            "post_velocity_px_per_frame": vfield(launch),
            "speed_pre_px_per_frame": None,
            "speed_post_px_per_frame": round(math.hypot(*launch), 3),
            "direction_change_deg": None, "turn_rate_deg": None,
            "speed_change_ratio": None, "confidence": round(conf, 3),
        })
        n_serves += 1

    # --- Serve de-duplication (real ball) -----------------------------------
    # A point has exactly one serve. Two serve detections within
    # serve_dedup_frames with NO rally shot between them = a pre-serve artifact
    # (e.g. the server bouncing the ball before serving) plus the real serve;
    # keep the one whose ball run is longer (the launch that starts the rally).
    n_serve_dedup = 0
    if contam_filter:
        serve_shots = sorted((s for s in shots if s["is_serve"]),
                             key=lambda s: s["frame"])
        drop_frames: set = set()
        for i in range(len(serve_shots) - 1):
            a, b = serve_shots[i], serve_shots[i + 1]
            if a["frame"] in drop_frames:
                continue
            if (b["frame"] - a["frame"] <= serve_dedup_frames
                    and not any(a["frame"] < imf < b["frame"]
                                for imf in impulse_frames)):
                la = run_bounds(a["frame"])[1] - run_bounds(a["frame"])[0]
                lb = run_bounds(b["frame"])[1] - run_bounds(b["frame"])[0]
                drop_frames.add(a["frame"] if la < lb else b["frame"])
        if drop_frames:
            shots = [s for s in shots
                     if not (s["is_serve"] and s["frame"] in drop_frames)]
            n_serves -= len(drop_frames)
            n_serve_dedup = len(drop_frames)

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
        "n_teleport_dropped": n_teleport_dropped,
        "n_rejected_handling": n_handling,
        "n_rejected_serve_blip": n_rejected_serve_blip,
        "n_rejected_teleport_in": n_rejected_teleport,
        "n_serve_deduped": n_serve_dedup,
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
    roles = load_track_roles(folder / "track_roles.json")
    user_tids = None
    participant_tids = None
    if roles is not None:
        user_tids = {tid for tid, r in roles.items() if r == "user"}
        # Only the four match participants may be credited with a shot; everyone
        # else in frame is on an adjacent court (role 'noise').
        participant_tids = {tid for tid, r in roles.items() if r != "noise"}
        n_noise = sum(1 for r in roles.values() if r == "noise")
        log.info(f"using track_roles.json: is_user from {len(user_tids)} user track(s); "
                 f"{len(participant_tids)} participant track(s), excluding {n_noise} "
                 f"noise (adjacent-court) track(s) from shot association")
    players_by_frame, n_player_rows, side_by_track = index_players(
        players_path, court["net_y_ft"], user_tids, participant_tids)
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

    # Resolution scaling: px thresholds were tuned at 1080p; scale them by
    # frame_width / REFERENCE_WIDTH_PX so they adapt to 4K and other resolutions.
    # Angle/ratio thresholds are scale-invariant and not scaled. An explicit CLI
    # px override is taken as absolute (not re-scaled).
    res_scale = (float(fw) / REFERENCE_WIDTH_PX) if fw else 1.0
    assoc_max_px = (args.assoc_max_px if args.assoc_max_px is not None
                    else ASSOC_MAX_PX * res_scale)
    max_ball_speed = (args.max_ball_speed_px_per_frame
                      if args.max_ball_speed_px_per_frame is not None
                      else MAX_BALL_SPEED_PX_PER_FRAME * res_scale)
    if abs(res_scale - 1.0) > 1e-6:
        log.info(f"resolution scale {res_scale:.3f} (frame_width {fw} / "
                 f"{REFERENCE_WIDTH_PX:.0f}); px thresholds scaled accordingly")

    # Frame-rate scaling: the frame-count windows were tuned at 30fps. Scale them
    # by fps/REFERENCE_FPS so the merge + velocity windows keep the same real-time
    # duration (e.g. the 0.2s merge window = 12 frames at 60fps, not 6) — this is
    # what collapses the per-strike duplicate detections on high-fps footage.
    fps_scale = float(fps) / REFERENCE_FPS
    impact_window = (args.impact_window_frames if args.impact_window_frames is not None
                     else max(1, int(round(IMPACT_WINDOW_FRAMES * fps_scale))))
    velocity_window = (args.velocity_window_frames if args.velocity_window_frames is not None
                       else max(1, int(round(VELOCITY_WINDOW_FRAMES * fps_scale))))
    if abs(fps_scale - 1.0) > 1e-6:
        log.info(f"fps scale {fps_scale:.3f} (fps {fps} / {REFERENCE_FPS:.0f}); "
                 f"impact_window={impact_window}, velocity_window={velocity_window} frames")

    params = {
        "fps": float(fps),
        "min_turn_rate_deg": args.min_turn_rate_deg,
        "min_speed_change_ratio": args.min_speed_change_ratio,
        "min_direction_change_deg": MIN_DIRECTION_CHANGE_DEG,
        "impact_window_frames": impact_window,
        "velocity_window_frames": velocity_window,
        "assoc_bbox_height_frac": ASSOC_BBOX_HEIGHT_FRAC,
        "assoc_max_px": assoc_max_px,
        "assoc_max_px_min": ASSOC_MAX_PX_MIN * res_scale,
        "min_ball_speed_px_per_frame": MIN_BALL_SPEED_PX_PER_FRAME * res_scale,
        "max_ball_speed_px_per_frame": max_ball_speed,
        "ball_coverage_warn_frac": BALL_COVERAGE_WARN_FRAC,
        "serve_gap_frames": int(round(MIN_SERVE_GAP_S * float(fps))),
        "handling_reset_frames": int(round(HANDLING_RESET_S * float(fps))),
        "handling_filter": ball_source == "real",
        "contamination_filter": ball_source == "real",
        "min_serve_run_frames": max(2, int(round(MIN_SERVE_RUN_S * float(fps)))),
        "teleport_in_px_per_frame": TELEPORT_IN_PX_PER_FRAME * res_scale,
        "serve_dedup_frames": int(round(SERVE_DEDUP_S * float(fps))),
        "net_y_ft": court["net_y_ft"],
        "resolution_scale": round(res_scale, 4),
        "reference_width_px": REFERENCE_WIDTH_PX,
        "fps_scale": round(fps_scale, 4),
        "reference_fps": REFERENCE_FPS,
    }

    log.info(f"ball={len(df_ball)} frames ({ball_source}); "
             f"players={n_player_rows} non-transient rows; "
             f"poses indexed for {len(poses)} (frame,track) pairs")

    shots, stats, warnings = detect(df_ball, players_by_frame, poses,
                                    court["image_to_court"], log, params,
                                    side_by_track)

    if ball_source == "synthetic":
        warnings.insert(0, "ball_source is 'synthetic': shots are derived from "
                        "PLACEHOLDER ball data and are not real detections.")

    log.info(f"detected {stats['n_shots']} shots "
             f"({stats['n_serves']} serves; "
             f"{stats['n_candidate_inflections']} impulse candidates, "
             f"{stats['n_merged_duplicates']} merged, "
             f"{stats['n_rejected_no_player']} no-player, "
             f"{stats['n_rejected_ball_gap']} gap-limited, "
             f"{stats['n_rejected_low_speed']} low-speed, "
             f"{stats['n_rejected_serve_blip']} serve-blip, "
             f"{stats['n_rejected_teleport_in']} teleport-in)")

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
    p.add_argument("--impact-window-frames", type=int, default=None,
                   dest="impact_window_frames",
                   help="absolute frame override (default: IMPACT_WINDOW_FRAMES scaled by fps/30)")
    p.add_argument("--velocity-window-frames", type=int, default=None,
                   dest="velocity_window_frames",
                   help="absolute frame override (default: VELOCITY_WINDOW_FRAMES scaled by fps/30)")
    p.add_argument("--assoc-max-px", type=float, default=None,
                   dest="assoc_max_px",
                   help="absolute px override (default: ASSOC_MAX_PX scaled by "
                        "frame_width/1920)")
    p.add_argument("--max-ball-speed-px-per-frame", type=float, default=None,
                   dest="max_ball_speed_px_per_frame",
                   help="absolute px override (default: MAX_BALL_SPEED scaled by "
                        "frame_width/1920)")
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

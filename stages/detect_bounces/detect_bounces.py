"""Stage 5.5 — detect bounces.

Find every ground bounce of the ball and emit bounces.json. Reuses Stage 5's
impulse signal (single-frame turn-rate spike or sudden speed jump) with the
OPPOSITE proximity rule: bounces happen AWAY from players (whereas strikes
happen AT players). A y-velocity-flip tiebreaker recovers bounces landing AT a
player's feet — a common pickleball play (dinks/drops/resets landing at the
opponent) that a pure proximity rule would mis-drop.

See stages/detect_bounces/contract.md for the full spec.

Usage:
    python -m stages.detect_bounces.detect_bounces data/test_clip
    python -m stages.detect_bounces.detect_bounces data/test_clip --force
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
STAGE_VERSION = "0.2.0"  # 0.2.0: real-ball adaptations (see contract "Real-ball adaptations")

# --- Detection defaults (see contract "Configuration") ----------------------
# Shared with Stage 5 — same impulse signal, same association radius.
MIN_TURN_RATE_DEG = 45.0
MIN_SPEED_CHANGE_RATIO = 0.35
IMPACT_WINDOW_FRAMES = 6
VELOCITY_WINDOW_FRAMES = 3
ASSOC_BBOX_HEIGHT_FRAC = 0.5
ASSOC_MAX_PX = 120.0
ASSOC_MAX_PX_MIN = 30.0
MIN_BALL_SPEED_PX_PER_FRAME = 1.5
BALL_COVERAGE_WARN_FRAC = 0.30

# Shot-frame exclusion window. DELIBERATELY TIGHTER than IMPACT_WINDOW_FRAMES
# (the NMS window). A bounce 4 frames before the receiver's strike is a real,
# distinguishable event — the wider IMPACT_WINDOW_FRAMES would mask it. Only
# candidates this close to a shot are treated as duplicates of that shot.
SHOT_FRAME_EXCLUSION_WINDOW = 3

# Y-velocity-flip floor. A real ground bounce reverses vertical direction
# (descending -> ascending); requiring this for every bounce (not just at-feet
# ones) rejects mid-air noise wobbles. NOTE: the flip currently uses WINDOWED
# velocity, which smears the sharp 1-frame bounce reversal — the next fix is a
# displacement-based reversal at the refined ground-contact frame (see contract
# follow-ups), which should recover synthetic recall without a high floor.
Y_FLIP_MIN_SPEED_PX_PER_FRAME = 2.0
# A ground bounce is a local MAXIMUM of the ball's pixel_y (it descends to the
# surface then rebounds); an arc APEX is a pixel_y minimum, so peak-detecting
# pixel_y ignores apexes. Require this much descent INTO and rebound OUT OF the
# peak (px @1920, scaled by frame_width/1920) to reject flat mid-air wobble.
BOUNCE_PROMINENCE_PX = 9.0
AT_FEET_CONFIDENCE_FACTOR = 0.7

# In-court classification (court is 20 ft wide x 44 ft long).
IN_COURT_TOLERANCE_FT = 0.25
COURT_WIDTH_FT = 20.0
COURT_LENGTH_FT = 44.0
NET_Y_FT = 22.0

# A real ground bounce projects accurately (the ball is on the surface) — so it
# lands in-court or, for a ball-out, within a few ft of the lines. A candidate
# projecting FAR beyond this is the ball high in the air (an arc apex, whose
# y-velocity also flips) or a noise point, NOT a ground bounce — reject it. Real
# ball only (the synthetic ball's bounces are all clean/in-court, so it's a no-op
# there).
BOUNCE_MAX_OUT_OF_COURT_FT = 8.0

# Court zones — match Stage 6 so cross-stage references stay consistent.
KITCHEN_MAX_DIST_FT = 9.0
BASELINE_MIN_DIST_FT = 17.0

FPS_TOLERANCE = 0.5
EPS = 1e-9

# Real-ball scaling (same as Stage 5): px thresholds tuned at 1080p, frame-count
# windows tuned at 30fps. Scale by frame_width/1920 and fps/30 for 4K/60fps.
REFERENCE_WIDTH_PX = 1920.0
REFERENCE_FPS = 30.0


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("detect_bounces")
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
        fail(f"ball.meta.json not found: {path}. Stage 5.5 requires the ball "
             f"metadata sidecar (carries fps and the synthetic flag).",
             FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ball(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"ball.parquet not found: {path}", FileNotFoundError)
    df = pd.read_parquet(path)
    need = {"frame_idx", "pixel_x", "pixel_y", "visible", "interpolated"}
    missing = need - set(df.columns)
    if missing:
        fail(f"ball.parquet missing columns: {sorted(missing)}", ValueError)
    df = df.sort_values("frame_idx").reset_index(drop=True)
    # Re-assert Stage 4 schema invariants (defense against bad/placeholder data).
    vis = df["visible"].to_numpy()
    interp = df["interpolated"].to_numpy()
    if np.any(vis & interp):
        fail("ball.parquet invariant violated: visible AND interpolated on the "
             "same row", ValueError)
    known = vis | interp
    xy_nan = df["pixel_x"].isna().to_numpy() | df["pixel_y"].isna().to_numpy()
    if np.any(known & xy_nan):
        fail("ball.parquet invariant violated: known row with NaN pixel coords",
             ValueError)
    if np.any(~known & ~xy_nan):
        fail("ball.parquet invariant violated: not-known row with non-NaN coords",
             ValueError)
    return df


def load_shots(path: Path) -> dict:
    if not path.exists():
        fail(f"shots.json not found: {path}. Stage 5.5 requires shots.json "
             f"to exclude shot-impact frames and to attach between_shots.",
             FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_players(path: Path) -> Tuple[Dict[int, List[dict]], int]:
    if not path.exists():
        fail(f"players.parquet not found: {path}", FileNotFoundError)
    df = pd.read_parquet(path)
    need = {"frame", "track_id", "transient",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    missing = need - set(df.columns)
    if missing:
        fail(f"players.parquet missing columns: {sorted(missing)}", ValueError)
    df = df[~df["transient"]]
    by_frame: Dict[int, List[dict]] = {}
    for r in df.itertuples(index=False):
        by_frame.setdefault(int(r.frame), []).append({
            "track_id": int(r.track_id),
            "bbox": (float(r.bbox_x1), float(r.bbox_y1),
                     float(r.bbox_x2), float(r.bbox_y2)),
        })
    return by_frame, len(df)


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


def classify_in_court(cx: float, cy: float, tol: float
                      ) -> Tuple[Optional[bool], Optional[str], str]:
    """Return (is_in_court, out_side, court_zone). court_zone is one of
    kitchen / transition / baseline / out / unknown."""
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return None, None, "unknown"
    in_x = -tol <= cx <= COURT_WIDTH_FT + tol
    in_y = -tol <= cy <= COURT_LENGTH_FT + tol
    if in_x and in_y:
        dist_from_net = abs(cy - NET_Y_FT)
        if dist_from_net <= KITCHEN_MAX_DIST_FT:
            zone = "kitchen"
        elif dist_from_net >= BASELINE_MIN_DIST_FT:
            zone = "baseline"
        else:
            zone = "transition"
        return True, None, zone
    # Out of court — pick the boundary with the largest violation.
    viol: List[Tuple[str, float]] = []
    if cx < -tol:
        viol.append(("left", -tol - cx))
    if cx > COURT_WIDTH_FT + tol:
        viol.append(("right", cx - (COURT_WIDTH_FT + tol)))
    if cy < -tol:
        viol.append(("near", -tol - cy))
    if cy > COURT_LENGTH_FT + tol:
        viol.append(("far", cy - (COURT_LENGTH_FT + tol)))
    out_side = max(viol, key=lambda v: v[1])[0] if viol else None
    return False, out_side, "out"


# --- Core detection ----------------------------------------------------------

def detect(df_ball: pd.DataFrame, shots: List[dict], players_by_frame,
           court_M: np.ndarray, log: logging.Logger, params: dict
           ) -> Tuple[List[dict], dict, List[str]]:
    n = len(df_ball)
    fx = df_ball["pixel_x"].to_numpy()
    fy = df_ball["pixel_y"].to_numpy()
    vis = df_ball["visible"].to_numpy()
    interp = df_ball["interpolated"].to_numpy()
    known = vis | interp
    warnings: List[str] = []

    known_idx = np.where(known)[0]
    if len(known_idx) == 0:
        return [], {"analyzed_frame_range": [0, 0],
                    "n_candidate_inflections": 0,
                    "n_rejected_at_shot_frame": 0,
                    "n_rejected_no_yflip": 0,
                    "n_rejected_low_speed": 0,
                    "n_rejected_in_ball_gap": 0,
                    "ball_visible_frac": 0.0}, \
               ["ball has no usable positions; zero bounces."]
    f_lo, f_hi = int(known_idx[0]), int(known_idx[-1])

    # --- Single-frame velocity, turn rate, speed-change ratio (matches Stage 5).
    def vel(i):
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
        s_in = float(np.linalg.norm(v_in))
        s_out = float(np.linalg.norm(v_out))
        sratio[i] = abs(s_out - s_in) / max(s_in, s_out, EPS)

    # --- Windowed velocity (nearest known neighbor within k frames). Used both
    # for the at-feet y-flip check and for reporting per-bounce speeds.
    k = params["velocity_window_frames"]

    def nearest_known(i: int, forward: bool) -> Optional[int]:
        for j in range(1, k + 1):
            t = i + j if forward else i - j
            if 0 <= t < n and known[t]:
                return t
        return None

    def windowed_velocity(i: int):
        fb = nearest_known(i, False)
        ff = nearest_known(i, True)
        if fb is None or ff is None:
            return None, None
        v_in = np.array([(fx[i] - fx[fb]) / (i - fb),
                         (fy[i] - fy[fb]) / (i - fb)])
        v_out = np.array([(fx[ff] - fx[i]) / (ff - i),
                          (fy[ff] - fy[i]) / (ff - i)])
        return v_in, v_out

    # --- Bounce candidates: local MAXIMA of the ball's pixel_y (the descent bottom
    # where a ground contact reverses vertical direction). An arc apex is a pixel_y
    # MINIMUM, so this primitive ignores apexes -- unlike the generic impulse signal
    # (single-frame turn/speed), which fired at both and buried the ~10 real bounces
    # under ~230 noise candidates on a jittery ball. Detect peaks on a gap-
    # interpolated, lightly-smoothed pixel_y and require real descent INTO and
    # rebound OUT OF the peak (prominence) to reject flat mid-air wobble; the y-flip
    # check below then re-confirms the vertical reversal on the raw trajectory.
    n_low_speed = 0
    n_gap_rejected = 0
    prom = params["bounce_prominence_px"]
    Wp = params["impact_window_frames"]
    yi = (np.interp(np.arange(n), known_idx, fy[known_idx])
          if len(known_idx) >= 2 else fy.copy())     # fill gaps so hidden bounces peak
    ys = np.convolve(yi, np.ones(3) / 3.0, mode="same")  # tame per-frame jitter
    cand: List[Tuple[int, float]] = []
    for i in range(Wp, n - Wp):
        if ys[i] < ys[i - Wp:i + Wp + 1].max() - EPS:
            continue                                   # not the local pixel_y max
        descent = ys[i] - ys[i - Wp:i].min()           # fell this far into the peak
        rebound = ys[i] - ys[i + 1:i + Wp + 1].min()   # rose this far back out
        if descent < prom or rebound < prom:
            continue
        if not known[max(0, i - 2):i + 3].any():
            continue                                   # peak sits deep inside a gap
        cand.append((i, float(min(descent, rebound))))
    n_candidates = len(cand)

    # --- Exclude candidates within SHOT_FRAME_EXCLUSION_WINDOW of any Stage-5
    # shot — those candidates are the same paddle-hit event Stage 5 already
    # captured. The window is intentionally TIGHTER than IMPACT_WINDOW_FRAMES
    # (the NMS window) so a bounce 4 frames before a strike is treated as a
    # separate event, not absorbed into the strike.
    #
    # CRUCIALLY, this step runs BEFORE NMS, not after. If we NMS'd first, a
    # bounce candidate at R-4 (the at-feet case) would get suppressed by the
    # stronger strike candidate at R within the ±6 NMS window — and never
    # reach the bounce-detection branch. Filtering shot-frames first removes
    # the strike candidates so the bounce candidate survives NMS.
    shot_frames = sorted(int(s["frame"]) for s in shots)
    W_shot = params["shot_frame_exclusion_window"]

    def near_shot(f: int) -> bool:
        for sf in shot_frames:
            if abs(f - sf) <= W_shot:
                return True
        return False

    pre_nms: List[Tuple[int, float]] = []
    n_rejected_at_shot_frame = 0
    for f, score in cand:
        if near_shot(f):
            n_rejected_at_shot_frame += 1
            continue
        pre_nms.append((f, score))

    # --- Non-maximum suppression within IMPACT_WINDOW_FRAMES on the remaining
    # candidates. Same window as Stage 5's NMS — but applied AFTER strikes are
    # filtered out, so it only dedupes bounce-signal clusters with each other.
    W = params["impact_window_frames"]
    pre_nms.sort(key=lambda c: c[1], reverse=True)
    accepted_frames: List[int] = []
    for f, _ in pre_nms:
        if any(abs(f - a) <= W for a in accepted_frames):
            continue
        accepted_frames.append(f)
    accepted_frames.sort()
    after_shot_filter = accepted_frames

    # --- Proximity classification + at-feet y-flip tiebreaker.
    bbox_frac = params["assoc_bbox_height_frac"]
    amax = params["assoc_max_px"]
    amin = params["assoc_max_px_min"]
    y_flip_floor = params["y_flip_min_speed_px_per_frame"]
    at_feet_factor = params["at_feet_confidence_factor"]
    tol = params["in_court_tolerance_ft"]

    def nearest_player(f: int, bx: float, by: float
                       ) -> Tuple[Optional[dict], float, float]:
        """Among non-transient players on frame f, return (player, distance, r)
        for the player whose bbox is closest to (bx, by). Distance is bbox
        rectangle distance; r is the perspective-scaled association radius for
        that player (matches Stage 5)."""
        best = None
        for p in players_by_frame.get(f, []):
            _, y1, _, y2 = p["bbox"]
            radius = min(max(bbox_frac * max(1.0, y2 - y1), amin), amax)
            d = bbox_distance(p["bbox"], bx, by)
            key = (d, p["track_id"])
            if best is None or key < best[0]:
                best = (key, p, radius)
        if best is None:
            return None, float("inf"), float("inf")
        return best[1], best[0][0], best[2]

    yflip_floor = 0.3 * params["resolution_scale"]

    def y_flipped(i: int) -> Tuple[Optional[bool], Optional[float], Optional[float]]:
        """A real ground bounce reverses vertical direction: descending (v_y > 0,
        pixel_y increasing) then ascending (v_y < 0). Measured on the SMOOTHED
        trajectory over the velocity window — the raw per-frame velocity is too
        jittery and its floor was tuned for sharp impulse hits, but a soft dink
        bounce rebounds gently (~prominence/window px/frame). A small floor passes
        real bounces while rejecting flat mid-air wobble."""
        lo, hi = max(0, i - k), min(n - 1, i + k)
        if i == lo or i == hi:
            return None, None, None
        v_y_in = float((ys[i] - ys[lo]) / (i - lo))
        v_y_out = float((ys[hi] - ys[i]) / (hi - i))
        flipped = bool((v_y_in > +yflip_floor) and (v_y_out < -yflip_floor))
        return flipped, v_y_in, v_y_out

    n_rejected_no_yflip = 0
    require_yflip_away = params["require_yflip_away"]  # real ball only
    surviving: List[Tuple[int, Optional[dict], float, float, bool]] = []
    # tuple = (frame, nearest_player_or_None_if_far, nearest_dist, radius, is_at_feet)
    # at-feet bounce (near a player): always require the y-flip tiebreaker.
    # away-from-player bounce: on REAL ball require the y-flip too (an impulse
    # with NO vertical reversal = the ball changing direction IN THE AIR, a noisy
    # mid-air wobble, not a ground contact). The synthetic placeholder is clean
    # (no mid-air noise), so it keeps the impulse-only behavior — that's why this
    # is gated to real ball. `is_at_feet` is just "at a player's feet" (proximity),
    # orthogonal to the court zone.
    for f in after_shot_filter:
        bx, by = float(fx[f]), float(fy[f])
        p, dist, radius = nearest_player(f, bx, by)
        at_player = p is not None and dist < radius
        if at_player or require_yflip_away:
            flipped, _, _ = y_flipped(f)
            if flipped is not True:
                n_rejected_no_yflip += 1
                continue
        surviving.append((f, p, dist, radius, at_player))

    # --- Build bounce records.
    def shot_context(f: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Returns (prev_shot_id, next_shot_id, frames_since_prev, frames_to_next)."""
        prev_id: Optional[int] = None
        prev_frame: Optional[int] = None
        next_id: Optional[int] = None
        next_frame: Optional[int] = None
        for s in shots:
            sf = int(s["frame"])
            if sf < f:
                prev_id = int(s["shot_id"])
                prev_frame = sf
            elif sf > f and next_id is None:
                next_id = int(s["shot_id"])
                next_frame = sf
                break
        fs_prev = (f - prev_frame) if prev_frame is not None else None
        fs_next = (next_frame - f) if next_frame is not None else None
        return prev_id, next_id, fs_prev, fs_next

    def quality(f: int) -> float:
        w0, w1 = max(0, f - W), min(n, f + W + 1)
        return float(vis[w0:w1].mean()) if w1 > w0 else 0.0

    bounces: List[dict] = []
    by_zone = {"kitchen": 0, "transition": 0, "baseline": 0, "out": 0, "unknown": 0}
    by_out_side = {"near": 0, "far": 0, "left": 0, "right": 0}
    n_at_feet = 0
    n_in_court = 0
    n_out = 0
    n_rejected_far_out = 0
    m_out = BOUNCE_MAX_OUT_OF_COURT_FT

    def ground_contact(fd: int) -> int:
        """Refine a bounce's detection frame to the true ground-contact frame =
        the ball's lowest point (max pixel_y, since y increases downward) within a
        small window. The y-flip can fire a few frames before/after contact; at
        contact the ball is on the surface, so its court projection — and zone —
        is most accurate (esp. on the perspective-compressed far court)."""
        w = params["velocity_window_frames"]
        best_f, best_y = fd, fy[fd]
        for g in range(max(f_lo, fd - w), min(f_hi, fd + w) + 1):
            if known[g] and fy[g] > best_y:
                best_f, best_y = g, fy[g]
        return best_f

    for fd, p, dist, radius, is_at_feet in surviving:
        f = ground_contact(fd)  # bounce frame + position refined to ground contact
        bx, by = float(fx[f]), float(fy[f])
        cx, cy = project_to_court(court_M, bx, by)
        # Reject apex/noise: a real ground bounce projects accurately (in-court or
        # near the lines for a ball-out); a candidate projecting far out is the
        # ball high in the air (arc apex) or noise, not a ground bounce.
        if (not math.isfinite(cx) or not math.isfinite(cy)
                or cx < -m_out or cx > COURT_WIDTH_FT + m_out
                or cy < -m_out or cy > COURT_LENGTH_FT + m_out):
            n_rejected_far_out += 1
            continue
        in_court, out_side, zone = classify_in_court(cx, cy, tol)
        if not math.isfinite(cx):
            warnings.append(f"bounce at frame {f}: court projection non-finite; "
                            f"court_xy_ft=null")
            court_xy = [None, None]
        else:
            court_xy = [round(cx, 2), round(cy, 2)]
        prev_id, next_id, fs_prev, fs_next = shot_context(f)

        flipped, v_y_in, v_y_out = y_flipped(f)
        v_in_full, v_out_full = windowed_velocity(f)
        if v_in_full is not None:
            s_pre = float(np.linalg.norm(v_in_full))
        else:
            s_pre = float("nan")
        if v_out_full is not None:
            s_post = float(np.linalg.norm(v_out_full))
        else:
            s_post = float("nan")

        # Confidence: blend impulse sharpness, distance-from-player normalized,
        # and ball-data quality around the bounce. At-feet bounces are
        # downweighted by AT_FEET_CONFIDENCE_FACTOR (they could still be a
        # Stage-5-missed shot that happens to have a y-flip).
        impulse_term = max(min(1.0, turn[fd] / 120.0), min(1.0, sratio[fd]))
        if is_at_feet:
            # Closer to the player ⇒ MORE confident the y-flip means at-feet
            # (rather than coincidence at the edge of the proximity radius).
            prox_term = 1.0 - min(1.0, dist / max(radius, EPS))
        else:
            # Farther from the player ⇒ more confidently a ground bounce.
            prox_term = min(1.0, dist / max(radius, EPS)) if math.isfinite(dist) else 1.0
        conf = float(np.clip(0.5 * impulse_term + 0.3 * prox_term + 0.2 * quality(f),
                             0.0, 1.0))
        if is_at_feet:
            conf *= at_feet_factor

        bounces.append({
            "bounce_id": 0,  # assigned after sort
            "frame": int(f),
            "t_sec": round(f / params["fps"], 3),
            "pixel_xy": [round(bx, 2), round(by, 2)],
            "court_xy_ft": court_xy,
            "is_in_court": in_court,
            "court_zone": zone,
            "out_side": out_side,
            "between_shots": [prev_id, next_id],
            "frames_since_prev_shot": fs_prev,
            "frames_to_next_shot": fs_next,
            "is_at_feet": bool(is_at_feet),
            "nearest_player_distance_px": (round(float(dist), 2)
                                            if math.isfinite(dist) else None),
            "nearest_player_track_id": (int(p["track_id"]) if (p is not None
                                         and is_at_feet) else None),
            "y_velocity_flipped": (None if flipped is None else bool(flipped)),
            "turn_rate_deg": round(float(turn[f]), 1),
            "speed_change_ratio": round(float(sratio[f]), 3),
            "ball_speed_pre_px_per_frame": (round(s_pre, 3)
                                             if math.isfinite(s_pre) else None),
            "ball_speed_post_px_per_frame": (round(s_post, 3)
                                              if math.isfinite(s_post) else None),
            "confidence": round(conf, 3),
        })
        by_zone[zone] = by_zone.get(zone, 0) + 1
        if out_side is not None:
            by_out_side[out_side] = by_out_side.get(out_side, 0) + 1
        if in_court is True:
            n_in_court += 1
        elif in_court is False:
            n_out += 1
        if is_at_feet:
            n_at_feet += 1

    bounces.sort(key=lambda b: b["frame"])
    for i, b in enumerate(bounces):
        b["bounce_id"] = i

    ball_visible_frac = float(vis.sum()) / n if n else 0.0
    if ball_visible_frac < params["ball_coverage_warn_frac"]:
        warnings.append(f"ball_visible_frac={ball_visible_frac:.2f} below "
                        f"{params['ball_coverage_warn_frac']:.2f}: bounce "
                        f"recall will be poor (ball seldom detected).")

    stats = {
        "n_bounces": len(bounces),
        "n_in_court": n_in_court,
        "n_out": n_out,
        "n_at_feet": n_at_feet,
        "n_candidate_inflections": n_candidates,
        "n_rejected_at_shot_frame": n_rejected_at_shot_frame,
        "n_rejected_no_yflip": n_rejected_no_yflip,
        "n_rejected_low_speed": n_low_speed,
        "n_rejected_in_ball_gap": n_gap_rejected,
        "n_rejected_far_out": n_rejected_far_out,
        "by_zone": {k: v for k, v in by_zone.items() if v > 0},
        "by_out_side": {k: v for k, v in by_out_side.items() if v > 0},
        "ball_visible_frac": round(ball_visible_frac, 4),
        "analyzed_frame_range": [f_lo, f_hi],
    }
    return bounces, stats, warnings


def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    court_path = folder / "court.json"
    ball_path = folder / "ball.parquet"
    ball_meta_path = folder / "ball.meta.json"
    players_path = folder / "players.parquet"
    shots_path = folder / "shots.json"
    out_path = folder / "bounces.json"

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.", FileExistsError)

    court = load_court(court_path)
    ball_meta = load_ball_meta(ball_meta_path)
    df_ball = load_ball(ball_path)
    players_by_frame, n_player_rows = index_players(players_path)
    shots_doc = load_shots(shots_path)

    fps = court["fps"] or ball_meta.get("video_fps")
    if fps is None or fps <= 0:
        fail("could not determine fps from court.json or ball.meta.json", ValueError)
    bfps = ball_meta.get("video_fps")
    if bfps and abs(float(bfps) - float(fps)) > FPS_TOLERANCE:
        fail(f"fps mismatch: court.json={fps}, ball.meta.json={bfps} "
             f"(> {FPS_TOLERANCE}). Refusing to run.", ValueError)

    ball_source = "synthetic" if ball_meta.get("synthetic") else "real"
    if ball_source == "synthetic":
        log.warning("ball_source is SYNTHETIC: bounces are placeholder-derived, "
                    "not real detections.")

    fw = court["frame_width"] or ball_meta.get("video_width")
    fh = court["frame_height"] or ball_meta.get("video_height")

    shots = sorted(shots_doc.get("shots", []), key=lambda s: s["frame"])

    # Resolution + fps scaling (see Stage 5): px thresholds scale by
    # frame_width/1920, frame-count windows by fps/30. Angle/ratio thresholds are
    # scale-invariant. Explicit CLI overrides are taken as absolute.
    res_scale = (float(fw) / REFERENCE_WIDTH_PX) if fw else 1.0
    fps_scale = float(fps) / REFERENCE_FPS
    sc_i = lambda base, ov: ov if ov is not None else max(1, int(round(base * fps_scale)))
    assoc_max_px = (args.assoc_max_px if args.assoc_max_px is not None
                    else ASSOC_MAX_PX * res_scale)
    y_flip_min = (args.y_flip_min_speed if args.y_flip_min_speed is not None
                  else Y_FLIP_MIN_SPEED_PX_PER_FRAME * res_scale)
    if abs(res_scale - 1.0) > 1e-6 or abs(fps_scale - 1.0) > 1e-6:
        log.info(f"scaling: res_scale={res_scale:.3f} (fw {fw}), "
                 f"fps_scale={fps_scale:.3f} (fps {fps})")

    params = {
        "fps": float(fps),
        "min_turn_rate_deg": args.min_turn_rate_deg,
        "min_speed_change_ratio": args.min_speed_change_ratio,
        "impact_window_frames": sc_i(IMPACT_WINDOW_FRAMES, args.impact_window_frames),
        "velocity_window_frames": sc_i(VELOCITY_WINDOW_FRAMES, args.velocity_window_frames),
        "shot_frame_exclusion_window": sc_i(SHOT_FRAME_EXCLUSION_WINDOW, args.shot_frame_exclusion_window),
        "assoc_bbox_height_frac": ASSOC_BBOX_HEIGHT_FRAC,
        "assoc_max_px": assoc_max_px,
        "assoc_max_px_min": ASSOC_MAX_PX_MIN * res_scale,
        "min_ball_speed_px_per_frame": MIN_BALL_SPEED_PX_PER_FRAME * res_scale,
        "bounce_prominence_px": BOUNCE_PROMINENCE_PX * res_scale,
        "y_flip_min_speed_px_per_frame": y_flip_min,
        "at_feet_confidence_factor": AT_FEET_CONFIDENCE_FACTOR,
        "in_court_tolerance_ft": args.in_court_tolerance_ft,
        "ball_coverage_warn_frac": BALL_COVERAGE_WARN_FRAC,
        "resolution_scale": round(res_scale, 4),
        "fps_scale": round(fps_scale, 4),
        "require_yflip_away": ball_source == "real",
    }

    log.info(f"ball={len(df_ball)} frames ({ball_source}); "
             f"players={n_player_rows} non-transient rows; "
             f"shots={len(shots)} from shots.json")

    bounces, stats, warnings = detect(df_ball, shots, players_by_frame,
                                      court["image_to_court"], log, params)

    if ball_source == "synthetic":
        warnings.insert(0, "ball_source is 'synthetic': bounces are derived "
                        "from PLACEHOLDER ball data and are not real detections.")

    log.info(f"detected {stats['n_bounces']} bounces "
             f"({stats['n_at_feet']} at-feet, "
             f"{stats['n_in_court']} in / {stats['n_out']} out; "
             f"{stats['n_candidate_inflections']} impulse candidates, "
             f"{stats['n_rejected_at_shot_frame']} at-shot-frame, "
             f"{stats['n_rejected_no_yflip']} no-yflip, "
             f"{stats['n_rejected_low_speed']} low-speed)")

    out = {
        "schema_version": SCHEMA_VERSION,
        "video_path": ball_meta.get("video_path"),
        "fps": float(fps),
        "frame_width": int(fw) if fw else None,
        "frame_height": int(fh) if fh else None,
        "ball_source": ball_source,
        "source_shots": str(shots_path),
        "params": params,
        "bounces": bounces,
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
    p = argparse.ArgumentParser(description="Stage 5.5 — detect bounces")
    p.add_argument("folder", type=Path,
                   help="per-video folder with court.json, ball.parquet, "
                        "ball.meta.json, players.parquet, shots.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--min-turn-rate-deg", type=float, default=MIN_TURN_RATE_DEG,
                   dest="min_turn_rate_deg")
    p.add_argument("--min-speed-change-ratio", type=float,
                   default=MIN_SPEED_CHANGE_RATIO, dest="min_speed_change_ratio")
    p.add_argument("--impact-window-frames", type=int, default=None,
                   dest="impact_window_frames", help="default: scaled by fps/30")
    p.add_argument("--velocity-window-frames", type=int,
                   default=None, dest="velocity_window_frames", help="default: scaled by fps/30")
    p.add_argument("--assoc-max-px", type=float, default=None,
                   dest="assoc_max_px", help="default: scaled by frame_width/1920")
    p.add_argument("--y-flip-min-speed", type=float,
                   default=None, dest="y_flip_min_speed",
                   help="Min |v_y| on each side for the y-velocity-flip tiebreaker "
                        "(default: scaled by frame_width/1920)")
    p.add_argument("--shot-frame-exclusion-window", type=int,
                   default=None,
                   dest="shot_frame_exclusion_window",
                   help="Drop candidates within +/-N frames of any Stage-5 shot "
                        "(default: scaled by fps/30)")
    p.add_argument("--in-court-tolerance-ft", type=float,
                   default=IN_COURT_TOLERANCE_FT, dest="in_court_tolerance_ft")
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

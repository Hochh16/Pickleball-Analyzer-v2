"""Stage 6 — classify shots.

Label each shot from shots.json with a stroke side (forehand/backhand), a shot
type (serve/drive/dink/drop/lob/overhead/reset/unknown), and a bounce-based
volley flag. Rule-based, with honest "unknown" when the signal is weak.

v1: real forehand/backhand only for the USER (handedness from roster.json,
mapped via is_user); non-user stroke side is "unknown" until a player-role-
classification stage exists. See stages/classify_shots/contract.md.

Usage:
    python -m stages.classify_shots.classify_shots data/test_clip [--force]
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
STAGE_VERSION = "0.4.0"  # 0.1.0 -> 0.2.0: is_volley consumed bounces.json.
                         # 0.2.0 -> 0.3.0 (real-ball): is_volley primary signal is
                         # a recall-focused LOCAL ball-trajectory scan between
                         # shots (the precision bounce list under-detects on the
                         # real ball -> false volleys); bounce list is now a
                         # fallback for occluded gaps. + lob requires below-drive
                         # speed; + fps/resolution scaling.
                         # 0.3.0 -> 0.4.0 (real-ball): LANDING-AWARE shot type --
                         # the airborne ball's pixel-speed is depth-corrupted and
                         # its court projection explodes, so when a real bounce
                         # landing exists (~21% of shots) the landing court_y drives
                         # the drive/drop/dink split (a sound, ground-projected
                         # signal); speed/arc remain the fallback otherwise. Adds
                         # features.landing_court_y + features.type_from_landing.
                         # See SYSTEM_DESIGN.md Stage 6 ledger for coverage limits.

# --- Config (see contract) --------------------------------------------------
LOB_MIN_ARC_FRAC = 0.35
DRIVE_DROP_ARC_SPLIT = 0.15  # tweener (16-25 ft/s) tiebreak: flatter=drive, loftier=drop
DRIVE_MIN_SPEED_FTPS = 25.0
DINK_MAX_SPEED_FTPS = 16.0
RESET_MIN_INCOMING_FTPS = 25.0
# Stage 5.7 ground-anchored HORIZONTAL speed (range/airtime) is an AVERAGE, so it
# runs lower than the instantaneous ppf speed; calibrated on operator ground truth
# (drill + match rally 10): clean dinks 11-23, drives 23-32 ft/s. Used only when the
# trajectory confidence clears TRAJ_SPEED_CONF_MIN, else the ppf speed + old thresholds.
DRIVE_MIN_SPEED_HORIZ_FTPS = 26.0
DINK_MAX_SPEED_HORIZ_FTPS = 23.0
TRAJ_SPEED_CONF_MIN = 0.6
POST_TRAJ_FRAMES = 15
MAX_ARC_FRAMES = 45          # cap the arc-measurement window (bounds dead-time gaps)
REFERENCE_FPS = 30.0         # frame-count windows tuned at 30fps; scale by fps/this for 60fps real footage
REFERENCE_WIDTH_PX = 1920.0  # px thresholds tuned at 1920-wide; scale by frame_width/this (4K = 2x)
VOLLEY_REBOUND_MIN_PX = 20.0  # min upward rebound after the low point to call a ground bounce (volley test)
VOLLEY_DESCENT_MIN_PX = 14.0  # min descent into the low point (ball clearly came down)
SIDE_CONF_FLOOR = 0.5
KITCHEN_MAX_DIST_FT = 9.0   # effective kitchen depth from net (court_zones)
BASELINE_MIN_DIST_FT = 17.0  # within ~5ft of the 22ft baseline
BOUNCE_MIN_TURN_DEG = 40.0   # single-frame turn between shots => ground bounce
LANDMARK_VIS_FLOOR = 0.5
NET_Y_FT = 22.0

SHOT_TYPES = {"serve", "drive", "dink", "drop", "lob", "reset", "unknown"}
# 'overhead' is a STROKE (above-the-head contact), recorded on stroke_side, not a
# tactical shot type. Stroke axis: forehand / backhand / overhead / unknown.
STROKE_SIDES = {"forehand", "backhand", "overhead", "unknown"}
STROKE_SIDES = {"forehand", "backhand", "unknown"}
EPS = 1e-9


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("classify_shots")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


# --- Loaders -----------------------------------------------------------------

def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"required input not found: {path}", FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_court(path: Path) -> dict:
    c = load_json(path)
    derived = c.get("derived", {}) or {}
    near = derived.get("pixels_per_foot_at_near_baseline")
    far = derived.get("pixels_per_foot_at_far_baseline")
    video = c.get("video", {}) or {}
    homo = c.get("homography", {}) or {}
    i2c = homo.get("image_to_court")
    img2court = np.array(i2c, dtype=float) if i2c is not None else None
    return {"ppf_near": near, "ppf_far": far, "fps": video.get("fps"),
            "image_to_court": img2court}


def project_court_y(court: dict, px: float, py: float) -> Optional[float]:
    """Project an image point to its court_y (feet). Ground-plane homography."""
    M = court.get("image_to_court")
    if M is None:
        return None
    v = M @ np.array([px, py, 1.0])
    if abs(v[2]) < 1e-9:
        return None
    return float(v[1] / v[2])


def front_foot_court_y(court: dict, pose: Optional[dict],
                       fallback: float) -> float:
    """Court_y of the player's FRONT foot = the ankle projecting CLOSEST to the
    net. A dinker leans in, so the front foot is within ~2 ft of the kitchen line
    while the bbox-bottom (the REAR foot, nearer the camera) reads several feet
    deeper -- using the rear foot mis-reads a kitchen dink as a transition/drop.
    Falls back to the bbox-foot court_y when pose/ankles/homography are missing."""
    if pose is None or court.get("image_to_court") is None:
        return fallback
    VIS = 0.3
    # Seed with the bbox foot so the result is NEVER deeper than it: the true
    # front foot is at least as close to the net as the bbox-bottom point. On the
    # NEAR side the bbox-bottom is the rear foot (reads too deep) and a front
    # ankle pulls it toward the net; on the FAR side the bbox-bottom is already
    # the front foot, so a noisy/occluded rear ankle can't push the read deeper.
    cands = [fallback]
    for xk, yk, vk in (("lax", "lay", "lav"), ("rax", "ray", "rav")):
        px, py, v = pose.get(xk), pose.get(yk), pose.get(vk)
        if px is None or py is None or (v is not None and v < VIS):
            continue
        cy = project_court_y(court, float(px), float(py))
        if cy is not None:
            cands.append(cy)
    # front foot = the foot nearest the net line (works on both court halves)
    return min(cands, key=lambda cy: abs(cy - NET_Y_FT))


def load_roster(path: Path, log: logging.Logger) -> Dict[str, str]:
    if not path.exists():
        log.warning(f"roster.json not found ({path}); user handedness unknown")
        return {}
    r = load_json(path)
    return r.get("handedness", {}) or {}


def index_players(path: Path) -> Dict[Tuple[int, int], dict]:
    if not path.exists():
        fail(f"players.parquet not found: {path}", FileNotFoundError)
    df = pd.read_parquet(path)
    out: Dict[Tuple[int, int], dict] = {}
    for r in df.itertuples(index=False):
        out[(int(r.frame), int(r.track_id))] = {
            "court_y": float(r.court_y_ft),
            "bbox": (float(r.bbox_x1), float(r.bbox_y1),
                     float(r.bbox_x2), float(r.bbox_y2)),
            "foot": (float(r.foot_x), float(r.foot_y)),
        }
    # also a per-frame list of all player pixel points (for bounce away-check)
    per_frame: Dict[int, List[Tuple[float, float]]] = {}
    for r in df.itertuples(index=False):
        per_frame.setdefault(int(r.frame), []).append((float(r.foot_x), float(r.foot_y)))
    return out, per_frame


def index_poses(path: Path) -> Dict[Tuple[int, int], dict]:
    if not path.exists():
        return {}
    cols = ["frame", "track_id", "pose_detected",
            "left_shoulder_x_px", "left_shoulder_y_px", "left_shoulder_visibility",
            "right_shoulder_x_px", "right_shoulder_y_px", "right_shoulder_visibility",
            "left_hip_y_px", "left_hip_visibility",
            "right_hip_y_px", "right_hip_visibility",
            "left_ankle_x_px", "left_ankle_y_px", "left_ankle_visibility",
            "right_ankle_x_px", "right_ankle_y_px", "right_ankle_visibility"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["pose_detected"]]
    out: Dict[Tuple[int, int], dict] = {}
    for r in df.itertuples(index=False):
        out[(int(r.frame), int(r.track_id))] = {
            "lsx": r.left_shoulder_x_px, "lsy": r.left_shoulder_y_px,
            "lsv": r.left_shoulder_visibility,
            "rsx": r.right_shoulder_x_px, "rsy": r.right_shoulder_y_px,
            "rsv": r.right_shoulder_visibility,
            "lhy": r.left_hip_y_px, "lhv": r.left_hip_visibility,
            "rhy": r.right_hip_y_px, "rhv": r.right_hip_visibility,
            "lax": r.left_ankle_x_px, "lay": r.left_ankle_y_px,
            "lav": r.left_ankle_visibility,
            "rax": r.right_ankle_x_px, "ray": r.right_ankle_y_px,
            "rav": r.right_ankle_visibility,
        }
    return out


def load_ball(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        fail(f"ball.parquet not found: {path}", FileNotFoundError)
    df = pd.read_parquet(path).sort_values("frame_idx").reset_index(drop=True)
    x = df["pixel_x"].to_numpy()
    y = df["pixel_y"].to_numpy()
    known = (df["visible"].to_numpy() | df["interpolated"].to_numpy())
    return x, y, known


# --- Feature helpers ---------------------------------------------------------

def ppf_at(court: dict, court_y: float) -> Optional[float]:
    near, far = court["ppf_near"], court["ppf_far"]
    if near is None or far is None:
        return None
    t = max(0.0, min(1.0, court_y / 44.0))
    return near + t * (far - near)


def speed_ftps(speed_pxpf: Optional[float], court: dict, court_y: float,
               fps: float) -> Optional[float]:
    if speed_pxpf is None:
        return None
    ppf = ppf_at(court, court_y)
    if ppf is None or ppf < EPS:
        return None
    return float(speed_pxpf) * fps / ppf


def zone_from_court_y(court_y: float) -> str:
    dist_from_net = abs(court_y - NET_Y_FT)
    if dist_from_net <= KITCHEN_MAX_DIST_FT:
        return "kitchen"
    if dist_from_net >= BASELINE_MIN_DIST_FT:
        return "baseline"
    return "transition"


def arc_height_frac(x, y, known, f0: int, f_end: int) -> Optional[float]:
    """Max upward (smaller-y) deviation of the post-shot trajectory from the
    straight contact->end chord, as a fraction of the chord length."""
    n = len(x)
    f1 = min(f_end, n - 1)
    pts = [(int(f), float(x[f]), float(y[f]))
           for f in range(f0, f1 + 1) if 0 <= f < n and known[f]]
    if len(pts) < 3:
        return None
    (fa, xa, ya), (fb, xb, yb) = pts[0], pts[-1]
    chord = math.hypot(xb - xa, yb - ya)
    if chord < EPS:
        return None
    max_up = 0.0
    for (f, px, py) in pts[1:-1]:
        t = (f - fa) / (fb - fa) if fb != fa else 0.0
        line_y = ya + t * (yb - ya)
        up = line_y - py  # positive when ball is ABOVE the chord (smaller y)
        if up > max_up:
            max_up = up
    return max_up / chord


def contact_height(impact_y: float, pose: Optional[dict]) -> str:
    if pose is None:
        return "mid"
    sh = [v for v, vis in ((pose["lsy"], pose["lsv"]), (pose["rsy"], pose["rsv"]))
          if vis >= LANDMARK_VIS_FLOOR and not _nan(v)]
    hp = [v for v, vis in ((pose["lhy"], pose["lhv"]), (pose["rhy"], pose["rhv"]))
          if vis >= LANDMARK_VIS_FLOOR and not _nan(v)]
    if sh and impact_y <= (sum(sh) / len(sh)):
        return "high"
    if hp and impact_y >= (sum(hp) / len(hp)):
        return "low"
    return "mid"


def _nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def stroke_side(impact_x: float, pose: Optional[dict],
                handedness: Optional[str]) -> Tuple[str, float]:
    if handedness not in ("left", "right") or pose is None:
        return "unknown", 0.0
    if (pose["lsv"] < LANDMARK_VIS_FLOOR or pose["rsv"] < LANDMARK_VIS_FLOOR
            or _nan(pose["lsx"]) or _nan(pose["rsx"])):
        return "unknown", 0.0
    lsx, rsx = float(pose["lsx"]), float(pose["rsx"])
    center_x = 0.5 * (lsx + rsx)
    shoulder_w = abs(rsx - lsx)
    if shoulder_w < EPS:
        return "unknown", 0.0
    # Anatomical right shoulder on the image-right => player's back to camera.
    facing_away = rsx > lsx
    body_right_sign = 1.0 if facing_away else -1.0
    offset = impact_x - center_x
    # Forehand is on the dominant-hand side of the body.
    dom_sign = body_right_sign if handedness == "right" else -body_right_sign
    side = "forehand" if (offset * dom_sign) > 0 else "backhand"
    conf = min(1.0, abs(offset) / (0.5 * shoulder_w))
    if conf < SIDE_CONF_FLOOR:
        return "unknown", round(conf, 3)
    return side, round(conf, 3)


def turn_deg(x, y, f) -> Optional[float]:
    n = len(x)
    if f - 1 < 0 or f + 1 >= n:
        return None
    a = np.array([x[f] - x[f - 1], y[f] - y[f - 1]])
    b = np.array([x[f + 1] - x[f], y[f + 1] - y[f]])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def build_bounces_between_index(bounces_doc: dict) -> Dict[Tuple[int, int], int]:
    """Map (prev_shot_id, next_shot_id) -> count of bounces sitting between them.
    Used by the volley check: is_volley = (count == 0)."""
    idx: Dict[Tuple[int, int], int] = {}
    for b in bounces_doc.get("bounces", []):
        bs = b.get("between_shots", [None, None])
        if bs[0] is None or bs[1] is None:
            continue
        key = (int(bs[0]), int(bs[1]))
        idx[key] = idx.get(key, 0) + 1
    return idx


def build_landing_index(bounces_doc: dict) -> Dict[int, float]:
    """shot_id -> LANDING court_y: the receiver-side ground-contact court_y of the
    first bounce after the shot. Bounces are ON THE GROUND, so they project to
    court coordinates reliably (unlike the airborne ball contact, whose ground-
    homography projection explodes — see KNOWN_ISSUES Stage 6 depth-speed). This
    is the SOUND signal for the drive/drop/dink/lob split: a drive lands deep, a
    drop/dink lands within the kitchen (+~2 ft)."""
    out: Dict[int, float] = {}
    for b in sorted(bounces_doc.get("bounces", []), key=lambda b: b.get("frame", 0)):
        bs = b.get("between_shots", [None, None])
        if bs[0] is None:
            continue
        sid = int(bs[0])
        if sid in out:
            continue  # keep the earliest bounce after the shot = its landing
        cxy = b.get("court_xy_ft")
        if cxy and cxy[1] is not None:
            out[sid] = float(cxy[1])
    return out


def bounced_between(by, bknown, f0: int, f1: int,
                    rebound_min_px: float, descent_min_px: float):
    """Recall-focused local test for the VOLLEY flag: did the ball bounce off the
    ground in the open interval (f0, f1) between two consecutive shots?

    This is deliberately decoupled from the precision-tuned Stage 5.5 bounce LIST
    (which exists for exact zone stats and filters out apex/in-air wobble). For
    is_volley we only need recall: a ground bounce shows up in screen space as a
    descending->ascending reversal of the ball -- pixel_y climbs to a clear low
    point (peak pixel_y) and then rebounds upward. The outgoing arc's apex is the
    opposite (a pixel_y minimum) and is correctly ignored.

    Returns True  (a down->up rebound occurred -> NOT a volley),
            False (continuous descent/flat into contact, no rebound -> volley),
            None  (inconclusive: too little visible trajectory to judge -- the
                   caller falls back to the bounce list).
    """
    n = len(bknown)
    fr = [k for k in range(f0 + 1, f1) if 0 <= k < n and bool(bknown[k])]
    if len(fr) < 5:
        return None  # occluded / too sparse -> let the caller fall back
    ys = np.array([by[k] for k in fr], dtype=float)
    # A ground bounce is an INTERIOR local peak in pixel_y (ball at a momentary
    # lowest-on-screen point) with the ball descending INTO it and rebounding UP
    # out of it. NOT the global pixel_y max: the trajectory usually starts at a
    # high pixel_y (the previous contact is low on screen) and the outgoing arc's
    # apex is a pixel_y MINIMUM -- both must be ignored. Scan for a peak that
    # dominates a small neighbourhood, with descent-in and rebound-out both real.
    for j in range(2, len(ys) - 2):
        lo, hi = max(0, j - 4), min(len(ys), j + 5)
        if ys[j] < ys[lo:hi].max():
            continue  # not the local peak in its window -> not the bounce point
        descent_in = ys[j] - ys[:j].min()      # fell from the arc apex down to here
        rebound = ys[j] - ys[j + 1:].min()     # rose back up afterwards
        if descent_in >= descent_min_px and rebound >= rebound_min_px:
            return True
    return False


# --- Classification ----------------------------------------------------------

def classify_type(is_serve, arc_frac, contact_h, post_ftps, pre_ftps, zone,
                  landing_y=None, receiver_zone=None, is_volley=False,
                  drive_min=DRIVE_MIN_SPEED_FTPS, dink_max=DINK_MAX_SPEED_FTPS):
    """Fused rule classifier for the TACTICAL shot type. The airborne ball's
    pixel-speed is depth-corrupted (a drive hit down-court reads slow) and its
    court projection explodes, so when a real bounce LANDING is available
    (`landing_y` = the ball's landing court_y) it drives the drive/drop/dink split
    — a *sound*, ground-projected signal. Falls back to arc + speed when there's no
    landing (volleys, missed bounces), at lower confidence.
    `post_ftps` is the outgoing shot speed; `drive_min`/`dink_max` are its
    thresholds. When Stage 5.7 supplies a confident GROUND-ANCHORED horizontal
    speed the caller passes it here with the horizontal-calibrated thresholds
    (~26/23); otherwise it's the depth-corrupted ppf speed with the old ~25/16.
    `receiver_zone` = the zone of the player about to receive (needed for a lob).
    NOTE: 'overhead' is a STROKE (above-the-head contact), not a tactical type —
    it's set on stroke_side by the caller; a high-contact ball is tactically a
    drive/put-away here. Returns (type, confidence)."""
    if is_serve:
        return "serve", 0.95
    # --- Volley (ball taken out of the air; no bounce, so no landing signal).
    #     Operator rules: classify from the hitter's zone + speed. At the kitchen a
    #     slow air-ball is a dink, a fast one a speed-up drive; taken out of the air
    #     from transition/baseline it's a drive. (A lob is judged on the PRIOR shot
    #     via the receiver running back — not modeled here yet.)
    if is_volley:
        if zone == "kitchen":
            if post_ftps is not None and post_ftps >= drive_min:
                return "drive", 0.6
            return "dink", 0.6
        return "drive", 0.6
    # A lob is a high lofted arc AND slow AND goes over a receiver AT THE KITCHEN
    # (a lob only makes sense against a player at the net; a soft high ball to
    # deep opponents is a drop/drive, not a lob). Resolve before the landing split.
    if (arc_frac is not None and arc_frac >= LOB_MIN_ARC_FRAC
            and (post_ftps is None or post_ftps < drive_min)
            and receiver_zone == "kitchen"):
        return "lob", min(1.0, max(0.6, arc_frac))

    # --- Landing-aware path: the SOUND signal (bounces project reliably) ---------
    if landing_y is not None:
        # soft landing within the kitchen + ~2 ft buffer (operator: a drop/dink
        # lands up to ~2 ft past the kitchen line, not only inside the kitchen).
        soft = abs(landing_y - NET_Y_FT) <= KITCHEN_MAX_DIST_FT
        if soft:
            # dink is hit from AT/NEAR the net (kitchen or transition — operator:
            # players dink from a step or two behind the kitchen line); a drop is
            # the soft shot from DEEP (baseline, the third-shot drop).
            return ("drop", 0.78) if zone == "baseline" else ("dink", 0.78)
        # Speed guard: a drive REQUIRES real pace. A slow ball near the net whose
        # landing read a bit deep (soft-shot depth is noisy for an airborne ball)
        # is still a dink, not a drive. Baseline stays out of this (could be a drop).
        if (post_ftps is not None and post_ftps <= dink_max
                and zone != "baseline"):
            return "dink", 0.7
        return "drive", 0.78   # deep landing + real pace = a flat fast ball

    # --- Fallback (no landing): arc + speed, lower confidence --------------------
    if post_ftps is not None and post_ftps >= drive_min:
        return "drive", 0.6
    if (pre_ftps is not None and pre_ftps >= RESET_MIN_INCOMING_FTPS
            and post_ftps is not None and post_ftps <= dink_max
            and zone != "baseline"):
        return "reset", 0.55
    if post_ftps is not None and post_ftps <= dink_max and zone in ("kitchen", "transition"):
        return "dink", 0.5   # slow ball hit from at/near the net = dink (a step back still dinks)
    if post_ftps is not None and post_ftps <= dink_max and zone == "baseline":
        return "drop", 0.45  # slow ball from deep = drop (third-shot drop)
    # Tweener (dink_max..drive_min) with no landing: speed is ambiguous, so resolve
    # by trajectory SHAPE -- flat => drive, lofted => drop.
    if post_ftps is not None and dink_max < post_ftps < drive_min:
        if arc_frac is not None and arc_frac >= DRIVE_DROP_ARC_SPLIT:
            return "drop", 0.4
        return "drive", 0.4
    return "unknown", 0.3


def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    shots_path = folder / "shots.json"
    out_path = folder / "classified.json"
    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.", FileExistsError)

    shots_doc = load_json(shots_path)
    bounces_doc = load_json(folder / "bounces.json")  # required: Stage 5.5 output
    court = load_court(folder / "court.json")
    roster = load_roster(folder / "roster.json", log)
    players, players_by_frame = index_players(folder / "players.parquet")
    poses = index_poses(folder / "poses.parquet")
    bx, by, bknown = load_ball(folder / "ball.parquet")

    fps = shots_doc.get("fps") or court["fps"]
    if not fps or fps <= 0:
        fail("could not determine fps", ValueError)
    # fps scaling: the arc/trajectory frame windows were tuned at 30fps; scale
    # them so they keep the same real-time duration on 60fps footage (speeds are
    # already in ft/s, so they need no scaling).
    fps_scale = float(fps) / REFERENCE_FPS
    max_arc_frames = max(1, int(round(MAX_ARC_FRAMES * fps_scale)))
    post_traj_frames = max(1, int(round(POST_TRAJ_FRAMES * fps_scale)))
    # resolution scaling: px thresholds were tuned at 1920-wide footage.
    frame_width = float(shots_doc.get("frame_width") or REFERENCE_WIDTH_PX)
    res_scale = frame_width / REFERENCE_WIDTH_PX
    volley_rebound_px = VOLLEY_REBOUND_MIN_PX * res_scale
    volley_descent_px = VOLLEY_DESCENT_MIN_PX * res_scale
    user_hand = roster.get("user")
    ball_source = shots_doc.get("ball_source", "real")

    # is_volley primary signal: a recall-focused LOCAL trajectory scan of the ball
    # between consecutive shots (did it bounce off the ground?). This is decoupled
    # from the precision-tuned Stage 5.5 bounce LIST, which under-detects bounces
    # on the noisy real ball (missed bounce -> false volley). The bounce list is
    # kept only as a fallback when the local trajectory is too occluded to judge.
    bounces_between = build_bounces_between_index(bounces_doc)
    # shot_id -> landing court_y (sound, ground-projected signal for shot type)
    landing_index = build_landing_index(bounces_doc)
    # shot_id -> Stage 5.7 ground-anchored horizontal speed (physical; replaces the
    # depth-corrupted ppf speed for dink/drive when confident). Optional input:
    # older bundles / pipelines without Stage 5.7 fall back to the ppf speed.
    traj_index: Dict[int, dict] = {}
    traj_path = folder / "trajectory.json"
    if traj_path.exists():
        for t in load_json(traj_path).get("shots", []):
            traj_index[int(t["shot_id"])] = t
        log.info(f"loaded Stage 5.7 trajectory speeds for {len(traj_index)} shots")

    shots = sorted(shots_doc.get("shots", []), key=lambda s: s["frame"])
    out_shots = []
    warnings = list(shots_doc.get("warnings", []))
    prev_frame = None
    prev_shot_id: Optional[int] = None

    for i, s in enumerate(shots):
        f = int(s["frame"])
        # arc is measured over the full OUTGOING segment (to the next shot),
        # capped, so a long lob's bow isn't truncated.
        next_frame = int(shots[i + 1]["frame"]) if i + 1 < len(shots) else None
        if next_frame is not None and next_frame - f <= max_arc_frames:
            arc_end = next_frame - 1
        else:
            arc_end = f + max_arc_frames
        tid = int(s["track_id"])
        is_user = bool(s.get("is_user"))
        is_serve = bool(s.get("is_serve"))
        impact_x, impact_y = s["impact_pixel_xy"]
        pdata = players.get((f, tid))
        pose = poses.get((f, tid))
        court_y = pdata["court_y"] if pdata else NET_Y_FT
        # zone uses the FRONT foot (ankle nearest the net): a dinker's front foot
        # is within ~2 ft of the kitchen line while the bbox-bottom is the rear
        # foot (nearer the camera) and reads several feet deeper. Speed still uses
        # court_y (player depth) for its pixels-per-foot scaling.
        zone_court_y = front_foot_court_y(court, pose, court_y)
        zone = zone_from_court_y(zone_court_y)

        post_ftps = speed_ftps(s.get("speed_post_px_per_frame"), court, court_y, fps)
        pre_ftps = speed_ftps(s.get("speed_pre_px_per_frame"), court, court_y, fps)
        arc_frac = arc_height_frac(bx, by, bknown, f, arc_end)
        contact_h = contact_height(float(impact_y), pose)

        # landing court_y from the first bounce after this shot (sound signal)
        landing_y = landing_index.get(int(s["shot_id"]))
        # receiver = the player about to hit next; their zone WHEN THIS SHOT is
        # struck decides whether a high slow ball is a lob (only vs a net player).
        receiver_zone = None
        if i + 1 < len(shots):
            r_tid = int(shots[i + 1]["track_id"])
            r_pd = players.get((f, r_tid))
            if r_pd is not None:
                r_cy = front_foot_court_y(court, poses.get((f, r_tid)),
                                          r_pd["court_y"])
                receiver_zone = zone_from_court_y(r_cy)
        # volley: a shot is a volley iff the ball did NOT bounce since the
        # previous shot. Primary = local trajectory scan (recall-focused);
        # fall back to the Stage 5.5 bounce list only when the ball is too
        # occluded between the two shots to judge locally. Computed BEFORE the
        # type so a volley (no landing) uses the volley rules, not the fallback.
        shot_id = int(s["shot_id"])
        if is_serve or prev_shot_id is None or prev_frame is None:
            is_volley, vol_conf = False, 0.9
        else:
            local = bounced_between(by, bknown, prev_frame, f,
                                    volley_rebound_px, volley_descent_px)
            if local is None:
                # inconclusive (occluded) -> fall back to the precision bounce list
                n_b = bounces_between.get((prev_shot_id, shot_id), 0)
                is_volley = (n_b == 0)
                vol_conf = 0.5
            else:
                is_volley = not local  # bounce found -> not a volley
                vol_conf = 0.85

        # Prefer the Stage 5.7 ground-anchored horizontal speed when confident: it's
        # physical (the ppf speed explodes on airborne balls). Different scale ->
        # horizontal-calibrated thresholds; otherwise the ppf speed + old thresholds.
        tj = traj_index.get(shot_id)
        # Consistency guard: if the ball was VOLLEYED (no bounce), a trajectory
        # "bounce" anchor is a phantom (a far-side false bounce) — its long range
        # reads as a confident-but-wrong drive. Distrust it and fall back.
        traj_phantom = (tj is not None and is_volley
                        and tj.get("anchor_type") == "bounce")
        if (tj is not None and not traj_phantom
                and tj.get("horizontal_speed_ftps") is not None
                and tj.get("confidence", 0.0) >= TRAJ_SPEED_CONF_MIN):
            speed_for_type = tj["horizontal_speed_ftps"]
            d_min, d_max = DRIVE_MIN_SPEED_HORIZ_FTPS, DINK_MAX_SPEED_HORIZ_FTPS
            speed_source = "trajectory_horizontal"
        else:
            speed_for_type = post_ftps
            d_min, d_max = DRIVE_MIN_SPEED_FTPS, DINK_MAX_SPEED_FTPS
            speed_source = "ppf_instantaneous"

        shot_type, type_conf = classify_type(is_serve, arc_frac, contact_h,
                                             speed_for_type, pre_ftps, zone, landing_y,
                                             receiver_zone, is_volley,
                                             drive_min=d_min, dink_max=d_max)

        # stroke side: forehand/backhand for the user (handedness known); an
        # above-the-head contact is an 'overhead' stroke regardless of handedness.
        hand = user_hand if is_user else None
        side, side_conf = stroke_side(float(impact_x), pose, hand)
        if contact_h == "high":
            side, side_conf = "overhead", 0.7

        out = dict(s)  # carry through all Stage 5 fields
        out.update({
            "stroke_side": side,
            "stroke_side_confidence": side_conf,
            "shot_type": shot_type,
            "shot_type_confidence": round(type_conf, 3),
            "is_volley": bool(is_volley),
            "is_volley_confidence": round(vol_conf, 3),
            "features": {
                "contact_zone": zone,
                "contact_front_foot_y": round(zone_court_y, 2),
                "contact_bbox_foot_y": round(court_y, 2),
                "post_speed_ftps": round(post_ftps, 2) if post_ftps is not None else None,
                "speed_used_ftps": round(speed_for_type, 2) if speed_for_type is not None else None,
                "speed_source": speed_source,
                "pre_speed_ftps": round(pre_ftps, 2) if pre_ftps is not None else None,
                "arc_height_frac": round(arc_frac, 3) if arc_frac is not None else None,
                "contact_height": contact_h,
                # landing court_y (sound shot-type signal) + whether the type came
                # from the landing path (reliable) vs the speed/arc fallback.
                "landing_court_y": round(landing_y, 2) if landing_y is not None else None,
                "type_from_landing": landing_y is not None,
                "handedness_used": hand,
                "handedness_known": hand in ("left", "right"),
            },
        })
        out_shots.append(out)
        prev_frame = f
        prev_shot_id = shot_id

    # stats
    from collections import Counter
    by_type = Counter(s["shot_type"] for s in out_shots)
    by_side = Counter(s["stroke_side"] for s in out_shots)
    stats = {
        "n_shots": len(out_shots),
        "by_shot_type": dict(by_type),
        "by_stroke_side": dict(by_side),
        "n_volley": sum(1 for s in out_shots if s["is_volley"]),
        "n_volley_fallback": sum(1 for s in out_shots if s["is_volley_confidence"] == 0.5),
        "n_unknown_type": by_type.get("unknown", 0),
        "n_unknown_side": by_side.get("unknown", 0),
    }

    if ball_source == "synthetic":
        msg = "ball_source is 'synthetic': classifications are derived from PLACEHOLDER ball data."
        if msg not in warnings:
            warnings.insert(0, msg)
        log.warning("ball_source is SYNTHETIC: classifications are placeholder-derived.")
    if user_hand not in ("left", "right"):
        warnings.append("user handedness unknown (roster.json); user stroke side will be 'unknown'.")

    log.info(f"classified {len(out_shots)} shots; types={dict(by_type)}; "
             f"sides={dict(by_side)}; volleys={stats['n_volley']}")

    out_doc = {
        "schema_version": SCHEMA_VERSION,
        "source_shots": str(shots_path),
        "ball_source": ball_source,
        "fps": float(fps),
        "params": {
            "lob_min_arc_frac": LOB_MIN_ARC_FRAC,
            "drive_min_speed_ftps": DRIVE_MIN_SPEED_FTPS,
            "dink_max_speed_ftps": DINK_MAX_SPEED_FTPS,
            "reset_min_incoming_ftps": RESET_MIN_INCOMING_FTPS,
            "post_traj_frames": post_traj_frames,
            "max_arc_frames": max_arc_frames,
            "fps_scale": round(fps_scale, 4),
            "resolution_scale": round(res_scale, 4),
            "volley_rebound_min_px": round(volley_rebound_px, 1),
            "volley_descent_min_px": round(volley_descent_px, 1),
            "bounce_min_turn_deg": BOUNCE_MIN_TURN_DEG,
        },
        "shots": out_shots,
        "stats": stats,
        "warnings": warnings,
        "stage_version": STAGE_VERSION,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)
        f.write("\n")
    log.info(f"wrote {out_path}")
    return out_doc


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 6 — classify shots")
    p.add_argument("folder", type=Path)
    p.add_argument("--force", action="store_true")
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

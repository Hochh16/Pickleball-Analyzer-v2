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
STAGE_VERSION = "0.7.0"  # 0.6.0 -> 0.7.0: RETURN OF SERVE is now its own shot type.
                         # Structural (the shot after the serve, from the other side),
                         # not a new signal. Previously every return scored as a drive:
                         # 0/4 on the operator's labels; now 3/4, and the clip yields 15
                         # returns vs the operator's truth of 14.
                         # 0.5.0 -> 0.6.0: a landing must be on the OPPOSITE side of
                         # the net (net hits excepted). 23% of landings were same-side
                         # mis-associations -- ball-handling bounces typed as landings.
                         # 0.4.0 -> 0.5.0: REMOVED the landing-path "speed guard"
                         # (a slow ball with a DEEP landing was called a dink). It
                         # contradicted the operator's ruling that the LANDING decides
                         # type, and rested on ball speed, a known-weak discriminator.
                         # dinks 36 -> 31 (truth 18) on pb_5_minute_outdoor-2.
                         # 0.1.0 -> 0.2.0: is_volley consumed bounces.json.
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
# Ratio of the anchored drive median to the net-crossing drive median
# (37.0 / 57.8), so the two speeds share the drive/dink thresholds.
NET_CROSS_SPEED_SCALE = 0.64

# Height-based volley test. The ball is 1-6 ft up in flight and reads 0.17-0.38 ft at a
# detected bounce across all four clips, so the two cases are far apart -- but `bias` scales
# absolute z (KNOWN_ISSUES), so the rule leans on the SHAPE (came down, went back up) and uses
# the absolute only to separate "near the ground" from "clearly not".
VOLLEY_GROUND_FT = 1.0      # zmin at or under this, with a rebound, is a ground contact
VOLLEY_AIRBORNE_FT = 1.5    # zmin above this never touched down -> a volley
VOLLEY_MIN_REBOUND_FT = 0.8  # the ball must visibly come back up out of the low point
VOLLEY_MIN_FRAMES = 4       # reconstructed frames needed between the two shots to decide

# WHAT THE CURRENT FEATURES CAN AND CANNOT DO (measured 2026-09-02, 100 labelled
# non-serve shots on the two reviewed clips). The operator's definitions:
#
#     dink   struck from the kitchen out to ~2 ft beyond (FRONT LEG), landing in the
#            opponent's kitchen, hit softly
#     drop   the same but struck from the transition zone or deeper; one that lands
#            mid-court is a FAILED drop, still a drop
#     drive  higher speed AND lower arc, together
#     lob    high arc AND long flight time
#     "no one criteria will work by itself"
#
# Each criterion separately:
#     hitter position   STRONG for dink vs drop -- front foot 7.6 ft from the net against
#                       20.0 ft, and the front foot beats the body centre (dink p75 falls
#                       from 11.3 to 8.7 ft, which is the operator's rule almost exactly)
#     landing depth     STRONG where it exists -- dink 6.9 ft past the net, drop 7.2,
#                       drive 13.1 -- but only ~45% of shots have one, and the gap is a
#                       60fps limit, not a detector fault (see build_landing_index)
#     arc               MODERATE, drop 0.272 against drive 0.155, but overlapping lob:
#                       drops reach 0.47 against a lob threshold of 0.35
#     flight time       MODERATE and consistent with the operator: dink 1.10s, drive 0.78,
#                       drop 1.07, lob 1.57
#     ball speed        UNRELIABLE. Pixel speed is INVERTED by camera distance (17 px/f for
#                       misread drops against 13 for real drives) and the physical ft/s
#                       reads 43 against 41. It is nonetheless what the no-landing path
#                       tests FIRST and ALONE.
#     body mechanics    WEAK as currently computed. Knee angle is IDENTICAL for dink and
#                       drop (158.2 deg both) and the post/pre speed ratio is LOWEST for
#                       drives (1.41 against 1.70 dink, 2.00 drop) -- the opposite of the
#                       aggressive acceleration that defines one. Only contact height
#                       carries anything: 31/49 drives are struck mid or high against
#                       8/26 dinks.
#     SWING speed       NOT MEASURABLE from this footage, though the operator is right that
#                       it should be decisive. Peak wrist speed normalised by the player's
#                       own pixel height is distance-invariant, which is exactly what ball
#                       pixel speed is not -- so it was the most promising idea available.
#                       It carries nothing: drives 2.06 body-lengths/sec against 2.01 for
#                       everything else, and at every threshold drives and non-drives are
#                       caught at the same rate (26/49 against 26/52 at 2.0). The joints are
#                       not the problem -- wrist visibility is 93-98% -- the SCALE is: the
#                       player is a median 148 px tall (p10 91), so a swing spans tens of
#                       pixels and its per-frame displacement is a few, comparable to the
#                       tracker's own jitter. A first attempt using the peak rather than a
#                       smoothed percentile gave dinks 19.8 body-lengths/sec, which is what
#                       measuring noise looks like.
#
# CORRECTION, and it matters: ball speed DOES separate a drive, contrary to what an earlier
# note here implied. That claim came from the misread drops only -- the subset where speed
# had already failed -- which is a biased sample. Measured over all labelled shots using the
# PHYSICAL trajectory speed and discarding the pixel-derived one, drives run 37.0 ft/s
# against dink 25.0, drop 27.0, lob 20.7. It is a real signal with heavy overlap (some drops
# are struck hard), not the absence of one. Mixing the pixel speed in is what hides it:
# together they read 30.7 against 24.5.
#
# And in COMBINATION: a search over 288 rules built from position, arc, speed and flight
# time -- including the front foot and the operator's own thresholds -- scores 60/100,
# exactly what the current classifier scores. Rules that lift drop recall to 55% pay for it
# in dink and lob. So the ceiling here is the FEATURES, not the rule, and the way past it
# is a new measurement rather than a better combination of these: swing path and backswing
# amplitude over time (not currently computed), or landing coverage.
#
# --- Fallback-path confidences: CALIBRATED against operator ground truth ------
# Measured on 21 operator-labelled shots (20 s drill + match rally 10), 2026-07-21:
#   landing-based path   : 73% accurate, reported 75%  -> honest, left alone
#   fallback (no landing): 33% accurate, reported 58%  -> OVERCONFIDENT by +24 pts
# The fallback fires exactly where we are blind (volleys / missed bounces). Stage 8
# averages these per-shot confidences (mv_sourced) into the metrics that drive the
# USAPA rating, so the overconfidence made a noisy read look precise. These values
# put the fallback's mean confidence near its measured accuracy, so a clip with many
# no-landing shots now yields LOWER confidence and a WIDER rating range instead of a
# falsely tight number. Re-measure if the classifier changes.
FB_DRIVE = 0.40      # was 0.60
FB_DINK = 0.35       # was 0.50 (0.60 on the volley branch)
FB_RESET = 0.35      # was 0.55
FB_DROP = 0.30       # was 0.45
FB_TWEENER = 0.25    # was 0.40 -- speed ambiguous, resolved only by arc shape
FB_UNKNOWN = 0.20    # was 0.30

# is_volley confidence, likewise CALIBRATED. Measured against operator volley truth
# on match rally 10 (clean ball track): 5/10 correct = ~50%, while the local-scan
# path reported 0.85. Bounce-vs-volley is the monocular precision floor (three
# independent height-free signals tested and defeated -- see docs/ACCURACY_LEDGER.md),
# so this must be reported as a SOFT signal. A shot that IS a serve genuinely cannot
# be a volley, so that structural case keeps its high confidence.
# Height decides it outright when the reconstruction covers the interval: "did the ball
# bounce" IS "did z reach the ground", and that is the question the pixel scan was guessing at.
# 0.75 is provisional -- it should be re-measured against operator volley truth once there is
# more of it than the one clip, exactly as the two below were.
VOL_CONF_HEIGHT = 0.75
VOL_CONF_SCAN = 0.55     # was 0.85 -- local trajectory scan concluded (no longer reached)
VOL_CONF_FALLBACK = 0.40  # was 0.50 -- fell back to the bounce list (no longer reached)
# No height for this interval: assume NOT a volley, at the base rate's own confidence.
# 17 of 98 shots are volleys, so this is right ~83% of the time.
VOL_CONF_PRIOR = 0.60
VOL_CONF_STRUCTURAL = 0.9  # serve / first shot of a rally: cannot be a volley
POST_TRAJ_FRAMES = 15
MAX_ARC_FRAMES = 45          # cap the arc-measurement window (bounds dead-time gaps)
REFERENCE_FPS = 30.0         # frame-count windows tuned at 30fps; scale by fps/this for 60fps real footage
REFERENCE_WIDTH_PX = 1920.0  # px thresholds tuned at 1920-wide; scale by frame_width/this (4K = 2x)
VOLLEY_REBOUND_MIN_PX = 20.0  # min upward rebound after the low point to call a ground bounce (volley test)
VOLLEY_DESCENT_MIN_PX = 14.0  # min descent into the low point (ball clearly came down)
SIDE_CONF_FLOOR = 0.35
KITCHEN_MAX_DIST_FT = 9.0   # effective kitchen depth from net (court_zones)
# "At the net" for the no-landing path, measured to the FRONT FOOT. Wider than the kitchen
# itself: the operator's dinks reach 8.7 ft at p75 and their drives start well behind, and
# scored over the no-landing shots this value separates them best.
NET_ZONE_MAX_FT = 11.0
BASELINE_MIN_DIST_FT = 17.0  # within ~5ft of the 22ft baseline
BOUNCE_MIN_TURN_DEG = 40.0   # single-frame turn between shots => ground bounce
LANDMARK_VIS_FLOOR = 0.5
NET_Y_FT = 22.0

# A RESET is not a type. Operator, 2026-08-26: "All resets are either drops or dinks and
# should be labeled as drops and dinks. In addition, can count drops and dinks as resets as
# well IF the previous shot was a drive. So resets don't add to overall shot total but are a
# qualifier on some of the drops and dinks." So `is_reset` is DERIVED below, and the shot
# total stays serve + return + drive + drop + dink + lob.
SHOT_TYPES = {"serve", "return", "drive", "dink", "drop", "lob", "unknown"}
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
    c2i = homo.get("court_to_image")
    img2court = np.array(i2c, dtype=float) if i2c is not None else None
    court2img = np.array(c2i, dtype=float) if c2i is not None else None
    return {"ppf_near": near, "ppf_far": far, "fps": video.get("fps"),
            "image_to_court": img2court, "court_to_image": court2img}


def project_court_y(court: dict, px: float, py: float) -> Optional[float]:
    """Project an image point to its court_y (feet). Ground-plane homography."""
    M = court.get("image_to_court")
    if M is None:
        return None
    v = M @ np.array([px, py, 1.0])
    if abs(v[2]) < 1e-9:
        return None
    return float(v[1] / v[2])


def contact_point_frontness(court: dict, pose: Optional[dict], hand: Optional[str],
                            body_court_y: float) -> Optional[float]:
    """Technique metric: is the paddle-hand wrist AHEAD of the body in the net-ward
    direction at contact ("in front", the mark of a clean strike) or behind it
    ("late"/jammed)? Computed in the IMAGE (no airborne projection): the homography
    only supplies the net-ward direction. Normalised by shoulder width so it is
    scale-invariant across near/far players. + = in front, - = late. Interpretation
    is per-shot-type (a drive should be well in front; a dink is naturally compact).
    Returns None when the pose/homography is missing."""
    c2i = court.get("court_to_image")
    if pose is None or c2i is None or hand not in ("left", "right"):
        return None
    VIS = 0.3
    wx = pose.get("rwx" if hand == "right" else "lwx")
    wy = pose.get("rwy" if hand == "right" else "lwy")
    wv = pose.get("rwv" if hand == "right" else "lwv")
    if wx is None or wy is None or (wv is not None and wv < VIS):
        return None
    lsx, rsx = pose.get("lsx"), pose.get("rsx")
    lhx, rhx = pose.get("lhx"), pose.get("rhx")
    lhy, rhy = pose.get("lhy"), pose.get("rhy")
    if None in (lsx, rsx, lhx, rhx, lhy, rhy):
        return None
    body_x = 0.5 * (float(lhx) + float(rhx))
    body_y = 0.5 * (float(lhy) + float(rhy))
    shoulder_w = abs(float(rsx) - float(lsx))
    if shoulder_w < EPS:
        return None
    # net-ward image direction at the player: a near player (court_y < net) hits
    # toward +court_y, a far player toward -court_y. Step 1 ft that way, project.
    sign = 1.0 if body_court_y < NET_Y_FT else -1.0

    def _proj(cx, cy):
        v = c2i @ np.array([cx, cy, 1.0])
        return np.array([v[0] / v[2], v[1] / v[2]]) if abs(v[2]) > EPS else None
    p0, p1 = _proj(10.0, body_court_y), _proj(10.0, body_court_y + sign)
    if p0 is None or p1 is None:
        return None
    netward = p1 - p0
    nn = np.linalg.norm(netward)
    if nn < EPS:
        return None
    netward /= nn
    wrist_off = np.array([float(wx) - body_x, float(wy) - body_y])
    return float(wrist_off @ netward / shoulder_w)


def knee_bend_angle(pose: Optional[dict]) -> Optional[float]:
    """Technique metric: athletic stance = how bent the knees are at contact. Returns
    the mean knee angle in degrees (hip-knee-ankle) over both legs with a visible
    chain: ~180 deg = straight/standing tall (a lower-level tell), smaller = loaded
    and ready to move. Pure angle in the image plane (no projection needed). None if
    no leg chain is visible."""
    if pose is None:
        return None
    VIS = 0.3
    angles = []
    for h, k, a in (("l", "l", "l"), ("r", "r", "r")):
        hip = (pose.get(f"{h}hx"), pose.get(f"{h}hy"), pose.get(f"{h}hv"))
        knee = (pose.get(f"{k}kx"), pose.get(f"{k}ky"), pose.get(f"{k}kv"))
        ank = (pose.get(f"{a}ax"), pose.get(f"{a}ay"), pose.get(f"{a}av"))
        if any(p[0] is None or p[1] is None or (p[2] is not None and p[2] < VIS)
               for p in (hip, knee, ank)):
            continue
        kh = np.array([hip[0] - knee[0], hip[1] - knee[1]], dtype=float)
        ka = np.array([ank[0] - knee[0], ank[1] - knee[1]], dtype=float)
        nn = np.linalg.norm(kh) * np.linalg.norm(ka)
        if nn < EPS:
            continue
        angles.append(float(np.degrees(np.arccos(np.clip(kh @ ka / nn, -1.0, 1.0)))))
    return round(sum(angles) / len(angles), 1) if angles else None


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
            "left_hip_x_px", "left_hip_y_px", "left_hip_visibility",
            "right_hip_x_px", "right_hip_y_px", "right_hip_visibility",
            "left_wrist_x_px", "left_wrist_y_px", "left_wrist_visibility",
            "right_wrist_x_px", "right_wrist_y_px", "right_wrist_visibility",
            "left_knee_x_px", "left_knee_y_px", "left_knee_visibility",
            "right_knee_x_px", "right_knee_y_px", "right_knee_visibility",
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
            "lhx": r.left_hip_x_px, "lhy": r.left_hip_y_px, "lhv": r.left_hip_visibility,
            "rhx": r.right_hip_x_px, "rhy": r.right_hip_y_px, "rhv": r.right_hip_visibility,
            "lwx": r.left_wrist_x_px, "lwy": r.left_wrist_y_px, "lwv": r.left_wrist_visibility,
            "rwx": r.right_wrist_x_px, "rwy": r.right_wrist_y_px, "rwv": r.right_wrist_visibility,
            "lkx": r.left_knee_x_px, "lky": r.left_knee_y_px, "lkv": r.left_knee_visibility,
            "rkx": r.right_knee_x_px, "rky": r.right_knee_y_px, "rkv": r.right_knee_visibility,
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


RESET_TYPES = ("drop", "dink")
# How far back the shot being answered may sit. Contacts in a rally are 0.5-2s apart; beyond
# this the previous shot belongs to a different exchange, and after a point ends the dead
# time is far longer than this.
RESET_MAX_GAP_S = 3.0


# A shot flagged as the SERVE whose ball demonstrably came over the net just before it is a
# RETURN: the serve happened ~1s earlier on the other side and we did not detect the contact.
# This is a physical statement, not a threshold, and it behaves like one -- measured over the
# 22 shots we call serves that the operator also labelled, the ball came from the other side
# for 3 of the 5 they call returns and 0 of the 16 they call serves. Partial recall, but it
# never misfires on a real serve, which is what a retype rule needs.
#
# The pre-contact ball SPAN was tried here first and is not safe: real serves span 5-52 ft and
# returns 27-106 ft, so retyping on span would fix 5 and break 6.
RETURN_LOOK_S = (1.0, 1.4)


def came_from_other_side(court_y_by_frame, frame: int, side: str, fps: float,
                         net_y_ft: float) -> bool:
    """Did the ball start beyond the net and end on the hitter's side, just before contact?"""
    if not court_y_by_frame or side not in ("near", "far"):
        return False
    for look in RETURN_LOOK_S:
        lo = int(frame - look * fps)
        ys = [court_y_by_frame[f] for f in range(lo, int(frame - 0.05 * fps))
              if f in court_y_by_frame]
        if len(ys) < 5:
            continue
        near = [y < net_y_ft for y in ys]
        k = max(2, len(near) // 3)
        head = sum(near[:k]) / k
        tail = sum(near[-k:]) / k
        if side == "near" and head < 0.35 and tail > 0.65:
            return True
        if side == "far" and head > 0.65 and tail < 0.35:
            return True
    return False


def behind_baseline_counts(players, roles_by_tid, frame: int,
                           court_len_ft: float, window: int = 3):
    """(behind the NEAR baseline, behind the FAR baseline) around `frame`.

    `players` is the (frame, track_id) -> {court_y, ...} index. Only tracks with a ROLE are
    counted: the tracker also produces noise tracks, and a stray detection behind a baseline
    would invent a server out of nothing.
    """
    seen = {}
    for (f, tid), p in players.items():
        if not (frame - window <= f <= frame + window):
            continue
        if roles_by_tid.get(int(tid)) in (None, "noise"):
            continue
        y = p.get("court_y")
        if y is not None and y == y:
            seen.setdefault(int(tid), []).append(float(y))
    near = far = 0
    for tid, ys in seen.items():
        m = sum(ys) / len(ys)
        if m < 0.0:
            near += 1
        elif m > court_len_ft:
            far += 1
    return near, far


def serve_side_from_formation(players, roles_by_tid, frame: int, court_len_ft: float):
    """Which side is SERVING, from where the players stand. None when it cannot tell.

    Operator, 2026-08-26: "not only are 2 players behind the baseline as well as an opposing
    player, but the ball should be seen and hit by the side with the 2 players behind the
    baseline. The return would always occur after the serve plus having the person hitting
    the ball be on the side with only 1 person behind the baseline."

    The COUNT alone cannot separate a serve from its return -- 1.2s later nobody has moved,
    and the formation reads the same. The ASYMMETRY can: server and partner are both back,
    the receiver is back alone. Measured against their labelled serves and returns on two
    clips, with ZERO crossovers:

        hitter's side has MORE behind    19 serves,  0 returns
        hitter's side has FEWER behind    0 serves, 20 returns
        equal                             5 serves,  3 returns

    So "more" means serve, "fewer" means return, and equal means we do not know.
    """
    near, far = behind_baseline_counts(players, roles_by_tid, frame, court_len_ft)
    if near > far:
        return "near"
    if far > near:
        return "far"
    return None


def retype_returns(shots: list, court_y_by_frame, fps: float, net_y_ft: float,
                  players=None, roles_by_tid=None,
                  court_len_ft: float = 44.0) -> int:
    """Relabel a "serve" that is really the RETURN.

    Two independent tests, tried in order of how well they measured:

    1. WHERE THE PLAYERS STAND. The serving side has two players behind its baseline, the
       receiving side one. Zero crossovers over the operator's labelled serves and returns
       (see serve_side_from_formation) -- a shot struck from the side with FEWER players back
       is not a serve.
    2. WHERE THE BALL CAME FROM. If it came over the net just before the contact, the shot
       answers something. 3 of 5 returns, 0 of 16 serves -- precise but partial, so it is the
       fallback for when the formation is symmetric and says nothing.

    `is_serve` is deliberately LEFT ALONE. It is what Stage 7 segments rallies on, and a new
    point really does begin around here -- the serve we missed is a second earlier. Clearing
    it would merge this point into the previous one, which is worse than a wrong type. Stage
    7's `opened_on_return` already records that the SERVER is the other side.
    """
    n = 0
    for s in shots:
        if s.get("shot_type") != "serve":
            continue
        side = s.get("hitter_side")
        verdict = None
        if players is not None and side in ("near", "far"):
            serving = serve_side_from_formation(players, roles_by_tid or {},
                                                int(s["frame"]), court_len_ft)
            if serving is not None:
                verdict = (serving != side)      # struck from the side that is NOT serving
        if verdict is None:
            verdict = came_from_other_side(court_y_by_frame, int(s["frame"]),
                                           side, fps, net_y_ft)
        if verdict:
            s["shot_type"] = "return"
            s["retyped_from_serve"] = True
            n += 1
    return n


def mark_resets(shots: list, fps: float) -> int:
    """A drop or a dink that ANSWERS A DRIVE is also a reset.

    Operator, 2026-08-26: "All resets are either drops or dinks and should be labeled as
    drops and dinks. In addition, can count drops and dinks as resets as well IF the previous
    shot was a drive. So resets don't add to overall shot total but are a qualifier on some
    of the drops and dinks."

    So `is_reset` is a QUALIFIER, never a type: every reset is already counted once as its
    drop or dink, and the shot total is unchanged. The previous shot must be the OPPONENT'S
    -- a reset answers the other side -- and recent enough to be the same exchange, so a
    drive that ended the last point is not something the next drop is resetting.
    """
    n = 0
    ss = sorted(shots, key=lambda s: int(s["frame"]))
    for i, s in enumerate(ss):
        s["is_reset"] = False
        if s.get("shot_type") not in RESET_TYPES or i == 0:
            continue
        prev = ss[i - 1]
        if (prev.get("shot_type") == "drive"
                and (int(s["frame"]) - int(prev["frame"])) / float(fps) <= RESET_MAX_GAP_S
                and prev.get("hitter_side") and s.get("hitter_side")
                and prev["hitter_side"] != s["hitter_side"]):
            s["is_reset"] = True
            n += 1
    return n


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
    drop/dink lands within the kitchen (+~2 ft).

    INFERRING the missing ones was tried and REVERTED. Where a landing cannot be detected
    because the reply lands on the same frame -- the largest single cause, and a physical
    limit at 60fps rather than a detector fault -- the landing can be estimated from where
    the replying player stood, corrected for the ~2.5 ft they stand behind it. It moves the
    right numbers and still does not pay:

        drop recall   35% -> 48%      dink recall  65% -> 92%
        drive recall  67% -> 41%      shot_type_correct 114 -> 109 over the harness clips

    The failure is a bias, not noise. Replies are often taken near the kitchen, so inferred
    depths cluster shallow and the classifier reads dink: dinks rose on EVERY clip
    (+14, +6, +7, +10) against an operator count that did not move. Two thirds of a drive's
    landings are inferred, so drive pays for all of it.

    Underneath that is a definition mismatch worth knowing before trying again. The
    operator's "drive" is about PACE -- a hard flat ball -- while landing depth measures
    DEPTH, and a drive taken early off the bounce lands short. Where the landing is really
    measured the two still separate (drop 7.2 ft, drive 13.1 ft); it is the inferred
    population that does not, because it is exactly the shots taken early.
    """
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


def bounced_between_3d(z_by_frame, f0: int, f1: int):
    """Did the ball touch the GROUND between two shots, from the 3-D reconstruction?

    True = it bounced (so the second shot is not a volley), False = it stayed up (a volley),
    None = the reconstruction does not cover the interval well enough to say, in which case
    the caller falls back to the pixel scan.

    This replaces a guess. The pixel test looks for a descend-then-rebound in image space,
    and bounce-vs-volley is recorded in docs/ACCURACY_LEDGER.md as the monocular precision
    floor with three height-free signals tried and defeated. Measured against operator volley
    truth the pixel scan scored 5/10, and the volley RATE it produces is 32-42% of shots on
    four venues against a truth of 17%. Height answers the question directly.

    The absolute threshold is used only to separate "near the ground" from "clearly not";
    the rebound is what carries the decision, because `bias` scales absolute z and a relative
    move does not inherit that error.

    KNOWN WEAKNESS, measured and not yet fixed. `zmin` is the minimum of 30-60 reconstructed
    samples, so one bad sample decides it, and 12-13% of reconstructed frames are exactly
    0.0 -- k is clipped at 1.0 in build_ball_3d, so z = H(1 - 1/k) collapses to zero
    whenever the size measurement falls out of range. That is a failed measurement, not a
    ball on the ground, and a single one turns a volley into a ground shot. It is the
    largest single cause of the missed volleys: 13 of the 20 across the two reviewed clips.

    Requiring the ball to HOLD low for consecutive frames was tried and REVERTED. On its
    own it changes nothing, because the caller reads None and True alike as "not a volley",
    so the only way to gain is to answer False more often. Doing that for the band where
    the ball dips below VOLLEY_AIRBORNE_FT but never holds at the ground is tempting -- that
    band is 64% volleys against the "not a volley" the caller assumes, so the prior really
    is backwards there -- but it does not pay:

        volley accuracy on labelled shots   93/113 -> 95/113   (+2)
        shot_type_correct                   107    -> 103      (-4)
        volleys flagged on outdoor-7        22     -> 42       (operator counts 27)

    Two reasons it fails. The volley TYPE path is the weaker one (56% vs 71% on court C),
    so every shot moved into it costs type accuracy; and flipping the whole band overshoots
    the operator's own volley total by more than the old under-call missed it. identity_gap
    improves hugely (27 -> 7 on outdoor-7) but that is mechanical -- the identity is
    shots - (volleys + bounces), so adding volleys closes it by construction and proves
    nothing.

    The way in is a discriminator WITHIN that 25-shot band, not a blanket flip.
    """
    if not z_by_frame or f1 - f0 < 2:
        return None
    seq = [z_by_frame[g] for g in range(f0 + 1, f1) if g in z_by_frame]
    if len(seq) < VOLLEY_MIN_FRAMES:
        return None
    k = min(range(len(seq)), key=lambda j: seq[j])
    zmin = seq[k]
    rose = max(seq[k + 1:], default=None)
    if zmin <= VOLLEY_GROUND_FT:
        if rose is None:
            return None                      # low at the very end -- cannot see a rebound
        if (rose - zmin) >= VOLLEY_MIN_REBOUND_FT:
            return True                      # down to the ground and back up: a bounce
        return None                          # low but never recovered -- ambiguous
    if zmin >= VOLLEY_AIRBORNE_FT:
        return False                         # never came near the ground: a volley
    return None


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
                  is_return=False,
                  landing_y=None, receiver_zone=None, is_volley=False,
                  drive_min=DRIVE_MIN_SPEED_FTPS, dink_max=DINK_MAX_SPEED_FTPS,
                  contact_dist_from_net=None):
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
    # RETURN OF SERVE — structural, not a signal we have to detect. The return is
    # the shot that answers the serve: the next contact, from the OTHER side of the
    # net. The operator counts returns as their own category (the ledger's identity
    # is 14 serves = 14 rallies = 14 returns), and Stage 6 previously had no such
    # type, so every return was scored as a `drive` -- 4 of 4 wrong on the operator's
    # labelled set, purely a taxonomy gap rather than a classification failure.
    if is_return:
        return "return", 0.9
    # --- Volley (ball taken out of the air; no bounce, so no landing signal).
    #     Operator rules: classify from the hitter's zone + speed. At the kitchen a
    #     slow air-ball is a dink, a fast one a speed-up drive; taken out of the air
    #     from transition/baseline it's a drive. (A lob is judged on the PRIOR shot
    #     via the receiver running back — not modeled here yet.)
    if is_volley:
        if zone == "kitchen":
            if post_ftps is not None and post_ftps >= drive_min:
                return "drive", FB_DRIVE
            return "dink", FB_DINK
        return "drive", FB_DRIVE
    # A lob is a high lofted arc AND slow AND goes over a receiver AT THE KITCHEN
    # (a lob only makes sense against a player at the net; a soft high ball to
    # deep opponents is a drop/drive, not a lob). Resolve before the landing split.
    if (arc_frac is not None and arc_frac >= LOB_MIN_ARC_FRAC
            and (post_ftps is None or post_ftps < drive_min)
            and receiver_zone == "kitchen"):
        return "lob", min(1.0, max(0.6, arc_frac))

    # --- Landing-aware path: the SOUND signal (bounces project reliably) ---------
    # OPERATOR TYPE TABLE (2026-08-03). Type is a function of WHERE THE HITTER WAS
    # x WHERE THE BALL LANDED (x pace), not of landing depth alone:
    #   "from the kitchen area: hit softly to the baseline = LOB; hard to the
    #    baseline or transition zone = DRIVE; into the kitchen area, or softly into
    #    the transition zone, = DINK."
    # The previous rule only tested landing-distance-from-net and so never combined
    # the two zones — which is why a soft ball lofted from the kitchen to the
    # baseline came out a DRIVE (both operator-labelled lobs were scored drives).
    # LOFTED vs FLAT is judged on ARC SHAPE first: arc is a trajectory property,
    # while ball speed is the signal this project has repeatedly found unreliable
    # (ACCURACY_LEDGER: "a weak discriminator by physics, not a fixable bug").
    # Speed is only the fallback when no arc is measurable.
    if landing_y is not None:
        land_zone = zone_from_court_y(landing_y)
        if arc_frac is not None:
            lofted = arc_frac >= DRIVE_DROP_ARC_SPLIT
            high_lob = arc_frac >= LOB_MIN_ARC_FRAC
        else:
            lofted = high_lob = (post_ftps is not None and post_ftps <= dink_max)
        if zone == "baseline":
            # from DEEP: a soft ball landing at the net is the third-shot drop.
            return ("drop", 0.78) if land_zone == "kitchen" else ("drive", 0.78)
        # hitter at/near the net (kitchen or transition)
        if land_zone == "kitchen":
            return "dink", 0.78
        if land_zone == "transition":
            return ("dink", 0.7) if lofted else ("drive", 0.78)
        # lands deep at the baseline: lofted from the net = LOB, flat = DRIVE
        return ("lob", 0.7) if high_lob else ("drive", 0.78)
        # NOTE: the old "speed guard" (a slow ball with a DEEP landing called a dink
        # anyway) was removed in v0.5.0 and is not reinstated here. It contradicted
        # the operator's ruling that the LANDING decides type, and rested on ball
        # speed. The table above expresses the same intent correctly: a soft ball
        # that lands deep is a LOB (from the net) or a DRIVE, never a dink.

    # --- Fallback (no landing): arc + speed, lower confidence --------------------
    # POSITION, THEN ARC, THEN SPEED -- and always an answer.
    #
    # This path types 19 of the operator's 23 drops (the landing branch sees only 4), and
    # it used to ask speed first and alone: `if speed >= drive_min: drive`. That sent every
    # hard-struck drop to "drive" -- they reach 43 ft/s against 41 for real drives, so
    # speed cannot separate them -- and never asked where the hitter stood, which is the
    # criterion the operator names for dink against drop and the strongest signal we have
    # (front foot 7.6 ft from the net for dinks against 20.0 for drops, from
    # players.parquet, which is ground truth rather than reconstruction).
    #
    # The result was a nearly degenerate classifier: on the no-landing shots it got 25 of
    # 30 drives and 1 of 8 dinks, 5 of 18 drops. Asking position first gives 18/30, 5/8,
    # 9/18 -- one more correct overall, and a usable answer for the two categories that
    # were being thrown away.
    #
    # A drive must be fast AND FLAT, which is the operator's definition. Where the arc
    # cannot be measured speed alone stands in, because every branch here must have an
    # exit: requiring both without that let a fast arced ball fall past everything onto
    # "unknown" -- 20 of them, and the score went from 97/142 to 85/142.
    # Speed still decides DRIVE on its own here. Requiring it to be flat as well -- the
    # operator's definition, and correct in principle -- was measured twice and costs more
    # than it returns: drive recall falls from 71% to 47% at every net-zone width tried,
    # because an arced hard ball then leaves this branch and is typed by position instead.
    # What position fixes is everything AFTER that test, which used to fall through a
    # speed-banded chain and out onto "unknown".
    _fast = post_ftps is not None and post_ftps >= drive_min
    _at_net = contact_dist_from_net is not None and contact_dist_from_net <= NET_ZONE_MAX_FT
    if _fast:
        return "drive", FB_DRIVE
    if _at_net:
        return "dink", FB_DINK
    if contact_dist_from_net is not None:
        return "drop", FB_DROP
    # (The old "reset" branch lived here: fast ball in, slow ball out, not from the
    # baseline. That is a real pattern, but it is a DROP or a DINK -- which one depends on
    # where it was struck, exactly as the rules below already decide. Whether it is also a
    # reset is answered by the previous shot, not by ball speed, so it is derived after
    # every type is known rather than competing with them here.)
    if post_ftps is not None and post_ftps <= dink_max and zone in ("kitchen", "transition"):
        return "dink", FB_DINK   # slow ball hit from at/near the net = dink
    if post_ftps is not None and post_ftps <= dink_max and zone == "baseline":
        return "drop", FB_DROP   # slow ball from deep = drop (third-shot drop)
    # Tweener (dink_max..drive_min) with no landing: speed is ambiguous, so resolve
    # by trajectory SHAPE -- flat => drive, lofted => drop.
    if post_ftps is not None and dink_max < post_ftps < drive_min:
        if arc_frac is not None and arc_frac >= DRIVE_DROP_ARC_SPLIT:
            return "drop", FB_TWEENER
        return "drive", FB_TWEENER
    return "unknown", FB_UNKNOWN


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
    # track_id -> Stage 2.5 role, so every player's handedness (the app collects it
    # for all four) can drive forehand/backhand -- not just the user's.
    roles_by_tid: Dict[int, str] = {}
    _tr = folder / "track_roles.json"
    if _tr.exists():
        for t, info in (load_json(_tr).get("track_roles", {}) or {}).items():
            try:
                roles_by_tid[int(t)] = info.get("role")
            except (TypeError, ValueError):
                continue
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
    # Ball height per frame, when tools/build_ball_3d.py has run. Optional input: without it
    # the volley test falls back to the pixel scan exactly as before.
    z_by_frame: Dict[int, float] = {}
    court_y_by_frame: Dict[int, float] = {}
    b3p = folder / "ball_3d.parquet"
    if b3p.exists():
        _b3full = pd.read_parquet(b3p, columns=["frame", "z_ft", "court_y_ft"])
        court_y_by_frame = {int(f): float(y) for f, y in
                            zip(_b3full["frame"], _b3full["court_y_ft"]) if y == y}
        b3 = pd.read_parquet(b3p, columns=["frame", "z_ft"])
        z_by_frame = {int(f): float(z) for f, z in zip(b3["frame"], b3["z_ft"])
                      if z == z}
        log.info(f"ball height available for {len(z_by_frame)} frames; "
                 f"volley decided by whether the ball reached the ground")

    traj_index: Dict[int, dict] = {}
    traj_path = folder / "trajectory.json"
    if traj_path.exists():
        for t in load_json(traj_path).get("shots", []):
            traj_index[int(t["shot_id"])] = t
        log.info(f"loaded Stage 5.7 trajectory speeds for {len(traj_index)} shots")

    shots = sorted(shots_doc.get("shots", []), key=lambda s: s["frame"])
    # Which serve-flagged shots are really RETURNS. Decided before typing so the shot that
    # FOLLOWS one is not typed "return" as well -- see is_return below.
    _court_len = float((court.get("court_geometry_feet") or {}).get("length_ft") or 44.0)
    really_returns = set()
    for _s in shots:
        if not _s.get("is_serve"):
            continue
        _side = _s.get("hitter_side")
        _verdict = None
        if _side in ("near", "far"):
            _serving = serve_side_from_formation(players, roles_by_tid,
                                                 int(_s["frame"]), _court_len)
            if _serving is not None:
                _verdict = (_serving != _side)
        if _verdict is None:
            _verdict = came_from_other_side(court_y_by_frame, int(_s["frame"]),
                                            _side, float(fps), NET_Y_FT)
        if _verdict:
            really_returns.add(int(_s.get("shot_id", -1)))
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
        # this player's handedness (user from roster; others via role) -- used by
        # both the stroke side and the contact-point technique metric.
        hand = user_hand if is_user else roster.get(roles_by_tid.get(tid) or "")
        # technique: contact point front-vs-late (paddle wrist net-ward of body).
        contact_front = contact_point_frontness(court, pose, hand, court_y)
        knee_deg = knee_bend_angle(pose)   # athletic stance: lower = more bent

        post_ftps = speed_ftps(s.get("speed_post_px_per_frame"), court, court_y, fps)
        pre_ftps = speed_ftps(s.get("speed_pre_px_per_frame"), court, court_y, fps)
        arc_frac = arc_height_frac(bx, by, bknown, f, arc_end)
        contact_h = contact_height(float(impact_y), pose)

        # landing court_y from the first bounce after this shot (sound signal)
        landing_y = landing_index.get(int(s["shot_id"]))
        # PHYSICAL CONSTRAINT (2026-08-02): a shot must land on the OPPOSITE side of
        # the net. A same-side landing is only physical for a NET HIT -- the ball
        # fails to cross and dies near the net (the operator counted 8 net hits in
        # this clip), so those are KEPT. Anything else same-side is a
        # mis-association: measured 16 of 70 landings (23%) were same-side, of which
        # 4 landed BEHIND a baseline and 6 within ~2 ft of the hitter's own feet --
        # ball-handling bounces, which cannot be the landing of the shot just struck.
        # Dropping the landing sends those shots to the fallback path rather than
        # typing them from a landing that belongs to a different event.
        if landing_y is not None:
            _hy = (s.get("hitter_court_xy_ft") or [None, None])[1]
            if _hy is not None and (_hy >= NET_Y_FT) == (landing_y >= NET_Y_FT)                     and abs(landing_y - NET_Y_FT) > KITCHEN_MAX_DIST_FT:
                landing_y = None
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
            is_volley, vol_conf = False, VOL_CONF_STRUCTURAL
        else:
            # Height first when the reconstruction covers the gap -- it answers the actual
            # question rather than inferring it from image-space motion. Then the pixel
            # scan, then the precision bounce list.
            height = bounced_between_3d(z_by_frame, prev_frame, f)
            if height is not None:
                is_volley = not height
                vol_conf = VOL_CONF_HEIGHT
            else:
                # Height could not see the interval. Fall back to the PRIOR, not to the
                # pixel scan: the operator's counts put volleys at 17 of 98 shots, so
                # "not a volley" is right 83% of the time, while the pixel scan measured
                # 5/10 against the same truth. Guessing with a coin when a loaded die is
                # available is strictly worse, and the scan's errors are not random -- it
                # over-calls volleys, which is exactly the 32-42% vs 17% discrepancy.
                is_volley = False
                vol_conf = VOL_CONF_PRIOR

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
        elif (tj is not None and not traj_phantom
              and tj.get("anchor_type") == "net_crossing"
              and tj.get("horizontal_speed_ftps") is not None):
            # The net-crossing speed, accepted BELOW the confidence floor because the
            # alternative is the pixel speed, which is inverted by camera distance and
            # actively wrong. It measures the approach to the net rather than the whole
            # flight, so it reads high on the same shots -- drives 57.8 ft/s against the
            # anchored 37.0 -- and is rescaled here so one set of thresholds serves both.
            # Worth it for coverage: 99% of shots against 63%, and a wider drive-vs-rest
            # margin (1.86x against 1.48x).
            speed_for_type = tj["horizontal_speed_ftps"] * NET_CROSS_SPEED_SCALE
            d_min, d_max = DRIVE_MIN_SPEED_HORIZ_FTPS, DINK_MAX_SPEED_HORIZ_FTPS
            speed_source = "net_crossing"
        else:
            speed_for_type = post_ftps
            d_min, d_max = DRIVE_MIN_SPEED_FTPS, DINK_MAX_SPEED_FTPS
            speed_source = "ppf_instantaneous"

        # a return answers the serve: previous shot was the serve AND this contact is
        # on the opposite side of the net (the receiving team plays it back).
        #
        # ...unless that "serve" is itself a return we mislabelled, which happens whenever
        # Stage 5 missed the real serve. Then the shot after it is the THIRD shot, not a
        # second return -- and calling it a return put 8 "returns" into the third-shot count
        # on the acceptance clip, where a third shot can never be one.
        prev = shots[i - 1] if i > 0 else None
        is_return = bool(prev is not None and prev.get("is_serve")
                         and int(prev.get("shot_id", -1)) not in really_returns
                         and prev.get("hitter_side") and s.get("hitter_side")
                         and prev["hitter_side"] != s["hitter_side"])
        # Distance from the net to the FRONT FOOT -- the operator's own reference ("kitchen
        # out to 2 feet beyond (front leg)"), and the tighter of the two: dinks sit at 8.7
        # ft at p75 by the front foot against 11.3 by the body centre. Falls back to the
        # body when no pose is available.
        _ff = (s.get("features") or {}).get("contact_front_foot_y")
        if _ff is None:
            _ff = locals().get("front_foot_y")
        _hxy = s.get("hitter_court_xy_ft") or [None, None]
        _cy = _ff if _ff is not None else (_hxy[1] if _hxy[1] is not None else None)
        _contact_dist_net = (abs(float(_cy) - _court_len / 2.0)
                             if _cy is not None else None)

        shot_type, type_conf = classify_type(is_serve, arc_frac, contact_h,
                                             speed_for_type, pre_ftps, zone,
                                             is_return, landing_y,
                                             receiver_zone, is_volley,
                                             drive_min=d_min, dink_max=d_max,
                                             contact_dist_from_net=_contact_dist_net)

        # stroke side: forehand/backhand for the user (handedness known); an
        # above-the-head contact is an 'overhead' stroke regardless of handedness.
        # Forehand/backhand for EVERY player: the roster carries handedness for
        # user/partner/opp_a/opp_b and stroke_side() derives facing from the pose,
        # so restricting this to the user threw away data we already collect (it
        # left 74 of 108 shots' stroke side "unknown").
        side, side_conf = stroke_side(float(impact_x), pose, hand)
        # Operator model: EVERY shot is a forehand or a backhand (serves are
        # forehands). An overhead is still struck with a FH or BH grip, so it is
        # NOT its own stroke category -- record it as a separate flag and keep the
        # FH/BH so the stroke counts satisfy the identity FH + BH = total shots.
        is_overhead = (contact_h == "high")
        # Operator: serves are forehands. A serve struck low often reads "unknown"
        # from shoulder geometry, so default a serve's stroke to forehand.
        if is_serve and side == "unknown" and hand in ("left", "right"):
            side, side_conf = "forehand", max(side_conf, 0.6)

        out = dict(s)  # carry through all Stage 5 fields
        out.update({
            "stroke_side": side,
            "stroke_side_confidence": side_conf,
            "is_overhead": bool(is_overhead),
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
                # technique: paddle-wrist net-ward of body at contact (+ in front,
                # - late), normalised by shoulder width. Interpreted per shot type.
                "contact_front": round(contact_front, 3) if contact_front is not None else None,
                # technique: mean knee angle (hip-knee-ankle) at contact; ~180 =
                # standing tall, lower = athletic/loaded.
                "knee_angle_deg": knee_deg,
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
    n_retyped = retype_returns(out_shots, court_y_by_frame, float(fps), NET_Y_FT,
                               players=players,
                               roles_by_tid=roles_by_tid,
                               court_len_ft=float(court["court_geometry_feet"]["length_ft"])
                               if court.get("court_geometry_feet") else 44.0)
    n_reset = mark_resets(out_shots, float(fps))

    from collections import Counter
    by_type = Counter(s["shot_type"] for s in out_shots)
    by_side = Counter(s["stroke_side"] for s in out_shots)
    stats = {
        "n_shots": len(out_shots),
        "by_shot_type": dict(by_type),
        "by_stroke_side": dict(by_side),
        "n_volley": sum(1 for s in out_shots if s["is_volley"]),
        "n_volley_fallback": sum(1 for s in out_shots if s["is_volley_confidence"] == 0.5),
        "n_volley_from_height": sum(1 for s in out_shots
                                    if s["is_volley_confidence"] == VOL_CONF_HEIGHT),
        # a QUALIFIER on drops and dinks, not a type -- these shots are already counted
        # once under by_shot_type, so this must never be added to the shot total
        "n_reset": n_reset,
        "n_retyped_serve_to_return": n_retyped,
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

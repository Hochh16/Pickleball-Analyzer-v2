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
STAGE_VERSION = "0.6.0"  # 0.5.1 -> 0.6.0: serve acceptance breaks ties on the
                         # RETURN. A feed lobbed back to the server looked like a
                         # serve, claimed the slot and BLOCKED the real serve behind
                         # it -- the false and missed serves were one bug. This also
                         # unblocked HANDLING_SPREAD_S (1.0s -> 8.0s).
                         # 0.5.0 -> 0.5.1: same-side runs keep the STRONGEST
                         # impact, not the last, when the run is tight -- the old
                         # rule deleted a 172-degree paddle reversal and kept a
                         # 26-degree wobble, which is what "wrong player" was.
                         # 0.4.0 -> 0.5.0: WRONG-OBJECT LATCH rejection. The
                         # single-ball tracker latches onto a neighbouring court's
                         # ball or a parked object beside the court; 10 of the 34
                         # operator-labelled false positives came from that, and
                         # the existing teleport-in gate had never fired once.
                         # 0.3.0 -> 0.4.0: GROUND-BALL rejection. Operator per-shot
                         # labels showed 30% of detected "shots" were balls rolled at
                         # the net, picked up, or bounced pre-serve. The ground
                         # homography is valid only at z=0, so an airborne struck ball
                         # projects far off court while a ground ball does not.
                         # 0.2.0: real-ball adaptations (see contract "Real-ball
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
# A same-side run spanning at least this long is genuine ball-handling (bounce, bounce,
# serve) where the real shot is LAST; anything tighter is a strike plus a tracking wobble,
# where the real shot is the STRONGEST. See reject_same_side_runs.
#
# Swept against BOTH scorers (tools.score_shots + tools.score_serves) on
# pb_5_minute_outdoor-7, AFTER the serve return tie-break made Stage 7 robust:
#
#   spread   false pos   wrong player   real kept   serve recall / prec
#     1.0s      29/34         4/7           91            86% / 75%
#     2.5s      27/34         2/7           90            86% / 86%
#     6-10s     23/34         1/7           94            86% / 86%    <- plateau, 8.0 chosen
#    15s+       22/34         1/7           94            71% / 77%
#
# Runs are already split at gaps over HANDLING_RESET_S (3s), so a run SPANNING 8s is a
# genuine repeated-bounce sequence before a serve, where the real shot is last. Everything
# tighter is a strike plus a tracking wobble, where the real shot is the strongest.
#
# This threshold could not be raised past 1.0s before the tie-break landed -- serve recall
# collapsed to 50%, because Stage 7 derives serves from the surviving shot sequence. Do not
# change it without re-running score_serves; the two stages are coupled through this value.
HANDLING_SPREAD_S = 8.0
# The handling filter rests on a statement about the PLAYER: you cannot legally hit twice in
# a row, so consecutive same-side impacts are one player handling the ball. That premise
# assumes detection is COMPLETE -- when a shot is missed, the two real shots either side of
# it become "consecutive same-side" and one of them is deleted too. Measured on the indoor
# clip, that accounted for 9 of the operator's 21 confirmed missed shots.
#
# The physically stronger statement is about the BALL: between two genuine shots it has to
# leave and come back, while during handling (a catch, a bounce, a carry) it stays near the
# hitter. Splitting a run wherever the ball made a real excursion is a plain 2-D distance in
# the image, so unlike the two fixes rejected before it (net-line crossing, opposing-player
# reach) it never has to decide which SIDE of the net the ball was on -- the question that
# needs a height one 6 ft camera cannot give.
#
# Swept through the FULL pipeline on all three scored clips (an offline sweep over the
# impulse shots alone disagrees, because serves are derived from the surviving shot sequence
# in structure_points and only the real pipeline shows that). Reference px @1920:
#
#   split at  | outdoor: real kept  false pos  wrong plr  serve | court B: in-play  junk  serve
#     off     |          95           22/34       1/7     93/81 |          59        11   70/88
#     800 px  |          95           22/34       1/7     93/81 |          59        11   70/88
#     700 px  |          96           22/34       1/7     86/80 |          61        10   80/89
#   * 600 px  |          97           22/34       1/7     86/80 |          61        10   80/89
#     500 px  |          98           22/34       2/7     86/80 |          61        11   80/89
#     400 px  |         101           22/34       3/7     79/73 |          62        11   80/89
#
# Held-out court C is flat at every threshold above 500 px (57 in-play, 12 junk, serve 80/80).
#
# 600 px is where the shot gain is exhausted before the costs start: below 500 px the
# wrong-player count climbs (recovered impacts start being credited to the wrong striker) and
# by 400 px serve accuracy collapses. Operator-labelled false positives never move at any
# threshold, on any clip -- the split returns shots the filter had over-deleted; it does not
# invent new ones.
#
# The one cost at 600 px is on the acceptance clip's serves: one true serve stops being
# detected (recall 93% -> 86%, which is the figure that stood before the tracker fix), traded
# against one serve GAINED on court B (70% -> 80%). Serves come out net level across the
# three clips; shots do not, and shots are what the operator counts.
SAME_SIDE_EXCURSION_PX = 600.0   # ref px @1920, scaled by frame_width/1920
# Adjacent-court contamination gates (real ball only). On a multi-court venue the
# single-ball detector grabs a NEIGHBORING court's ball when ours is occluded,
# producing phantom shots/serves. Two trajectory-coherence gates reject them:
RALLY_GAP_S = 6.0  # same-player repeats within this window = ball-handling in one point
MIN_SERVE_RUN_S = 0.13  # a real serve launches a SUSTAINED run; a blip serve
                        # (other-court ball appearing briefly) does not. (8f @60fps)
TELEPORT_IN_PX_PER_FRAME = 40.0  # ref px/frame @1920 (scaled by frame_width/1920):
                        # an impulse impact whose ball run TELEPORTED in (jumped
                        # from where our ball actually was) is the other court's ball.
# LATCH rejection (operator per-shot review of all 125 detected shots, 2026-08-18). The
# single-ball tracker sometimes locks onto a DIFFERENT object for a second or more -- a
# neighbouring court's ball, or a lawnmower parked beside the court -- and 10 of the 34
# labelled false positives come from exactly that. TELEPORT_IN_PX_PER_FRAME cannot see it:
# it measures the jump into the contact's run, but by the contact the tracker is settled on
# the wrong object and the step is small. The tell is the jump at the START of the latch.
# Threshold is physical: across 84 operator-confirmed real shots the ball never exceeds 101
# px/frame at 60fps/3840px, so 150 px/frame (75 ref @1920) leaves 50% headroom. Measured
# in-pipeline: 5 of the 10 removed for the loss of 1 real shot (net -4 errors).
# A 0.5s window loses no real shot but removes only 2, and 0.75s removes 4 and
# still costs the same 1 -- 1.0s is the best net error count of the three.
LATCH_JUMP_PX_PER_FRAME = 75.0   # ref px/frame @1920 (scaled by frame_width/1920)
LATCH_WINDOW_S = 1.0  # half-window scanned either side of the contact
SERVE_DEDUP_S = 2.0     # two serve detections this close with no rally shot between
                        # = a pre-serve artifact + the real serve; keep the longer run.
# Unified point-boundary detection (operator method 2026-07-27). A rally is
# SERVE -> ... -> POINT-END, one of each, alternating. No single rule carries it;
# combine weak cues + the structural one-each constraint + the hitter's robust
# GROUND depth (not the airborne/dead-time-contaminated ball).
SERVE_BEHIND_BASELINE_FT = 21.0  # hitter dist_from_net beyond this = behind the baseline
                                 # (baseline is 22 ft from the net; 1 ft tolerance for
                                 # position noise. A shot from IN FRONT of the baseline is
                                 # never a serve -- operator rule 2026-07-27.)
SERVE_OPEN_GAP_S = 3.0           # >= this gap before a deep shot = it opens a point
POINT_RETURN_S = 2.5             # opposite-side shot within this = the shot was RETURNED (rally continues)
POINT_DEAD_GAP_S = 3.0           # dead time after an un-returned shot = the point ENDED
# A second acceptance path for a serve: two real serves are separated by a whole point
# + retrieval, so a deep point-opener this long after the last accepted serve is a new
# serve even when its point-end went undetected (the mutual constraint alone over-rejects
# whenever a return-timing point-end is missed). Recovers far-side / missed-end serves.
POINT_MIN_INTER_SERVE_S = 10.0
# GROUND-BALL rejection (operator per-shot labels, 2026-08-03). 30% of detected
# "shots" were not shots at all -- balls rolled at the net, a ball picked up, a ball
# bounced before serving. They are separable on PHYSICS, not on rally structure: the
# ground homography is valid only at z=0, so an airborne struck ball projects far off
# the court while a ball ON the ground projects sensibly. Real shots measured <= 0.63
# grounded; the junk this catches starts at 0.76.
GROUNDED_WINDOW_S = 1.0
GROUNDED_MAX_FRAC = 0.70    # >= this share of samples on the ground => not a shot
GROUND_PLAUSIBLE_MARGIN_FT = 8.0
COURT_WIDTH_FT = 20.0
COURT_LENGTH_FT = 44.0
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


def grounded_fraction(M: np.ndarray, fx, fy, known, frame: int, half_window: int,
                      min_samples: int = 6) -> Optional[float]:
    """Fraction of nearby ball samples whose GROUND projection is plausible.

    The discriminator for "the ball is on the court surface, not in flight". The
    ground homography is only valid at z=0: a ball in the AIR projects far off the
    court (the ledger measured court_y 75-150 ft for airborne balls), while a ball
    ON the ground projects sensibly. So a high fraction means the ball was never
    really airborne through this event -- it was rolling, sitting, or being bounced.

    Measured against operator per-shot labels (2026-08-03, 21 real shots vs 9
    labelled not-a-shot): real shots reach at most **0.63**; the between-point balls
    it catches start at **0.76** -- a genuine gap, not a tuned edge. Catches balls
    ROLLED at the net, a ball being picked up, and pre-serve bouncing. It does NOT
    catch a ball THROWN to the server (that really is airborne) -- see KNOWN_ISSUES.

    Returns None when too few samples exist to judge (never reject on no evidence).
    """
    lo, hi = max(0, frame - half_window), min(len(fx) - 1, frame + half_window)
    n = ok = 0
    for i in range(lo, hi + 1):
        if not known[i]:
            continue
        n += 1
        cx, cy = project_to_court(M, float(fx[i]), float(fy[i]))
        if (math.isfinite(cx) and math.isfinite(cy)
                and -GROUND_PLAUSIBLE_MARGIN_FT <= cx <= COURT_WIDTH_FT + GROUND_PLAUSIBLE_MARGIN_FT
                and -GROUND_PLAUSIBLE_MARGIN_FT <= cy <= COURT_LENGTH_FT + GROUND_PLAUSIBLE_MARGIN_FT):
            ok += 1
    return (ok / n) if n >= min_samples else None



def max_latch_jump_pxpf(fx, fy, known, frame: int, half: int) -> float:
    """Largest frame-to-frame ball displacement ANYWHERE within +/-`half` frames of `frame`.

    The discriminator for "the tracker was following something that is not our ball". The
    single-ball detector sometimes latches onto a neighbouring court's ball or a parked
    object beside the court and stays latched for a second or more.

    `teleport_in_pxpf` cannot see this. It measures the jump INTO the run containing the
    contact, but by the contact frame the tracker has settled on the wrong object and the
    step is small again -- measured 3-11 px/frame on the operator's labelled cases, LOWER
    than real shots at up to 43, which is why that gate has never fired
    (n_rejected_teleport_in = 0). The implausible jump is at the START of the latch, so this
    scans a window rather than a single frame.

    The threshold is physical rather than tuned: across 84 operator-confirmed real shots the
    ball never exceeds 101 px/frame at 60fps/3840px, while latched junk reaches 2783.
    """
    lo, hi = max(0, frame - half), min(len(fx) - 1, frame + half)
    prev = None
    worst = 0.0
    for i in range(lo, hi + 1):
        if not known[i]:
            continue
        if prev is not None:
            d = math.hypot(fx[i] - fx[prev], fy[i] - fy[prev]) / max(i - prev, 1)
            if d > worst:
                worst = d
        prev = i
    return worst


# --- Core detection ----------------------------------------------------------

def reject_same_track_repeats(shots: List[dict], gap_frames: int
                              ) -> Tuple[List[dict], int]:
    """Collapse consecutive shots by the SAME player within a point. A player
    physically cannot strike the ball twice in a row in a rally (the ball must go
    to an opponent first), so two consecutive same-track impacts are ball-handling
    — the near player (nearest the corner camera, hence most visible) tapping /
    bouncing the ball between points. Keep the LAST (the real strike follows the
    handling). Only collapses within `gap_frames` (a genuinely new point, after
    ball retrieval, legitimately re-starts with the same server). Measured on
    pb_5_minute_outdoor-2: 108 → 99 shots (operator truth 98), near/far 63/45 →
    54/45. This complements the net-side filter, which splits runs at a short reset
    and so misses same-track repeats spaced further apart."""
    shots_sorted = sorted(shots, key=lambda x: x["frame"])
    kept: List[dict] = []
    n_dropped = 0
    for s in shots_sorted:
        if (kept and kept[-1]["track_id"] == s["track_id"]
                and (s["frame"] - kept[-1]["frame"]) < gap_frames):
            # Same player again within a point = ball-handling. Keep ONE, but never
            # discard a SERVE (it is the rally start): if either impact is the serve,
            # the survivor is the serve; otherwise keep the LAST (handling precedes
            # the real strike).
            if kept[-1].get("is_serve"):
                pass                       # keep the serve already held
            elif s.get("is_serve"):
                kept[-1] = s               # promote to the serve
            else:
                kept[-1] = s               # keep the last of a handling run
            n_dropped += 1
        else:
            kept.append(s)
    return kept, n_dropped


def reject_same_side_runs(shots: List[dict], side_by_track: Dict[int, str],
                          reset_frames: int, fps: float = 60.0,
                          ball_xy: Optional[Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]] = None,
                          excursion_px: Optional[float] = None
                          ) -> Tuple[List[dict], int]:
    """Net-side alternation / ball-handling rejection. Every rally shot crosses
    the net, so the striker's net side must alternate; a run of consecutive
    same-side impacts means the ball stayed on one side — a player catching /
    holding / bouncing the ball between points, not rally shots (you can't
    legally hit twice in a row).

    Within each same-side run, keep the STRONGEST impact and drop the rest.

    This kept the LAST impact until 2026-08-18, on the reasoning that handling precedes
    the real shot (you catch/bounce, THEN serve/hit). That reasoning holds for genuine
    handling but inverts on the case the operator reported as "wrong player": a real
    strike followed by a weak spurious inflection is also a same-side run, and keeping the
    LAST deletes the strike and keeps the junk. Measured at t=200s -- the partner's dink
    turns the ball 172 degrees at 199.60s and was dropped, while a 26.6-degree wobble at
    200.13s survived and was attributed to the user. Same story at 284.45s (156 vs 11.3)
    and 286.63s (180 vs 0.0).

    Impact strength uses the same score as candidate selection -- max(turn/180, speed
    ratio) -- so a paddle reversal outranks a tracking wobble regardless of order. For a
    true handling run the real shot is still the most impulsive event in the run, so the
    original case is preserved.

    Runs are split by a side change, a gap longer than reset_frames (a new rally), or -- when
    `ball_xy` and `excursion_px` are given -- by the BALL having left and come back between
    two impacts. That last split is what stops a missed shot from costing a second real one:
    the run premise ("nobody hit it in between") is exactly what a missed shot violates, and
    a ball that travelled `excursion_px` away from the first impact did not stay in the
    hitter's hands. See SAME_SIDE_EXCURSION_PX for the measurement.

    `ball_xy` is (fx, fy, known) indexed by frame. Returns (kept, n_dropped)."""
    fps_local = float(fps) or 60.0
    shots_sorted = sorted(shots, key=lambda x: x["frame"])
    kept: List[dict] = []
    n_dropped = 0
    run: List[dict] = []
    prev_side: Optional[str] = None
    prev_frame: Optional[int] = None

    def strength(s) -> float:
        turn = float(s.get("turn_rate_deg") or 0.0) / 180.0
        ratio = min(1.0, float(s.get("speed_change_ratio") or 0.0))
        return max(turn, ratio)

    def flush():
        nonlocal n_dropped
        if not run:
            return
        # Both rules are right about different runs, and they separate on TIMING.
        # Genuine ball-handling (bounce, bounce, serve) is spread over seconds and the
        # real shot is LAST. A real strike followed by a tracking wobble is tight -- the
        # wobble lands within half a second -- and the real shot is the STRONGEST.
        span = (run[-1]["frame"] - run[0]["frame"]) / max(fps_local, 1.0)
        kept.append(run[-1] if span >= HANDLING_SPREAD_S else max(run, key=strength))
        n_dropped += len(run) - 1

    def ball_left(f0: int, f1: int) -> bool:
        """Did the ball travel `excursion_px` away from the impact at f0 before f1?"""
        if ball_xy is None or not excursion_px:
            return False
        fx_, fy_, known_ = ball_xy
        if not (0 <= f0 < len(fx_)) or not known_[f0]:
            return False
        hi = min(f1 + 1, len(fx_))
        for g in range(f0, hi):
            if known_[g] and math.hypot(fx_[g] - fx_[f0], fy_[g] - fy_[f0]) >= excursion_px:
                return True
        return False

    for s in shots_sorted:
        side = side_by_track.get(s["track_id"])
        same_run = (run and side is not None and side == prev_side
                    and prev_frame is not None
                    and (s["frame"] - prev_frame) <= reset_frames
                    and not ball_left(prev_frame, s["frame"]))
        if same_run:
            run.append(s)
        else:
            flush()
            run = [s]
        prev_side, prev_frame = side, s["frame"]
    flush()
    kept.sort(key=lambda x: x["frame"])
    return kept, n_dropped


def structure_points(shots: List[dict], net_y_ft: float, behind_baseline_ft: float,
                     open_gap_frames: int, return_frames: int,
                     dead_gap_frames: int, min_inter_serve_frames: int) -> int:
    """Unified point-boundary detection (operator method 2026-07-27). A rally is
    SERVE -> ... -> POINT-END, one of each, alternating. Combine weak cues with the
    structural one-each constraint so no single rule has to carry it. Sets `is_serve`
    and `is_between_point` on each shot in place; returns the serve count.

    - POINT-END: a shot the opponent does NOT return (no opposite-side shot within
      `return_frames`) AND dead time follows (>= `dead_gap_frames` to the next shot).
    - SERVE: struck from BEHIND the baseline (>= `behind_baseline_ft` from the net,
      robust ground depth) AND opens a point (>= `open_gap_frames` since the prior shot).
    - MUTUAL CONSTRAINT: accept a serve only if a POINT ENDED since the previous
      accepted serve -> drops the deep between-point balls/returns that look serve-like
      on depth+gap alone (two serves with no end between = one is spurious).
    - BETWEEN-POINT: shots after a rally's point-end and before the next serve (and any
      before the first serve / after the last) -> dead-time returns-to-server, thrown
      balls. Flagged is_between_point for downstream to exclude from rally evaluation."""
    ss = sorted(shots, key=lambda s: int(s["frame"]))
    N = len(ss)

    def side(i):
        return ss[i].get("hitter_side")

    def dist(i):
        hy = (ss[i].get("hitter_court_xy_ft") or [None, None])[1]
        return abs(hy - net_y_ft) if hy is not None else 0.0

    def gap_prev(i):
        return (ss[i]["frame"] - ss[i - 1]["frame"]) if i > 0 else open_gap_frames + 1

    def gap_next(i):
        return (ss[i + 1]["frame"] - ss[i]["frame"]) if i + 1 < N else dead_gap_frames + 1

    def returned(i):
        s = side(i)
        if s is None:
            return None
        for j in range(i + 1, N):
            dt = ss[j]["frame"] - ss[i]["frame"]
            if dt <= 0:
                continue
            if dt > return_frames:
                break
            if side(j) is not None and side(j) != s:
                return True
        return False

    def serve_cand(i):
        return dist(i) >= behind_baseline_ft and gap_prev(i) >= open_gap_frames

    ends = {i for i in range(N)
            if not serve_cand(i) and returned(i) is False
            and gap_next(i) >= dead_gap_frames}

    accepted: List[int] = []
    for i in range(N):
        if not serve_cand(i):
            continue
        end_since = bool(accepted) and any(accepted[-1] < e < i for e in ends)
        gap_since = (not accepted
                     or (ss[i]["frame"] - ss[accepted[-1]]["frame"]) >= min_inter_serve_frames)
        if not accepted or end_since or gap_since:
            accepted.append(i)
        elif returned(i) is True and returned(accepted[-1]) is not True:
            # RETURN TIE-BREAK. Acceptance is otherwise greedy first-wins, so a deep
            # between-point ball (a feed lobbed back to the server) claims the slot and
            # then BLOCKS the real serve behind it via the mutual constraint -- measured
            # on the acceptance clip, the false serve at 120.5s blocked the real one at
            # 128.8s, and 295.9s blocked 302.9s. The two error classes are one bug.
            #
            # A serve is answered: the opposing side plays the ball back. A feed is not.
            # Measured over the labelled serves, 8 of 11 real serves draw a reply within
            # `return_frames` against 1 of 5 false ones. Too weak to gate on outright --
            # 3 real serves go unanswered because the REPLY was missed, not because it
            # never happened -- but decisive when choosing between two candidates for the
            # same slot, which is the only thing it is used for here.
            accepted[-1] = i
    accepted_set = set(accepted)

    # Between-point = dead-time balls AFTER a rally's point-end and before the next
    # serve (returns-to-server / thrown balls). Do NOT flag pre-first-serve shots:
    # a real first rally can precede the first DETECTED serve (its serve may be missed),
    # and dropping it would delete real play. Segmentation keeps that first burst.
    between = set()
    bounds = accepted + [N]
    for k, si in enumerate(accepted):
        nxt = bounds[k + 1]
        eb = [e for e in ends if si < e < nxt]
        if eb:                                   # trim dead-time balls after the point-end
            between.update(range(eb[-1] + 1, nxt))
        # no detected end -> keep the whole span (a missed end must not drop real play)

    n = 0
    for i in range(N):
        ss[i]["is_serve"] = i in accepted_set
        ss[i]["is_between_point"] = i in between
        n += int(i in accepted_set)
    return n


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
    latch_thresh = params["latch_jump_px_per_frame"]
    latch_half = params["latch_window_frames"]
    n_rejected_serve_blip = 0
    n_rejected_teleport = 0
    n_rejected_latch = 0

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
        # LATCH gate. The gate above has never once fired (n_rejected_teleport_in = 0 on
        # the acceptance clip) because the AND with the blip length makes it unreachable,
        # so it defends nothing. This one is independent and needs no run-length test:
        # a displacement this large is not a ball trajectory at any speed, so whatever the
        # tracker was following, it was not our ball.
        if (contam_filter
                and max_latch_jump_pxpf(fx, fy, known, f, latch_half) >= latch_thresh):
            n_rejected_latch += 1
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
            shots, side_by_track or {}, params["handling_reset_frames"],
            params["fps"], ball_xy=(fx, fy, known),
            excursion_px=params.get("same_side_excursion_px"))
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
        # Already captured as an impulse shot? Then PROMOTE that shot to a serve
        # rather than discarding the serve evidence. Skipping here left the shot
        # with is_serve=False, so real serves went unflagged: 11 of 18 rallies had
        # no serve at all, which starves rally segmentation and makes the third
        # shot (a core USAPA item) unidentifiable.
        near_impulse = [sf for sf in impulse_frames if abs(f - sf) <= W]
        if near_impulse:
            target = min(near_impulse, key=lambda sf: abs(f - sf))
            for sh in shots:
                if sh["frame"] == target and not sh["is_serve"]:
                    sh["is_serve"] = True
                    sh["detection_method"] = "impulse+serve_appearance"
                    n_serves += 1
                    break
            continue
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
    # FINAL pass: collapse same-PLAYER repeats within a point (ball-handling). Runs
    # here, after serves are merged in, so it catches every same-track adjacency
    # (the mid-stream net-side filter splits runs at a short reset and misses those
    # spaced further apart). A player cannot strike twice in a row in a rally.
    n_same_track = 0
    if params.get("handling_filter"):
        shots, n_same_track = reject_same_track_repeats(shots, params["rally_gap_frames"])
    shots.sort(key=lambda s: s["frame"])
    # Drop GROUND BALLS: events where the ball never actually flew (rolled at the net,
    # picked up, bounced before a serve). Physics-based, so it needs no knowledge of
    # rally boundaries -- unlike the cross-net "in-play" test, which the same operator
    # labels showed kills 9 of 21 REAL shots and misses 4 of 9 junk ones, because a
    # ball rolled back at the net does cross the net.
    n_ground_ball = 0
    if params.get("contamination_filter"):
        kept = []
        for s in shots:
            gf = grounded_fraction(court_M, fx, fy, known, int(s["frame"]),
                                   params["grounded_window_frames"])
            if gf is not None and gf >= params["grounded_max_frac"]:
                n_ground_ball += 1
                continue
            kept.append(s)
        shots = kept
    # Unified point-boundary detection (operator method): re-derive is_serve + flag
    # between-point balls from the SERVE->...->POINT-END structure (combines depth,
    # return-timing, dead-time + the one-serve-per-point constraint). Real ball only.
    if params.get("contamination_filter"):
        n_serves = structure_points(
            shots, net_y_ft=params["net_y_ft"],
            behind_baseline_ft=params["serve_behind_baseline_ft"],
            open_gap_frames=params["serve_open_gap_frames"],
            return_frames=params["point_return_frames"],
            dead_gap_frames=params["point_dead_gap_frames"],
            min_inter_serve_frames=params["point_min_inter_serve_frames"])
    for s in shots:
        s.setdefault("is_between_point", False)
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
        "n_rejected_latch": n_rejected_latch,
        "n_serve_deduped": n_serve_dedup,
        "n_rejected_ground_ball": n_ground_ball,
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
    same_side_excursion = (args.same_side_excursion_px
                           if args.same_side_excursion_px is not None
                           else SAME_SIDE_EXCURSION_PX * res_scale)
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
        "rally_gap_frames": int(round(RALLY_GAP_S * float(fps))),
        "handling_reset_frames": int(round(HANDLING_RESET_S * float(fps))),
        "same_side_excursion_px": same_side_excursion,
        "handling_filter": ball_source == "real",
        "contamination_filter": ball_source == "real",
        "min_serve_run_frames": max(2, int(round(MIN_SERVE_RUN_S * float(fps)))),
        "teleport_in_px_per_frame": TELEPORT_IN_PX_PER_FRAME * res_scale,
        "latch_jump_px_per_frame": LATCH_JUMP_PX_PER_FRAME * res_scale,
        "latch_window_frames": int(round(LATCH_WINDOW_S * float(fps))),
        "serve_dedup_frames": int(round(SERVE_DEDUP_S * float(fps))),
        "serve_behind_baseline_ft": SERVE_BEHIND_BASELINE_FT,
        "serve_open_gap_frames": int(round(SERVE_OPEN_GAP_S * float(fps))),
        "point_return_frames": int(round(POINT_RETURN_S * float(fps))),
        "point_dead_gap_frames": int(round(POINT_DEAD_GAP_S * float(fps))),
        "point_min_inter_serve_frames": int(round(POINT_MIN_INTER_SERVE_S * float(fps))),
        "grounded_window_frames": int(round(GROUNDED_WINDOW_S * float(fps))),
        "grounded_max_frac": GROUNDED_MAX_FRAC,
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
             f"{stats['n_rejected_teleport_in']} teleport-in, "
             f"{stats['n_rejected_latch']} wrong-object latch)")

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
    p.add_argument("--same-side-excursion-px", type=float, default=None,
                   dest="same_side_excursion_px",
                   help="absolute px override (default: SAME_SIDE_EXCURSION_PX scaled by "
                        "frame_width/1920); a same-side run splits where the ball "
                        "travelled this far, 0 to disable the split")
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

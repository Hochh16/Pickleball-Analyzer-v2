"""Stage 7 — segment rallies.

Group the per-shot stream from Stage 6 (classified.json) into rallies and tag
each rally with how it ended (serve-fault / double-bounce / ball-out /
net-or-short / ball-not-returned / ball-off-frame / unknown). Boundaries come
from is_serve; end_reasons come from the bounce stream (bounces.json from
Stage 5.5), including side-of-net reasoning for hitter vs receiver error
attribution.

Role-blind v1: no winner_side, no track_roles.json dependency. server_track_id
is carried through from the serve shot (no role inference needed).

See stages/segment_rallies/contract.md for the full spec.

Usage:
    python -m stages.segment_rallies.segment_rallies data/test_clip
    python -m stages.segment_rallies.segment_rallies data/test_clip --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
STAGE_VERSION = "0.3.0"  # 0.2.0 -> 0.3.0 (real ball): minimum-rally filter drops
                         # between-point / after-game net-tapping micro-rallies
                         # (short AND few-shot). 0.2.0: ball-OUT-OF-PLAY rally
                         # boundaries, hitter_side sides, real-ball unknown
                         # end_reason, courtesy-feed drop. Gated to real ball;
                         # synthetic path unchanged.

# --- Config (matches contract) ----------------------------------------------
SERVE_FAULT_MAX_FRAMES = 60       # quick-next-serve = serve fault (~2s @ 30fps)
NET_Y_FT = 22.0                   # net line in court coordinates
KITCHEN_DEPTH_FT = 7.0            # kitchen extends 7 ft from net (each side)
REFERENCE_FPS = 30.0              # frame-count params tuned at 30fps; scale by fps/this
# Real-ball rally boundaries: a point breaks when the BALL GOES OUT OF PLAY, not
# when a hit is merely missed. During a point the ball is in flight (known almost
# every frame, tiny <~0.25s absences); between points it is dead (picked up /
# reset) for 3-4s. A sustained not-in-play run is the physical, general boundary
# signal (a raw inter-shot time-gap falsely splits a rally wherever a shot was
# missed). 1.5s sits far above in-rally occlusions and far below real dead time.
# Seconds (fps-independent). Gated to the real ball (synthetic keeps is_serve-only).
BALL_DEAD_RUN_SEC = 1.5
# Every rally shot crosses the net, so consecutive rally shots ALTERNATE sides. Two
# consecutive SAME-side shots this far apart mean the ball never went to the opponent
# in between — that is ball-handling / a reset between points, not one rally. This
# catches dead time the out-of-play rule misses, because the ball is often still
# visible and moving while a player bounces it before serving (measured: 3.6 s at 90 %
# visibility, 12 px/frame, four same-side contacts). A SHORT same-side gap is left
# alone: that is a missed opponent shot mid-rally, which must NOT split the rally.
SAME_SIDE_STALL_SEC = 2.0
# Absolute ceiling on the time between consecutive contacts WITHIN one rally,
# regardless of side. In real play contacts are 0.5-2 s apart (a high lob ~3 s at
# most); even one missed shot only doubles that. Gaps beyond this are dead time
# between points that the out-of-play rule can miss whenever the ball stays visible
# (a player bouncing/holding it). Observed pre-fix: rallies carrying 5-15 s gaps.
MAX_INTRA_RALLY_GAP_SEC = 4.0

# Minimum-rally filter (real ball). A real point is a sustained exchange; between
# points / after the game players stand at the net and tap the ball a couple times
# (pb_2min rallies 6 & 7: 0.8s/1.1s, 2 shots each — one even off a FALSELY detected
# serve). Drop a segment only when it is BOTH shorter than MIN_RALLY_SEC AND has
# fewer than MIN_RALLY_SHOTS shots — conservative AND-logic, so a segment that is
# EITHER long OR has many shots is always kept. A lone serve-fault (n_shots==1) is
# never treated as a micro-rally (guarded), so real serves put into play survive.
MIN_RALLY_SEC = 2.0
MIN_RALLY_SHOTS = 3

END_REASONS = {"serve-fault", "double-bounce", "ball-out", "net-or-short",
               "ball-not-returned", "ball-off-frame", "unknown"}

EPS = 1e-9


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("segment_rallies")
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


def load_ball_known(path: Path):
    """Per-frame 'ball in play' (visible|interpolated), indexed by frame_idx, as
    a numpy bool array. Drives the real-ball rally boundary (ball-out-of-play
    runs). Returns None if ball.parquet is absent (caller falls back to serves)."""
    if not path.exists():
        return None
    import numpy as np
    import pandas as pd
    df = pd.read_parquet(path, columns=["frame_idx", "visible", "interpolated"])
    df = df.sort_values("frame_idx")
    n = int(df["frame_idx"].max()) + 1
    known = np.zeros(n, dtype=bool)
    idx = df["frame_idx"].to_numpy()
    known[idx] = (df["visible"].to_numpy() | df["interpolated"].to_numpy())
    return known


def load_court_fps(path: Path) -> Optional[float]:
    """Fallback fps source if classified.json doesn't carry it."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        c = json.load(f)
    video = c.get("video", {}) or {}
    return video.get("fps")


# --- Side helpers ------------------------------------------------------------

def side_of_net(court_y: Optional[float]) -> Optional[str]:
    """Return 'near' if court_y < NET_Y_FT, 'far' if > NET_Y_FT, None if
    indeterminate (null / NaN)."""
    if court_y is None:
        return None
    try:
        cy = float(court_y)
    except (TypeError, ValueError):
        return None
    if cy != cy:  # NaN check
        return None
    if cy < NET_Y_FT:
        return "near"
    return "far"


def bounce_in_receivers_kitchen(bounce_court_y: Optional[float],
                                 server_side: Optional[str]) -> Optional[bool]:
    """A serve is a kitchen-fault if its bounce lands in the RECEIVER's kitchen:
    - Server on near side → receiver's kitchen is y in [22, 29].
    - Server on far side  → receiver's kitchen is y in [15, 22].
    Returns None if either input is indeterminate."""
    if bounce_court_y is None or server_side is None:
        return None
    try:
        by = float(bounce_court_y)
    except (TypeError, ValueError):
        return None
    if by != by:
        return None
    if server_side == "near":
        return NET_Y_FT <= by <= NET_Y_FT + KITCHEN_DEPTH_FT
    if server_side == "far":
        return NET_Y_FT - KITCHEN_DEPTH_FT <= by <= NET_Y_FT
    return None


# --- Boundary segmentation ---------------------------------------------------

def longest_dead_run(a: int, b: int, ball_known) -> int:
    """Longest run of consecutive NOT-known (ball out of play) frames strictly
    between frames a and b. The key rally-boundary signal: during a point the
    ball is in flight (known) with only tiny absences; between points it is dead
    (picked up / reset) for a long stretch."""
    n = len(ball_known)
    best = cur = 0
    for k in range(a + 1, b):
        if 0 <= k < n and not ball_known[k]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def segment_rallies(shots: List[dict],
                    ball_known=None,
                    ball_dead_run_frames: Optional[int] = None,
                    gap_split: bool = False,
                    same_side_stall_frames: Optional[int] = None,
                    max_gap_frames: Optional[int] = None
                    ) -> Tuple[List[List[dict]], List[dict]]:
    """Split the shot stream into rallies. Returns (rally_shot_lists,
    dropped_shots). Each rally is a list of shot dicts.

    Synthetic path (`gap_split=False`, the contract's v1): a new rally starts at
    every `is_serve` shot; shots before the first serve are dropped.

    Real-ball path (`gap_split=True`): Stage 5 under-detects serves AND shots, so
    is_serve alone merges points while a raw inter-shot time-gap falsely splits a
    rally wherever a hit was missed. The robust, GENERAL boundary is whether the
    BALL WENT OUT OF PLAY: a new rally starts when the ball had a sustained
    not-in-play run (`>= ball_dead_run_frames`, the ball picked up / reset between
    points) since the previous shot, OR at a flagged serve. A missed shot leaves
    the ball flying (short absences) and does NOT break the rally. Then non-rally
    segments are dropped: a single shot that is not a flagged serve is a
    between-points courtesy feed or isolated noise, not a rally."""
    if not gap_split:
        rallies: List[List[dict]] = []
        pre_rally: List[dict] = []
        cur: Optional[List[dict]] = None
        for s in shots:
            if s.get("is_serve"):
                if cur is not None:
                    rallies.append(cur)
                cur = [s]
            else:
                if cur is None:
                    pre_rally.append(s)
                else:
                    cur.append(s)
        if cur is not None:
            rallies.append(cur)
        return rallies, pre_rally

    # Real-ball: split on a sustained ball-out-of-play run OR a flagged serve.
    segments: List[List[dict]] = []
    cur = None
    prev_frame: Optional[int] = None
    prev_side: Optional[str] = None
    for s in shots:
        f = int(s["frame"])
        dead = (longest_dead_run(prev_frame, f, ball_known)
                if (prev_frame is not None and ball_known is not None
                    and ball_dead_run_frames is not None) else 0)
        # Same-side stall: consecutive shots on the SAME side of the net, far apart
        # in time = the ball never crossed = ball-handling between points, not one
        # rally. (A short same-side gap is a missed opponent shot; do not split.)
        side = s.get("hitter_side")
        same_side_stall = bool(
            same_side_stall_frames is not None and prev_side is not None
            and side is not None and side == prev_side and prev_frame is not None
            and (f - prev_frame) >= same_side_stall_frames)
        # Absolute ceiling: no rally has consecutive contacts this far apart.
        too_long = bool(max_gap_frames is not None and prev_frame is not None
                        and (f - prev_frame) >= max_gap_frames)
        starts_new = bool(s.get("is_serve")) or same_side_stall or too_long or (
            ball_dead_run_frames is not None and dead >= ball_dead_run_frames)
        if cur is None:
            cur = [s]
        elif starts_new:
            segments.append(cur)
            cur = [s]
        else:
            cur.append(s)
        prev_frame = f
        prev_side = side
    if cur is not None:
        segments.append(cur)

    rallies, dropped = [], []
    for seg in segments:
        if len(seg) == 1 and not seg[0].get("is_serve"):
            dropped.extend(seg)  # courtesy feed / isolated non-serve hit
        else:
            rallies.append(seg)
    return rallies, dropped


def drop_micro_rallies(rally_groups: List[List[dict]], fps: float,
                       min_sec: float = MIN_RALLY_SEC,
                       min_shots: int = MIN_RALLY_SHOTS
                       ) -> Tuple[List[List[dict]], List[List[dict]]]:
    """Remove spurious micro-rallies (between-point / after-game net-tapping).
    A segment is dropped only when it is BOTH shorter than `min_sec` AND has
    fewer than `min_shots` shots — so anything either long OR with many shots is
    kept. `n_shots > 1` is required to drop, so a lone serve-fault (a real serve
    put into play, n_shots==1) is never removed here. Duration is the shot span
    (first→last shot); the ending bounce isn't needed to judge "too short to be a
    point". Returns (kept_groups, dropped_groups)."""
    kept: List[List[dict]] = []
    dropped: List[List[dict]] = []
    for g in rally_groups:
        n = len(g)
        span_sec = ((int(g[-1]["frame"]) - int(g[0]["frame"])) / fps
                    if fps > 0 else 0.0)
        if n > 1 and n < min_shots and span_sec < min_sec:
            dropped.append(g)
        else:
            kept.append(g)
    return kept, dropped


# --- End-reason classification ----------------------------------------------

def classify_rally(rally_shots: List[dict], bounces: List[dict],
                    next_rally_serve_frame: Optional[int],
                    serve_fault_max_frames: int = SERVE_FAULT_MAX_FRAMES,
                    real_ball: bool = False
                    ) -> Tuple[str, float, Optional[int], dict]:
    """Returns (end_reason, confidence, ending_bounce_id, end_signals).
    Implements the rule table in the contract: serve-fault > double-bounce >
    net-or-short > ball-out > ball-not-returned > ball-off-frame > unknown.

    On the real ball (`real_ball=True`) the zero-bounce case is labeled
    **unknown**, not "ball-off-frame": with real bounce recall the absence of a
    detected rally-ending bounce almost always means the bounce was MISSED, not
    that the ball flew off-frame, so the off-frame inference (and its hitter-error
    attribution) is not warranted. Synthetic keeps "ball-off-frame" (clean ball)."""
    last_shot = rally_shots[-1]
    last_shot_id = int(last_shot["shot_id"])
    last_frame = int(last_shot["frame"])
    n_shots = len(rally_shots)
    serve = rally_shots[0]
    # Side comes from the HITTING PLAYER's ground position (Stage 5 hitter_side),
    # NOT the airborne ball-contact projection (impact_court_xy_ft), which is
    # garbage through the ground homography for an elevated contact. Fall back to
    # the old projection only if hitter_side is absent (pre-0.3.0 shots).
    server_side = serve.get("hitter_side") or side_of_net(
        (serve.get("impact_court_xy_ft") or [None, None])[1])
    hitter_side = last_shot.get("hitter_side") or side_of_net(
        (last_shot.get("impact_court_xy_ft") or [None, None])[1])

    # Find post-last-shot bounces: between_shots[0] == last_shot_id.
    post = [b for b in bounces
            if b.get("between_shots") and b["between_shots"][0] == last_shot_id]
    post.sort(key=lambda b: int(b["frame"]))
    n_post = len(post)
    last_bounce = post[-1] if post else None
    last_bounce_in_court = (last_bounce.get("is_in_court")
                            if last_bounce is not None else None)
    last_bounce_out_side = (last_bounce.get("out_side")
                            if last_bounce is not None else None)
    last_bounce_court_xy = (last_bounce.get("court_xy_ft") or [None, None]
                            if last_bounce is not None else [None, None])
    last_bounce_side = side_of_net(last_bounce_court_xy[1]
                                    if last_bounce_court_xy else None)
    last_bounce_in_kitchen = bounce_in_receivers_kitchen(
        last_bounce_court_xy[1] if last_bounce_court_xy else None,
        server_side)
    frames_to_next_serve = (next_rally_serve_frame - last_frame
                            if next_rally_serve_frame is not None else None)

    end_signals = {
        "n_bounces_after_last_shot": n_post,
        "last_bounce_in_court": last_bounce_in_court,
        "last_bounce_out_side": last_bounce_out_side,
        "last_bounce_side": last_bounce_side,
        "last_bounce_in_kitchen": last_bounce_in_kitchen,
        "hitter_side": hitter_side,
        "server_side": server_side,
        "frames_to_next_serve": frames_to_next_serve,
    }

    # Rule 1: serve-fault (n_shots == 1).
    if n_shots == 1:
        # First post-serve bounce gives the strongest signal.
        first_post = post[0] if post else None
        if first_post is not None and first_post.get("is_in_court") is False:
            return ("serve-fault", 0.9, int(first_post["bounce_id"]), end_signals)
        if first_post is not None and first_post is last_bounce \
                and last_bounce_in_kitchen is True:
            return ("serve-fault", 0.9, int(first_post["bounce_id"]), end_signals)
        # Check the kitchen flag on the first bounce too (not just last)
        if first_post is not None:
            first_bounce_court_xy = (first_post.get("court_xy_ft")
                                      or [None, None])
            first_in_kitchen = bounce_in_receivers_kitchen(
                first_bounce_court_xy[1] if first_bounce_court_xy else None,
                server_side)
            if first_in_kitchen is True:
                return ("serve-fault", 0.9,
                        int(first_post["bounce_id"]), end_signals)
        if (frames_to_next_serve is not None and 0 < frames_to_next_serve
                <= serve_fault_max_frames):
            return ("serve-fault", 0.7,
                    int(first_post["bounce_id"]) if first_post else None,
                    end_signals)
        return ("serve-fault", 0.5,
                int(first_post["bounce_id"]) if first_post else None,
                end_signals)

    # Rule 2: double-bounce.
    if n_post >= 2:
        return ("double-bounce", 0.85,
                int(last_bounce["bounce_id"]), end_signals)

    # Rule 3: net-or-short (only for IN-COURT bounces on hitter's side; an
    # out-of-court bounce that happens to project to hitter's side is still a
    # ball-out, not a net hit).
    if (n_post >= 1 and last_bounce is not None
            and last_bounce.get("is_in_court") is True
            and last_bounce_side is not None and hitter_side is not None
            and last_bounce_side == hitter_side):
        return ("net-or-short", 0.8,
                int(last_bounce["bounce_id"]), end_signals)

    # Rule 4: ball-out.
    if (n_post == 1 and last_bounce is not None
            and last_bounce.get("is_in_court") is False):
        return ("ball-out", 0.85,
                int(last_bounce["bounce_id"]), end_signals)

    # Rule 5: ball-not-returned.
    if (n_post >= 1 and last_bounce is not None
            and last_bounce.get("is_in_court") is True):
        # If sides are determinate AND last bounce is on hitter's side, rule 3
        # already matched. Otherwise treat as receiver-side or
        # indeterminate-but-in-court → receiver missed.
        return ("ball-not-returned", 0.75,
                int(last_bounce["bounce_id"]), end_signals)

    # Rule 6: ball-off-frame (synthetic) / unknown (real). Zero post-last-shot
    # bounces with play stopped. On the clean synthetic ball this implies the
    # ball flew off-frame (hitter error); on the real ball it almost always means
    # the rally-ending bounce was simply missed, so label it honestly "unknown".
    if n_post == 0 and frames_to_next_serve is not None \
            and frames_to_next_serve > 0:
        return (("unknown", 0.3, None, end_signals) if real_ball
                else ("ball-off-frame", 0.5, None, end_signals))

    # Rule 7: unknown.
    return ("unknown", 0.3, None, end_signals)


# --- Main pipeline -----------------------------------------------------------

def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    classified_path = folder / "classified.json"
    bounces_path = folder / "bounces.json"
    court_path = folder / "court.json"
    out_path = folder / "rallies.json"

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.",
             FileExistsError)

    classified = load_json(classified_path)
    bounces_doc = load_json(bounces_path)
    if bounces_doc.get("schema_version") != 1:
        fail(f"bounces.json schema_version={bounces_doc.get('schema_version')} "
             f"unexpected (Stage 7 v1 expects 1)", ValueError)

    fps = classified.get("fps") or load_court_fps(court_path)
    if fps is None or fps <= 0:
        fail("could not determine fps from classified.json or court.json",
             ValueError)

    ball_source = classified.get("ball_source") or bounces_doc.get("ball_source") or "real"
    if ball_source == "synthetic":
        log.warning("ball_source is SYNTHETIC: rally end_reasons are "
                    "placeholder-derived.")

    # fps scaling: frame-count params were tuned at 30fps. The dead-ball-run
    # threshold is in seconds (fps-independent). Ball-dead-run rally splitting +
    # courtesy-feed drop are real-ball adaptations (Stage 5 under-detects serves
    # AND shots); synthetic keeps is_serve-only.
    fps_scale = float(fps) / REFERENCE_FPS
    serve_fault_max_frames = max(1, int(round(SERVE_FAULT_MAX_FRAMES * fps_scale)))
    ball_dead_run_frames = max(1, int(round(BALL_DEAD_RUN_SEC * float(fps))))
    same_side_stall_frames = max(1, int(round(SAME_SIDE_STALL_SEC * float(fps))))
    max_intra_gap_frames = max(1, int(round(MAX_INTRA_RALLY_GAP_SEC * float(fps))))
    gap_split = (ball_source == "real")

    # Ball visibility drives the rally boundary on the real ball: a point breaks
    # only when the ball goes out of play (sustained not-in-play run).
    ball_known = load_ball_known(folder / "ball.parquet") if gap_split else None

    shots = sorted(classified.get("shots", []), key=lambda s: int(s["frame"]))
    bounces = sorted(bounces_doc.get("bounces", []),
                     key=lambda b: int(b["frame"]))

    rally_groups, pre_rally = segment_rallies(
        shots, ball_known=ball_known,
        ball_dead_run_frames=ball_dead_run_frames, gap_split=gap_split,
        same_side_stall_frames=same_side_stall_frames,
        max_gap_frames=max_intra_gap_frames)

    # Minimum-rally filter (real ball only, like the other real-world-phenomenon
    # gates): drop between-point/after-game net-tapping bursts. Synthetic keeps
    # every is_serve rally so its acceptance bars are unperturbed. Dropped shots
    # roll into the unassigned bucket so shot accounting still reconciles.
    micro_rallies: List[List[dict]] = []
    if gap_split:
        rally_groups, micro_rallies = drop_micro_rallies(
            rally_groups, float(fps), args.min_rally_sec, args.min_rally_shots)
        for g in micro_rallies:
            pre_rally.extend(g)
    if not rally_groups:
        log.warning("no rallies found; emitting empty rallies list")

    out_rallies: List[dict] = []
    for ri, rally_shots in enumerate(rally_groups):
        next_serve_frame = (int(rally_groups[ri + 1][0]["frame"])
                            if ri + 1 < len(rally_groups) else None)
        end_reason, conf, ending_bid, signals = classify_rally(
            rally_shots, bounces, next_serve_frame, serve_fault_max_frames,
            real_ball=(ball_source == "real"))

        last_shot = rally_shots[-1]
        last_frame = int(last_shot["frame"])
        serve = rally_shots[0]
        start_frame = int(serve["frame"])
        # End frame is max(last shot frame, ending bounce frame).
        if ending_bid is not None:
            ebf = next((int(b["frame"]) for b in bounces
                        if int(b["bounce_id"]) == ending_bid), last_frame)
            end_frame = max(last_frame, ebf)
        else:
            end_frame = last_frame

        out_rallies.append({
            "rally_id": ri,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_t_sec": round(start_frame / fps, 3),
            "end_t_sec": round(end_frame / fps, 3),
            "duration_sec": round((end_frame - start_frame) / fps, 3),
            "shot_ids": [int(s["shot_id"]) for s in rally_shots],
            "n_shots": len(rally_shots),
            "serve_shot_id": int(serve["shot_id"]),
            "server_track_id": int(serve["track_id"]),
            "server_is_user": bool(serve.get("is_user", False)),
            # True when this rally's start was inferred from a dead-time gap
            # rather than a flagged serve (Stage 5 serve-detection gap). Stage 8
            # should treat server attribution / serve-fault stats as lower
            # confidence for these. Always False on the synthetic path.
            "serve_is_inferred": bool(gap_split and not serve.get("is_serve")),
            "end_reason": end_reason,
            "end_reason_confidence": round(conf, 3),
            "ending_bounce_id": ending_bid,
            "end_signals": signals,
        })

    # Stats.
    by_end_reason: Dict[str, int] = {}
    for r in out_rallies:
        er = r["end_reason"]
        by_end_reason[er] = by_end_reason.get(er, 0) + 1
    total_shots_in_rallies = sum(r["n_shots"] for r in out_rallies)
    mean_n = (total_shots_in_rallies / len(out_rallies)
              if out_rallies else 0.0)
    mean_dur = (sum(r["duration_sec"] for r in out_rallies) / len(out_rallies)
                if out_rallies else 0.0)
    stats = {
        "n_rallies": len(out_rallies),
        "by_end_reason": by_end_reason,
        "total_shots_in_rallies": total_shots_in_rallies,
        "unassigned_shots": len(pre_rally),
        "mean_rally_length": round(mean_n, 3),
        "mean_rally_duration_sec": round(mean_dur, 3),
    }

    warnings: List[str] = []
    if ball_source == "synthetic":
        warnings.append("ball_source is 'synthetic': rally end_reasons are "
                        "derived from PLACEHOLDER ball data.")
    if micro_rallies:
        spans = [f"{round((int(g[-1]['frame'])-int(g[0]['frame']))/float(fps),2)}s"
                 f"/{len(g)}sh" for g in micro_rallies]
        warnings.append(
            f"{len(micro_rallies)} micro-rally segment(s) dropped as non-rally "
            f"(< {args.min_rally_sec}s AND < {args.min_rally_shots} shots; "
            f"between-point / after-game net-tapping): {spans}. Their shots are "
            f"counted as unassigned.")
    if pre_rally:
        kind = ("non-rally shot(s) (pre-first-rally, courtesy/between-point "
                "feeds, or micro-rally)" if gap_split else
                "shot(s) preceded the first serve")
        warnings.append(f"{len(pre_rally)} {kind} were dropped (unassigned). "
                        f"Shot_ids: {sorted(int(s['shot_id']) for s in pre_rally)}")
    if not out_rallies:
        warnings.append("no rallies emitted (no serve/rally boundaries found)")

    log.info(f"segmented {len(out_rallies)} rallies; "
             f"by_end_reason={by_end_reason}; "
             f"unassigned_shots={len(pre_rally)}")

    out = {
        "schema_version": SCHEMA_VERSION,
        "source_classified": str(classified_path),
        "source_bounces": str(bounces_path),
        "ball_source": ball_source,
        "fps": float(fps),
        "params": {
            "serve_fault_max_frames": serve_fault_max_frames,
            "net_y_ft": NET_Y_FT,
            "kitchen_depth_ft": KITCHEN_DEPTH_FT,
            "fps_scale": round(fps_scale, 4),
            "gap_split": gap_split,
            "ball_dead_run_sec": BALL_DEAD_RUN_SEC if gap_split else None,
            "ball_dead_run_frames": ball_dead_run_frames if gap_split else None,
            "ball_known_loaded": ball_known is not None,
            "min_rally_sec": args.min_rally_sec if gap_split else None,
            "min_rally_shots": args.min_rally_shots if gap_split else None,
            "n_micro_rallies_dropped": len(micro_rallies),
        },
        "rallies": out_rallies,
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
    p = argparse.ArgumentParser(description="Stage 7 — segment rallies")
    p.add_argument("folder", type=Path,
                   help="per-video folder with classified.json, bounces.json, court.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--min-rally-sec", type=float, default=MIN_RALLY_SEC,
                   dest="min_rally_sec",
                   help="drop a rally shorter than this AND below --min-rally-shots "
                        "(real ball only; between-point net-tapping)")
    p.add_argument("--min-rally-shots", type=int, default=MIN_RALLY_SHOTS,
                   dest="min_rally_shots",
                   help="shot-count half of the minimum-rally filter (see --min-rally-sec)")
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

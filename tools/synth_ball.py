"""Synthetic placeholder ball generator.

Real ball detection is paused (Stage 4.5). To develop and smoke-test the
downstream pipeline (Stage 5 shot detection and beyond) we generate a SYNTHETIC
ball trajectory whose impacts are placed at REAL player positions (from
players.parquet), so Stage 5 has true events to find. The output matches
Stage 4's ball.parquet schema EXACTLY, plus a `synthetic: true` flag in the
meta sidecar and a ground-truth sidecar (ball_synth_truth.json) that Stage 5's
smoke test grades against.

This is a PLACEHOLDER. Stages built on it must validate ball plausibility and
fail loudly on bad input (see Stage 5 contract "Defenses"). When real ball
detection (v4) exists, the synthetic data is removed and downstream stages are
re-validated against real, noisy trajectories.

Usage:
    python tools/synth_ball.py data/test_clip --seed 1234
    python tools/synth_ball.py data/test_clip --seed 1234 --gap-frac 0.15 --force

Reads from <folder>: video.mp4, court.json, players.parquet, poses.parquet (opt).
Writes into <folder>: ball.parquet, ball.meta.json, ball_synth_truth.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1            # ball.parquet schema (matches Stage 4)
TRUTH_SCHEMA_VERSION = 1
TOOL_VERSION = "0.1.0"

# --- Generation defaults (tunable via CLI) ----------------------------------
RALLY_MIN_SHOTS = 3
RALLY_MAX_SHOTS = 9
HIT_GAP_MIN_S = 0.35         # min time between consecutive hits in a rally
HIT_GAP_MAX_S = 1.20         # max time between consecutive hits
DEAD_TIME_MIN_S = 1.0        # min dead time between rallies (ball not visible)
DEAD_TIME_MAX_S = 3.0
ARC_HEIGHT_MIN_PX = 20.0     # gentle gravity bump on each inter-hit segment
ARC_HEIGHT_MAX_PX = 100.0
PADDLE_HEIGHT_FRAC = 0.25    # impact point height up from bbox top (0=top,1=bottom)
IMPACT_JITTER_PX = 4.0       # small noise on impact placement
SYNTH_CONFIDENCE = 1.0       # confidence written for synthetic visible frames

# On-court eligibility for placing hits (exclude adjacent-court contamination).
COURT_Y_MIN_FT = -10.0
COURT_Y_MAX_FT = 44.0
COURT_X_MIN_FT = -6.0
COURT_X_MAX_FT = 26.0
NET_Y_FT = 22.0              # for soft side-alternation


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


# --- Loaders -----------------------------------------------------------------

def load_video_meta(video_path: Path) -> Tuple[int, int, int, float]:
    if not video_path.exists():
        fail(f"video not found: {video_path}", FileNotFoundError)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"OpenCV could not open video: {video_path}", RuntimeError)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    if n < 3:
        fail(f"video has {n} frames; need >= 3", ValueError)
    return n, w, h, fps


def load_court(court_path: Path) -> dict:
    if not court_path.exists():
        fail(f"court.json not found: {court_path}", FileNotFoundError)
    with court_path.open("r", encoding="utf-8") as f:
        c = json.load(f)
    video = c.get("video", {}) or {}
    return {
        "fps": video.get("fps"),
        "frame_width": video.get("frame_width"),
        "frame_height": video.get("frame_height"),
    }


def load_players(players_path: Path) -> pd.DataFrame:
    if not players_path.exists():
        fail(f"players.parquet not found: {players_path}", FileNotFoundError)
    df = pd.read_parquet(players_path)
    need = {"frame", "track_id", "is_user", "transient",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "court_x_ft", "court_y_ft"}
    missing = need - set(df.columns)
    if missing:
        fail(f"players.parquet missing columns: {sorted(missing)}", ValueError)
    return df


def in_scope_track_ids(poses_path: Path) -> Optional[set]:
    """The tracks that survived Stage 3's strict scope filter are exactly the
    track_ids present in poses.parquet. Use them if available (cleanest set of
    real on-court players); otherwise return None and fall back to a court-
    position filter on players.parquet."""
    if not poses_path.exists():
        return None
    df = pd.read_parquet(poses_path, columns=["track_id"])
    return set(int(t) for t in df["track_id"].unique())


def load_wrists(poses_path: Path) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """(frame, track_id) -> representative wrist pixel point (mean of visible
    wrists). The ball is struck at the paddle, which is held at the hand/wrist,
    so this is the physically correct contact location to place a synthetic hit
    (and is exactly what Stage 5 associates impacts on)."""
    if not poses_path.exists():
        return {}
    cols = ["frame", "track_id", "pose_detected",
            "left_wrist_x_px", "left_wrist_y_px", "left_wrist_visibility",
            "right_wrist_x_px", "right_wrist_y_px", "right_wrist_visibility"]
    df = pd.read_parquet(poses_path, columns=cols)
    df = df[df["pose_detected"]]
    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for r in df.itertuples(index=False):
        pts = []
        if r.left_wrist_visibility >= 0.5 and not math.isnan(r.left_wrist_x_px):
            pts.append((float(r.left_wrist_x_px), float(r.left_wrist_y_px)))
        if r.right_wrist_visibility >= 0.5 and not math.isnan(r.right_wrist_x_px):
            pts.append((float(r.right_wrist_x_px), float(r.right_wrist_y_px)))
        if pts:
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            out[(int(r.frame), int(r.track_id))] = (mx, my)
    return out


# --- Eligibility -------------------------------------------------------------

def build_eligible(players: pd.DataFrame, scope_ids: Optional[set],
                   wrists: Dict[Tuple[int, int], Tuple[float, float]]) -> Dict[int, List[dict]]:
    """frame -> list of eligible hitter dicts {track_id, is_user, x, y, side}.

    A detection is eligible to be a hitter if it is non-transient, on the
    user's court (not adjacent-court contamination), and (if poses exist) a
    Stage-3 in-scope track. The impact point is the player's wrist (paddle
    contact point) when a pose is available, else paddle height in the upper
    part of the bbox.
    """
    df = players[~players["transient"]].copy()
    df = df[
        (df["court_y_ft"] >= COURT_Y_MIN_FT)
        & (df["court_y_ft"] <= COURT_Y_MAX_FT)
        & (df["court_x_ft"] >= COURT_X_MIN_FT)
        & (df["court_x_ft"] <= COURT_X_MAX_FT)
    ]
    if scope_ids is not None:
        df = df[df["track_id"].isin(scope_ids)]

    eligible: Dict[int, List[dict]] = {}
    for row in df.itertuples(index=False):
        key = (int(row.frame), int(row.track_id))
        if key in wrists:
            cx, cy = wrists[key]
        else:
            cx = 0.5 * (row.bbox_x1 + row.bbox_x2)
            cy = row.bbox_y1 + PADDLE_HEIGHT_FRAC * (row.bbox_y2 - row.bbox_y1)
        eligible.setdefault(int(row.frame), []).append({
            "track_id": int(row.track_id),
            "is_user": bool(row.is_user),
            "x": float(cx),
            "y": float(cy),
            "side": "near" if row.court_y_ft < NET_Y_FT else "far",
        })
    return eligible


# --- Rally construction ------------------------------------------------------

def pick_hitter(cands: List[dict], rng, prev: Optional[dict]) -> dict:
    """Prefer a player on the opposite side of the net from the previous hitter
    (so the ball crosses the court — bigger, more realistic direction changes),
    then any different player, then anyone."""
    if prev is not None:
        opp = [c for c in cands if c["side"] != prev["side"]]
        if opp:
            return opp[rng.integers(len(opp))]
        diff = [c for c in cands if c["track_id"] != prev["track_id"]]
        if diff:
            return diff[rng.integers(len(diff))]
    return cands[rng.integers(len(cands))]


def build_rallies(eligible: Dict[int, List[dict]], n_frames: int, fps: float,
                  rng) -> List[List[dict]]:
    """Return a list of rallies; each rally is an ordered list of hit dicts
    {frame, track_id, is_user, x, y}."""
    hit_gap = (max(1, int(HIT_GAP_MIN_S * fps)), max(2, int(HIT_GAP_MAX_S * fps)))
    dead = (max(1, int(DEAD_TIME_MIN_S * fps)), max(2, int(DEAD_TIME_MAX_S * fps)))
    frames_with_players = sorted(eligible.keys())
    if not frames_with_players:
        return []

    def jhit(frame: int, p: dict) -> dict:
        """Build a hit record, jittering the impact point by a small paddle-
        scale offset so association is non-degenerate (the ball sits just off
        the hand, not exactly on the wrist landmark)."""
        return {
            "frame": frame,
            "track_id": p["track_id"],
            "is_user": p["is_user"],
            "x": float(p["x"] + rng.normal(0.0, IMPACT_JITTER_PX)),
            "y": float(p["y"] + rng.normal(0.0, IMPACT_JITTER_PX)),
            "side": p["side"],
        }

    rallies: List[List[dict]] = []
    cursor = frames_with_players[0]
    last_frame = frames_with_players[-1]

    while cursor <= last_frame - hit_gap[0]:
        here = eligible.get(cursor, [])
        if not here:
            # advance to next frame that has an eligible player
            nxt = [f for f in frames_with_players if f > cursor]
            if not nxt:
                break
            cursor = nxt[0]
            continue

        n_shots = int(rng.integers(RALLY_MIN_SHOTS, RALLY_MAX_SHOTS + 1))
        rally: List[dict] = []
        prev = pick_hitter(here, rng, None)
        rally.append(jhit(cursor, prev))
        f = cursor
        for _ in range(n_shots - 1):
            dframes = int(rng.integers(hit_gap[0], hit_gap[1] + 1))
            f_next = f + dframes
            if f_next > last_frame:
                break
            cands = eligible.get(f_next, [])
            if not cands:
                break
            nxt = pick_hitter(cands, rng, prev)
            rally.append(jhit(f_next, nxt))
            f, prev = f_next, nxt

        if len(rally) >= 2:
            rallies.append(rally)
            cursor = f + int(rng.integers(dead[0], dead[1] + 1))
        else:
            cursor = f + hit_gap[0]

    return rallies


# --- Ball rendering ----------------------------------------------------------

def render_ball(rallies: List[List[dict]], n_frames: int, w: int, h: int,
                gap_frac: float, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render pixel-space ball positions. Returns (x, y, visible) arrays of
    length n_frames. Between two consecutive hits the ball follows a straight
    line plus a sinusoidal up-bump (gravity flavor); the meeting point at each
    hit is a sharp velocity change (the impulse Stage 5 detects). The apex of
    each bump is a GRADUAL change that Stage 5 must NOT call a shot."""
    x = np.full(n_frames, np.nan, dtype=np.float64)
    y = np.full(n_frames, np.nan, dtype=np.float64)
    vis = np.zeros(n_frames, dtype=bool)

    for rally in rallies:
        for k in range(len(rally) - 1):
            a, b = rally[k], rally[k + 1]
            f0, f1 = a["frame"], b["frame"]
            L = f1 - f0
            if L <= 0:
                continue
            dist = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            arc_h = float(rng.uniform(ARC_HEIGHT_MIN_PX, ARC_HEIGHT_MAX_PX))
            arc_h = min(arc_h, 0.35 * dist + ARC_HEIGHT_MIN_PX)  # keep apex gentle
            for i in range(L):  # fills [f0, f1)
                t = i / L
                px = a["x"] + t * (b["x"] - a["x"])
                py = a["y"] + t * (b["y"] - a["y"]) - arc_h * math.sin(math.pi * t)
                f = f0 + i
                x[f] = px
                y[f] = py
                vis[f] = True
        # final hit frame gets its own impact point
        last = rally[-1]
        x[last["frame"]] = last["x"]
        y[last["frame"]] = last["y"]
        vis[last["frame"]] = True
        # Follow-through: the last hit redirects the ball, so render a short
        # OUTGOING segment in a new direction. This gives the last hit an
        # outgoing velocity (=> a detectable impulse), then the ball goes
        # not-visible (landed / out). The FIRST hit of a rally (serve) has no
        # incoming ball and stays undetectable by a direction-change detector,
        # by design (see Stage 5 contract).
        if len(rally) >= 2:
            prev = rally[-2]
            in_dir = np.array([last["x"] - prev["x"], last["y"] - prev["y"]])
            nrm = np.linalg.norm(in_dir)
            if nrm > 1e-6:
                in_dir = in_dir / nrm
                sign = 1.0 if rng.random() < 0.5 else -1.0
                ang = math.radians(float(rng.uniform(90.0, 170.0)) * sign)
                ca, sa = math.cos(ang), math.sin(ang)
                out_dir = np.array([ca * in_dir[0] - sa * in_dir[1],
                                    sa * in_dir[0] + ca * in_dir[1]])
                ftL = int(rng.integers(8, 18))
                dist = float(rng.uniform(100.0, 400.0))
                fx, fy = last["x"] + out_dir[0] * dist, last["y"] + out_dir[1] * dist
                f0 = last["frame"]
                for i in range(1, ftL + 1):
                    f = f0 + i
                    if f >= n_frames:
                        break
                    t = i / ftL
                    x[f] = last["x"] + t * (fx - last["x"])
                    y[f] = last["y"] + t * (fy - last["y"])
                    vis[f] = True

    # clamp to frame bounds
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    # NaN where not visible (clip turned NaN into nan still; restore explicitly)
    x[~vis] = np.nan
    y[~vis] = np.nan

    # inject detection gaps (simulate Stage 4 misses) on in-flight frames only
    if gap_frac > 0:
        flight = np.where(vis)[0]
        n_drop = int(round(gap_frac * len(flight)))
        if n_drop > 0:
            drop = rng.choice(flight, size=n_drop, replace=False)
            vis[drop] = False
            x[drop] = np.nan
            y[drop] = np.nan

    return x, y, vis


# --- Output ------------------------------------------------------------------

def write_outputs(folder: Path, x, y, vis, rallies, n_frames, w, h, fps,
                  court_path: Path, video_path: Path, seed: int,
                  gap_frac: float, params: dict, force: bool) -> dict:
    out_parquet = folder / "ball.parquet"
    out_meta = folder / "ball.meta.json"
    out_truth = folder / "ball_synth_truth.json"
    for p in (out_parquet, out_meta, out_truth):
        if p.exists() and not force:
            fail(f"output exists: {p}. Use --force to overwrite.", FileExistsError)

    conf = np.full(n_frames, np.nan, dtype=np.float32)
    conf[vis] = np.float32(SYNTH_CONFIDENCE)

    df = pd.DataFrame({
        "schema_version": np.full(n_frames, SCHEMA_VERSION, dtype="int64"),
        "frame_idx": np.arange(n_frames, dtype="int64"),
        "pixel_x": x.astype("float64"),
        "pixel_y": y.astype("float64"),
        "visible": vis.astype(bool),
        "confidence": conf.astype("float32"),
        "interpolated": np.zeros(n_frames, dtype=bool),
    })[["schema_version", "frame_idx", "pixel_x", "pixel_y",
        "visible", "confidence", "interpolated"]]
    df.to_parquet(out_parquet, engine="pyarrow", index=False)

    # ground-truth hits (flatten rallies). The first hit of each rally is a
    # SERVE: no incoming ball trajectory, so a direction-change detector cannot
    # see it. Mark it so Stage 5's smoke test grades recall on the DETECTABLE
    # (non-serve) population.
    hits = []
    for rally in rallies:
        for j, hit in enumerate(rally):
            hits.append({
                "hit_id": 0,
                "frame": int(hit["frame"]),
                "track_id": int(hit["track_id"]),
                "is_user": bool(hit["is_user"]),
                "pixel_xy": [round(float(hit["x"]), 2), round(float(hit["y"]), 2)],
                "is_serve": bool(j == 0),
            })
    hits.sort(key=lambda hh: hh["frame"])
    for i, hh in enumerate(hits):
        hh["hit_id"] = i
    n_serves = sum(1 for hh in hits if hh["is_serve"])

    n_visible = int(vis.sum())
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "video_path": str(video_path),
        "video_frame_count": n_frames,
        "video_fps": fps,
        "video_width": w,
        "video_height": h,
        "court_path": str(court_path),
        "generator": {
            "tool": "tools/synth_ball.py",
            "tool_version": TOOL_VERSION,
            "seed": seed,
            "gap_frac": gap_frac,
            "params": params,
            "n_rallies": len(rallies),
            "n_hits": len(hits),
        },
        "stats": {
            "frames_visible": n_visible,
            "frames_not_visible": n_frames - n_visible,
            "ball_visible_frac": (n_visible / n_frames) if n_frames else 0.0,
        },
        "completed_at_utc": now,
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    truth = {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "synthetic": True,
        "fps": fps,
        "seed": seed,
        "n_hits": len(hits),
        "n_serves": n_serves,
        "n_detectable": len(hits) - n_serves,
        "hits": hits,
    }
    with out_truth.open("w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)
        f.write("\n")

    return {"n_rallies": len(rallies), "n_hits": len(hits),
            "frames_visible": n_visible, "ball_visible_frac": meta["stats"]["ball_visible_frac"]}


# --- Main --------------------------------------------------------------------

def run(folder: Path, seed: int, gap_frac: float, force: bool) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    video_path = folder / "video.mp4"
    court_path = folder / "court.json"
    players_path = folder / "players.parquet"
    poses_path = folder / "poses.parquet"

    n_frames, w, h, vid_fps = load_video_meta(video_path)
    court = load_court(court_path)
    fps = court["fps"] or vid_fps
    if not fps or fps <= 0:
        fail("could not determine fps from court.json or video", ValueError)
    if court["frame_width"]:
        w = int(court["frame_width"])
    if court["frame_height"]:
        h = int(court["frame_height"])

    players = load_players(players_path)
    scope_ids = in_scope_track_ids(poses_path)
    wrists = load_wrists(poses_path)

    rng = np.random.default_rng(seed)
    eligible = build_eligible(players, scope_ids, wrists)
    if not eligible:
        fail("no eligible on-court players found in players.parquet; cannot "
             "place synthetic hits", ValueError)
    rallies = build_rallies(eligible, n_frames, fps, rng)
    if not rallies:
        fail("could not build any rallies (too few eligible player frames)",
             ValueError)

    x, y, vis = render_ball(rallies, n_frames, w, h, gap_frac, rng)

    params = {
        "rally_shots_range": [RALLY_MIN_SHOTS, RALLY_MAX_SHOTS],
        "hit_gap_s_range": [HIT_GAP_MIN_S, HIT_GAP_MAX_S],
        "dead_time_s_range": [DEAD_TIME_MIN_S, DEAD_TIME_MAX_S],
        "arc_height_px_range": [ARC_HEIGHT_MIN_PX, ARC_HEIGHT_MAX_PX],
        "paddle_height_frac": PADDLE_HEIGHT_FRAC,
    }
    stats = write_outputs(folder, x, y, vis, rallies, n_frames, w, h, fps,
                          court_path, video_path, seed, gap_frac, params, force)
    return stats


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a synthetic placeholder ball.parquet")
    p.add_argument("folder", type=Path,
                   help="per-video folder with video.mp4, court.json, players.parquet")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gap-frac", type=float, default=0.0, dest="gap_frac",
                   help="fraction of in-flight frames to drop (simulate detection gaps)")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if not (0.0 <= args.gap_frac < 1.0):
        print(f"--gap-frac must be in [0, 1), got {args.gap_frac}", file=sys.stderr)
        return 2
    try:
        stats = run(args.folder, args.seed, args.gap_frac, args.force)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        print(f"synth_ball error: {e}", file=sys.stderr)
        return 1
    print(f"synth_ball: wrote ball.parquet + meta + truth to {args.folder}")
    print(f"  rallies={stats['n_rallies']} hits={stats['n_hits']} "
          f"visible_frames={stats['frames_visible']} "
          f"({stats['ball_visible_frac']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

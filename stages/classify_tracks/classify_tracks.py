"""Stage 2.5 — classify tracks into player roles.

Map ByteTrack track_ids (players.parquet) to logical roles:
user / partner / opp_left / opp_right / noise. A role is a set of track_ids over
time (ByteTrack swaps IDs on crossings). See contract for the full spec.

v1 is the VIDEO-FREE core: noise filter -> near/far side -> seed the user from
clicks -> separate user/partner with the "two people at once" simultaneity
constraint + click-anchored motion continuity + perspective-normalized height
(so matching team kit doesn't break it) -> provisional opponent L/R. Multi-region
clothing-colour matching is a documented fast-follow (helps the easy
different-colour case; height + continuity carry the same-colour case).

Usage:
    python -m stages.classify_tracks.classify_tracks data/test_clip [--force]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
STAGE_VERSION = "0.1.0"

# --- Config -----------------------------------------------------------------
NOISE_MIN_LIFETIME_S = 1.0
NOISE_COURT_Y_MIN_FT = -8.0
NOISE_COURT_Y_MAX_FT = 44.0
NOISE_MIN_IN_COURT_FRAC = 0.15
HEIGHT_PCTL = 75            # percentile of bbox height (approx standing)
HEIGHT_TOL_FT = 0.9         # height-similarity tolerance
SIMULTANEITY_MAX = 0.30     # frame-overlap with the user => can't be the user
CONTINUITY_MAX_GAP_S = 4.0  # max time to link a gap segment to a user segment
CONTINUITY_MAX_DIST_FT = 12.0
USER_ASSIGN_FLOOR = 0.45    # combined score to claim a gap segment as user
ROLES = ("user", "partner", "opp_left", "opp_right", "noise")
EPS = 1e-9


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("classify_tracks")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


def load_court(path: Path) -> dict:
    if not path.exists():
        fail(f"court.json not found: {path}", FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        c = json.load(f)
    geom = c.get("court_geometry_feet", {}) or {}
    derived = c.get("derived", {}) or {}
    video = c.get("video", {}) or {}
    width = geom.get("width_ft", 20.0)
    length = geom.get("length_ft", 44.0)
    return {
        "width_ft": float(width), "length_ft": float(length),
        "net_y": float(length) / 2.0,
        "ppf_near": derived.get("pixels_per_foot_at_near_baseline"),
        "ppf_far": derived.get("pixels_per_foot_at_far_baseline"),
        "fps": video.get("fps") or 30.0,
    }


def ppf_at(court: dict, court_y: float) -> Optional[float]:
    near, far = court["ppf_near"], court["ppf_far"]
    if near is None or far is None:
        return None
    t = max(0.0, min(1.0, court_y / court["length_ft"]))
    return near + t * (far - near)


def court_dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def summarize_tracks(df: pd.DataFrame, court: dict) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for tid, t in df.groupby("track_id"):
        t = t.sort_values("frame")
        frames = t["frame"].to_numpy()
        cy = t["court_y_ft"].to_numpy()
        cx = t["court_x_ft"].to_numpy()
        med_y = float(np.nanmedian(cy))
        med_x = float(np.nanmedian(cx))
        bbox_h = (t["bbox_y2"] - t["bbox_y1"]).to_numpy()
        h_px = float(np.nanpercentile(bbox_h, HEIGHT_PCTL)) if len(bbox_h) else np.nan
        ppf = ppf_at(court, med_y)
        height_ft = (h_px / ppf) if (ppf and ppf > EPS and not np.isnan(h_px)) else np.nan
        f0, f1 = int(frames[0]), int(frames[-1])
        out[int(tid)] = {
            "track_id": int(tid),
            "n": int(len(frames)),
            "f0": f0, "f1": f1,
            "lifetime_s": (f1 - f0 + 1) / court["fps"],
            "med_x": med_x, "med_y": med_y,
            "in_court_frac": float(t["in_court"].mean()),
            "is_user_frac": float(t["is_user"].mean()),
            "height_ft": height_ft,
            "frame_set": set(int(f) for f in frames),
            "first_pos": (float(cx[0]), float(cy[0])),
            "last_pos": (float(cx[-1]), float(cy[-1])),
        }
    return out


def height_sim(a: float, b: float) -> float:
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return 0.5  # uninformative
    return max(0.0, 1.0 - abs(a - b) / HEIGHT_TOL_FT)


def continuity_score(cand: dict, user_tracks: List[dict], fps: float) -> float:
    """How well `cand` connects (in time + court position) to a user segment's
    boundary — i.e. the user moving continuously into/out of this segment."""
    best = 0.0
    for u in user_tracks:
        # user segment ends just before candidate starts
        if u["f1"] <= cand["f0"]:
            dt_s = (cand["f0"] - u["f1"]) / fps
            if 0 <= dt_s <= CONTINUITY_MAX_GAP_S:
                d = court_dist(u["last_pos"], cand["first_pos"])
                if d <= CONTINUITY_MAX_DIST_FT:
                    best = max(best, (1 - dt_s / CONTINUITY_MAX_GAP_S)
                               * (1 - d / CONTINUITY_MAX_DIST_FT))
        # user segment starts just after candidate ends
        if u["f0"] >= cand["f1"]:
            dt_s = (u["f0"] - cand["f1"]) / fps
            if 0 <= dt_s <= CONTINUITY_MAX_GAP_S:
                d = court_dist(cand["last_pos"], u["first_pos"])
                if d <= CONTINUITY_MAX_DIST_FT:
                    best = max(best, (1 - dt_s / CONTINUITY_MAX_GAP_S)
                               * (1 - d / CONTINUITY_MAX_DIST_FT))
    return best


def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    players_path = folder / "players.parquet"
    out_path = folder / "track_roles.json"
    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.", FileExistsError)
    if not players_path.exists():
        fail(f"players.parquet not found: {players_path}", FileNotFoundError)

    court = load_court(folder / "court.json")
    fps = court["fps"]
    df = pd.read_parquet(players_path)
    need = {"frame", "track_id", "is_user", "court_x_ft", "court_y_ft",
            "in_court", "bbox_y1", "bbox_y2"}
    missing = need - set(df.columns)
    if missing:
        fail(f"players.parquet missing columns: {sorted(missing)}", ValueError)

    total_frames = int(df["frame"].max()) + 1
    tracks = summarize_tracks(df, court)
    role: Dict[int, dict] = {}  # tid -> {role, confidence, basis}

    # 1. Noise
    for tid, tr in tracks.items():
        if (tr["lifetime_s"] < NOISE_MIN_LIFETIME_S
                or not (NOISE_COURT_Y_MIN_FT <= tr["med_y"] <= NOISE_COURT_Y_MAX_FT)
                or tr["in_court_frac"] < NOISE_MIN_IN_COURT_FRAC):
            role[tid] = {"role": "noise", "confidence": 0.9, "basis": "out-of-court/short"}

    live = [tr for tid, tr in tracks.items() if tid not in role]
    near = [tr for tr in live if tr["med_y"] < court["net_y"]]
    far = [tr for tr in live if tr["med_y"] >= court["net_y"]]

    # 2. Seed the user from clicks
    seed_user = [tr for tr in near if tr["is_user_frac"] > 0.0]
    if not seed_user:
        fail("no is_user rows found; Stage 2 must resolve at least one user "
             "click before track classification can seed the user role", ValueError)
    for tr in seed_user:
        role[tr["track_id"]] = {"role": "user", "confidence": 0.95, "basis": "click"}

    # user identity: frame-weighted mean height, and the set of user-present frames
    hw = [(tr["height_ft"], tr["n"]) for tr in seed_user if not np.isnan(tr["height_ft"])]
    user_height = (sum(h * n for h, n in hw) / sum(n for _, n in hw)) if hw else np.nan
    user_frames = set()
    for tr in seed_user:
        user_frames |= tr["frame_set"]
    user_tracks = list(seed_user)

    # 3. Near non-seed tracks -> user (gap segments) or partner
    near_candidates = sorted([tr for tr in near if tr["track_id"] not in role],
                             key=lambda t: t["f0"])
    for tr in near_candidates:
        overlap = len(tr["frame_set"] & user_frames) / max(1, tr["n"])
        if overlap > SIMULTANEITY_MAX:
            # present at the same time as the user -> the partner
            role[tr["track_id"]] = {"role": "partner", "confidence": 0.7,
                                    "basis": "simultaneous-with-user"}
            continue
        # gap candidate: could be a user segment during a click gap
        cont = continuity_score(tr, user_tracks, fps)
        hsim = height_sim(tr["height_ft"], user_height)
        score = 0.6 * cont + 0.4 * hsim
        if score >= USER_ASSIGN_FLOOR:
            role[tr["track_id"]] = {"role": "user", "confidence": round(score, 3),
                                    "basis": "continuity+height"}
            user_frames |= tr["frame_set"]
            user_tracks.append(tr)
        else:
            role[tr["track_id"]] = {"role": "partner", "confidence": round(1 - score, 3),
                                    "basis": "near-not-user"}

    # 4. Opponents L/R (provisional by court_x)
    half_x = court["width_ft"] / 2.0
    for tr in far:
        side = "opp_left" if tr["med_x"] < half_x else "opp_right"
        role[tr["track_id"]] = {"role": side, "confidence": 0.5, "basis": "far-side-x"}

    # 5. Aggregate roles + stats
    roles_agg: Dict[str, dict] = {r: {"track_ids": [], "n_frames": 0} for r in ROLES if r != "noise"}
    frames_by_role: Dict[str, set] = {r: set() for r in roles_agg}
    for tid, info in role.items():
        r = info["role"]
        if r == "noise":
            continue
        roles_agg[r]["track_ids"].append(tid)
        frames_by_role[r] |= tracks[tid]["frame_set"]
    for r in roles_agg:
        roles_agg[r]["n_frames"] = len(frames_by_role[r])
        roles_agg[r]["track_ids"].sort()

    is_user_frames = set(int(f) for f in df.loc[df["is_user"], "frame"].unique())
    user_cov = len(frames_by_role["user"]) / total_frames if total_frames else 0.0
    was_is_user = len(is_user_frames) / total_frames if total_frames else 0.0

    noise_ids = sorted(tid for tid, info in role.items() if info["role"] == "noise")
    stats = {
        "n_tracks": len(tracks),
        "n_assigned": sum(1 for i in role.values() if i["role"] != "noise"),
        "n_noise": len(noise_ids),
        "user_frame_coverage": round(user_cov, 4),
        "user_frame_coverage_was_is_user": round(was_is_user, 4),
    }
    log.info(f"roles: user={len(roles_agg['user']['track_ids'])} tracks/"
             f"{roles_agg['user']['n_frames']}f, "
             f"partner={len(roles_agg['partner']['track_ids'])}, "
             f"opp_left={len(roles_agg['opp_left']['track_ids'])}, "
             f"opp_right={len(roles_agg['opp_right']['track_ids'])}, "
             f"noise={len(noise_ids)}")
    log.info(f"user coverage {was_is_user:.1%} (clicks) -> {user_cov:.1%} (roles)")

    out_doc = {
        "schema_version": SCHEMA_VERSION,
        "roles": roles_agg,
        "track_roles": {str(tid): info for tid, info in sorted(role.items())},
        "noise_track_ids": noise_ids,
        "stats": stats,
        "params": {
            "noise_min_lifetime_s": NOISE_MIN_LIFETIME_S,
            "simultaneity_max": SIMULTANEITY_MAX,
            "continuity_max_gap_s": CONTINUITY_MAX_GAP_S,
            "continuity_max_dist_ft": CONTINUITY_MAX_DIST_FT,
            "height_tol_ft": HEIGHT_TOL_FT,
            "user_assign_floor": USER_ASSIGN_FLOOR,
        },
        "warnings": [],
        "stage_version": STAGE_VERSION,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)
        f.write("\n")
    log.info(f"wrote {out_path}")
    return out_doc


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2.5 — classify tracks into roles")
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

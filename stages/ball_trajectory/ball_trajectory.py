"""Stage 5.7 — ball trajectory (physics), Phase 1: ground-anchored speed.

Never measure ball speed from the airborne ball's raw ground-homography
projection (it explodes to physically-impossible values — 261, 117 ft/s on match
rally 10). Instead anchor on the two GROUND points we can trust and derive the
horizontal motion between them:

  near anchor = the hitter's FRONT foot (on the ground, where the ball was struck)
  far anchor  = the bounce (z=0) if the shot bounced, else the NEXT hitter's foot
                (the ball was volleyed out of the air)

  horizontal_speed_ftps = distance(near, far) / airtime

This average horizontal speed separates a slow dink/drop from a fast drive without
any camera calibration. Height / apex / volley determination are Phase 2.

See stages/ball_trajectory/contract.md for the full spec.

Usage:
    python -m stages.ball_trajectory.ball_trajectory data/test_clip
    python -m stages.ball_trajectory.ball_trajectory data/test_clip --force
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
STAGE_VERSION = "0.1.0"  # Phase 1: ground-anchored horizontal speed
PHASE = 1

NET_Y_FT = 22.0
NET_CROSS_FRAMES = 3     # consecutive frames past the net before a crossing counts
# Below BOUNCE_CONF and NEXT_CONTACT_CONF: this measures the ball's approach to the net
# rather than its full flight, so it is a coarser number even where it is available.
NET_CROSS_CONF = 0.55
EPS = 1e-9

# --- Defaults (court feet / seconds — resolution & fps independent) ----------
MIN_AIRTIME_S = 0.10          # below this the frame gap is too small to trust
MAX_RANGE_FT = 44.0           # a single shot can't travel more than court length
COURT_Y_VALID_MIN = -3.0      # homography-reliable band (court is 0..44)
COURT_Y_VALID_MAX = 47.0
MAX_VOLLEY_GAP_S = 1.5        # a volley's next contact is soon; a new serve is later
BOUNCE_CONF = 0.85
NEXT_CONTACT_CONF = 0.6
ANKLE_VIS_MIN = 0.3


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("ball_trajectory")
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
    homog = c.get("homography", {}) or {}
    if "image_to_court" not in homog:
        fail("court.json.homography missing image_to_court", ValueError)
    M = np.array(homog["image_to_court"], dtype=np.float64)
    if M.shape != (3, 3):
        fail(f"image_to_court must be 3x3, got {M.shape}", ValueError)
    video = c.get("video", {}) or {}
    return {"image_to_court": M, "fps": video.get("fps")}


def index_players(path: Path) -> Dict[Tuple[int, int], dict]:
    if not path.exists():
        fail(f"players.parquet not found: {path}", FileNotFoundError)
    df = pd.read_parquet(path)
    has_rel = "court_pos_reliable" in df.columns
    out: Dict[Tuple[int, int], dict] = {}
    for r in df.itertuples(index=False):
        out[(int(r.frame), int(r.track_id))] = {
            "cx": float(r.court_x_ft), "cy": float(r.court_y_ft),
            "reliable": bool(getattr(r, "court_pos_reliable", True)) if has_rel else True,
        }
    return out


def index_pose_ankles(path: Path) -> Dict[Tuple[int, int], dict]:
    if not path.exists():
        return {}
    cols = ["frame", "track_id", "pose_detected",
            "left_ankle_x_px", "left_ankle_y_px", "left_ankle_visibility",
            "right_ankle_x_px", "right_ankle_y_px", "right_ankle_visibility"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["pose_detected"]]
    out: Dict[Tuple[int, int], dict] = {}
    for r in df.itertuples(index=False):
        out[(int(r.frame), int(r.track_id))] = {
            "lax": r.left_ankle_x_px, "lay": r.left_ankle_y_px, "lav": r.left_ankle_visibility,
            "rax": r.right_ankle_x_px, "ray": r.right_ankle_y_px, "rav": r.right_ankle_visibility,
        }
    return out


# --- Geometry ----------------------------------------------------------------

def project_to_court(M: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    v = M @ np.array([px, py, 1.0])
    if abs(v[2]) < EPS or not np.all(np.isfinite(v)):
        return float("nan"), float("nan")
    return float(v[0] / v[2]), float(v[1] / v[2])


def front_foot_court_xy(M: np.ndarray, pose: Optional[dict],
                        prow: Optional[dict]) -> Optional[Tuple[float, float]]:
    """Court (x, y) of the hitter's FRONT foot = the ankle nearest the net.
    On the near side the bbox-bottom is the REAR foot (reads too deep), so we
    prefer the ankle nearest the net; seeded with the bbox foot so it can never
    read DEEPER than that (protects the far side). Falls back to the bbox foot."""
    cands: List[Tuple[float, float]] = []
    if prow is not None and math.isfinite(prow["cx"]) and math.isfinite(prow["cy"]):
        cands.append((prow["cx"], prow["cy"]))
    if pose is not None:
        for xk, yk, vk in (("lax", "lay", "lav"), ("rax", "ray", "rav")):
            px, py, v = pose.get(xk), pose.get(yk), pose.get(vk)
            if px is None or py is None or (v is not None and v < ANKLE_VIS_MIN):
                continue
            cx, cy = project_to_court(M, float(px), float(py))
            if math.isfinite(cx) and math.isfinite(cy):
                cands.append((cx, cy))
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c[1] - NET_Y_FT))


def y_in_band(y: float) -> bool:
    return COURT_Y_VALID_MIN <= y <= COURT_Y_VALID_MAX


NET_TOL_FT = 2.0  # a landing within this of the net counts as "crossed"


def crosses_net(near_y: float, far_y: float) -> bool:
    """Every legit shot sends the ball over the net, so its landing / the next
    contact must be on the OPPOSITE side of the net line from the hitter (or right
    at the net). A 'landing' on the hitter's own side is a mis-assigned bounce."""
    if abs(far_y - NET_Y_FT) <= NET_TOL_FT:
        return True
    return (near_y - NET_Y_FT) * (far_y - NET_Y_FT) < 0


# WHY A THIRD OF SHOTS CARRY NO PHYSICAL SPEED (measured 2026-09-02, 202 shots on the two
# reviewed clips). The speed is range/airtime, so it needs an ANCHOR -- the shot's landing
# bounce, or the next contact -- and 59 shots get neither. That is not one fault:
#
#     27  are rally-ENDERS or between-point contacts. Nothing came back, so there is no
#         next contact and usually no in-court landing. Unmeasurable by definition, not a
#         failure, and the classifier should not be waiting on a number that cannot exist.
#     32  are mid-rally shots that should have had one. Their landing bounce was either
#         never detected or was rejected here for not crossing the net, AND the next
#         contact was unusable -- most often because it is on the SAME side, which in a
#         rally means a shot between them was missed.
#
# A further 19 shots have a speed that is deliberately distrusted (a volley with a bounce
# anchor) and 17 fall under TRAJ_SPEED_CONF_MIN in Stage 6.
#
# So the mid-rally gap is downstream of missed shots and missed bounces rather than of the
# arithmetic here, and it matters: with the pixel-derived speed mixed in, drives read 30.7
# ft/s against 24.5 for everything else, while on the physical speed alone they read 37.0
# against 25.0. The fallback dilutes a real signal.
#
# One threshold here is worth revisiting on its own: max_volley_gap_s is 1.5s, but the
# operator's flight times put dinks at a median 1.10s with p75 1.55 and lobs at 1.57. A
# lob's next contact is therefore outside the window by construction.


def anchor_ok(near: Tuple[float, float], far: Tuple[float, float],
              rng: float, max_range_ft: float) -> bool:
    """Physical sanity: the range can't exceed the court length, and the ball must
    have crossed the net. A violation means the anchor is wrong, not slow/fast."""
    return rng <= max_range_ft and crosses_net(near[1], far[1])


# --- Core --------------------------------------------------------------------

def build_landing_index(bounces: List[dict]) -> Dict[int, dict]:
    """shot_id -> its landing bounce (the FIRST bounce after that shot). A double
    bounce between two shots keeps the earlier one (the ball's landing)."""
    out: Dict[int, dict] = {}
    for b in bounces:
        bs = b.get("between_shots") or [None, None]
        prev = bs[0]
        if prev is None:
            continue
        cxy = b.get("court_xy_ft") or [None, None]
        if cxy[0] is None or cxy[1] is None:
            continue
        prev = int(prev)
        if prev not in out or b["frame"] < out[prev]["frame"]:
            out[prev] = b
    return out


def net_crossing_speed(ball_y: Dict[int, float], f: int, side: str, hitter_y: float,
                       fps: float, look_s: float = 2.5) -> Optional[float]:
    """Distance to the net over the time the ball takes to get across it.

    The operator's idea, and it is the right shape for this system. The anchored speed
    needs the shot's landing bounce or the next contact, and 59 of 202 shots have neither
    -- a rally-ender has nothing coming back, a missed bounce leaves nothing to measure
    to. Every shot that crosses the net crosses the net.

    Both inputs are the reliable kind. The hitter's court position is ground truth from
    players.parquet, and WHEN the ball gets across is a sustained, relative question about
    the 3-D track, which is the class this reconstruction answers well -- unlike absolute
    per-frame position, which it gets wrong about 30% of the time.

    The crossing must be SUSTAINED. Taking the first frame that reads across dates it
    frames too early, and since the distance is fixed that inflates the speed: dinks came
    out at 96 ft/s and one shot at 396. Requiring the ball to stay across gives dink 25.6,
    drop 36.9, drive 57.8, lob 16.1 -- the right order, and a wider drive-vs-rest margin
    than the anchored speed manages (1.86x against 1.48x) on 99% of shots against 63%.
    """
    if not ball_y or side not in ("near", "far") or hitter_y is None:
        return None
    run, first = 0, None
    for g in range(f + 1, f + int(look_s * fps)):
        v = ball_y.get(g)
        if v is None:
            continue
        past = (v > NET_Y_FT) if side == "near" else (v < NET_Y_FT)
        if past:
            if run == 0:
                first = g
            run += 1
            if run >= NET_CROSS_FRAMES:
                dt = (first - f) / fps
                return abs(float(hitter_y) - NET_Y_FT) / dt if dt > 0 else None
        else:
            run = 0
    return None


def compute(shots: List[dict], landing: Dict[int, dict],
            players: Dict[Tuple[int, int], dict],
            poses: Dict[Tuple[int, int], dict],
            M: np.ndarray, fps: float, params: dict, log: logging.Logger,
            ball_y: Optional[Dict[int, float]] = None
            ) -> Tuple[List[dict], dict]:
    max_gap_frames = params["max_volley_gap_s"] * fps
    min_airtime = params["min_airtime_s"]
    by_index = {i: s for i, s in enumerate(shots)}
    n_bounce = n_next = n_none = n_net = 0
    results: List[dict] = []

    for i, s in enumerate(shots):
        sid = int(s["shot_id"])
        f = int(s["frame"])
        tid = int(s["track_id"])
        near = front_foot_court_xy(M, poses.get((f, tid)), players.get((f, tid)))

        # Build candidate anchors in priority order and pick the FIRST that passes
        # physical sanity (range <= court length, ball crossed the net, airtime big
        # enough). This lets a mis-assigned bounce fall back to the next contact.
        cands: List[Tuple[str, Tuple[float, float], int]] = []
        b = landing.get(sid)
        if b is not None:
            cands.append(("bounce",
                          (float(b["court_xy_ft"][0]), float(b["court_xy_ft"][1])),
                          int(b["frame"])))
        nxt = by_index.get(i + 1)
        if (nxt is not None and not bool(nxt.get("is_serve"))
                and 0 < (int(nxt["frame"]) - f) <= max_gap_frames):
            nf, ntid = int(nxt["frame"]), int(nxt["track_id"])
            nfar = front_foot_court_xy(M, poses.get((nf, ntid)), players.get((nf, ntid)))
            if nfar is not None:
                cands.append(("next_contact", nfar, nf))

        anchor_type, far, f_far = "none", None, None
        speed = rng = airtime = None
        conf = 0.0
        if near is not None:
            for atype, cfar, cff in cands:
                cair = (cff - f) / fps
                crng = math.hypot(cfar[0] - near[0], cfar[1] - near[1])
                if cair < min_airtime or not anchor_ok(near, cfar, crng, params["max_range_ft"]):
                    continue
                anchor_type, far, f_far = atype, cfar, cff
                airtime, rng = cair, crng
                speed = rng / airtime
                conf = BOUNCE_CONF if atype == "bounce" else NEXT_CONTACT_CONF
                if not (y_in_band(near[1]) and y_in_band(far[1])):
                    conf *= 0.5
                if airtime < 0.15:
                    conf *= 0.5
                break

        if anchor_type == "none" and near is not None:
            ns = net_crossing_speed(ball_y or {}, f, s.get("hitter_side"), near[1], fps)
            if ns is not None:
                anchor_type, speed, conf = "net_crossing", ns, NET_CROSS_CONF

        if anchor_type == "bounce":
            n_bounce += 1
        elif anchor_type == "next_contact":
            n_next += 1
        elif anchor_type == "net_crossing":
            n_net += 1
        else:
            n_none += 1

        results.append({
            "shot_id": sid,
            "horizontal_speed_ftps": round(speed, 2) if speed is not None else None,
            "range_ft": round(rng, 2) if rng is not None else None,
            "airtime_s": round(airtime, 3) if airtime is not None else None,
            "anchor_type": anchor_type,
            "hitter_court_xy_ft": [round(near[0], 2), round(near[1], 2)] if near else None,
            "anchor_court_xy_ft": [round(far[0], 2), round(far[1], 2)] if far else None,
            "confidence": round(conf, 3),
        })

    stats = {"shots": len(shots), "anchor_bounce": n_bounce,
             "anchor_next_contact": n_next, "anchor_net_crossing": n_net, "anchor_none": n_none}
    log.info(f"trajectory for {len(shots)} shots; anchors: bounce={n_bounce} "
             f"next_contact={n_next} net_crossing={n_net} none={n_none}")
    return results, stats


def run(folder: Path, args, log: logging.Logger) -> dict:
    folder = Path(folder)
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    out_path = folder / "trajectory.json"
    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.", FileExistsError)

    court = load_court(folder / "court.json")
    fps = court["fps"]
    if fps is None or fps <= 0:
        fail("could not determine fps from court.json", ValueError)
    fps = float(fps)

    shots_doc = load_json(folder / "shots.json")
    shots = shots_doc.get("shots", shots_doc) if isinstance(shots_doc, dict) else shots_doc
    bounces_doc = load_json(folder / "bounces.json")
    bounces = bounces_doc.get("bounces", bounces_doc) if isinstance(bounces_doc, dict) else bounces_doc
    players = index_players(folder / "players.parquet")
    poses = index_pose_ankles(folder / "poses.parquet")

    params = {
        "fps": fps,
        "min_airtime_s": float(args.min_airtime_s),
        "max_range_ft": float(args.max_range_ft),
        "max_volley_gap_s": float(args.max_volley_gap_s),
        "court_y_valid_min": COURT_Y_VALID_MIN,
        "court_y_valid_max": COURT_Y_VALID_MAX,
        "bounce_conf": BOUNCE_CONF,
        "next_contact_conf": NEXT_CONTACT_CONF,
    }

    landing = build_landing_index(bounces)
    _b3 = folder / "ball_3d.parquet"
    ball_y = {}
    if _b3.exists():
        _df = pd.read_parquet(_b3, columns=["frame", "court_y_ft"])
        ball_y = {int(f): float(v) for f, v in zip(_df["frame"], _df["court_y_ft"])
                  if v == v}
    results, stats = compute(shots, landing, players, poses,
                             court["image_to_court"], fps, params, log, ball_y)

    out = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "source_shots": str(folder / "shots.json"),
        "source_bounces": str(folder / "bounces.json"),
        "params": params,
        "shots": results,
        "stats": stats,
        "stage_version": STAGE_VERSION,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    log.info(f"wrote {out_path}")
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 5.7 — ball trajectory (Phase 1)")
    p.add_argument("folder", type=Path,
                   help="per-video folder with court.json, shots.json, bounces.json, "
                        "players.parquet, poses.parquet")
    p.add_argument("--force", action="store_true")
    p.add_argument("--min-airtime-s", type=float, default=MIN_AIRTIME_S,
                   dest="min_airtime_s")
    p.add_argument("--max-range-ft", type=float, default=MAX_RANGE_FT,
                   dest="max_range_ft")
    p.add_argument("--max-volley-gap-s", type=float, default=MAX_VOLLEY_GAP_S,
                   dest="max_volley_gap_s")
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

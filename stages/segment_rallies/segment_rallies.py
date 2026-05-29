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
STAGE_VERSION = "0.1.0"

# --- Config (matches contract) ----------------------------------------------
SERVE_FAULT_MAX_FRAMES = 60       # quick-next-serve = serve fault (~2s @ 30fps)
NET_Y_FT = 22.0                   # net line in court coordinates
KITCHEN_DEPTH_FT = 7.0            # kitchen extends 7 ft from net (each side)

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

def segment_rallies(shots: List[dict]) -> Tuple[List[List[dict]], List[dict]]:
    """Split the shot stream into rallies by is_serve. Returns
    (rally_shot_lists, pre_rally_shots). Each rally is a list of shot dicts,
    starting with a serve. pre_rally_shots are shots before the first serve."""
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


# --- End-reason classification ----------------------------------------------

def classify_rally(rally_shots: List[dict], bounces: List[dict],
                    next_rally_serve_frame: Optional[int]
                    ) -> Tuple[str, float, Optional[int], dict]:
    """Returns (end_reason, confidence, ending_bounce_id, end_signals).
    Implements the rule table in the contract: serve-fault > double-bounce >
    net-or-short > ball-out > ball-not-returned > ball-off-frame > unknown."""
    last_shot = rally_shots[-1]
    last_shot_id = int(last_shot["shot_id"])
    last_frame = int(last_shot["frame"])
    n_shots = len(rally_shots)
    serve = rally_shots[0]
    serve_court = serve.get("impact_court_xy_ft") or [None, None]
    server_side = side_of_net(serve_court[1] if serve_court else None)
    last_court = last_shot.get("impact_court_xy_ft") or [None, None]
    hitter_side = side_of_net(last_court[1] if last_court else None)

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
                <= SERVE_FAULT_MAX_FRAMES):
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

    # Rule 6: ball-off-frame.
    if n_post == 0 and frames_to_next_serve is not None \
            and frames_to_next_serve > 0:
        return ("ball-off-frame", 0.5, None, end_signals)

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

    shots = sorted(classified.get("shots", []), key=lambda s: int(s["frame"]))
    bounces = sorted(bounces_doc.get("bounces", []),
                     key=lambda b: int(b["frame"]))

    rally_groups, pre_rally = segment_rallies(shots)
    if not rally_groups:
        log.warning("no serves found; emitting empty rallies list")

    out_rallies: List[dict] = []
    for ri, rally_shots in enumerate(rally_groups):
        next_serve_frame = (int(rally_groups[ri + 1][0]["frame"])
                            if ri + 1 < len(rally_groups) else None)
        end_reason, conf, ending_bid, signals = classify_rally(
            rally_shots, bounces, next_serve_frame)

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
    if pre_rally:
        warnings.append(f"{len(pre_rally)} shot(s) preceded the first serve "
                        f"and were dropped (unassigned). "
                        f"Shot_ids: {[int(s['shot_id']) for s in pre_rally]}")
    if not out_rallies:
        warnings.append("no rallies emitted (no is_serve shots in classified.json)")

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
            "serve_fault_max_frames": SERVE_FAULT_MAX_FRAMES,
            "net_y_ft": NET_Y_FT,
            "kitchen_depth_ft": KITCHEN_DEPTH_FT,
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

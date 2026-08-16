"""Stage 7.9 — Aggregate: union N analysed videos into one virtual session.

Contract: stages/aggregate/contract.md. The operator's specification is
"work the same as if the multiple videos were all in one video", which is testable:

    compute_metrics(union(A, B)) == compute_metrics(concat_video(A, B))

This stage produces NO statistics. It writes streams with byte-identical schemas to a
real session, so Stages 8-11 run on the result unchanged — a cumulative report is the
existing report over a bigger input. That is why aggregation sits BELOW Stage 8 rather
than combining metrics.json: a median of medians is not the median, and averaging the
confidence envelope would make two 6-rally videos read as LESS certain than they are.

Usage:
    python -m stages.aggregate.aggregate --out data/_collections/c1 \
        --member data/pb_2min --member data/pb_3min
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1
STAGE_VERSION = "0.1.0"
ROLES = ("user", "partner", "opp_a", "opp_b")


def fail(msg: str):
    raise RuntimeError(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("aggregate")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


def _read(folder: Path, name: str) -> dict:
    p = folder / name
    if not p.exists():
        fail(f"{folder.name} is missing {name} — run the pipeline on it first")
    return json.loads(p.read_text(encoding="utf-8"))


def member_span_sec(cl: dict, ra: dict, bo: dict) -> float:
    """Playing time contributed by one member. match_span_sec is the SUM of these, not
    wall-clock between videos: cumulative stats describe time on court, and a literal
    concatenation of the files would agree."""
    ts = ([s["t_sec"] for s in cl["shots"]] +
          [b["t_sec"] for b in bo["bounces"]] +
          [r["end_t_sec"] for r in ra["rallies"]])
    return float(max(ts) - min(ts)) if ts else 0.0


NET_Y_FT = 22.0
POSE_ANKLE_MIN_VIS = 0.3


def _project_front_foot(q, court: dict) -> None:
    """Add front_foot_court_x_ft / _y_ft to a member's poses, using ITS court.

    Mirrors compute_metrics.pose_front_foot: the front foot is the visible ankle nearest
    the net, because a player's bbox bottom is the BACK foot for a net-facing player and
    reads several feet too deep. Doing it here, once per member, is what makes the union
    venue-independent.
    """
    import numpy as np

    H = (court.get("homography", {}) or {}).get("image_to_court")
    need = {"left_ankle_x_px", "right_ankle_x_px", "left_ankle_visibility"}
    if H is None or not need <= set(q.columns):
        q["front_foot_court_x_ft"] = float("nan")
        q["front_foot_court_y_ft"] = float("nan")
        return
    M = np.asarray(H, dtype=np.float64)

    def proj(xc, yc):
        pts = np.column_stack([q[xc].to_numpy(dtype=np.float64),
                               q[yc].to_numpy(dtype=np.float64), np.ones(len(q))])
        o = pts @ M.T
        return o[:, 0] / o[:, 2], o[:, 1] / o[:, 2]

    lx, ly = proj("left_ankle_x_px", "left_ankle_y_px")
    rx, ry = proj("right_ankle_x_px", "right_ankle_y_px")
    ok_l = q["left_ankle_visibility"].to_numpy(dtype=np.float64) >= POSE_ANKLE_MIN_VIS
    ok_r = q["right_ankle_visibility"].to_numpy(dtype=np.float64) >= POSE_ANKLE_MIN_VIS
    dl = np.where(ok_l, np.abs(ly - NET_Y_FT), np.inf)
    dr = np.where(ok_r, np.abs(ry - NET_Y_FT), np.inf)
    use_left = dl <= dr
    fx, fy = np.where(use_left, lx, rx), np.where(use_left, ly, ry)
    bad = ~np.isfinite(fx) | ~np.isfinite(fy) | (np.minimum(dl, dr) == np.inf)
    q["front_foot_court_x_ft"] = np.where(bad, np.nan, fx)
    q["front_foot_court_y_ft"] = np.where(bad, np.nan, fy)


def union(members: list[Path], log: logging.Logger,
          pool_opponents: bool = True) -> dict:
    """Concatenate every stream with IDs renumbered and time offset.

    Every cross-reference is rewritten with the IDs. A dangling reference afterwards is a
    hard failure, not a warning: a silently broken shot_ids link would surface as a rally
    with missing shots in the cumulative report, which is indistinguishable from a real
    detection gap.
    """
    out_shots, out_rallies, out_bounces, out_players, out_poses = [], [], [], [], []
    missing_poses: list[str] = []
    role_of_track: dict[int, str] = {}
    track_conf: dict[int, float] = {}
    shot_off = rally_off = bounce_off = track_off = frame_off = 0
    t_off = 0.0
    synthetic = False
    gated: set[str] = set()
    provenance = []

    for vi, folder in enumerate(members):
        cl = _read(folder, "classified.json")
        ra = _read(folder, "rallies.json")
        bo = _read(folder, "bounces.json")
        tr = _read(folder, "track_roles.json")
        pl = pd.read_parquet(folder / "players.parquet")

        # --- reliability propagates WORST-CASE, never averaged. A metric untrustworthy
        # in one member is untrustworthy in the total; the alternative presents a
        # contaminated number as clean, which is the most damaging thing here.
        if str(cl.get("ball_source", "")).lower() != "real":
            synthetic = True
        mt = folder / "metrics.json"
        if mt.exists():
            rel = json.loads(mt.read_text(encoding="utf-8")).get("reliability", {})
            gated.update(rel.get("synthetic_gated", []))
            if rel.get("synthetic_ball"):
                synthetic = True

        sid = {s["shot_id"]: s["shot_id"] + shot_off for s in cl["shots"]}
        bid = {b["bounce_id"]: b["bounce_id"] + bounce_off for b in bo["bounces"]}
        tid = {int(t): int(t) + track_off for t in tr["track_roles"]}

        for s in cl["shots"]:
            m = dict(s)
            m["shot_id"] = sid[s["shot_id"]]
            m["video_id"], m["video_frame"] = vi, s["frame"]
            m["frame"] = s["frame"] + frame_off
            m["t_sec"] = round(s["t_sec"] + t_off, 4)
            if s.get("track_id") is not None:
                m["track_id"] = tid.get(int(s["track_id"]), int(s["track_id"]) + track_off)
            out_shots.append(m)

        for r in ra["rallies"]:
            m = dict(r)
            m["rally_id"] = r["rally_id"] + rally_off
            m["video_id"] = vi
            m["shot_ids"] = [sid[x] for x in r["shot_ids"]]
            if r.get("serve_shot_id") is not None:
                m["serve_shot_id"] = sid[r["serve_shot_id"]]
            if r.get("server_track_id") is not None:
                m["server_track_id"] = tid.get(int(r["server_track_id"]),
                                               int(r["server_track_id"]) + track_off)
            for k in ("start_frame", "end_frame"):
                m[k] = r[k] + frame_off
            for k in ("start_t_sec", "end_t_sec"):
                m[k] = round(r[k] + t_off, 4)
            out_rallies.append(m)

        for b in bo["bounces"]:
            m = dict(b)
            m["bounce_id"] = bid[b["bounce_id"]]
            m["video_id"], m["video_frame"] = vi, b["frame"]
            m["frame"] = b["frame"] + frame_off
            m["t_sec"] = round(b["t_sec"] + t_off, 4)
            m["between_shots"] = [None if x is None else sid[x] for x in b["between_shots"]]
            out_bounces.append(m)

        p = pl.copy()
        p["video_id"] = vi
        p["video_frame"] = p["frame"]
        p["frame"] = p["frame"] + frame_off
        p["track_id"] = p["track_id"].map(lambda t: tid.get(int(t), int(t) + track_off))
        p["t_sec"] = p["t_sec"] + t_off
        out_players.append(p)

        # Poses MUST travel with the players. Stage 8 uses the pose front-foot for court
        # position and falls back to the bbox foot when poses are absent — silently, with
        # only a warning — which UNDER-COUNTS kitchen time. Omitting them made a
        # cumulative rating (3.23) come out below both of its inputs (4.14 and 3.78),
        # because strategy is the heaviest category. They also carry ready-position and
        # knee-bend, so without them the report loses its body-mechanics findings.
        pp = folder / "poses.parquet"
        if pp.exists():
            q = pd.read_parquet(pp)
            q["video_id"] = vi
            q["video_frame"] = q["frame"]
            q["frame"] = q["frame"] + frame_off
            q["track_id"] = q["track_id"].map(lambda t: tid.get(int(t), int(t) + track_off))
            if "t_sec" in q.columns:
                q["t_sec"] = q["t_sec"] + t_off
            # Pose landmarks are PIXELS, so they only mean something alongside the
            # homography of the video they came from. A union spans venues and can carry
            # only one court.json, so project here, per member, with that member's own
            # court — otherwise every venue after the first lands off-court.
            _project_front_foot(q, _read(folder, "court.json"))
            out_poses.append(q)
        else:
            missing_poses.append(folder.name)

        for t, info in tr["track_roles"].items():
            role = info["role"]
            # Contract D1: cumulatively, opp_a and opp_b are not the same two humans from
            # video to video, so keeping them apart is meaningless precision. Pool them
            # HERE, in the stream, so Stage 8 computes one opponents bucket natively --
            # merging two per-role metric blocks afterwards would mean re-deriving means
            # and confidences by hand, the exact arithmetic this stage exists to avoid.
            # `partner` is NOT pooled into it: the partner is on the user's team, and
            # team.near / team.far and error attribution are built on that split.
            if pool_opponents and role == "opp_b":
                role = "opp_a"
            role_of_track[tid[int(t)]] = role
            track_conf[tid[int(t)]] = info.get("confidence", 0.0)

        span = member_span_sec(cl, ra, bo)
        provenance.append({"video_id": vi, "session": folder.name, "span_sec": round(span, 3),
                           "n_shots": len(cl["shots"]), "n_rallies": len(ra["rallies"]),
                           "n_bounces": len(bo["bounces"]), "fps": cl.get("fps"),
                           "ball_source": cl.get("ball_source")})
        log.info("%s: %d shots, %d rallies, %d bounces, %.1fs",
                 folder.name, len(cl["shots"]), len(ra["rallies"]), len(bo["bounces"]), span)

        shot_off = max(sid.values(), default=shot_off - 1) + 1
        rally_off += len(ra["rallies"])
        bounce_off = max(bid.values(), default=bounce_off - 1) + 1
        track_off = max(tid.values(), default=track_off - 1) + 1
        t_off += span
        # +1 so no frame index is shared between members
        frame_off = int(max(p["frame"].max(),
                            max((x["frame"] for x in out_shots), default=0),
                            max((x["frame"] for x in out_bounces), default=0))) + 1

    # --- integrity: every cross-reference must resolve ---
    known_shots = {s["shot_id"] for s in out_shots}
    for r in out_rallies:
        missing = [x for x in r["shot_ids"] if x not in known_shots]
        if missing:
            fail(f"rally {r['rally_id']} references unknown shots {missing}")
    for b in out_bounces:
        for x in b["between_shots"]:
            if x is not None and x not in known_shots:
                fail(f"bounce {b['bounce_id']} references unknown shot {x}")
    for name, rows in (("shots", out_shots), ("bounces", out_bounces)):
        frames = [r["frame"] for r in rows]
        if len(frames) != len(set(frames)):
            fail(f"{name}: frame collision after union — offsets are wrong")
    players = pd.concat(out_players, ignore_index=True)
    unknown = set(players["track_id"].unique()) - set(role_of_track)
    if unknown:
        fail(f"{len(unknown)} track ids in players.parquet have no role")

    poses = pd.concat(out_poses, ignore_index=True) if out_poses else None
    return {"shots": out_shots, "rallies": out_rallies, "bounces": out_bounces,
            "players": players, "poses": poses, "missing_poses": missing_poses, "role_of_track": role_of_track, "track_conf": track_conf,
            "synthetic": synthetic, "gated": sorted(gated), "provenance": provenance,
            "span_sec": t_off, "pooled_opponents": pool_opponents}


def write(u: dict, members: list[Path], out: Path, log: logging.Logger) -> None:
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    src = _read(members[0], "classified.json")
    fps = src.get("fps")
    ball_source = "synthetic" if u["synthetic"] else "real"

    def base(extra: dict) -> dict:
        d = {"schema_version": src.get("schema_version", 1), "fps": fps,
             "ball_source": ball_source, "stage_version": STAGE_VERSION,
             "completed_at_utc": now, "aggregated_from": [m.name for m in members],
             "warnings": []}
        d.update(extra)
        return d

    (out / "classified.json").write_text(json.dumps(base({
        "source_shots": "aggregate", "params": src.get("params", {}),
        "shots": u["shots"], "stats": {"n_shots": len(u["shots"])}}), indent=1),
        encoding="utf-8")
    (out / "rallies.json").write_text(json.dumps(base({
        "source_classified": "aggregate", "source_bounces": "aggregate",
        "params": _read(members[0], "rallies.json").get("params", {}),
        "rallies": u["rallies"], "stats": {"n_rallies": len(u["rallies"])}}), indent=1),
        encoding="utf-8")
    (out / "bounces.json").write_text(json.dumps(base({
        "source_shots": "aggregate",
        "params": _read(members[0], "bounces.json").get("params", {}),
        "bounces": u["bounces"], "stats": {"n_bounces": len(u["bounces"])}}), indent=1),
        encoding="utf-8")
    u["players"].to_parquet(out / "players.parquet", index=False)
    if u.get("poses") is not None:
        u["poses"].to_parquet(out / "poses.parquet", index=False)
    else:
        (out / "poses.parquet").unlink(missing_ok=True)

    # Roles carry through; TRACK ids do not. Note the union keeps all four roles rather
    # than pooling opp_a/opp_b (contract D1) -- pooling happens at REPORT time, so no
    # information is destroyed here and Stage 8 needs no change.
    roles = {r: {"track_ids": sorted(t for t, v in u["role_of_track"].items() if v == r)}
             for r in ROLES}
    for r, d in roles.items():
        d["n_frames"] = int((u["players"]["track_id"].isin(d["track_ids"])).sum())
    (out / "track_roles.json").write_text(json.dumps({
        "schema_version": 1,
        "roles": roles,
        "track_roles": {str(t): {"role": v, "confidence": u["track_conf"].get(t, 0.0),
                                 "basis": "aggregated"}
                        for t, v in sorted(u["role_of_track"].items())},
        "noise_track_ids": sorted(t for t, v in u["role_of_track"].items() if v == "noise"),
        "stats": {"n_tracks": len(u["role_of_track"])},
        "params": {}, "warnings": [], "stage_version": STAGE_VERSION,
        "completed_at_utc": now}, indent=1), encoding="utf-8")

    for f in ("court.json", "court_zones.json", "roster.json"):
        p = members[0] / f
        if p.exists():
            (out / f).write_bytes(p.read_bytes())

    # NOT collection.json -- app/collections.py owns that name for the membership
    # record in this same folder. Two writers, one filename, is a silent corruption
    # waiting for a rebuild to be interrupted.
    (out / "union.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "built_at_utc": now,
        "members": u["provenance"], "total_span_sec": round(u["span_sec"], 3),
        "ball_source": ball_source, "synthetic_gated": u["gated"],
        "pooled_opponents": u["pooled_opponents"],
        "stage_version": STAGE_VERSION}, indent=1), encoding="utf-8")

    log.info("union -> %s: %d shots, %d rallies, %d bounces, %d players rows, %.1fs",
             out, len(u["shots"]), len(u["rallies"]), len(u["bounces"]),
             len(u["players"]), u["span_sec"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", action="append", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-pool-opponents", action="store_true",
                    help="keep opp_a/opp_b separate (they are different people per "
                         "video, so the default pools them -- contract D1)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    log = setup_logging(a.log_level)
    try:
        u = union(a.member, log, pool_opponents=not a.no_pool_opponents)
        write(u, a.member, a.out, log)
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

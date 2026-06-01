"""Stage 11 — render annotated video.

Pure consumer: draws what upstream stages decided onto the actual source video,
and emits a synchronized timeline.json + standalone heatmap PNGs. Recomputes
nothing — rally end_reasons, shot types, roles, rating, plan are all copied from
the upstream JSON; the only geometry here is DRAWING (projecting known court
coordinates through the known homography, colormapping known grids).

When ball_source == 'synthetic', a persistent watermark is burned in so a
rendered demo is never mistaken for validated output.

See stages/render/contract.md for the full spec.

Usage:
    python -m stages.render.render data/test_clip
    python -m stages.render.render data/test_clip --force --max-seconds 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
STAGE_VERSION = "0.1.0"

# --- Config (matches contract) ----------------------------------------------
TRAIL_FRAMES = 10
SHOT_MARKER_FRAMES = 6
BOUNCE_MARKER_FRAMES = 6
MINIMAP_W_PX = 200
VIDEO_FOURCC = "mp4v"

# BGR colors
ROLE_COLORS = {
    "user": (0, 200, 0), "partner": (220, 130, 0),
    "opp_left": (0, 0, 220), "opp_right": (0, 150, 255),
    "noise": (150, 150, 150),
}
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COURT_COLOR = (0, 255, 255)
NET_COLOR = (200, 200, 200)
BALL_COLOR = (0, 255, 255)
IN_COLOR = (0, 220, 0)
OUT_COLOR = (0, 0, 230)
SYNTH_COLOR = (0, 165, 255)

COURT_LEN_FT = 44.0
COURT_WID_FT = 20.0
NET_Y_FT = 22.0
KITCHEN_Y = (15.0, 29.0)   # net +/- 7 ft

POSE_EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

EVENT_TYPE_RANK = {"rally_start": 0, "shot": 1, "bounce": 2, "rally_end": 3}


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("render")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


def load_json_opt(path: Path, log: logging.Logger, label: str) -> Optional[dict]:
    if not path.exists():
        log.warning(f"optional input missing: {path.name} ({label} layer skipped)")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not read {path.name}: {e} ({label} layer skipped)")
        return None


def load_parquet_opt(path: Path, log: logging.Logger, label: str
                     ) -> Optional[pd.DataFrame]:
    if not path.exists():
        log.warning(f"optional input missing: {path.name} ({label} layer skipped)")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not read {path.name}: {e} ({label} layer skipped)")
        return None


# --- Geometry ----------------------------------------------------------------

def make_projector(court: dict):
    M = np.array(court["homography"]["court_to_image"], dtype=float)

    def project(cx: float, cy: float) -> Tuple[int, int]:
        v = M @ np.array([cx, cy, 1.0])
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))
    return project


# --- Index builders (pure consumer; just reorganize upstream data) ----------

def build_role_map(track_roles: Optional[dict]) -> Dict[int, str]:
    if not track_roles:
        return {}
    return {int(t): v.get("role", "noise")
            for t, v in (track_roles.get("track_roles", {}) or {}).items()}


def build_player_index(df: Optional[pd.DataFrame], role_of: Dict[int, str]
                       ) -> Dict[int, list]:
    idx: Dict[int, list] = defaultdict(list)
    if df is None:
        return idx
    for r in df.itertuples(index=False):
        role = role_of.get(int(r.track_id))
        if role is None or role == "noise":
            continue
        idx[int(r.frame)].append({
            "tid": int(r.track_id), "role": role,
            "bbox": (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2),
            "court": (float(r.court_x_ft), float(r.court_y_ft)),
        })
    return idx


def build_ball_index(df: Optional[pd.DataFrame]) -> Dict[int, Tuple[float, float, bool]]:
    idx: Dict[int, Tuple[float, float, bool]] = {}
    if df is None:
        return idx
    for r in df.itertuples(index=False):
        idx[int(r.frame_idx)] = (float(r.pixel_x), float(r.pixel_y), bool(r.visible))
    return idx


def build_event_index(items: list, key="frame") -> Dict[int, list]:
    idx: Dict[int, list] = defaultdict(list)
    for it in items:
        idx[int(it[key])].append(it)
    return idx


# --- Drawing primitives ------------------------------------------------------

def draw_court(frame, project):
    corners = [(0, 0), (COURT_WID_FT, 0), (COURT_WID_FT, COURT_LEN_FT),
               (0, COURT_LEN_FT)]
    pts = np.array([project(*c) for c in corners], dtype=np.int32)
    cv2.polylines(frame, [pts], True, COURT_COLOR, 2, cv2.LINE_AA)
    # net + kitchen lines
    cv2.line(frame, project(0, NET_Y_FT), project(COURT_WID_FT, NET_Y_FT),
             NET_COLOR, 2, cv2.LINE_AA)
    for ky in KITCHEN_Y:
        cv2.line(frame, project(0, ky), project(COURT_WID_FT, ky),
                 COURT_COLOR, 1, cv2.LINE_AA)


def draw_players(frame, players: list):
    for p in players:
        x1, y1, x2, y2 = [int(round(v)) for v in p["bbox"]]
        c = ROLE_COLORS.get(p["role"], ROLE_COLORS["noise"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, p["role"], (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)


def draw_ball(frame, ball_idx, frame_no, synthetic, trail=True):
    if trail:
        for k in range(1, TRAIL_FRAMES + 1):
            b = ball_idx.get(frame_no - k)
            if b and b[2]:
                rad = max(1, 4 - k // 3)
                cv2.circle(frame, (int(b[0]), int(b[1])), rad, BALL_COLOR, -1,
                           cv2.LINE_AA)
    b = ball_idx.get(frame_no)
    if b and b[2]:
        col = SYNTH_COLOR if synthetic else BALL_COLOR
        cv2.circle(frame, (int(b[0]), int(b[1])), 6, col, 2, cv2.LINE_AA)


def draw_shots(frame, shot_idx, frame_no, labels=False):
    for d in range(-SHOT_MARKER_FRAMES, SHOT_MARKER_FRAMES + 1):
        for s in shot_idx.get(frame_no + d, []):
            xy = s.get("impact_pixel_xy")
            if not xy:
                continue
            x, y = int(xy[0]), int(xy[1])
            cv2.drawMarker(frame, (x, y), WHITE, cv2.MARKER_TILTED_CROSS, 16, 2)
            if labels and d == 0:
                txt = s.get("shot_type", "?")
                if s.get("is_volley"):
                    txt += " (v)"
                cv2.putText(frame, txt, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_bounces(frame, bounce_idx, frame_no):
    for d in range(-BOUNCE_MARKER_FRAMES, BOUNCE_MARKER_FRAMES + 1):
        for b in bounce_idx.get(frame_no + d, []):
            xy = b.get("pixel_xy")
            if not xy:
                continue
            col = IN_COLOR if b.get("is_in_court") else OUT_COLOR
            cv2.circle(frame, (int(xy[0]), int(xy[1])), 7, col, 2, cv2.LINE_AA)


def draw_pose(frame, pose_rows: list):
    for row in pose_rows:
        def pt(name):
            x, y = row.get(f"{name}_x_px"), row.get(f"{name}_y_px")
            if x is None or y is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return int(x), int(y)
        for a, b in POSE_EDGES:
            pa, pb = pt(a), pt(b)
            if pa and pb:
                cv2.line(frame, pa, pb, (200, 255, 200), 1, cv2.LINE_AA)


def draw_banner(frame, text: str):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 34), BLACK, -1)
    cv2.putText(frame, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2,
                cv2.LINE_AA)


def draw_watermark(frame):
    h, w = frame.shape[:2]
    txt = "SYNTHETIC BALL - placeholder analysis"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = (w - tw) // 2
    cv2.rectangle(frame, (x - 8, h - 40), (x + tw + 8, h - 12), BLACK, -1)
    cv2.putText(frame, txt, (x, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                SYNTH_COLOR, 2, cv2.LINE_AA)


def draw_hud(frame, rating: Optional[dict], plan: Optional[dict]):
    lines = []
    if rating and rating.get("rating"):
        rt = rating["rating"]
        lines.append(f"USAPA ~{rt.get('band')} ({rt.get('estimate')})")
        rng = rt.get("range")
        if rng:
            lines.append(f"range {rng[0]}-{rng[1]} conf {rt.get('confidence')}")
    else:
        lines.append("rating unavailable")
    if plan and plan.get("focus_areas"):
        fa = plan["focus_areas"][0]
        tag = "*" if fa.get("confidence") == "provisional" else ""
        lines.append(f"Focus: {fa.get('dimension')}{tag}")
    x0, y0 = 10, 44
    box_h = 20 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 4), (x0 + 300, y0 + box_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (x0, y0 + 16 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def make_minimap(players: list, bounces_now: list) -> np.ndarray:
    w = MINIMAP_W_PX
    h = int(w * COURT_LEN_FT / COURT_WID_FT)
    mm = np.full((h, w, 3), 30, np.uint8)
    m = 6

    def to_px(cx, cy):
        x = m + (cx / COURT_WID_FT) * (w - 2 * m)
        y = m + (cy / COURT_LEN_FT) * (h - 2 * m)
        return int(x), int(y)
    cv2.rectangle(mm, to_px(0, 0), to_px(COURT_WID_FT, COURT_LEN_FT),
                  COURT_COLOR, 1)
    cv2.line(mm, to_px(0, NET_Y_FT), to_px(COURT_WID_FT, NET_Y_FT), NET_COLOR, 1)
    for ky in KITCHEN_Y:
        cv2.line(mm, to_px(0, ky), to_px(COURT_WID_FT, ky), COURT_COLOR, 1)
    for p in players:
        cx, cy = p["court"]
        if np.isnan(cx) or np.isnan(cy):
            continue
        cv2.circle(mm, to_px(cx, cy), 4, ROLE_COLORS.get(p["role"]), -1)
    for b in bounces_now:
        xy = b.get("court_xy_ft")
        if xy and xy[0] is not None:
            col = IN_COLOR if b.get("is_in_court") else OUT_COLOR
            cv2.circle(mm, to_px(xy[0], xy[1]), 3, col, -1)
    return mm


def composite_minimap(frame, mm):
    h, w = mm.shape[:2]
    fh, fw = frame.shape[:2]
    y0, x0 = fh - h - 10, fw - w - 10
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + w + 2, y0 + h + 2), WHITE, 1)
    frame[y0:y0 + h, x0:x0 + w] = mm


# --- Timeline ----------------------------------------------------------------

def build_timeline(rallies, classified, bounces, rating, plan, role_of,
                   video_path, fps, frame_count, rendered_range, ball_source,
                   layers, warnings) -> dict:
    events = []
    for r in (rallies or {}).get("rallies", []):
        srole = role_of.get(int(r.get("server_track_id", -1)))
        events.append({"frame": int(r["start_frame"]),
                       "t_sec": round(r["start_frame"] / fps, 3),
                       "type": "rally_start", "rally_id": r["rally_id"],
                       "server_role": srole})
        events.append({"frame": int(r["end_frame"]),
                       "t_sec": round(r["end_frame"] / fps, 3),
                       "type": "rally_end", "rally_id": r["rally_id"],
                       "end_reason": r["end_reason"],
                       "end_reason_confidence": r.get("end_reason_confidence")})
    for s in (classified or {}).get("shots", []):
        events.append({"frame": int(s["frame"]),
                       "t_sec": round(s["frame"] / fps, 3), "type": "shot",
                       "shot_id": s["shot_id"], "role": role_of.get(int(s["track_id"])),
                       "shot_type": s.get("shot_type"),
                       "stroke_side": s.get("stroke_side"),
                       "is_volley": s.get("is_volley"), "is_serve": s.get("is_serve")})
    for b in (bounces or {}).get("bounces", []):
        events.append({"frame": int(b["frame"]),
                       "t_sec": round(b["frame"] / fps, 3), "type": "bounce",
                       "bounce_id": b["bounce_id"],
                       "is_in_court": b.get("is_in_court"),
                       "court_zone": b.get("court_zone")})
    events.sort(key=lambda e: (e["frame"], EVENT_TYPE_RANK.get(e["type"], 9)))

    summary = {"rated_role": "user", "synthetic_ball": ball_source == "synthetic"}
    if rating:
        summary["rating"] = rating.get("rating")
    if plan:
        summary["target_band"] = plan.get("target", {}).get("band")
        summary["focus_areas"] = [
            {"priority": f["priority"], "dimension": f["dimension"],
             "confidence": f["confidence"]}
            for f in plan.get("focus_areas", [])]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_video": str(video_path),
        "fps": float(fps),
        "frame_count": frame_count,
        "rendered_range": rendered_range,
        "ball_source": ball_source,
        "duration_sec": round(frame_count / fps, 2),
        "summary": summary,
        "events": events,
        "layers_rendered": layers,
        "warnings": warnings,
        "stage_version": STAGE_VERSION,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


# --- Heatmap PNGs ------------------------------------------------------------

def render_heatmap_png(grid: list, out_path: Path, title: str):
    arr = np.array(grid, dtype=float)
    if arr.size == 0 or arr.max() <= 0:
        arr = np.zeros_like(arr) if arr.size else np.zeros((22, 10))
    norm = (arr / arr.max() * 255).astype(np.uint8) if arr.max() > 0 \
        else arr.astype(np.uint8)
    # upscale to a court-proportioned image (rows=y=44ft, cols=x=20ft)
    cell = 18
    img = cv2.resize(norm, (arr.shape[1] * cell, arr.shape[0] * cell),
                     interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    cv2.rectangle(color, (0, 0), (color.shape[1] - 1, color.shape[0] - 1),
                  WHITE, 1)
    # net line at y=22ft
    ny = int(color.shape[0] * NET_Y_FT / COURT_LEN_FT)
    cv2.line(color, (0, ny), (color.shape[1], ny), WHITE, 1)
    cv2.putText(color, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
                cv2.LINE_AA)
    cv2.imwrite(str(out_path), color)


# --- Main --------------------------------------------------------------------

def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    video_path = folder / "video.mp4"
    court_path = folder / "court.json"
    if not video_path.exists():
        fail(f"required input not found: {video_path}", FileNotFoundError)
    if not court_path.exists():
        fail(f"required input not found: {court_path}", FileNotFoundError)

    out_video = folder / "annotated.mp4"
    out_timeline = folder / "timeline.json"
    if not args.heatmaps_only and out_video.exists() and not args.force:
        fail(f"output exists: {out_video}. Use --force to overwrite.",
             FileExistsError)
    if out_timeline.exists() and not args.force:
        fail(f"output exists: {out_timeline}. Use --force to overwrite.",
             FileExistsError)

    court = json.loads(court_path.read_text(encoding="utf-8"))
    project = make_projector(court)

    players_df = load_parquet_opt(folder / "players.parquet", log, "players")
    poses_df = load_parquet_opt(folder / "poses.parquet", log, "pose") if args.pose else None
    ball_df = load_parquet_opt(folder / "ball.parquet", log, "ball")
    track_roles = load_json_opt(folder / "track_roles.json", log, "roles")
    ball_meta = load_json_opt(folder / "ball.meta.json", log, "ball-meta")
    classified = load_json_opt(folder / "classified.json", log, "shots")
    bounces = load_json_opt(folder / "bounces.json", log, "bounces")
    rallies = load_json_opt(folder / "rallies.json", log, "rallies")
    metrics = load_json_opt(folder / "metrics.json", log, "heatmaps")
    rating = load_json_opt(folder / "rating.json", log, "hud-rating")
    plan = load_json_opt(folder / "improvement_plan.json", log, "hud-plan")

    ball_source = "real"
    for src in (classified, bounces):
        if src and src.get("ball_source"):
            ball_source = src["ball_source"]
            break
    if ball_meta and ball_meta.get("synthetic"):
        ball_source = "synthetic"
    synthetic = ball_source == "synthetic"
    if synthetic:
        log.warning("ball_source is SYNTHETIC: burning placeholder watermark.")

    role_of = build_role_map(track_roles)
    player_idx = build_player_index(players_df, role_of)
    ball_idx = build_ball_index(ball_df)
    shot_idx = build_event_index((classified or {}).get("shots", []))
    bounce_idx = build_event_index((bounces or {}).get("bounces", []))
    rally_list = sorted((rallies or {}).get("rallies", []),
                        key=lambda r: r["start_frame"])
    pose_idx: Dict[int, list] = defaultdict(list)
    if poses_df is not None:
        for row in poses_df[poses_df.get("is_user", False) == True].to_dict("records"):  # noqa: E712
            pose_idx[int(row["frame"])].append(row)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"could not open video: {video_path}")
    fps = court.get("video", {}).get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = max(0, args.start_frame or 0)
    end = args.end_frame if args.end_frame is not None else frame_count
    if args.max_seconds is not None:
        end = min(end, start + int(args.max_seconds * fps))
    end = min(end, frame_count)
    if start >= end:
        fail(f"empty render range [{start},{end})", ValueError)

    warnings: List[str] = []
    if synthetic:
        warnings.append("ball_source is 'synthetic': annotated video + analysis "
                        "are PLACEHOLDER (watermark burned in).")
    for label, obj in [("players", players_df), ("track_roles", track_roles),
                       ("classified", classified), ("bounces", bounces),
                       ("rallies", rallies), ("rating", rating),
                       ("improvement_plan", plan), ("metrics", metrics)]:
        if obj is None:
            warnings.append(f"{label} unavailable: related overlay omitted.")

    layers = ["court"]
    if player_idx:
        layers.append("players")
    if ball_idx:
        layers += ["ball"] + (["trail"] if not args.no_trail else [])
    if shot_idx:
        layers.append("shots")
    if bounce_idx:
        layers.append("bounces")
    if rally_list:
        layers.append("rally_banner")
    if not args.no_hud and (rating or plan):
        layers.append("hud")
    if not args.no_minimap and player_idx:
        layers.append("minimap")
    if args.pose and pose_idx:
        layers.append("pose")
    if synthetic:
        layers.append("watermark")

    # --- render video ---
    if not args.heatmaps_only:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
        writer = cv2.VideoWriter(str(out_video), fourcc, args.fps_out or fps,
                                 (w, h))
        if not writer.isOpened():
            fail(f"could not open VideoWriter for {out_video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for fno in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            draw_court(frame, project)
            players_now = player_idx.get(fno, [])
            if "players" in layers:
                draw_players(frame, players_now)
            if args.pose and "pose" in layers:
                draw_pose(frame, pose_idx.get(fno, []))
            if "ball" in layers:
                draw_ball(frame, ball_idx, fno, synthetic, not args.no_trail)
            if "shots" in layers:
                draw_shots(frame, shot_idx, fno, args.labels)
            if "bounces" in layers:
                draw_bounces(frame, bounce_idx, fno)
            if rally_list:
                cur = None
                for r in rally_list:
                    if r["start_frame"] <= fno:
                        cur = r
                    else:
                        break
                if cur is not None:
                    srole = role_of.get(int(cur.get("server_track_id", -1)), "?")
                    txt = f"Rally {cur['rally_id']} | shots {cur['n_shots']} | server {srole}"
                    if fno >= cur["end_frame"]:
                        txt += f" | ended: {cur['end_reason']}"
                    draw_banner(frame, txt)
            if "hud" in layers:
                draw_hud(frame, rating, plan)
            if "minimap" in layers:
                bn = [b for d in range(-BOUNCE_MARKER_FRAMES, BOUNCE_MARKER_FRAMES + 1)
                      for b in bounce_idx.get(fno + d, [])]
                composite_minimap(frame, make_minimap(players_now, bn))
            if synthetic:
                draw_watermark(frame)
            writer.write(frame)
        writer.release()
        log.info(f"wrote {out_video} ({end - start} frames)")
    cap.release()

    # --- timeline ---
    timeline = build_timeline(rallies, classified, bounces, rating, plan,
                              role_of, video_path, fps, frame_count,
                              [start, end], ball_source, layers, warnings)
    out_timeline.write_text(json.dumps(timeline, indent=2) + "\n",
                            encoding="utf-8")
    log.info(f"wrote {out_timeline} ({len(timeline['events'])} events)")

    # --- heatmap PNGs ---
    heatmaps_written = []
    if metrics and metrics.get("heatmaps"):
        hm = metrics["heatmaps"]
        for role, grid in (hm.get("player_position", {}) or {}).items():
            p = folder / f"heatmap_position_{role}.png"
            render_heatmap_png(grid, p, f"position: {role}")
            heatmaps_written.append(p.name)
        if hm.get("ball_landing"):
            p = folder / "heatmap_ball_landing.png"
            render_heatmap_png(hm["ball_landing"], p, "ball landing")
            heatmaps_written.append(p.name)
        log.info(f"wrote {len(heatmaps_written)} heatmap PNG(s)")
    else:
        warnings.append("metrics.json heatmaps unavailable: no heatmap PNGs.")

    return {"timeline": timeline, "heatmaps": heatmaps_written,
            "video": str(out_video) if not args.heatmaps_only else None}


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 11 — render annotated video")
    p.add_argument("folder", type=Path, help="per-video folder")
    p.add_argument("--force", action="store_true")
    p.add_argument("--start-frame", type=int, default=0, dest="start_frame")
    p.add_argument("--end-frame", type=int, default=None, dest="end_frame")
    p.add_argument("--max-seconds", type=float, default=None, dest="max_seconds")
    p.add_argument("--fps-out", type=float, default=None, dest="fps_out")
    p.add_argument("--pose", action="store_true", help="draw pose skeleton")
    p.add_argument("--labels", action="store_true", help="shot-type text labels")
    p.add_argument("--no-trail", action="store_true", dest="no_trail")
    p.add_argument("--no-minimap", action="store_true", dest="no_minimap")
    p.add_argument("--no-hud", action="store_true", dest="no_hud")
    p.add_argument("--heatmaps-only", action="store_true", dest="heatmaps_only")
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

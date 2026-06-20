"""Stage 8 — compute metrics.

Aggregate every upstream stream (classified.json, rallies.json, bounces.json,
players.parquet, track_roles.json) into one metrics.json: match-level summary,
per-player (per-role) breakdowns, error attribution, team positioning + movement
(REAL data), numeric heatmap grids, and structural placeholders for Tier-B
ball-derived metrics (pending real ball detection v4).

Pure aggregation — no new detection. Correctness is enforced by reconciliation
invariants (counts sum, by_end_reason matches Stage 7), not by ball-derived
accuracy, because the ball is still synthetic.

See stages/compute_metrics/contract.md for the full spec.

Usage:
    python -m stages.compute_metrics.compute_metrics data/test_clip
    python -m stages.compute_metrics.compute_metrics data/test_clip --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SCHEMA_VERSION = 1
STAGE_VERSION = "0.1.0"

# --- Config (matches contract) ----------------------------------------------
HEATMAP_BIN_FT = 2.0          # court grid bin -> 10 cols (x) x 22 rows (y)
ROLE_CONF_FLOOR = 0.55        # role_confidence below this -> role_contaminated
NET_Y_FT = 22.0               # net line (= length_ft / 2)            [Stage 6]
KITCHEN_MAX_DIST_FT = 9.0     # effective kitchen depth from net      [Stage 6]
BASELINE_MIN_DIST_FT = 17.0   # within ~5ft of own baseline -> baseline [Stage 6]
COURT_LEN_FT = 44.0
COURT_WID_FT = 20.0
MOVE_MIN_STEP_FT = 0.25       # per-frame foot delta below this = jitter
RALLY_LEN_BUCKETS = ["1", "2-4", "5-8", "9+"]

# NOTE: opponents are identity-based (opp_a/opp_b), NOT position L/R (Stage 2.5).
# Any left/right-by-court_x semantics in this stage is stale and belongs to the
# deferred real-ball Stage 8 rework (SYSTEM_DESIGN.md #7); the vocab is renamed
# here for consistency only.
PLAYING_ROLES = ["user", "partner", "opp_a", "opp_b"]
NEAR_ROLES = ["user", "partner"]
FAR_ROLES = ["opp_a", "opp_b"]
HITTER_ERRORS = {"ball-out", "net-or-short", "ball-off-frame"}
RECEIVER_ERRORS = {"double-bounce", "ball-not-returned"}

EPS = 1e-9


def fail(msg: str, exc=RuntimeError):
    raise exc(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("compute_metrics")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"required input not found: {path}", FileNotFoundError)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --- Zone / lateral helpers (Stage 6 source of truth) -----------------------

def zone_from_court_y(court_y: float) -> str:
    """Depth zone by distance from the net. Verbatim from Stage 6
    (classify_shots.zone_from_court_y) so a shot's contact_zone and a player's
    standing zone always agree."""
    dist_from_net = abs(court_y - NET_Y_FT)
    if dist_from_net <= KITCHEN_MAX_DIST_FT:
        return "kitchen"
    if dist_from_net >= BASELINE_MIN_DIST_FT:
        return "baseline"
    return "transition"


def lateral_from_court_x(court_x: float) -> str:
    """Left / center / right by court_x thirds. Convention: left = low court_x,
    right = high court_x (court-coordinate, NOT player-egocentric). This is a
    position descriptor only; opponents are identity-based (opp_a/opp_b)."""
    third = COURT_WID_FT / 3.0
    if court_x < third:
        return "left"
    if court_x < 2.0 * third:
        return "center"
    return "right"


def in_extent(x: float, y: float) -> bool:
    return 0.0 <= x < COURT_WID_FT and 0.0 <= y < COURT_LEN_FT


# --- Heatmap grid ------------------------------------------------------------

N_COLS = int(round(COURT_WID_FT / HEATMAP_BIN_FT))   # 10
N_ROWS = int(round(COURT_LEN_FT / HEATMAP_BIN_FT))   # 22


def new_grid() -> List[List[int]]:
    return [[0 for _ in range(N_COLS)] for _ in range(N_ROWS)]


def bin_positions(positions: List[Tuple[float, float]]) -> Tuple[List[List[int]], int]:
    """Accumulate (x,y) court positions into a row-major grid. Returns
    (grid, n_in_extent). Out-of-extent positions are dropped (not clamped)."""
    grid = new_grid()
    n_in = 0
    for x, y in positions:
        if not in_extent(x, y):
            continue
        col = int(x / HEATMAP_BIN_FT)
        row = int(y / HEATMAP_BIN_FT)
        if 0 <= row < N_ROWS and 0 <= col < N_COLS:
            grid[row][col] += 1
            n_in += 1
    return grid, n_in


# --- Role mapping ------------------------------------------------------------

def build_role_maps(track_roles_doc: Optional[dict], shots: List[dict],
                    log: logging.Logger) -> Tuple[Dict[int, str], Dict[str, List[int]],
                                                   Dict[str, float], bool]:
    """Returns (tid_to_role, role_to_tids, role_confidence, degraded).
    degraded=True means we fell back to user-only attribution from is_user."""
    if track_roles_doc is not None:
        if track_roles_doc.get("schema_version") != 1:
            fail(f"track_roles.json schema_version="
                 f"{track_roles_doc.get('schema_version')} unexpected (expects 1)",
                 ValueError)
        roles = track_roles_doc.get("roles", {}) or {}
        track_roles = track_roles_doc.get("track_roles", {}) or {}
        role_to_tids = {r: [int(t) for t in roles.get(r, {}).get("track_ids", [])]
                        for r in PLAYING_ROLES}
        tid_to_role: Dict[int, str] = {}
        for r in PLAYING_ROLES:
            for t in role_to_tids[r]:
                tid_to_role[t] = r
        # role confidence = mean of member tracks' confidences (track_roles dict)
        role_confidence: Dict[str, float] = {}
        for r in PLAYING_ROLES:
            confs = [float(track_roles[str(t)]["confidence"])
                     for t in role_to_tids[r] if str(t) in track_roles]
            role_confidence[r] = round(sum(confs) / len(confs), 3) if confs else 0.0
        return tid_to_role, role_to_tids, role_confidence, False

    # --- Degraded fallback: user-only from is_user ---
    log.warning("track_roles.json unavailable: falling back to user-only "
                "attribution from is_user; partner/opponents will be empty.")
    user_tids = sorted({int(s["track_id"]) for s in shots if s.get("is_user")})
    role_to_tids = {"user": user_tids, "partner": [], "opp_a": [], "opp_b": []}
    tid_to_role = {t: "user" for t in user_tids}
    role_confidence = {"user": 0.95 if user_tids else 0.0,
                       "partner": 0.0, "opp_a": 0.0, "opp_b": 0.0}
    return tid_to_role, role_to_tids, role_confidence, True


# --- Aggregation helpers -----------------------------------------------------

def count_by(items, key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        v = key(it)
        out[v] = out.get(v, 0) + 1
    return out


def rally_len_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 8:
        return "5-8"
    return "9+"


def shot_mix(shots: List[dict]) -> dict:
    n_volley = sum(1 for s in shots if s.get("is_volley"))
    return {
        "by_shot_type": count_by(shots, lambda s: s.get("shot_type", "unknown")),
        "by_stroke_side": count_by(shots, lambda s: s.get("stroke_side", "unknown")),
        "n_volley": n_volley,
        "volley_rate": round(n_volley / len(shots), 3) if shots else 0.0,
    }


def safe_stats(vals: List[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {"mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "max": round(max(vals), 3)}


# --- Position / movement (real data) ----------------------------------------

def role_valid_rows(df: pd.DataFrame, track_ids: List[int]) -> pd.DataFrame:
    """Non-transient rows for a role's tracks with a finite court position."""
    if not track_ids:
        return df.iloc[0:0]
    sub = df[df["track_id"].isin(track_ids) & (~df["transient"])]
    sub = sub[sub["court_x_ft"].notna() & sub["court_y_ft"].notna()]
    return sub


def role_frame_pos(sub: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    """frame -> (mean court_x, mean court_y) for a role's rows (averages any
    duplicate rows at the same frame)."""
    if sub.empty:
        return {}
    g = sub.groupby("frame")[["court_x_ft", "court_y_ft"]].mean()
    return {int(f): (float(r.court_x_ft), float(r.court_y_ft))
            for f, r in g.iterrows()}


def compute_position(sub: pd.DataFrame, role: str, fps: float,
                     rally_windows: List[Tuple[int, int]], n_rallies: int) -> dict:
    fpos = role_frame_pos(sub)
    n_frames = len(fpos)
    own_far = role in FAR_ROLES
    if n_frames == 0:
        return {
            "n_frames": 0,
            "zone_time_frac": {"kitchen": 0.0, "transition": 0.0, "baseline": 0.0},
            "lateral_time_frac": {"left": 0.0, "center": 0.0, "right": 0.0},
            "area_time_frac": {f"{d}-{l}": 0.0 for d in
                               ("kitchen", "transition", "baseline")
                               for l in ("left", "center", "right")},
            "court_coverage_frac": 0.0,
            "mean_court_xy_ft": [0.0, 0.0],
            "movement": {"distance_ft_total": 0.0, "distance_ft_per_rally": 0.0,
                         "distance_ft_per_min": 0.0},
        }

    zone_ct = {"kitchen": 0, "transition": 0, "baseline": 0}
    lat_ct = {"left": 0, "center": 0, "right": 0}
    area_ct = {f"{d}-{l}": 0 for d in ("kitchen", "transition", "baseline")
               for l in ("left", "center", "right")}
    xs, ys = [], []
    for (x, y) in fpos.values():
        d = zone_from_court_y(y)
        zone_ct[d] += 1
        # lateral only meaningful inside the court width; clamp for binning
        lx = min(max(x, 0.0), COURT_WID_FT - EPS)
        l = lateral_from_court_x(lx)
        lat_ct[l] += 1
        area_ct[f"{d}-{l}"] += 1
        xs.append(x)
        ys.append(y)

    zone_frac = {k: round(v / n_frames, 4) for k, v in zone_ct.items()}
    lat_frac = {k: round(v / n_frames, 4) for k, v in lat_ct.items()}
    area_frac = {k: round(v / n_frames, 4) for k, v in area_ct.items()}

    # coverage over own half (cells with >=1 visit) / cells in own half
    half_rows = range(N_ROWS // 2, N_ROWS) if own_far else range(0, N_ROWS // 2)
    visited = set()
    for (x, y) in fpos.values():
        if not in_extent(x, y):
            continue
        col = int(x / HEATMAP_BIN_FT)
        row = int(y / HEATMAP_BIN_FT)
        if row in half_rows and 0 <= col < N_COLS:
            visited.add((row, col))
    n_half_cells = (N_ROWS // 2) * N_COLS
    coverage = round(len(visited) / n_half_cells, 4) if n_half_cells else 0.0

    # movement: integrate path length above jitter floor
    total_dist = 0.0
    rally_dist = 0.0
    rallies_present = set()
    prev = None
    for f in sorted(fpos.keys()):
        x, y = fpos[f]
        in_rally_idx = _frame_rally_index(f, rally_windows)
        if in_rally_idx is not None:
            rallies_present.add(in_rally_idx)
        if prev is not None:
            pf, px, py = prev
            step = math.hypot(x - px, y - py)
            if step >= MOVE_MIN_STEP_FT:
                total_dist += step
                if in_rally_idx is not None:
                    rally_dist += step
        prev = (f, x, y)

    active_sec = n_frames / fps if fps > 0 else 0.0
    per_min = round(total_dist / (active_sec / 60.0), 2) if active_sec > 0 else 0.0
    n_present = len(rallies_present)
    per_rally = round(rally_dist / n_present, 2) if n_present else 0.0

    return {
        "n_frames": n_frames,
        "zone_time_frac": zone_frac,
        "lateral_time_frac": lat_frac,
        "area_time_frac": area_frac,
        "court_coverage_frac": coverage,
        "mean_court_xy_ft": [round(statistics.mean(xs), 2),
                             round(statistics.mean(ys), 2)],
        "movement": {
            "distance_ft_total": round(total_dist, 2),
            "distance_ft_per_rally": per_rally,
            "distance_ft_per_min": per_min,
        },
    }


def _frame_rally_index(frame: int, windows: List[Tuple[int, int]]) -> Optional[int]:
    for i, (a, b) in enumerate(windows):
        if a <= frame <= b:
            return i
    return None


def compute_team(side: str, roles: List[str],
                 role_fpos: Dict[str, Dict[int, Tuple[float, float]]],
                 role_positions: Dict[str, dict],
                 role_contaminated: Dict[str, bool]) -> dict:
    a, b = roles
    fa, fb = role_fpos.get(a, {}), role_fpos.get(b, {})
    common = sorted(set(fa.keys()) & set(fb.keys()))
    both_kitchen = 0
    spacings = []
    for f in common:
        ax, ay = fa[f]
        bx, by = fb[f]
        if abs(ay - NET_Y_FT) <= KITCHEN_MAX_DIST_FT and \
                abs(by - NET_Y_FT) <= KITCHEN_MAX_DIST_FT:
            both_kitchen += 1
        spacings.append(math.hypot(ax - bx, ay - by))
    n_common = len(common)
    out = {
        "roles": roles,
        "n_frames_both_present": n_common,
        "both_at_kitchen_frac": round(both_kitchen / n_common, 4) if n_common else 0.0,
        "spacing_ft": {
            "mean": round(statistics.mean(spacings), 2) if spacings else 0.0,
            "median": round(statistics.median(spacings), 2) if spacings else 0.0,
            "min": round(min(spacings), 2) if spacings else 0.0,
            "max": round(max(spacings), 2) if spacings else 0.0,
        },
        "transition_time_frac": {
            r: role_positions.get(r, {}).get("zone_time_frac", {}).get("transition", 0.0)
            for r in roles
        },
    }
    if side == "far":
        out["role_contaminated"] = any(role_contaminated.get(r, False) for r in roles)
    return out


# --- Tier-B pending block (structure only; values null until real ball) -----

def pending_real_ball_block() -> dict:
    return {
        "_comment": (
            "Tier-B metrics. STRUCTURALLY present so the output shape is stable "
            "and downstream/UI can bind to it now, but VALUE is null in v1: they "
            "need trustworthy ball trajectories, and computing them against the "
            "synthetic ball would be placeholder-only. Each entry documents what "
            "it will contain once ball detection v4 lands. See KNOWN_ISSUES.md "
            "'Synthetic ball' section."),
        "forced_vs_unforced_errors": {
            "status": "pending_real_ball", "value": None,
            "description": (
                "Splits each committed error into 'forced' (error off a fast "
                "incoming ball: pre-speed >= FORCED_MIN_INCOMING_FTPS) vs "
                "'unforced'. Will populate {by_owner: {<role>: {forced, unforced, "
                "unforced_rate}}, match: {forced, unforced, unforced_rate}}. "
                "High-value input to the Stage 9 USAPA rating."),
        },
        "dink_shot_tolerance": {
            "status": "pending_real_ball", "value": None,
            "description": (
                "Average consecutive dinks, and average total shots, sustained "
                "before the rally-ending error. Will populate {match: "
                "{mean_dinks_before_error, mean_shots_before_error}, players: "
                "{<role>: {...}}}. Needs reliable shot-type + rally sequencing."),
        },
        "third_shot_drop_outcome": {
            "status": "pending_real_ball", "value": None,
            "description": (
                "Whether each third-shot drop SUCCEEDED (hitting team won the "
                "ensuing kitchen approach). Will populate {n_drops, n_successful, "
                "success_rate, by_server_role}. Needs post-drop trajectory."),
        },
        "opponent_backhand_targeting": {
            "status": "pending_real_ball", "value": None,
            "description": (
                "Uses roster handedness + shot-direction geometry to measure how "
                "often a player targets an opponent's BACKHAND and the win rate. "
                "Will populate {by_role: {<role>: {n_shots_to_opp_backhand, "
                "frac_to_backhand, point_win_rate_when_to_backhand}}}. Needs shot "
                "direction from real ball + reliable opponent roles."),
        },
    }


# --- Main pipeline -----------------------------------------------------------

def run(folder: Path, args, log: logging.Logger) -> dict:
    if not folder.is_dir():
        fail(f"not a folder: {folder}", FileNotFoundError)
    classified_path = folder / "classified.json"
    rallies_path = folder / "rallies.json"
    bounces_path = folder / "bounces.json"
    players_path = folder / "players.parquet"
    track_roles_path = folder / "track_roles.json"
    roster_path = folder / "roster.json"
    court_path = folder / "court.json"
    out_path = folder / "metrics.json"

    if out_path.exists() and not args.force:
        fail(f"output exists: {out_path}. Use --force to overwrite.",
             FileExistsError)

    # Required structural inputs.
    classified = load_json(classified_path)
    if classified.get("schema_version") != 1:
        fail(f"classified.json schema_version={classified.get('schema_version')} "
             f"unexpected (expects 1)", ValueError)
    if not players_path.exists():
        fail(f"required input not found: {players_path}", FileNotFoundError)
    court = load_json(court_path)

    fps = classified.get("fps") or (court.get("video", {}) or {}).get("fps")
    if fps is None or fps <= 0:
        fail("could not determine fps from classified.json or court.json", ValueError)

    # Optional / degradable inputs.
    rallies_doc = load_json(rallies_path) if rallies_path.exists() else None
    if rallies_doc is not None and rallies_doc.get("schema_version") != 1:
        fail(f"rallies.json schema_version={rallies_doc.get('schema_version')} "
             f"unexpected (expects 1)", ValueError)
    bounces_doc = load_json(bounces_path) if bounces_path.exists() else None
    if bounces_doc is not None and bounces_doc.get("schema_version") != 1:
        fail(f"bounces.json schema_version={bounces_doc.get('schema_version')} "
             f"unexpected (expects 1)", ValueError)
    track_roles_doc = load_json(track_roles_path) if track_roles_path.exists() else None
    roster = load_json(roster_path) if roster_path.exists() else {}
    handedness = (roster.get("handedness", {}) or {})

    ball_source = (classified.get("ball_source")
                   or (bounces_doc or {}).get("ball_source") or "real")
    is_synth = ball_source == "synthetic"
    if is_synth:
        log.warning("ball_source is SYNTHETIC: all ball-derived metrics are "
                    "PLACEHOLDER (see reliability.synthetic_gated).")

    shots = classified.get("shots", [])
    shot_by_id = {int(s["shot_id"]): s for s in shots}
    bounces = (bounces_doc or {}).get("bounces", [])
    rallies = (rallies_doc or {}).get("rallies", [])

    warnings: List[str] = []

    # --- Role maps ---
    tid_to_role, role_to_tids, role_confidence, degraded = build_role_maps(
        track_roles_doc, shots, log)
    if degraded:
        warnings.append("track_roles.json unavailable: per-player attribution "
                        "degraded to user-only (from is_user); partner/opponent "
                        "blocks are empty.")
    role_contaminated = {r: role_confidence.get(r, 0.0) < args.role_conf_floor
                         for r in PLAYING_ROLES}
    assigned_tids = {t for tids in role_to_tids.values() for t in tids}

    def role_of(tid: Optional[int]) -> Optional[str]:
        if tid is None:
            return None
        return tid_to_role.get(int(tid))

    # --- Match summary ---
    rally_lengths = [int(r["n_shots"]) for r in rallies]
    rally_durs = [float(r["duration_sec"]) for r in rallies]
    rl_dist = {b: 0 for b in RALLY_LEN_BUCKETS}
    for n in rally_lengths:
        rl_dist[rally_len_bucket(n)] += 1

    by_end_reason = (rallies_doc or {}).get("stats", {}).get("by_end_reason", {}) \
        if rallies_doc else {}
    if not by_end_reason and rallies:
        by_end_reason = count_by(rallies, lambda r: r["end_reason"])

    n_serves = len(rallies)
    n_serve_faults = by_end_reason.get("serve-fault", 0)

    # third shot
    third_shots = []
    for r in rallies:
        if int(r["n_shots"]) >= 3 and len(r["shot_ids"]) >= 3:
            s = shot_by_id.get(int(r["shot_ids"][2]))
            if s is not None:
                third_shots.append(s)
    third_by_type = count_by(third_shots, lambda s: s.get("shot_type", "unknown"))
    third_drop_rate = (round(third_by_type.get("drop", 0) / len(third_shots), 3)
                       if third_shots else 0.0)

    n_in = sum(1 for b in bounces if b.get("is_in_court") is True)
    n_out = sum(1 for b in bounces if b.get("is_in_court") is False)

    # match frame span
    all_frames = [int(s["frame"]) for s in shots] + [int(b["frame"]) for b in bounces]
    match_span_sec = (round((max(all_frames) - min(all_frames)) / fps, 2)
                      if all_frames else 0.0)

    match = {
        "n_rallies": len(rallies),
        "n_shots": len(shots),
        "n_bounces": len(bounces),
        "match_span_sec": match_span_sec,
        "rally_length_shots": {**safe_stats([float(n) for n in rally_lengths]),
                               "distribution": rl_dist},
        "rally_duration_sec": safe_stats(rally_durs),
        "by_end_reason": by_end_reason,
        "serve": {
            "n_serves": n_serves,
            "n_serve_faults": n_serve_faults,
            "serve_fault_rate": round(n_serve_faults / n_serves, 4) if n_serves else 0.0,
        },
        "shot_mix": shot_mix(shots),
        "third_shot": {
            "n_rallies_ge_3_shots": len(third_shots),
            "by_shot_type": third_by_type,
            "drop_rate": third_drop_rate,
        },
        "bounce_in_out": {
            "n_in": n_in, "n_out": n_out,
            "in_rate": round(n_in / (n_in + n_out), 4) if (n_in + n_out) else 0.0,
        },
    }

    # --- Error attribution ---
    by_owner: Dict[str, int] = {}
    by_er_owner: Dict[Tuple[str, str, str], int] = {}
    errors_committed: Dict[str, int] = {r: 0 for r in PLAYING_ROLES}

    def add_owner(owner: str, end_reason: str, kind: str):
        by_owner[owner] = by_owner.get(owner, 0) + 1
        by_er_owner[(end_reason, owner, kind)] = \
            by_er_owner.get((end_reason, owner, kind), 0) + 1

    for r in rallies:
        er = r["end_reason"]
        sig = r.get("end_signals", {}) or {}
        if er == "serve-fault":
            owner = role_of(r.get("server_track_id")) or "unattributed"
            add_owner(owner, er, "server")
            if owner in errors_committed:
                errors_committed[owner] += 1
        elif er in HITTER_ERRORS:
            last_sid = int(r["shot_ids"][-1]) if r["shot_ids"] else None
            last_shot = shot_by_id.get(last_sid) if last_sid is not None else None
            owner = (role_of(last_shot["track_id"]) if last_shot else None) or "unattributed"
            add_owner(owner, er, "hitter")
            if owner in errors_committed:
                errors_committed[owner] += 1
        elif er in RECEIVER_ERRORS:
            hs = sig.get("hitter_side")
            if hs == "near":
                owner = "team_far"
            elif hs == "far":
                owner = "team_near"
            else:
                owner = "unknown"
            add_owner(owner, er, "receiver")
        else:  # unknown
            add_owner("unknown", er, "unknown")

    error_attribution = {
        "by_owner": by_owner,
        "by_end_reason_and_owner": [
            {"end_reason": k[0], "owner": k[1], "owner_kind": k[2], "count": v}
            for k, v in sorted(by_er_owner.items())
        ],
        "notes": [
            "Server / hitter errors attribute to a specific role via track_id.",
            "Receiver errors (double-bounce, ball-not-returned) attribute to the "
            "receiving TEAM (team_near/team_far) — the specific receiver of two "
            "players is not identifiable in v1.",
            "unknown end_reason -> 'unknown' owner; shots whose track_id maps to "
            "no role -> 'unattributed'.",
        ],
    }

    # --- players.parquet load + per-role position ---
    df = pd.read_parquet(players_path)
    rally_windows = [(int(r["start_frame"]), int(r["end_frame"])) for r in rallies]
    n_rallies = len(rallies)

    role_positions: Dict[str, dict] = {}
    role_fpos: Dict[str, Dict[int, Tuple[float, float]]] = {}
    player_pos_heatmaps: Dict[str, List[List[int]]] = {}
    pos_heatmap_in_extent: Dict[str, int] = {}

    for r in PLAYING_ROLES:
        sub = role_valid_rows(df, role_to_tids[r])
        role_positions[r] = compute_position(sub, r, fps, rally_windows, n_rallies)
        fpos = role_frame_pos(sub)
        role_fpos[r] = fpos
        grid, n_ext = bin_positions(list(fpos.values()))
        player_pos_heatmaps[r] = grid
        pos_heatmap_in_extent[r] = n_ext

    # --- per-role shot stats ---
    players_out: Dict[str, dict] = {}
    n_attributed_shots = 0
    for r in PLAYING_ROLES:
        tids = set(role_to_tids[r])
        rshots = [s for s in shots if int(s["track_id"]) in tids]
        n_attributed_shots += len(rshots)
        rserves = [s for s in rshots if s.get("is_serve")]
        # role serve-faults: rallies this role served AND end_reason serve-fault
        rsf = sum(1 for ra in rallies
                  if role_of(ra.get("server_track_id")) == r
                  and ra["end_reason"] == "serve-fault")
        speeds = [s["features"]["post_speed_ftps"] for s in rshots
                  if not s.get("is_serve") and s.get("features", {}).get("post_speed_ftps") is not None]
        players_out[r] = {
            "role_confidence": role_confidence.get(r, 0.0),
            "role_contaminated": bool(role_contaminated.get(r, False)),
            "handedness": handedness.get(r, "unknown"),
            "track_ids": role_to_tids[r],
            "n_shots": len(rshots),
            "shot_mix": shot_mix(rshots),
            "serve": {
                "n_serves": len(rserves),
                "n_serve_faults": rsf,
                "serve_fault_rate": round(rsf / len(rserves), 4) if rserves else 0.0,
            },
            "errors_committed": errors_committed[r],
            "mean_post_speed_ftps": round(statistics.mean(speeds), 2) if speeds else None,
            "position": role_positions[r],
        }

    unattributed_shots = [s for s in shots if int(s["track_id"]) not in assigned_tids]
    players_out["unattributed"] = {
        "n_shots": len(unattributed_shots),
        "note": "shots whose track_id is noise or maps to no role",
    }

    # reconciliation guard (loud failure on drift)
    if n_attributed_shots + len(unattributed_shots) != len(shots):
        fail(f"shot reconciliation failed: attributed {n_attributed_shots} + "
             f"unattributed {len(unattributed_shots)} != total {len(shots)}")
    if sum(by_owner.values()) != len(rallies):
        fail(f"error-owner reconciliation failed: sum(by_owner)="
             f"{sum(by_owner.values())} != n_rallies {len(rallies)}")

    # --- team ---
    team = {
        "near": compute_team("near", NEAR_ROLES, role_fpos, role_positions, role_contaminated),
        "far": compute_team("far", FAR_ROLES, role_fpos, role_positions, role_contaminated),
    }

    # --- heatmaps ---
    ball_positions = [tuple(b["court_xy_ft"]) for b in bounces
                      if b.get("court_xy_ft") and b["court_xy_ft"][0] is not None]
    ball_grid, _ = bin_positions([(float(x), float(y)) for x, y in ball_positions])
    heatmaps = {
        "grid": {
            "bin_ft": HEATMAP_BIN_FT,
            "x_min_ft": 0.0, "x_max_ft": COURT_WID_FT, "n_cols": N_COLS,
            "y_min_ft": 0.0, "y_max_ft": COURT_LEN_FT, "n_rows": N_ROWS,
            "row_major": True,
            "note": ("cell [r][c] covers x in [c*bin,(c+1)*bin), y in "
                     "[r*bin,(r+1)*bin). Counts only; Stage 11 normalizes + renders."),
        },
        "player_position": {r: player_pos_heatmaps[r] for r in PLAYING_ROLES},
        "ball_landing": ball_grid,
    }

    # --- reliability + warnings ---
    reliability = {
        "synthetic_ball": is_synth,
        "synthetic_gated": ["match.by_end_reason", "match.serve", "match.shot_mix",
                            "match.third_shot", "match.bounce_in_out",
                            "error_attribution", "heatmaps.ball_landing",
                            "players.*.shot_mix", "players.*.serve",
                            "players.*.errors_committed",
                            "players.*.mean_post_speed_ftps"],
        "real_data": ["players.*.position", "players.*.position.movement",
                      "heatmaps.player_position", "team.near", "team.far",
                      "match.rally_length_shots", "match.rally_duration_sec"],
        "pending": ["pending_real_ball.forced_vs_unforced_errors",
                    "pending_real_ball.dink_shot_tolerance",
                    "pending_real_ball.third_shot_drop_outcome",
                    "pending_real_ball.opponent_backhand_targeting"],
    }

    if is_synth:
        warnings.append("ball_source is 'synthetic': all ball-derived metrics "
                        "are PLACEHOLDER. See reliability.synthetic_gated.")
    for r in PLAYING_ROLES:
        if role_contaminated.get(r) and role_to_tids[r]:
            warnings.append(
                f"{r} role_confidence {role_confidence[r]} < floor "
                f"{args.role_conf_floor}: stats may be contaminated by "
                f"adjacent-court tracks (Stage 2.5 known issue).")
    if rallies_doc is None:
        warnings.append("rallies.json unavailable: rally / error / serve metrics "
                        "are empty.")
    if bounces_doc is None:
        warnings.append("bounces.json unavailable: ball-landing heatmap + in/out "
                        "rate are empty.")

    log.info(f"metrics: {len(rallies)} rallies, {len(shots)} shots, "
             f"{len(bounces)} bounces; by_owner={by_owner}")

    out = {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            "classified": str(classified_path),
            "rallies": str(rallies_path) if rallies_doc is not None else None,
            "bounces": str(bounces_path) if bounces_doc is not None else None,
            "players": str(players_path),
            "track_roles": str(track_roles_path) if track_roles_doc is not None else None,
        },
        "ball_source": ball_source,
        "fps": float(fps),
        "params": {
            "heatmap_bin_ft": HEATMAP_BIN_FT,
            "role_conf_floor": args.role_conf_floor,
            "net_y_ft": NET_Y_FT,
            "kitchen_max_dist_ft": KITCHEN_MAX_DIST_FT,
            "baseline_min_dist_ft": BASELINE_MIN_DIST_FT,
        },
        "match": match,
        "error_attribution": error_attribution,
        "players": players_out,
        "team": team,
        "heatmaps": heatmaps,
        "pending_real_ball": pending_real_ball_block(),
        "reliability": reliability,
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
    p = argparse.ArgumentParser(description="Stage 8 — compute metrics")
    p.add_argument("folder", type=Path,
                   help="per-video folder with classified.json, rallies.json, "
                        "bounces.json, players.parquet, track_roles.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--heatmap-bin-ft", type=float, default=HEATMAP_BIN_FT,
                   dest="heatmap_bin_ft")
    p.add_argument("--role-conf-floor", type=float, default=ROLE_CONF_FLOOR,
                   dest="role_conf_floor")
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

"""Stage 2.5 — classify tracks into player roles.

Map ByteTrack track_ids (players.parquet) to logical roles:
user / partner / opp_a / opp_b / noise. A role is a set of track_ids over
time (ByteTrack swaps IDs on crossings). See contract for the full spec.

Noise filter -> near/far side -> seed the user from clicks -> separate
user/partner with the "two people at once" simultaneity constraint +
click-anchored motion continuity + perspective-normalized height (so matching
team kit doesn't break it). Opponents are grouped into two stable IDENTITIES
opp_a / opp_b by the same two-anchor appearance + continuity re-id (NOT position
L/R -- they switch sides), at honestly moderate confidence (far-side crops are
small, so appearance colour is noisier). See SYSTEM_DESIGN.md foundation #2.

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
SEED_EARLY_WINDOW_S = 12.0  # opening window to read the user's starting corner
SEED_CORNER_MIN_SEP_FT = 2.0  # min court_x gap between near players to seed confidently
N_APPEARANCE_SAMPLES = 12   # frames sampled per track for the color signature
HSV_H_BINS, HSV_S_BINS = 12, 8   # upper/lower-body HSV histogram resolution
APP_W, HGT_W, CONT_W = 0.60, 0.25, 0.15  # cue weights in user/partner assignment
OPP_CONF_CAP = 0.75  # far-side appearance is noisier -> cap opponent conf < near-side 0.95
ROLES = ("user", "partner", "opp_a", "opp_b", "noise")
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
    user_inputs = c.get("user_inputs", {}) or {}
    width = geom.get("width_ft", 20.0)
    length = geom.get("length_ft", 44.0)
    return {
        "width_ft": float(width), "length_ft": float(length),
        "net_y": float(length) / 2.0,
        "ppf_near": derived.get("pixels_per_foot_at_near_baseline"),
        "ppf_far": derived.get("pixels_per_foot_at_far_baseline"),
        "fps": video.get("fps") or 30.0,
        "user_baseline": user_inputs.get("user_baseline", "near"),
        "user_starting_corner": user_inputs.get("user_starting_corner"),
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


# --- Appearance (multi-region clothing-color) re-id -------------------------

def _hsv_hist(bgr_crop) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    # Hue + Saturation only (ignore Value) so lighting changes matter less.
    hist = cv2.calcHist([hsv], [0, 1], None, [HSV_H_BINS, HSV_S_BINS],
                        [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_track_appearance(df: pd.DataFrame, track_ids, video_path: Path,
                             log: logging.Logger,
                             n_samples: int = N_APPEARANCE_SAMPLES) -> Dict[int, dict]:
    """Sample frames per track, crop the bbox, and build median upper-/lower-body
    HSV histograms (separate, since teammates often share a top color but differ
    in bottoms). Returns {tid: {"upper": hist, "lower": hist}}; {} if no usable
    video, which makes the caller fall back to height + continuity.

    Frames are fetched in a SINGLE SEQUENTIAL PASS rather than random-seeking to
    each sample: on 4K H.264 a `CAP_PROP_POS_FRAMES` seek re-decodes from the
    nearest keyframe and costs seconds each (~80 seeks was the dominant cost of
    this stage). Sequential grab()/retrieve() — decoding only the sampled frames —
    is ~10x faster and, being exact, avoids OpenCV's imprecise-seek frame drift."""
    try:
        import cv2
    except Exception:
        log.warning("opencv unavailable; appearance re-id disabled")
        return {}
    if not Path(video_path).exists():
        log.warning(f"no {video_path}; appearance re-id disabled "
                    "(falling back to height + continuity)")
        return {}
    want = set(int(t) for t in track_ids)
    cols = ["track_id", "frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    sub = df[df["track_id"].isin(want)][cols]

    # Plan which (tid, bbox) crops to grab on each sampled frame — same sample
    # indices as before, just gathered so we can fetch them in frame order.
    plan: Dict[int, list] = {}
    for tid, t in sub.groupby("track_id"):
        t = t.sort_values("frame")
        if len(t) == 0:
            continue
        idx = np.unique(np.linspace(0, len(t) - 1,
                                    min(n_samples, len(t))).astype(int))
        for r in t.iloc[idx].itertuples(index=False):
            plan.setdefault(int(r.frame), []).append(
                (int(tid), r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2))
    if not plan:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"could not open {video_path}; appearance re-id disabled")
        return {}
    max_frame = max(plan)
    hists: Dict[int, dict] = {int(t): {"uh": [], "lh": []} for t in want}
    fno = -1
    try:
        while fno < max_frame:
            if not cap.grab():         # advance; decode deferred to retrieve()
                break
            fno += 1
            crops = plan.get(fno)
            if not crops:
                continue
            ok, fr = cap.retrieve()    # decode only the sampled frames
            if not ok:
                continue
            H, W = fr.shape[:2]
            for tid, bx1, by1, bx2, by2 in crops:
                x1 = max(0, min(W - 1, int(bx1)))
                x2 = max(x1 + 1, min(W, int(bx2)))
                y1 = max(0, min(H - 1, int(by1)))
                y2 = max(y1 + 1, min(H, int(by2)))
                crop = fr[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 4:
                    continue
                mid = crop.shape[0] // 2
                hists[tid]["uh"].append(_hsv_hist(crop[:mid]))
                hists[tid]["lh"].append(_hsv_hist(crop[mid:]))
    finally:
        cap.release()

    feats: Dict[int, dict] = {}
    for tid, h in hists.items():
        if h["uh"]:
            feats[tid] = {"upper": np.median(np.stack(h["uh"]), axis=0),
                          "lower": np.median(np.stack(h["lh"]), axis=0)}
    log.info(f"appearance: built color signatures for {len(feats)}/{len(want)} "
             "tracks (single sequential pass)")
    return feats


def combine_feats(feat_list) -> Optional[dict]:
    """Average several tracks' signatures (e.g. multiple click-seeded user
    tracks) into one. None if none have features."""
    fl = [f for f in feat_list if f is not None]
    if not fl:
        return None
    return {"upper": np.mean(np.stack([f["upper"] for f in fl]), axis=0),
            "lower": np.mean(np.stack([f["lower"] for f in fl]), axis=0)}


def appearance_sim(fa: Optional[dict], fb: Optional[dict]) -> Optional[float]:
    """Cosine similarity in [0,1] of concatenated upper+lower histograms; None if
    either signature is missing (caller then leans on height + continuity)."""
    if fa is None or fb is None:
        return None
    a = np.concatenate([fa["upper"], fa["lower"]])
    b = np.concatenate([fb["upper"], fb["lower"]])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > EPS else 0.0


def seed_user_by_corner(near: List[dict], df: pd.DataFrame, court: dict,
                        log: logging.Logger):
    """Geometric default seed (no clicks): among near-side tracks, the user is
    the one in `user_starting_corner` during the opening window.

    Court origin (0,0) is the user's near-LEFT corner, so on the near side
    `starting_corner="left"` => smallest court_x, "right" => largest.
    Returns (seed_list, confidence, warnings).
    """
    warnings: List[str] = []
    corner = court.get("user_starting_corner")
    if corner not in ("left", "right"):
        return [], 0.0, [f"user_starting_corner is {corner!r} (expected "
                         f"'left'/'right'); cannot seed the user geometrically"]
    if not near:
        return [], 0.0, ["no near-side tracks to seed the user from"]

    f_min = int(df["frame"].min())
    early_cut = f_min + SEED_EARLY_WINDOW_S * court["fps"]
    early = df[(df["frame"] >= f_min) & (df["frame"] <= early_cut)]
    cand = []  # (track, early_median_court_x)
    for tr in near:
        rows = early[early["track_id"] == tr["track_id"]]
        if len(rows):
            ex = float(np.nanmedian(rows["court_x_ft"].to_numpy()))
            if not np.isnan(ex):
                cand.append((tr, ex))
    if not cand:
        cand = [(tr, tr["med_x"]) for tr in near if not np.isnan(tr["med_x"])]
        if cand:
            warnings.append("no near-side track present in the opening "
                            f"{SEED_EARLY_WINDOW_S:.0f}s window; seeded the user "
                            "from overall median position (less reliable)")
    if not cand:
        return [], 0.0, warnings + ["no near-side track has a usable court_x"]

    # left -> smallest court_x, right -> largest
    cand.sort(key=lambda c: c[1], reverse=(corner == "right"))
    user_tr, user_x = cand[0]
    if len(cand) >= 2:
        sep = abs(user_x - cand[1][1])
        conf = float(min(0.9, 0.5 + 0.4 * min(1.0, sep / court["width_ft"])))
        if sep < SEED_CORNER_MIN_SEP_FT:
            warnings.append(
                f"near players are close in the opening window (dx={sep:.1f}ft); "
                f"user/partner seed by starting corner is ambiguous — add "
                f"user_clicks.json to override if the user role looks wrong")
            conf = min(conf, 0.5)
    else:
        conf = 0.7  # only one near track early (singles, or partner not yet seen)
    log.info(f"geometric user seed: track {user_tr['track_id']} "
             f"(starting_corner={corner}, early court_x={user_x:.1f}ft, "
             f"conf={conf:.2f})")
    return [user_tr], conf, warnings


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
    if court.get("user_baseline") == "far":
        fail("court.json user_baseline='far' is not supported by Stage 2.5 v1 "
             "(near = user/partner pool, far = opponents). Use the near baseline.",
             ValueError)
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

    # 2. Seed the user — clicks override geometry. Default (no clicks) seeds the
    #    user from court.json's user_starting_corner (the intended Stage-1 design).
    seed_warnings: List[str] = []
    seed_user = [tr for tr in near if tr["is_user_frac"] > 0.0]
    if seed_user:
        seed_basis, seed_conf = "click", 0.95
    else:
        seed_user, seed_conf, seed_warnings = seed_user_by_corner(near, df, court, log)
        seed_basis = "starting-corner"
    if not seed_user:
        fail("could not seed the user: no user_clicks.json and geometric seeding "
             "from court.json's user_starting_corner found no near-side player in "
             "the starting corner during the opening window. Provide "
             "user_clicks.json to override.", ValueError)
    for tr in seed_user:
        role[tr["track_id"]] = {"role": "user", "confidence": round(seed_conf, 3),
                                "basis": seed_basis}

    # user identity: frame-weighted mean height, and the set of user-present frames
    hw = [(tr["height_ft"], tr["n"]) for tr in seed_user if not np.isnan(tr["height_ft"])]
    user_height = (sum(h * n for h, n in hw) / sum(n for _, n in hw)) if hw else np.nan
    user_frames = set()
    for tr in seed_user:
        user_frames |= tr["frame_set"]
    user_tracks = list(seed_user)

    # 3. Near non-seed tracks -> user or partner (two-anchor appearance re-id).
    #    Anchor the user on the seed and the partner on the longest near track
    #    simultaneous with the user (provably a different person), then assign
    #    each remaining near track to whichever anchor it resembles most by
    #    appearance (primary) + height + continuity. Global and gap/side-switch
    #    robust, unlike local 4 s continuity linking.
    near_nonseed = [tr for tr in near if tr["track_id"] not in role]
    partner_anchor = None
    for tr in sorted(near_nonseed, key=lambda t: -t["n"]):
        if len(tr["frame_set"] & user_frames) / max(1, tr["n"]) > SIMULTANEITY_MAX:
            partner_anchor = tr
            break

    feat_ids = ({tr["track_id"] for tr in seed_user}
                | {tr["track_id"] for tr in near_nonseed})
    if partner_anchor:
        feat_ids.add(partner_anchor["track_id"])
    # One sequential decode pass for EVERY track we'll need a signature for (near
    # for user/partner, far for opponents — disjoint sets) so the video is decoded
    # once, not twice.
    all_feats = extract_track_appearance(
        df, feat_ids | {t["track_id"] for t in far}, folder / "video.mp4", log)
    feats = all_feats
    user_feat = combine_feats([feats.get(tr["track_id"]) for tr in seed_user])
    partner_feat = feats.get(partner_anchor["track_id"]) if partner_anchor else None
    partner_height = partner_anchor["height_ft"] if partner_anchor else np.nan
    partner_tracks = [partner_anchor] if partner_anchor else []

    def _note_partner(tr, f):
        nonlocal partner_feat, partner_height
        partner_tracks.append(tr)
        if partner_feat is None and f is not None:
            partner_feat = f
        if np.isnan(partner_height) and not np.isnan(tr["height_ft"]):
            partner_height = tr["height_ft"]

    for tr in sorted(near_nonseed, key=lambda t: t["f0"]):
        f = feats.get(tr["track_id"])
        if len(tr["frame_set"] & user_frames) / max(1, tr["n"]) > SIMULTANEITY_MAX:
            # overlaps the user in time -> cannot be the user -> partner
            role[tr["track_id"]] = {"role": "partner", "confidence": 0.8,
                                    "basis": "simultaneous-with-user"}
            _note_partner(tr, f)
            continue
        asim_u, asim_p = appearance_sim(f, user_feat), appearance_sim(f, partner_feat)
        hsim_u = height_sim(tr["height_ft"], user_height)
        hsim_p = height_sim(tr["height_ft"], partner_height)
        cont_u = continuity_score(tr, user_tracks, fps)
        cont_p = continuity_score(tr, partner_tracks, fps)
        if asim_u is not None and asim_p is not None:
            u_score = APP_W * asim_u + HGT_W * hsim_u + CONT_W * cont_u
            p_score = APP_W * asim_p + HGT_W * hsim_p + CONT_W * cont_p
            basis = "appearance+height"
        else:
            u_score = 0.6 * cont_u + 0.4 * hsim_u
            p_score = 0.6 * cont_p + 0.4 * hsim_p
            basis = "continuity+height"
        conf = round(float(min(0.95, 0.5 + abs(u_score - p_score))), 3)
        if u_score >= p_score:
            role[tr["track_id"]] = {"role": "user", "confidence": conf, "basis": basis}
            user_frames |= tr["frame_set"]
            user_tracks.append(tr)
        else:
            role[tr["track_id"]] = {"role": "partner", "confidence": conf, "basis": basis}
            _note_partner(tr, f)

    # 4. Opponents -> two stable IDENTITIES opp_a / opp_b by appearance + continuity.
    #    The far-side mirror of the user/partner two-anchor re-id: anchor opp_a on
    #    the longest far track, opp_b on the longest far track SIMULTANEOUS with it
    #    (provably a different person), then assign each remaining far track to
    #    whichever identity it resembles (appearance primary + height + continuity),
    #    with the can't-be-two-places simultaneity hard constraint. Identity-based,
    #    NOT position L/R -- opponents switch sides, so a court_x split is unstable.
    #    Confidence is honestly MODERATE: far-side crops are small, so appearance
    #    color is noisier than near-side (cap below the near-side 0.95). The
    #    per-frame side, when a stat needs it, is derived downstream from position.
    #    See SYSTEM_DESIGN.md foundation #2.
    if far:
        far_by_len = sorted(far, key=lambda t: -t["n"])
        a_anchor = far_by_len[0]
        b_anchor = None
        for tr in far_by_len[1:]:
            if len(tr["frame_set"] & a_anchor["frame_set"]) / max(1, tr["n"]) > SIMULTANEITY_MAX:
                b_anchor = tr
                break
        opp_feats = all_feats   # extracted in the single pass above
        a_feat = opp_feats.get(a_anchor["track_id"])
        a_height = a_anchor["height_ft"]
        a_tracks = [a_anchor]
        a_frames = set(a_anchor["frame_set"])
        b_feat = opp_feats.get(b_anchor["track_id"]) if b_anchor else None
        b_height = b_anchor["height_ft"] if b_anchor else np.nan
        b_tracks = [b_anchor] if b_anchor else []
        b_frames = set(b_anchor["frame_set"]) if b_anchor else set()
        role[a_anchor["track_id"]] = {"role": "opp_a", "confidence": 0.6,
                                      "basis": "longest-far-anchor"}
        if b_anchor is not None:
            role[b_anchor["track_id"]] = {"role": "opp_b", "confidence": 0.6,
                                          "basis": "simultaneous-with-opp_a"}

        def _note_opp(which, tr, f):
            nonlocal a_feat, a_height, b_feat, b_height
            if which == "opp_a":
                a_tracks.append(tr); a_frames.update(tr["frame_set"])
                if a_feat is None and f is not None: a_feat = f
                if np.isnan(a_height) and not np.isnan(tr["height_ft"]): a_height = tr["height_ft"]
            else:
                b_tracks.append(tr); b_frames.update(tr["frame_set"])
                if b_feat is None and f is not None: b_feat = f
                if np.isnan(b_height) and not np.isnan(tr["height_ft"]): b_height = tr["height_ft"]

        for tr in sorted(far, key=lambda t: t["f0"]):
            tid = tr["track_id"]
            if tid in role:  # an anchor, already assigned
                continue
            f = opp_feats.get(tid)
            sim_a = len(tr["frame_set"] & a_frames) / max(1, tr["n"]) > SIMULTANEITY_MAX
            sim_b = (b_anchor is not None
                     and len(tr["frame_set"] & b_frames) / max(1, tr["n"]) > SIMULTANEITY_MAX)
            if sim_a and not sim_b and b_anchor is not None:
                role[tid] = {"role": "opp_b", "confidence": 0.6, "basis": "simultaneous-with-opp_a"}
                _note_opp("opp_b", tr, f); continue
            if sim_b and not sim_a:
                role[tid] = {"role": "opp_a", "confidence": 0.6, "basis": "simultaneous-with-opp_b"}
                _note_opp("opp_a", tr, f); continue
            asim_a, asim_b = appearance_sim(f, a_feat), appearance_sim(f, b_feat)
            hsim_a, hsim_b = height_sim(tr["height_ft"], a_height), height_sim(tr["height_ft"], b_height)
            cont_a, cont_b = continuity_score(tr, a_tracks, fps), continuity_score(tr, b_tracks, fps)
            if asim_a is not None and asim_b is not None:
                a_score = APP_W * asim_a + HGT_W * hsim_a + CONT_W * cont_a
                b_score = APP_W * asim_b + HGT_W * hsim_b + CONT_W * cont_b
                basis = "appearance+height"
            else:
                a_score = 0.6 * cont_a + 0.4 * hsim_a
                b_score = 0.6 * cont_b + 0.4 * hsim_b
                basis = "continuity+height"
            # far-side appearance is noisier -> cap confidence below near-side 0.95
            conf = round(float(min(OPP_CONF_CAP, 0.5 + abs(a_score - b_score))), 3)
            if b_anchor is None or a_score >= b_score:
                role[tid] = {"role": "opp_a", "confidence": conf, "basis": basis}
                _note_opp("opp_a", tr, f)
            else:
                role[tid] = {"role": "opp_b", "confidence": conf, "basis": basis}
                _note_opp("opp_b", tr, f)

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
             f"opp_a={len(roles_agg['opp_a']['track_ids'])}, "
             f"opp_b={len(roles_agg['opp_b']['track_ids'])}, "
             f"noise={len(noise_ids)}")
    log.info(f"user coverage {was_is_user:.1%} ({seed_basis} seed) -> "
             f"{user_cov:.1%} (roles)")

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
        "warnings": list(seed_warnings),
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

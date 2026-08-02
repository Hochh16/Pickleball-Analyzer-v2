"""PROTOTYPE — the ball-event timeline, tested on one rally with known truth.

Step 1 of docs/DESIGN_REVIEW_2026-08.md. Deliberately falsifiable: if this does not
reproduce the operator's 12 shots / 5 volleys on rally 10, the timeline idea is wrong
and we stop rather than build on it.

WHAT IS DIFFERENT FROM THE PIPELINE (the whole point):
  today  Stage 5 finds shots. Stage 5.5 then finds bounces *inside the intervals
         Stage 5 produced*, capped at one per interval. Stage 6 then infers
         volley = "no bounce since the previous shot". So bounces are a function of
         shots, and volleys are a function of both -- change shot detection and the
         volley count moves although no volley changed. `shots = volleys + bounces`
         is an aspiration that currently fails by 5.
  here   ONE candidate pool -> arbitrated ONCE into a single ordered event stream of
         CONTACT / BOUNCE -> every statistic is a VIEW over that stream. A frame
         cannot be both. Volleys are READ OFF the timeline, never inferred from the
         absence of something. `shots = volleys + bounces` holds BY CONSTRUCTION.

Detection primitives are deliberately UNCHANGED from the validated stages -- this
tests the architecture, not new detectors:
  CONTACT  impulse (turn-rate spike or speed jump) coinciding with a player
  BOUNCE   pixel_y descent-peak with prominence + vertical-velocity flip, away from
           players (an arc apex is a pixel_y MINIMUM, so this ignores apexes)

    python tools/proto_timeline.py --clip data/pb_outdoor2_excerpt --start 1684 --end 2544
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Thresholds copied from the validated stages, at their reference scale.
MIN_TURN_RATE_DEG = 45.0
MIN_SPEED_CHANGE_RATIO = 0.35
MIN_BALL_SPEED_PX_PER_FRAME = 1.5
IMPACT_WINDOW_FRAMES = 6
VELOCITY_WINDOW_FRAMES = 3
ASSOC_BBOX_HEIGHT_FRAC = 0.5
ASSOC_MAX_PX = 120.0
ASSOC_MAX_PX_MIN = 30.0
BOUNCE_PROMINENCE_PX = 9.0
YFLIP_FLOOR = 0.3
REFERENCE_WIDTH_PX = 1920.0
REFERENCE_FPS = 30.0
EPS = 1e-9

# Operator truth for rally 10 (docs/ACCURACY_LEDGER "Match-clip validation").
TRUTH = {"contacts": 12, "volleys": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args()
    D = Path(args.clip)

    court = json.load(open(D / "court.json", encoding="utf-8"))
    vid = court.get("video", {}) or {}
    fps = float(vid.get("fps") or 60.0)
    res = float(vid.get("frame_width") or 3840.0) / REFERENCE_WIDTH_PX
    fs = fps / REFERENCE_FPS
    W = int(round(IMPACT_WINDOW_FRAMES * fs))
    k = int(round(VELOCITY_WINDOW_FRAMES * fs))
    min_speed = MIN_BALL_SPEED_PX_PER_FRAME * res
    prom = BOUNCE_PROMINENCE_PX * res
    yflip = YFLIP_FLOOR * res
    amax, amin = ASSOC_MAX_PX * res, ASSOC_MAX_PX_MIN * res

    ball = pd.read_parquet(D / "ball.parquet").sort_values("frame_idx")
    n = int(ball["frame_idx"].max()) + 1
    fx = np.full(n, np.nan)
    fy = np.full(n, np.nan)
    known = np.zeros(n, bool)
    for r in ball.itertuples():
        i = int(r.frame_idx)
        if r.visible or r.interpolated:
            fx[i], fy[i], known[i] = r.pixel_x, r.pixel_y, True

    roles = {int(t): i["role"] for t, i in
             json.load(open(D / "track_roles.json", encoding="utf-8"))["track_roles"].items()}
    players = pd.read_parquet(D / "players.parquet")
    players = players[players.track_id.map(lambda t: roles.get(int(t), "noise")) != "noise"]
    by_frame = {}
    for r in players.itertuples():
        by_frame.setdefault(int(r.frame), []).append(
            (int(r.track_id), r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2))

    # WRIST positions — the paddle hand. Stage 5 associates on these first and only
    # falls back to the bbox, and that matters enormously: bbox distance reads ZERO
    # whenever the ball merely overlaps a player's box, which at 4K makes almost every
    # impulse look like a contact. A first version of this prototype used bbox only
    # and produced 38 contacts against a truth of 12.
    WRIST_VIS_FLOOR = 0.5
    pos = pd.read_parquet(D / "poses.parquet",
                          columns=["frame", "track_id",
                                   "left_wrist_x_px", "left_wrist_y_px", "left_wrist_visibility",
                                   "right_wrist_x_px", "right_wrist_y_px", "right_wrist_visibility"])
    wrists = {}
    for r in pos.itertuples():
        w = []
        if r.left_wrist_visibility >= WRIST_VIS_FLOOR and not math.isnan(r.left_wrist_x_px):
            w.append((float(r.left_wrist_x_px), float(r.left_wrist_y_px)))
        if r.right_wrist_visibility >= WRIST_VIS_FLOOR and not math.isnan(r.right_wrist_x_px):
            w.append((float(r.right_wrist_x_px), float(r.right_wrist_y_px)))
        if w:
            wrists[(int(r.frame), int(r.track_id))] = w

    def nearest_player(f, bx, by):
        """Closest participant: wrist first (the paddle hand), bbox as fallback."""
        best = (None, float("inf"), 0.0)
        for tid, x1, y1, x2, y2 in by_frame.get(f, []):
            radius = min(max(ASSOC_BBOX_HEIGHT_FRAC * max(1.0, y2 - y1), amin), amax)
            ws = wrists.get((f, tid))
            if ws:
                d = min(math.hypot(bx - wx, by - wy) for wx, wy in ws)
            else:
                dx = max(x1 - bx, 0, bx - x2)
                dy = max(y1 - by, 0, by - y2)
                d = math.hypot(dx, dy)
            if d < best[1]:
                best = (tid, d, radius)
        return best

    def vel(i):
        a = b = None
        for j in range(1, k + 1):
            if i - j >= 0 and known[i - j]:
                a = i - j
                break
        for j in range(0, k + 1):
            if i + j < n and known[i + j]:
                b = i + j
                break
        if a is None or b is None or b == a:
            return None
        return np.array([(fx[b] - fx[a]) / (b - a), (fy[b] - fy[a]) / (b - a)])

    # ---- candidate pool -----------------------------------------------------
    lo, hi = args.start, args.end
    cands = []          # (frame, kind, score)

    # CONTACT candidates: impulse signature
    for i in range(max(1, lo), min(n - 1, hi + 1)):
        if not known[i]:
            continue
        v_in, v_out = vel(i), vel(i + 1)
        if v_in is None or v_out is None:
            continue
        s_in, s_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if max(s_in, s_out) < min_speed:
            continue
        cos = float(np.dot(v_in, v_out) / max(s_in * s_out, EPS))
        turn = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        sratio = abs(s_out - s_in) / max(s_in, s_out, EPS)
        if turn >= MIN_TURN_RATE_DEG or sratio >= MIN_SPEED_CHANGE_RATIO:
            cands.append((i, "CONTACT", max(turn / 180.0, min(1.0, sratio))))

    # BOUNCE candidates: pixel_y descent-peak + prominence + y-flip
    ki = np.where(known)[0]
    yi = np.interp(np.arange(n), ki, fy[ki]) if len(ki) >= 2 else fy.copy()
    ys = np.convolve(yi, np.ones(3) / 3.0, mode="same")
    for i in range(max(W, lo), min(n - W, hi + 1)):
        if ys[i] < ys[i - W:i + W + 1].max() - EPS:
            continue
        descent = ys[i] - ys[i - W:i].min()
        rebound = ys[i] - ys[i + 1:i + W + 1].min()
        if descent < prom or rebound < prom:
            continue
        if not known[max(0, i - 2):i + 3].any():
            continue
        a, b = max(0, i - k), min(n - 1, i + k)
        if i in (a, b):
            continue
        if not ((ys[i] - ys[a]) / (i - a) > yflip and (ys[b] - ys[i]) / (b - i) < -yflip):
            continue
        cands.append((i, "BOUNCE", float(min(descent, rebound))))

    # ---- ARBITRATE ONCE into a single ordered stream ------------------------
    # A frame is one physical event. Where both primitives fire, player proximity
    # decides: a paddle contact happens AT a player, a ground bounce away from one.
    # Today this arbitration is implicit and one-directional (Stage 5 wins, then
    # Stage 5.5 excludes near-shot candidates), which is what lets the two layers
    # disagree and breaks `shots = volleys + bounces`.
    cands.sort(key=lambda c: -c[2])
    events = []
    for f, kind, score in cands:
        if any(abs(f - e["frame"]) <= W for e in events):
            continue
        tid, dist, radius = nearest_player(f, fx[f], fy[f]) if known[f] else (None, 1e9, 0)
        at_player = tid is not None and dist < radius
        events.append({"frame": f, "kind": "CONTACT" if at_player else "BOUNCE",
                       "primitive": kind, "track_id": tid, "dist": dist,
                       "role": roles.get(tid, "?") if tid else None, "score": score})
    events.sort(key=lambda e: e["frame"])

    # ---- VIEWS over the timeline -------------------------------------------
    contacts = [e for e in events if e["kind"] == "CONTACT"]
    bounces = [e for e in events if e["kind"] == "BOUNCE"]
    for i, c in enumerate(contacts):
        prev = contacts[i - 1]["frame"] if i else None
        c["is_volley"] = (i > 0 and not any(prev < b["frame"] < c["frame"] for b in bounces))
    volleys = [c for c in contacts if c.get("is_volley")]

    print(f"\nPROTOTYPE TIMELINE — {D.name} frames {lo}-{hi} "
          f"({(hi - lo) / fps:.1f}s @ {fps:g}fps)\n")
    print(f"  {'frame':>6s} {'t':>6s}  {'event':8s} {'via':8s} {'role':8s} "
          f"{'dist_px':>8s}  note")
    for e in events:
        t = (e["frame"] - lo) / fps
        note = ""
        if e["kind"] == "CONTACT":
            note = "VOLLEY" if e.get("is_volley") else "(bounced before it)"
        print(f"  {e['frame']:>6d} {t:>6.2f}  {e['kind']:8s} {e['primitive']:8s} "
              f"{str(e['role'] or '-'):8s} {e['dist']:>8.0f}  {note}")

    nc, nb, nv = len(contacts), len(bounces), len(volleys)
    print(f"\n  contacts {nc}   bounces {nb}   volleys {nv}")
    print(f"  IDENTITY  contacts == volleys + bounces:  "
          f"{nc} vs {nv}+{nb}={nv + nb}  "
          f"{'HOLDS BY CONSTRUCTION' if nc == nv + nb else f'OFF BY {nc - nv - nb:+d}'}")
    print(f"\n  vs operator truth (rally 10): contacts {nc}/{TRUTH['contacts']}, "
          f"volleys {nv}/{TRUTH['volleys']}, bounces {nb}/"
          f"{TRUTH['contacts'] - TRUTH['volleys']} (implied by the identity)\n")


if __name__ == "__main__":
    main()

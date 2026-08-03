"""Validation render for PLAYER IDENTITY (Stage 2.5 roles).

The whole per-player report rests on one question nobody has ever checked with
their eyes: **is the player we call `user` actually the user?** Stage 2.5 seeds
the user geometrically from `user_starting_corner` over an opening window, and on
`pb_5_minute_outdoor-2` it emits `near players are close in the opening window
(dx=0.2ft); user/partner seed is ambiguous`. If that seed is flipped, every
"your" stat in the report belongs to the partner -- and no smoke test can tell,
because a flipped assignment is perfectly self-consistent.

So this is the cheap test before any Stage 2.5 rebuild. Two parts:

1. **Automatic** (no operator effort) -- the checks code CAN make:
   role coverage per frame, and *impossible* frames where one role is carried by
   two tracks at once. Printed as a report.
2. **Rendered** (operator's eye) -- contact sheets of sampled frames with every
   tracked player boxed and labelled with its role, track_id, confidence and
   basis. Sampled at the DETECTED SERVE frames by default, because that is where
   operator ground truth already exists ("0:48 partner", "3:39 opp"), plus a
   uniform sweep to catch a mid-clip swap.

    python tools/verify_identity.py --clip data/pb_5_minute_outdoor-2

Writes PNGs into <clip>/_identity_check/ and prints the consistency report.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PLAY_ROLES = ("user", "partner", "opp_a", "opp_b")

# BGR, deliberately far apart so a swap is obvious at a glance
ROLE_COLOR = {
    "user": (0, 255, 0),
    "partner": (255, 200, 0),
    "opp_a": (255, 0, 255),
    "opp_b": (0, 140, 255),
}


def load(clip: Path, name: str):
    return json.load(open(clip / name, encoding="utf-8"))


def draw_court(img, court: dict, envelope: bool = True) -> None:
    """Project THE USER'S court onto the frame.

    Non-negotiable: at a multi-court venue you cannot tell by eye which court is
    which -- the user's court is a thin foreshortened sliver near the top of frame
    while a neighbouring court can dominate the view. Judging "is that player on our
    court?" without these lines drawn produces confident, wrong answers.
    """
    H = np.array(court["homography"]["court_to_image"])
    w = float(court.get("court_geometry_feet", {}).get("width_ft", 20.0))
    ln = float(court.get("court_geometry_feet", {}).get("length_ft", 44.0))
    kd = float(court.get("court_geometry_feet", {}).get("kitchen_depth_ft", 7.0))

    def c2i(x, y):
        v = H @ np.array([x, y, 1.0])
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    def line(a, b, col, th):
        cv2.line(img, c2i(*a), c2i(*b), col, th)

    white, net_col, kit = (255, 255, 255), (0, 220, 255), (200, 200, 60)
    for y in (0.0, ln):                       # baselines
        line((0, y), (w, y), white, 4)
    for x in (0.0, w):                        # sidelines
        line((x, 0), (x, ln), white, 4)
    line((0, ln / 2), (w, ln / 2), net_col, 5)              # net
    for y in (ln / 2 - kd, ln / 2 + kd):                     # kitchen lines
        line((0, y), (w, y), kit, 3)
    cv2.putText(img, "YOUR far baseline", c2i(1.0, ln - 1.5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
    cv2.putText(img, "NET", c2i(0.3, ln / 2 - 0.8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, net_col, 2)
    # The operator's PLAYER play envelope — players legitimately serve from behind the
    # baseline and chase wide, so this, not the rectangle, bounds "is this person on
    # our court". It does NOT apply to the BALL: operator 2026-08-02, "the ball must
    # bounce within the court parameters" (22x20 each side) — anything outside is OUT.
    # So it is suppressed wherever the question is about the ball (e.g. shot labelling),
    # since showing it there invites judging a landing against the wrong boundary.
    if envelope:
        ex, ey = 5.0, 15.0
        env = [(-ex, -ey), (w + ex, -ey), (w + ex, ln + ey), (-ex, ln + ey)]
        for a, b in zip(env, env[1:] + env[:1]):
            line(a, b, (120, 255, 120), 2)


def tc(frame: int, fps: float) -> str:
    t = frame / fps
    return f"{int(t // 60)}:{t % 60:04.1f}"


def consistency_report(players: pd.DataFrame, fps: float) -> None:
    """The checks code can make alone -- no operator, no video."""
    played = players[players.role.isin(PLAY_ROLES)]
    per_frame = played.groupby(["frame", "role"]).track_id.nunique().unstack(fill_value=0)
    n_frames = len(per_frame)
    print(f"\n=== automatic consistency ({n_frames} frames with any role) ===")
    print(f"{'role':10s} {'tracks':>7s} {'present':>9s} {'absent':>9s} {'DUPLICATE':>10s}")
    for r in PLAY_ROLES:
        if r not in per_frame:
            print(f"{r:10s} {'-':>7s}   role never assigned")
            continue
        col = per_frame[r]
        n_tracks = played[played.role == r].track_id.nunique()
        dup = int((col > 1).sum())
        print(f"{r:10s} {n_tracks:7d} {int((col >= 1).sum()):9d} "
              f"{int((col == 0).sum()):9d} {dup:10d}")
    print("  DUPLICATE = frames where one role is on two tracks at once "
          "(provably wrong; a person can't be in two places).")
    print("  NOTE: zero duplicates does NOT mean identity is correct -- a fully "
          "swapped user/partner is self-consistent. That is what the render is for.")


def role_timeline(players: pd.DataFrame, roles_meta: dict, fps: float,
                  role: str, top: int = 12) -> None:
    """Which track_ids carry a role, and when -- shows fragmentation + hand-offs."""
    sub = players[players.role == role]
    if sub.empty:
        return
    spans = (sub.groupby("track_id")
                .agg(first=("frame", "min"), last=("frame", "max"), n=("frame", "size"))
                .sort_values("n", ascending=False))
    print(f"\n--- {role}: {len(spans)} track_ids, {int(spans.n.sum())} rows "
          f"(showing {min(top, len(spans))} largest) ---")
    for tid, row in spans.head(top).iterrows():
        meta = roles_meta.get(str(tid), {})
        print(f"  tid {tid:>5}  {tc(row['first'], fps):>7s}-{tc(row['last'], fps):<7s} "
              f"{int(row['n']):6d} rows  conf={meta.get('confidence', 0):.2f} "
              f"{meta.get('basis', '?')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--sweep", type=int, default=10,
                    help="uniformly-spaced extra frames across the clip")
    ap.add_argument("--at", default="",
                    help="comma list of extra timestamps (m:ss or seconds)")
    ap.add_argument("--no-serves", action="store_true",
                    help="skip the detected-serve frames")
    ap.add_argument("--width", type=int, default=1600, help="output width (px)")
    ap.add_argument("--no-render", action="store_true",
                    help="consistency report + timelines only, no video decode")
    args = ap.parse_args()
    D = Path(args.clip)

    court = load(D, "court.json")
    roles_meta = load(D, "track_roles.json")["track_roles"]
    players = pd.read_parquet(D / "players.parquet")
    players["role"] = players.track_id.astype(str).map(
        lambda t: roles_meta.get(t, {}).get("role", "noise"))

    shots = load(D, "shots.json")
    fps = float(shots.get("fps") or 60.0)

    consistency_report(players, fps)
    for r in PLAY_ROLES:
        role_timeline(players, roles_meta, fps, r)

    if args.no_render:
        return

    # --- pick the frames to show the operator -------------------------------
    picks = []  # (frame, label)
    if not args.no_serves:
        for s in shots["shots"]:
            if s.get("is_serve"):
                meta = roles_meta.get(str(s["track_id"]), {})
                picks.append((int(s["frame"]),
                              f"SERVE by {meta.get('role', '?')} "
                              f"(tid {s['track_id']}, conf {meta.get('confidence', 0):.2f}, "
                              f"{s.get('hitter_side', '?')})"))
    for tok in filter(None, (t.strip() for t in args.at.split(","))):
        sec = (int(tok.split(":")[0]) * 60 + float(tok.split(":")[1])
               if ":" in tok else float(tok))
        picks.append((int(round(sec * fps)), "operator-requested"))
    if args.sweep > 0:
        last = int(players.frame.max())
        for f in np.linspace(last * 0.02, last * 0.98, args.sweep).astype(int):
            picks.append((int(f), "sweep"))
    picks.sort()

    out_dir = D / "_identity_check"
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(D / "video.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {D / 'video.mp4'}")
    print(f"\n=== rendering {len(picks)} frames -> {out_dir} ===")

    by_frame = {f: g for f, g in players[players.role.isin(PLAY_ROLES)].groupby("frame")}

    for frame, label in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            print(f"  !! could not read frame {frame}")
            continue
        draw_court(img, court)
        rows = by_frame.get(frame)
        seen = []
        if rows is not None:
            for _, r in rows.iterrows():
                col = ROLE_COLOR[r.role]
                p1 = (int(r.bbox_x1), int(r.bbox_y1))
                p2 = (int(r.bbox_x2), int(r.bbox_y2))
                cv2.rectangle(img, p1, p2, col, 5)
                meta = roles_meta.get(str(r.track_id), {})
                txt = (f"{r.role.upper()}  tid{r.track_id}  "
                       f"c{meta.get('confidence', 0):.2f}  {meta.get('basis', '?')}")
                # label above the box, or inside it if we're at the top edge
                ty = p1[1] - 12 if p1[1] > 40 else p2[1] + 34
                cv2.putText(img, txt, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 0), 7)
                cv2.putText(img, txt, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, col, 3)
                seen.append(r.role)

        h, w = img.shape[:2]
        img = cv2.resize(img, (args.width, int(h * args.width / w)))
        missing = [r for r in PLAY_ROLES if r not in seen]
        banner = (f"f{frame}  {tc(frame, fps)}   {label}"
                  + (f"   MISSING: {','.join(missing)}" if missing else ""))
        bar = np.zeros((44, img.shape[1], 3), np.uint8)
        cv2.putText(bar, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (255, 255, 255), 2)
        # legend so the operator never has to guess a colour
        x = img.shape[1] - 420
        for r in PLAY_ROLES:
            cv2.putText(bar, r, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        ROLE_COLOR[r], 2)
            x += 105
        img = np.vstack([bar, img])

        name = f"{frame:06d}_{label.split()[0].lower().strip('(')}.png"
        cv2.imwrite(str(out_dir / name), img)
        print(f"  wrote {name}  ({label})")

    cap.release()
    print(f"\ndone -> {out_dir}")
    print("OPERATOR CHECK: in every frame, is the GREEN box you, and the "
          "CYAN box your partner? A consistent swap is the failure we are hunting.")


if __name__ == "__main__":
    main()

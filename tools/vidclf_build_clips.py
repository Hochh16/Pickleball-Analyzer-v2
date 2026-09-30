r"""Phase 0 of the video-classifier experiment: one short labelled clip per Stage 5 candidate.

See docs/VIDEO_CLASSIFIER_SCOPE.md. The question the classifier will answer is "was THIS moment a
paddle strike?", from the pixels around it. So for every impulse candidate Stage 5 considers,
before any of its filters run, this cuts ~0.5 s of video centred on the candidate, cropped around
the ball and resized to 224x224, and labels it from the operator's truth store.

Inputs, per clip:
  data/_vidclf/work/<clip>/shot_discards.json   candidates, regenerated on a COPY of the clip's
                                                inputs (the real folders' Stage 5 output is left
                                                untouched so court A's review numbering holds)
  data/<clip>/session.json, court.json          the source video and the court geometry
  docs/truth/                                   via tools.truth_store.known()

Outputs (data/ is gitignored):
  data/_vidclf/clips/<clip>/<clip_id>.mp4       16 frames, 224x224
  data/_vidclf/clips/manifest.csv               one row per clip, label and provenance
  data/_vidclf/clips/contact_sheet_<clip>.jpg   a visual check of real and junk crops

Labels are matched ONE-TO-ONE, shortest pair first, within 0.35 s -- the same matcher every
earlier experiment used, so results stay comparable. Court A's clips are built and marked
"heldout"; nothing in this script reads its labels to choose anything.

    python -m tools.vidclf_build_clips
    python -m tools.vidclf_build_clips --clip pb_3_min_indoor_1_court_c
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.truth_store import known  # noqa: E402

DEV = ["pb_5_minute_outdoor-7", "pb_3_min_indoor_1_court_c", "pb_3_min_indoor_1_court_b"]
HELD_OUT = ["pb_5_min_indoor_1_court_a"]
WORK = Path("data/_vidclf/work")
OUT = Path("data/_vidclf/clips")

OFFSETS = list(range(-16, 16, 2))   # 16 frames, ~0.5 s at 60 fps, centred on the candidate
SIZE = 224
CROP_FT = 14.0                      # crop side in court feet at the candidate's depth
CROP_MIN_PX, CROP_MAX_PX = 320, 2000
MATCH_S = 0.35


# --- pure helpers (tested) ---------------------------------------------------------------------
def one_to_one(a: Sequence[float], b: Sequence[float], tol: float) -> Dict[int, int]:
    """{index in a: index in b} for pairs within tol, shortest first, each index used once."""
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                   if abs(x - y) <= tol)
    ui, uj, out = set(), set(), {}
    for _, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        out[i] = j
    return out


def pixels_per_foot_at(y_px: float, court: dict) -> float:
    """Court scale at an image row, interpolated between the far and near baselines.

    The camera sits behind the near baseline, so the scale grows toward the bottom of the image:
    a far player is ~45 px/ft at 4K and a near one ~100. A fixed crop would show a far player as
    a speck and cut a near player off at the knees.
    """
    ys = sorted(float(p[1]) for p in court["user_inputs"]["court_corners_image"])
    far_y, near_y = (ys[0] + ys[1]) / 2.0, (ys[2] + ys[3]) / 2.0
    d = court["derived"]
    near, far = float(d["pixels_per_foot_at_near_baseline"]), float(d["pixels_per_foot_at_far_baseline"])
    if near_y == far_y:
        return near
    k = (y_px - far_y) / (near_y - far_y)
    return far + max(-0.5, min(1.5, k)) * (near - far)


def crop_box(x: float, y: float, court: dict, frame_w: int, frame_h: int) -> Tuple[int, int, int]:
    """(left, top, side) of a square crop centred on the candidate, kept inside the frame."""
    side = int(round(CROP_FT * pixels_per_foot_at(y, court)))
    side = max(CROP_MIN_PX, min(CROP_MAX_PX, side, frame_w, frame_h))
    left = int(round(x - side / 2))
    top = int(round(y - side / 2))
    left = max(0, min(frame_w - side, left))
    top = max(0, min(frame_h - side, top))
    return left, top, side


def label_candidates(cand_times: Sequence[float], truth_shots: List[dict]) -> List[Optional[dict]]:
    """For each candidate, the truth shot it is matched to, or None (junk)."""
    tt = [float(s["t_sec"]) for s in truth_shots]
    m = one_to_one(cand_times, tt, MATCH_S)
    return [truth_shots[m[i]] if i in m else None for i in range(len(cand_times))]


# --- building ----------------------------------------------------------------------------------
def build_clip(clip: str, role: str, writer_rows: list, log=print) -> int:
    import cv2
    disc = json.loads((WORK / clip / "shot_discards.json").read_text(encoding="utf-8"))
    cands = sorted(disc["candidates"], key=lambda c: int(c["frame"]))
    court = json.loads((Path("data") / clip / "court.json").read_text(encoding="utf-8"))
    session = json.loads((Path("data") / clip / "session.json").read_text(encoding="utf-8"))
    video = session["video_path"]
    truth = [s for s in known(Path("data") / clip)["shots"] if not s.get("not_a_shot")]
    labels = label_candidates([float(c["t_sec"]) for c in cands], truth)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # which candidates need which frames; each candidate collects its 16 crops as they stream by
    need: Dict[int, List[int]] = {}
    boxes, stacks, last = {}, {}, {}
    for i, c in enumerate(cands):
        x, y = (c.get("impact_pixel_xy") or [None, None])[:2]
        if x is None:
            continue
        boxes[i] = crop_box(float(x), float(y), court, fw, fh)
        frames = [min(max(0, int(c["frame"]) + o), n_frames - 1) for o in OFFSETS]
        stacks[i] = [None] * len(frames)
        last[i] = max(frames)
        for k, f in enumerate(frames):
            need.setdefault(f, []).append((i, k))

    out_dir = OUT / clip
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    written, t0, f = 0, time.time(), 0
    pending = set(stacks)
    max_need = max(need) if need else -1
    while f <= max_need:
        if f in need:
            ok, img = cap.read()
            if not ok:
                break
            for i, k in need[f]:
                left, top, side = boxes[i]
                stacks[i][k] = cv2.resize(img[top:top + side, left:left + side], (SIZE, SIZE),
                                          interpolation=cv2.INTER_AREA)
        else:
            if not cap.grab():      # decode-free skip for frames nobody needs
                break
        done = [i for i in pending if last[i] <= f]
        for i in done:
            pending.discard(i)
            frames = [s for s in stacks[i] if s is not None]
            if len(frames) != len(OFFSETS):
                stacks[i] = None
                continue
            c = cands[i]
            cid = f"{clip}_{int(c['frame']):06d}"
            w = cv2.VideoWriter(str(out_dir / f"{cid}.mp4"), fourcc, 30.0, (SIZE, SIZE))
            for fr in frames:
                w.write(fr)
            w.release()
            t = labels[i]
            writer_rows.append({
                "clip_id": cid, "video": clip, "role": role, "frame": int(c["frame"]),
                "t_sec": float(c["t_sec"]), "label": int(t is not None),
                "truth_t_sec": "" if t is None else float(t["t_sec"]),
                "truth_type": "" if t is None else (t.get("type") or ""),
                "side": c.get("hitter_side") or "", "no_player": int(bool(c.get("no_player"))),
                "crop_left": boxes[i][0], "crop_top": boxes[i][1], "crop_side": boxes[i][2]})
            stacks[i] = None
            written += 1
        f += 1
    cap.release()
    n_real = sum(1 for r in writer_rows if r["video"] == clip and r["label"] == 1)
    log(f"{clip:32s} [{role}] {written} clips ({n_real} real) of {len(cands)} candidates "
        f"in {time.time() - t0:.0f}s")
    return written


def contact_sheet(clip: str, rows: list, per_class: int = 6) -> Optional[Path]:
    """Middle frame of a few real and a few junk clips, so the crops can be checked by eye."""
    import cv2
    tiles = []
    for lab in (1, 0):
        pick = [r for r in rows if r["video"] == clip and r["label"] == lab][:per_class]
        row = []
        for r in pick:
            cap = cv2.VideoCapture(str(OUT / clip / f"{r['clip_id']}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, len(OFFSETS) // 2)
            ok, img = cap.read()
            cap.release()
            if ok:
                cv2.putText(img, "REAL" if lab else "junk", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0) if lab else (0, 0, 255), 2)
                row.append(img)
        while len(row) < per_class:
            row.append(np.zeros((SIZE, SIZE, 3), np.uint8))
        tiles.append(np.hstack(row))
    p = OUT / f"contact_sheet_{clip}.jpg"
    cv2.imwrite(str(p), np.vstack(tiles))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", default=None)
    a = ap.parse_args(argv)
    clips = a.clip or (DEV + HELD_OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list = []
    for c in clips:
        build_clip(c, "heldout" if c in HELD_OUT else "dev", rows)
        contact_sheet(c, rows)
    man = OUT / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dev = [r for r in rows if r["role"] == "dev"]
    print(f"\nwrote {man}: {len(rows)} clips; dev {len(dev)} "
          f"({sum(r['label'] for r in dev)} real), held out {len(rows) - len(dev)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

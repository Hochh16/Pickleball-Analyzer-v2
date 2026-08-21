"""Propose WHICH frames to label next: the frames where the detector is UNRELIABLE.

Corrected 2026-08-20, after the first batch. This tool was built to find balls the detector
was MISSING. It does not find those, and they barely exist: measured against the operator's
labels the detector's recall is **94-96%** with a median error of ~4 px, and dropout runs
bracketed by confident detections either side (where the ball cannot have teleported, so it
was there) total only ~226 frames in a whole 5-minute clip.

What it actually finds is the real defect — **the detector inventing a ball where there is
none**, which it does in 25% of frames indoors and 49% outdoors. That single failure is
behind the wrong-object latch, the between-point false shots, and rally-end precision.

Low confidence is the signal, and it is a strong one:

| | median confidence |
|---|---|
| known hallucinations (n=388) | **0.33** |
| true detections (n=2,890) | **0.84** |

Choosing the threshold is a purity/coverage trade, measured across all three labelled clips:

| conf below | hallucinations caught | true detections swept in | purity |
|---|---|---|---|
| 0.35 | 54% | 4% | 66% |
| **0.40** | **64%** | **5%** | **62%** |
| 0.50 | 77% | 10% | 50% |
| 0.60 | 86% | 19% | 38% |

At 0.40 roughly two thirds of proposed frames are genuine negatives, and the third that turn
out to hold a ball are hard positives worth having anyway. Both labels teach something.

Ranges are contiguous because `label_ball.py` walks frames in order; jumping around costs far
more than the label itself. Expect to press the "not visible" key for most of a batch — that
is the point, not a sign the batch is wrong.

Usage:
    python -m tools.propose_labels data/pb_5_minute_outdoor-7 --labels data/pb_5min
    python -m tools.propose_labels data/pb_5_minute_outdoor-7 --labels data/pb_5min --budget 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOW_CONF = 0.40         # 62% of detections below this are hallucinations (measured)
GAP_MIN_FRAMES = 4      # a dropout shorter than this is normal occlusion, not a failure
MERGE_GAP_S = 1.0       # stretches closer than this are merged into one range
MIN_RANGE_S = 0.5       # ignore ranges too short to be worth navigating to
MAX_DROPOUT = 0.98      # almost pure dropout carries little to label either way
SAMPLE_EVERY = 3        # label_ball.py's stride, used to estimate the label count


def load_labelled(label_dir: Path) -> set[int]:
    p = label_dir / "ball_labels.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    return {int(x["frame_idx"]) for x in d.get("labels", [])}


def rally_windows(clip: Path) -> list[tuple[float, float]]:
    p = clip / "rallies.json"
    if not p.exists():
        return []
    out = []
    for r in json.loads(p.read_text(encoding="utf-8")).get("rallies", []):
        a = float(r.get("start_t_sec", 0.0))
        out.append((a, a + float(r.get("duration_sec") or 0.0)))
    return out


def propose(clip: Path, labelled: set[int], fps: float,
            windows: list[tuple[float, float]] | None = None) -> list[dict]:
    ball = pd.read_parquet(clip / "ball.parquet").sort_values("frame_idx")
    fr = ball.frame_idx.to_numpy()
    vis = ball.visible.to_numpy()
    conf = ball.confidence.to_numpy()

    trouble = np.zeros(len(fr), bool)
    # low-confidence detections
    trouble |= vis & (np.nan_to_num(conf, nan=0.0) < LOW_CONF)
    # dropouts: runs of not-visible at least GAP_MIN_FRAMES long
    i = 0
    while i < len(fr):
        if not vis[i]:
            j = i
            while j + 1 < len(fr) and not vis[j + 1]:
                j += 1
            if (j - i + 1) >= GAP_MIN_FRAMES:
                trouble[i:j + 1] = True
            i = j + 1
        else:
            i += 1

    # already-labelled frames are not worth proposing again
    if labelled:
        lab = np.array([f in labelled for f in fr], bool)
        # a labelled REGION, not just the exact sampled frames
        lo, hi = min(labelled), max(labelled)
        trouble &= ~((fr >= lo) & (fr <= hi))
        del lab

    ranges: list[dict] = []
    i = 0
    merge = int(MERGE_GAP_S * fps)
    while i < len(fr):
        if trouble[i]:
            j = i
            while j + 1 < len(fr) and (trouble[j + 1] or fr[j + 1] - fr[j] <= merge):
                if not trouble[j + 1] and (fr[j + 1] - fr[i]) > merge * 3:
                    break
                j += 1
            a, b = int(fr[i]), int(fr[j])
            # NOT restricted to rally windows any more. The first batch was, on the theory that
            # in-play frames were the valuable ones; it landed on a stretch that was 82%
            # hallucination, which is exactly what is wanted. Between-point frames are where
            # the detector invents balls, so excluding them excluded the target.
            if (b - a) / fps >= MIN_RANGE_S:
                seg = slice(i, j + 1)
                ranges.append({
                    "start_frame": a, "end_frame": b,
                    "start_s": round(a / fps, 2), "end_s": round(b / fps, 2),
                    "seconds": round((b - a) / fps, 2),
                    "n_labels": int((b - a) // SAMPLE_EVERY) + 1,
                    "dropout_frac": round(float(np.mean(~vis[seg])), 2),
                    "median_conf": round(float(np.nanmedian(np.where(vis[seg], conf[seg], np.nan)))
                                         if vis[seg].any() else 0.0, 2)})
            i = j + 1
        else:
            i += 1
    # A stretch that is almost entirely dropout is usually the ball genuinely gone -- off
    # camera, or lying still after a point -- not the detector failing to find it. Those
    # frames still teach "not visible", but they carry far less per label than a stretch
    # where the model DOES see something and is unsure, so they go last.
    for r in ranges:
        r["likely_absent"] = r["dropout_frac"] > MAX_DROPOUT
    # Lowest confidence first: that is where the detector is most likely inventing a ball.
    ranges.sort(key=lambda r: (r["likely_absent"], r["median_conf"]))
    return ranges


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path, help="folder with ball.parquet (an analysed session)")
    ap.add_argument("--labels", type=Path, default=None,
                    help="folder holding ball_labels.json (defaults to the clip)")
    ap.add_argument("--budget", type=int, default=300,
                    help="stop once this many labels have been proposed")
    ap.add_argument("--video", type=Path, default=None,
                    help="source video, for printing a ready-to-run label_ball command")
    a = ap.parse_args(argv)

    court = json.loads((a.clip / "court.json").read_text(encoding="utf-8"))
    fps = float(court["video"]["fps"])
    labelled = load_labelled(a.labels or a.clip)
    ranges = propose(a.clip, labelled, fps, None)

    # Merge stretches that are close together. Each separate labelling run costs a fresh
    # sequential seek -- minutes on a 4K clip -- so a few long ranges beat many short ones,
    # even though some easy frames come along for the ride.
    merge_frames = int(4.0 * fps)
    picked: list[dict] = []
    for r in sorted(ranges, key=lambda x: (x["likely_absent"], x["median_conf"])):
        if sum(x["n_labels"] for x in picked) >= a.budget:
            break
        placed = False
        for q in picked:
            if (r["start_frame"] - q["end_frame"] <= merge_frames
                    and r["start_frame"] >= q["start_frame"]):
                q["end_frame"] = max(q["end_frame"], r["end_frame"])
                q["median_conf"] = min(q["median_conf"], r["median_conf"])
                placed = True
                break
            if (q["start_frame"] - r["end_frame"] <= merge_frames
                    and r["end_frame"] <= q["end_frame"]):
                q["start_frame"] = min(q["start_frame"], r["start_frame"])
                q["median_conf"] = min(q["median_conf"], r["median_conf"])
                placed = True
                break
        if not placed:
            picked.append(dict(r))
        for q in picked:
            q["n_labels"] = int((q["end_frame"] - q["start_frame"]) // SAMPLE_EVERY) + 1

    picked.sort(key=lambda r: -r["n_labels"])
    total = sum(r["n_labels"] for r in picked)

    print(f"{a.clip.name}: {len(ranges)} unreliable stretches outside the labelled region")
    if labelled:
        print(f"  already labelled: frames {min(labelled)}-{max(labelled)} "
              f"({len(labelled)} labels)")
    print()
    print(f"  {'frames':>16}{'start':>9}{'end':>9}{'labels':>8}{'conf':>7}")
    for r in picked:
        span = f"{r['start_frame']}-{r['end_frame']}"
        print(f"  {span:>16}"
              f"{r['start_frame'] / fps:>8.1f}s{r['end_frame'] / fps:>8.1f}s"
              f"{r['n_labels']:>8}{r['median_conf']:>7.2f}")
    print()
    print(f"  {total} labels in {len(picked)} run(s)  (budget {a.budget})")
    print("  expect to press the not-visible key often -- that is the point")
    if a.video:
        out = (a.labels or a.clip) / "ball_labels.json"
        print()
        for r in picked:
            print(f'python tools/label_ball.py --video "{a.video}" --out {out} '
                  f'--start-frame {r["start_frame"]} --end-frame {r["end_frame"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

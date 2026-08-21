"""Propose WHICH frames to label next, ranked by how much the detector is struggling there.

Labelling is the most expensive thing in this project — `indoor_C1_3min`'s 1,555 labels took
a session running 16:43 to 23:48 — so where those hours go matters more than how many there
are. Labelling the next unlabelled stretch in sequence spends them uniformly, and most of a
rally is easy for the detector already.

Two measurements say where the effort should go instead:

* Coverage is worst exactly where the model is weakest. The outdoor clips are 13-24% labelled
  against 43% for indoor B1/C1, and outdoor is the harder venue — the ball measures 10.5 px
  there against 16.5 px indoors, and tracking losses skew harder to small balls (52% of losses
  fall in the smallest third of sizes, against 43% indoors).
* The detector announces its own failures. Confidence collapses from ~0.78 to ~0.40 in the
  frames immediately before it loses the ball, so the frames worth labelling are already
  identifiable without anyone watching the video.

So this ranks unlabelled stretches by trouble: gaps where the track dropped out, and runs of
low-confidence detections. It emits contiguous RANGES rather than scattered frames, because
`label_ball.py` steps through frames in order and jumping around costs far more per label
than the label itself.

A proposal is not ground truth. A gap may be the ball genuinely out of shot, and those frames
are still worth labelling — "not visible" is a valid label and teaches the detector not to
hallucinate.

Usage:
    python -m tools.propose_labels data/pb_5_minute_outdoor-7 --labels data/pb_5min
    python -m tools.propose_labels data/pb_5_minute_outdoor-7 --labels data/pb_5min --budget 600
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOW_CONF = 0.50         # detections below this are the model saying it is unsure
GAP_MIN_FRAMES = 4      # a dropout shorter than this is normal occlusion, not a failure
MERGE_GAP_S = 1.0       # stretches closer than this are merged into one range
MIN_RANGE_S = 0.5       # ignore ranges too short to be worth navigating to
MAX_DROPOUT = 0.85      # above this the ball is probably genuinely absent, not missed
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
            in_rally = (not windows) or any(
                lo - 2.0 <= a / fps <= hi + 2.0 or lo - 2.0 <= b / fps <= hi + 2.0
                for lo, hi in windows)
            if in_rally and (b - a) / fps >= MIN_RANGE_S:
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
    ranges.sort(key=lambda r: (r["likely_absent"], r["median_conf"], -r["dropout_frac"]))
    return ranges


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path, help="folder with ball.parquet (an analysed session)")
    ap.add_argument("--labels", type=Path, default=None,
                    help="folder holding ball_labels.json (defaults to the clip)")
    ap.add_argument("--budget", type=int, default=500,
                    help="stop once this many labels have been proposed")
    a = ap.parse_args(argv)

    court = json.loads((a.clip / "court.json").read_text(encoding="utf-8"))
    fps = float(court["video"]["fps"])
    labelled = load_labelled(a.labels or a.clip)
    windows = rally_windows(a.clip)
    ranges = propose(a.clip, labelled, fps, windows)

    print(f"{a.clip.name}: {len(ranges)} trouble stretches outside the labelled region")
    if labelled:
        print(f"  already labelled: frames {min(labelled)}-{max(labelled)} "
              f"({len(labelled)} labels)")
    print()
    if windows:
        print(f"  restricted to the {len(windows)} detected rally windows")
    print(f"  {'start':>9}{'end':>9}{'secs':>7}{'labels':>8}{'dropout':>9}{'conf':>7}  note")
    used = 0
    for r in ranges:
        if used >= a.budget:
            break
        used += r["n_labels"]
        print(f"  {r['start_s']:>8.1f}s{r['end_s']:>8.1f}s{r['seconds']:>7.1f}"
              f"{r['n_labels']:>8}{r['dropout_frac']:>9.0%}{r['median_conf']:>7.2f}"
              f"  {'ball probably absent' if r['likely_absent'] else 'model unsure'}")
    print()
    print(f"  {used} labels across {min(len(ranges), sum(1 for _ in ranges))} stretches "
          f"(budget {a.budget})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

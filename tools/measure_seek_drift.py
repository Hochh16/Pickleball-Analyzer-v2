"""Measure the old labeling tool's seek drift at EVERY labeled frame.

Background: `label_ball.py` used to display a frame via
`cap.set(CAP_PROP_POS_FRAMES, idx); cap.read()`, which on long-GOP H.264 returns a
frame near idx rather than idx itself. Clicks were therefore filed against the wrong
frame (fixed in a98158b).

The first recovery pass (f99ad9e) sampled the drift every 400 frames and interpolated.
That is not good enough: indoor_B1's measured curve is non-monotonic (+5 at f1400, +4
at f1800, +3 at f2200, +5 at f2600), so a line drawn between sample points is wrong in
between, and a one-frame error still misplaces a fast ball by tens of pixels.

Seeking is deterministic, so there is no need to interpolate at all: reproduce the old
tool's exact call for each labeled frame and identify what came back. Identification is
by direct image match against frames_720/, which is trustworthy because it was written
by a single sequential decode.

Output: data/<clip>/_drift_dense.json
    {"drift": {"<frame>": k}, "ambiguous": [...], "stats": {...}}

Usage:
    python -m tools.measure_seek_drift data/indoor_B1_3min
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SEARCH_LO, SEARCH_HI = -12, 20   # candidate offsets to test around the requested frame
SIG_W, SIG_H = 320, 180          # match resolution: coarse enough to be cheap, fine
                                 # enough that player/paddle motion separates neighbours
MARGIN = 1.15                    # best must beat runner-up by this factor to be trusted


def signature(img) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)


class SigCache:
    """Lazily loaded signatures for frames_720/, pruned as the window slides."""

    def __init__(self, folder: Path):
        self.dir = folder / "frames_720"
        self.sig: dict[int, np.ndarray] = {}

    def get(self, idx: int):
        if idx in self.sig:
            return self.sig[idx]
        p = self.dir / f"{idx}.jpg"
        if not p.exists():
            return None
        img = cv2.imread(str(p))
        if img is None:
            return None
        s = signature(img)
        self.sig[idx] = s
        return s

    def prune(self, keep_from: int) -> None:
        for k in [k for k in self.sig if k < keep_from]:
            del self.sig[k]


def measure(folder: Path, out: Path) -> int:
    doc = json.loads((folder / "ball_labels.json").read_text(encoding="utf-8"))
    # measure against the frames the operator ORIGINALLY filed under
    wanted = sorted({int(l.get("remapped_from", l["frame_idx"])) for l in doc["labels"]})
    if not wanted:
        print(f"{folder.name}: no labels")
        return 1

    cache = SigCache(folder)
    cap = cv2.VideoCapture(str(folder / "video.mp4"))
    if not cap.isOpened():
        print(f"{folder.name}: cannot open video")
        return 1

    drift, ambiguous, misses = {}, [], 0
    for n, idx in enumerate(wanted):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)     # exactly what the old tool did
        ok, frame = cap.read()
        if not ok:
            misses += 1
            continue
        sig = signature(frame)
        cache.prune(idx + SEARCH_LO - 4)

        scores = []
        for k in range(SEARCH_LO, SEARCH_HI + 1):
            c = cache.get(idx + k)
            if c is not None:
                scores.append((float(np.abs(c - sig).mean()), k))
        if not scores:
            misses += 1
            continue
        scores.sort()
        best_err, best_k = scores[0]
        runner = scores[1][0] if len(scores) > 1 else best_err * 99
        # a flat scene (nobody moving) genuinely cannot be resolved; record it so the
        # caller can fall back rather than trust a coin flip
        if best_err > 0.5 and runner < best_err * MARGIN:
            ambiguous.append(idx)
        drift[idx] = best_k

        if n % 200 == 0:
            print(f"  {folder.name}: {n}/{len(wanted)}  f{idx} -> {best_k:+d}")
    cap.release()

    vals = np.array(list(drift.values()))
    stats = {
        "n_measured": len(drift),
        "n_ambiguous": len(ambiguous),
        "n_unreadable": misses,
        "min": int(vals.min()), "max": int(vals.max()),
        "nonzero_frac": float((vals != 0).mean()),
    }
    out.write_text(json.dumps({"drift": {str(k): int(v) for k, v in drift.items()},
                               "ambiguous": ambiguous, "stats": stats}, indent=1),
                   encoding="utf-8")
    print(f"{folder.name}: {stats}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    return measure(a.folder, a.out or a.folder / "_drift_dense.json")


if __name__ == "__main__":
    raise SystemExit(main())

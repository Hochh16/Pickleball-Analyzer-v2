"""Eyeball check for the label re-indexing (see commit f99ad9e).

The operator's clicks were right; `label_ball.py` attached them to the wrong frame
because `cap.set(CAP_PROP_POS_FRAMES)` drifts on long-GOP H.264. Labels were moved to
the frame the tool actually displayed. Two automatic metrics agree the remap is
correct (ball-colour present, model peak error 15.1px -> 1.8px), but neither is proof
a human would accept, so this renders the comparison for the operator:

    LEFT  = the crop at the frame the label was FILED under (pre-remap)
    RIGHT = the crop at the frame the label now points to  (post-remap)

Same pixel position in both, marked with a crosshair. If the remap is right, the ball
sits under the crosshair on the RIGHT and is displaced or absent on the LEFT.

Crops come from frames_720/ (built by sequential decode, so frame-exact) and are
upscaled, because the ball is only a few pixels wide at 720p.

Usage:
    python -m tools.verify_remap data/indoor_B1_3min
    python -m tools.verify_remap data/indoor_B1_3min --n 24 --out sheet.png
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

CROP = 56          # half-width of the crop in 720p pixels
ZOOM = 4           # upscale so a ~5px ball is visible
COLS = 4           # pairs per row


def load_labels(folder: Path) -> tuple[dict, list[dict]]:
    doc = json.loads((folder / "ball_labels.json").read_text(encoding="utf-8"))
    labels = [l for l in doc["labels"]
              if l.get("ball_visible") and l.get("pixel_x") is not None]
    return doc, labels


def read_720(folder: Path, idx: int):
    p = folder / "frames_720" / f"{idx}.jpg"
    return cv2.imread(str(p)) if p.exists() else None


def crop_at(img, cx: float, cy: float):
    """Crop a CROP-radius box around (cx, cy), zero-padded at the frame edge."""
    h, w = img.shape[:2]
    x0, y0 = int(round(cx)) - CROP, int(round(cy)) - CROP
    out = np.zeros((CROP * 2, CROP * 2, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + CROP * 2), min(h, y0 + CROP * 2)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    out = cv2.resize(out, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    c = CROP * ZOOM
    # crosshair with a gap, so the ball itself is never covered
    for dx in (-1, 0, 1):
        cv2.line(out, (c + dx, c - 26), (c + dx, c - 9), (0, 255, 255), 1)
        cv2.line(out, (c + dx, c + 9), (c + dx, c + 26), (0, 255, 255), 1)
        cv2.line(out, (c - 26, c + dx), (c - 9, c + dx), (0, 255, 255), 1)
        cv2.line(out, (c + 9, c + dx), (c + 26, c + dx), (0, 255, 255), 1)
    return out


def label_bar(w: int, text: str, colour) -> np.ndarray:
    bar = np.zeros((26, w, 3), np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return bar


def build(folder: Path, n: int, seed: int, out: Path) -> int:
    doc, labels = load_labels(folder)
    moved = [l for l in labels if l.get("drift")]
    if not moved:
        print(f"{folder.name}: no label was moved; nothing to verify.")
        return 1

    # stratify across the distinct drift values so every regime of the curve is shown,
    # not just the dense middle of the clip
    by_drift: dict[int, list] = {}
    for l in moved:
        by_drift.setdefault(int(l["drift"]), []).append(l)
    rng = random.Random(seed)
    picks: list[dict] = []
    while len(picks) < n and any(by_drift.values()):
        for d in sorted(by_drift):
            pool = by_drift[d]
            if pool and len(picks) < n:
                picks.append(pool.pop(rng.randrange(len(pool))))
    picks.sort(key=lambda l: l["frame_idx"])

    sx = doc["video_width"] / 1280.0   # labels are in source (4K) pixels
    sy = doc["video_height"] / 720.0

    tiles, kept = [], 0
    for l in picks:
        new, old = int(l["frame_idx"]), int(l["remapped_from"])
        a, b = read_720(folder, old), read_720(folder, new)
        if a is None or b is None:
            continue
        cx, cy = l["pixel_x"] / sx, l["pixel_y"] / sy
        pair = np.hstack([crop_at(a, cx, cy),
                          np.full((CROP * 2 * ZOOM, 3, 3), 60, np.uint8),
                          crop_at(b, cx, cy)])
        head = np.hstack([
            label_bar(CROP * 2 * ZOOM, f"was f{old}", (150, 150, 255)),
            np.full((26, 3, 3), 60, np.uint8),
            label_bar(CROP * 2 * ZOOM, f"now f{new}  ({l['drift']:+d})", (150, 255, 150)),
        ])
        tiles.append(np.vstack([head, pair]))
        kept += 1

    if not tiles:
        print(f"{folder.name}: frames_720 cache is missing the sampled frames.")
        return 1

    rows = []
    for i in range(0, len(tiles), COLS):
        row = tiles[i:i + COLS]
        while len(row) < COLS:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    title = np.zeros((34, sheet.shape[1], 3), np.uint8)
    cv2.putText(title, f"{folder.name}: LEFT = frame the label was filed under, "
                       f"RIGHT = frame it now points to. Ball should be under the "
                       f"RIGHT crosshair.",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([title, sheet]))
    drifts = sorted({int(l["drift"]) for l in picks})
    print(f"{folder.name}: {kept} pairs, drift values shown {drifts} -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out = a.out or a.folder / "_remap_check.png"
    return build(a.folder, a.n, a.seed, out)


if __name__ == "__main__":
    raise SystemExit(main())

"""Re-index ball labels onto the frame the labeling tool actually displayed.

Reads the untouched originals (ball_labels.PRE_REMAP.json) plus the per-label drift
measured by tools.measure_seek_drift, and rewrites ball_labels.json. Always working
from the PRE_REMAP copy means this is idempotent and re-runnable as the drift
measurement improves — it never remaps an already-remapped file.

Ambiguous frames (a still scene, where consecutive frames cannot be told apart by
image match) take the drift of the nearest confidently measured label. That is safe
precisely because the scene is still: if neighbouring frames are indistinguishable,
the ball is not moving either, so an off-by-one costs nothing.

Usage:
    python -m tools.remap_labels data/indoor_B1_3min --dry-run
    python -m tools.remap_labels data/indoor_B1_3min --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NOTE = ("frame indices corrected: label_ball used cap.set(CAP_PROP_POS_FRAMES), which "
        "drifts on long-GOP H.264, so the tool displayed frame N+k while recording the "
        "click as frame N. Drift re-measured PER LABEL (not interpolated) by "
        "tools.measure_seek_drift and each label moved to the frame actually shown.")


def resolve_drift(orig: list[int], dense: dict) -> dict[int, int]:
    """frame -> drift, with ambiguous/unmeasured frames filled from nearest neighbour."""
    raw = {int(k): int(v) for k, v in dense["drift"].items()}
    amb = set(dense.get("ambiguous", []))
    good = sorted(f for f in raw if f not in amb)
    if not good:
        raise SystemExit("no confidently measured frame; cannot remap")
    ga = np.array(good)
    out = {}
    for f in orig:
        if f in raw and f not in amb:
            out[f] = raw[f]
        else:
            out[f] = raw[int(ga[np.abs(ga - f).argmin()])]
    return out


def run(folder: Path, apply: bool) -> int:
    src = folder / "ball_labels.PRE_REMAP.json"
    if not src.exists():
        print(f"{folder.name}: no ball_labels.PRE_REMAP.json (nothing to remap from)")
        return 1
    dense_p = folder / "_drift_dense.json"
    if not dense_p.exists():
        print(f"{folder.name}: run tools.measure_seek_drift first")
        return 1

    doc = json.loads(src.read_text(encoding="utf-8"))
    dense = json.loads(dense_p.read_text(encoding="utf-8"))
    labels = doc["labels"]
    origs = [int(l["frame_idx"]) for l in labels]
    drift = resolve_drift(origs, dense)

    seen, kept, collided = set(), [], 0
    for l in labels:
        o = int(l["frame_idx"])
        k = drift[o]
        n = o + k
        if n in seen:
            collided += 1
            continue
        seen.add(n)
        m = dict(l)
        m["frame_idx"] = n
        m["remapped_from"] = o
        m["drift"] = k
        kept.append(m)
    kept.sort(key=lambda l: l["frame_idx"])

    vals = np.array([l["drift"] for l in kept])
    hist = {int(d): int((vals == d).sum()) for d in sorted(set(vals.tolist()))}
    print(f"{folder.name}: {len(labels)} labels -> {len(kept)} kept, {collided} collided, "
          f"{int((vals != 0).sum())} shifted")
    print(f"  drift histogram: {hist}")
    prev = json.loads((folder / "ball_labels.json").read_text(encoding="utf-8"))
    pmap = {int(l.get("remapped_from", l["frame_idx"])): int(l["frame_idx"])
            for l in prev["labels"]}
    changed = sum(1 for l in kept if pmap.get(l["remapped_from"]) != l["frame_idx"])
    print(f"  differs from the current ball_labels.json at {changed} labels")

    if not apply:
        print("  (dry run; pass --apply to write)")
        return 0

    doc["labels"] = kept
    doc["remap_note"] = NOTE
    (folder / "ball_labels.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"  wrote {folder / 'ball_labels.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return run(a.folder, a.apply)


if __name__ == "__main__":
    raise SystemExit(main())

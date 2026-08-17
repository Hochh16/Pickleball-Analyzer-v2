"""Score classified shot TYPES against the operator's labels.

The operator labelled shots in data/pb_5_minute_outdoor-2/_labeling/*.csv with a
`true_type` (drive / drop / dink / lob / serve / return / reset, plus non-shots marked
"between points"). This scores our `shot_type` against those, so a change to Stage 6 can
be judged on the operator's counts rather than on whether the code looks better.

Labels are matched on FRAME, not shot_id: shot ids are renumbered whenever detection
changes, so an id-keyed comparison silently compares different shots (this bit us before —
it produced a bogus "0 of 9 junk removed"). A tolerance is allowed because changing the
ball model moves a detected contact by a few frames.

Non-shot labels ("between points", "not ours", ...) are reported separately: they are the
between-point problem, a different accepted limitation, and mixing them into a type score
would hide movement in either.

Usage:
    python -m tools.score_shot_types data/pb_5_minute_outdoor-2
    python -m tools.score_shot_types data/_tmp --labels data/pb_5_minute_outdoor-2
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

TOL_FRAMES = 20
REAL_TYPES = {"drive", "drop", "dink", "lob", "serve", "return", "reset"}


def parse_clock(s: str) -> float | None:
    """'01:50.2' or '00:06.2' -> seconds."""
    s = (s or "").strip()
    if ":" not in s:
        return None
    mm, _, ss = s.partition(":")
    try:
        return int(mm) * 60 + float(ss)
    except ValueError:
        return None


def load_labels(label_dir: Path, fps: float) -> list[dict]:
    """Operator labels keyed to a FRAME.

    Only one of the two label files carries a `frame` column; the other has `time` only.
    Both are keyed on the clip's timeline rather than shot_id, which is renumbered every
    time detection changes.
    """
    out = []
    for name in ("labels.csv", "labels_block1.csv"):
        p = label_dir / "_labeling" / name
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            t = (r.get("true_type") or "").strip().lower()
            if not t:
                continue
            fr = (r.get("frame") or "").strip()
            if fr.isdigit():
                frame = int(fr)
            else:
                sec = parse_clock(r.get("time", ""))
                if sec is None:
                    continue
                frame = int(round(sec * fps))
            out.append({"frame": frame, "true_type": t, "src": name,
                        "role": (r.get("hitter_role") or "").strip()})
    return out


def score(clip: Path, labels: list[dict]) -> dict:
    shots = json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"]
    by_frame = sorted((int(s["frame"]), s) for s in shots)

    def nearest(fr: int):
        best, bd = None, TOL_FRAMES + 1
        for f, s in by_frame:
            d = abs(f - fr)
            if d < bd:
                best, bd = s, d
        return best

    real = [l for l in labels if l["true_type"] in REAL_TYPES]
    nonshot = [l for l in labels if l["true_type"] not in REAL_TYPES]
    hit = miss = unmatched = 0
    confusion: Counter = Counter()
    for l in real:
        s = nearest(l["frame"])
        if s is None:
            unmatched += 1
            continue
        got = (s.get("shot_type") or "?").strip().lower()
        if got == l["true_type"]:
            hit += 1
        else:
            miss += 1
            confusion[f'{l["true_type"]} -> {got}'] += 1
    # a non-shot label that we still emit as a shot is a false positive
    fp = sum(1 for l in nonshot if nearest(l["frame"]) is not None)
    return {"n_real": len(real), "hit": hit, "miss": miss, "unmatched": unmatched,
            "acc": hit / len(real) if real else 0.0,
            "n_nonshot": len(nonshot), "nonshot_kept": fp, "confusion": confusion}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--labels", type=Path, default=None,
                    help="folder holding _labeling/ (defaults to the clip)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    fps = float(json.loads((a.clip / "classified.json").read_text(encoding="utf-8"))
                .get("fps") or 60.0)
    labels = load_labels(a.labels or a.clip, fps)
    if not labels:
        print("no labels found")
        return 1
    r = score(a.clip, labels)
    print(f"{a.clip.name}: shot type {r['hit']}/{r['n_real']} = {r['acc']:.0%} correct "
          f"({r['miss']} wrong, {r['unmatched']} not detected within {TOL_FRAMES}f)")
    print(f"  non-shot labels still emitted as shots: {r['nonshot_kept']}/{r['n_nonshot']}"
          f"  (between-point problem, tracked separately)")
    if a.verbose and r["confusion"]:
        for k, n in r["confusion"].most_common():
            print(f"    {k}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

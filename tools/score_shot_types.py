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

# 1 second, not the 0.33s used at first. The labels carry a hand-typed CLOCK time (one
# file has no frame column at all), so sub-second matching measures the operator's
# stopwatch rather than our detection: at 0.33s only 22/32 labels matched, at 1s it is
# 26/32. Tightening this manufactured a "41% of shots are missed" result that was not real.
TOL_FRAMES = 60
REAL_TYPES = {"drive", "drop", "dink", "lob", "serve", "return", "reset"}
# The operator's full type-label set. It lives in one analysed folder but describes the
# SOURCE VIDEO, so any clip of the same video can be scored against it -- and must be, or a
# thin clip-local file silently takes its place. Keyed to that video's timeline, so it is
# only ever used when the source videos match.
SHARED_LABELS = Path("data/pb_5_minute_outdoor-2")


def source_video(clip: Path) -> str | None:
    """Basename of the video a clip was analysed from, for matching label sets."""
    for name in ("ball.meta.json", "session.json"):
        p = clip / name
        if not p.exists():
            continue
        try:
            v = json.loads(p.read_text(encoding="utf-8")).get("video_path")
        except (OSError, json.JSONDecodeError):
            continue
        if v and Path(str(v)).name not in ("video.mp4", ""):
            return Path(str(v)).name
    return None


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


def load_from_truth_store(clip: Path) -> list[dict]:
    """Type labels from the accumulating per-video truth store, when it has any.

    One home for the operator's input. The scattered `_labeling/labels*.csv` files are what
    the store was built FROM -- reading both would double-count, and a clip-local file
    shadowing a fuller set is the exact bug that made shot typing read 58% when it was 31%.
    """
    try:
        from tools.truth_store import known_shots
    except ImportError:
        return []
    out = []
    for s in known_shots(clip):
        ty = (s.get("type") or "").strip().lower()
        if ty:
            out.append({"frame": None, "t_sec": float(s["t_sec"]), "true_type": ty,
                        "src": "truth_store", "role": s.get("hitter") or ""})
    return out


def load_labels(label_dir: Path, fps: float) -> list[dict]:
    """Operator labels keyed to a FRAME.

    Reads EVERY labels*.csv in the folder. It used to name two files explicitly, which
    quietly mattered: a clip carrying its own thin `_labeling/labels.csv` shadowed the fuller
    set next door, and the default label dir is the clip itself. On the acceptance clip that
    meant scoring 12 labels — 4 drives, 6 serves, 1 return and **no drops at all** — instead
    of the 32 available, which include 3 drops, 3 dinks and 2 lobs. A drop could only ever be
    counted as an error, never as a success, and a change to drop detection was rejected on
    that basis. Same shape as the ground-ball filter recorded as "solved" against a
    measurement that had stopped applying: the scorer has to see the thing it claims to score.

    Only one of the label files carries a `frame` column; the others have `time` only. All
    are keyed on the clip's timeline rather than shot_id, which is renumbered every time
    detection changes. Duplicates (the same frame and type in two files) are dropped.
    """
    out = []
    seen: set[tuple[int, str]] = set()
    for p in sorted((label_dir / "_labeling").glob("labels*.csv")):
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
            key = (frame, t)
            if key in seen:
                continue          # the same shot labelled in two files
            seen.add(key)
            out.append({"frame": frame, "true_type": t, "src": p.name,
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
                    help="folder holding _labeling/ (defaults to the clip, then to the "
                         "shared label set the operator actually built)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    fps = float(json.loads((a.clip / "classified.json").read_text(encoding="utf-8"))
                .get("fps") or 60.0)
    store = load_from_truth_store(a.clip) if a.labels is None else []
    if store:
        for l in store:
            l["frame"] = int(round(l["t_sec"] * fps))
        labels = store
        print(f"  scoring against the truth store: {len(labels)} typed shots")
    else:
        labels = load_labels(a.labels or a.clip, fps)
    # Fall back to the shared set, and prefer it when the clip's own is a thin subset --
    # a partial label file is worse than none, because it looks like a score.
    if a.labels is None and source_video(a.clip) == source_video(SHARED_LABELS):
        shared = load_labels(SHARED_LABELS, fps)
        if len(shared) > len(labels):
            if labels:
                print(f"  note: {a.clip.name}/_labeling has only {len(labels)} labels; "
                      f"using the {len(shared)} in {SHARED_LABELS.name} instead "
                      f"(pass --labels {a.clip} to force the clip's own)")
            labels = shared
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

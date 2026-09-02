"""What is in the shots we emit: real play, junk the operator already named, or neither.

    python -m tools.shot_precision_score [clip ...]

Type accuracy answers "of the shots we call a drive, how many are drives". It says nothing
about shots that should not be there at all, and those cost more: a false positive inflates
every count in the report -- shots, dinks, third shots, the rally it lands in -- and unlike
a mistyped shot it cannot be corrected downstream, only removed.

The operator has adjudicated these. Their review marks each of our detections as a real
shot or as not-a-shot, and the truth store keeps the second kind in `false_positives`. So
every detection can be placed in one of three buckets, and only the third is unknown:

    real        matched to a shot the operator labelled
    known junk  matched to something they explicitly called not-a-shot
    unexplained matched to neither -- either genuinely new, or the detection moved since
                the review and no longer lines up with what they looked at

Matching is one-to-one and side-constrained, for the reason that keeps recurring here:
consecutive contacts are often under a second apart, so nearest-within-a-window silently
pairs a detection with its neighbour and the answer moves with the window.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.truth_store import known

MATCH_S = 0.6


def clock(t: float) -> str:
    return f"{int(t)//60}:{t % 60:05.2f}"


def _pair(items, shots, fps, window=MATCH_S, sided=True):
    """One-to-one, shortest pair first. items are dicts with t_sec (and maybe side)."""
    cand = []
    for i, it in enumerate(items):
        t = float(it["t_sec"])
        side = it.get("side") if sided else None
        for j, s in enumerate(shots):
            d = abs(int(s["frame"]) / fps - t)
            if d > window:
                continue
            if side and s.get("hitter_side") and s["hitter_side"] != side:
                continue
            cand.append((d, i, j))
    cand.sort()
    taken_i, taken_j, pairs = set(), set(), {}
    for d, i, j in cand:
        if i in taken_i or j in taken_j:
            continue
        taken_i.add(i)
        taken_j.add(j)
        pairs[j] = i
    return pairs


def score_clip(clip: Path) -> dict:
    cl = json.loads((clip / "classified.json").read_text(encoding="utf-8"))
    fps = cl.get("fps") or 60.0
    shots = sorted(cl["shots"], key=lambda s: int(s["frame"]))
    st = known(clip)
    real = [s for s in st["shots"] if not s.get("not_a_shot")]
    junk = [f for f in st.get("false_positives", [])]

    real_of = _pair(real, shots, fps)
    junk_of = _pair(junk, shots, fps, sided=False)

    rows = []
    for j, s in enumerate(shots):
        if j in real_of:
            kind = "real"
        elif j in junk_of:
            kind = "known junk"
        else:
            kind = "unexplained"
        rows.append({"clip": clip.name, "j": j, "kind": kind,
                     "t": int(s["frame"]) / fps,
                     "type": s.get("shot_type"),
                     "between": bool(s.get("is_between_point")),
                     "note": (junk[junk_of[j]].get("notes") or "") if j in junk_of else ""})
    return {"clip": clip.name, "rows": rows,
            "n_real_labelled": len(real), "n_junk_labelled": len(junk),
            "n_real_found": len(real_of)}


def main(clips=None) -> int:
    from tools.serve_score import CLIPS as DEFAULT
    allrows, tot = [], Counter()
    for name in (clips or DEFAULT):
        clip = Path("data") / name
        if not (clip / "classified.json").exists():
            print(f"  {name}: not analysed, skipping")
            continue
        r = score_clip(clip)
        allrows += r["rows"]
        c = Counter(x["kind"] for x in r["rows"])
        print(f"\n=== {r['clip']}: {len(r['rows'])} shots emitted")
        for k in ("real", "known junk", "unexplained"):
            print(f"    {c.get(k, 0):>4}  {k}")
        print(f"    operator labelled {r['n_real_labelled']} real "
              f"({r['n_real_found']} of them found) and {r['n_junk_labelled']} junk")
        tot.update(c)

    n = sum(tot.values())
    if not n:
        return 1
    print(f"\n=== {n} shots emitted across the clips")
    for k in ("real", "known junk", "unexplained"):
        print(f"    {tot.get(k, 0):>4}  {k}  ({tot.get(k, 0)/n:.0%})")
    print(f"\n  precision against what the operator adjudicated: "
          f"{tot['real']}/{tot['real'] + tot['known junk']} "
          f"({tot['real']/max(1, tot['real'] + tot['known junk']):.0%})")

    junk = [r for r in allrows if r["kind"] == "known junk"]
    if junk:
        print(f"\n  the junk we still emit, by the type we gave it:")
        for k, v in Counter(r["type"] for r in junk).most_common():
            print(f"    {v:>4}  {k}")
        inr = sum(1 for r in junk if not r["between"])
        print(f"\n  {inr} of {len(junk)} are NOT flagged between-point, so they reach the "
              f"rally counts")
        print(f"\n  what the operator said about them:")
        words = Counter()
        for r in junk:
            note = (r["note"] or "").lower()
            for key in ("feed", "pick", "adjacent", "between", "bounce", "serve",
                        "not hit", "partner", "net"):
                if key in note:
                    words[key] += 1
        for k, v in words.most_common(8):
            print(f"    {v:>4}  mentions '{k}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))

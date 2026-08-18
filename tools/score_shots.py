"""Score detected shots against the operator's per-shot review.

This is the acceptance test the project did not have. The ground-ball filter was recorded
in KNOWN_ISSUES as solved ("real shots <= 0.63, junk starts at 0.76 -- a genuine gap") and
marked not-to-be-re-litigated. When the ball detector was replaced with TrackNet, that gap
vanished -- measured 2026-08-18, the filter's score on 34 operator-labelled false positives
is a median 0.25 against a median 0.25 for real shots, and it now rejects 2 of 34. Nobody
noticed for two weeks because nothing re-ran the measurement. A filter tuned against one
ball track does not transfer to another, so every such filter needs a standing score.

Truth is data/<clip>/shot_review.json: the operator watched a continuous render of all 125
detected shots and reported every error, by cause.

Matching is on CLOCK TIME, never on shot_id. Shot ids are renumbered whenever detection
changes, so an id-keyed comparison silently compares different shots -- that bug already
produced a bogus "0 of 9 junk removed" result once.

TOL_S is 1.0s because the operator's times are hand-typed off a stopwatch; a tighter window
measures their reflexes rather than our detection (at 0.33s an earlier pass manufactured a
"41% of shots are missed" result that was not real).

Usage:
    python -m tools.score_shots data/pb_5_minute_outdoor-7
    python -m tools.score_shots data/_tmp --truth data/pb_5_minute_outdoor-7
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TOL_S = 1.0


def load(clip: Path, truth: Path) -> tuple[list[dict], dict]:
    shots = json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"]
    review = json.loads((truth / "shot_review.json").read_text(encoding="utf-8"))
    return shots, review


def claim(times: list[float], shots: list[dict], tol: float) -> tuple[set[int], list[float]]:
    """Greedy nearest match, each detected shot claimable once.

    Returns (indices of shots claimed, truth times nothing matched).
    """
    pairs = sorted((abs(s["t_sec"] - t), i, j)
                   for j, t in enumerate(times)
                   for i, s in enumerate(shots) if abs(s["t_sec"] - t) <= tol)
    used_s: set[int] = set()
    used_t: set[int] = set()
    for _, i, j in pairs:
        if i in used_s or j in used_t:
            continue
        used_s.add(i)
        used_t.add(j)
    return used_s, [t for j, t in enumerate(times) if j not in used_t]


def score(shots: list[dict], review: dict, tol: float = TOL_S) -> dict:
    fps_ = review.get("false_positives", [])
    fp_idx, fp_unmatched = claim([d["t_sec"] for d in fps_], shots, tol)
    # which causes are still being emitted as shots
    still: Counter = Counter()
    by_time = {round(d["t_sec"], 2): d["cause"] for d in fps_}
    for i in fp_idx:
        near = min(by_time, key=lambda t: abs(t - shots[i]["t_sec"]))
        still[by_time[near]] += 1

    missed = review.get("missed", [])
    found_idx, still_missed = claim([d["t_sec"] for d in missed], shots, tol)

    wp = review.get("wrong_player", [])
    wp_idx, _ = claim([d["t_sec"] for d in wp], shots, tol)
    # Shots carry `is_user` (a bool), not a role string. Every operator note in this class
    # reads "was a shot by partner, not the user", i.e. said=user / actually=partner, so
    # the error is still-true `is_user`.
    wrong = 0
    for i in wp_idx:
        near = min(wp, key=lambda d: abs(d["t_sec"] - shots[i]["t_sec"]))
        if bool(shots[i].get("is_user")) is (near["said"] == "user"):
            wrong += 1

    real = len(shots) - len(fp_idx)
    return {
        "n_detected": len(shots),
        "false_positives": len(fp_idx),
        "fp_by_cause": still,
        "fp_gone": len(fps_) - len(fp_idx),
        "n_fp_labelled": len(fps_),
        "missed_recovered": len(found_idx),
        "still_missed": len(still_missed),
        "n_missed_labelled": len(missed),
        "wrong_player": wrong,
        "n_wp_labelled": len(wp),
        "real_kept": real,
        "precision": real / len(shots) if shots else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--truth", type=Path, default=None,
                    help="folder holding shot_review.json (defaults to the clip)")
    ap.add_argument("--tol", type=float, default=TOL_S)
    a = ap.parse_args(argv)

    shots, review = load(a.clip, a.truth or a.clip)
    r = score(shots, review, a.tol)

    base = review.get("summary", {})
    print(f"{a.clip.name}: {r['n_detected']} detected "
          f"(operator baseline {base.get('detected', '?')})")
    print()
    print(f"  false positives still emitted : {r['false_positives']}/{r['n_fp_labelled']}"
          f"   ({r['fp_gone']} removed)")
    for c, n in r["fp_by_cause"].most_common():
        print(f"      {c:<32}{n}")
    print(f"  operator-missed shots found   : {r['missed_recovered']}/{r['n_missed_labelled']}")
    print(f"  attributed to the wrong player: {r['wrong_player']}/{r['n_wp_labelled']}")
    print()
    print(f"  precision (of labelled junk)  : {r['precision']:.0%}"
          f"   -- {r['real_kept']} real shots kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

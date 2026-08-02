"""Score the whole pipeline against the OPERATOR'S COUNTS — the acceptance test.

`docs/ACCURACY_LEDGER.md`: "These counts are the acceptance test. Validate every
change against THEM, never against the previous run." This tool is that check, in
one command, across every stage a change can touch — so a fix to one stage can't
quietly wreck another. Motivated by the operator's standing rule: *no change lands
without considering its impact on other stages* (there is a project history of
fixing one stage and looping back on the damage).

    python tools/score_acceptance.py --clip data/pb_5_minute_outdoor-2
    python tools/score_acceptance.py --clip <clip> --save before.json
    python tools/score_acceptance.py --clip <clip> --compare before.json

`--save` snapshots the current numbers; `--compare` diffs a later run against that
snapshot AND against truth, so "did this change help or hurt?" is answered per
metric rather than by eyeballing one headline number.
"""
import argparse
import json
from pathlib import Path

# Operator ground truth for pb_5_minute_outdoor-2 (docs/ACCURACY_LEDGER.md).
# MATCH-level counts, then the USER-level counts (the USAPA rating is per-user).
TRUTH_MATCH = {
    "shots": 98,
    # CORRECTED 2026-08-01: the operator re-checked all 14 listed serve timestamps
    # and confirmed every one is a real serve -> 14 serves = 14 rallies (was 13).
    # Two timestamps also shifted: the 1:29 serve is really at 1:33, and 4:58 at 5:01.
    "serves": 14,
    "rallies": 14,
    "dinks": 18,
    "volleys": 17,
    "bounces": 81,
}
TRUTH_USER = {
    "user_shots": 24,
    "user_drive": 12,
    "user_serve": 4,
    "user_dink": 6,
    "user_drop": 2,
    "user_volley": 6,
    "user_forehand": 15,
    "user_backhand": 9,
}
# Clips other than the acceptance clip have no operator truth; scoring still runs
# and prints the raw numbers, but every row is marked "no truth".
TRUTH_CLIP = "pb_5_minute_outdoor-2"


def load(clip: Path, name: str):
    p = clip / name
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))


def collect(clip: Path) -> dict:
    m = {}
    shots = load(clip, "shots.json")
    if shots:
        s = shots["shots"]
        m["shots"] = len(s)
        m["serves"] = sum(1 for x in s if x.get("is_serve"))
        m["user_shots"] = sum(1 for x in s if x.get("is_user"))
    bounces = load(clip, "bounces.json")
    if bounces:
        m["bounces"] = len(bounces["bounces"])
    rallies = load(clip, "rallies.json")
    if rallies:
        m["rallies"] = len(rallies["rallies"])
    cls = load(clip, "classified.json")
    if cls:
        c = cls["shots"]
        by_user = [x for x in c if x.get("is_user")]
        m["volleys"] = sum(1 for x in c if x.get("is_volley"))
        m["dinks"] = sum(1 for x in c if x.get("shot_type") == "dink")
        for t in ("drive", "serve", "dink", "drop", "volley"):
            key = f"user_{t}"
            m[key] = (sum(1 for x in by_user if x.get("is_volley")) if t == "volley"
                      else sum(1 for x in by_user if x.get("shot_type") == t))
        m["user_forehand"] = sum(1 for x in by_user if x.get("stroke_side") == "forehand")
        m["user_backhand"] = sum(1 for x in by_user if x.get("stroke_side") == "backhand")
    rating = load(clip, "rating.json")
    if rating:
        m["_rating_estimate"] = rating.get("estimate")
        m["_rating_band"] = rating.get("band")
    return m


def fmt_row(name, got, truth, base):
    """One metric line: value, error vs truth, and movement vs the snapshot."""
    g = "-" if got is None else f"{got}"
    if truth is None:
        err, verdict = "no truth", ""
    else:
        d = got - truth if got is not None else None
        err = "-" if d is None else f"{d:+d} ({abs(d) / truth:.0%})" if truth else f"{d:+d}"
        verdict = "" if d is None else ("EXACT" if d == 0 else "")
    move = ""
    if base is not None and got is not None and truth is not None:
        was, now = abs(base - truth), abs(got - truth)
        if base == got:
            move = "unchanged"
        elif now < was:
            move = f"BETTER (was {base})"
        elif now > was:
            move = f"WORSE (was {base})"
        else:
            move = f"same error (was {base})"
    elif base is not None and got is not None and base != got:
        move = f"was {base}"
    return f"  {name:16s} {g:>6s}  truth {str(truth):>5s}  {err:>14s}  {verdict}{move}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--save", help="write these numbers to a snapshot file")
    ap.add_argument("--compare", help="diff against a snapshot file")
    args = ap.parse_args()
    clip = Path(args.clip)

    got = collect(clip)
    has_truth = clip.name == TRUTH_CLIP
    base = json.load(open(args.compare, encoding="utf-8")) if args.compare else {}

    print(f"\nACCEPTANCE SCORECARD — {clip.name}")
    if not has_truth:
        print("  (no operator truth for this clip — raw numbers only)")
    if args.compare:
        print(f"  comparing against {args.compare}")

    for title, truth_map in (("MATCH", TRUTH_MATCH), ("USER", TRUTH_USER)):
        print(f"\n{title}")
        for k, t in truth_map.items():
            print(fmt_row(k, got.get(k), t if has_truth else None, base.get(k)))

    # The single best self-check in the system (ACCURACY_LEDGER): every shot is
    # either volleyed out of the air or lands exactly once.
    sh, vo, bo = got.get("shots"), got.get("volleys"), got.get("bounces")
    if None not in (sh, vo, bo):
        d = sh - (vo + bo)
        print(f"\nIDENTITY  shots == volleys + bounces:  "
              f"{sh} vs {vo}+{bo}={vo + bo}   {'HOLDS' if d == 0 else f'OFF BY {d:+d}'}")
    if got.get("_rating_estimate") is not None:
        print(f"RATING    estimate {got['_rating_estimate']} band {got.get('_rating_band')}")

    if has_truth:
        scored = [(k, got.get(k), t) for k, t in {**TRUTH_MATCH, **TRUTH_USER}.items()
                  if got.get(k) is not None]
        if scored:
            err = sum(abs(g - t) / t for _, g, t in scored) / len(scored)
            print(f"\nMEAN ABSOLUTE ERROR across {len(scored)} scored metrics: {err:.1%}")

    if args.save:
        json.dump(got, open(args.save, "w", encoding="utf-8"), indent=1)
        print(f"\nsnapshot -> {args.save}")
    print()


if __name__ == "__main__":
    main()

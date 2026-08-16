"""Count shot-to-player attribution errors using rules of the game, not labels.

Motivation: on the indoor clip the operator's own serve/return counts did not match the
report, while a render CONFIRMED the roles were the right people. Roles right, individual
shots on the wrong player. That needs a measurement before a fix, and pickleball supplies
two contradictions that need no ground truth at all:

  ALTERNATION   Within a rally the two teams must strictly alternate. There is no second
                touch in pickleball, so two consecutive shots by the same team is
                impossible — it means at least one of them is on the wrong player.

  SERVE/RETURN  The serving team cannot hit the return of its own serve. This is the
                special case of alternation the operator hit first, reported separately
                because it is the one that corrupts the serve and return counts.

Both are properties of the SEQUENCE, so they cannot be satisfied by chance in a long
rally: a random attribution violates alternation about half the time.

WHAT THE VIOLATIONS TURNED OUT TO MEAN (measured 2026-08-15)
------------------------------------------------------------
Not misattribution. MISSING SHOTS.

    clip          legal alternating pairs      same-team pairs
    outdoor-7     n=98, median gap 1.15s       n=10, median gap 4.70s   (4.1x)
    indoor        n=77, median gap 1.24s       n= 3, median gap 3.45s   (2.8x)

A normal shot-to-shot gap is ~1.2 s, so a 4.7 s gap has room for two or three shots. The
true sequence is far -> (missed near shot) -> far, and the pair only looks impossible
because the shot between them was never detected. A repair that re-attributed one of the
two shots to the other side was built and correctly did nothing: no other-side player was
in range, because no player was near the ball — the ball was not detected there at all.

So this tool measures RECALL, not attribution. Read a rising same-team rate as "shots are
being missed", and use the gap ratio to confirm.

It still cannot see the error that prompted it: a mix-up between the two players on the
SAME side preserves alternation. That is what corrupts an individual's serve/return
counts, and settling it needs the operator's eye on specific frames, not a rule.

Usage:
    python -m tools.check_attribution data/pb_5_minute_outdoor-7 data/pb_3_min_indoor_1_court_b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NEAR = {"user", "partner"}
FAR = {"opp_a", "opp_b"}


def team_of(role: str) -> str | None:
    return "near" if role in NEAR else "far" if role in FAR else None


def check(folder: Path, verbose: bool = False) -> dict:
    cl = {int(s["shot_id"]): s for s in
          json.loads((folder / "classified.json").read_text(encoding="utf-8"))["shots"]}
    rallies = json.loads((folder / "rallies.json").read_text(encoding="utf-8"))["rallies"]
    roles = json.loads((folder / "track_roles.json").read_text(encoding="utf-8"))["track_roles"]

    def role_of(shot) -> str | None:
        t = shot.get("track_id")
        return None if t is None else roles.get(str(int(t)), {}).get("role")

    pairs = same_team = unknown = 0
    serve_ret_bad = serve_ret_total = 0
    details = []

    for r in rallies:
        ids = [i for i in r.get("shot_ids", []) if i in cl]
        teams = [team_of(role_of(cl[i]) or "") for i in ids]
        for k in range(1, len(ids)):
            a, b = teams[k - 1], teams[k]
            if a is None or b is None:
                unknown += 1
                continue
            pairs += 1
            if a == b:
                same_team += 1
                if verbose and len(details) < 12:
                    details.append(
                        f"    rally {r['rally_id']:>3} t={cl[ids[k]]['t_sec']:7.1f}s  "
                        f"{role_of(cl[ids[k-1]])} -> {role_of(cl[ids[k]])} (both {a})")
        # serve/return: the receiving team must hit shot 2
        if len(ids) >= 2 and teams[0] is not None and teams[1] is not None:
            serve_ret_total += 1
            if teams[0] == teams[1]:
                serve_ret_bad += 1

    return {"clip": folder.name, "rallies": len(rallies), "pairs": pairs,
            "same_team": same_team, "unknown": unknown,
            "serve_ret_bad": serve_ret_bad, "serve_ret_total": serve_ret_total,
            "details": details}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    print("Teams must strictly alternate within a rally (no second touch in pickleball).")
    print("A same-team pair means at least one of those two shots is on the wrong player.")
    print()
    print(f"{'clip':<28}{'rallies':>8}{'pairs':>7}{'same-team':>11}{'rate':>8}"
          f"{'serve=return team':>19}")
    rc = 0
    for f in a.folders:
        if not (f / "classified.json").exists():
            print(f"{f.name:<28} not analysed")
            continue
        r = check(f, a.verbose)
        rate = r["same_team"] / r["pairs"] if r["pairs"] else 0.0
        print(f"{r['clip']:<28}{r['rallies']:>8}{r['pairs']:>7}{r['same_team']:>11}"
              f"{rate:>7.1%}{r['serve_ret_bad']:>10}/{r['serve_ret_total']:<8}")
        if r["details"]:
            for d in r["details"]:
                print(d)
        if r["same_team"]:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

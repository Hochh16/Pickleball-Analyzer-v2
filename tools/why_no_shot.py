"""For each labelled serve with no detected contact, name the filter that dropped it.

    python -m tools.why_no_shot [clip ...]

Requires shot_discards.json, which stage 5 writes beside shots.json.

Why: stats give totals -- 315 rejected by handling, 273 with no player -- and a total
cannot say what happened to the serve at 1:34.22. Working from the totals produced two
confident wrong answers (the "server has the ball" threshold, then camera placement),
each only disproved after being measured. This reads the actual fate of the candidates
around a labelled serve, so the filter is named rather than inferred.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.serve_score import CLIPS, clock
from tools.truth_store import known

SEARCH_S = 0.5      # a candidate this close to the labelled strike is that strike


def main(clips=None) -> int:
    for name in (clips or CLIPS):
        clip = Path("data") / name
        dpath = clip / "shot_discards.json"
        if not dpath.exists():
            print(f"  {name}: no shot_discards.json — re-run stage 5 to produce it")
            continue

        doc = json.loads(dpath.read_text(encoding="utf-8"))
        fps = doc.get("fps") or 60.0
        discards = doc["discards"]

        cl = json.loads((clip / "classified.json").read_text(encoding="utf-8"))
        kept = {int(s["frame"]) for s in cl["shots"]}

        serves = sorted((s for s in known(clip)["shots"]
                         if s.get("type") == "serve" and not s.get("not_a_shot")),
                        key=lambda s: float(s["t_sec"]))

        print(f"\n=== {name}   ({len(discards)} discarded candidates recorded)")
        for tr in serves:
            t, side = float(tr["t_sec"]), tr.get("side")
            f0 = int(round(t * fps))
            near_kept = [f for f in kept if abs(f - f0) <= SEARCH_S * fps]
            if near_kept:
                continue                       # a contact survived here; not our problem
            near = [d for d in discards if abs(d["frame"] - f0) <= SEARCH_S * fps]
            print(f"  {clock(t):>8}  {str(tr.get('hitter')):<9} side={str(side):<5} "
                  f"no contact kept")
            if not near:
                print("            nothing was even proposed here — the impulse detector "
                      "never fired")
                continue
            by_frame = {d["frame"]: d for d in discards}
            shown = 0
            for d in sorted(near, key=lambda d: abs(d["frame"] - f0)):
                # Follow merges to the candidate that actually won, and report ITS fate --
                # that is the real cause; "merged" only says where to look next.
                chain, cur, seen = [], d, set()
                while (cur["reason"] == "merged_into_nearby_candidate"
                       and cur.get("merged_into") is not None
                       and cur["merged_into"] not in seen):
                    seen.add(cur["frame"])
                    nxt = cur["merged_into"]
                    chain.append(nxt)
                    if nxt in kept:
                        cur = {"reason": "KEPT as a shot", "frame": nxt}
                        break
                    if nxt not in by_frame:
                        cur = {"reason": "vanished (neither kept nor discarded)",
                               "frame": nxt}
                        break
                    cur = by_frame[nxt]
                extra = {k: v for k, v in d.items()
                         if k not in ("frame", "reason", "merged_into")}
                via = f" -> f{' -> f'.join(str(c) for c in chain)}" if chain else ""
                print(f"            f{d['frame']} ({(d['frame']-f0)/fps:+.2f}s) "
                      f"{d['reason']}{via}"
                      + (f" => {cur['reason']}" if chain else "")
                      + (f"  {extra}" if extra else ""))
                shown += 1
                if shown >= 4:
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))

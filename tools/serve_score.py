"""Score serve detection against the operator's labels, and say WHY each one fails.

Run before and after any change to serve handling:

    python -m tools.serve_score

Why this exists: the serve numbers have been "improved" more than once without a standing
measure, and a fix that helps one clip while quietly costing another is indistinguishable
from a fix that works. The buckets here are the acceptance test.

MATCHING. A truth serve is matched to a detection only if the detection is struck from the
SAME SIDE. Matching on a time window alone is not stable: a serve and its return are about
1.0-1.3s apart, so a wide window pairs a serve with the other side's return and a narrow
one calls a real match a miss. Measured on the two labelled clips, the time-only count ran
7 -> 3 -> 2 -> 0 as the window widened from 0.4s to 1.5s, while the side-constrained count
settles at 3 and stays there from 0.75s out to 1.5s. The window was deciding the answer;
the side constraint is what makes it stop.

FALSE ACCEPTS MATTER AS MUCH AS MISSES. A false serve does not merely add one: acceptance
enforces one serve per point, so a false accept BLOCKS the real serve behind it. On the
outdoor clip the false accept at 3:06.43 blocked the real serve at 3:13.00, and 4:39.82
blocked 4:42.53. That is why n_false is reported next to the misses rather than buried.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.truth_store import known

# Wide enough to absorb contact-timing jitter, narrow enough that the side constraint is
# doing the work rather than the clock. The count is flat from 0.75 to 1.5s.
MATCH_WINDOW_S = 1.0

CLIPS = ["pb_5_minute_outdoor-12", "pb_3_min_indoor_1_court_c"]

MISSED = "contact never detected"
NOT_ACCEPTED = "detected, not accepted as a serve"
WRONG_PLAYER = "accepted, credited to the wrong player"
CORRECT = "correct"


def _norm(role):
    """Truth says user / partner / opponent; roles say user / partner / opp_a / opp_b."""
    if role is None:
        return None
    r = str(role).lower()
    return "opponent" if r.startswith("opp") else r


def clock(t: float) -> str:
    return f"{int(t)//60}:{t % 60:05.2f}"


def score_clip(clip: Path, window: float = MATCH_WINDOW_S) -> dict:
    """Bucket every labelled serve, and count accepted serves that match no label."""
    cl = json.loads((clip / "classified.json").read_text(encoding="utf-8"))
    fps = cl.get("fps") or 60.0
    shots = sorted(cl["shots"], key=lambda s: int(s["frame"]))

    roles = json.loads((clip / "track_roles.json").read_text(encoding="utf-8"))
    tid_role = {int(t): (i or {}).get("role")
                for t, i in (roles.get("track_roles") or {}).items()}

    truth = known(clip)
    serves = sorted((s for s in truth["shots"]
                     if s.get("type") == "serve" and not s.get("not_a_shot")),
                    key=lambda s: float(s["t_sec"]))

    rows = []
    for tr in serves:
        t, side = float(tr["t_sec"]), tr.get("side")
        want = _norm(tr.get("hitter"))
        cand = [s for s in shots
                if abs(int(s["frame"]) / fps - t) <= window
                and (side is None or s.get("hitter_side") == side)]
        if not cand:
            rows.append({"t": t, "hitter": want, "bucket": MISSED, "note": ""})
            continue
        best = min(cand, key=lambda s: abs(int(s["frame"]) / fps - t))
        got = _norm(tid_role.get(int(best.get("track_id", -1))))
        if not best.get("is_serve"):
            rows.append({"t": t, "hitter": want, "bucket": NOT_ACCEPTED,
                         "note": f"called {best.get('shot_type') or '?'}"})
        elif want and got and want != got:
            rows.append({"t": t, "hitter": want, "bucket": WRONG_PLAYER,
                         "note": f"credited {got}"})
        else:
            rows.append({"t": t, "hitter": want, "bucket": CORRECT, "note": ""})

    truth_t = [float(s["t_sec"]) for s in serves]
    accepted = [s for s in shots if s.get("is_serve")]
    false_accepts = [s for s in accepted
                     if not any(abs(int(s["frame"]) / fps - x) <= window for x in truth_t)]
    return {"clip": clip.name, "fps": fps, "rows": rows,
            "n_accepted": len(accepted), "n_false": len(false_accepts),
            "false_t": [int(s["frame"]) / fps for s in false_accepts]}


def main(clips=None) -> int:
    results = []
    for name in (clips or CLIPS):
        clip = Path("data") / name
        if not (clip / "classified.json").exists():
            print(f"  {name}: not analysed, skipping")
            continue
        results.append(score_clip(clip))

    if not results:
        print("no analysed clips with serve truth")
        return 1

    order = [CORRECT, MISSED, NOT_ACCEPTED, WRONG_PLAYER]
    totals = {b: 0 for b in order}
    user = {b: 0 for b in order}
    n_accepted = n_false = 0

    for r in results:
        print(f"\n=== {r['clip']}")
        for row in r["rows"]:
            totals[row["bucket"]] += 1
            if row["hitter"] == "user":
                user[row["bucket"]] += 1
            if row["bucket"] != CORRECT:
                note = f"  ({row['note']})" if row["note"] else ""
                print(f"    {clock(row['t']):>8}  {str(row['hitter']):<9} "
                      f"{row['bucket']}{note}")
        n_accepted += r["n_accepted"]
        n_false += r["n_false"]
        print(f"    accepted {r['n_accepted']}, of which {r['n_false']} match no label")

    n = sum(totals.values())
    print(f"\n=== {n} labelled serves across {len(results)} clip(s)")
    for b in order:
        print(f"  {totals[b]:>3} / {n}   {b}")
    print(f"\n=== the user's own serves ({sum(user.values())})")
    for b in order:
        if user[b]:
            print(f"  {user[b]:>3}   {b}")
    print(f"\n=== accepted serves: {n_accepted}, false: {n_false}"
          f" ({n_false / n_accepted:.0%})" if n_accepted else "")
    print("  A false accept also BLOCKS the real serve behind it (one serve per point),")
    print("  so this number must come down for the misses to come down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))

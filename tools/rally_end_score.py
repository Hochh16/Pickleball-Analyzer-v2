r"""How a rally ended, scored against the operator's rally truth.

The serve/return card reads "In-play rate & faults", and both halves come from the rally's
END REASON: a serve fault is a rally that ended on the serve. Before changing how faults
are decided there has to be a standing number for how often the end reason is right at all
-- otherwise a change that moves the fault count has nothing to be judged against.

WHAT THE TRUTH HAS. Court C is the only video where the operator gave an end reason for
every rally (10 of them). Court B gives rally boundaries and shot counts but no reasons.
The outdoor video gives 16 end notes in free text ("hit into net. End of rally"), which
carry a reason but no rally boundary. So: 26 ends with a reason, and among all of them
exactly ONE serve fault -- court C at 39.7s, "serve was out. Hit long." A serve fault is a
rare event, and no amount of tuning can be validated against a single instance; the honest
target is the end reason as a whole, of which the fault is one bucket.

MATCHING. Detected rallies are paired to truth rallies by TEMPORAL OVERLAP, not by end
time. Detected rallies run long -- court C's rally 3 is 39.78-51.60s against a truth
39.70-41.62s -- so a window around the end time pairs the wrong things, which is the
failure mode that has bitten every scorer in this repo. Overlap is stable under that.

    python -m tools.rally_end_score
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.truth_store import known  # noqa: E402

CLIPS = ["pb_3_min_indoor_1_court_c", "pb_3_min_indoor_1_court_b", "pb_5_minute_outdoor-12"]

# Our vocabulary -> the operator's. double-bounce and not-returned are the same event in
# his words: the ball bounced twice because nobody got to it.
OURS = {"ball-out": "out", "net-or-short": "net", "ball-not-returned": "not-returned",
        "double-bounce": "not-returned", "serve-fault": "serve-fault",
        "unknown": "unknown"}

_NET = re.compile(r"\binto the net\b|\bhit into net\b|\bwas into net\b|\bnet\b", re.I)
_OUT = re.compile(r"\bout\b|\bhit long\b|\bwent long\b", re.I)
_NOT_RET = re.compile(r"winning shot|not[- ]returned|missed by|could not get", re.I)


def note_reason(note: str) -> Optional[str]:
    """The reason a free-text end note asserts, or None if it asserts none.

    Order matters: "partner hit it out. Never made it back over the net" is an OUT that
    mentions the net, and "winning shot | net" is a net error the operator also called a
    winner. Out and net are physical claims about the ball; not-returned is a claim about
    the players, so it yields to both.
    """
    n = note or ""
    if _OUT.search(n):
        return "out"
    if _NET.search(n):
        return "net"
    if _NOT_RET.search(n):
        return "not-returned"
    return None


def truth_ends(clip: Path) -> List[dict]:
    """One row per rally the operator described: {start, end, reason, n_shots}.

    rally_truth is preferred -- it carries boundaries -- and rally_ends fills in for the
    videos where he annotated the ending shot but not the rally.
    """
    d = known(clip)
    rows = []
    for r in d.get("rally_truth") or []:
        reason = r.get("end_reason")
        # A rally that ended on its only shot ended on the SERVE. The operator writes that
        # as "out" plus a note; the distinction our pipeline draws is a separate bucket.
        if reason and r.get("n_shots") == 1:
            reason = "serve-fault"
        rows.append({"start": float(r["start_t_sec"]), "end": float(r["end_t_sec"]),
                     "reason": reason, "n_shots": r.get("n_shots")})
    if rows:
        return sorted(rows, key=lambda r: r["start"])
    for e in d.get("rally_ends") or []:
        # The review sheet's END_REASON dropdown where given; the notes regex otherwise.
        reason = e.get("reason") or note_reason(e.get("notes", ""))
        if reason is None:
            continue
        t = float(e.get("rally_over_t_sec") or e["t_sec"])
        rows.append({"start": None, "end": t, "reason": reason, "n_shots": None})
    return sorted(rows, key=lambda r: r["end"])


def pair_by_overlap(truth: List[dict], det: List[dict]) -> List[tuple]:
    """One-to-one, greediest overlap first. Falls back to nearest END for truth rows with
    no boundary (the free-text notes), which is all they can support."""
    pairs, used = [], set()
    scored = []
    for ti, t in enumerate(truth):
        for di, d in enumerate(det):
            if t["start"] is None:
                # No boundary: the end note belongs to the rally that contains it, else
                # the nearest end. Score is negative distance so bigger is better.
                inside = d["start"] <= t["end"] <= d["end"]
                s = 1000.0 if inside else -abs(d["end"] - t["end"])
                if not inside and abs(d["end"] - t["end"]) > 6.0:
                    continue
            else:
                s = min(t["end"], d["end"]) - max(t["start"], d["start"])
                if s <= 0:
                    continue
            scored.append((s, ti, di))
    for _s, ti, di in sorted(scored, key=lambda x: -x[0]):
        if ti in used or ("d", di) in used:
            continue
        used.add(ti)
        used.add(("d", di))
        pairs.append((truth[ti], det[di]))
    for ti, t in enumerate(truth):
        if ti not in used:
            pairs.append((t, None))
    for di, d in enumerate(det):
        if ("d", di) not in used:
            pairs.append((None, d))
    return pairs


def score_clip(name: str) -> dict:
    clip = Path("data") / name
    cl = json.loads((clip / "classified.json").read_text(encoding="utf-8"))
    ra = json.loads((clip / "rallies.json").read_text(encoding="utf-8"))
    fps = cl.get("fps") or 60.0
    det = [{"start": int(r["start_frame"]) / fps, "end": int(r["end_frame"]) / fps,
            "reason": OURS.get(r.get("end_reason"), r.get("end_reason")),
            "raw": r.get("end_reason"), "n_shots": r.get("n_shots"),
            "conf": r.get("end_reason_confidence")}
           for r in ra["rallies"]]
    truth = [t for t in truth_ends(clip) if t["reason"]]
    if not truth:
        # Court B has rally boundaries but no end reasons. Counting its 10 rallies as
        # "spurious" would say we invented them, when the operator simply never said.
        return {"clip": name, "ok": 0, "n": 0, "rows": [], "confusion": Counter(),
                "missed": 0, "spurious": 0, "n_truth": 0, "n_det": len(det),
                "skipped": "no end reasons in truth"}
    pairs = pair_by_overlap(truth, det)

    ok = n_scored = 0
    confusion: Counter = Counter()
    unmatched_truth = unmatched_det = 0
    rows = []
    for t, d in pairs:
        if t is None:
            unmatched_det += 1
            confusion[f"(no rally) -> {d['reason']}"] += 1
            rows.append((None, d, "SPURIOUS"))
            continue
        if d is None:
            unmatched_truth += 1
            confusion[f"{t['reason']} -> (missed)"] += 1
            rows.append((t, None, "MISSED"))
            continue
        n_scored += 1
        hit = t["reason"] == d["reason"]
        ok += hit
        if not hit:
            confusion[f"{t['reason']} -> {d['reason']}"] += 1
        rows.append((t, d, "ok" if hit else "WRONG"))
    return {"clip": name, "ok": ok, "n": n_scored, "rows": rows,
            "confusion": confusion, "missed": unmatched_truth,
            "spurious": unmatched_det, "n_truth": len(truth), "n_det": len(det)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tot_ok = tot_n = tot_missed = tot_spur = 0
    conf: Counter = Counter()
    sf_hit = sf_want = sf_got = 0
    for name in CLIPS:
        r = score_clip(name)
        if r.get("skipped"):
            print(f"\n{name}  --  skipped: {r['skipped']} ({r['n_det']} rallies detected)")
            continue
        print(f"\n{name}  --  {r['n_truth']} rallies in truth, {r['n_det']} detected")
        for t, d, verdict in r["rows"]:
            tw = f"{t['start']:6.1f}-{t['end']:6.1f}" if t and t["start"] is not None \
                else (f"     ~{t['end']:6.1f}" if t else "            ")
            dw = f"{d['start']:6.1f}-{d['end']:6.1f}" if d else "            "
            print(f"   truth {tw} {str(t['reason']) if t else '-':<13}"
                  f"| ours {dw} {str(d['reason']) if d else '-':<13} {verdict}")
        tot_ok += r["ok"]
        tot_n += r["n"]
        tot_missed += r["missed"]
        tot_spur += r["spurious"]
        conf.update(r["confusion"])
        for t, d, _v in r["rows"]:
            if t and t["reason"] == "serve-fault":
                sf_want += 1
                if d and d["reason"] == "serve-fault":
                    sf_hit += 1
            if d and d["reason"] == "serve-fault":
                sf_got += 1

    print(f"\n=== END REASON: {tot_ok}/{tot_n} correct on paired rallies "
          f"({tot_ok / tot_n:.0%})" if tot_n else "\n=== no pairs")
    print(f"    {tot_missed} truth rallies with no detected rally, "
          f"{tot_spur} detected rallies with no truth rally")
    print(f"    serve faults: {sf_hit} of {sf_want} found, {sf_got} claimed")
    print("    confusions (truth -> ours):")
    for k, n in conf.most_common(12):
        print(f"      {n:>3}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

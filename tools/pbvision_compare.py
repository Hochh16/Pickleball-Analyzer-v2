r"""Score PB Vision's output and our pipeline against the operator's review of the same video.

Recommended in docs/SESSION_HANDOFF.md (2026-10-01) as the cheapest, most decisive test left: if
a commercial product gets far more right from the SAME tripod footage, the footage is not the
limit; if it struggles too, the camera position caps everyone. Headline numbers are not compared
-- two systems can land on similar totals for different reasons -- every shot is matched to the
review one-to-one, the same way every experiment in docs/ACCURACY_LEDGER.md is scored.

Input is PB Vision's "insights" JSON (rallies -> shots with contact time, hitter, type tags,
volley flag, and where the final ball went). Our side is the clip's classified.json and
rallies.json; the truth is the store (tools/truth_store.py).

    python -m tools.pbvision_compare data/pb_5_min_indoor_1_court_a "<path to insights.json>"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rally_end_score import note_reason  # noqa: E402
from tools.truth_store import known  # noqa: E402

NET_Y_FT = 22.0
COURT_W_FT, COURT_L_FT = 20.0, 44.0
SHOT_TOL_S, LOOSE_TOL_S, SERVE_TOL_S, END_TOL_S = 0.35, 1.0, 1.0, 2.0

# our classifier's vocabulary -> the operator's; PB's tags are mapped in pb_shots()
OUR_END = {"ball-out": "out", "net-or-short": "net", "ball-not-returned": "not-returned",
           "double-bounce": "not-returned", "serve-fault": "serve-fault"}


def one_to_one(a: Sequence[float], b: Sequence[float], tol: float) -> Dict[int, int]:
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                   if abs(x - y) <= tol)
    ui, uj, out = set(), set(), {}
    for _, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        out[i] = j
    return out


def pb_end_reason(final: dict) -> Optional[str]:
    """How PB Vision says the point ended, in the operator's words, from the final ball."""
    m = final.get("resulting_ball_movement") or {}
    if m.get("crossed_net") is False:
        return "net"
    end = ((m.get("trajectory") or {}).get("end") or {}).get("location") or {}
    x, y = end.get("x"), end.get("y")
    if x is None or y is None:
        return None
    if not (0.0 <= x <= COURT_W_FT) or not (0.0 <= y <= COURT_L_FT):
        return "out"
    return "not-returned"


def pb_near_is_high_y(cv: Optional[dict]) -> Optional[bool]:
    """Which end of PB Vision's court is nearer the camera, from its own camera fit.

    PB Vision's court coordinates are not oriented like ours: on court A its camera sits at
    y = 57 ft, past ITS far end, so the camera-side half is y > 22. Read from the cv file rather
    than assumed -- scored the other way round, side came out a perfect 0 of 86.
    """
    try:
        y = float(cv["camera"]["cameraSegments"][0]["position"]["y"])
    except (TypeError, KeyError, IndexError, ValueError):
        return None
    return y > NET_Y_FT


def pb_shots(insights: dict, near_high: Optional[bool] = None) -> tuple:
    """(shots, rallies) in the same shape as ours: t, type, side, volley; rally end and reason."""
    shots, rallies = [], []
    for r in insights["rallies"]:
        rs = r["shots"]
        for k, s in enumerate(rs):
            tags = " ".join(s.get("tags") or {})
            typ = next((t for t in ("serve", "drive", "dink", "drop", "smash", "lob")
                        if f"type;{t}" in tags), None)
            if typ is None and k == 1:
                typ = "return"          # PB does not tag returns; the shot after the serve is one
            pos = (s.get("player_positions") or [])
            pid = s.get("player_id")
            py = pos[pid]["y"] if pid is not None and pid < len(pos) else None
            shots.append({"t": s["start_ms"] / 1000.0, "type": typ,
                          "side": (None if py is None or near_high is None else
                                   ("near" if (py > NET_Y_FT) == near_high else "far")),
                          "volley": s.get("is_volley")})
        if rs:
            rallies.append({"last": rs[-1]["start_ms"] / 1000.0, "reason": pb_end_reason(rs[-1])})
    return shots, rallies


def our_shots(folder: Path) -> tuple:
    cl = json.loads((folder / "classified.json").read_text(encoding="utf-8"))["shots"]
    by_id = {int(s["shot_id"]): s for s in cl}
    shots = [{"t": float(s["t_sec"]), "type": s.get("shot_type"), "side": s.get("hitter_side"),
              "volley": s.get("is_volley")} for s in cl]
    rallies = []
    for r in json.loads((folder / "rallies.json").read_text(encoding="utf-8"))["rallies"]:
        ids = [int(i) for i in r.get("shot_ids", []) if int(i) in by_id]
        if ids:
            rallies.append({"last": float(by_id[ids[-1]]["t_sec"]),
                            "reason": OUR_END.get(r.get("end_reason"))})
    return shots, rallies


def score(shots: List[dict], rallies: List[dict], truth: List[dict], ends: List[dict]) -> dict:
    tt = [float(s["t_sec"]) for s in truth]
    st = [s["t"] for s in shots]
    m = one_to_one(st, tt, SHOT_TOL_S)
    loose = one_to_one(st, tt, LOOSE_TOL_S)
    pairs = [(shots[i], truth[j]) for i, j in m.items()]
    typed = [(g, t) for g, t in pairs if t.get("type")]
    sided = [(g, t) for g, t in pairs if t.get("side")]
    voll = [(g, t) for g, t in pairs if t.get("volley") is not None and g.get("volley") is not None]
    ts = [float(s["t_sec"]) for s in truth if s.get("type") == "serve"]
    ss = [s["t"] for s in shots if s["type"] == "serve"]
    ms = one_to_one(ss, ts, SERVE_TOL_S)
    et = [float(e["t_sec"]) for e in ends]
    er = [e.get("reason") or note_reason(e.get("notes", "")) for e in ends]
    me = one_to_one([r["last"] for r in rallies], et, END_TOL_S)
    offs = [st[i] - tt[j] for i, j in loose.items()]
    return {
        "emitted": len(st), "found": len(m), "junk": len(st) - len(m),
        "found_1s": len(loose), "junk_1s": len(st) - len(loose),
        "clock_offset_s": round(median(offs), 3) if offs else None,
        "type": (sum(g["type"] == t["type"] for g, t in typed), len(typed)),
        "side": (sum(g["side"] == t["side"] for g, t in sided), len(sided)),
        "volley": (sum(bool(g["volley"]) == bool(t["volley"]) for g, t in voll), len(voll)),
        "serves": (len(ms), len(ts), len(ss) - len(ms)),
        "rallies": len(rallies),
        "ends": (len(me), len(et)),
        "end_reason": (sum(rallies[i]["reason"] == er[j] for i, j in me.items()),
                       sum(1 for i, j in me.items() if er[j])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("insights", type=Path)
    ap.add_argument("--cv", type=Path, default=None,
                    help="PB Vision's cv JSON; needed to orient near/far (side is skipped without it)")
    a = ap.parse_args(argv)
    store = known(a.clip)
    truth = sorted((s for s in store["shots"] if not s.get("not_a_shot")),
                   key=lambda s: float(s["t_sec"]))
    ends = sorted(store.get("rally_ends", []), key=lambda e: float(e["t_sec"]))
    ins = json.loads(a.insights.read_text(encoding="utf-8"))
    cv = json.loads(a.cv.read_text(encoding="utf-8")) if a.cv else None
    res = {"PB Vision": score(*pb_shots(ins, pb_near_is_high_y(cv)), truth, ends),
           "our pipeline": score(*our_shots(a.clip), truth, ends)}
    n_serve = sum(1 for s in truth if s.get("type") == "serve")
    print(f"{a.clip.name}: operator's review = {len(truth)} real shots, {n_serve} serves, "
          f"{len(ends)} rally ends\n")
    rows = [("shots reported", lambda r: f"{r['emitted']}"),
            ("real shots found (within 0.35 s)", lambda r: f"{r['found']} / {len(truth)}"),
            ("junk (not a real shot)", lambda r: f"{r['junk']}"),
            ("found within 1.0 s / junk", lambda r: f"{r['found_1s']} / {r['junk_1s']}"),
            ("clock offset vs review (median)", lambda r: f"{r['clock_offset_s']:+.2f} s"),
            ("shot type right", lambda r: "{} / {}".format(*r["type"])),
            ("near/far side right", lambda r: "{} / {}".format(*r["side"])),
            ("volley vs not right", lambda r: "{} / {}".format(*r["volley"])),
            ("serves found / false", lambda r: "{} / {}  ({} false)".format(*r["serves"])),
            ("rallies", lambda r: f"{r['rallies']}"),
            ("rally ends within 2 s", lambda r: "{} / {}".format(*r["ends"])),
            ("how the point ended, right", lambda r: "{} / {}".format(*r["end_reason"]))]
    w = max(len(k) for k, _ in rows)
    print(f"{'':{w}s}   {'PB Vision':>18s}   {'our pipeline':>18s}")
    for k, f in rows:
        print(f"{k:{w}s}   {f(res['PB Vision']):>18s}   {f(res['our pipeline']):>18s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

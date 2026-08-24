r"""One accumulating truth file per SOURCE VIDEO. Nothing the operator says is asked twice.

The problem this fixes, in the operator's words: *"The info I provide on shots should be saved
as truths and built upon so I don't have to keep reviewing the same info. MANY of these
mistakes have been identified multiple times."*

They are right, and the cause is structural. Truth about one video is currently spread across
`shot_review.json`, `missed_review.json`, `truth.json`, `serve_strikes.json`, two or three
`_labeling/labels*.csv`, and counts written only into `docs/ACCURACY_LEDGER.md` — different
shapes, different folders, and no merge step. Every review therefore started from nothing and
re-collected facts already given.

So: **one file per source video**, not per analysed folder. Folders get re-created whenever a
clip is re-analysed (there are eight `pb_5_minute_outdoor-*` folders for one video); the video
is what the truth is about.

Everything is keyed on **clock time**, never on shot_id. Shot ids are renumbered whenever
detection changes, and an id-keyed comparison silently compares different shots — that has
already produced one bogus result on this project.

A fact is never overwritten by a later import. If two sources disagree about the same moment,
both are kept with their provenance and the conflict is reported, because a silent overwrite
is how a correction gets lost.

    python -m tools.truth_store --import-all              # absorb every existing source
    python -m tools.truth_store --report
    python -m tools.truth_store --import-review data/<clip>   # a filled-in shot_review.xlsx
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STORE_DIR = Path("docs/truth")
SCHEMA_VERSION = 1
MATCH_TOL_S = 1.0        # the operator's times are hand-typed; same window score_shots uses

REAL_TYPES = {"serve", "return", "drive", "dink", "drop", "lob", "reset"}

# Which source wins when two disagree about the same shot. The operator watching the whole
# annotated video and correcting a named claim is the best evidence available; an old CSV from
# a labelling session that used a different workflow is the weakest. Without this the FIRST
# import won, which meant a 2026-08-17 spreadsheet overruled a review done today -- exactly
# backwards, and it produced 117 "conflicts" that were nothing of the kind.
AUTHORITY = {"review": 3, "operator_json": 2, "labels_csv": 1, "unknown": 0}

# The operator writes "opponent"; track_roles calls the same person opp_a or opp_b. Comparing
# those raw manufactures a disagreement out of a naming difference.
def norm_hitter(h) -> Optional[str]:
    h = str(h or "").strip().lower()
    if not h:
        return None
    if h.startswith("opp"):
        return "opponent"
    if h in ("user", "me", "self"):
        return "user"
    if h.startswith("partner"):
        return "partner"
    return h


def store_path(video: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in video)
    return STORE_DIR / f"{safe}.json"


def source_video(clip: Path) -> Optional[str]:
    """The video a clip was analysed from — the key the truth is filed under."""
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


def empty(video: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "video": video,
            "shots": [], "false_positives": [], "dead_intervals": [],
            "serve_strikes": [], "totals": {}, "provenance": []}


def load(video: str) -> dict:
    p = store_path(video)
    if not p.exists():
        return empty(video)
    return json.loads(p.read_text(encoding="utf-8"))


def save(doc: dict) -> Path:
    p = store_path(doc["video"])
    p.parent.mkdir(parents=True, exist_ok=True)
    doc["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    for k in ("shots", "false_positives", "serve_strikes"):
        doc[k] = sorted(doc[k], key=lambda r: float(r.get("t_sec", 0)))
    doc["dead_intervals"] = sorted(doc["dead_intervals"], key=lambda r: r[0])
    p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return p


def _find(rows: List[dict], t: float, tol: float = MATCH_TOL_S,
          claimed: Optional[set] = None) -> Optional[dict]:
    """Nearest stored shot within `tol`, skipping any already claimed in this import.

    One-to-one on purpose. A ±1 s window is right for hand-typed times but wide enough to
    span three shots of a kitchen exchange, and without the claim set one stored shot
    absorbed several distinct ones and then "conflicted" with all of them.
    """
    best, bd = None, tol + 1e-9
    for r in rows:
        if claimed is not None and id(r) in claimed:
            continue
        d = abs(float(r.get("t_sec", -999)) - t)
        if d < bd:
            best, bd = r, d
    return best


def add_shot(doc: dict, t: float, *, type_: Optional[str] = None,
             hitter: Optional[str] = None, side: Optional[str] = None,
             volley: Optional[bool] = None, detected: Optional[bool] = None,
             source: str = "", notes: str = "", kind: str = "unknown",
             claimed: Optional[set] = None, volley_explicit: bool = False) -> str:
    """Merge one shot fact. Returns 'new' | 'enriched' | 'agreed' | 'CONFLICT'.

    Enrichment only ever ADDS fields that were unknown. A field already recorded is never
    replaced by a later import — if they differ the row keeps the original, records the
    disagreement, and the caller reports it. Losing a correction silently is the failure this
    whole file exists to prevent.
    """
    hitter = norm_hitter(hitter)
    auth = AUTHORITY.get(kind, 0)
    ex = _find(doc["shots"], t, claimed=claimed)
    if ex is None:
        row = {k: v for k, v in
               {"t_sec": round(float(t), 2), "type": type_, "hitter": hitter,
                "side": side, "volley": volley, "detected": detected,
                "volley_explicit": volley_explicit or None,
                "source": source, "notes": notes, "authority": auth}.items()
               if v is not None and v != ""}
        doc["shots"].append(row)
        if claimed is not None:
            claimed.add(id(row))
        return "new"
    if claimed is not None:
        claimed.add(id(ex))
    prev_auth = int(ex.get("authority", 0))
    verdict = "agreed"
    for key, val in (("type", type_), ("hitter", hitter), ("side", side),
                     ("volley", volley), ("detected", detected)):
        if val is None or val == "":
            continue
        cur = ex.get(key)
        if cur is None:
            ex[key] = val
            if verdict == "agreed":
                verdict = "enriched"
        elif cur == val:
            continue
        elif auth > prev_auth:
            # A better source disagrees: it wins, and what it replaced is kept so the change
            # is inspectable rather than invisible.
            ex.setdefault("superseded", []).append(
                {"field": key, "was": cur, "now": val, "by": source})
            ex[key] = val
            verdict = "updated"
        elif auth < prev_auth:
            continue                     # a weaker source disagreeing is not news
        else:
            ex.setdefault("conflicts", []).append(
                {"field": key, "kept": cur, "rejected": val, "from": source})
            verdict = "CONFLICT"
    if volley_explicit:
        # The operator actually judged this one. Distinguishing that from a value inherited
        # by "blank means agree" matters: scoring against inherited values would be scoring
        # against ourselves, and would read as agreement no matter how wrong we are.
        ex["volley_explicit"] = True
    if auth > prev_auth:
        ex["authority"] = auth
        ex["source"] = source
    if notes and notes not in (ex.get("notes") or ""):
        ex["notes"] = ((ex.get("notes") or "") + " | " + notes).strip(" |")
    return verdict


def note_provenance(doc: dict, source: str, counts: Dict[str, int]) -> None:
    doc["provenance"].append({"at": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "source": source, **counts})


# ---------------------------------------------------------------- importers

def import_review_xlsx(doc: dict, clip: Path) -> Dict[str, int]:
    """A filled-in shot_review.xlsx: corrections, confirmations, and missed shots."""
    from openpyxl import load_workbook
    from tools.shot_review_sheet import OUT_NAME, parse_clock
    p = clip / "_labeling" / OUT_NAME
    if not p.exists():
        return {}
    ws = load_workbook(p, data_only=True).active
    hdr = next((r for r in range(1, 20)
                if str(ws.cell(row=r, column=1).value or "").strip() == "#"), None)
    if hdr is None:
        return {}
    col = {str(ws.cell(row=hdr, column=i).value or "").strip(): i
           for i in range(1, ws.max_column + 1)}
    for need in ("CORRECT_TYPE", "CORRECT_VOLLEY", "notes"):
        col.setdefault(need, {"CORRECT_TYPE": 7, "CORRECT_VOLLEY": 8, "notes": 9}[need])
    c = Counter()
    claimed: set = set()
    src = f"shot_review.xlsx / {clip.name}"
    for r in range(hdr + 1, ws.max_row + 1):
        n = ws.cell(row=r, column=1).value
        if isinstance(n, str) and not str(n).strip().isdigit():
            continue                                     # the worked-example row
        t = parse_clock(ws.cell(row=r, column=2).value)
        ours = str(ws.cell(row=r, column=5).value or "").strip().lower()
        corr = str(ws.cell(row=r, column=col["CORRECT_TYPE"]).value or "").strip().lower()
        # column layout differs between sheet generations (an ALREADY KNOWN column was
        # inserted at G), so find the columns by HEADER rather than by position -- a review
        # silently read from the wrong column is how 26 missed shots were lost once already.
        volley_txt = str(ws.cell(row=r, column=col["CORRECT_VOLLEY"]).value or "").strip().lower()
        notes = str(ws.cell(row=r, column=col["notes"]).value or "").strip()
        if t is None:
            if notes:
                doc.setdefault("free_notes", []).append({"source": src, "note": notes})
                c["free_notes"] += 1
            continue
        ty = corr or ours
        if not ty:
            continue
        vol = {"yes": True, "no": False}.get(volley_txt)
        # A BLANK is a deliberate confirmation, not an absence -- the operator: "if I did not
        # mark it as wrong, then I deliberately considered it to be correct." So every row in
        # a completed review is the operator's own answer, and scoring against it is scoring
        # against them, not against ourselves.
        #
        # This holds only while a review covers every row. A PARTIAL review must say so, or
        # its untouched rows would be recorded as confirmations of things nobody looked at.
        vol_explicit = True
        if vol is None and not corr and str(ws.cell(row=r, column=6).value or "").strip():
            vol = str(ws.cell(row=r, column=6).value).strip().lower() == "yes"
        detected = n not in (None, "")
        if ty == "not a shot":
            ex = _find(doc["false_positives"], t)
            if ex is None:
                doc["false_positives"].append({"t_sec": round(t, 2), "source": src,
                                               "notes": notes})
                c["false_positive"] += 1
            continue
        c["confirmed"] += 1 if not corr else 0
        c[add_shot(doc, t, type_=ty, hitter=str(ws.cell(row=r, column=3).value or "") or None,
                   side=str(ws.cell(row=r, column=4).value or "") or None, volley=vol,
                   detected=detected, source=src, notes=notes,
                   kind="review", claimed=claimed,
                   volley_explicit=vol_explicit)] += 1
        if not detected:
            c["missed"] += 1
    return dict(c)


def import_legacy(doc: dict, clip: Path) -> Dict[str, int]:
    """Everything the project already knew about this video, in its scattered forms."""
    c = Counter()
    claimed: set = set()
    src = f"legacy / {clip.name}"

    rv = clip / "shot_review.json"
    if rv.exists():
        d = json.loads(rv.read_text(encoding="utf-8"))
        for fp in d.get("false_positives", []):
            t = float(fp["t_sec"])
            if _find(doc["false_positives"], t) is None:
                doc["false_positives"].append({"t_sec": round(t, 2),
                                               "cause": fp.get("cause"),
                                               "notes": fp.get("note", ""), "source": src})
                c["false_positive"] += 1
        for m in d.get("missed", []):
            c[add_shot(doc, float(m["t_sec"]), detected=False, source=src,
                       notes=m.get("note", ""), kind="operator_json",
                       claimed=claimed)] += 1
            c["missed"] += 1
        for w in d.get("wrong_player", []):
            c[add_shot(doc, float(w["t_sec"]), hitter=w.get("actually"), source=src,
                       notes=w.get("note", ""), kind="operator_json",
                       claimed=claimed)] += 1

    mr = clip / "missed_review.json"
    if mr.exists():
        for m in json.loads(mr.read_text(encoding="utf-8")).get("missed", []):
            c[add_shot(doc, float(m["t_sec"]), type_=(m.get("shot_type") or None),
                       hitter=(m.get("who") or None), detected=False, source=src,
                       notes=m.get("note", ""), kind="operator_json",
                       claimed=claimed)] += 1
            c["missed"] += 1

    lab = clip / "_labeling"
    if lab.is_dir():
        for f in sorted(lab.glob("labels*.csv")):
            # labels_from_review.csv is GENERATED from shot_review.xlsx, which is imported
            # separately at higher authority. Reading it too would re-enter the same facts
            # as a weaker source and manufacture conflicts with their own origin.
            if f.name == "labels_from_review.csv":
                continue
            with f.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    ty = (row.get("true_type") or "").strip().lower()
                    if ty not in REAL_TYPES:
                        continue
                    from tools.shot_review_sheet import parse_clock
                    t = parse_clock(row.get("time")) if row.get("time") else None
                    if t is None and row.get("frame"):
                        try:
                            t = float(row["frame"]) / _fps(clip)
                        except (ValueError, ZeroDivisionError):
                            t = None
                    if t is None:
                        continue
                    vol = {"y": True, "n": False}.get((row.get("true_volley") or "").strip())
                    c[add_shot(doc, t, type_=ty, volley=vol,
                               hitter=(row.get("hitter_role") or None),
                               source=f"{src}:{f.name}",
                               notes=(row.get("notes") or ""),
                               kind="labels_csv", claimed=claimed)] += 1

    ss = clip / "serve_strikes.json"
    if ss.exists():
        for s in json.loads(ss.read_text(encoding="utf-8")).get("strikes", []):
            if s.get("skipped") or s.get("t_sec") is None:
                continue
            if _find(doc["serve_strikes"], float(s["t_sec"])) is None:
                doc["serve_strikes"].append({"t_sec": round(float(s["t_sec"]), 2),
                                             "source": src})
                c["serve_strike"] += 1

    tj = clip / "truth.json"
    if tj.exists():
        d = json.loads(tj.read_text(encoding="utf-8"))
        for k, v in (d.get("totals") or {}).items():
            doc["totals"].setdefault(k, v)
        pts = d.get("points") or []
        for a, b in zip(pts, pts[1:]):
            lo, hi = float(a.get("end_t_sec", 0)), float(b.get("start_t_sec", 0))
            if hi > lo and not any(abs(x[0] - lo) < 0.5 for x in doc["dead_intervals"]):
                doc["dead_intervals"].append([round(lo, 2), round(hi, 2)])
                c["dead_interval"] += 1
    return dict(c)


def _fps(clip: Path) -> float:
    try:
        return float(json.loads((clip / "court.json").read_text(encoding="utf-8"))
                     ["video"]["fps"])
    except (OSError, KeyError, json.JSONDecodeError):
        return 60.0


def known_shots(clip: Path) -> List[dict]:
    """Every shot fact known about the video this clip came from. The accessor the rest of
    the codebase should use — scorers and the review sheet both read this, so the operator's
    input has exactly one home."""
    v = source_video(clip)
    return load(v)["shots"] if v else []


def known(clip: Path) -> dict:
    v = source_video(clip)
    return load(v) if v else empty("?")


def report(video: str) -> int:
    doc = load(video)
    shots = doc["shots"]
    typed = [s for s in shots if s.get("type")]
    missed = [s for s in shots if s.get("detected") is False]
    print(f"{video}\n  stored at {store_path(video)}")
    print(f"\n  {len(shots):>4} shots known")
    print(f"  {len(typed):>4} with a type" +
          ("   " + ", ".join(f"{k} {v}" for k, v in
                             Counter(s["type"] for s in typed).most_common()) if typed else ""))
    print(f"  {len(missed):>4} we FAILED to detect")
    print(f"  {len(doc['false_positives']):>4} detections that are not shots")
    print(f"  {len(doc['serve_strikes']):>4} marked serve strikes")
    print(f"  {len(doc['dead_intervals']):>4} known between-point intervals")
    if doc.get("free_notes"):
        print(f"  {len(doc['free_notes']):>4} free-text notes")
    sup = [s for s in shots if s.get("superseded")]
    if sup:
        print(f"  {len(sup):>4} updated by a better source (the change is kept, inspectable)")
    conf = [s for s in shots if s.get("conflicts")]
    if conf:
        print(f"\n  {len(conf)} CONFLICT(s) — two sources disagree, original kept:")
        for s in conf[:10]:
            for cf in s["conflicts"]:
                print(f"    {s['t_sec']:>8.2f}s  {cf['field']}: kept {cf['kept']!r}, "
                      f"rejected {cf['rejected']!r} (from {cf['from']})")
    if doc["totals"]:
        print(f"\n  totals: {json.dumps(doc['totals'])}")
    print(f"\n  provenance:")
    for p in doc["provenance"][-8:]:
        bits = ", ".join(f"{k} {v}" for k, v in p.items() if k not in ("at", "source"))
        print(f"    {p['at'][:19]}  {p['source']:<40} {bits}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import-review", type=Path, default=None, metavar="CLIP")
    ap.add_argument("--import-legacy", type=Path, default=None, metavar="CLIP")
    ap.add_argument("--import-all", action="store_true",
                    help="absorb every source under data/ into the per-video stores")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--video", default=None, help="which video to report on")
    a = ap.parse_args(argv)

    if a.import_all:
        clips = sorted(p for p in Path("data").iterdir()
                       if p.is_dir() and (p / "ball.meta.json").exists())
        touched = {}
        for clip in clips:
            v = source_video(clip)
            if not v:
                continue
            doc = touched.get(v) or load(v)
            # weakest source first, so the strongest one supersedes rather than colliding
            got = Counter(import_legacy(doc, clip))
            got.update(import_review_xlsx(doc, clip))
            if got:
                note_provenance(doc, f"import from {clip.name}", dict(got))
                print(f"  {clip.name:<34} -> {v:<28} {dict(got)}")
            touched[v] = doc
        for v, doc in touched.items():
            print(f"saved {save(doc)}")
        return 0

    for flag, fn in ((a.import_review, import_review_xlsx), (a.import_legacy, import_legacy)):
        if flag is None:
            continue
        v = source_video(flag)
        if not v:
            raise SystemExit(f"cannot tell which video {flag} came from")
        doc = load(v)
        got = fn(doc, flag)
        note_provenance(doc, f"{fn.__name__} from {flag.name}", got)
        print(f"{flag.name} -> {v}: {got}")
        print(f"saved {save(doc)}")
        return 0

    if a.report:
        vids = ([a.video] if a.video else
                sorted(p.stem for p in STORE_DIR.glob("*.json")) if STORE_DIR.is_dir() else [])
        if not vids:
            raise SystemExit("no truth stored yet — run --import-all")
        for i, v in enumerate(vids):
            doc = load(v) if (STORE_DIR / f"{v}.json").exists() else None
            if doc is None:
                continue
            if i:
                print()
            report(doc["video"])
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

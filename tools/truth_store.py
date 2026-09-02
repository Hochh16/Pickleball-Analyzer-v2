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
# How far a rally-level end may trail the ending SHOT and still be the same point. Measured
# 0.93-1.87s across court C's ten points; 2.5 covers that without reaching the 1.08s gap
# between two genuinely distinct ends on the acceptance clip (which are same-source anyway).
END_SHADOW_S = 2.5
# Between two sources of EQUAL authority the LATER one wins, per the operator: "use the last
# one I built as the truth if there is a conflict between any reviews." Passed as `seq`, which
# the caller increments per import.


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


# --- reading the operator's free-text notes ---------------------------------------------
# They record more than the columns ask for: "in the xls I noted in the comments when either
# the rally ended and/or it was a winning shot". Their note at 32.57s says the rest --
# "Corrected this and end of rally multiple times. Doesn't seem to capture this info in one
# place" -- which is the whole reason this store exists. A first pass at the phrasing they
# actually used; anything not matched is kept verbatim so nothing is lost to a regex.
import re as _re

NOT_A_SHOT_RE = _re.compile(r"(not a sh(o|i)rt|not a shot|no shot)", _re.I)
RALLY_END_RE = _re.compile(
    r"(rally (ended|ends|is over|over)|end of rally|winning shot|hit (it )?out"
    r"|never made it back|missed by (user|opponent|partner))", _re.I)
THIRD_DROP_RE = _re.compile(r"3rd shot drop", _re.I)


# The operator types the shot the way a player says it: "drive/volley", "backhand drive",
# "3rd shot drive". Compared raw against the canonical set those are simply absent -- and
# worse, tools/score_shot_types treats an unrecognised type as a NON-SHOT label, so six real
# shots on court B were being reported as false positives over a wording difference. Eight of
# that clip's twenty-one typed shots were unusable this way.
#
# Splitting them also RECOVERS information: "drive/volley" says the shot was a volley, which
# is a field of its own, and "3rd shot drop" says where in the rally it fell.
#
# Word matching is done by SPLITTING, not by a word-boundary regex. Writing one here has now
# twice produced a literal backspace character instead of a boundary, and the pattern then
# matches nothing while looking correct -- it silently discarded 30 of the operator's notes
# once already.
# "reset" is deliberately NOT here. Operator, 2026-08-26: every reset is a drop or a dink,
# and a drop/dink IS a reset when it answers a drive -- a qualifier, derived in Stage 6, not
# a type. Left out of the canonical set, an older "reset" label passes through unchanged and
# stays visible for the operator to resolve as drop or dink.
CANON_TYPES = ("serve", "return", "drive", "drop", "dink", "lob")
STROKE_WORDS = ("backhand", "forehand", "bh", "fh", "handed", "two")


def norm_type(raw):
    """'drive/volley' -> ('drive', True, None); '3rd shot drop' -> ('drop', None, '3rd').

    Returns (type, volley_or_None, rally_position_or_None). An input yielding no canonical
    word comes back UNCHANGED: a vocabulary this cannot read must stay visible rather than be
    dropped or coerced into the nearest guess.
    """
    s = str(raw or "").strip().lower()
    if not s:
        return (None, None, None)
    if s in CANON_TYPES:
        return (s, None, None)
    words = [w for w in _re.split(r"[^a-z0-9]+", s) if w]
    volley = True if "volley" in words else None
    third = "3rd" if "3rd" in words else None
    hits = [w for w in words
            if w in CANON_TYPES and w not in STROKE_WORDS]
    if len(set(hits)) == 1:
        return (hits[0], volley, third)
    if volley and not hits:
        # "volley" alone is not a type -- it says HOW the shot was taken, not what it was
        return (None, True, third)
    return (s, volley, third)


def read_note(note: str) -> dict:
    """What a free-text note asserts. Flags only -- the note itself is always kept."""
    n = note or ""
    return {"not_a_shot": bool(NOT_A_SHOT_RE.search(n)),
            "rally_end": bool(RALLY_END_RE.search(n)),
            "third_shot_drop": bool(THIRD_DROP_RE.search(n))}


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
          claimed: Optional[set] = None, key: Optional[str] = None) -> Optional[dict]:
    """Nearest stored shot within `tol`, skipping any already claimed in this import.

    A row that carries a `key` matches its OWN previous entry first, whatever the times say.
    Without that the store was not idempotent: re-running the same import appended clones
    (8 pairs of them), because two review rows less than the ±1 s window apart would each
    claim the other's entry and the loser appended a fresh one. Duplicated truth inflates
    the shot total and every rate computed from it.

    One-to-one on purpose. A ±1 s window is right for hand-typed times but wide enough to
    span three shots of a kitchen exchange, and without the claim set one stored shot
    absorbed several distinct ones and then "conflicted" with all of them.
    """
    if key:
        for r in rows:
            if r.get("key") == key:
                return r
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
             claimed: Optional[set] = None, volley_explicit: bool = False,
             seq: int = 0, key: Optional[str] = None,
             existing: Optional[dict] = None, assigned: bool = False) -> str:
    """Merge one shot fact. Returns 'new' | 'enriched' | 'agreed' | 'CONFLICT'.

    Enrichment only ever ADDS fields that were unknown. A field already recorded is never
    replaced by a later import — if they differ the row keeps the original, records the
    disagreement, and the caller reports it. Losing a correction silently is the failure this
    whole file exists to prevent.
    """
    hitter = norm_hitter(hitter)
    # Split the operator's phrasing into the fields it actually carries. "drive/volley" is a
    # drive AND a volley; scoring it as a type named "drive/volley" matches nothing, and the
    # volley it states is thrown away. Only FILL a volley the caller did not judge -- a
    # CORRECT_VOLLEY column the operator filled in must win over a word in the type.
    type_, ty_volley, ty_third = norm_type(type_)
    if ty_volley is not None and volley is None:
        volley = ty_volley
        volley_explicit = True          # they said "volley"; that is a judgement, not a guess
    auth = AUTHORITY.get(kind, 0)
    # `assigned` means the caller matched this row globally (shortest pair first) and is
    # telling us which entry it belongs to -- `existing=None` then means "no entry, create
    # one", not "look one up". Row-order greedy matching cascades: once row #26 took the
    # entry nearest IT, row #27 took the one that belonged to #28, and three of the
    # operator's notes landed on the wrong shots.
    ex = existing if assigned else _find(doc["shots"], t, claimed=claimed, key=key)
    if ex is None:
        row = {k: v for k, v in
               {"t_sec": round(float(t), 2), "type": type_, "hitter": hitter,
                "side": side, "volley": volley, "detected": detected,
                "rally_position": ty_third,
                "volley_explicit": volley_explicit or None,
                "source": source, "notes": notes, "authority": auth, "seq": seq,
                "key": key}.items()
               if v is not None and v != ""}
        doc["shots"].append(row)
        if claimed is not None:
            claimed.add(id(row))
        return "new"
    if claimed is not None:
        claimed.add(id(ex))
    if key and not ex.get("key"):
        ex["key"] = key                  # adopt the identity so the NEXT import is stable
    prev_auth = int(ex.get("authority", 0))
    prev_seq = int(ex.get("seq", 0))
    newer = (auth, seq) > (prev_auth, prev_seq)
    older = (auth, seq) < (prev_auth, prev_seq)
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
        elif newer:
            # A better source disagrees: it wins, and what it replaced is kept so the change
            # is inspectable rather than invisible.
            ex.setdefault("superseded", []).append(
                {"field": key, "was": cur, "now": val, "by": source})
            ex[key] = val
            verdict = "updated"
        elif older:
            continue                     # an older or weaker source disagreeing is not news
        else:
            ex.setdefault("conflicts", []).append(
                {"field": key, "kept": cur, "rejected": val, "from": source})
            verdict = "CONFLICT"
    if type_ and ex.get("not_a_shot") and newer:
        # A retraction has to be reversible. An OLDER review called #18 "no shot, ball
        # rolling along back fence" at 48.617s; shot numbers were renumbered before the next
        # review, and at that time the operator's LATEST sheet says #19, a return that ended
        # the point. Their rule is explicit -- "use the last one I built as the truth if
        # there is a conflict between any reviews" -- but not_a_shot was one-way, so the
        # older note won and a real return counted as junk.
        ex["not_a_shot"] = False
        ex.setdefault("superseded", []).append(
            {"field": "not_a_shot", "was": True, "now": False, "by": source})
        for i, f in enumerate(doc.get("false_positives", [])):
            if abs(float(f.get("t_sec", -999)) - t) <= MATCH_TOL_S:
                doc["false_positives"].pop(i)
                break                    # ...and it is no longer a false positive either
    if ty_third and not ex.get("rally_position"):
        ex["rally_position"] = ty_third      # "3rd shot drop" says where in the rally it fell
    if volley_explicit:
        # The operator actually judged this one. Distinguishing that from a value inherited
        # by "blank means agree" matters: scoring against inherited values would be scoring
        # against ourselves, and would read as agreement no matter how wrong we are.
        ex["volley_explicit"] = True
    if newer:
        # ...including the TIME. A shot first recorded from missed_shots.csv at 2:19.20 and
        # later confirmed by review row #54 at 2:18.22 kept the older, hand-typed time, and
        # then read as sitting 0.9 s outside the rally it is plainly in. Every scorer matches
        # on a time window, so a stale t_sec is not cosmetic -- it silently moves a shot.
        if abs(float(ex.get("t_sec", t)) - float(t)) > 0.01:
            ex.setdefault("superseded", []).append(
                {"field": "t_sec", "was": ex.get("t_sec"), "now": round(float(t), 2),
                 "by": source})
            ex["t_sec"] = round(float(t), 2)
        ex["authority"] = auth
        ex["seq"] = seq
        ex["source"] = source
    if notes and notes not in (ex.get("notes") or ""):
        ex["notes"] = ((ex.get("notes") or "") + " | " + notes).strip(" |")
    return verdict


def fold_shadowed_legacy(doc: dict) -> int:
    """Drop a legacy label that a reviewed row already covers.

    The review sheet enumerates EVERY shot in the video, so a `labels*.csv` entry within the
    match window of a reviewed row is the same shot written down twice -- and the operator
    was explicit that the last sheet they built wins any conflict. Left in place the pair
    counts as two shots, which is the same silent inflation the retraction fix removed.
    """
    # to a fixed point: folding one row frees the claim that was hiding the next, and a
    # single pass left the store still changing on the following run
    total = 0
    while True:
        n = _fold_once(doc)
        total += n
        if not n:
            break
    return (total + _demote_stale_legacy(doc) + _fold_shadowed_ends(doc)
            + _fold_shadowed_fps(doc))


def _demote_stale_legacy(doc: dict) -> int:
    """Inside the span a review sheet covers, the review is the complete account.

    The sheet lists every shot we detected AND lets the operator add the ones we missed, so a
    label from an older file at a time the sheet covers, with no row of its own, is
    contradicted by it. Four such labels stood as confirmed real shots at 0:42.20, 1:16.30,
    3:59.20 and 4:58.10 while the latest sheet marks the nearby rows "not a shot" -- and at
    three of those times we detect nothing at all. Operator's rule: "use the last one I built
    as the truth if there is a conflict between any reviews", and their decision to apply it
    here (2026-08-24).

    Demoted, not deleted: the entries move to `superseded_shots` so the older reading stays
    inspectable and can be restored if a later review disagrees.
    """
    keyed = [s for s in doc["shots"] if s.get("key")]
    if not keyed:
        return 0                          # no review for this video: nothing overrules
    lo = min(float(s["t_sec"]) for s in keyed)
    hi = max(float(s["t_sec"]) for s in keyed)
    keep, moved = [], []
    for s in doc["shots"]:
        if (not s.get("key") and not s.get("not_a_shot")
                and lo <= float(s.get("t_sec", -999)) <= hi):
            moved.append(s)
            continue
        keep.append(s)
    if not moved:
        return 0
    doc["shots"] = keep
    prior = doc.setdefault("superseded_shots", [])
    for s in moved:
        # the legacy import re-creates these every run; do not re-record them
        if not any(abs(float(x.get("t_sec", -999)) - float(s["t_sec"])) < 0.01
                   and x.get("source") == s.get("source") for x in prior):
            prior.append({**s, "superseded_by": "the review sheet covering this span"})
    return len(moved)


def _fold_shadowed_fps(doc: dict) -> int:
    """A junk mark from an older file that a REVIEW row already covers is the same detection.

    Once review rows got their own keyed entries they stopped merging with the legacy ones,
    which is right for two distinct detections 0.27s apart -- and wrong for the same
    detection recorded twice. Seven junk moments on the acceptance clip were being counted
    twice, at identical timestamps, which inflates the junk total and every rate built on it.
    """
    fps = doc.get("false_positives") or []
    review = [f for f in fps if f.get("key")]
    if not review:
        return 0
    keep, moved = [], []
    for f in fps:
        if f.get("key"):
            keep.append(f)
        elif any(abs(float(r["t_sec"]) - float(f["t_sec"])) <= MATCH_TOL_S for r in review):
            moved.append(f)
        else:
            keep.append(f)
    if not moved:
        return 0
    doc["false_positives"] = keep
    prior = doc.setdefault("superseded_false_positives", [])
    for f in moved:
        if not any(abs(float(x["t_sec"]) - float(f["t_sec"])) < 0.01
                   and x.get("source") == f.get("source") for x in prior):
            prior.append(f)
    return len(moved)


def _fold_shadowed_ends(doc: dict) -> int:
    """A rally END marked on the SHOT and the rally-level end TIME are one point, not two.

    The review sheet's RALLY_END marks the shot that ended the point; truth.json's
    `end_t_sec` marks when the rally was over, which trails the ending strike by 0.9-1.9s
    (measured across court C's ten points). They are the same event, so keeping both doubles
    the rally-end count -- court C reported 20 ends for 10 points.

    Both facts are kept -- the shot-level end carries `rally_over_t_sec` -- because they
    answer different questions: which shot ended the point, and when the ball was finally
    done. Only the COUNT is deduplicated.

    Matched ACROSS SOURCES only, never within one. Two ends 1.08s apart from the same review
    are two real points on the acceptance clip, and a window alone would merge them.
    """
    ends = doc.get("rally_ends") or []
    review = [e for e in ends if "shot_review" in str(e.get("source", ""))]
    if not review:
        return 0
    keep, moved = [], []
    for e in ends:
        if e in review:
            keep.append(e)
            continue
        near = [r for r in review
                if abs(float(r["t_sec"]) - float(e["t_sec"])) <= END_SHADOW_S]
        if near:
            # KEEP BOTH FACTS, COUNT ONE END. Operator, 2026-08-26: "rally ending last shot
            # and rally ending time should be compatible info but should not double the count
            # of rally ends." They are two measurements of one point -- the shot that ended it
            # and the moment the ball was done -- and each answers a different question, so
            # neither is discarded. The rally-level time rides along on the shot-level end.
            r = min(near, key=lambda x: abs(float(x["t_sec"]) - float(e["t_sec"])))
            r["rally_over_t_sec"] = round(float(e["t_sec"]), 2)
            r["rally_over_source"] = e.get("source")
            if e.get("notes") and e["notes"] not in (r.get("notes") or ""):
                r["notes"] = ((r.get("notes") or "") + " | " + e["notes"]).strip(" |")
            moved.append(e)
        else:
            keep.append(e)
    if not moved:
        return 0
    doc["rally_ends"] = keep
    return len(moved)


def _fold_once(doc: dict) -> int:
    keyed = [s for s in doc["shots"] if s.get("key")]
    drop = []
    for s in doc["shots"]:
        if s.get("key"):
            continue
        near = _find(keyed, float(s.get("t_sec", -999)))
        if near is not None:
            seen = {"t_sec": s.get("t_sec"), "type": s.get("type"),
                    "hitter": s.get("hitter"), "source": s.get("source")}
            # the legacy import re-creates this row on every run, so the fold must not keep
            # re-recording it -- the note would grow without bound
            if seen not in near.setdefault("also_seen", []):
                near["also_seen"].append(seen)
            drop.append(id(s))
    if drop:
        doc["shots"] = [s for s in doc["shots"] if id(s) not in drop]
    return len(drop)


def note_provenance(doc: dict, source: str, counts: Dict[str, int]) -> None:
    doc["provenance"].append({"at": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "source": source, **counts})


# ---------------------------------------------------------------- importers

def _has_operator_marks(ws, hdr: int, col: dict) -> bool:
    """Has anyone actually reviewed this sheet, or is it as we generated it?

    Any of: a filled CORRECT_* / NOT_A_SHOT / RALLY_END / notes cell, or a row the operator
    added by hand (a time with no row number, i.e. a shot we missed entirely).
    """
    marks = [col[k] for k in ("CORRECT_TYPE", "CORRECT_VOLLEY", "NOT_A_SHOT", "RALLY_END",
                              "notes") if k in col]
    for r in range(hdr + 1, ws.max_row + 1):
        n = ws.cell(row=r, column=1).value
        if isinstance(n, str) and not str(n).strip().isdigit():
            continue                                     # the worked-example row
        if any(str(ws.cell(row=r, column=i).value or "").strip() for i in marks):
            return True
        if n in (None, "") and str(ws.cell(row=r, column=2).value or "").strip():
            return True                                  # a hand-added missed shot
    return False


def import_review_xlsx(doc: dict, clip: Path) -> Dict[str, int]:
    """A filled-in shot_review.xlsx: corrections, confirmations, and missed shots."""
    from openpyxl import load_workbook
    from tools.shot_review_sheet import OUT_NAME, parse_clock
    p = clip / "_labeling" / OUT_NAME
    if not p.exists():
        return {}
    ws = load_workbook(p, data_only=True).active
    seq = int(p.stat().st_mtime)          # "use the last one I built" -- newer file wins
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
    if not _has_operator_marks(ws, hdr, col):
        # A blank row means "the operator looked at this and agreed" -- but ONLY in a sheet
        # the operator actually worked through. An untouched, freshly built sheet is all
        # blank rows, so importing it would file OUR OWN detections as operator truth at the
        # highest authority and every accuracy figure for that clip would then be us scoring
        # ourselves. This is the same class of bug as the label file that shadowed a fuller
        # set: it does not error, it just quietly turns into a good score.
        print(f"  {clip.name}: shot_review.xlsx has no operator marks -- not imported "
              f"(a prepopulated sheet is our output, not truth)")
        return {}
    parsed = []
    for r in range(hdr + 1, ws.max_row + 1):
        n0 = ws.cell(row=r, column=1).value
        if isinstance(n0, str) and not str(n0).strip().isdigit():
            continue                                     # the worked-example row
        parsed.append((r, n0, parse_clock(ws.cell(row=r, column=2).value)))

    # Assign each row to at most one existing entry, SHORTEST PAIR FIRST across the whole
    # sheet, before touching anything. One-to-one: a sheet row is one of our detections (or
    # one the operator added), so two rows can never be the same stored shot.
    assign: Dict[int, dict] = {}
    taken: set = set()
    by_key = {s.get("key"): s for s in doc["shots"] if s.get("key")}
    cand = []
    for r, n0, tt in parsed:
        if tt is None:
            continue
        k = f"{src}#{str(n0).strip()}" if n0 not in (None, "") else f"{src}@{tt:.2f}"
        own = by_key.get(k)
        if own is not None and id(own) not in taken:
            assign[r] = own                              # its own entry from a prior import
            taken.add(id(own))
            continue
        for s in doc["shots"]:
            d = abs(float(s.get("t_sec", -999)) - tt)
            if d <= MATCH_TOL_S:
                cand.append((d, r, id(s), s))
    for d, r, sid, s in sorted(cand, key=lambda x: (x[0], x[1])):
        if r in assign or sid in taken:
            continue
        assign[r] = s
        taken.add(sid)

    for r, n, t in parsed:
        ours = str(ws.cell(row=r, column=col.get("our_type", 5)).value or "").strip().lower()
        corr = str(ws.cell(row=r, column=col["CORRECT_TYPE"]).value or "").strip().lower()
        known_prev = (str(ws.cell(row=r, column=col["ALREADY KNOWN"]).value or "").strip().lower()
                      if "ALREADY KNOWN" in col else "")
        # column layout differs between sheet generations (an ALREADY KNOWN column was
        # inserted at G), so find the columns by HEADER rather than by position -- a review
        # silently read from the wrong column is how 26 missed shots were lost once already.
        volley_txt = str(ws.cell(row=r, column=col["CORRECT_VOLLEY"]).value or "").strip().lower()
        notes = str(ws.cell(row=r, column=col["notes"]).value or "").strip()
        if t is None:
            if notes:
                fn = doc.setdefault("free_notes", [])
                # a set of facts, not an append log: re-importing must not duplicate them
                if not any(x.get("note") == notes and x.get("source") == src for x in fn):
                    fn.append({"source": src, "note": notes})
                    c["free_notes"] += 1
            continue
        flags = read_note(notes)
        # Dedicated columns win over the note text: a column cannot be silently mis-parsed,
        # and a regex over free text already lost 30 of these once.
        if "NOT_A_SHOT" in col and str(ws.cell(row=r, column=col["NOT_A_SHOT"]).value
                                       or "").strip().lower().startswith("y"):
            flags["not_a_shot"] = True
        if "RALLY_END" in col and str(ws.cell(row=r, column=col["RALLY_END"]).value
                                      or "").strip().lower().startswith("y"):
            flags["rally_end"] = True
        detected_row = n not in (None, "")
        row_key = (f"{src}#{str(n).strip()}" if detected_row else f"{src}@{t:.2f}")
        ty = corr or (known_prev.split("  ")[0] if known_prev else "") or ours
        if flags["not_a_shot"]:
            # The operator recorded these in the notes because the sheet had nowhere else to
            # put them: "not a shot. Between rallies. Opponent feeding ball to their partner".
            # Reading them is the difference between 0 and ~30 false positives from this
            # review.
            ty = "not a shot"
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
        c_ov = col.get("our_volley", 6)
        if vol is None and not corr and str(ws.cell(row=r, column=c_ov).value or "").strip():
            vol = str(ws.cell(row=r, column=c_ov).value).strip().lower() == "yes"
        detected = n not in (None, "")
        if flags["rally_end"]:
            if not any(abs(x["t_sec"] - t) < 0.5 for x in doc.setdefault("rally_ends", [])):
                doc["rally_ends"].append({"t_sec": round(t, 2), "source": src,
                                          "notes": notes})
                c["rally_end"] += 1
        if flags["third_shot_drop"]:
            doc.setdefault("third_shot_drops", [])
            if not any(abs(x - t) < 0.5 for x in doc["third_shot_drops"]):
                doc["third_shot_drops"].append(round(t, 2))
                c["third_shot_drop"] += 1
        if ty == "not a shot":
            # Keyed and one-to-one, exactly like the shots list. Proximity alone swallowed 4
            # of the operator's 21 NOT_A_SHOT marks on court C: two junk detections 0.27s
            # apart are two detections, and collapsing them under-counts the junk we emit --
            # which flatters every false-positive figure computed from it.
            ex = next((f for f in doc["false_positives"] if f.get("key") == row_key), None)
            if ex is None and not row_key:
                ex = _find(doc["false_positives"], t)
            if ex is None:
                doc["false_positives"].append({"t_sec": round(t, 2), "source": src,
                                               "notes": notes, "key": row_key})
                c["false_positive"] += 1
            else:
                ex["t_sec"] = round(t, 2)
                if notes and notes not in (ex.get("notes") or ""):
                    ex["notes"] = ((ex.get("notes") or "") + " | " + notes).strip(" |")
            # ...and retract it from the shots list. An earlier, weaker source may already
            # have recorded this moment as a typed shot; leaving that behind means the same
            # detection counts as BOTH a real shot and a false positive, which inflates the
            # real-shot total and quietly flatters every accuracy figure computed from it.
            prior = assign.get(r)
            if prior is not None and int(prior.get("authority", 0)) <= AUTHORITY["review"]:
                prior["not_a_shot"] = True
                prior["type"] = None
                prior["source"] = src
                prior["notes"] = ((prior.get("notes") or "") + " | " + notes).strip(" |")
                c["retracted"] += 1
            continue
        c["confirmed"] += 1 if not corr else 0
        # A corrected hitter REPLACES ours. Until these columns existed the operator wrote
        # it in the notes -- "shot was by opponent on far side", "dink by partner" -- and
        # this importer, reading only the structured columns, kept the wrong player on all
        # four. Server attribution is scored against this exact field.
        _hit = str(ws.cell(row=r, column=col.get("hitter", 3)).value or "").strip() or None
        _side = str(ws.cell(row=r, column=col.get("side", 4)).value or "").strip() or None
        if col.get("CORRECT_HITTER"):
            _c = str(ws.cell(row=r, column=col["CORRECT_HITTER"]).value or "").strip()
            if _c:
                _hit = _c
        if col.get("CORRECT_SIDE"):
            _c = str(ws.cell(row=r, column=col["CORRECT_SIDE"]).value or "").strip()
            if _c:
                _side = _c
        c[add_shot(doc, t, key=row_key, type_=ty,
                   hitter=_hit,
                   side=_side,
                   volley=vol,
                   detected=detected, source=src, notes=notes,
                   kind="review", claimed=claimed, existing=assign.get(r), assigned=True,
                   volley_explicit=vol_explicit, seq=seq)] += 1
        if not detected:
            c["missed"] += 1

    # An OLDER review's false-positive claim cannot stand where the LATEST sheet says there
    # is a real shot. The old review names SHOT NUMBERS -- "#18 is mislabeled", "#124, #125
    # were not shots" -- and the numbering changed between reviews, so those notes now land
    # on different shots. Seven times were counted as junk AND as a confirmed shot because of
    # it, including the return at 48.62 that ends a point. Operator's rule, verbatim: "use
    # the last one I built as the truth if there is a conflict between any reviews."
    confirmed = [s for s in doc["shots"]
                 if not s.get("not_a_shot") and str(s.get("key") or "").startswith(src)]
    keep, dropped = [], []
    for f in doc["false_positives"]:
        if f.get("source") != src and any(
                abs(float(s["t_sec"]) - float(f["t_sec"])) <= 0.4 for s in confirmed):
            dropped.append(f)
            continue
        keep.append(f)
    if dropped:
        doc["false_positives"] = keep
        prior = doc.setdefault("superseded_false_positives", [])
        for f in dropped:
            # the legacy import re-creates these every run, so this list must be a SET of
            # facts, not an append log -- it reached 21 entries for 7 overruled claims
            if not any(abs(float(x.get("t_sec", -999)) - float(f["t_sec"])) < 0.01
                       and x.get("source") == f.get("source") for x in prior):
                prior.append(f)
        c["fp_overruled_by_latest_review"] = len(dropped)
    return dict(c)


def import_corrections_csv(doc: dict, clip: Path) -> Dict[str, int]:
    """`_labeling/corrections.csv` — corrections the operator gave OUTSIDE a review sheet.

    They correct things in conversation ("reset at 1:11.24 is my mistake, it's a drop"), and
    those have to land somewhere durable. Editing their own xlsx is wrong twice over: it is
    their file, and Excel holds a lock on it half the time. A direct edit to the store is
    worse -- it is rebuilt from sources on every import, so the correction would silently
    vanish at the next `--import-all`.

    Columns: time, type, hitter, volley, not_a_shot, note. Review authority, and a seq above
    any sheet's mtime so a correction given after a review wins over it.
    """
    p = clip / "_labeling" / "corrections.csv"
    if not p.exists():
        return {}
    from tools.shot_review_sheet import parse_clock
    c = Counter()
    src = f"corrections.csv / {clip.name}"
    seq = int(p.stat().st_mtime) + 10 ** 9      # always newer than a sheet
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = parse_clock(row.get("time"))
            if t is None:
                continue
            note = (row.get("note") or "").strip()
            if str(row.get("not_a_shot") or "").strip().lower().startswith("y"):
                if _find(doc["false_positives"], t) is None:
                    doc["false_positives"].append({"t_sec": round(t, 2), "source": src,
                                                   "notes": note})
                    c["false_positive"] += 1
                prior = _find(doc["shots"], t)
                if prior is not None:
                    prior["not_a_shot"] = True
                    prior["type"] = None
                    c["retracted"] += 1
                continue
            vol = {"yes": True, "no": False}.get(
                str(row.get("volley") or "").strip().lower())
            c[add_shot(doc, t, type_=(row.get("type") or "").strip().lower() or None,
                       hitter=(row.get("hitter") or "").strip() or None, volley=vol,
                       source=src, notes=note, kind="review",
                       volley_explicit=vol is not None, seq=seq)] += 1
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
        for pt in pts:
            # The operator's own point ENDS. They were sitting in per-clip truth.json where
            # only tools/detect_rally_ends.py --score could see them, so rally-end accuracy
            # could be measured on the indoor clips and nowhere else. Note the convention
            # caveat: this file's start_t_sec is NOT the serve strike (early by ~1.06 s
            # indoors), so treat end_t_sec as the operator's mark of the point ending rather
            # than a frame-accurate event, and score it with a window.
            et = pt.get("end_t_sec")
            if et is None:
                continue
            et = float(et)
            if not any(abs(float(x["t_sec"]) - et) < 0.5 for x in doc.setdefault("rally_ends", [])):
                doc["rally_ends"].append({"t_sec": round(et, 2), "source": src,
                                          "notes": (pt.get("end_reason") or "")})
                c["rally_end"] += 1
            # The rally as the operator described it: who served, how many shots, how it
            # ended. Only the end TIME was being read, so per-rally server truth for two
            # videos sat unused -- and shot-to-player attribution is a known weak point, so
            # that is the one axis where truth was most worth having.
            rt = doc.setdefault("rally_truth", [])
            st = pt.get("start_t_sec")
            if st is not None and not any(abs(float(x["start_t_sec"]) - float(st)) < 0.5
                                          for x in rt):
                rt.append({"start_t_sec": round(float(st), 2), "end_t_sec": round(et, 2),
                           "server": norm_hitter(pt.get("server")),
                           "n_shots": pt.get("n_shots"),
                           "end_reason": pt.get("end_reason"),
                           "note": pt.get("note") or "", "source": src})
                c["rally_truth"] += 1
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
    print(f"  {len(doc.get('rally_ends') or []):>4} rally ENDS read from the notes")
    print(f"  {len(doc.get('third_shot_drops') or []):>4} third shots the operator called a DROP")
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
            got.update(import_corrections_csv(doc, clip))
            if got:
                note_provenance(doc, f"import from {clip.name}", dict(got))
                print(f"  {clip.name:<34} -> {v:<28} {dict(got)}")
            touched[v] = doc
        for v, doc in touched.items():
            n = fold_shadowed_legacy(doc)
            if n:
                print(f"  {v}: folded {n} legacy label(s) into the reviewed row for the "
                      f"same shot")
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

r"""Prepopulated shot-review workbook — the operator CORRECTS rather than labels from scratch.

Why this shape. Two labelling workflows were built and both were rejected by the operator for
the same underlying reason: a shot cannot be typed from a short clip around its contact.
*"You can't determine a shot at the point of contact without seeing it in context of where it
is coming from and where it is going."* A reel of snippets fails, and so does a
one-keypress-per-shot window — neither shows the rally.

So the review pairs two things:

  * the WHOLE video, annotated in order with a running clock, each shot numbered and labelled
    with the type we assigned (`python -m tools.annotate_full <clip>`)
  * this workbook, one row per shot, prepopulated with that same number, its time, who hit it
    and what we called it

The operator watches the video and fills `CORRECT_TYPE` only where we are wrong. Everything
else stays blank. That is a fraction of the work of labelling 124 shots from nothing, and the
numbers on screen and in the sheet are the same numbers.

It also fixes the gap that made the previous tool unusable: **there was no way to report a
shot we MISSED.** Blank rows at the bottom exist for exactly that — put in the time and the
type, leave `#` empty.

Reading it back is `--score`, which writes the corrections into the CSV form
`tools/score_shot_types.py` already reads, so nothing downstream needs changing.

Usage:
    python -m tools.shot_review_sheet data/pb_5_minute_outdoor-7
    python -m tools.shot_review_sheet data/pb_5_minute_outdoor-7 --score
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FONT = "Arial"
VALID = ["serve", "return", "drive", "dink", "drop", "lob", "not a shot"]
N_BLANK_ROWS = 30          # for shots we missed entirely
OUT_NAME = "shot_review.xlsx"
CSV_NAME = "labels_from_review.csv"
CSV_FIELDS = ["shot_no", "frame", "shot_id", "time", "hitter_role", "hitter_side",
              "true_type", "true_volley", "true_in", "notes"]


def clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def parse_clock(s) -> Optional[float]:
    """Seconds from whatever the operator actually typed.

    The first version accepted only "M:SS.s" and silently dropped everything else -- which
    threw away all 26 missed shots from a completed review, and reported "0 shots we MISSED"
    while doing it. The formats that appeared in practice:

        :18.37                      leading colon, seconds only
        :1:08                       leading colon, minutes and seconds
        1:03.82                     the documented form
        datetime.time(0, 3, 14, 500000)   Excel silently retyped the cell

    A parser for a human-filled sheet has to take what humans and Excel produce, and a value
    it cannot read must be REPORTED, never skipped -- losing operator input without saying so
    is the worst failure this tool can have.
    """
    import datetime as _dt
    if isinstance(s, (_dt.time, _dt.datetime)):
        return (s.hour * 3600 + s.minute * 60 + s.second + s.microsecond / 1e6)
    if isinstance(s, _dt.timedelta):
        return s.total_seconds()
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s or "").strip().strip(":")
    if not s:
        return None
    parts = s.split(":")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return vals[0] * 60 + vals[1]
    if len(vals) == 3:
        return vals[0] * 3600 + vals[1] * 60 + vals[2]
    return None


def rows_for(clip: Path) -> List[dict]:
    """One row per detected shot, numbered exactly as tools/annotate_full numbers them."""
    shots = json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"]
    roles: Dict[int, str] = {}
    rp = clip / "track_roles.json"
    if rp.exists():
        for role, info in json.loads(rp.read_text(encoding="utf-8"))["roles"].items():
            for tid in info.get("track_ids", []):
                roles[int(tid)] = role
    out = []
    for i, s in enumerate(sorted(shots, key=lambda x: x["frame"])):
        out.append({
            "n": i + 1,
            "shot_id": int(s["shot_id"]),
            "frame": int(s["frame"]),
            "t": float(s["t_sec"]),
            "hitter": roles.get(int(s.get("track_id", -1)), "?"),
            "side": s.get("hitter_side") or "?",
            "our_type": (s.get("shot_type") or "?"),
            "our_volley": "yes" if s.get("is_volley") else "no",
        })
    return out


def build(clip: Path, out_path: Path) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    rows = rows_for(clip)
    wb = Workbook()
    ws = wb.active
    ws.title = "shots"

    title = Font(name=FONT, size=12, bold=True)
    head = Font(name=FONT, size=10, bold=True, color="FFFFFF")
    body = Font(name=FONT, size=10)
    note = Font(name=FONT, size=10, italic=True, color="555555")
    fill_head = PatternFill("solid", fgColor="333333")
    fill_edit = PatternFill("solid", fgColor="FFF2CC")     # the columns to type in
    fill_ours = PatternFill("solid", fgColor="F2F2F2")     # what we currently say
    thin = Side(style="thin", color="BBBBBB")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws["A1"] = f"Shot review — {clip.name}"
    ws["A1"].font = title
    ws["A2"] = ("Watch _labeling/<clip>_annotated.mp4. Each shot is numbered on screen and "
                "labelled with the type we assigned.")
    ws["A3"] = ("Fill CORRECT_TYPE only where we are WRONG. A BLANK row is recorded as you "
                "CONFIRMING we are right — so if you stop part-way, say where you stopped.")
    ws["A6"] = ("Rows with a green ALREADY KNOWN value have been reviewed before — SKIP THEM "
                "unless that stored answer is wrong.")
    ws["A4"] = ("Missed a shot entirely? Use the blank rows at the bottom: time + "
                "CORRECT_TYPE, leave # empty.")
    ws["A7"] = ("Put 'y' in NOT_A_SHOT for a detection that is not a shot (a feed, a "
                "pick-up, an adjacent court), and 'y' in RALLY_END for the shot that ENDED "
                "the point (into the net, hit out, a winner).")
    ws["A5"] = ("Valid types: " + ", ".join(VALID)
                + "   —   a RESET is not a type: label it drop or dink, and we mark it a "
                  "reset automatically when it answers a drive.")
    for r in (2, 3, 4, 5, 6, 7):
        ws[f"A{r}"].font = note

    # What the truth store already knows about this video, so a shot reviewed once is not
    # put in front of the operator again. This is the whole point of the store: "the info I
    # provide on shots should be saved as truths and built upon so I don't have to keep
    # reviewing the same info."
    try:
        from tools.truth_store import known as _known, MATCH_TOL_S as _TOL
        store = _known(clip)
    except Exception:                                    # noqa: BLE001 - optional input
        store, _TOL = {"shots": [], "false_positives": []}, 1.0

    def already(t_sec):
        best, bd = None, _TOL + 1e-9
        for s in store.get("shots", []):
            d = abs(float(s.get("t_sec", -999)) - t_sec)
            if d < bd and s.get("type"):
                best, bd = s, d
        return best

    # NOT_A_SHOT and RALLY_END are their own columns now. The operator had to record both in
    # free text last time -- "not a shot. Between rallies. Opponent feeding ball to their
    # partner" -- because the sheet only asked for a corrected TYPE. Sixteen false positives
    # and sixteen rally ends were sitting in the notes, and a regex bug meant the first pass
    # read none of them. A column cannot be silently mis-parsed.
    # The operator's own rally windows, where they have described them. Their objection to
    # the first labelling tool was exactly this: "you can't determine a shot at the point of
    # contact without seeing it in context of where it is coming from and where it is going."
    # A rally number per row is that context, and it comes from what they already told us --
    # it is never our inference, so it cannot mislead them with our own error.
    rally_truth = sorted((store.get("rally_truth") or []),
                         key=lambda r: float(r["start_t_sec"]))

    def rally_of(t_sec):
        for i, r in enumerate(rally_truth, start=1):
            if float(r["start_t_sec"]) - 0.5 <= t_sec <= float(r["end_t_sec"]) + 0.5:
                return str(i)
        return "between points" if rally_truth else ""

    hr = 9 if not rally_truth else 11
    headers = ["#", "time", "your rally", "hitter", "side", "our_type", "our_volley",
               "ALREADY KNOWN", "NOT_A_SHOT", "RALLY_END", "CORRECT_TYPE",
               "CORRECT_VOLLEY", "notes"]
    if rally_truth:
        # The operator counted the shots in each rally. Showing their count against ours says
        # exactly which rally to hunt in for a shot we missed -- the alternative is watching
        # three minutes hoping to notice an absence.
        short = []
        for i, r in enumerate(rally_truth, start=1):
            theirs = r.get("n_shots")
            ours = sum(1 for x in rows
                       if float(r["start_t_sec"]) - 0.5 <= x["t"]
                       <= float(r["end_t_sec"]) + 0.5)
            if theirs and int(theirs) != ours:
                short.append(f"rally {i} (you counted {theirs}, we have {ours})")
        ws["A8"] = ("You already told us the shot count per rally. We disagree on: "
                    + ("; ".join(short) if short else "none — the counts all match")
                    + ".")
        ws["A8"].font = note
        ws["A9"] = ("The 'your rally' column is YOUR rally windows, not our guess. A row "
                    "marked 'between points' falls outside every rally you described — those "
                    "are the likeliest NOT_A_SHOT rows.")
        ws["A9"].font = note

    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=hr, column=c, value=h)
        cell.font = head
        cell.fill = fill_head
        cell.alignment = Alignment(horizontal="center")
        cell.border = box

    # one worked example, so the expected format is unambiguous
    ex = hr + 1
    for c, v in enumerate([" e.g. 12", "1:03.82", "3", "user", "near", "drive", "no",
                           "", "", "", "drop", "",
                           "was a soft third shot, not a drive"], start=1):
        cell = ws.cell(row=ex, column=c, value=v)
        cell.font = note
        cell.border = box

    fill_known = PatternFill("solid", fgColor="E2EFDA")     # already reviewed: skip it
    first = ex + 1
    n_known = 0
    for i, r in enumerate(rows):
        rr = first + i
        prev = already(r["t"])
        kn = ""
        if prev:
            n_known += 1
            kn = prev["type"] + ("  (agrees)" if prev["type"] == r["our_type"]
                                 else f"  (you said {prev['type']})")
        vals = [r["n"], clock(r["t"]), rally_of(r["t"]), r["hitter"], r["side"],
                r["our_type"], r["our_volley"], kn, "", "", "", "", ""]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=rr, column=c, value=v)
            cell.font = body
            cell.border = box
            if c in (6, 7):
                cell.fill = fill_ours
            if c == 3 and v == "between points":
                cell.font = note          # outside every rally they described: junk candidate
            if c == 8 and prev:
                cell.fill = fill_known
            if c in (9, 10, 11, 12, 13) and not prev:
                cell.fill = fill_edit
    last = first + len(rows) - 1

    blank_hdr = last + 2
    ws.cell(row=blank_hdr, column=1, value="SHOTS WE MISSED — add below (time + "
                                          "CORRECT_TYPE; leave # blank)").font = title
    for i in range(N_BLANK_ROWS):
        rr = blank_hdr + 1 + i
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=rr, column=c, value="")
            cell.font = body
            cell.border = box
            if c in (2, 9, 10, 11, 12, 13):
                cell.fill = fill_edit
    last_blank = blank_hdr + N_BLANK_ROWS

    # Dropdowns and widths BY HEADER, never by letter. The layout has gained a column three
    # times now (ALREADY KNOWN, then NOT_A_SHOT / RALLY_END, now the rally), and each time a
    # hard-coded letter put a control on the wrong column -- silently, because a dropdown on
    # the wrong column still looks like a working sheet.
    from openpyxl.utils import get_column_letter
    letter = {h: get_column_letter(i) for i, h in enumerate(headers, start=1)}
    for name, choices in (("CORRECT_TYPE", VALID), ("CORRECT_VOLLEY", ("yes", "no")),
                          ("NOT_A_SHOT", ("y",)), ("RALLY_END", ("y",))):
        d = DataValidation(type="list", formula1='"' + ",".join(choices) + '"',
                           allow_blank=True)
        ws.add_data_validation(d)
        d.add(f"{letter[name]}{first}:{letter[name]}{last_blank}")

    for h, w in (("#", 7), ("time", 10), ("your rally", 13), ("hitter", 10), ("side", 7),
                 ("our_type", 12), ("our_volley", 12), ("ALREADY KNOWN", 20),
                 ("NOT_A_SHOT", 12), ("RALLY_END", 11), ("CORRECT_TYPE", 15),
                 ("CORRECT_VOLLEY", 15), ("notes", 46)):
        if h in letter:
            ws.column_dimensions[letter[h]].width = w
    ws.freeze_panes = ws[f"A{first}"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path


def score(clip: Path, xlsx: Path) -> int:
    """Read the corrections back and write the CSV tools/score_shot_types.py already globs.

    A blank CORRECT_TYPE means "you were right", so the row is emitted with OUR type as the
    truth. That is the point of a correction sheet: silence is agreement, and agreement is
    still a label.
    """
    from openpyxl import load_workbook
    if not xlsx.exists():
        raise SystemExit(f"{xlsx} not found — build it first")
    ws = load_workbook(xlsx, data_only=True).active
    court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
    fps = float(court["video"]["fps"])

    hdr_row = next((r for r in range(1, 30)
                    if str(ws.cell(row=r, column=1).value or "").strip() == "#"), None)
    if hdr_row is None:
        raise SystemExit("could not find the header row in the sheet")
    # By header, never by position: the layout has gained columns twice, and a review read
    # from the wrong column is how 26 missed shots were lost.
    col = {str(ws.cell(row=hdr_row, column=i).value or "").strip(): i
           for i in range(1, ws.max_column + 1)}
    C_TYPE = col.get("CORRECT_TYPE", 8)
    C_VOL = col.get("CORRECT_VOLLEY", 9)
    C_NOTE = col.get("notes", 10)
    C_KNOWN = col.get("ALREADY KNOWN")
    C_NAS = col.get("NOT_A_SHOT")
    C_OURS = col.get("our_type", 5)
    C_HIT = col.get("hitter", 3)
    C_SIDE = col.get("side", 4)

    out, n_corr, n_agree, n_missed = [], 0, 0, 0
    unparsed: List[tuple] = []
    for r in range(hdr_row + 1, ws.max_row + 1):
        n = ws.cell(row=r, column=1).value
        tstr = ws.cell(row=r, column=2).value
        ours = str(ws.cell(row=r, column=C_OURS).value or "").strip().lower()
        known_prev = (str(ws.cell(row=r, column=C_KNOWN).value or "").strip().lower()
                      if C_KNOWN else "")
        corr = str(ws.cell(row=r, column=C_TYPE).value or "").strip().lower()
        vol = str(ws.cell(row=r, column=C_VOL).value or "").strip().lower()
        notes = str(ws.cell(row=r, column=C_NOTE).value or "").strip()
        if C_NAS and str(ws.cell(row=r, column=C_NAS).value or "").strip().lower().startswith("y"):
            corr = "not a shot"
        t = parse_clock(tstr)
        if t is None:
            if any(str(ws.cell(row=r, column=c).value or "").strip()
                   for c in (C_HIT, C_SIDE, C_OURS, C_TYPE, C_NOTE)):
                unparsed.append((r, tstr))
            continue
        if isinstance(n, str) and not str(n).strip().isdigit():
            continue                      # the worked example row
        true_type = corr or ours
        if not true_type:
            continue
        if n in (None, ""):
            n_missed += 1
            notes = (notes + " | reported as a MISSED shot").strip(" |")
        elif corr:
            n_corr += 1
        else:
            n_agree += 1
        out.append({"shot_no": n or "", "frame": int(round(t * fps)),
                    "shot_id": n or "", "time": clock(t),
                    "hitter_role": str(ws.cell(row=r, column=C_HIT).value or ""),
                    "hitter_side": str(ws.cell(row=r, column=C_SIDE).value or ""),
                    "true_type": true_type,
                    "true_volley": {"yes": "y", "no": "n"}.get(vol, ""),
                    "true_in": "", "notes": notes})

    dest = clip / "_labeling" / CSV_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"{clip.name}: {len(out)} labels written to {dest}")
    print(f"  {n_agree} confirmed as already correct")
    print(f"  {n_corr} corrected")
    print(f"  {n_missed} shots we MISSED entirely")
    if unparsed:
        print(f"\n  {len(unparsed)} row(s) had content but an UNREADABLE time -- "
              f"these were NOT counted:")
        for r, v in unparsed[:12]:
            print(f"    row {r}: time={v!r}")
        if len(unparsed) > 12:
            print(f"    ...and {len(unparsed) - 12} more")
    if n_corr + n_agree + n_missed:
        print(f"\n  now run:  python -m tools.regression --clip {clip}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--score", action="store_true",
                    help="read the filled-in sheet back and write the label CSV")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing sheet even if it has been filled in")
    a = ap.parse_args(argv)
    if not a.clip.is_dir():
        raise SystemExit(f"not a folder: {a.clip}")
    out = a.out or a.clip / "_labeling" / OUT_NAME
    if a.score:
        return score(a.clip, out)
    if not (a.clip / "classified.json").exists():
        raise SystemExit(f"{a.clip}/classified.json missing — analyse the clip first")
    # Never overwrite a sheet that has work in it. The operator's time is the scarce input
    # here, and rebuilding on top of a filled-in review would destroy hours of it silently.
    if out.exists() and not a.force:
        try:
            from openpyxl import load_workbook
            ws = load_workbook(out, data_only=True).active
            hdr = next((r for r in range(1, 20)
                        if str(ws.cell(row=r, column=1).value or "").strip() == "#"), None)
            filled = 0
            if hdr:
                cols = [c for c in range(1, ws.max_column + 1)
                        if "CORRECT" in str(ws.cell(row=hdr, column=c).value or "").upper()]
                cols += [c for c in range(1, ws.max_column + 1)
                         if str(ws.cell(row=hdr, column=c).value or "").strip() == "notes"]
                for r in range(hdr + 2, ws.max_row + 1):
                    if any(str(ws.cell(row=r, column=c).value or "").strip() for c in cols):
                        filled += 1
            if filled:
                msg = (f"{out} already has {filled} filled-in row(s). "
                       "Refusing to overwrite your review. Either import it first "
                       f"(python -m tools.truth_store --import-review {a.clip}), "
                       "or pass --force / --out <other.xlsx>.")
                raise SystemExit(msg)
        except SystemExit:
            raise
        except Exception:                                # noqa: BLE001
            pass
    p = build(a.clip, out)
    n = len(rows_for(a.clip))
    print(f"wrote {p}  ({n} shots prepopulated, {N_BLANK_ROWS} blank rows for missed ones)")
    try:
        from tools.truth_store import known as _k
        kn = sum(1 for s in _k(a.clip).get("shots", []) if s.get("type"))
        if kn:
            print(f"  {kn} shots already have a stored answer — those rows are marked green "
                  f"and can be skipped")
    except Exception:                                    # noqa: BLE001
        pass
    print(f"\nnext: python -m tools.annotate_full {a.clip}")
    print(f"      watch it, correct the sheet, then:")
    print(f"      python -m tools.shot_review_sheet {a.clip} --score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

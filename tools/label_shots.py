"""Operator shot-labelling: build a review REEL + a CSV, then score against it.

Why this exists. Every accuracy claim about shot TYPE currently rests on operator
COUNTS, not per-shot labels. That is why "user dinks 6 = 6, EXACT" turned out to be
coincidence: only 1 of those 6 came from the sound landing signal, the other 5 from
low-confidence fallbacks. A right total built from wrong decisions will not survive a
different clip. Per-shot truth is the only way to measure precision and recall
instead of inferring from totals.

Two modes:

  BUILD   python tools/label_shots.py --clip data/pb_5_minute_outdoor-2 --shots 1-30
          Writes <clip>/_labeling/reel.mp4 and <clip>/_labeling/labels.csv.
          The reel shows each shot as a short segment: the court projected, the ball
          with a trail, a contact flash on the strike frame, and a big SHOT #id.
          It deliberately does NOT show our predicted type -- seeing the guess biases
          the label, and an unbiased label is the whole point. (--show-prediction
          overrides this if you would rather correct than label from scratch.)

  SCORE   python tools/label_shots.py --clip data/pb_5_minute_outdoor-2 --score
          Reads the filled labels.csv and reports per-type precision / recall / F1
          and a confusion matrix against classified.json.

Filling it in: open labels.csv, watch reel.mp4, and put the true type in `true_type`
for every row. Blank rows are skipped, so you can stop part-way and still score what
you did. Optional `true_volley` (y/n) and `true_in` (y/n -- a good shot bounces
INSIDE the 22x20 half; anything outside is out) sharpen the volley and in/out checks.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.verify_identity import draw_court  # noqa: E402

VALID_TYPES = ["serve", "return", "drive", "dink", "drop", "lob", "reset", "other"]


def load(clip: Path, name: str):
    return json.load(open(clip / name, encoding="utf-8"))


def parse_range(spec: str, n: int):
    """'1-30' or '3,7,9' or 'all' -> a set of 1-based positions."""
    if not spec or spec == "all":
        return set(range(1, n + 1))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return {i for i in out if 1 <= i <= n}


def build(clip: Path, args):
    shots = sorted(load(clip, "shots.json")["shots"], key=lambda s: s["frame"])
    fps = float(load(clip, "shots.json").get("fps") or 60.0)
    court = load(clip, "court.json")
    roles = {int(t): i["role"] for t, i in
             load(clip, "track_roles.json")["track_roles"].items()}
    cls = {s["shot_id"]: s for s in load(clip, "classified.json")["shots"]}
    import pandas as pd
    ball = pd.read_parquet(clip / "ball.parquet").set_index("frame_idx")

    picks = parse_range(args.shots, len(shots))
    sel = [(i, s) for i, s in enumerate(shots, 1) if i in picks]
    if not sel:
        raise SystemExit("no shots selected")

    pre = int(round(args.pre * fps))
    post = int(round(args.post * fps))
    # frame -> list of (position, shot) so one sequential pass can serve every segment
    need = defaultdict(list)
    for pos, s in sel:
        for f in range(max(0, s["frame"] - pre), s["frame"] + post + 1):
            need[f].append((pos, s))
    last = max(need) if need else 0

    out_dir = clip / "_labeling"
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(clip / "video.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip / 'video.mp4'}")
    W = args.width
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 3840
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 2160
    H = int(src_h * W / src_w)
    raw_path = out_dir / "_reel_raw.mp4"
    vw = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Segments are written in shot order, so buffer per shot then emit.
    buf = defaultdict(list)
    trail = deque(maxlen=12)
    print(f"decoding to frame {last} for {len(sel)} shots "
          f"({args.pre:g}s before / {args.post:g}s after each)...")
    f = 0
    while f <= last:
        if f not in need:
            if not cap.grab():          # skip cheaply -- no decode
                break
            f += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        # envelope=False: the player run-out envelope must not be shown when the
        # question is where the BALL bounced (operator: ball stays within 22x20).
        draw_court(img, court, envelope=False)
        bx = by = None
        if f in ball.index:
            r = ball.loc[f]
            if bool(r["visible"]) and not np.isnan(r["pixel_x"]):
                bx, by = int(r["pixel_x"]), int(r["pixel_y"])
                trail.append((bx, by))
        for i, (tx, ty) in enumerate(trail):
            cv2.circle(img, (tx, ty), 5, (0, 140, 255), -1)
        if bx is not None:
            cv2.circle(img, (bx, by), 20, (0, 0, 255), 4)
        for pos, s in need[f]:
            d = f - s["frame"]
            if abs(d) <= 2:              # contact flash
                cv2.circle(img, tuple(int(v) for v in s["impact_pixel_xy"]),
                           60, (255, 255, 255), 6)
            small = cv2.resize(img, (W, H))
            hdr = (f"SHOT #{pos}   {int(s['frame'] / fps // 60)}:"
                   f"{s['frame'] / fps % 60:04.1f}   "
                   f"{roles.get(s['track_id'], '?')} / {s.get('hitter_side', '?')}"
                   f"   {'CONTACT' if abs(d) <= 2 else ''}")
            if args.show_prediction:
                c = cls.get(s["shot_id"], {})
                hdr += f"   [pred {c.get('shot_type')}]"
            bar = np.zeros((46, W, 3), np.uint8)
            cv2.putText(bar, hdr, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2)
            buf[pos].append(np.vstack([bar, small[:H - 46]]))
        f += 1
    cap.release()

    for pos, _ in sel:
        for frame in buf.get(pos, []):
            vw.write(frame)
    vw.release()
    # Re-encode to H.264 so the reel plays in any viewer and is a fraction of the
    # size (mp4v wrote 68 MB for 66 s). ffmpeg comes from the imageio-ffmpeg wheel,
    # same as tools/compress_video.py; if it is missing we keep the raw file.
    final = out_dir / "reel.mp4"
    try:
        import subprocess
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw_path),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)],
                       check=True, capture_output=True)
        raw_path.unlink()
    except Exception as e:
        raw_path.replace(final)
        print(f"  (H.264 re-encode unavailable: {e}; kept mp4v)")

    csv_path = out_dir / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        # `frame` is the STABLE key. shot_id is assigned by enumeration, so any
        # change to detection renumbers every shot and silently invalidates labels
        # keyed on it -- that bit us on 2026-08-03. Always match on frame.
        w.writerow(["shot_no", "frame", "shot_id", "time", "hitter_role",
                    "hitter_side", "true_type", "true_volley", "true_in", "notes"])
        for pos, s in sel:
            t = s["frame"] / fps
            w.writerow([pos, s["frame"], s["shot_id"], f"{int(t // 60)}:{t % 60:04.1f}",
                        roles.get(s["track_id"], "?"), s.get("hitter_side", ""),
                        "", "", "", ""])
    print(f"\nwrote {out_dir / 'reel.mp4'}  ({len(sel)} shots)")
    print(f"wrote {csv_path}")
    print(f"\nvalid true_type values: {', '.join(VALID_TYPES)}")
    print("true_volley: y/n (taken out of the air).  true_in: y/n (bounced inside "
          "the 22x20 half; outside = out).")
    print("Fill true_type for every row you judge; blanks are skipped when scoring.")


def score(clip: Path):
    csv_path = clip / "_labeling" / "labels.csv"
    if not csv_path.exists():
        raise SystemExit(f"no labels at {csv_path} — run the build mode first")
    cls_shots = load(clip, "classified.json")["shots"]
    shots = {s["shot_id"]: s for s in load(clip, "shots.json")["shots"]}
    by_frame = {}
    for c in cls_shots:
        fr = shots.get(c["shot_id"], {}).get("frame")
        if fr is not None:
            by_frame[int(fr)] = c

    def lookup(row):
        """Find our classification for a labelled row. Match on FRAME (stable);
        fall back to shot_id only for label files written before the frame column
        existed -- shot_id is re-enumerated whenever detection changes."""
        fr = (row.get("frame") or "").strip()
        if fr:
            f = int(float(fr))
            best = min(by_frame, key=lambda k: abs(k - f), default=None)
            return by_frame.get(best, {}) if best is not None and abs(best - f) <= 3 else {}
        t = (row.get("time") or "").strip()
        if t and ":" in t:
            m, sec = t.split(":")
            f = int(round((int(m) * 60 + float(sec)) * 60.0))
            best = min(by_frame, key=lambda k: abs(k - f), default=None)
            return by_frame.get(best, {}) if best is not None and abs(best - f) <= 4 else {}
        return {}

    all_rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    # A label of "between pts/shots/points" (or a blank with a note) means the
    # detection is NOT A SHOT at all -- a feed, a ball rolled back, ball handling.
    # These are FALSE POSITIVES of shot detection and must be scored separately:
    # folding them into the type confusion matrix would hide the biggest error class.
    between = {"between pts", "between shots", "between points", "between point",
               "between", "not a shot", "none"}
    junk, rows = [], []
    for r in all_rows:
        t = (r.get("true_type") or "").strip().lower()
        if t in between or (not t and (r.get("notes") or "").strip()):
            junk.append(r)
        elif t:
            rows.append(r)
    n_lab = len(rows) + len(junk)
    if n_lab:
        print(f"\nSHOT DETECTION — is a detected 'shot' a real shot?  ({n_lab} labelled)")
        print(f"  real shots                          : {len(rows)}")
        print(f"  NOT a shot (feed / roll / handling) : {len(junk)}"
              f"   -> {len(junk) / n_lab:.0%} of detections are FALSE POSITIVES")
        if junk:
            c = defaultdict(int)
            for r in junk:
                c[(lookup(r).get("shot_type") or "NOT DETECTED")] += 1
            print(f"  the pipeline typed them as: "
                  f"{', '.join(f'{k} x{v}' for k, v in sorted(c.items()))}")
    if not rows:
        raise SystemExit("labels.csv has no real-shot true_type values yet")

    conf = defaultdict(int)
    vol_ok = vol_n = 0
    for r in rows:
        truth = r["true_type"].strip().lower()
        got = lookup(r)
        pred = (got.get("shot_type") or "NOT-DETECTED").lower()
        conf[(truth, pred)] += 1
        tv = (r.get("true_volley") or "").strip().lower()
        if tv in ("y", "n"):
            vol_n += 1
            if (tv == "y") == bool(got.get("is_volley")):
                vol_ok += 1

    types = sorted({t for t, _ in conf} | {p for _, p in conf})
    print(f"\nSHOT-TYPE ACCURACY — {len(rows)} labelled shots\n")
    print("  truth \\ pred   " + "".join(f"{p[:7]:>9s}" for p in types))
    for t in types:
        if not any(conf[(t, p)] for p in types):
            continue
        print(f"  {t:<14s} " + "".join(f"{conf[(t, p)] or '':>9}" for p in types))

    print(f"\n  {'type':<10s} {'n':>4s} {'precision':>10s} {'recall':>8s} {'F1':>7s}")
    total_ok = 0
    for t in types:
        tp = conf[(t, t)]
        fp = sum(conf[(o, t)] for o in types if o != t)
        fn = sum(conf[(t, o)] for o in types if o != t)
        n = tp + fn
        if not n and not fp:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n if n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        total_ok += tp
        print(f"  {t:<10s} {n:>4d} {prec:>10.0%} {rec:>8.0%} {f1:>7.2f}")
    print(f"\n  overall type accuracy: {total_ok}/{len(rows)} = {total_ok / len(rows):.0%}")
    if vol_n:
        print(f"  is_volley accuracy   : {vol_ok}/{vol_n} = {vol_ok / vol_n:.0%}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--shots", default="1-30",
                    help="'1-30', '3,7,9', or 'all' (1-based, in frame order)")
    ap.add_argument("--pre", type=float, default=0.8, help="seconds before contact")
    ap.add_argument("--post", type=float, default=1.4,
                    help="seconds after contact (long enough to see the LANDING, "
                         "which is what decides type)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--show-prediction", action="store_true",
                    help="overlay our predicted type (biases the label — off by default)")
    ap.add_argument("--score", action="store_true", help="score a filled labels.csv")
    args = ap.parse_args()
    clip = Path(args.clip)
    score(clip) if args.score else build(clip, args)


if __name__ == "__main__":
    main()

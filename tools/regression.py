r"""Every scorer, every clip, one table — and a baseline to compare against.

Why this exists. Each accuracy claim on this project has its own scorer, and checking a change
meant running a dozen commands and reading the numbers by eye. That is how two things went
wrong in one session: `score_shot_types` silently read a clip-local label file containing no
drops, so shot typing read 58% when it was 31% and a change was rejected on a number that was
not measuring what it claimed; and a filter recorded as "solved" went inert for two weeks
because nothing re-ran the measurement that justified it.

So: one command, every scorer, every clip that has truth for it, printed as one table and
diffed against a saved baseline. A refactor that should not move a number can be shown not to
have moved one — which is exactly what the Stage 4 rewrite needed and had to be assembled by
hand instead.

    python -m tools.regression                 # score everything, diff against the baseline
    python -m tools.regression --save          # record the current numbers as the baseline
    python -m tools.regression --rerun         # re-run Stages 5-11 first, then score
    python -m tools.regression --clip data/x   # just one clip

Exit code is 1 if anything moved against the baseline, so it can gate a commit.

The baseline lives in docs/REGRESSION_BASELINE.json and is meant to be committed: a number
changing is a reviewable event, and the diff in that file says what changed and by how much.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import score_rally_shots, score_serves, score_shot_types, score_shots

BASELINE = Path("docs/REGRESSION_BASELINE.json")

# Clips carrying operator truth of some kind. A clip with none is still worth scoring for the
# structural checks (the shots = volleys + bounces identity), so it is listed too.
CLIPS = [
    "data/pb_5_minute_outdoor-7",
    "data/pb_3_min_indoor_1_court_b",
    "data/pb_3_min_indoor_1_court_c",
    "data/pb_5_min_indoor_1_court_a",
]

# Stages 5-11: everything below the GPU vision pass. ~24 s for a 5-minute clip, so --rerun is
# cheap now; before the Stage 4 work it was ~30 minutes and this would not have been usable.
POST_STAGES = ["detect_shots", "detect_bounces", "ball_trajectory", "classify_shots",
               "segment_rallies", "compute_metrics", "rate", "plan_improvement"]

# Operator counts for the source video "PB 5 minute outdoor" (docs/ACCURACY_LEDGER.md,
# corrected 2026-08-01). Keyed by source video, not by clip folder, because several analysed
# folders share one video and the truth belongs to the video.
VIDEO_TRUTH = {
    "PB 5 minute outdoor.mp4": {"shots_truth": 98, "volleys_truth": 17, "bounces_truth": 81,
                                "serves_truth": 14, "dinks_truth": 18},
}


def source_video(clip: Path) -> Optional[str]:
    for name in ("ball.meta.json", "session.json"):
        d = _load(clip, name)
        if d and d.get("video_path"):
            n = Path(str(d["video_path"])).name
            if n not in ("video.mp4", ""):
                return n
    return None


def _load(clip: Path, name: str) -> Optional[dict]:
    p = clip / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def rerun_post(clip: Path) -> None:
    for s in POST_STAGES:
        r = subprocess.run([sys.executable, "-m", f"stages.{s}.{s}", str(clip), "--force"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{s} failed on {clip.name}:\n{(r.stderr or r.stdout)[-600:]}")


def measure(clip: Path) -> Dict[str, object]:
    """Every number we can put a truth against, plus the structural checks.

    Keys are flat and stable so the baseline diff is readable. A key absent from a clip means
    that clip has no truth for it — not that it scored zero.
    """
    m: Dict[str, object] = {}
    cls = _load(clip, "classified.json")
    if not cls:
        return m
    shots = cls["shots"]
    m["shots"] = len(shots)
    m["volleys"] = sum(1 for s in shots if s.get("is_volley"))
    m["dinks"] = sum(1 for s in shots if s.get("shot_type") == "dink")
    b = _load(clip, "bounces.json")
    m["bounces"] = len(b["bounces"]) if b else None
    # The identity every clip must satisfy: a shot is volleyed out of the air or lands once.
    # The single best structural check we have, and it needs no operator truth.
    if m["bounces"] is not None:
        m["identity_gap"] = m["shots"] - (m["volleys"] + m["bounces"])
    ral = _load(clip, "rallies.json")
    m["rallies"] = len(ral["rallies"]) if ral else None
    rat = _load(clip, "rating.json")
    if rat:
        m["rating"] = round(float(rat["rating"]["estimate"]), 2)
        m["rating_confidence"] = round(float(rat["rating"]["confidence"]), 2)

    # --- operator counts for the whole video ---------------------------------
    # The identity shots = volleys + bounces is the best self-check here, but it only says
    # the terms are CONSISTENT. These say whether they are RIGHT: volleys read 32-42% of
    # shots across four venues against an operator truth of 17%.
    vt = VIDEO_TRUTH.get(source_video(clip) or "")
    if vt:
        m.update(vt)

    # --- per-shot review (false positives / missed / wrong player) -------------
    if (clip / "shot_review.json").exists():
        sc = score_shots.score(shots, _load(clip, "shot_review.json"))
        m["fp_emitted"] = sc["false_positives"]
        m["fp_labelled"] = sc["n_fp_labelled"]
        m["missed_recovered"] = sc["missed_recovered"]
        m["missed_labelled"] = sc["n_missed_labelled"]
        m["wrong_player"] = sc.get("wrong_player")
        m["real_shots_kept"] = len(shots) - sc["false_positives"]

    # --- operator point windows (in-rally vs between-point) --------------------
    if (clip / "truth.json").exists():
        try:
            sh, points = score_rally_shots.load(clip)
            rs = score_rally_shots.score(sh, points)
            m["in_rally_shots"] = rs["n_in_rallies"]
            m["in_rally_truth"] = rs["n_truth"]
            m["between_point_shots"] = len(rs["outside"])
        except (KeyError, json.JSONDecodeError):
            pass

    # --- confirmed missed shots (court B's timestamped review) ----------------
    mr = _load(clip, "missed_review.json")
    if mr:
        found, _ = score_shots.claim([d["t_sec"] for d in mr["missed"]], shots,
                                     score_shots.TOL_S)
        m["confirmed_missed_recovered"] = len(found)
        m["confirmed_missed_total"] = len(mr["missed"])

    # --- serves -------------------------------------------------------------
    truth_sv = score_serves.truth_serves(clip)
    if truth_sv:
        got = score_serves.detected_serves(clip)
        mt = score_serves.match(truth_sv, got)
        m["serve_recall"] = round(len(mt["hits"]) / len(truth_sv), 2)
        m["serve_precision"] = round(len(mt["hits"]) / len(got), 2) if got else None
        strikes = score_serves.strike_times(clip)
        if strikes:
            sm = score_serves.match(strikes, got)
            errs = sorted(abs(g - t) for t, g, _ in sm["hits"])
            if errs:
                m["serve_timing_median_s"] = round(errs[len(errs) // 2], 2)

    # --- what the operator has told us about this video, cumulatively ---------
    try:
        from tools import truth_store as _ts
        st = _ts.known(clip)
        kn_missed = [s for s in st["shots"] if s.get("detected") is False]
        if st["shots"]:
            m["truth_shots_known"] = len(st["shots"])
            m["truth_missed_known"] = len(kn_missed)
            m["truth_fp_known"] = len(st["false_positives"])
            found, _ = score_shots.claim([s["t_sec"] for s in kn_missed], shots,
                                         score_shots.TOL_S)
            m["truth_missed_recovered"] = len(found)
            # Volleys, scored ONLY against shots the operator actually judged. The rest carry
            # a value inherited by "blank means agree", and scoring against those would be
            # scoring against ourselves -- it reads as agreement however wrong we are.
            ex = [s for s in st["shots"]
                  if s.get("volley_explicit") and s.get("volley") is not None]
            if ex:
                idx, _ = score_shots.claim([s["t_sec"] for s in ex], shots,
                                           score_shots.TOL_S)
                miss = inv = okv = 0
                for i in idx:
                    near = min(ex, key=lambda s: abs(s["t_sec"] - shots[i]["t_sec"]))
                    got = bool(shots[i].get("is_volley"))
                    want = bool(near["volley"])
                    okv += want == got
                    miss += want and not got
                    inv += (not want) and got
                m["volley_judged"] = len(idx)
                m["volley_correct"] = okv
                m["volley_missed"] = miss
                m["volley_invented"] = inv
    except Exception:                                    # noqa: BLE001
        pass

    # --- shot types ---------------------------------------------------------
    # The accumulating truth store is the one home for the operator's input; the scattered
    # labels*.csv files are what it was built from, so reading both would double-count.
    fps_ = float((cls.get("fps") or 60.0))
    labels = score_shot_types.load_from_truth_store(clip)
    for l in labels:
        l["frame"] = int(round(l["t_sec"] * fps_))
    if not labels:
        labels = score_shot_types.load_labels(clip, fps_)
        if (score_shot_types.source_video(clip)
                == score_shot_types.source_video(score_shot_types.SHARED_LABELS)):
            shared = score_shot_types.load_labels(score_shot_types.SHARED_LABELS, fps_)
            if len(shared) > len(labels):
                labels = shared
    if labels:
        st = score_shot_types.score(clip, labels)
        m["shot_type_correct"] = st["hit"]
        m["shot_type_labelled"] = st["n_real"]

    return m


def render(results: Dict[str, Dict], base: Dict[str, Dict]) -> int:
    """One table per clip, with the baseline delta beside anything that moved."""
    keys_order = ["shots", "shots_truth", "volleys", "volleys_truth",
                  "bounces", "bounces_truth", "identity_gap", "rallies",
                  "fp_emitted", "fp_labelled", "real_shots_kept",
                  "missed_recovered", "missed_labelled", "wrong_player",
                  "confirmed_missed_recovered", "confirmed_missed_total",
                  "in_rally_shots", "in_rally_truth", "between_point_shots",
                  "serve_recall", "serve_precision", "serve_timing_median_s",
                  "shot_type_correct", "shot_type_labelled",
                  "truth_shots_known", "truth_missed_known", "truth_missed_recovered",
                  "truth_fp_known", "volley_judged", "volley_correct",
                  "volley_missed", "volley_invented",
                  "dinks", "rating", "rating_confidence"]
    n_moved = 0
    for name, m in results.items():
        b = base.get(name, {})
        print(f"\n=== {name} ===")
        if not m:
            print("  (no classified.json — not analysed)")
            continue
        for k in keys_order:
            if k not in m or m[k] is None:
                continue
            old = b.get(k)
            if old is None or old == m[k]:
                flag = ""
            else:
                try:
                    d = m[k] - old
                    flag = f"   <-- was {old} ({d:+g})"
                except TypeError:
                    flag = f"   <-- was {old}"
                n_moved += 1
            print(f"  {k:<28}{str(m[k]):>10}{flag}")
        extra = [k for k in m if k not in keys_order]
        for k in sorted(extra):
            print(f"  {k:<28}{str(m[k]):>10}")
    return n_moved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true",
                    help="write the current numbers as the new baseline")
    ap.add_argument("--rerun", action="store_true",
                    help="re-run Stages 5-11 on each clip before scoring")
    ap.add_argument("--clip", action="append", default=None,
                    help="score only this clip (repeatable)")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    a = ap.parse_args(argv)

    clips = [Path(c) for c in (a.clip or CLIPS)]
    missing = [c for c in clips if not c.is_dir()]
    if missing:
        print("skipping absent clips: " + ", ".join(c.name for c in missing))
    clips = [c for c in clips if c.is_dir()]
    if not clips:
        raise SystemExit("no clips to score")

    base = {}
    if a.baseline.exists():
        base = json.loads(a.baseline.read_text(encoding="utf-8")).get("clips", {})

    results = {}
    for c in clips:
        if a.rerun:
            print(f"re-running Stages 5-11 on {c.name}...", flush=True)
            rerun_post(c)
        results[c.name] = measure(c)

    moved = render(results, base)

    if a.save:
        a.baseline.parent.mkdir(parents=True, exist_ok=True)
        merged = dict(base)
        merged.update(results)
        a.baseline.write_text(
            json.dumps({"note": "Written by tools/regression.py --save. A change here is a "
                                "reviewable event: the diff says which number moved and by "
                                "how much.",
                        "clips": merged}, indent=1) + "\n", encoding="utf-8")
        print(f"\nbaseline written: {a.baseline}")
        return 0

    print()
    if not base:
        print("no baseline yet — run with --save once the numbers are known good")
        return 0
    if moved:
        print(f"{moved} number(s) MOVED against the baseline (see the <-- markers)")
        return 1
    print("no change against the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

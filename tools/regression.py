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
#
# These are the stages app/pipeline.py runs, IN ITS ORDER AND WITH ITS
# ARGUMENTS. This had drifted from the app: the point-ends stage was missing and
# segment_rallies ran without --use-rally-ends, so --rerun rebuilt a pipeline the app never
# runs. With unmodified code that alone moved 16 baseline numbers -- rally_end_within_2s
# halving on every clip, junk_in_rallies up 10 on one -- which makes --rerun evidence about
# a code change indistinguishable from evidence about the harness.
POST_STAGES = [
    # TWO passes of shots+bounces, mirroring app/pipeline.py -- see the comment there. The
    # harness omits build_ball_3d between them (it decodes the video, which is 98% of the
    # cost); everything else has to match, or --rerun measures a pipeline nobody ships.
    ("stages.detect_shots.detect_shots", ["--force", "--no-bounces"]),
    ("stages.detect_bounces.detect_bounces", ["--force"]),
    ("stages.detect_shots.detect_shots", ["--force"]),
    ("stages.detect_bounces.detect_bounces", ["--force"]),
    ("stages.ball_trajectory.ball_trajectory", ["--force"]),
    ("stages.classify_shots.classify_shots", ["--force"]),
    ("tools.detect_rally_ends", ["--force"]),
    ("stages.segment_rallies.segment_rallies", ["--force", "--use-rally-ends"]),
    ("stages.compute_metrics.compute_metrics", ["--force"]),
    ("stages.rate.rate", ["--force"]),
    ("stages.plan_improvement.plan_improvement", ["--force"]),
]

# Operator counts for the source video "PB 5 minute outdoor" (docs/ACCURACY_LEDGER.md,
# corrected 2026-08-01). Keyed by source video, not by clip folder, because several analysed
# folders share one video and the truth belongs to the video.
VIDEO_TRUTH = {
    # volleys_truth 17 and dinks_truth 18 are the 2026-07-22 acceptance figures. BOTH are
    # superseded per-clip by the operator's shot-by-shot review where one exists (see
    # below); they remain only as the fallback for a clip he has not reviewed.
    "PB 5 minute outdoor.mp4": {"shots_truth": 98, "volleys_truth": 17, "bounces_truth": 81,
                                # dinks_truth is the 2026-07-22 acceptance figure for
                                # pb_5_minute_outdoor-2. The operator's own shot-by-shot
                                # review of THIS clip counts 32, so the truth store wins
                                # below; 18 is kept only as the fallback for a clip with no
                                # review. Reading 34-vs-18 as a threefold over-count -- which
                                # I did -- is what a stale truth costs.
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
    for module, args in POST_STAGES:
        r = subprocess.run([sys.executable, "-m", module, str(clip), *args],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(
                f"{module} failed on {clip.name}:\n{(r.stderr or r.stdout)[-600:]}")


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

    # ...but prefer the TRUTH STORE when it has adjudicated this clip. shot_review.json is
    # a per-clip snapshot that stops being written once the store exists: outdoor-7's is
    # dated 2026-08-18 and lists 34 false positives against the store's 54, so
    # real_shots_kept = len(shots) - 34 counted the removal of junk the snapshot had never
    # heard of as the loss of real play -- and it did, on a filter that provably removed
    # none. Third stale-truth trap in this table after dinks_truth and volleys_truth.
    try:
        from tools.shot_precision_score import score_clip as _prec
        _rows = _prec(clip)["rows"]
        if any(r["kind"] != "unexplained" for r in _rows):
            m["real_shots_kept"] = sum(1 for r in _rows if r["kind"] == "real")
            m["fp_emitted"] = sum(1 for r in _rows if r["kind"] == "known junk")
    except (OSError, ValueError, KeyError, StopIteration):
        pass

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

    # --- who served, against the operator's per-rally truth -------------------
    # 20 rallies of server truth across the two indoor courts, previously measured by
    # nothing. Shot-to-player attribution is a known weak point and this is the one axis
    # where the operator had already written the answer down.
    try:
        from tools import truth_store as _ts3
        rt = _ts3.known(clip).get("rally_truth") or []
        ra3 = _load(clip, "rallies.json")
        roles = _load(clip, "track_roles.json")
        if rt and ra3 and roles:
            by_track = {}
            for role, info in (roles.get("roles") or {}).items():
                for tid in (info.get("track_ids") or []):
                    by_track[int(tid)] = _ts3.norm_hitter(role)
            ok = judged = side_ok = side_judged = 0
            for r in rt:
                if not r.get("server"):
                    continue
                a, b = float(r["start_t_sec"]), float(r["end_t_sec"])
                # our rally whose span overlaps the operator's most
                best, bov = None, 0.0
                for our in ra3["rallies"]:
                    ov = min(b, float(our["end_t_sec"])) - max(a, float(our["start_t_sec"]))
                    if ov > bov:
                        best, bov = our, ov
                if best is None:
                    continue
                got = by_track.get(int(best.get("server_track_id", -1)))
                if got is None:
                    continue
                judged += 1
                ok += int(got == r["server"])
                # The serving SIDE, scored separately. It is what the ball-span correction
                # actually fixes: knowing the serve came from the near end does not say
                # whether it was the user or their partner, so the two must not be reported
                # as one number.
                want = "far" if r["server"] == "opponent" else "near"
                mine = best.get("server_side") or (
                    "far" if got == "opponent" else "near")
                side_judged += 1
                side_ok += int(mine == want)
            if judged:
                m["server_correct"] = ok
                m["server_judged"] = judged
            if side_judged:
                m["server_side_correct"] = side_ok
                m["server_side_judged"] = side_judged
    except (KeyError, OSError, ValueError, TypeError):
        pass

    # --- serves -------------------------------------------------------------
    # CAUTION: serve_recall/serve_precision here match detections against truth.json's
    # start_t_sec -- the RALLY START, not the strike -- with a 3.0s tolerance. That is loose
    # enough for a FALSE serve to satisfy a real rally, so removing junk can LOWER the
    # recall. Measured on pb_5_minute_outdoor-7: 2 of 12 hits were satisfied by a false
    # serve (the opponent's return at 1:35.50 credited to the rally starting 1:33.00, and
    # 5:03.17 to 5:01.00 -- both cases where the operator's serve was missed and the return
    # was accepted in its place). The formation tie-break removed two such false serves and
    # this number FELL 0.93 -> 0.86 while the operator-scored result rose 10/14 -> 11/14.
    # Use `python -m tools.serve_score` to judge a serve change; it matches the operator's
    # labelled serves side-constrained, so junk cannot satisfy it.
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
            # The operator's shot-by-shot review supersedes the 2026-07-22 acceptance
            # counts wherever it covers the same video. Those constants describe a
            # DIFFERENT analysed folder of it, and reading 34 dinks against a stale 18
            # looks like a threefold over-count when the review says 32.
            # ...but ONLY when the review covers the whole clip. Court B's typed shots are
            # every one of them a shot we MISSED (it came from a missed-shot review), so its
            # "2 dinks" is two dinks among the misses, not the clip's total -- and printing
            # that beside our 20 invents a tenfold error out of nothing.
            typed = [s for s in st["shots"] if s.get("type") and not s.get("not_a_shot")]
            n_missed = sum(1 for s in typed if s.get("detected") is False)
            whole_clip = typed and n_missed < len(typed) / 2
            if whole_clip:
                for field, ty in (("dinks_truth", "dink"), ("serves_truth", "serve")):
                    m[field] = sum(1 for s in typed if s.get("type") == ty)
                # VOLLEYS the same way, and for the same reason dinks are done this way.
                # The table carried volleys_truth = 17 from the 2026-07-22 acceptance run
                # while the operator's own shot-by-shot review of this video judges every
                # one of its 103 live shots and counts 23. Scoring a volley change against
                # 17 made a move toward his count look like an over-count away from it.
                judged = [s for s in typed if s.get("volley") is not None]
                if judged:
                    m["volleys_truth"] = sum(1 for s in judged if s.get("volley"))
            # What the rally gate is actually for: keeping known junk OUT of the rally
            # stream while leaving real play in. Neither `fp_emitted` nor `in_rally_shots`
            # could see it -- the first reads classified.json, which the gate never
            # rewrites, and the second needs a `points` truth file only two clips have. The
            # gate's whole trade was invisible in this table while it was being tuned.
            ra = _load(clip, "rallies.json")
            if ra and st["false_positives"]:
                spans = [(float(r["start_t_sec"]), float(r["end_t_sec"]))
                         for r in ra["rallies"]]
                def _inside(x):
                    return any(a - 0.2 <= x <= b + 0.2 for a, b in spans)
                m["junk_in_rallies"] = sum(
                    1 for f in st["false_positives"] if _inside(float(f["t_sec"])))
                # Only shots WE ACTUALLY EMITTED can be "left outside a rally" -- an
                # operator label with no detection near it is a missed shot, a different
                # problem with its own metric. Counting those made this read 6 when 2 was
                # the honest answer, and 4 of the 6 turned out to be stale labels from an
                # older file that the latest review contradicts.
                real_t = [float(s["t_sec"]) for s in st["shots"]
                          if not s.get("not_a_shot") and s.get("detected") is not False]
                # claim() returns (claimed SHOT indices, unmatched TRUTH times) -- the
                # second is the one that says which operator times we never emitted.
                _, unmatched = score_shots.claim(real_t, shots, score_shots.TOL_S)
                never = {round(x, 3) for x in unmatched}
                m["real_outside_rallies"] = sum(
                    1 for x in real_t if round(x, 3) not in never and not _inside(x))
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

    # --- rally ends, and what gating on them would be worth -------------------
    # The operator: "if we can accurately know when a rally ends, every shot outside the
    # serve to rally can be ignored." Measured on their boundaries that trade is 30 junk for
    # 2 real shots -- decisive. So rally-end accuracy is now a first-class number, not a
    # detail: it is the only thing between us and that.
    try:
        from tools import truth_store as _ts2
        st2 = _ts2.known(clip)
        t_ends = sorted(e["t_sec"] for e in st2.get("rally_ends") or [])
        if t_ends and ral:
            ours = sorted(float(r["start_t_sec"]) + float(r.get("duration_sec") or 0)
                          for r in ral["rallies"])
            errs = sorted(abs(min(ours, key=lambda o: abs(o - x)) - x) for x in t_ends)
            m["rally_end_truth"] = len(t_ends)
            m["rally_end_within_2s"] = sum(1 for e in errs if e <= 2.0)
            m["rally_end_median_err_s"] = round(errs[len(errs) // 2], 2)
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
        # How much of that sample is shots we had MISSED. It matters: court B's typed shots
        # are ALL shots we failed to detect -- the hardest cases by construction -- so its
        # rate is not comparable to a clip reviewed shot by shot, and reading the two side by
        # side invites exactly the wrong conclusion. Made visible rather than explained in a
        # comment nobody reads at the moment of comparison.
        try:
            from tools import truth_store as _ts2
            typed = [s for s in _ts2.known(clip)["shots"]
                     if s.get("type") and not s.get("not_a_shot")]
            if typed:
                miss = sum(1 for s in typed if s.get("detected") is False)
                m["shot_type_sample_was_missed"] = miss
        except (KeyError, OSError):
            pass

    return m


def render(results: Dict[str, Dict], base: Dict[str, Dict]) -> int:
    """One table per clip, with the baseline delta beside anything that moved."""
    keys_order = ["shots", "shots_truth", "volleys", "volleys_truth",
                  "bounces", "bounces_truth", "identity_gap", "rallies",
                  "fp_emitted", "fp_labelled", "real_shots_kept",
                  "missed_recovered", "missed_labelled", "wrong_player",
                  "confirmed_missed_recovered", "confirmed_missed_total",
                  "in_rally_shots", "in_rally_truth", "between_point_shots",
                  "junk_in_rallies", "real_outside_rallies",
                  "server_correct", "server_judged",
                  "server_side_correct", "server_side_judged",
                  "serve_recall", "serve_precision", "serve_timing_median_s",
                  "shot_type_correct", "shot_type_labelled",
                  "shot_type_sample_was_missed",
                  "truth_shots_known", "truth_missed_known", "truth_missed_recovered",
                  "truth_fp_known", "volley_judged", "volley_correct",
                  "volley_missed", "volley_invented",
                  "rally_end_truth", "rally_end_within_2s", "rally_end_median_err_s",
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

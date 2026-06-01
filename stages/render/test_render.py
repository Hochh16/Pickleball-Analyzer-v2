"""Stage 11 — Smoke test.

Rendering can't be graded pixel-exactly, so the test gates on runs-without-crash
+ output well-formedness + timeline reconciliation + pure-consumer invariants,
on a SHORT frame range for speed.

End-to-end chain (synth -> S5 -> S5.5 -> S6 -> S7 -> S2.5 -> S8 -> S9 -> S10)
then Stage 11 on a short range.

Requires data/test_clip/ with video.mp4, court.json, court_zones.json,
players.parquet, poses.parquet, roster.json, user_clicks.json.

Usage:
    python -m stages.render.test_render
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2

from stages.detect_shots.detect_shots import main as detect_main
from stages.detect_bounces.detect_bounces import main as bounces_main
from stages.classify_shots.classify_shots import main as classify_main
from stages.segment_rallies.segment_rallies import main as rallies_main
from stages.classify_tracks.classify_tracks import main as roles_main
from stages.compute_metrics.compute_metrics import main as metrics_main
from stages.rate.rate import main as rate_main
from stages.plan_improvement.plan_improvement import main as plan_main
from stages.render.render import main as render_main

TEST_FOLDER = Path("data/test_clip")
SEED = 1234
START_FRAME = 1000
MAX_SECONDS = 2.0   # ~60 frames

EVENT_TYPES = {"rally_start", "rally_end", "shot", "bounce"}
INPUT_FILES = ["court.json", "players.parquet", "classified.json",
               "bounces.json", "rallies.json", "metrics.json", "rating.json",
               "improvement_plan.json", "ball.parquet"]


def _fail(m): print(f"  FAIL: {m}")
def _pass(m): print(f"  PASS: {m}")


def check_fixtures() -> bool:
    needed = ["video.mp4", "court.json", "court_zones.json", "players.parquet",
              "poses.parquet", "roster.json", "user_clicks.json"]
    missing = [f for f in needed if not (TEST_FOLDER / f).exists()]
    if missing:
        print(f"Missing fixtures in {TEST_FOLDER}: {missing}")
        return False
    return True


def gen_ball() -> bool:
    cmd = [sys.executable, "tools/synth_ball.py", str(TEST_FOLDER),
           "--seed", str(SEED), "--force"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  synth_ball failed:\n{r.stderr}")
        return False
    return True


def run_chain_to_s10() -> bool:
    for stage_main in (detect_main, bounces_main, classify_main, rallies_main,
                       metrics_main, rate_main, plan_main):
        if stage_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
            _fail(f"stage {stage_main.__module__} crashed")
            return False
    return True


def render(extra=None) -> int:
    argv = [str(TEST_FOLDER), "--force", "--start-frame", str(START_FRAME),
            "--max-seconds", str(MAX_SECONDS), "--log-level", "ERROR"]
    return render_main(argv + (extra or []))


def load(name): return json.load((TEST_FOLDER / name).open(encoding="utf-8"))


def stat_snapshot():
    return {f: (TEST_FOLDER / f).stat().st_size if (TEST_FOLDER / f).exists()
            else None for f in INPUT_FILES}


# --- Conditions --------------------------------------------------------------

def cond_outputs_exist() -> bool:
    ok = ((TEST_FOLDER / "annotated.mp4").exists()
          and (TEST_FOLDER / "timeline.json").exists()
          and any(TEST_FOLDER.glob("heatmap_*.png")))
    (_pass if ok else _fail)("outputs exist: annotated.mp4 + timeline.json + "
                             "heatmap PNGs" if ok else "missing outputs")
    return ok


def cond_video(fps_src) -> bool:
    cap = cv2.VideoCapture(str(TEST_FOLDER / "annotated.mp4"))
    if not cap.isOpened():
        _fail("annotated.mp4 won't open")
        return False
    n, w, h = int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    cap.release()
    src = cv2.VideoCapture(str(TEST_FOLDER / "video.mp4"))
    sw, sh = int(src.get(3)), int(src.get(4))
    src.release()
    expected = int(MAX_SECONDS * fps_src)
    ok = (abs(n - expected) <= 2 and w == sw and h == sh)
    (_pass if ok else _fail)(
        f"video well-formed: {n} frames (~{expected}), {w}x{h} == source {sw}x{sh}"
        if ok else f"video bad: {n} frames vs ~{expected}, {w}x{h} vs {sw}x{sh}")
    return ok


def cond_timeline_schema(tl) -> bool:
    if tl.get("schema_version") != 1:
        _fail("bad schema_version")
        return False
    ev = tl.get("events", [])
    frames = [e["frame"] for e in ev]
    if frames != sorted(frames):
        _fail("events not sorted by frame")
        return False
    for e in ev:
        if e["type"] not in EVENT_TYPES:
            _fail(f"bad event type {e['type']}")
            return False
        if e["type"] == "shot" and "shot_id" not in e:
            _fail("shot event missing shot_id")
            return False
        if e["type"] == "rally_end" and "end_reason" not in e:
            _fail("rally_end missing end_reason")
            return False
    _pass(f"timeline schema valid: {len(ev)} events, sorted, types OK")
    return True


def cond_reconciliation(tl) -> bool:
    cl, bn, ra = load("classified.json"), load("bounces.json"), load("rallies.json")
    ct = {t: sum(1 for e in tl["events"] if e["type"] == t) for t in EVENT_TYPES}
    failures = []
    if ct["shot"] != len(cl["shots"]):
        failures.append(f"shots {ct['shot']} != {len(cl['shots'])}")
    if ct["bounce"] != len(bn["bounces"]):
        failures.append(f"bounces {ct['bounce']} != {len(bn['bounces'])}")
    if ct["rally_start"] != len(ra["rallies"]) or ct["rally_end"] != len(ra["rallies"]):
        failures.append(f"rallies {ct['rally_start']}/{ct['rally_end']} != "
                        f"{len(ra['rallies'])}")
    # copied-field check: a shot event's shot_type matches classified.json
    by_id = {s["shot_id"]: s for s in cl["shots"]}
    for e in tl["events"]:
        if e["type"] == "shot":
            src = by_id.get(e["shot_id"])
            if src and e.get("shot_type") != src.get("shot_type"):
                failures.append(f"shot {e['shot_id']} type mismatch")
                break
    if failures:
        _fail(f"reconciliation: {failures[:3]}")
        return False
    _pass(f"timeline reconciliation (pure consumer): shots/bounces/rallies "
          f"counts + copied fields match source")
    return True


def cond_synthetic(tl) -> bool:
    ok = (tl.get("ball_source") == "synthetic"
          and tl["summary"].get("synthetic_ball") is True
          and "watermark" in tl.get("layers_rendered", [])
          and any("synthetic" in w.lower() or "placeholder" in w.lower()
                  for w in tl.get("warnings", [])))
    (_pass if ok else _fail)(
        "synthetic propagation: ball_source synthetic, summary flag, watermark "
        "layer, warning" if ok else "synthetic propagation incomplete")
    return ok


def cond_heatmaps() -> bool:
    pngs = sorted(TEST_FOLDER.glob("heatmap_*.png"))
    if not pngs:
        _fail("no heatmap PNGs")
        return False
    for p in pngs:
        im = cv2.imread(str(p))
        if im is None or im.shape[0] < 50 or im.shape[1] < 50:
            _fail(f"{p.name} invalid/too small")
            return False
    _pass(f"heatmap PNGs valid: {len(pngs)} files, all open + plausible dims")
    return True


def cond_hud_summary(tl) -> bool:
    rating, plan = load("rating.json"), load("improvement_plan.json")
    s = tl["summary"]
    ok = (s.get("rating") == rating.get("rating")
          and s.get("target_band") == plan.get("target", {}).get("band")
          and len(s.get("focus_areas", [])) == len(plan.get("focus_areas", [])))
    (_pass if ok else _fail)(
        "summary matches rating/plan (copied, not recomputed)"
        if ok else "summary does not match rating/plan")
    return ok


def cond_pure_consumer(before) -> bool:
    after = stat_snapshot()
    changed = [f for f in INPUT_FILES if before.get(f) != after.get(f)]
    ok = not changed
    (_pass if ok else _fail)(
        "pure-consumer: no input file modified by render"
        if ok else f"render modified inputs: {changed}")
    return ok


def cond_degradation() -> bool:
    """Hide rating.json -> still renders, rating omitted from summary, warning."""
    r = TEST_FOLDER / "rating.json"
    bak = TEST_FOLDER / "rating.json.bak"
    if not r.exists():
        _fail("rating.json missing before degradation test")
        return False
    r.rename(bak)
    try:
        rc = render()
        if rc != 0:
            _fail("render crashed without rating.json")
            return False
        tl = load("timeline.json")
        ok = ("rating" not in tl["summary"]
              and any("rating" in w.lower() for w in tl.get("warnings", []))
              and (TEST_FOLDER / "annotated.mp4").exists())
        (_pass if ok else _fail)(
            "degradation: hide rating.json -> renders, rating omitted, warned"
            if ok else "degradation behavior wrong")
        return ok
    finally:
        bak.rename(r)
        render()  # restore full timeline


# --- Runner ------------------------------------------------------------------

def run_smoke_test() -> int:
    print(f"Stage 11 smoke test - fixture: {TEST_FOLDER}")
    print()
    if not check_fixtures():
        return 1
    for stale in ("annotated.mp4", "timeline.json"):
        p = TEST_FOLDER / stale
        if p.exists():
            p.unlink()
    for p in TEST_FOLDER.glob("heatmap_*.png"):
        p.unlink()

    if roles_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
        _fail("classify_tracks (Stage 2.5) crashed")
        return 1
    if not gen_ball() or not run_chain_to_s10():
        return 1

    src = cv2.VideoCapture(str(TEST_FOLDER / "video.mp4"))
    fps_src = src.get(cv2.CAP_PROP_FPS)
    src.release()

    results = []
    before = stat_snapshot()
    if render() != 0:
        _fail("render crashed on clean run")
        return 1
    tl = load("timeline.json")

    print("Checking conditions:")
    results.append(cond_outputs_exist())
    results.append(cond_video(fps_src))
    results.append(cond_timeline_schema(tl))
    results.append(cond_reconciliation(tl))
    results.append(cond_synthetic(tl))
    results.append(cond_heatmaps())
    results.append(cond_hud_summary(tl))
    results.append(cond_pure_consumer(before))
    results.append(cond_degradation())

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())

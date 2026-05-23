"""Stage 6 — Smoke test.

Two layers:
  (1) Deterministic UNIT checks of the shot-type rule logic (clear-cut feature
      inputs -> expected type). This validates the rules independent of footage
      geometry. (End-to-end synthetic lob arcs clip at the top of this footage's
      frame, so arc-based lob accuracy is not a reliable end-to-end gate -- see
      contract Known follow-ups.)
  (2) END-TO-END run (synth_ball -> Stage 5 -> Stage 6) graded on what IS
      reliable: schema/consistency, serves -> shot_type "serve", is_volley
      accuracy vs the synthetic bounce truth, and a sane unknown rate. Plus an
      injected-gap variant that must not crash.

Requires data/test_clip/ with video.mp4, court.json, players.parquet,
poses.parquet, roster.json (from Stages 1-3 + setup).

Usage:
    python -m stages.classify_shots.test_classify_shots
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from stages.classify_shots.classify_shots import (
    classify_type, stroke_side, main as classify_main,
)
from stages.detect_shots.detect_shots import main as detect_main

TEST_FOLDER = Path("data/test_clip")
SEED = 1234
IS_VOLLEY_BAR = 0.70
UNKNOWN_TYPE_MAX_FRAC = 0.40
GAP_FRAC = 0.20

REQUIRED_CLS_KEYS = {
    "stroke_side", "stroke_side_confidence", "shot_type", "shot_type_confidence",
    "is_volley", "is_volley_confidence", "features",
}
SHOT_TYPES = {"serve", "drive", "dink", "drop", "lob", "overhead", "reset", "unknown"}
STROKE_SIDES = {"forehand", "backhand", "unknown"}


def _fail(m): print(f"  FAIL: {m}")
def _pass(m): print(f"  PASS: {m}")


def check_fixtures() -> bool:
    needed = ["video.mp4", "court.json", "players.parquet", "poses.parquet", "roster.json"]
    missing = [f for f in needed if not (TEST_FOLDER / f).exists()]
    if missing:
        print(f"Missing fixtures in {TEST_FOLDER}: {missing}")
        return False
    return True


def gen_ball(gap_frac: float) -> bool:
    cmd = [sys.executable, "tools/synth_ball.py", str(TEST_FOLDER),
           "--seed", str(SEED), "--force"]
    if gap_frac > 0:
        cmd += ["--gap-frac", str(gap_frac)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  synth_ball failed:\n{r.stderr}")
        return False
    return True


def load(name): return json.load((TEST_FOLDER / name).open(encoding="utf-8"))


# ---- (1) Unit checks of the rule logic --------------------------------------

def unit_checks() -> bool:
    # (is_serve, arc_frac, contact_h, post_ftps, pre_ftps, zone) -> expected
    cases = [
        ((True, None, "mid", None, None, "baseline"), "serve"),
        ((False, 0.50, "mid", 15.0, 10.0, "baseline"), "lob"),
        ((False, 0.10, "high", 40.0, 10.0, "transition"), "overhead"),
        ((False, 0.10, "mid", 40.0, 10.0, "baseline"), "drive"),
        ((False, 0.10, "mid", 10.0, 30.0, "transition"), "reset"),
        ((False, 0.10, "mid", 10.0, 10.0, "kitchen"), "dink"),
        ((False, 0.10, "mid", 10.0, 10.0, "baseline"), "drop"),
        ((False, 0.10, "mid", 20.0, 10.0, "transition"), "unknown"),
    ]
    bad = []
    for args, expected in cases:
        got, _ = classify_type(*args)
        if got != expected:
            bad.append(f"{args} -> {got} (expected {expected})")
    if bad:
        _fail("shot-type rule logic:\n      " + "\n      ".join(bad))
        return False
    _pass(f"shot-type rule logic: all {len(cases)} clear-cut cases classify correctly")
    return True


def unit_checks_side() -> bool:
    """Stroke side must flip for left- vs right-handed players (and handle the
    camera-facing mirror). Guards left-handed-user support."""
    away = {"lsx": 600.0, "lsv": 0.9, "rsx": 700.0, "rsv": 0.9}     # back to camera
    toward = {"lsx": 700.0, "lsv": 0.9, "rsx": 600.0, "rsv": 0.9}   # facing camera
    right_contact = 720.0  # image-right of the 650 center
    checks = [
        (right_contact, away, "right", "forehand"),
        (right_contact, away, "left", "backhand"),
        (right_contact, toward, "right", "backhand"),   # mirrored
        (right_contact, toward, "left", "forehand"),
        (right_contact, away, "unknown", "unknown"),
    ]
    bad = []
    for cx, pose, hand, expected in checks:
        side, _ = stroke_side(cx, pose, hand)
        if side != expected:
            bad.append(f"contact={cx} hand={hand} -> {side} (expected {expected})")
    if bad:
        _fail("stroke-side (left/right + facing) logic:\n      " + "\n      ".join(bad))
        return False
    _pass("stroke-side logic: left/right handedness + camera-facing mirror all correct")
    return True


# ---- (2) End-to-end checks ---------------------------------------------------

def match_truth(shots, truth, W=6):
    by_frame = truth["hits"]

    def find(f):
        best = None
        for h in by_frame:
            if abs(h["frame"] - f) <= W and (best is None or abs(h["frame"] - f) < abs(best["frame"] - f)):
                best = h
        return best
    return find


def cond_schema(cls, shots_doc) -> bool:
    cs, ss = cls["shots"], shots_doc["shots"]
    if [s["shot_id"] for s in cs] != [s["shot_id"] for s in ss]:
        _fail("classified.json shot_ids not 1:1 with shots.json")
        return False
    for s in cs:
        if not REQUIRED_CLS_KEYS <= set(s.keys()):
            _fail(f"shot {s['shot_id']} missing keys {REQUIRED_CLS_KEYS - set(s.keys())}")
            return False
        if s["shot_type"] not in SHOT_TYPES:
            _fail(f"bad shot_type {s['shot_type']}")
            return False
        if s["stroke_side"] not in STROKE_SIDES:
            _fail(f"bad stroke_side {s['stroke_side']}")
            return False
        for k in ("shot_type_confidence", "stroke_side_confidence", "is_volley_confidence"):
            if not (0.0 <= s[k] <= 1.0):
                _fail(f"{k}={s[k]} out of [0,1]")
                return False
    _pass(f"classified.json valid: {len(cs)} shots, 1:1 with shots.json, all fields/categories OK")
    return True


def run_smoke_test() -> int:
    print(f"Stage 6 smoke test - fixture: {TEST_FOLDER}")
    print()
    if not check_fixtures():
        return 1
    for stale in ("classified.json",):
        p = TEST_FOLDER / stale
        if p.exists():
            p.unlink()

    results = []

    print("Unit checks (rule logic):")
    results.append(unit_checks())
    results.append(unit_checks_side())
    print()

    # --- Phase A: gap variant must not crash ---
    print(f"Phase A: gap variant (--gap-frac {GAP_FRAC})")
    if not gen_ball(GAP_FRAC):
        return 1
    if detect_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
        _fail("Stage 5 crashed on gap variant")
        return 1
    rc = classify_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"])
    ok_gap = (rc == 0 and (TEST_FOLDER / "classified.json").exists()
              and len(load("classified.json")["shots"]) > 0)
    (_pass if ok_gap else _fail)("gap variant completed without crash")
    results.append(ok_gap)
    print()

    # --- Phase B: clean variant graded ---
    print("Phase B: clean variant")
    if not gen_ball(0.0):
        return 1
    if detect_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
        _fail("Stage 5 crashed on clean variant")
        return 1
    if classify_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
        _fail("Stage 6 crashed on clean variant")
        return 1

    cls = load("classified.json")
    shots_doc = load("shots.json")
    truth = load("ball_synth_truth.json")
    find = match_truth(cls["shots"], truth)

    print("Checking conditions:")
    results.append(cond_schema(cls, shots_doc))

    ok_src = (cls["ball_source"] == "synthetic"
              and any("synthetic" in w.lower() or "placeholder" in w.lower()
                      for w in cls["warnings"]))
    (_pass if ok_src else _fail)("ball_source=synthetic propagated with warning")
    results.append(ok_src)

    serve_bad = [s for s in cls["shots"] if s["is_serve"] and s["shot_type"] != "serve"]
    ok_serve = not serve_bad
    (_pass if ok_serve else _fail)(f"all serves classified as shot_type=serve ({serve_bad and len(serve_bad)} bad)")
    results.append(ok_serve)

    vc = vt = 0
    for s in cls["shots"]:
        h = find(s["frame"])
        if h and not h["is_serve"]:
            vt += 1
            if s["is_volley"] == h["is_volley"]:
                vc += 1
    vol_acc = vc / vt if vt else 0.0
    ok_vol = vol_acc >= IS_VOLLEY_BAR
    (_pass if ok_vol else _fail)(f"is_volley accuracy {vol_acc:.3f} (bar {IS_VOLLEY_BAR})")
    results.append(ok_vol)

    n = len(cls["shots"])
    unk = cls["stats"]["n_unknown_type"]
    ok_unk = (unk / n) < UNKNOWN_TYPE_MAX_FRAC if n else False
    (_pass if ok_unk else _fail)(f"unknown shot_type rate {unk}/{n}={unk/max(n,1):.2f} (< {UNKNOWN_TYPE_MAX_FRAC})")
    results.append(ok_unk)

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())

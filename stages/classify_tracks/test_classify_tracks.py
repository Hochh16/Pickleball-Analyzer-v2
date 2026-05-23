"""Stage 2.5 — Smoke test.

Runs classify_tracks against data/test_clip/ and verifies role assignment.
There's no full role ground truth, so checks combine partial truth (the user's
clicks), geometric consistency, noise rejection, and the core value metric
(user coverage rising above the click baseline).

Usage:
    python -m stages.classify_tracks.test_classify_tracks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stages.classify_tracks.classify_tracks import load_court, main as classify_main

TEST_FOLDER = Path("data/test_clip")
COVERAGE_GAIN_BAR = 0.10
ROLES = {"user", "partner", "opp_left", "opp_right", "noise"}
PLAYING = {"user", "partner", "opp_left", "opp_right"}


def _fail(m): print(f"  FAIL: {m}")
def _pass(m): print(f"  PASS: {m}")


def check_fixtures() -> bool:
    needed = ["court.json", "players.parquet"]
    missing = [f for f in needed if not (TEST_FOLDER / f).exists()]
    if missing:
        print(f"Missing fixtures in {TEST_FOLDER}: {missing} (run Stage 2 first)")
        return False
    return True


def track_med_y(df: pd.DataFrame) -> dict:
    return {int(t): float(np.nanmedian(g["court_y_ft"]))
            for t, g in df.groupby("track_id")}


def cond_schema(d: dict) -> bool:
    if d.get("schema_version") != 1:
        _fail("schema_version != 1")
        return False
    for tid, info in d["track_roles"].items():
        if info["role"] not in ROLES:
            _fail(f"track {tid} bad role {info['role']}")
            return False
        if not (0.0 <= info["confidence"] <= 1.0):
            _fail(f"track {tid} confidence out of [0,1]")
            return False
    # roles aggregate consistent with track_roles
    for r, agg in d["roles"].items():
        for tid in agg["track_ids"]:
            if d["track_roles"][str(tid)]["role"] != r:
                _fail(f"roles[{r}] lists track {tid} but track_roles disagrees")
                return False
    _pass(f"track_roles.json valid: {len(d['track_roles'])} tracks, roles consistent")
    return True


def cond_click_agreement(d: dict, df: pd.DataFrame) -> bool:
    user_tids = set(int(t) for t in df.loc[df["is_user"], "track_id"].unique())
    bad = [t for t in user_tids if d["track_roles"].get(str(t), {}).get("role") != "user"]
    if bad:
        _fail(f"clicked user tracks not labeled user: {bad}")
        return False
    _pass(f"click agreement: all {len(user_tids)} clicked tracks are role 'user'")
    return True


def cond_roles_and_sides(d: dict, med_y: dict, net_y: float) -> bool:
    for r in PLAYING:
        if not d["roles"][r]["track_ids"]:
            _fail(f"role '{r}' has no tracks")
            return False
    # user + partner predominantly near (median of medians < net); opps far
    def side_median(role):
        ys = [med_y[t] for t in d["roles"][role]["track_ids"] if t in med_y]
        return float(np.median(ys)) if ys else float("nan")
    near = max(side_median("user"), side_median("partner"))
    far = min(side_median("opp_left"), side_median("opp_right"))
    if not (near < net_y):
        _fail(f"user/partner not near-side (median y {near:.1f} >= net {net_y})")
        return False
    if not (far >= net_y):
        _fail(f"opponents not far-side (median y {far:.1f} < net {net_y})")
        return False
    _pass(f"4 roles populated; user/partner near (<{net_y:.0f}), opponents far (>={net_y:.0f})")
    return True


def cond_noise_rejection(d: dict, med_y: dict) -> bool:
    # adjacent-court tracks (median y beyond the far baseline) must be noise
    adj = [t for t, y in med_y.items() if y > 44.0]
    if not adj:
        _pass("no adjacent-court (y>44) tracks present; vacuously OK")
        return True
    bad = [t for t in adj if d["track_roles"].get(str(t), {}).get("role") != "noise"]
    if bad:
        _fail(f"adjacent-court tracks (y>44) not marked noise: {bad[:10]}")
        return False
    _pass(f"all {len(adj)} adjacent-court (y>44) tracks marked noise")
    return True


def cond_coverage(d: dict) -> bool:
    cov = d["stats"]["user_frame_coverage"]
    was = d["stats"]["user_frame_coverage_was_is_user"]
    if cov < was + COVERAGE_GAIN_BAR:
        _fail(f"user coverage {cov:.3f} did not rise >= {COVERAGE_GAIN_BAR} above "
              f"click baseline {was:.3f}")
        return False
    _pass(f"user coverage rose {was:.1%} -> {cov:.1%} (>= +{COVERAGE_GAIN_BAR:.0%})")
    return True


def run_smoke_test() -> int:
    print(f"Stage 2.5 smoke test - fixture: {TEST_FOLDER}")
    print()
    if not check_fixtures():
        return 1
    out = TEST_FOLDER / "track_roles.json"
    if out.exists():
        out.unlink()

    print("Running classify_tracks.main()...")
    rc = classify_main([str(TEST_FOLDER), "--force", "--log-level", "WARNING"])
    if rc != 0 or not out.exists():
        print(f"FAIL: classify_tracks returned {rc} / no output")
        return 1
    d = json.load(out.open(encoding="utf-8"))
    df = pd.read_parquet(TEST_FOLDER / "players.parquet")
    court = load_court(TEST_FOLDER / "court.json")
    med_y = track_med_y(df)

    print("Checking conditions:")
    results = [
        cond_schema(d),
        cond_click_agreement(d, df),
        cond_roles_and_sides(d, med_y, court["net_y"]),
        cond_noise_rejection(d, med_y),
        cond_coverage(d),
    ]
    print()
    print(f"{sum(results)}/{len(results)} conditions passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())

"""Stage 4 (v4) — smoke test.

The v4 detector forward pass is a heavy 720p TrackNet that is GPU-bound and far
too slow to run inside a unit test on CPU (~12 s/frame). The detector's
per-frame accuracy is already graded by the training run's validation_report
(val recall 0.90 / fp 0.02 same court, 0.54 / 0.02 cross court). What is NOT
covered there — and what this test gates on — is the court-agnostic trajectory
post-processing (postprocess) and the output schema invariants, which we can
drive deterministically with synthetic detection dicts (no model, no video).

Covers:
  1. Confident detections survive and are emitted visible with their confidence.
  2. An isolated velocity outlier (far from BOTH neighbors) is dropped.
  3. A short gap between confirmed detections is linearly interpolated and
     marked interpolated (visible=False, confidence NaN).
  4. A long gap (> MAX_GAP_FRAMES) is left not-visible (all NaN).
  5. Schema invariants hold on every emitted row: columns, contiguity over the
     requested frame list, and the per-row visible/interpolated/neither state
     machine (matches synth_ball / Stage 11 consumer expectations).

Usage:
    python -m stages.track_ball.test_track_ball_v4

Exit codes:
    0  all checks passed
    1  one or more checks failed
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from stages.track_ball.track_ball_v4 import (
    postprocess, MAX_GAP_FRAMES, OUTLIER_MAX_STEP_PX, SCHEMA_VERSION,
)


def _fail(m): print(f"  FAIL: {m}"); return False
def _pass(m): print(f"  PASS: {m}"); return True


def rows_to_df(rows):
    """Mirror the DataFrame construction in run() so invariants are tested on
    the same dtypes the parquet would carry."""
    df = pd.DataFrame(rows)
    df.insert(0, "schema_version", SCHEMA_VERSION)
    df["visible"] = df["visible"].astype(bool)
    df["interpolated"] = df["interpolated"].astype(bool)
    df["confidence"] = df["confidence"].astype("float32")
    return df


# ---------- schema invariants (v4: frame_idx is the requested range) ----------

def check_schema_invariants(df: pd.DataFrame, frames: list) -> bool:
    ok = True
    expected_cols = {"schema_version", "frame_idx", "pixel_x", "pixel_y",
                     "visible", "confidence", "interpolated"}
    cols = set(df.columns)
    if cols != expected_cols:
        ok = _fail(f"columns {sorted(cols)} != {sorted(expected_cols)}") and ok

    if not (df["schema_version"] == SCHEMA_VERSION).all():
        ok = _fail("schema_version not uniform") and ok

    if list(df["frame_idx"]) != list(frames):
        ok = _fail("frame_idx does not match the requested frame list") and ok

    both = df["visible"] & df["interpolated"]
    if both.any():
        ok = _fail(f"{int(both.sum())} rows are both visible and interpolated") and ok

    vis = df[df["visible"]]
    if vis[["pixel_x", "pixel_y", "confidence"]].isna().any().any():
        ok = _fail("a visible row has NaN x/y/confidence") and ok

    interp = df[df["interpolated"]]
    if interp[["pixel_x", "pixel_y"]].isna().any().any():
        ok = _fail("an interpolated row has NaN x/y") and ok
    if interp["confidence"].notna().any():
        ok = _fail("an interpolated row has non-NaN confidence") and ok

    neither = df[~df["visible"] & ~df["interpolated"]]
    if neither[["pixel_x", "pixel_y", "confidence"]].notna().any().any():
        ok = _fail("a not-visible row has non-NaN data") and ok

    if ok:
        _pass(f"schema invariants hold on all {len(df)} rows")
    return ok


# ---------- logic checks ----------

def test_confident_detections_survive() -> bool:
    # ball drifting steadily right — every frame a clean detection
    dets = {f: (100.0 + 10 * f, 200.0, 0.8) for f in range(5)}
    frames = list(range(5))
    df = rows_to_df(postprocess(dets, frames))
    ok = True
    if int(df["visible"].sum()) != 5:
        ok = _fail(f"expected 5 visible, got {int(df['visible'].sum())}") and ok
    if not np.allclose(df["confidence"].to_numpy(), 0.8, atol=1e-5):
        ok = _fail("confidence not carried through on visible rows") and ok
    if ok:
        _pass("confident detections survive as visible with their confidence")
    return ok and check_schema_invariants(df, frames)


def test_outlier_rejected() -> bool:
    # frames 0,1,3,4 form a smooth line at y=200; frame 2 teleports far away.
    step = OUTLIER_MAX_STEP_PX + 50
    dets = {0: (100.0, 200.0, 0.7), 1: (110.0, 200.0, 0.7),
            2: (110.0 + 2 * step, 200.0 + 2 * step, 0.7),  # impossible from both sides
            3: (130.0, 200.0, 0.7), 4: (140.0, 200.0, 0.7)}
    frames = list(range(5))
    df = rows_to_df(postprocess(dets, frames))
    r2 = df[df["frame_idx"] == 2].iloc[0]
    ok = True
    if r2["visible"]:
        ok = _fail("isolated velocity outlier was NOT dropped") and ok
    # neighbours 1 and 3 are 2 frames apart -> short gap -> interpolated
    if not r2["interpolated"]:
        ok = _fail("dropped-outlier frame should be interpolated from neighbours") and ok
    else:
        # interpolated midpoint should sit on the smooth line near x=120, y=200
        if abs(r2["pixel_x"] - 120.0) > 1.0 or abs(r2["pixel_y"] - 200.0) > 1.0:
            ok = _fail(f"interp landed at ({r2['pixel_x']:.1f},{r2['pixel_y']:.1f}), "
                       f"expected ~(120,200)") and ok
    if ok:
        _pass("velocity outlier dropped, then interpolated back onto the line")
    return ok and check_schema_invariants(df, frames)


def test_short_gap_interpolated() -> bool:
    # detections only at frames 0 and 4 (gap of 4 <= MAX_GAP); 1..3 unseen
    dets = {0: (100.0, 100.0, 0.9), 4: (140.0, 140.0, 0.9)}
    frames = list(range(5))
    assert 4 <= MAX_GAP_FRAMES
    df = rows_to_df(postprocess(dets, frames))
    ok = True
    interp = df[df["interpolated"]]
    if list(interp["frame_idx"]) != [1, 2, 3]:
        ok = _fail(f"expected frames 1,2,3 interpolated, got {list(interp['frame_idx'])}") and ok
    # linear: frame 2 is the midpoint -> (120,120)
    r2 = df[df["frame_idx"] == 2].iloc[0]
    if abs(r2["pixel_x"] - 120.0) > 1e-6 or abs(r2["pixel_y"] - 120.0) > 1e-6:
        ok = _fail(f"midpoint interp ({r2['pixel_x']},{r2['pixel_y']}) != (120,120)") and ok
    if r2["visible"] or not pd.isna(r2["confidence"]):
        ok = _fail("interpolated row must be visible=False, confidence NaN") and ok
    if ok:
        _pass("short gap linearly interpolated and flagged interpolated")
    return ok and check_schema_invariants(df, frames)


def test_long_gap_left_missing() -> bool:
    # gap of MAX_GAP_FRAMES + 2 between detections -> not interpolated
    gap = MAX_GAP_FRAMES + 2
    dets = {0: (100.0, 100.0, 0.9), gap: (200.0, 200.0, 0.9)}
    frames = list(range(gap + 1))
    df = rows_to_df(postprocess(dets, frames))
    ok = True
    middle = df[(df["frame_idx"] > 0) & (df["frame_idx"] < gap)]
    if middle["visible"].any() or middle["interpolated"].any():
        ok = _fail("a gap longer than MAX_GAP_FRAMES should be left not-visible") and ok
    if not middle[["pixel_x", "pixel_y", "confidence"]].isna().all().all():
        ok = _fail("not-visible rows must be all-NaN") and ok
    if ok:
        _pass(f"gap of {gap} (> MAX_GAP_FRAMES={MAX_GAP_FRAMES}) left not-visible")
    return ok and check_schema_invariants(df, frames)


def test_offset_frame_range() -> bool:
    # frames don't start at 0 — Stage 4 runs on arbitrary [start,end) windows
    dets = {300: (50.0, 60.0, 0.6), 301: (55.0, 62.0, 0.6)}
    frames = list(range(300, 305))
    df = rows_to_df(postprocess(dets, frames))
    return check_schema_invariants(df, frames)


def main() -> int:
    tests = [
        ("confident detections survive", test_confident_detections_survive),
        ("velocity outlier rejected", test_outlier_rejected),
        ("short gap interpolated", test_short_gap_interpolated),
        ("long gap left missing", test_long_gap_left_missing),
        ("offset frame range", test_offset_frame_range),
    ]
    print("Stage 4 (v4) smoke test — trajectory post-processing + schema\n")
    results = []
    for name, fn in tests:
        print(f"[{name}]")
        results.append(bool(fn()))
        print()
    n_pass = sum(results)
    print(f"OVERALL: {'PASS' if all(results) else 'FAIL'} "
          f"({n_pass}/{len(results)} checks passed)")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

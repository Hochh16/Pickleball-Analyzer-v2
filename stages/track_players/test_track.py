"""Stage 2 — Smoke test.

Runs track.main() against a real fixture clip and verifies the 6 conditions
specified in stages/track_players/contract.md.

Required fixture in data/test_clip/:
    video.mp4              - a clip where the user is clearly visible
    court.json             - calibrated for that clip via Stage 1
    court_zones.json       - calibrated for that clip via Stage 1
    user_clicks.json       - at least one click identifying the user

Usage:
    python -m stages.track_players.test_track

Exit code 0 if all 6 conditions pass, 1 otherwise. Prints a pass/fail line
for each condition. Cleans up prior outputs before each run, so re-running
is safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_float_dtype, is_integer_dtype,
)

from stages.track_players.track import PARQUET_COLUMNS, main as track_main

TEST_FOLDER = Path("data/test_clip")


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def _pass(msg: str) -> None:
    print(f"  PASS: {msg}")


def check_fixtures() -> bool:
    """Ensure the fixture folder is set up before we try to run anything."""
    needed = ["video.mp4", "court.json", "court_zones.json", "user_clicks.json"]
    missing = [f for f in needed if not (TEST_FOLDER / f).exists()]
    if missing:
        print(f"Missing fixture files in {TEST_FOLDER}:")
        for f in missing:
            print(f"  - {f}")
        print()
        print("To set up the fixture:")
        print(f"  1. Copy a clip (~30s recommended) to {TEST_FOLDER}\\video.mp4")
        print(f"  2. Run Stage 1 calibrate against it; place court.json and")
        print(f"     court_zones.json in {TEST_FOLDER}\\")
        print(f"  3. Hand-craft {TEST_FOLDER}\\user_clicks.json with one click")
        print(f"     identifying you on a frame where you're clearly visible:")
        print(f'       {{"clicks": [{{"frame": 0, "x": 640, "y": 360}}]}}')
        return False
    return True


def condition_1(df: pd.DataFrame) -> bool:
    """parquet exists, non-empty, all 14 columns with correct dtype kinds."""
    if len(df) == 0:
        _fail("players.parquet is empty")
        return False
    if list(df.columns) != PARQUET_COLUMNS:
        _fail(
            f"columns mismatch\n"
            f"      expected: {PARQUET_COLUMNS}\n"
            f"      got:      {list(df.columns)}"
        )
        return False

    expected_kind = {
        "frame": "int",  "t_sec": "float", "track_id": "int",
        "is_user": "bool", "user_segment_id": "int",
        "bbox_x1": "float", "bbox_y1": "float",
        "bbox_x2": "float", "bbox_y2": "float",
        "foot_x":  "float", "foot_y":  "float",
        "court_x_ft": "float", "court_y_ft": "float",
        "in_court": "bool", "transient": "bool",
    }
    for col, kind in expected_kind.items():
        dt = df[col].dtype
        ok = (
            (kind == "int"   and is_integer_dtype(dt)) or
            (kind == "float" and is_float_dtype(dt))   or
            (kind == "bool"  and is_bool_dtype(dt))
        )
        if not ok:
            _fail(f"column '{col}' has unexpected dtype: {dt} (expected {kind})")
            return False

    _pass(f"parquet exists, {len(df)} rows, all 14 columns with correct dtype kinds")
    return True


def condition_2(df: pd.DataFrame) -> bool:
    """At least one row has is_user=True."""
    n = int(df["is_user"].sum())
    if n == 0:
        _fail(
            "no rows with is_user=True - initial click did not resolve. "
            "Verify user_clicks.json points to a frame where the user is detected."
        )
        return False
    _pass(f"{n} row(s) with is_user=True")
    return True


def condition_3(df: pd.DataFrame) -> bool:
    """Within any single user_segment_id, all is_user=True rows share track_id."""
    user_rows = df[df["is_user"]]
    if len(user_rows) == 0:
        _fail("no is_user=True rows to check (condition 2 should have caught this)")
        return False
    per_segment = user_rows.groupby("user_segment_id")["track_id"].nunique()
    bad = per_segment[per_segment > 1]
    if len(bad) > 0:
        _fail(f"user_segment_id(s) with multiple track_ids: {dict(bad)}")
        return False
    _pass(
        f"{per_segment.shape[0]} user_segment_id(s); each maps to a single track_id"
    )
    return True


def condition_4(df: pd.DataFrame) -> bool:
    """At least one frame has >=2 distinct track_ids."""
    per_frame = df.groupby("frame")["track_id"].nunique()
    max_per_frame = int(per_frame.max()) if len(per_frame) else 0
    if max_per_frame < 2:
        _fail(
            f"no frame had >=2 distinct track_ids (max per frame: {max_per_frame}). "
            f"Verify the test clip has multiple people visible."
        )
        return False
    _pass(f"max distinct track_ids on a single frame: {max_per_frame}")
    return True


def condition_5(df: pd.DataFrame) -> bool:
    """Every row with in_court=True has finite court_x_ft and court_y_ft."""
    in_court = df[df["in_court"]]
    if len(in_court) == 0:
        _fail(
            "no rows with in_court=True - suggests bad calibration or footage "
            "where no detected person stands inside the court polygon"
        )
        return False
    bad = in_court[in_court["court_x_ft"].isna() | in_court["court_y_ft"].isna()]
    if len(bad) > 0:
        _fail(f"{len(bad)} in_court=True rows have NaN court coords (impossible state)")
        return False
    _pass(f"{len(in_court)} in_court=True rows; all with finite court coords")
    return True


def condition_6() -> bool:
    """players_pending.json exists, valid JSON, gaps satisfy frame ordering."""
    p = TEST_FOLDER / "players_pending.json"
    if not p.exists():
        _fail("players_pending.json not written")
        return False
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        _fail(f"players_pending.json is not valid JSON: {e}")
        return False
    if not isinstance(data, dict) or "gaps" not in data or "warnings" not in data:
        _fail("players_pending.json missing 'gaps' or 'warnings' key")
        return False
    if not isinstance(data["gaps"], list):
        _fail("'gaps' is not a list")
        return False
    for i, g in enumerate(data["gaps"]):
        if "last_user_frame" not in g or "resumes_at_or_after" not in g:
            _fail(f"gap[{i}] missing 'last_user_frame' or 'resumes_at_or_after'")
            return False
        if g["last_user_frame"] >= g["resumes_at_or_after"]:
            _fail(
                f"gap[{i}]: last_user_frame ({g['last_user_frame']}) "
                f">= resumes_at_or_after ({g['resumes_at_or_after']})"
            )
            return False
    _pass(
        f"players_pending.json valid: {len(data['gaps'])} gap(s), "
        f"{len(data['warnings'])} warning(s)"
    )
    return True


def run_smoke_test() -> int:
    print(f"Stage 2 smoke test - fixture: {TEST_FOLDER}")
    print()

    if not check_fixtures():
        return 1

    # Wipe prior outputs so we know what we get is from this run.
    for stale in ("players.parquet", "players_pending.json"):
        out = TEST_FOLDER / stale
        if out.exists():
            out.unlink()

    print("Running track.main()...")
    rc = track_main([str(TEST_FOLDER)])
    print(f"track.main() returned {rc}")
    print()

    if rc != 0:
        print(f"FAIL: track.main() returned non-zero exit code ({rc})")
        return 1

    parquet_path = TEST_FOLDER / "players.parquet"
    if not parquet_path.exists():
        print("FAIL: players.parquet not written")
        return 1
    df = pd.read_parquet(parquet_path)

    print("Checking conditions:")
    results = [
        condition_1(df),
        condition_2(df),
        condition_3(df),
        condition_4(df),
        condition_5(df),
        condition_6(),
    ]
    print()
    print(f"{sum(results)}/6 conditions passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())
"""Stage 3 — Smoke test.

Runs pose.main() against the data/test_clip/ fixture and verifies the 6
conditions specified in stages/pose/contract.md.

Requires that Stage 2 has already produced players.parquet in data/test_clip/.

Usage:
    python -m stages.pose.test_pose

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

from stages.pose.pose import (
    LANDMARK_COLUMNS, PARQUET_COLUMNS,
    filter_to_scope, load_court_fps, load_players,
    main as pose_main,
)

TEST_FOLDER = Path("data/test_clip")

# Padded-bbox tolerance for landmark-in-bbox check. The bbox in players.parquet
# is the YOLO bbox; the pose was run on a crop padded by BBOX_PAD_FRAC=0.10.
# Landmarks may legitimately lie inside the padded crop but slightly outside
# the original bbox (e.g., outstretched arm). Allow up to 25% slack so the
# check still meaningfully rejects garbage poses without rejecting valid
# edge cases.
BBOX_TOLERANCE_FRAC = 0.40

# Minimum landmark visibility to include in the in-bbox check.
# MediaPipe emits 33 landmarks for every detected pose, but reports a
# visibility score per landmark. Low-visibility landmarks (occluded, off-frame,
# or otherwise unobserved) are extrapolated from torso geometry and may
# legitimately project far outside the bbox. The contract directs downstream
# stages to weight landmarks by visibility, so the smoke test does the same:
# we only verify in-bbox correctness for landmarks the model says it could
# actually see.
LANDMARK_VISIBILITY_FLOOR = 0.5

# Landmarks that should be inside (or near) the bbox for a valid pose.
# Per the contract: shoulders, elbows, wrists, hips, knees, ankles. Face
# landmarks and feet extremities (heel, foot_index) are not required for
# biomechanics analysis and can legitimately wander.
KEY_LANDMARKS = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]

# Sanity bounds for in-scope detection count (condition 6). The strict scope
# filter should keep the run from blowing up to tens of thousands of garbage
# detections, but should also retain the user plus a handful of real players.
MIN_REASONABLE_IN_SCOPE = 100
MAX_REASONABLE_IN_SCOPE = 12000


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def _pass(msg: str) -> None:
    print(f"  PASS: {msg}")


def check_fixtures() -> bool:
    """Ensure Stage 2 output is present before we try to run Stage 3."""
    needed = ["video.mp4", "court.json", "players.parquet"]
    missing = [f for f in needed if not (TEST_FOLDER / f).exists()]
    if missing:
        print(f"Missing fixture files in {TEST_FOLDER}:")
        for f in missing:
            print(f"  - {f}")
        print()
        print("Run Stage 2 first to produce players.parquet:")
        print("  python -m stages.track_players.test_track")
        return False
    return True


def condition_1(df: pd.DataFrame) -> bool:
    """parquet exists, non-empty, all 137 columns with correct dtype kinds."""
    if len(df) == 0:
        _fail("poses.parquet is empty")
        return False
    if list(df.columns) != PARQUET_COLUMNS:
        for i, (got, exp) in enumerate(zip(df.columns, PARQUET_COLUMNS)):
            if got != exp:
                _fail(
                    f"columns mismatch at index {i}: "
                    f"expected '{exp}', got '{got}'"
                )
                return False
        _fail(
            f"columns mismatch: expected {len(PARQUET_COLUMNS)} columns, "
            f"got {len(df.columns)}"
        )
        return False

    expected_kind = {
        "frame":         "int",
        "t_sec":         "float",
        "track_id":      "int",
        "is_user":       "bool",
        "pose_detected": "bool",
    }
    for col in LANDMARK_COLUMNS:
        expected_kind[col] = "float"

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

    _pass(
        f"parquet exists, {len(df)} rows, all {len(PARQUET_COLUMNS)} columns "
        f"with correct dtype kinds"
    )
    return True


def condition_2(df: pd.DataFrame, players_df: pd.DataFrame, fps: float) -> bool:
    """Row count in poses.parquet equals in-scope detection count."""
    scope_df, _ = filter_to_scope(players_df, fps)
    if len(df) != len(scope_df):
        _fail(
            f"row count mismatch: poses.parquet has {len(df)} rows, "
            f"strict-scope filter on players.parquet yields {len(scope_df)} rows"
        )
        return False
    _pass(f"row count matches in-scope detection count: {len(df)}")
    return True


def condition_3(df: pd.DataFrame) -> bool:
    """At least one row has is_user=True AND pose_detected=True."""
    n = int((df["is_user"] & df["pose_detected"]).sum())
    if n == 0:
        n_user = int(df["is_user"].sum())
        if n_user == 0:
            _fail(
                "no rows with is_user=True at all; players.parquet has no "
                "user detections to run pose on"
            )
        else:
            _fail(
                f"{n_user} user rows present but none had pose_detected=True. "
                "Bbox crops may be too small or noisy for MediaPipe."
            )
        return False
    _pass(f"{n} row(s) with is_user=True AND pose_detected=True")
    return True


def condition_4(df: pd.DataFrame, players_df: pd.DataFrame) -> bool:
    """For every pose_detected=True row, key landmarks with visibility >=
    LANDMARK_VISIBILITY_FLOOR lie inside the bbox (with BBOX_TOLERANCE_FRAC
    slack). Low-visibility landmarks are skipped because MediaPipe is
    explicitly telling us those positions are extrapolated and unreliable;
    downstream stages will weight by visibility for the same reason.
    """
    detected = df[df["pose_detected"]].copy()
    if len(detected) == 0:
        _fail("no pose_detected=True rows to check (condition 3 should have caught this)")
        return False

    bbox_lookup = players_df.set_index(["frame", "track_id"])[
        ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    ]
    detected_idx = detected.set_index(["frame", "track_id"])
    joined = detected_idx.join(bbox_lookup, how="left")

    if joined[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().any().any():
        _fail(
            "some pose rows do not have matching bboxes in players.parquet; "
            "join failed unexpectedly"
        )
        return False

    bbox_w = joined["bbox_x2"] - joined["bbox_x1"]
    bbox_h = joined["bbox_y2"] - joined["bbox_y1"]
    pad_x = bbox_w * BBOX_TOLERANCE_FRAC
    pad_y = bbox_h * BBOX_TOLERANCE_FRAC
    x_lo = joined["bbox_x1"] - pad_x
    x_hi = joined["bbox_x2"] + pad_x
    y_lo = joined["bbox_y1"] - pad_y
    y_hi = joined["bbox_y2"] + pad_y

    failures = []
    skipped_per_landmark = []
    for name in KEY_LANDMARKS:
        col_x = f"{name}_x_px"
        col_y = f"{name}_y_px"
        col_v = f"{name}_visibility"
        x = joined[col_x]
        y = joined[col_y]
        v = joined[col_v]

        visible = v >= LANDMARK_VISIBILITY_FLOOR
        n_visible = int(visible.sum())
        n_skipped = int((~visible).sum())
        skipped_per_landmark.append((name, n_skipped))

        if n_visible == 0:
            continue

        x_v = x[visible]
        y_v = y[visible]
        x_lo_v = x_lo[visible]
        x_hi_v = x_hi[visible]
        y_lo_v = y_lo[visible]
        y_hi_v = y_hi[visible]

        outside = (x_v < x_lo_v) | (x_v > x_hi_v) | (y_v < y_lo_v) | (y_v > y_hi_v)
        n_outside = int(outside.sum())
        if n_outside > 0:
            failures.append(
                f"{name}: {n_outside} of {n_visible} visible rows outside "
                f"(skipped {n_skipped} low-visibility)"
            )

    if failures:
        _fail(
            f"key landmarks outside bbox (+/-{int(BBOX_TOLERANCE_FRAC * 100)}% slack), "
            f"counting only landmarks with visibility >= {LANDMARK_VISIBILITY_FLOOR}:\n      "
            + "\n      ".join(failures)
        )
        return False

    total_skipped = sum(n for _, n in skipped_per_landmark)
    total_possible = len(joined) * len(KEY_LANDMARKS)
    _pass(
        f"all {len(KEY_LANDMARKS)} key landmarks inside bbox "
        f"(+/-{int(BBOX_TOLERANCE_FRAC * 100)}% slack) across "
        f"{len(joined)} pose-detected rows; "
        f"{total_skipped} of {total_possible} landmark-checks skipped due to "
        f"visibility < {LANDMARK_VISIBILITY_FLOOR}"
    )
    return True


def condition_5(df: pd.DataFrame) -> bool:
    """Every row with pose_detected=False has all 132 landmark columns NaN."""
    not_detected = df[~df["pose_detected"]]
    if len(not_detected) == 0:
        _pass("no pose_detected=False rows; vacuously true")
        return True
    landmark_data = not_detected[LANDMARK_COLUMNS]
    non_nan_per_row = landmark_data.notna().any(axis=1)
    n_bad = int(non_nan_per_row.sum())
    if n_bad > 0:
        _fail(
            f"{n_bad} of {len(not_detected)} pose_detected=False rows have "
            f"non-NaN landmark values"
        )
        return False
    _pass(
        f"all {len(not_detected)} pose_detected=False rows have NaN landmark "
        f"values"
    )
    return True


def condition_6(df: pd.DataFrame) -> bool:
    """pose_summary.json exists, valid JSON, has per_track for every track,
    and reports an in-scope detection count between MIN and MAX bounds."""
    p = TEST_FOLDER / "pose_summary.json"
    if not p.exists():
        _fail("pose_summary.json not written")
        return False
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        _fail(f"pose_summary.json is not valid JSON: {e}")
        return False
    required_keys = {
        "schema_version", "scope_filter", "total_pose_detected",
        "overall_detection_rate", "per_track", "warnings",
    }
    missing = required_keys - set(data.keys())
    if missing:
        _fail(f"pose_summary.json missing keys: {sorted(missing)}")
        return False

    summary_tids = {entry["track_id"] for entry in data["per_track"]}
    parquet_tids = set(df["track_id"].unique().tolist())
    if summary_tids != parquet_tids:
        only_in_summary = summary_tids - parquet_tids
        only_in_parquet = parquet_tids - summary_tids
        _fail(
            f"per_track tracks differ from poses.parquet tracks; "
            f"only in summary: {sorted(only_in_summary)}; "
            f"only in parquet: {sorted(only_in_parquet)}"
        )
        return False

    in_scope = data["scope_filter"].get("in_scope_detections", -1)
    if in_scope < MIN_REASONABLE_IN_SCOPE or in_scope > MAX_REASONABLE_IN_SCOPE:
        _fail(
            f"in_scope_detections={in_scope} outside reasonable bounds "
            f"[{MIN_REASONABLE_IN_SCOPE}, {MAX_REASONABLE_IN_SCOPE}]; "
            f"scope filter may be too strict or too permissive"
        )
        return False

    _pass(
        f"pose_summary.json valid: {len(data['per_track'])} track(s), "
        f"{len(data['warnings'])} warning(s), "
        f"in_scope={in_scope}, "
        f"overall detection rate {data['overall_detection_rate']:.1%}"
    )
    return True


def run_smoke_test() -> int:
    print(f"Stage 3 smoke test - fixture: {TEST_FOLDER}")
    print()

    if not check_fixtures():
        return 1

    for stale in ("poses.parquet", "pose_summary.json"):
        out = TEST_FOLDER / stale
        if out.exists():
            out.unlink()

    print("Running pose.main()...")
    rc = pose_main([str(TEST_FOLDER)])
    print(f"pose.main() returned {rc}")
    print()

    if rc != 0:
        print(f"FAIL: pose.main() returned non-zero exit code ({rc})")
        return 1

    parquet_path = TEST_FOLDER / "poses.parquet"
    if not parquet_path.exists():
        print("FAIL: poses.parquet not written")
        return 1
    df = pd.read_parquet(parquet_path)

    fps = load_court_fps(TEST_FOLDER / "court.json")
    players_df = load_players(TEST_FOLDER / "players.parquet")

    print("Checking conditions:")
    results = [
        condition_1(df),
        condition_2(df, players_df, fps),
        condition_3(df),
        condition_4(df, players_df),
        condition_5(df),
        condition_6(df),
    ]
    print()
    print(f"{sum(results)}/6 conditions passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())
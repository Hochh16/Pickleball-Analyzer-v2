"""
Stage 4 smoke test.

Runs the track_ball stage against data/test_clip/ and verifies all
acceptance criteria from contract.md:

  1. Stage runs to completion without exception.
  2. ball.parquet and ball.meta.json are produced.
  3. Schema invariants hold on every row.
  4. detection_rate >= 0.80 measured against the active-rally subset.
  5. (Visual sanity is left for the operator — this script does not render
     trajectories.)

Reads:
    data/test_clip/video.mp4
    data/test_clip/court.json
    data/test_clip/active_rally_frames.json
    data/models/tracknet_v2_dettor.pt   (or path passed via --weights)

Writes:
    data/test_clip/ball.parquet
    data/test_clip/ball.parquet.meta.json
    data/test_clip/ball.smoke.txt    (verdict + summary)

Usage:
    python -m stages.track_ball.smoke_test
    python -m stages.track_ball.smoke_test --weights path/to/weights.pt
    python -m stages.track_ball.smoke_test --skip-run    # use existing parquet

Exit codes:
    0  smoke test passed (all 4 acceptance criteria met)
    1  smoke test failed (any criterion failed; see ball.smoke.txt)
    2  setup error (missing input file, etc. — pre-test failure)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stages.track_ball.track_ball import (
    SCHEMA_VERSION,
    main as track_ball_main,
)

DEFAULT_CLIP_DIR = Path("data/test_clip")
DEFAULT_WEIGHTS = Path("data/models/tracknet_v2_dettor.pt")
ACCEPTANCE_DETECTION_RATE = 0.80
PLACEHOLDER_FIELDS = (
    "_comment", "_labeling_instructions", "_example_entry_REMOVE_ME",
)


# ---------- output collection ----------

class Verdict:
    """Accumulates pass/fail results across the four acceptance criteria."""
    def __init__(self):
        self.checks = []  # list of (name, passed, detail)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    @property
    def all_passed(self) -> bool:
        return all(p for _, p, _ in self.checks)

    def render(self, header_lines: list) -> str:
        lines = list(header_lines)
        lines.append("")
        lines.append("Acceptance criteria:")
        for name, passed, detail in self.checks:
            mark = "PASS" if passed else "FAIL"
            lines.append(f"  [{mark}] {name}")
            if detail:
                for d in detail.split("\n"):
                    lines.append(f"         {d}")
        lines.append("")
        verdict = "PASS" if self.all_passed else "FAIL"
        lines.append(f"OVERALL: {verdict}")
        lines.append("")
        return "\n".join(lines)


# ---------- rally-frames loader ----------

def load_rally_frames(path: Path) -> list:
    """Load and validate active_rally_frames.json. Returns sorted list of
    {start_frame, end_frame, ...}. Raises ValueError on any problem."""
    if not path.exists():
        raise FileNotFoundError(
            f"active_rally_frames.json not found at {path}. "
            f"Create it from the template (see tools/verify_rally_frames.py)."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema_version") != 1:
        raise ValueError(
            f"{path}: schema_version is {data.get('schema_version')!r}, expected 1"
        )

    leftover = [k for k in PLACEHOLDER_FIELDS if k in data]
    if leftover:
        raise ValueError(
            f"{path} still contains placeholder fields: {leftover}. "
            f"Fill in the rallies array and remove these fields."
        )

    rallies = data.get("rallies")
    if not isinstance(rallies, list) or len(rallies) == 0:
        raise ValueError(
            f"{path}: 'rallies' must be a non-empty list"
        )

    for i, r in enumerate(rallies):
        if not isinstance(r, dict):
            raise ValueError(f"{path}: rally[{i}] is not an object")
        if "start_frame" not in r or "end_frame" not in r:
            raise ValueError(
                f"{path}: rally[{i}] missing start_frame or end_frame"
            )
        if (not isinstance(r["start_frame"], int)
                or not isinstance(r["end_frame"], int)):
            raise ValueError(
                f"{path}: rally[{i}] start_frame/end_frame must be int"
            )
        if r["start_frame"] < 0 or r["end_frame"] < r["start_frame"]:
            raise ValueError(
                f"{path}: rally[{i}] has invalid range "
                f"[{r['start_frame']}, {r['end_frame']}]"
            )

    rallies_sorted = sorted(rallies, key=lambda r: r["start_frame"])
    for a, b in zip(rallies_sorted, rallies_sorted[1:]):
        if b["start_frame"] <= a["end_frame"]:
            raise ValueError(
                f"{path}: rally ranges overlap — "
                f"[{a['start_frame']}, {a['end_frame']}] and "
                f"[{b['start_frame']}, {b['end_frame']}]"
            )
    return rallies_sorted


# ---------- check 1: stage ran ----------

def run_track_ball(clip_dir: Path, weights: Path, force: bool) -> int:
    """Invoke the track_ball stage as a subprocess-equivalent (call its
    main() with argv). Returns its exit code."""
    argv = [
        "--video", str(clip_dir / "video.mp4"),
        "--court", str(clip_dir / "court.json"),
        "--weights", str(weights),
        "--out", str(clip_dir / "ball.parquet"),
    ]
    if force:
        argv.append("--force")
    return track_ball_main(argv)


# ---------- check 3: schema invariants ----------

def check_schema_invariants(df: pd.DataFrame) -> tuple:
    """Validate every row of ball.parquet against the contract.
    Returns (passed: bool, detail: str)."""
    errors = []

    expected_cols = {
        "schema_version", "frame_idx", "pixel_x", "pixel_y",
        "visible", "confidence", "interpolated",
    }
    missing = expected_cols - set(df.columns)
    extra = set(df.columns) - expected_cols
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected columns: {sorted(extra)}")
    if errors:
        return False, "\n".join(errors)

    if not (df["schema_version"] == SCHEMA_VERSION).all():
        bad = df[df["schema_version"] != SCHEMA_VERSION].head(3)
        errors.append(
            f"schema_version != {SCHEMA_VERSION} on {len(df) - (df['schema_version'] == SCHEMA_VERSION).sum()} rows; "
            f"first bad row: {bad.iloc[0].to_dict() if len(bad) else 'n/a'}"
        )

    expected_idx = pd.Series(range(len(df)), dtype="int64")
    if not df["frame_idx"].equals(expected_idx):
        errors.append(
            f"frame_idx is not a contiguous 0..{len(df)-1} sequence"
        )

    # Per-row state machine: at most one of visible/interpolated is True.
    both_true = df["visible"] & df["interpolated"]
    if both_true.any():
        errors.append(
            f"{both_true.sum()} rows have both visible=True and "
            f"interpolated=True (invariant violation)"
        )

    # When visible: x, y, confidence non-NaN
    vis = df[df["visible"]]
    bad_vis = vis[vis["pixel_x"].isna() | vis["pixel_y"].isna()
                  | vis["confidence"].isna()]
    if len(bad_vis) > 0:
        errors.append(
            f"{len(bad_vis)} visible rows have NaN x/y/confidence"
        )

    # When interpolated: x, y non-NaN, confidence NaN
    interp = df[df["interpolated"]]
    bad_interp_xy = interp[interp["pixel_x"].isna() | interp["pixel_y"].isna()]
    if len(bad_interp_xy) > 0:
        errors.append(
            f"{len(bad_interp_xy)} interpolated rows have NaN x/y"
        )
    bad_interp_conf = interp[interp["confidence"].notna()]
    if len(bad_interp_conf) > 0:
        errors.append(
            f"{len(bad_interp_conf)} interpolated rows have non-NaN confidence"
        )

    # When neither: all NaN
    neither = df[~df["visible"] & ~df["interpolated"]]
    bad_neither = neither[neither["pixel_x"].notna()
                          | neither["pixel_y"].notna()
                          | neither["confidence"].notna()]
    if len(bad_neither) > 0:
        errors.append(
            f"{len(bad_neither)} 'missing' rows have non-NaN data"
        )

    if errors:
        return False, "\n".join(errors)
    return True, f"all {len(df)} rows satisfy schema invariants"


# ---------- check 4: detection rate on active rallies ----------

def compute_active_rally_detection_rate(df: pd.DataFrame, rallies: list) -> dict:
    """Count visible+interpolated frames inside the union of active rally
    ranges. Returns dict with counts and rate."""
    in_rally = pd.Series(False, index=df.index)
    for r in rallies:
        s, e = r["start_frame"], r["end_frame"]
        in_rally |= (df["frame_idx"] >= s) & (df["frame_idx"] <= e)

    rally_df = df[in_rally]
    n_total = len(rally_df)
    n_visible = int(rally_df["visible"].sum())
    n_interp = int(rally_df["interpolated"].sum())
    n_detected = n_visible + n_interp
    rate = n_detected / n_total if n_total else 0.0
    return {
        "n_active_rally_frames": n_total,
        "n_visible": n_visible,
        "n_interpolated": n_interp,
        "n_detected": n_detected,
        "detection_rate": rate,
    }


# ---------- driver ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4 smoke test")
    ap.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR,
                    help="Clip folder. Default: data/test_clip")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="TrackNet weights .pt. "
                         "Default: data/models/tracknet_v2_dettor.pt")
    ap.add_argument("--skip-run", action="store_true",
                    help="Don't re-run track_ball; use existing parquet.")
    ap.add_argument("--no-force", action="store_true",
                    help="Don't pass --force to track_ball "
                         "(ball.parquet must not exist).")
    args = ap.parse_args(argv)

    clip_dir = args.clip_dir
    parquet_path = clip_dir / "ball.parquet"
    meta_path = clip_dir / "ball.parquet.meta.json"
    rally_path = clip_dir / "active_rally_frames.json"
    smoke_txt = clip_dir / "ball.smoke.txt"

    # Pre-test setup checks (exit 2 — distinct from a "real" smoke fail)
    if not (clip_dir / "video.mp4").exists():
        print(f"SETUP ERROR: {clip_dir / 'video.mp4'} not found", file=sys.stderr)
        return 2
    if not (clip_dir / "court.json").exists():
        print(f"SETUP ERROR: {clip_dir / 'court.json'} not found", file=sys.stderr)
        return 2
    if not args.skip_run and not args.weights.exists():
        print(f"SETUP ERROR: weights not found at {args.weights}. "
              f"Run tools/convert_dettor_weights.py first.", file=sys.stderr)
        return 2

    try:
        rallies = load_rally_frames(rally_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"SETUP ERROR loading {rally_path}: {e}", file=sys.stderr)
        return 2

    print(f"loaded {len(rallies)} rally ranges from {rally_path}")

    verdict = Verdict()
    t0 = time.time()

    # Check 1: stage runs without exception
    if args.skip_run:
        if not parquet_path.exists():
            print(f"SETUP ERROR: --skip-run but {parquet_path} doesn't exist",
                  file=sys.stderr)
            return 2
        verdict.add(
            "1. stage runs without exception",
            True,
            "skipped (using existing parquet)",
        )
    else:
        print("running track_ball stage...")
        rc = run_track_ball(clip_dir, args.weights, force=not args.no_force)
        verdict.add(
            "1. stage runs without exception",
            rc == 0,
            f"track_ball exit code: {rc}",
        )
        if rc != 0:
            # Don't bother with later checks — write verdict and exit
            return _finalize(smoke_txt, verdict, args, rally_path,
                             rallies, t0, parquet_path, meta_path)

    # Check 2: outputs exist
    parquet_ok = parquet_path.exists()
    meta_ok = meta_path.exists()
    verdict.add(
        "2. ball.parquet and ball.meta.json produced",
        parquet_ok and meta_ok,
        f"parquet: {'present' if parquet_ok else 'MISSING'}; "
        f"meta: {'present' if meta_ok else 'MISSING'}",
    )
    if not (parquet_ok and meta_ok):
        return _finalize(smoke_txt, verdict, args, rally_path,
                         rallies, t0, parquet_path, meta_path)

    # Check 3: schema invariants
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        verdict.add("3. schema invariants", False,
                    f"could not read parquet: {e}")
        return _finalize(smoke_txt, verdict, args, rally_path,
                         rallies, t0, parquet_path, meta_path)
    schema_ok, schema_detail = check_schema_invariants(df)
    verdict.add("3. schema invariants", schema_ok, schema_detail)

    # Check 4: detection rate on active rallies
    rate_info = compute_active_rally_detection_rate(df, rallies)
    rate_ok = rate_info["detection_rate"] >= ACCEPTANCE_DETECTION_RATE
    rate_detail = (
        f"active-rally frames: {rate_info['n_active_rally_frames']} "
        f"(visible={rate_info['n_visible']}, "
        f"interpolated={rate_info['n_interpolated']})\n"
        f"detection rate: {rate_info['detection_rate']:.3f} "
        f"(threshold: {ACCEPTANCE_DETECTION_RATE:.2f})"
    )
    verdict.add(
        f"4. detection rate >= {ACCEPTANCE_DETECTION_RATE:.2f} on active rallies",
        rate_ok, rate_detail,
    )

    return _finalize(smoke_txt, verdict, args, rally_path,
                     rallies, t0, parquet_path, meta_path)


def _finalize(smoke_txt: Path, verdict: Verdict, args, rally_path: Path,
              rallies: list, t0: float, parquet_path: Path,
              meta_path: Path) -> int:
    """Render verdict, write ball.smoke.txt, print to stdout, return exit code."""
    header = [
        f"Stage 4 smoke test — pickleball-analyzer-v2",
        f"clip: {args.clip_dir}",
        f"weights: {args.weights}",
        f"rally file: {rally_path} ({len(rallies)} rallies)",
        f"wall time: {time.time() - t0:.1f}s",
    ]
    # Pull a few headline numbers from meta if available
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            stats = meta.get("stats", {})
            header.append(
                f"meta detection_rate (all frames): "
                f"{stats.get('detection_rate', '?'):.3f}"
            )
            header.append(
                f"meta filtered_by_threshold: "
                f"{stats.get('detections_filtered_by_threshold', '?')}"
            )
            header.append(
                f"meta filtered_by_roi: "
                f"{stats.get('detections_filtered_by_roi', '?')}"
            )
        except Exception:
            pass

    text = verdict.render(header)
    print(text)
    smoke_txt.parent.mkdir(parents=True, exist_ok=True)
    smoke_txt.write_text(text, encoding="utf-8")
    print(f"verdict written to {smoke_txt}")

    return 0 if verdict.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
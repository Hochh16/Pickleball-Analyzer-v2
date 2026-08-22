"""Stage 5 — Smoke test.

Generates a synthetic ball (impacts at real player positions) with tools/
synth_ball.py, runs Stage 5, and verifies the 6 conditions in
stages/detect_shots/contract.md against the synthetic ground truth.

Requires data/test_clip/ to already contain (from Stages 1-3):
    video.mp4, court.json, players.parquet, poses.parquet

Usage:
    python -m stages.detect_shots.test_detect_shots

Exit 0 if all conditions pass, 1 otherwise. Leaves the folder with a CLEAN
(gap-free) synthetic ball + shots.json. Re-running is safe.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stages.detect_shots.detect_shots import main as detect_main, reject_same_side_runs

TEST_FOLDER = Path("data/test_clip")
SEED = 1234

# Acceptance bars (synthetic data; see contract).
RECALL_BAR = 0.80
PLAYER_MATCH_BAR = 0.80
PRECISION_BAR = 0.70
SERVE_RECALL_BAR = 0.70
GAP_FRAC = 0.20

REQUIRED_TOP_KEYS = {
    "schema_version", "video_path", "fps", "ball_source", "params",
    "shots", "stats", "warnings", "stage_version",
}
REQUIRED_SHOT_KEYS = {
    "shot_id", "frame", "t_sec", "track_id", "is_user", "is_serve",
    "detection_method", "impact_pixel_xy", "impact_court_xy_ft",
    "player_distance_px", "assoc_basis", "pre_velocity_px_per_frame",
    "post_velocity_px_per_frame", "direction_change_deg", "turn_rate_deg",
    "speed_change_ratio", "confidence",
}


def _fail(m): print(f"  FAIL: {m}")
def _pass(m): print(f"  PASS: {m}")


def check_fixtures() -> bool:
    needed = ["video.mp4", "court.json", "players.parquet", "poses.parquet"]
    missing = [f for f in needed if not (TEST_FOLDER / f).exists()]
    if missing:
        print(f"Missing fixtures in {TEST_FOLDER}: {missing}")
        print("Run Stage 2 then Stage 3 first:")
        print("  python -m stages.track_players.test_track")
        print("  python -m stages.pose.test_pose")
        return False
    return True


def gen_ball(gap_frac: float) -> bool:
    cmd = [sys.executable, "tools/synth_ball.py", str(TEST_FOLDER),
           "--seed", str(SEED), "--force"]
    if gap_frac > 0:
        cmd += ["--gap-frac", str(gap_frac)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  synth_ball failed (rc={r.returncode}):\n{r.stderr}")
        return False
    return True


def run_stage5() -> int:
    return detect_main([str(TEST_FOLDER), "--force", "--log-level", "WARNING"])


def load(name): return json.load((TEST_FOLDER / name).open(encoding="utf-8"))


def grade(shots, truth):
    """Return (recall, player_match, precision, serve_recall) on non-serve hits."""
    W = shots["params"]["impact_window_frames"]
    sh = shots["shots"]
    det = [h for h in truth["hits"] if not h["is_serve"]]
    serves = [h for h in truth["hits"] if h["is_serve"]]

    def matched(h):
        return [s for s in sh if abs(s["frame"] - h["frame"]) <= W]

    rec = sum(1 for h in det if matched(h))
    pmatch = sum(1 for h in det
                 if any(s["track_id"] == h["track_id"] for s in matched(h)))

    def matched_serve(h):  # truth serve recovered by a shot flagged is_serve
        return [s for s in sh if s.get("is_serve")
                and abs(s["frame"] - h["frame"]) <= W]

    srec = sum(1 for h in serves if matched_serve(h))

    def s_match(s):
        return any(abs(s["frame"] - h["frame"]) <= W for h in truth["hits"])

    spur = sum(1 for s in sh if not s_match(s))
    recall = rec / len(det) if det else 0.0
    player_match = pmatch / rec if rec else 0.0
    precision = (len(sh) - spur) / len(sh) if sh else 0.0
    serve_recall = srec / len(serves) if serves else 0.0
    return recall, player_match, precision, serve_recall


def condition_1(shots) -> bool:
    if set(shots.keys()) < REQUIRED_TOP_KEYS:
        _fail(f"shots.json missing top keys: {REQUIRED_TOP_KEYS - set(shots.keys())}")
        return False
    if shots["schema_version"] != 1:
        _fail(f"schema_version != 1: {shots['schema_version']}")
        return False
    sh = shots["shots"]
    if not sh:
        _fail("no shots produced")
        return False
    for i, s in enumerate(sh):
        if set(s.keys()) < REQUIRED_SHOT_KEYS:
            _fail(f"shot {i} missing keys: {REQUIRED_SHOT_KEYS - set(s.keys())}")
            return False
    frames = [s["frame"] for s in sh]
    if frames != sorted(frames):
        _fail("shots not sorted by frame")
        return False
    if [s["shot_id"] for s in sh] != list(range(len(sh))):
        _fail("shot_id not contiguous from 0")
        return False
    _pass(f"shots.json valid: {len(sh)} shots, sorted, contiguous shot_id, all fields present")
    return True


def condition_2(shots) -> bool:
    if shots["ball_source"] != "synthetic":
        _fail(f"ball_source != synthetic: {shots['ball_source']}")
        return False
    if not any("synthetic" in w.lower() or "placeholder" in w.lower()
               for w in shots["warnings"]):
        _fail("no synthetic/placeholder warning present")
        return False
    _pass("ball_source=synthetic with placeholder warning present")
    return True


def run_smoke_test() -> int:
    print(f"Stage 5 smoke test - fixture: {TEST_FOLDER}")
    print()
    if not check_fixtures():
        return 1
    for stale in ("shots.json",):
        p = TEST_FOLDER / stale
        if p.exists():
            p.unlink()

    results = []

    # --- Phase A: injected-gap variant (condition 6) ---
    print(f"Phase A: gap variant (--gap-frac {GAP_FRAC})")
    if not gen_ball(GAP_FRAC):
        return 1
    rc = run_stage5()
    if rc != 0:
        _fail(f"Stage 5 crashed on gap variant (rc={rc})")
        results.append(False)
    else:
        shots_g = load("shots.json")
        truth_g = load("ball_synth_truth.json")
        rec_g, _, prec_g, _ = grade(shots_g, truth_g)
        ok6 = (len(shots_g["shots"]) > 0 and prec_g >= PRECISION_BAR and rec_g > 0.0)
        (_pass if ok6 else _fail)(
            f"gap variant completed; recall={rec_g:.3f} (>0, degraded), "
            f"precision={prec_g:.3f} (>= {PRECISION_BAR}); no fabrication")
        results.append(ok6)
    print()

    # --- Phase B: clean variant (conditions 1-5) ---
    print("Phase B: clean variant")
    if not gen_ball(0.0):
        return 1
    rc = run_stage5()
    if rc != 0:
        _fail(f"Stage 5 crashed on clean variant (rc={rc})")
        return 1
    shots = load("shots.json")
    truth = load("ball_synth_truth.json")
    recall, player_match, precision, serve_recall = grade(shots, truth)

    print("Checking conditions:")
    results.append(condition_1(shots))
    results.append(condition_2(shots))

    ok3 = recall >= RECALL_BAR
    (_pass if ok3 else _fail)(f"non-serve recall {recall:.3f} (bar {RECALL_BAR})")
    results.append(ok3)

    ok4 = player_match >= PLAYER_MATCH_BAR
    (_pass if ok4 else _fail)(f"non-serve player-match {player_match:.3f} (bar {PLAYER_MATCH_BAR})")
    results.append(ok4)

    ok5 = precision >= PRECISION_BAR
    (_pass if ok5 else _fail)(f"precision {precision:.3f} (bar {PRECISION_BAR})")
    results.append(ok5)

    ok6 = serve_recall >= SERVE_RECALL_BAR
    (_pass if ok6 else _fail)(f"serve recall {serve_recall:.3f} (bar {SERVE_RECALL_BAR}) "
                              f"- via the separate appearance signal")
    results.append(ok6)

    # --- Phase C: teleport-drop robustness + resolution-scale sanity ---
    print()
    print("Phase C: teleport-drop + resolution scaling")
    bp = TEST_FOLDER / "ball.parquet"
    bdf = pd.read_parquet(bp)  # clean ball from Phase B
    vis_idx = bdf.index[bdf["visible"]].tolist()
    # pick a frame whose immediate neighbors are also visible, so the injected
    # jump is an unambiguous impossible pair
    tgt = next((i for i in vis_idx[len(vis_idx) // 4: -1]
                if (i - 1) in vis_idx and (i + 1) in vis_idx), vis_idx[len(vis_idx) // 2])
    orig_x = float(bdf.loc[tgt, "pixel_x"])
    bdf.loc[tgt, "pixel_x"] = orig_x + 5000.0  # physically impossible jump
    bdf.to_parquet(bp, index=False)
    rc = run_stage5()
    if rc != 0:
        _fail("Stage 5 crashed on injected teleport (should DROP it, not crash)")
        results.append(False)
    else:
        sj = load("shots.json")
        nd = sj["stats"].get("n_teleport_dropped", 0)
        rs = sj["params"].get("resolution_scale")
        amax = sj["params"].get("assoc_max_px")
        # 1080p test_clip => res_scale 1.0 => px thresholds unchanged (no regression)
        okC = (nd >= 1 and rs is not None and abs(rs - 1.0) < 1e-6
               and amax is not None and abs(amax - 120.0) < 1e-6)
        (_pass if okC else _fail)(
            f"injected teleport dropped without crashing (n_teleport_dropped={nd}); "
            f"resolution_scale={rs} at 1080p (=1.0), assoc_max_px={amax} (=120 base)")
        results.append(okC)
    # restore the clean value so the fixture isn't left corrupted
    bdf.loc[tgt, "pixel_x"] = orig_x
    bdf.to_parquet(bp, index=False)

    # --- Phase D: net-side alternation filter (ball-handling rejection) ---
    print()
    print("Phase D: net-side alternation filter")
    side = {1: "near", 2: "far"}
    # an alternating rally is fully kept
    rally = [{"frame": f, "track_id": t} for f, t in [(0, 1), (20, 2), (40, 1), (60, 2)]]
    k_rally, d_rally = reject_same_side_runs(rally, side, 90)
    # A TIGHT same-side run is a strike plus a tracking wobble: keep the STRONGEST, not the
    # last. (This asserted "last" until v0.5.1, which is what "wrong player" turned out to
    # be -- the old rule deleted a 172-degree paddle reversal and kept a 26-degree wobble.)
    run = [{"frame": f, "track_id": 1, "turn_rate_deg": t, "speed_change_ratio": 0.0}
           for f, t in [(100, 20.0), (114, 172.0), (130, 15.0), (150, 26.0), (168, 10.0)]]
    k_run, d_run = reject_same_side_runs(run, side, 90)
    # A run SPREAD over seconds is genuine handling (bounce, bounce, serve): keep the LAST.
    spread = [{"frame": f, "track_id": 1, "turn_rate_deg": t, "speed_change_ratio": 0.0}
              for f, t in [(0, 172.0), (300, 20.0), (560, 30.0)]]
    k_spread, d_spread = reject_same_side_runs(spread, side, 600, 60.0)
    # same side after a long gap (> reset) is a new rally, kept
    newrally = [{"frame": 0, "track_id": 1}, {"frame": 200, "track_id": 1}]
    k_new, d_new = reject_same_side_runs(newrally, side, 90)
    okD = (len(k_rally) == 4 and d_rally == 0
           and len(k_run) == 1 and d_run == 4 and k_run[0]["frame"] == 114
           and len(k_spread) == 1 and d_spread == 2 and k_spread[0]["frame"] == 560
           and len(k_new) == 2 and d_new == 0)
    (_pass if okD else _fail)(
        f"alternation filter: rally kept {len(k_rally)}/4 (drop {d_rally}); "
        f"tight run -> strongest kept {[s['frame'] for s in k_run]} (drop {d_run}); "
        f"spread run -> last kept {[s['frame'] for s in k_spread]} (drop {d_spread}); "
        f"new-rally kept {len(k_new)}/2")
    results.append(okD)

    # --- Phase D2: the ball-excursion split inside a same-side run ---
    # The run premise is "nobody hit it in between", which is exactly what a MISSED shot
    # violates -- and then the filter deletes a second real shot. A ball that travelled a
    # long way from the first impact and came back did not stay in the hitter's hands, so
    # the two impacts are separate shots however the sides were attributed.
    print()
    print("Phase D2: ball-excursion split")
    n_fr = 200
    known_arr = np.ones(n_fr, dtype=bool)
    # ball parked at the hitter the whole time: genuine handling, no split
    still_x = np.full(n_fr, 500.0)
    still_y = np.full(n_fr, 900.0)
    k_still, d_still = reject_same_side_runs(
        run, side, 90, 60.0, ball_xy=(still_x, still_y, known_arr), excursion_px=450.0)
    # ball leaves and comes back between every pair: separate shots, nothing dropped
    away_x = still_x.copy()
    for a, b in zip((100, 114, 130, 150), (114, 130, 150, 168)):
        away_x[(a + b) // 2] = 500.0 + 900.0
    k_away, d_away = reject_same_side_runs(
        run, side, 90, 60.0, ball_xy=(away_x, still_y, known_arr), excursion_px=450.0)
    # no ball given -> unchanged from the shipped behaviour
    k_none, d_none = reject_same_side_runs(run, side, 90, 60.0)
    okD2 = (len(k_still) == 1 and d_still == 4
            and len(k_away) == 5 and d_away == 0
            and len(k_none) == len(k_run) and d_none == d_run)
    (_pass if okD2 else _fail)(
        f"excursion split: parked ball still collapses to {len(k_still)}/5 (drop {d_still}); "
        f"ball leaving between each impact keeps {len(k_away)}/5 (drop {d_away}); "
        f"no ball data leaves the old behaviour ({len(k_none)} kept, drop {d_none})")
    results.append(okD2)

    # --- Phase D3: the net-crossing split, from the 3-D reconstruction ---
    # Only available on a re-run (ball_3d.parquet needs bounces from Stage 5.5), and only
    # trustworthy as a SUSTAINED crossing -- the per-frame reconstruction is noisy enough
    # that a single outlier frame would split nearly every pair.
    print()
    print("Phase D3: net-crossing split")
    net = 22.0
    parked = {f: 10.0 for f in range(90, 180)}                     # never leaves the near side
    k_park, d_park = reject_same_side_runs(
        run, side, 90, 60.0, ball_court_y=parked, net_y_ft=net, cross_frames=10)
    blip = dict(parked)
    for f in (120, 121):                                          # two stray frames only
        blip[f] = 40.0
    k_blip, d_blip = reject_same_side_runs(
        run, side, 90, 60.0, ball_court_y=blip, net_y_ft=net, cross_frames=10)
    over = dict(parked)
    for a, b in zip((100, 114, 130, 150), (114, 130, 150, 168)):
        for f in range((a + b) // 2 - 6, (a + b) // 2 + 6):        # 12 frames on the far side
            over[f] = 40.0
    k_over, d_over = reject_same_side_runs(
        run, side, 90, 60.0, ball_court_y=over, net_y_ft=net, cross_frames=10)
    okD3 = (len(k_park) == 1 and d_park == 4
            and len(k_blip) == 1 and d_blip == 4
            and len(k_over) == 5 and d_over == 0)
    (_pass if okD3 else _fail)(
        f"net-crossing split: ball staying near-side collapses to {len(k_park)}/5; "
        f"a 2-frame blip past the net does NOT split it ({len(k_blip)}/5); "
        f"a sustained crossing between each impact keeps {len(k_over)}/5")
    results.append(okD3)

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())

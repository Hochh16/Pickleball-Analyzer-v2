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

from stages.detect_shots.detect_shots import (main as detect_main,
                                              reject_same_side_runs,
                                              structure_points)

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
    k_rally, d_rally, _, _ = reject_same_side_runs(rally, side, 90)
    # A TIGHT same-side run is a strike plus a tracking wobble: keep the STRONGEST, not the
    # last. (This asserted "last" until v0.5.1, which is what "wrong player" turned out to
    # be -- the old rule deleted a 172-degree paddle reversal and kept a 26-degree wobble.)
    run = [{"frame": f, "track_id": 1, "turn_rate_deg": t, "speed_change_ratio": 0.0}
           for f, t in [(100, 20.0), (114, 172.0), (130, 15.0), (150, 26.0), (168, 10.0)]]
    k_run, d_run, _, _ = reject_same_side_runs(run, side, 90)
    # A run SPREAD over seconds is genuine handling (bounce, bounce, serve): keep the LAST.
    spread = [{"frame": f, "track_id": 1, "turn_rate_deg": t, "speed_change_ratio": 0.0}
              for f, t in [(0, 172.0), (300, 20.0), (560, 30.0)]]
    k_spread, d_spread, _, _ = reject_same_side_runs(spread, side, 600, 60.0)
    # same side after a long gap (> reset) is a new rally, kept
    newrally = [{"frame": 0, "track_id": 1}, {"frame": 200, "track_id": 1}]
    k_new, d_new, _, _ = reject_same_side_runs(newrally, side, 90)
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
    k_still, d_still, _, _ = reject_same_side_runs(
        run, side, 90, 60.0, ball_xy=(still_x, still_y, known_arr), excursion_px=450.0)
    # ball leaves and comes back between every pair: separate shots, nothing dropped
    away_x = still_x.copy()
    for a, b in zip((100, 114, 130, 150), (114, 130, 150, 168)):
        away_x[(a + b) // 2] = 500.0 + 900.0
    k_away, d_away, _, _ = reject_same_side_runs(
        run, side, 90, 60.0, ball_xy=(away_x, still_y, known_arr), excursion_px=450.0)
    # no ball given -> unchanged from the shipped behaviour
    k_none, d_none, _, _ = reject_same_side_runs(run, side, 90, 60.0)
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
    k_park, d_park, _, _ = reject_same_side_runs(
        run, side, 90, 60.0, ball_court_y=parked, net_y_ft=net, cross_frames=10)
    blip = dict(parked)
    for f in (120, 121):                                          # two stray frames only
        blip[f] = 40.0
    k_blip, d_blip, _, _ = reject_same_side_runs(
        run, side, 90, 60.0, ball_court_y=blip, net_y_ft=net, cross_frames=10)
    over = dict(parked)
    for a, b in zip((100, 114, 130, 150), (114, 130, 150, 168)):
        for f in range((a + b) // 2 - 6, (a + b) // 2 + 6):        # 12 frames on the far side
            over[f] = 40.0
    k_over, d_over, _, _ = reject_same_side_runs(
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



def _shot(frame, side, dist_from_net, net_y=22.0):
    """A shot at `frame`, hit by `side`, that far from the net."""
    y = net_y - dist_from_net if side == "near" else net_y + dist_from_net
    return {"frame": frame, "hitter_side": side, "hitter_court_xy_ft": [10.0, y]}


def _args(fps=60.0):
    return dict(net_y_ft=22.0, behind_baseline_ft=21.0,
                open_gap_frames=int(3.0 * fps), return_frames=int(2.5 * fps),
                dead_gap_frames=int(3.0 * fps), min_inter_serve_frames=int(10.0 * fps))


def test_junk_on_the_servers_own_side_does_not_hide_the_serve():
    """The operator's most-reported pattern is false shots just before a serve, and they do
    real damage: a detection 2.45s before the confirmed serve at 0:46.57 shortened the gap
    below the 3s "opens a point" threshold, so the serve was never even a candidate and its
    whole point -- serve and return -- vanished from the analysis.

    Every rally shot crosses the net, so the gap that matters is to the last contact from the
    OTHER side. Junk on the server's own side cannot hide their serve.
    """
    fps = 60.0
    shots = [_shot(0, "near", 25.0),                    # a serve, 0.0s
             _shot(int(1.0 * fps), "far", 25.0),        # returned
             _shot(int(20.0 * fps), "far", 8.0),        # junk, own side, 20.0s
             _shot(int(22.5 * fps), "far", 30.0),       # THE SERVE, 2.5s after the junk
             _shot(int(23.5 * fps), "near", 20.0)]      # answered
    structure_points(shots, **_args(fps))
    assert [bool(s["is_serve"]) for s in shots] == [True, False, False, True, False]


def test_a_relaxed_candidate_must_be_answered():
    """A serve is played back; a ball handled in dead time is not. Without this the
    relaxation opened a rally at 3:31 made of four junk shots and no real ones."""
    fps = 60.0
    shots = [_shot(0, "near", 25.0),
             _shot(int(1.0 * fps), "far", 25.0),
             _shot(int(20.0 * fps), "far", 8.0),
             _shot(int(22.5 * fps), "far", 30.0)]       # deep, but nobody replies
    structure_points(shots, **_args(fps))
    assert not shots[3]["is_serve"]


def test_a_relaxed_candidate_never_displaces_a_strict_one():
    """Weaker evidence may fill a slot the strict rule left empty, never take one it filled.
    Letting it displace cost court B its 1:44 serve to a candidate 2.9s later -- recall +1
    outdoors and -1 indoors, a wash. Restricted this way it is +1 with nothing lost."""
    fps = 60.0
    shots = [_shot(0, "near", 25.0),                    # strict serve
             _shot(int(2.9 * fps), "near", 30.0),       # relaxed candidate, same slot
             _shot(int(3.9 * fps), "far", 20.0)]        # answers the SECOND one
    structure_points(shots, **_args(fps))
    assert shots[0]["is_serve"] and not shots[1]["is_serve"]


def test_a_discarded_serve_comes_back_when_the_server_had_the_ball():
    """Operator, 2026-08-27: "once I know which side is serving, I can tell which shot is a
    serve by a) they have the ball and b) ball moves forward toward the net" -- (b) being
    what separates a serve from an underhand feed to their own partner.

    reject_same_side_runs keeps ONE contact per same-side run, and a serve sits in a run with
    the server's own bouncing, so it is routinely the one discarded. All three conditions are
    needed: on the discards, formation+side alone admitted 16 for 7 serves, adding an
    underhand test made it worse (56), and adding HAS-THE-BALL + moves-forward gave 13.
    """
    from stages.detect_shots.detect_shots import restore_serves
    import numpy as np
    import pandas as pd
    fps, L, net = 60.0, 44.0, 22.0
    F = 300
    # serving side is NEAR: two players behind y=0, one behind y=44
    formation = pd.DataFrame(
        [{"frame": f, "track_id": t, "court_y_ft": y}
         for f in range(F - 5, F + 6)
         for t, y in ((1, -3.0), (2, -5.0), (3, 47.0), (4, 30.0))])
    # the server holds the ball: it stays on them through the window before contact
    players_px = {(f, 1): (500.0, 800.0, 200.0) for f in range(F - 120, F + 2)}
    n = F + 60
    bx = np.full(n, 505.0)
    by = np.full(n, 810.0)
    known = np.ones(n, dtype=bool)
    # ...and afterwards the ball heads to the far side
    court_y = {f: 2.0 for f in range(F - 120, F + 1)}
    court_y.update({f: 2.0 + (f - F) * 0.5 for f in range(F + 1, F + 50)})

    disc = [{"frame": F, "track_id": 1}]
    got = restore_serves([{"frame": 10, "track_id": 1}], disc, {1: "near"}, formation,
                         players_px, bx, by, known, court_y, L, net, fps)
    assert len(got) == 1 and got[0]["restored_as_serve"] is True

    # ball never leaves -> a feed or handling, not a serve
    flat = {f: 2.0 for f in range(F - 120, F + 50)}
    assert restore_serves([{"frame": 10, "track_id": 1}], [dict(d) for d in disc],
                          {1: "near"}, formation, players_px, bx, by, known, flat,
                          L, net, fps) == []
    # struck by the RECEIVING side -> not a serve
    assert restore_serves([{"frame": 10, "track_id": 3}], [{"frame": F, "track_id": 3}],
                          {3: "far"}, formation, players_px, bx, by, known, court_y,
                          L, net, fps) == []
    # a shot we already kept sits right there -> we are recovering a MISSING serve only
    assert restore_serves([{"frame": F - 30, "track_id": 1}], [dict(d) for d in disc],
                          {1: "near"}, formation, players_px, bx, by, known, court_y,
                          L, net, fps) == []


def test_the_receiving_pair_stacks_for_a_serve():
    """The formation tie-break rests on this reading of the court, so pin the reading.

    A serve has the receiving team split -- returner deep, partner at the non-volley line.
    Nothing mid-rally looks like that, which is why it separates a real serve (84%) from a
    false accept (30%) where every cue about the BALL's later flight tied.
    """
    import pandas as pd

    from stages.detect_shots.detect_shots import receiver_at_kitchen

    L = 44.0

    def frame_at(ys, frame=100):
        return pd.DataFrame([{"frame": f, "track_id": i, "court_y_ft": y}
                             for i, y in enumerate(ys) for f in range(frame - 4, frame + 5)])

    # near side serves: receivers are the far pair (y > 22). One deep behind the far
    # baseline, one up at the far kitchen line (22 + 7 = 29).
    stacked = frame_at([-2.0, -3.0, 46.0, 29.2])
    assert receiver_at_kitchen(stacked, 100, "near", L) is True

    # mid-rally: both receivers at mid-court, neither at the kitchen line
    midcourt = frame_at([-2.0, -3.0, 36.0, 38.0])
    assert receiver_at_kitchen(midcourt, 100, "near", L) is False

    # the far side serving flips which pair counts as receivers
    far_serve = frame_at([-1.0, 15.1, 46.0, 47.0])
    assert receiver_at_kitchen(far_serve, 100, "far", L) is True

    # no evidence is not evidence of absence
    assert receiver_at_kitchen(None, 100, "near", L) is None
    empty = pd.DataFrame(columns=["frame", "track_id", "court_y_ft"])
    assert receiver_at_kitchen(empty, 100, "near", L) is None
    assert receiver_at_kitchen(pd.DataFrame(), 100, "near", L) is None
    assert receiver_at_kitchen(stacked, 100, None, L) is None

"""Stage 8 — Smoke test.

End-to-end run (synth_ball -> S5 -> S5.5 -> S6 -> S7 -> S2.5 -> S8) graded on
RECONCILIATION + schema + reliability (Stage 8 is aggregation; there is no
per-player ground truth on real tracks, so arithmetic invariants are the gate):

  - schema/consistency of metrics.json
  - shot reconciliation: sum(per-role n_shots) + unattributed == total shots
  - error reconciliation: sum(by_owner) == n_rallies
  - end_reason passthrough == rallies.json stats.by_end_reason
  - serve metric (n_serves == n_rallies, faults == serve-fault bucket)
  - rally-length stats recomputed from rallies.json
  - position stats (real): zone/lateral/area sum ~1, area marginals == zone,
    coverage in [0,1], movement finite, team positioning valid
  - heatmap integrity: grid sums == in-extent counts (recomputed via helpers)
  - reliability + synthetic propagation; real-data families NOT synth-gated
  - pending_real_ball block all null + listed under reliability.pending
  - role-confidence contamination flag
  - degradation: hide track_roles.json -> user-only fallback, no crash
Plus an injected-gap variant that must not crash.

Requires data/test_clip/ with video.mp4, court.json, court_zones.json,
players.parquet, poses.parquet, roster.json, user_clicks.json.

Usage:
    python -m stages.compute_metrics.test_compute_metrics
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd

from stages.detect_shots.detect_shots import main as detect_main
from stages.detect_bounces.detect_bounces import main as bounces_main
from stages.classify_shots.classify_shots import main as classify_main
from stages.segment_rallies.segment_rallies import main as rallies_main
from stages.classify_tracks.classify_tracks import main as roles_main
from stages.compute_metrics.compute_metrics import (
    main as metrics_main, role_valid_rows, role_frame_pos, bin_positions,
    pose_front_foot, role_front_foot_pos, scope_to_rally_frames,
    rally_len_bucket, PLAYING_ROLES,
)

TEST_FOLDER = Path("data/test_clip")
SEED = 1234
GAP_FRAC = 0.20

REQUIRED_TOP_KEYS = {
    "schema_version", "sources", "ball_source", "fps", "params", "match",
    "error_attribution", "players", "team", "heatmaps", "pending_real_ball",
    "reliability", "warnings", "stage_version", "completed_at_utc",
}
VALID_OWNERS = set(PLAYING_ROLES) | {"team_near", "team_far", "unattributed", "unknown"}


def _fail(m): print(f"  FAIL: {m}")
def _pass(m): print(f"  PASS: {m}")

# --- v2 confidence-wrapper helpers ------------------------------------------
WRAP_KEYS = {"value", "confidence", "n", "limited_by"}
LIMITERS = {"sample_size", "measurement", "known_limit", "detection_floor"}


def _is_wrapped(x) -> bool:
    return isinstance(x, dict) and WRAP_KEYS <= set(x.keys())


def _v(metric):
    """Unwrap a {value, confidence, n, limited_by} metric to its raw value."""
    return metric["value"] if _is_wrapped(metric) else metric


def check_fixtures() -> bool:
    needed = ["video.mp4", "court.json", "court_zones.json", "players.parquet",
              "poses.parquet", "roster.json", "user_clicks.json"]
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


def run_chain(force_log="ERROR") -> bool:
    """S5 -> S5.5 -> S6 -> S7 -> S8 (roles run separately; ball-independent)."""
    for stage_main in (detect_main, bounces_main, classify_main, rallies_main,
                        metrics_main):
        if stage_main([str(TEST_FOLDER), "--force", "--log-level", force_log]) != 0:
            _fail(f"stage {stage_main.__module__} crashed")
            return False
    return True


def load(name): return json.load((TEST_FOLDER / name).open(encoding="utf-8"))


# --- Conditions --------------------------------------------------------------

def cond_schema(m) -> bool:
    if m.get("schema_version") != 2:
        _fail(f"bad schema_version {m.get('schema_version')}")
        return False
    if not REQUIRED_TOP_KEYS <= set(m.keys()):
        _fail(f"missing top keys {REQUIRED_TOP_KEYS - set(m.keys())}")
        return False
    # every by_* count is a non-negative int (unwrap the v2 metric wrappers)
    for d in (_v(m["match"]["shot_mix"]["by_shot_type"]),
              _v(m["match"]["shot_mix"]["by_stroke_side"]),
              _v(m["match"]["by_end_reason"]), _v(m["error_attribution"]["by_owner"])):
        for k, v in d.items():
            if not isinstance(v, int) or v < 0:
                _fail(f"bad count {k}={v}")
                return False
    _pass("metrics.json valid: schema_version=2, all top keys present, counts >= 0")
    return True


def cond_shot_reconciliation(m) -> bool:
    role_sum = sum(_v(m["players"][r]["n_shots"]) for r in PLAYING_ROLES)
    unattr = _v(m["players"]["unattributed"]["n_shots"])
    total = _v(m["match"]["n_shots"])
    ok = role_sum + unattr == total
    (_pass if ok else _fail)(
        f"shot reconciliation: {role_sum} role + {unattr} unattributed == "
        f"{total} total" if ok else
        f"shot reconciliation FAILED: {role_sum}+{unattr} != {total}")
    return ok


def cond_error_reconciliation(m) -> bool:
    by_owner = _v(m["error_attribution"]["by_owner"])
    bad = [o for o in by_owner if o not in VALID_OWNERS]
    s = sum(by_owner.values())
    n = _v(m["match"]["n_rallies"])
    ok = (s == n) and not bad
    (_pass if ok else _fail)(
        f"error reconciliation: sum(by_owner)={s} == n_rallies={n}, owners valid"
        if ok else f"error reconciliation FAILED: sum={s} vs n_rallies={n}, "
        f"bad owners={bad}")
    return ok


def cond_end_reason_passthrough(m, rallies_doc) -> bool:
    expect = rallies_doc.get("stats", {}).get("by_end_reason", {})
    got = _v(m["match"]["by_end_reason"])
    ok = got == expect
    (_pass if ok else _fail)(
        f"by_end_reason passthrough matches Stage 7 exactly: {got}" if ok else
        f"by_end_reason mismatch: metrics={got} vs rallies={expect}")
    return ok


def cond_serve(m) -> bool:
    s = _v(m["match"]["serve"])
    faults = _v(m["match"]["by_end_reason"]).get("serve-fault", 0)
    ok = (s["n_serves"] == _v(m["match"]["n_rallies"])
          and s["n_serve_faults"] == faults
          and 0.0 <= s["serve_fault_rate"] <= 1.0)
    (_pass if ok else _fail)(
        f"serve: n_serves={s['n_serves']}==n_rallies, faults={s['n_serve_faults']}"
        f"==serve-fault bucket, rate={s['serve_fault_rate']}" if ok else
        f"serve metric inconsistent: {s}, faults bucket={faults}")
    return ok


def cond_rally_length(m, rallies_doc) -> bool:
    lengths = [int(r["n_shots"]) for r in rallies_doc.get("rallies", [])]
    rl = _v(m["match"]["rally_length_shots"])
    if not lengths:
        ok = rl["distribution"] == {} or sum(rl["distribution"].values()) == 0
        (_pass if ok else _fail)("rally length: empty handled")
        return ok
    exp_mean = round(statistics.mean(lengths), 3)
    exp_dist = {"1": 0, "2-4": 0, "5-8": 0, "9+": 0}
    for n in lengths:
        exp_dist[rally_len_bucket(n)] += 1
    ok = (abs(rl["mean"] - exp_mean) < 1e-2
          and rl["distribution"] == exp_dist
          and sum(rl["distribution"].values()) == len(lengths))
    (_pass if ok else _fail)(
        f"rally-length stats match recompute: mean={rl['mean']}, dist={rl['distribution']}"
        if ok else f"rally-length mismatch: got {rl}, expected mean={exp_mean} "
        f"dist={exp_dist}")
    return ok


def cond_position(m) -> bool:
    failures = []
    for r in PLAYING_ROLES:
        pos = _v(m["players"][r]["position"])
        if pos["n_frames"] == 0:
            continue
        for key in ("zone_time_frac", "lateral_time_frac", "area_time_frac"):
            vals = pos[key].values()
            if any(not (0.0 <= v <= 1.0) for v in vals):
                failures.append(f"{r}.{key} out of [0,1]")
            if abs(sum(vals) - 1.0) > 1e-3:
                failures.append(f"{r}.{key} sums to {sum(vals):.4f} != 1")
        # area marginals == zone
        for depth in ("kitchen", "transition", "baseline"):
            marg = sum(v for k, v in pos["area_time_frac"].items()
                       if k.startswith(depth + "-"))
            if abs(marg - pos["zone_time_frac"][depth]) > 1e-3:
                failures.append(f"{r} area-marginal {depth} {marg:.4f} != zone "
                                f"{pos['zone_time_frac'][depth]}")
        if not (0.0 <= pos["court_coverage_frac"] <= 1.0):
            failures.append(f"{r} coverage out of [0,1]")
        mvt = pos["movement"]
        if mvt["distance_ft_total"] < 0 or mvt["distance_ft_per_min"] < 0 \
                or mvt["distance_ft_per_rally"] < 0:
            failures.append(f"{r} negative movement")
    if _v(m["players"]["user"]["position"])["n_frames"] <= 0:
        failures.append("user position n_frames == 0 (clicks guarantee frames)")
    if failures:
        _fail(f"position stats: {len(failures)} issue(s); first 3: {failures[:3]}")
        return False
    _pass("position stats (real): fractions sum to 1, area==zone marginals, "
          "coverage/movement valid, user has frames")
    return True


def cond_team(m) -> bool:
    failures = []
    for side in ("near", "far"):
        t = _v(m["team"][side])
        if not (0.0 <= t["both_at_kitchen_frac"] <= 1.0):
            failures.append(f"{side} both_at_kitchen_frac out of [0,1]")
        sp = t["spacing_ft"]
        if t["n_frames_both_present"] > 0:
            if not (sp["min"] <= sp["median"] <= sp["max"] and sp["mean"] >= 0):
                failures.append(f"{side} spacing ordering bad: {sp}")
    if _v(m["team"]["near"])["n_frames_both_present"] <= 0:
        failures.append("near team n_frames_both_present == 0")
    if failures:
        _fail(f"team positioning: {failures[:3]}")
        return False
    _pass("team positioning (real): both_at_kitchen_frac/spacing valid, "
          "near team has common frames")
    return True


def cond_heatmaps(m) -> bool:
    """Recompute each role's in-extent foot-position count via the stage's own
    helpers; it must equal the player_position grid sum. And ball_landing sum
    must equal bounces with an in-extent court projection."""
    df = pd.read_parquet(TEST_FOLDER / "players.parquet")
    # mirror production: front foot (net-most ankle) where pose exists, bbox foot else
    poses_p = TEST_FOLDER / "poses.parquet"
    poses_df = pd.read_parquet(poses_p) if poses_p.exists() else None
    court = load("court.json")
    i2c = (court.get("homography", {}) or {}).get("image_to_court")
    pose_ff = pose_front_foot(poses_df, i2c) if (poses_df is not None and i2c) else {}
    rally_windows = [(int(x["start_frame"]), int(x["end_frame"]))
                     for x in load("rallies.json").get("rallies", [])]
    failures = []
    for r in PLAYING_ROLES:
        tids = m["players"][r]["track_ids"]
        sub = role_valid_rows(df, tids)
        fpos = role_front_foot_pos(role_frame_pos(sub), pose_ff, tids)
        fpos = scope_to_rally_frames(fpos, rally_windows)
        _, n_ext = bin_positions(list(fpos.values()))
        grid = _v(m["heatmaps"]["player_position"][r])
        gsum = sum(sum(row) for row in grid)
        if gsum != n_ext:
            failures.append(f"{r}: grid sum {gsum} != in-extent {n_ext}")
        if len(grid) != m["heatmaps"]["grid"]["n_rows"] or \
                any(len(row) != m["heatmaps"]["grid"]["n_cols"] for row in grid):
            failures.append(f"{r}: grid shape wrong")
    # ball landing
    bounces = load("bounces.json").get("bounces", [])
    def in_ext(b):
        xy = b.get("court_xy_ft")
        return xy and xy[0] is not None and 0 <= xy[0] < 20 and 0 <= xy[1] < 44
    exp_ball = sum(1 for b in bounces if in_ext(b))
    ball_sum = sum(sum(row) for row in _v(m["heatmaps"]["ball_landing"]))
    if ball_sum != exp_ball:
        failures.append(f"ball_landing sum {ball_sum} != in-extent bounces {exp_ball}")
    if failures:
        _fail(f"heatmap integrity: {failures[:3]}")
        return False
    _pass("heatmap integrity: all player grids + ball_landing reconcile with "
          "in-extent counts")
    return True


def cond_reliability(m) -> bool:
    rel = m["reliability"]
    ok = rel.get("synthetic_ball") is True
    ok = ok and any("synthetic" in w.lower() or "placeholder" in w.lower()
                    for w in m["warnings"])
    # real-data families NOT in synthetic_gated
    real = set(rel["real_data"])
    gated = set(rel["synthetic_gated"])
    for key in ("players.*.position", "heatmaps.player_position", "team.near"):
        if key not in real or key in gated:
            ok = False
    (_pass if ok else _fail)(
        "reliability: synthetic_ball=true + warning; position/team/player-heatmap "
        "are real_data, not synthetic_gated" if ok else
        "reliability misclassified real vs synthetic-gated families")
    return ok


def cond_pending(m) -> bool:
    pend = m["pending_real_ball"]
    listed = set(k.split(".", 1)[1] for k in m["reliability"]["pending"])
    keys = {k for k in pend if not k.startswith("_")}
    ok = keys == listed and len(keys) == 4
    for k in keys:
        e = pend[k]
        if e.get("value") is not None or e.get("status") != "pending_real_ball" \
                or not e.get("description"):
            ok = False
    (_pass if ok else _fail)(
        f"pending_real_ball: 4 entries all null + described + listed in "
        f"reliability.pending" if ok else
        f"pending block invalid: keys={keys} listed={listed}")
    return ok


def cond_contamination_flag(m) -> bool:
    floor = m["params"]["role_conf_floor"]
    ok = True
    flagged = []
    for r in PLAYING_ROLES:
        p = m["players"][r]
        if p["track_ids"] and p["role_confidence"] < floor:
            if not p["role_contaminated"]:
                ok = False
            else:
                flagged.append(r)
            if not any(r in w for w in m["warnings"]):
                ok = False
    (_pass if ok else _fail)(
        f"contamination flag: low-confidence roles {flagged} flagged + warned"
        if ok else "contamination flag/warning missing for a low-conf role")
    return ok


def cond_truth_tie(m, rallies_doc) -> bool:
    ok = _v(m["match"]["n_rallies"]) == len(rallies_doc.get("rallies", []))
    bio = _v(m["match"]["bounce_in_out"])
    bounces = load("bounces.json").get("bounces", [])
    proj = sum(1 for b in bounces if b.get("is_in_court") in (True, False))
    ok = ok and (bio["n_in"] + bio["n_out"] == proj)
    (_pass if ok else _fail)(
        f"truth tie: n_rallies matches Stage 7 ({_v(m['match']['n_rallies'])}); "
        f"bounce in+out ({bio['n_in']}+{bio['n_out']}) == projected bounces ({proj})"
        if ok else "truth tie failed")
    return ok


def cond_confidence(m) -> bool:
    """v2 wrapper integrity (assertion #13). On the synthetic clip we assert
    SHAPE + limited_by + the fixed constants — confidence VALUES are validated
    separately on real data."""
    failures = []
    reps = [m["match"]["n_shots"], m["match"]["rally_length_shots"],
            m["match"]["by_end_reason"], m["match"]["bounce_in_out"],
            m["match"]["shot_mix"]["by_shot_type"],
            m["error_attribution"]["by_owner"],
            m["players"]["user"]["position"],
            m["players"]["user"]["mean_post_speed_ftps"],
            m["heatmaps"]["ball_landing"]]
    for w in reps:
        if not _is_wrapped(w):
            failures.append(f"not wrapped: {str(w)[:48]}")
            continue
        if not (0.0 <= w["confidence"] <= 1.0):
            failures.append(f"confidence out of [0,1]: {w['confidence']}")
        if not isinstance(w["n"], int) or w["n"] < 0:
            failures.append(f"bad n: {w['n']}")
        if w["limited_by"] not in LIMITERS:
            failures.append(f"bad limited_by: {w['limited_by']}")
    # known-limit speed metric pinned to SPEED_CONF
    sp = m["players"]["user"]["mean_post_speed_ftps"]
    if _is_wrapped(sp):
        if sp["limited_by"] != "known_limit":
            failures.append(f"speed limited_by {sp['limited_by']} != known_limit")
        if abs(sp["confidence"] - m["params"]["speed_conf"]) > 1e-9:
            failures.append(f"speed confidence {sp['confidence']} != SPEED_CONF")
    # structural census metric
    if m["match"]["match_span_sec"]["limited_by"] != "detection_floor":
        failures.append("match_span_sec limited_by != detection_floor")
    # recall-blind-spot banner present
    if not any("recall" in w.lower() for w in m["warnings"]):
        failures.append("recall-blind-spot banner missing from warnings")
    if failures:
        _fail(f"confidence wrappers: {len(failures)} issue(s); first 3: {failures[:3]}")
        return False
    _pass("confidence wrappers (v2): shape + limited_by + SPEED_CONF + "
          "detection_floor + recall banner all valid")
    return True


def cond_degradation() -> bool:
    """Hide track_roles.json -> stage falls back to user-only, no crash."""
    tr = TEST_FOLDER / "track_roles.json"
    backup = TEST_FOLDER / "track_roles.json.bak"
    if not tr.exists():
        _fail("track_roles.json missing before degradation test")
        return False
    tr.rename(backup)
    try:
        rc = metrics_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"])
        if rc != 0:
            _fail("degradation: stage crashed without track_roles.json")
            return False
        m = load("metrics.json")
        ok = (_v(m["players"]["user"]["n_shots"]) >= 0
              and _v(m["players"]["partner"]["n_shots"]) == 0
              and _v(m["players"]["opp_a"]["n_shots"]) == 0
              and m["sources"]["track_roles"] is None
              and any("track_roles" in w for w in m["warnings"]))
        (_pass if ok else _fail)(
            "degradation: user-only fallback, opponents empty, warned, no crash"
            if ok else "degradation fallback did not behave as specified")
        return ok
    finally:
        backup.rename(tr)
        # restore the full-role metrics.json for any later inspection
        metrics_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"])


# --- Runner ------------------------------------------------------------------

def run_smoke_test() -> int:
    print(f"Stage 8 smoke test - fixture: {TEST_FOLDER}")
    print()
    if not check_fixtures():
        return 1
    for stale in ("metrics.json",):
        p = TEST_FOLDER / stale
        if p.exists():
            p.unlink()

    # Roles are ball-independent; build once up front.
    if roles_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
        _fail("classify_tracks (Stage 2.5) crashed")
        return 1

    results = []

    # --- Phase A: gap variant must not crash ---
    print(f"Phase A: gap variant (--gap-frac {GAP_FRAC})")
    if not gen_ball(GAP_FRAC):
        return 1
    ok_gap = run_chain() and (TEST_FOLDER / "metrics.json").exists()
    (_pass if ok_gap else _fail)("gap variant completed without crash")
    results.append(ok_gap)
    print()

    # --- Phase B: clean variant graded ---
    print("Phase B: clean variant")
    if not gen_ball(0.0):
        return 1
    if not run_chain():
        return 1

    m = load("metrics.json")
    rallies_doc = load("rallies.json")

    print("Checking conditions:")
    results.append(cond_schema(m))
    results.append(cond_shot_reconciliation(m))
    results.append(cond_error_reconciliation(m))
    results.append(cond_end_reason_passthrough(m, rallies_doc))
    results.append(cond_serve(m))
    results.append(cond_rally_length(m, rallies_doc))
    results.append(cond_position(m))
    results.append(cond_team(m))
    results.append(cond_heatmaps(m))
    results.append(cond_reliability(m))
    results.append(cond_pending(m))
    results.append(cond_contamination_flag(m))
    results.append(cond_truth_tie(m, rallies_doc))
    results.append(cond_confidence(m))
    results.append(cond_degradation())

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())

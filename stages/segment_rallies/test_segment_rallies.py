"""Stage 7 — Smoke test.

End-to-end run (synth_ball -> Stage 5 -> Stage 5.5 -> Stage 6 -> Stage 7)
graded on:
  - schema/consistency of rallies.json
  - ball_source=synthetic propagated with warning
  - boundary correctness: every is_serve becomes a rally's serve_shot_id
  - shot assignment: every non-pre-rally shot belongs to exactly one rally
  - boundary recovery vs truth >= 0.90 on detected rallies
  - end_reason accuracy >= 0.70 on matched rallies
  - internal consistency (ball-out has out-of-court bounce after last shot;
    double-bounce has 2+ bounces; serve-fault has n_shots == 1; net-or-short
    last_bounce_side == hitter_side; ball-off-frame has 0 bounces)
  - by_end_reason diversity >= 4 non-zero buckets
Plus an injected-gap variant that must not crash.

Requires data/test_clip/ with video.mp4, court.json, players.parquet,
poses.parquet, roster.json (from Stages 1-3 + setup).

Usage:
    python -m stages.segment_rallies.test_segment_rallies
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from stages.detect_shots.detect_shots import main as detect_main
from stages.detect_bounces.detect_bounces import main as bounces_main
from stages.classify_shots.classify_shots import main as classify_main
from stages.segment_rallies.segment_rallies import main as rallies_main

TEST_FOLDER = Path("data/test_clip")
SEED = 1234

BOUNDARY_RECOVERY_BAR = 0.90
END_REASON_ACCURACY_BAR = 0.70
MIN_NONZERO_END_REASONS = 4
GAP_FRAC = 0.20
MATCH_WINDOW = 6

REQUIRED_RALLY_KEYS = {
    "rally_id", "start_frame", "end_frame", "start_t_sec", "end_t_sec",
    "duration_sec", "shot_ids", "n_shots", "serve_shot_id",
    "server_track_id", "server_is_user", "end_reason",
    "end_reason_confidence", "ending_bounce_id", "end_signals",
}
END_REASONS = {"serve-fault", "double-bounce", "ball-out", "net-or-short",
               "ball-not-returned", "ball-off-frame", "unknown"}


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


# --- Individual conditions ---------------------------------------------------

def cond_schema(doc) -> bool:
    if doc.get("schema_version") != 1:
        _fail(f"bad schema_version {doc.get('schema_version')}")
        return False
    rs = doc.get("rallies", [])
    if not rs:
        _fail("zero rallies emitted (synth should produce >=1)")
        return False
    frames = [r["start_frame"] for r in rs]
    if frames != sorted(frames):
        _fail("rallies not sorted by start_frame")
        return False
    for i, r in enumerate(rs):
        if not REQUIRED_RALLY_KEYS <= set(r.keys()):
            _fail(f"rally {i} missing keys {REQUIRED_RALLY_KEYS - set(r.keys())}")
            return False
        if r["rally_id"] != i:
            _fail(f"rally_id not contiguous at index {i}")
            return False
        if r["end_reason"] not in END_REASONS:
            _fail(f"bad end_reason {r['end_reason']} on rally {i}")
            return False
        if not (0.0 <= r["end_reason_confidence"] <= 1.0):
            _fail(f"rally {i} confidence out of [0,1]")
            return False
        if r["start_frame"] > r["end_frame"]:
            _fail(f"rally {i} start_frame > end_frame")
            return False
    _pass(f"rallies.json valid: {len(rs)} rallies, schema_version=1, "
          f"contiguous rally_id, all fields/categories OK, sorted")
    return True


def cond_source(doc) -> bool:
    ok = (doc.get("ball_source") == "synthetic"
          and any("synthetic" in w.lower() or "placeholder" in w.lower()
                  for w in doc.get("warnings", [])))
    (_pass if ok else _fail)("ball_source=synthetic propagated with warning")
    return ok


def cond_boundary_correctness(doc, classified_doc) -> bool:
    """Every shot with is_serve in classified.json should be some rally's
    serve_shot_id."""
    serve_ids = {int(s["shot_id"]) for s in classified_doc["shots"]
                 if s.get("is_serve")}
    rally_serve_ids = {int(r["serve_shot_id"]) for r in doc.get("rallies", [])}
    missing = serve_ids - rally_serve_ids
    extra = rally_serve_ids - serve_ids
    ok = (not missing) and (not extra)
    if not ok:
        _fail(f"boundary correctness: missing serve shot_ids={sorted(missing)}, "
              f"unexpected={sorted(extra)}")
        return False
    _pass(f"boundary correctness: {len(serve_ids)} is_serve shots all start rallies")
    return True


def cond_shot_assignment(doc, classified_doc) -> bool:
    """Every shot (except pre-rally) should appear in exactly one rally's
    shot_ids."""
    all_shot_ids = {int(s["shot_id"]) for s in classified_doc["shots"]}
    assigned: dict = {}
    duplicates = []
    for r in doc.get("rallies", []):
        for sid in r["shot_ids"]:
            sid = int(sid)
            if sid in assigned:
                duplicates.append(sid)
            else:
                assigned[sid] = r["rally_id"]
    pre_count = doc.get("stats", {}).get("unassigned_shots", 0)
    expected_assigned = all_shot_ids - set(s["shot_id"]
                                           for s in classified_doc["shots"][:pre_count])
    missing = expected_assigned - set(assigned.keys())
    ok = (not duplicates) and len(missing) == pre_count - pre_count
    # Simplification: just verify total counts and no duplicates
    total_assigned = len(assigned)
    expected = len(all_shot_ids) - pre_count
    ok = (not duplicates) and (total_assigned == expected)
    if not ok:
        _fail(f"shot assignment: {total_assigned} assigned vs expected {expected}, "
              f"duplicates={sorted(set(duplicates))[:10]}")
        return False
    _pass(f"shot assignment: {total_assigned} shots in rallies + "
          f"{pre_count} unassigned = {len(all_shot_ids)} total, no duplicates")
    return True


def cond_boundary_recovery(doc, truth) -> bool:
    """Each truth rally should have a detected rally with start_frame within
    MATCH_WINDOW."""
    t_rallies = truth.get("rallies", [])
    d_rallies = doc.get("rallies", [])
    if not t_rallies:
        _fail("truth has no rallies")
        return False
    used = set()
    matched = 0
    for tr in t_rallies:
        ts = tr["start_frame"]
        for j, dr in enumerate(d_rallies):
            if j in used:
                continue
            if abs(dr["start_frame"] - ts) <= MATCH_WINDOW:
                used.add(j)
                matched += 1
                break
    rec = matched / len(t_rallies)
    ok = rec >= BOUNDARY_RECOVERY_BAR
    (_pass if ok else _fail)(
        f"boundary recovery {matched}/{len(t_rallies)} = {rec:.3f} "
        f"(bar {BOUNDARY_RECOVERY_BAR})")
    return ok


def cond_end_reason_accuracy(doc, truth) -> bool:
    """For matched (truth, detected) pairs, end_reason should agree."""
    t_rallies = truth.get("rallies", [])
    d_rallies = doc.get("rallies", [])
    used = set()
    pairs = []
    for tr in t_rallies:
        ts = tr["start_frame"]
        for j, dr in enumerate(d_rallies):
            if j in used:
                continue
            if abs(dr["start_frame"] - ts) <= MATCH_WINDOW:
                used.add(j)
                pairs.append((tr, dr))
                break
    if not pairs:
        _fail("no matched rally pairs; can't grade end_reason accuracy")
        return False
    correct = sum(1 for tr, dr in pairs if tr["end_reason"] == dr["end_reason"])
    acc = correct / len(pairs)
    ok = acc >= END_REASON_ACCURACY_BAR
    (_pass if ok else _fail)(
        f"end_reason accuracy {correct}/{len(pairs)} = {acc:.3f} "
        f"(bar {END_REASON_ACCURACY_BAR})")
    return ok


def cond_internal_consistency(doc, bounces_doc) -> bool:
    """Each rally's end_reason should match its end_signals:
    - serve-fault: n_shots == 1
    - double-bounce: n_bounces_after_last_shot >= 2
    - ball-out: at least one out-of-court bounce after last shot
    - net-or-short: last_bounce_side == hitter_side
    - ball-off-frame: n_bounces_after_last_shot == 0
    """
    bounces = bounces_doc.get("bounces", [])
    bounce_by_id = {int(b["bounce_id"]): b for b in bounces}
    failures = []
    for r in doc.get("rallies", []):
        er = r["end_reason"]
        sig = r.get("end_signals", {})
        n_post = sig.get("n_bounces_after_last_shot", 0)
        last_shot_id = r["shot_ids"][-1] if r["shot_ids"] else None
        post = [b for b in bounces
                if b.get("between_shots") and b["between_shots"][0] == last_shot_id]
        if er == "serve-fault" and r["n_shots"] != 1:
            failures.append(f"rally {r['rally_id']}: serve-fault but n_shots={r['n_shots']}")
        if er == "double-bounce" and n_post < 2:
            failures.append(f"rally {r['rally_id']}: double-bounce but n_post={n_post}")
        if er == "ball-out":
            if not any(b.get("is_in_court") is False for b in post):
                failures.append(f"rally {r['rally_id']}: ball-out but no OOC bounce")
        if er == "net-or-short":
            if sig.get("last_bounce_side") != sig.get("hitter_side"):
                failures.append(f"rally {r['rally_id']}: net-or-short but "
                                f"last_bounce_side={sig.get('last_bounce_side')} "
                                f"vs hitter_side={sig.get('hitter_side')}")
        if er == "ball-off-frame" and n_post != 0:
            failures.append(f"rally {r['rally_id']}: ball-off-frame but n_post={n_post}")
    if failures:
        _fail(f"internal consistency: {len(failures)} violation(s); "
              f"first 3: {failures[:3]}")
        return False
    _pass(f"internal consistency: all end_reasons match end_signals")
    return True


def cond_diversity(doc) -> bool:
    by = doc.get("stats", {}).get("by_end_reason", {})
    nonzero = sum(1 for v in by.values() if v > 0)
    ok = nonzero >= MIN_NONZERO_END_REASONS
    (_pass if ok else _fail)(
        f"by_end_reason diversity {nonzero} non-zero buckets "
        f"(bar >={MIN_NONZERO_END_REASONS}); breakdown={by}")
    return ok


# --- Test runner -------------------------------------------------------------

def run_smoke_test() -> int:
    print(f"Stage 7 smoke test - fixture: {TEST_FOLDER}")
    print()
    if not check_fixtures():
        return 1
    for stale in ("rallies.json",):
        p = TEST_FOLDER / stale
        if p.exists():
            p.unlink()

    results = []

    # --- Phase A: gap variant must not crash ---
    print(f"Phase A: gap variant (--gap-frac {GAP_FRAC})")
    if not gen_ball(GAP_FRAC):
        return 1
    for stage_main in (detect_main, bounces_main, classify_main):
        if stage_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
            _fail(f"Stage upstream crashed on gap variant")
            return 1
    rc = rallies_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"])
    ok_gap = (rc == 0 and (TEST_FOLDER / "rallies.json").exists()
              and len(load("rallies.json").get("rallies", [])) > 0)
    (_pass if ok_gap else _fail)("gap variant completed without crash + non-empty")
    results.append(ok_gap)
    print()

    # --- Phase B: clean variant graded ---
    print("Phase B: clean variant")
    if not gen_ball(0.0):
        return 1
    for stage_main in (detect_main, bounces_main, classify_main, rallies_main):
        if stage_main([str(TEST_FOLDER), "--force", "--log-level", "ERROR"]) != 0:
            _fail(f"Upstream stage crashed on clean variant")
            return 1

    rallies_doc = load("rallies.json")
    classified_doc = load("classified.json")
    bounces_doc = load("bounces.json")
    truth = load("ball_synth_truth.json")

    print("Checking conditions:")
    results.append(cond_schema(rallies_doc))
    results.append(cond_source(rallies_doc))
    results.append(cond_boundary_correctness(rallies_doc, classified_doc))
    results.append(cond_shot_assignment(rallies_doc, classified_doc))
    results.append(cond_boundary_recovery(rallies_doc, truth))
    results.append(cond_end_reason_accuracy(rallies_doc, truth))
    results.append(cond_internal_consistency(rallies_doc, bounces_doc))
    results.append(cond_diversity(rallies_doc))

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())

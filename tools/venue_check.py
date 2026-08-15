"""Decide whether a venue is measurable well enough to trust.

Contract D4: a venue whose measurement quality is materially worse is excluded from a
collection rather than merged, because a number that describes neither venue is worse
than a missing one. Deciding that needs recall, which a new unlabelled venue does not
have — so it is predicted from the unlabelled signals in tools.venue_signals.

WHAT THIS SHIPS, AND WHAT IT DOES NOT
-------------------------------------
This was designed as a three-bucket call (supported / marginal / not_supported) validated
leave-one-venue-out. **That validation is impossible on the data we have**, and the reason
is structural rather than a tuning problem: of four venues, exactly one is supported and
exactly one is not_supported. Hold out the failing venue and the training set contains
nothing that fails, so the not_supported boundary is undefined — it fits to -inf and
admits everything. You cannot learn a boundary from a set with nothing on one side of it.

So the three-bucket rule is NOT shipped. What ships is the narrow claim the data does
support: a BINARY gate that catches a venue the detector effectively cannot see.

    p90_conf   how confident the detector is when it is most confident. Preferred over
               det_rate because it is robust to how much DEAD TIME a clip contains — a
               clip with long gaps between rallies has a low detection rate without being
               a bad venue.
    det_rate   corroboration, never on its own.

Measured separation, per clip (the gate runs on one clip, so per-clip is what matters):

    p90_conf   worst working 0.762   vs failing 0.216   ->  3.5x
    det_rate   worst working 0.237   vs failing 0.025   ->  9.5x

Thresholds sit at the geometric midpoint of those gaps. A clip is refused if EITHER
signal falls below its threshold: wrongly refusing a venue is an inconvenience, wrongly
admitting one silently contaminates every cumulative number.

HONEST LIMITS — read before trusting a refusal
  - Calibrated on exactly ONE failing venue. It detects "looks like pb_3min_indoor",
    which may not cover every way a venue can be hard.
  - Ranking is perfect on n=4 venues (Spearman +1.000 for det_rate / mean_conf / p90_conf
    against true recall), but n=4 is n=4.
  - It does NOT predict recall and must not be reported as if it did.
  - Thresholds are tied to the weights they were fitted with. Re-run --validate after any
    model change.
  - The middle ground is unvalidated: court2 measures 0.533 recall locally and is
    ADMITTED. Being admitted is not a promise of quality.

Usage:
    python -m tools.venue_check --validate
    python -m tools.venue_check data/some_new_clip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIGNALS = ("det_rate", "mean_conf", "p90_conf", "contrast", "continuity")
SUPPORTED, MARGINAL = 0.80, 0.50

# Geometric midpoints of the measured PER-CLIP gaps between the worst working clip and
# the failing one. Geometric rather than arithmetic because these behave like ratios, so
# the threshold sits clear of both ends by the same factor instead of hugging the larger
# value.
#
# REFITTED 2026-08-15 against the DEPLOYED weights (Run 3). Thresholds are tied to the
# model that produced the signals, so this must be redone on every model change.
#     p90_conf   worst working 0.841 (pb_3min)  vs failing 0.263  ->  3.2x
#     det_rate   worst working 0.550 (court2)   vs failing 0.060  ->  9.2x
# The previous values (0.406 / 0.077, fitted to the old baseline) would still have
# classified all eight clips correctly, so this is a margin improvement, not a fix.
T_P90, T_DET = 0.470, 0.182

# Motion guard — CONDITIONAL, and unvalidated. It targets the one failure mode the
# confidence signals structurally cannot see: a detector locked onto something STATIC
# (a yellow shirt, a cone, a ball on the next court), where confidence stays high while
# recall collapses.
#
# It must only be applied where motion is actually measurable. Measured values:
#     clip              recall  det_rate  n_steps  moving_frac
#     pb_3min_indoor     0.050      0.03        1         1.00
#     pb_3min_court2     0.533      0.24       11         1.00
#     indoor_C1          0.750      0.55       32         0.97
#     pb_2min            0.933      0.66       38         0.61
#
# moving_frac is ANTI-correlated with quality here, so it is NOT a quality predictor and
# must never be used as one. Two reasons, both real: measuring motion needs CONSECUTIVE
# detections, which a failing venue barely produces (n_steps=1 on pb_3min_indoor — one
# sample), and on a good venue the ball genuinely sits still a lot (held between points,
# resting on court), which drags moving_frac DOWN. The original reasoning — "a real ball
# flies, noise teleports" — is backwards in the sparse-detection regime.
#
# A static lock-on, by contrast, produces MANY confident detections, so n_steps is large
# and moving_frac collapses toward 0. Hence: only judge when there is enough motion data,
# and set the bar far below the lowest healthy venue (pb_2min at 0.61).
MIN_STEPS_FOR_MOTION = 20     # below this, motion is not measurable; do not judge on it
T_MOVE = 0.20                 # ~3x clear of the lowest healthy venue

# Which clips are the same physical venue. Four venues, not eight clips — the unit of
# generalisation is the VENUE.
VENUE_OF = {
    "pb_2min": "home", "pb_3min": "home", "pb_4min": "home", "pb_5min": "home",
    "pb_3min_court2": "court2",
    "indoor_B1_3min": "indoor_1", "indoor_C1_3min": "indoor_1",
    "pb_3min_indoor": "indoor_2",
}


def load(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if "error" not in v and v.get("recall") is not None}


def by_venue(rows: dict) -> dict:
    """Venue -> mean of its clips' signals. Averaged so home's four clips do not outvote
    court2's one."""
    groups: dict[str, list] = {}
    for clip, r in rows.items():
        groups.setdefault(VENUE_OF.get(clip, clip), []).append(r)
    agg = {}
    for ven, rs in groups.items():
        agg[ven] = {s: float(np.mean([r[s] for r in rs if r.get(s) is not None]))
                    for s in SIGNALS if any(r.get(s) is not None for r in rs)}
        agg[ven]["recall"] = float(np.mean([r["recall"] for r in rs]))
        agg[ven]["n_clips"] = len(rs)
    return agg


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if len(a) < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def score(r: dict) -> tuple[str, str]:
    """(verdict, reason) for one clip. Binary by design — see the module docstring."""
    p90, det = r.get("p90_conf"), r.get("det_rate")
    if p90 is None or det is None:
        return "unknown", "signals missing"
    if p90 < T_P90:
        return "not_supported", f"p90_conf {p90:.3f} < {T_P90:.3f}"
    if det < T_DET:
        return "not_supported", f"det_rate {det:.3f} < {T_DET:.3f}"
    # Conditional motion guard: only where motion is measurable at all. Unvalidated —
    # no venue in the calibration set fails this way, so it has never fired in anger.
    mv, ns = r.get("moving_frac"), r.get("n_steps", 0)
    if mv is not None and ns >= MIN_STEPS_FOR_MOTION and mv < T_MOVE:
        return "not_supported", (f"moving_frac {mv:.2f} < {T_MOVE:.2f} over {ns} steps — "
                                 f"detections look parked, not ball-like")
    return "ok", f"p90_conf {p90:.3f}, det_rate {det:.3f}"


def nearest_known(r: dict, rows: dict) -> str:
    """Which calibration clip this most resembles, so an 'ok' carries context instead of
    sounding like a quality guarantee."""
    best, bd = None, 1e9
    for c, k in rows.items():
        if k.get("p90_conf") is None:
            continue
        d = abs(k["p90_conf"] - r["p90_conf"]) + abs(k["det_rate"] - r["det_rate"])
        if d < bd:
            best, bd = c, d
    return "" if best is None else (
        f"closest calibration clip: {best} (measured recall {rows[best]['recall']:.2f})")


def validate(rows: dict) -> int:
    venues = by_venue(rows)
    print(f"{len(rows)} labelled clips across {len(venues)} venues\n")
    print(f"{'venue':<10}{'clips':>6}{'recall':>8}   " + "".join(f"{s:>11}" for s in SIGNALS))
    for ven, v in sorted(venues.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"{ven:<10}{v['n_clips']:>6}{v['recall']:>8.3f}   " +
              "".join(f"{v[s]:>11.3f}" if s in v else f"{'-':>11}" for s in SIGNALS))

    print(f"\nrank agreement with true recall (venue level, n={len(venues)}):")
    for s, r in sorted(((s, spearman([v[s] for v in venues.values() if s in v],
                                     [v["recall"] for v in venues.values() if s in v]))
                        for s in SIGNALS), key=lambda kv: -abs(kv[1])):
        print(f"   {s:<12}{r:+.3f}")

    print("\nleave-one-venue-out is NOT reported: of 4 venues exactly 1 is supported and")
    print("exactly 1 is not_supported, so holding either out deletes that bucket from the")
    print("fit and its boundary becomes undefined. Structural, not a tuning problem.\n")

    print("BINARY GATE on every calibration clip (must refuse the failing venue only):")
    bad_admitted, good_refused = [], []
    for c, r in sorted(rows.items(), key=lambda kv: kv[1]["p90_conf"]):
        verdict, why = score(r)
        truth_bad = r["recall"] < MARGINAL
        flag = ""
        if truth_bad and verdict != "not_supported":
            bad_admitted.append(c); flag = "   <-- ADMITTED A BAD VENUE"
        elif not truth_bad and verdict == "not_supported":
            good_refused.append(c); flag = "   <-- refused a usable venue"
        print(f"   {c:<20}recall {r['recall']:.3f} -> {verdict:<14} {why}{flag}")

    if bad_admitted:
        print(f"\nFAILS ITS PURPOSE: {bad_admitted} would contaminate a collection")
        return 1
    print(f"\nPASSES on the calibration set: 0 bad venues admitted, "
          f"{len(good_refused)} usable venues refused.")
    print("Necessary, not sufficient — the gate was fitted on this same set. It earns")
    print("real confidence only when a NEW failing venue is measured against it.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=Path)
    ap.add_argument("--signals", type=Path, default=Path("data/_venue_signals.json"))
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args(argv)

    if not a.signals.exists():
        print(f"no {a.signals}; run tools.venue_signals first")
        return 1
    rows = load(a.signals)
    if a.validate or a.folder is None:
        return validate(rows)

    import subprocess
    import sys as _sys
    tmp = Path("data/_venue_one.json")
    subprocess.run([_sys.executable, "-m", "tools.venue_signals", str(a.folder),
                    "--out", str(tmp)], check=True)
    r = json.loads(tmp.read_text(encoding="utf-8"))[a.folder.name]
    verdict, why = score(r)
    print(f"\n{a.folder.name}: {verdict.upper()}  ({why})")
    ctx = nearest_known(r, rows)
    if ctx:
        print(f"   {ctx}")
    if verdict == "not_supported":
        print("   -> excluded from collections (contract D4). This means the ball is not")
        print("      reliably visible to the detector here, NOT that the video is bad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

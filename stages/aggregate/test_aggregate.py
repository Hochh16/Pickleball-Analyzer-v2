"""Stage 7.9 — acceptance test for the union.

The operator's specification is "work the same as if the multiple videos were all in one
video". This tests exactly that, with no new labelling:

  1. Take a real analysed clip and PARTITION its detected streams at a rally boundary
     into two pseudo-sessions.
  2. Union them back with Stage 7.9.
  3. Run Stage 8 on the union. Every count must equal the whole clip's.

Why partition the STREAMS rather than re-run detection on two half-videos (which is what
the contract first proposed): cutting a clip also cuts players.parquet mid-track, so
tracks lose history in a way genuinely separate videos never would — a real second video
is tracked from scratch. Measured, that route collapsed 34 detected shots to 13 across
the halves. That conflates two questions and only one is this stage's job. Aggregation
must lose nothing when streams are recombined; detector stability under truncated input
is a separate question, recorded as untested rather than silently folded in here.

The rally boundary still matters: partition mid-rally and one rally becomes two, which
would be a property of the cut, not an aggregation bug.

Counts and distributions must match EXACTLY. Time-derived values are not compared — the
union sums member spans by design, so it differs from the whole clip's wall-clock.

Usage:
    python -m stages.aggregate.test_aggregate
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

CLIP = Path("data/pb_2min")
WORK = Path("data/_agg_test")
CARRY = ["court.json", "court_zones.json", "roster.json", "ball.meta.json",
         "track_roles.json"]


def run(mod: str, folder: Path) -> bool:
    r = subprocess.run([sys.executable, "-m", mod, str(folder), "--force",
                        "--log-level", "ERROR"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL: {mod} on {folder.name}: {r.stderr.strip()[-300:]}")
    return r.returncode == 0


def cut_frame(rallies: list) -> int:
    """A frame strictly between two rallies — the only lossless place to partition."""
    if len(rallies) < 4:
        raise SystemExit(f"need >= 4 rallies to partition; got {len(rallies)}")
    mid = len(rallies) // 2
    return int((rallies[mid - 1]["end_frame"] + rallies[mid]["start_frame"]) // 2)


def write_part(clip: Path, dest: Path, docs: dict, lo: int, hi: int) -> dict:
    """One pseudo-session holding every stream item whose frame falls in [lo, hi)."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in CARRY:
        if (clip / f).exists():
            shutil.copy(clip / f, dest / f)

    shots = [dict(s) for s in docs["classified"]["shots"] if lo <= s["frame"] < hi]
    keep = {int(s["shot_id"]) for s in shots}
    rallies = [dict(r) for r in docs["rallies"]["rallies"] if lo <= r["start_frame"] < hi]
    bounces = [dict(b) for b in docs["bounces"]["bounces"] if lo <= b["frame"] < hi]
    # A bounce's shot references must stay inside this part, or the union's integrity
    # check would rightly reject a dangling reference.
    for b in bounces:
        b["between_shots"] = [x if (x is not None and int(x) in keep) else None
                              for x in b["between_shots"]]

    for name, key, rows in (("classified", "shots", shots),
                            ("rallies", "rallies", rallies),
                            ("bounces", "bounces", bounces)):
        d = dict(docs[name])
        d[key] = rows
        (dest / f"{name}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")

    pl = docs["players"]
    pl[(pl["frame"] >= lo) & (pl["frame"] < hi)].to_parquet(dest / "players.parquet",
                                                            index=False)
    return {"shots": len(shots), "rallies": len(rallies), "bounces": len(bounces)}


def counts(folder: Path) -> dict:
    m = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["match"]
    out = {k: m[k]["value"] for k in ("n_shots", "n_rallies", "n_bounces")}
    out["rally_len_dist"] = m["rally_length_shots"]["value"]["distribution"]
    out["shot_mix"] = m["shot_mix"]["by_shot_type"]["value"]
    return out


def main() -> int:
    print("Stage 7.9 acceptance test - partition at a rally boundary, union, compare")
    print()
    if not (CLIP / "rallies.json").exists():
        print(f"missing {CLIP}/rallies.json - run the pipeline on it first")
        return 1
    if WORK.exists():
        shutil.rmtree(WORK)

    docs = {n: json.loads((CLIP / f"{n}.json").read_text(encoding="utf-8"))
            for n in ("classified", "rallies", "bounces")}
    docs["players"] = pd.read_parquet(CLIP / "players.parquet")
    rallies = docs["rallies"]["rallies"]
    cut = cut_frame(rallies)
    end = int(max(s["frame"] for s in docs["classified"]["shots"])) + 10**6
    print(f"{CLIP.name}: {len(docs['classified']['shots'])} shots, "
          f"{len(rallies)} rallies - partitioning at frame {cut}")

    a, b = WORK / "part_a", WORK / "part_b"
    ca = write_part(CLIP, a, docs, 0, cut)
    cb = write_part(CLIP, b, docs, cut, end)
    print(f"  part_a: {ca}")
    print(f"  part_b: {cb}")
    if ca["shots"] + cb["shots"] != len(docs["classified"]["shots"]):
        print("  FAIL: the partition itself dropped shots, before aggregation ran")
        return 1

    u = WORK / "union"
    r = subprocess.run([sys.executable, "-m", "stages.aggregate.aggregate",
                        "--member", str(a), "--member", str(b), "--out", str(u),
                        "--log-level", "ERROR"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL: aggregate: {r.stderr.strip()[-400:]}")
        return 1
    for folder in (CLIP, u):
        if not run("stages.compute_metrics.compute_metrics", folder):
            return 1

    whole, union = counts(CLIP), counts(u)
    print()
    print("%-16s %12s %12s" % ("metric", "whole", "union"))
    ok = True
    for k in ("n_shots", "n_rallies", "n_bounces"):
        hit = whole[k] == union[k]
        ok &= hit
        print("%-16s %12s %12s   %s" % (k, whole[k], union[k], "OK" if hit else "MISMATCH"))
    for k in ("rally_len_dist", "shot_mix"):
        hit = whole[k] == union[k]
        ok &= hit
        print("%-16s %12s %12s   %s" % (k, "", "", "OK" if hit else "MISMATCH"))
        if not hit:
            print(f"    whole: {whole[k]}")
            print(f"    union: {union[k]}")

    print()
    print("PASS - the union reproduces the whole clip exactly" if ok
          else "FAIL - union does not match the whole clip")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

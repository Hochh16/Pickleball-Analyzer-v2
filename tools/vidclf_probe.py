r"""Phase 1 of the video-classifier experiment: is there signal, and is data the limit?

See docs/VIDEO_CLASSIFIER_SCOPE.md. A FROZEN pretrained video model turns each candidate clip
(tools/vidclf_build_clips.py) into a vector; only a logistic regression is trained on top. That
is the regime where hundreds of examples can be enough, and it is what the learning curve below
is for: it says whether more labelled video would help or whether the pixels just do not carry
the answer.

Backbone: torchvision S3D, Kinetics-400 weights. Chosen for a BSD-licensed code path (a shipped
app cannot depend on a non-commercial model) and because it runs at ~1.6 s per clip on this
machine's CPU -- ~1 hour for the whole set, no GPU needed.

Everything is leave-one-VIDEO-out over the three dev videos. Court A's clips are embedded but
never used to train or to choose anything; `--heldout` scores it, once, with a threshold chosen
on dev.

    python -m tools.vidclf_probe embed          # cached; safe to stop and resume
    python -m tools.vidclf_probe probe          # dev only: AUC, learning curve, found/junk
    python -m tools.vidclf_probe probe --heldout
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vidclf_build_clips import DEV, HELD_OUT, MATCH_S, OUT, one_to_one  # noqa: E402

EMB = OUT / "embeddings_s3d.npz"
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SEEDS = 5
NMS_S = 0.20
THRESHOLDS = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]


def manifest() -> List[dict]:
    with (OUT / "manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["label"] = int(r["label"])
        r["t_sec"] = float(r["t_sec"])
    return rows


# --- embedding -------------------------------------------------------------------------------
def embed(batch: int = 8) -> None:
    import cv2
    import torch
    from torchvision.models.video import S3D_Weights, s3d

    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = manifest()
    done: Dict[str, np.ndarray] = {}
    if EMB.exists():
        z = np.load(EMB, allow_pickle=False)
        done = dict(zip(z["ids"].tolist(), z["x"]))
    todo = [r for r in rows if r["clip_id"] not in done]
    print(f"{len(rows)} clips, {len(done)} already embedded, {len(todo)} to go")
    if not todo:
        return
    weights = S3D_Weights.KINETICS400_V1
    model = s3d(weights=weights).eval()
    tf = weights.transforms()

    def features(x):
        # S3D: features -> avgpool -> (dropout, 1x1x1 conv classifier). Stop before the
        # classifier: the 1024-d pooled vector is the representation being probed.
        # Mean over whatever time/space remains, as torchvision's own forward() does after
        # the classifier, so a clip of a different length cannot change the vector's shape.
        return model.avgpool(model.features(x)).mean(dim=(2, 3, 4))

    def load(r):
        cap = cv2.VideoCapture(str(OUT / r["video"] / f"{r['clip_id']}.mp4"))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cap.release()
        v = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)   # T,C,H,W uint8
        return tf(v)                                                  # C,T,H,W float

    t0 = time.time()
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        x = torch.stack([load(r) for r in chunk])
        with torch.inference_mode():
            f = features(x).numpy().astype(np.float32)
        for r, v in zip(chunk, f):
            done[r["clip_id"]] = v
        if (k // batch) % 10 == 0 or k + batch >= len(todo):
            ids = np.array(list(done), dtype="U80")
            np.savez(EMB, ids=ids, x=np.stack([done[i] for i in ids]))
            rate = (time.time() - t0) / (k + len(chunk))
            print(f"  {k + len(chunk)}/{len(todo)}  {rate:.2f} s/clip  "
                  f"~{rate * (len(todo) - k - len(chunk)) / 60:.0f} min left", flush=True)


# --- probing ---------------------------------------------------------------------------------
def truth_times(video: str) -> List[float]:
    from tools.truth_store import known
    return sorted(float(s["t_sec"]) for s in known(Path("data") / video)["shots"]
                  if not s.get("not_a_shot"))


def shipped_times(video: str) -> List[float]:
    shots = json.loads((Path("data") / video / "shots.json").read_text(encoding="utf-8"))["shots"]
    return sorted(float(s["t_sec"]) for s in shots)


def emit(t: np.ndarray, p: np.ndarray, thr: float) -> List[float]:
    """Accept candidates at or above thr; within NMS_S keep the most confident."""
    kept: List[int] = []
    for i in np.argsort(-p):
        if p[i] < thr:
            break
        if any(abs(t[i] - t[k]) < NMS_S for k in kept):
            continue
        kept.append(int(i))
    return sorted(float(t[i]) for i in kept)


def score(times: Sequence[float], truth: Sequence[float]):
    m = one_to_one(list(times), list(truth), MATCH_S)
    return len(m), len(times) - len(m)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_predict(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
                                                           max_iter=4000))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def probe(heldout: bool) -> int:
    rows = manifest()
    z = np.load(EMB, allow_pickle=False)
    vec = dict(zip(z["ids"].tolist(), z["x"]))
    rows = [r for r in rows if r["clip_id"] in vec]
    by = {v: [r for r in rows if r["video"] == v] for v in DEV + HELD_OUT}
    X = {v: np.stack([vec[r["clip_id"]] for r in by[v]]) for v in by if by[v]}
    y = {v: np.array([r["label"] for r in by[v]]) for v in by if by[v]}
    t = {v: np.array([r["t_sec"] for r in by[v]]) for v in by if by[v]}
    for v in DEV:
        print(f"{v:32s} clips {len(y[v]):4d}  real {int(y[v].sum()):3d}")

    # 1. learning curve: leave one dev video out, train on a FRACTION of the other two
    print("\nLEARNING CURVE (leave-one-video-out AUC; mean of 5 subsamples; 0.5 = no signal)")
    rng = np.random.default_rng(0)
    full_p = {}
    for frac in FRACTIONS:
        cells = []
        for te in DEV:
            tr = [v for v in DEV if v != te]
            Xtr, ytr = np.vstack([X[v] for v in tr]), np.concatenate([y[v] for v in tr])
            aucs = []
            for s in range(SEEDS if frac < 1 else 1):
                if frac < 1:
                    idx = np.concatenate([rng.choice(np.where(ytr == c)[0],
                                                     max(2, int(round(frac * (ytr == c).sum()))),
                                                     replace=False) for c in (0, 1)])
                else:
                    idx = np.arange(len(ytr))
                p = fit_predict(Xtr[idx], ytr[idx], X[te])
                aucs.append(auc(y[te], p))
                if frac == 1:
                    full_p[te] = p
            cells.append(float(np.mean(aucs)))
        n_real = int(round(frac * sum(int(y[v].sum()) for v in DEV) * 2 / 3))
        print(f"  {int(frac * 100):3d}% of training data (~{n_real} real per fold): "
              + "  ".join(f"{v[-7:]} {c:.3f}" for v, c in zip(DEV, cells))
              + f"   mean {np.mean(cells):.3f}")

    # 2. what it would EMIT, against the rules, threshold chosen across the dev videos
    best, best_val = None, -1e9
    for thr in THRESHOLDS:
        val = sum(np.subtract(*score(emit(t[v], full_p[v], thr), truth_times(v))) for v in DEV)
        if val > best_val:
            best, best_val = thr, val
    print(f"\nEMITTED SHOTS at threshold {best:.2f} (chosen on dev, found minus junk)")
    for v in DEV:
        f, j = score(emit(t[v], full_p[v], best), truth_times(v))
        sf, sj = score(shipped_times(v), truth_times(v))
        print(f"  {v:32s} rules {sf}/{len(truth_times(v))} junk {sj}   |   video model {f} junk {j}")

    if heldout:
        v = HELD_OUT[0]
        Xtr = np.vstack([X[d] for d in DEV])
        ytr = np.concatenate([y[d] for d in DEV])
        p = fit_predict(Xtr, ytr, X[v])
        f, j = score(emit(t[v], p, best), truth_times(v))
        sf, sj = score(shipped_times(v), truth_times(v))
        print(f"\n=== HELD-OUT {v}: AUC {auc(y[v], p):.3f}   rules {sf}/{len(truth_times(v))} "
              f"junk {sj}   |   video model {f} junk {j}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["embed", "probe"])
    ap.add_argument("--heldout", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "embed":
        embed()
        return 0
    return probe(a.heldout)


if __name__ == "__main__":
    raise SystemExit(main())

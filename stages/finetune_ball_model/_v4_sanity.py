"""CPU sanity check for the v4 training pipeline (no GPU, no real training).
Proves: densification, heatmap target, TrackNet instantiation at 720p, focal
loss + forward/backward, and the recall eval all work before Colab time.
"""
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader

from stages.finetune_ball_model._v4_data import (
    load_samples, densify_labels, make_heatmap, VideoFrameReader, PROC_H, PROC_W)
from stages.finetune_ball_model.train_v4 import (
    build_model, focal_loss, BallDataset, evaluate)

LB = Path("data/pb_2min/ball_labels.json")

print("=== 1. densification ===")
raw = json.loads(LB.read_text())["labels"]
dens = densify_labels(raw)
print(f"labels: raw {len(raw)} -> densified {len(dens)}")
samples = load_samples(LB, "pb_2min")
vis = [s for s in samples if s.visible]
print(f"samples {len(samples)}, visible {len(vis)}, not-visible {len(samples)-len(vis)}")

print("=== 2. heatmap target ===")
hm = make_heatmap(640.0, 360.0)
print(f"heatmap shape {hm.shape}, peak {hm.max():.3f}, nonzero {(hm>0).sum()}")
assert abs(hm.max() - 1.0) < 1e-6, "heatmap peak must be exactly 1.0"
assert make_heatmap(None, None).max() == 0.0, "not-visible -> zero heatmap"

print("=== 3. model instantiation @ (720,1280) out=1 ===")
dev = "cpu"
model = build_model(dev)
nparams = sum(p.numel() for p in model.parameters())
print(f"TrackNet params: {nparams/1e6:.2f}M")

print("=== 4. forward/backward + focal loss (few steps, CPU) ===")
reader = VideoFrameReader()
# mix visible + not-visible so focal sees positives and negatives
mix = vis[:3] + [s for s in samples if not s.visible][:1]
ds = BallDataset(mix, reader=reader, augment=True)
dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for step in range(4):
    for stack, tgt, px, py in dl:
        t0 = time.time()
        pred = model(stack)
        assert pred.shape == tgt.shape, f"shape {pred.shape} vs {tgt.shape}"
        loss = focal_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        print(f"  step {step}: loss {float(loss):.4f} "
              f"pred[{float(pred.min()):.3f},{float(pred.max()):.3f}] "
              f"{time.time()-t0:.1f}s")
        break  # one batch per step
import math
assert all(math.isfinite(l) for l in losses), "non-finite loss!"
print(f"losses {[round(l,3) for l in losses]} | last<first: {losses[-1] < losses[0]}")

print("=== 5. recall eval ===")
m = evaluate(model, vis[:6], dev, reader)
print(f"eval(6 visible, untrained model): {m}")
reader.release()
print("\nSANITY OK — pipeline runs end to end on CPU.")

"""Stage 4.5 v4 — TrackNet training (focal loss, importable + Colab-driven).

The training logic lives here (not buried in a notebook) so it is CPU-testable
locally and the Colab notebook just imports and calls `train()`. This is a
deliberate fix for v1/v2, where notebook-only code made the loss bugs
(confidently-wrong BCE, predict-zero MSE) hard to catch.

Key fixes baked in:
  - CenterNet-style penalty-reduced FOCAL loss on a Gaussian heatmap target.
    No trivial "predict zero" minimum (v2's failure); no "confidently wrong"
    incentive (v1's failure).
  - Center-frame supervision from 3 stacked frames (temporal disambiguation).
  - Validation = detection RECALL within a pixel tolerance, NOT loss.

See stages/finetune_ball_model/contract_v4.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stages.track_ball._tracknet_model import TrackNet
from stages.finetune_ball_model._v4_data import (
    Sample, VideoFrameReader, build_input_stack, sample_target,
    PROC_H, PROC_W,
)

TOL_PX_PROC = 6          # detection correct if peak within this many proc px
CONF_THRESH = 0.30       # min heatmap peak to count as a detection


# --- Focal loss (CenterNet / CornerNet penalty-reduced) ---------------------

def focal_loss(pred: torch.Tensor, gt: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0,
               eps: float = 1e-6) -> torch.Tensor:
    """pred, gt: (N,1,H,W) in [0,1]. gt is a Gaussian heatmap with peak==1.0 at
    the ball. Positives = pixels where gt==1; everything else is a
    penalty-reduced negative (weight (1-gt)^beta). No predict-zero minimum:
    a frame with a ball MUST light up its peak or pos_loss stays large."""
    pred = pred.clamp(eps, 1.0 - eps)
    pos = gt.eq(1.0).float()
    neg = 1.0 - pos
    pos_loss = ((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = ((1 - gt) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg
    n_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


# --- Dataset -----------------------------------------------------------------

class BallDataset(Dataset):
    def __init__(self, samples: List[Sample], reader: Optional[VideoFrameReader] = None,
                 augment: bool = False):
        self.samples = samples
        self.reader = reader or VideoFrameReader()
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        stack = build_input_stack(self.reader, s)
        if stack is None:
            stack = np.zeros((9, PROC_H, PROC_W), np.float32)
            tgt = np.zeros((1, PROC_H, PROC_W), np.float32)
            return torch.from_numpy(stack), torch.from_numpy(tgt), -1.0, -1.0
        tgt = sample_target(s)[None]  # (1,H,W)
        if self.augment:
            stack, tgt = _augment(stack, tgt)
        # peak coords for recall metric (proc res); (-1,-1) if not visible
        if s.visible:
            px = s.x * (PROC_W / s.src_w)
            py = s.y * (PROC_H / s.src_h)
        else:
            px = py = -1.0
        return (torch.from_numpy(stack), torch.from_numpy(tgt.astype(np.float32)),
                float(px), float(py))


def _augment(stack: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Photometric jitter (generalization lever) + horizontal flip. Geometry of
    the court is irrelevant to a ball detector, so flip is safe."""
    if np.random.rand() < 0.5:
        stack = stack[:, :, ::-1].copy()
        tgt = tgt[:, :, ::-1].copy()
    # brightness/contrast jitter applied identically to all 3 frames
    if np.random.rand() < 0.7:
        gain = 1.0 + (np.random.rand() - 0.5) * 0.4
        bias = (np.random.rand() - 0.5) * 0.1
        stack = np.clip(stack * gain + bias, 0.0, 1.0).astype(np.float32)
    return stack, tgt


# --- Peak extraction + recall metric ----------------------------------------

def heatmap_peak(hm: np.ndarray) -> Tuple[int, int, float]:
    """Return (x, y, value) of the heatmap max."""
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    return int(ix), int(iy), float(hm[iy, ix])


@torch.no_grad()
def evaluate(model, samples: List[Sample], device, reader: VideoFrameReader,
             tol: float = TOL_PX_PROC, conf: float = CONF_THRESH) -> dict:
    """Detection recall (visible frames localized within tol) + false-positive
    rate (peaks above conf on not-visible frames)."""
    model.eval()
    tp = fn = fp = tn = 0
    for s in samples:
        stack = build_input_stack(reader, s)
        if stack is None:
            continue
        x = torch.from_numpy(stack)[None].to(device)
        pred = model(x)[0, 0].cpu().numpy()
        px, py, val = heatmap_peak(pred)
        detected = val >= conf
        if s.visible:
            gx = s.x * (PROC_W / s.src_w)
            gy = s.y * (PROC_H / s.src_h)
            if detected and np.hypot(px - gx, py - gy) <= tol:
                tp += 1
            else:
                fn += 1
        else:
            fp += 1 if detected else 0
            tn += 1 if not detected else 0
    recall = tp / max(tp + fn, 1)
    fp_rate = fp / max(fp + tn, 1)
    return {"recall": recall, "fp_rate": fp_rate, "tp": tp, "fn": fn,
            "fp": fp, "tn": tn}


# --- Train -------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 6
    lr: float = 1e-3
    holdout_clip: str = ""
    out_path: str = "data/models/ball_model_v4.pt"
    num_workers: int = 2


def build_model(device) -> TrackNet:
    # out_channels=1: single heatmap for the CENTER of the 3 input frames.
    return TrackNet(in_channels=9, out_channels=1,
                    input_shape=(PROC_H, PROC_W)).to(device)


def train(train_samples: List[Sample], val_samples: List[Sample],
          cfg: TrainConfig, device=None, log: Optional[logging.Logger] = None):
    log = log or logging.getLogger("train_v4")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    reader = VideoFrameReader()
    ds = BallDataset(train_samples, reader=reader, augment=True)
    # num_workers=0 here because VideoCapture isn't fork-safe; Colab can raise it
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    best_recall = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        tot = 0.0
        for stack, tgt, _, _ in dl:
            stack, tgt = stack.to(device), tgt.to(device)
            pred = model(stack)
            loss = focal_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        metrics = evaluate(model, val_samples, device, reader)
        log.info(f"epoch {epoch}: loss {tot/max(len(dl),1):.4f} "
                 f"val_recall {metrics['recall']:.3f} fp {metrics['fp_rate']:.3f}")
        if metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "input_shape": (PROC_H, PROC_W),
                        "in_channels": 9, "out_channels": 1,
                        "val_recall": best_recall}, cfg.out_path)
    reader.release()
    return best_recall

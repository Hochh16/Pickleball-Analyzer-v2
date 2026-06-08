"""Stage 4.5 v4 — frame-cache (JPEG) dataset + manifest-driven train/eval.

This is the path the Colab notebook uses: reads small 720p JPEGs produced by
prepare_v4.py (fork-safe, fast) instead of seeking 4K video. Reuses the focal
loss, model, heatmap, and peak/recall logic from _v4_data/train_v4.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from stages.finetune_ball_model._v4_data import make_heatmap, PROC_H, PROC_W
from stages.finetune_ball_model.train_v4 import (
    focal_loss, build_model, heatmap_peak, TrainConfig, TOL_PX_PROC, CONF_THRESH)


def load_manifest(folder: Path) -> Tuple[dict, Path]:
    """Return (manifest, base_dir). base_dir is where frames_dir lives."""
    folder = Path(folder)
    mpath = folder / "v4_manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    return m, folder


class CacheDataset(Dataset):
    """Reads 3-frame JPEG windows from one or more prepared clip folders."""
    def __init__(self, manifests: List[Tuple[dict, Path]], augment: bool = False):
        self.items = []
        for m, base in manifests:
            fdir = base / m.get("frames_dir", "frames_720")
            for s in m["samples"]:
                self.items.append((s, fdir))
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _read(self, fdir: Path, idx: int) -> Optional[np.ndarray]:
        img = cv2.imread(str(fdir / f"{idx}.jpg"))
        if img is None:
            return None
        if img.shape[:2] != (PROC_H, PROC_W):
            img = cv2.resize(img, (PROC_W, PROC_H))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, i):
        s, fdir = self.items[i]
        frs = [self._read(fdir, f) for f in s["frames"]]
        if any(f is None for f in frs):
            stack = np.zeros((9, PROC_H, PROC_W), np.float32)
            tgt = np.zeros((1, PROC_H, PROC_W), np.float32)
            return torch.from_numpy(stack), torch.from_numpy(tgt), -1.0, -1.0
        stack = np.concatenate([f.transpose(2, 0, 1) for f in frs], 0).astype(np.float32)
        if s["visible"]:
            tgt = make_heatmap(s["x_proc"], s["y_proc"])[None]
            px, py = float(s["x_proc"]), float(s["y_proc"])
        else:
            tgt = np.zeros((1, PROC_H, PROC_W), np.float32)
            px = py = -1.0
        if self.augment:
            if np.random.rand() < 0.5:
                stack = stack[:, :, ::-1].copy()
                tgt = tgt[:, :, ::-1].copy()
                if px >= 0:
                    px = PROC_W - 1 - px
            if np.random.rand() < 0.7:
                gain = 1.0 + (np.random.rand() - 0.5) * 0.4
                bias = (np.random.rand() - 0.5) * 0.1
                stack = np.clip(stack * gain + bias, 0, 1).astype(np.float32)
        return (torch.from_numpy(stack), torch.from_numpy(tgt.astype(np.float32)),
                px, py)


@torch.no_grad()
def evaluate_cache(model, ds: CacheDataset, device, batch_size=8,
                   tol=TOL_PX_PROC, conf=CONF_THRESH) -> dict:
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    tp = fn = fp = tn = 0
    for stack, tgt, px, py in dl:
        pred = model(stack.to(device)).cpu().numpy()
        for b in range(pred.shape[0]):
            x, y, val = heatmap_peak(pred[b, 0])
            detected = val >= conf
            gx, gy = float(px[b]), float(py[b])
            if gx >= 0:
                if detected and np.hypot(x - gx, y - gy) <= tol:
                    tp += 1
                else:
                    fn += 1
            else:
                fp += 1 if detected else 0
                tn += 1 if not detected else 0
    return {"recall": tp / max(tp + fn, 1), "fp_rate": fp / max(fp + tn, 1),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def train_from_manifests(train_folders: List[Path], val_folder: Path,
                         cfg: TrainConfig, test_folder: Optional[Path] = None,
                         device=None, log: Optional[logging.Logger] = None,
                         num_workers: int = 4) -> dict:
    """Train on the train clips; EARLY-STOP on the val clip (same court);
    measure honest generalization on the TEST clip (different court, never used
    for training or checkpoint selection). Saves best-by-val-recall weights +
    a validation_report.json with both val and test recall.

    With the current data: train = same-court clips, val = a held-out
    same-court clip, test = the different-court clip."""
    log = log or logging.getLogger("train_v4_cache")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_mans = [load_manifest(f) for f in train_folders]
    val_man = load_manifest(val_folder)
    train_ds = CacheDataset(train_mans, augment=True)
    val_ds = CacheDataset([val_man], augment=False)
    test_ds = test_man = None
    if test_folder is not None:
        test_man = load_manifest(test_folder)
        test_ds = CacheDataset([test_man], augment=False)
    log.info(f"train {len(train_ds)} | val {len(val_ds)} ({val_man[0]['clip']}) "
             f"| test {len(test_ds) if test_ds else 0} "
             f"({test_man[0]['clip'] if test_man else 'none'}) | device {device}")

    model = build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=num_workers)
    best_val = -1.0
    best_test = None
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        tot = 0.0
        for stack, tgt, _, _ in dl:
            pred = model(stack.to(device))
            loss = focal_loss(pred, tgt.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        vm = evaluate_cache(model, val_ds, device)
        row = {"epoch": epoch, "loss": tot / max(len(dl), 1),
               "val_recall": vm["recall"], "val_fp": vm["fp_rate"]}
        history.append(row)
        msg = (f"epoch {epoch}: loss {row['loss']:.4f} "
               f"val_recall {vm['recall']:.3f} fp {vm['fp_rate']:.3f}")
        if vm["recall"] > best_val:
            best_val = vm["recall"]
            tm = evaluate_cache(model, test_ds, device) if test_ds else None
            best_test = tm["recall"] if tm else None
            if tm:
                row["test_recall"] = tm["recall"]
                row["test_fp"] = tm["fp_rate"]
                msg += f" | TEST_recall {tm['recall']:.3f} fp {tm['fp_rate']:.3f}"
            outp = Path(cfg.out_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "input_shape": (PROC_H, PROC_W),
                        "in_channels": 9, "out_channels": 1,
                        "val_recall": best_val, "val_clip": val_man[0]["clip"],
                        "test_recall": best_test,
                        "test_clip": test_man[0]["clip"] if test_man else None},
                       outp)
            (outp.parent / "validation_report.json").write_text(
                json.dumps({"best_val_recall": best_val,
                            "test_recall_at_best_val": best_test,
                            "val_clip": val_man[0]["clip"],
                            "test_clip": test_man[0]["clip"] if test_man else None,
                            "history": history}, indent=2) + "\n",
                encoding="utf-8")
        log.info(msg)
    log.info(f"best val recall {best_val:.3f} | test recall (generalization) "
             f"{best_test if best_test is not None else 'n/a'}")
    return {"best_val_recall": best_val, "test_recall": best_test,
            "history": history}

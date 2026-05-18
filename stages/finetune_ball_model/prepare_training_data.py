"""
Stage 4.5 — prepare training data.

Convert per-video ball_labels.json files into TrackNetV2 training samples.
For each labeled frame i, load the consecutive frame triple (i-2, i-1, i),
resize to 288x512, generate a Gaussian heatmap target at the labeled
position, and save as a single .npz file per sample.

Split: samples from the --val-video go to val/, all others to train/.

Storage layout (DEVIATES from contract's single .npy with shape (12,H,W)):
inputs stored as uint8 (lossless), targets as float16 (lossless for the
sparse Gaussian peaks used here). One .npz per sample with named arrays
'input' (9,H,W) and 'target' (3,H,W). The training notebook casts to
float32 on load. Total dataset size at ~10.7k samples: ~22 GB vs the
contract's ~72 GB with float32 in a single .npy.

Target semantics: same Gaussian peak on all 3 channels at the labeled
position when ball_visible=true; all-zero when ball_visible=false.
Channels 0/1 supervise the model on adjacent-frame predictions even
though Stage 4 inference only reads channel 2; this preserves Dettor's
3-channel prior and lets the shared encoder learn ball features without
contradictory zero-supervision signals.

Usage:
    python -m stages.finetune_ball_model.prepare_training_data \
        --labels data/test_clip/ball_labels.json \
        --labels data/indoor_b/ball_labels.json \
        --labels data/indoor_c/ball_labels.json \
        --labels data/outdoor/ball_labels.json \
        --out-dir data/training \
        --val-video outdoor
"""
import argparse
import datetime as dt
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

SCHEMA_VERSION = 1
STAGE_VERSION = "0.1.0"

LABELS_SCHEMA_VERSION = 1  # ball_labels.json schema this script accepts

# TrackNetV2 native input resolution (must match stages.track_ball)
MODEL_H = 288
MODEL_W = 512

PER_VIDEO_MIN_LABELS = 200  # contract acceptance criterion (warn-only here;
                            # enforced hard in stages/finetune_ball_model/
                            # smoke_test.py)


# ---------- helpers ----------

def fail(msg: str, exc_cls=RuntimeError):
    raise exc_cls(msg)


def setup_logging(level: str) -> logging.Logger:
    log = logging.getLogger("prepare_training_data")
    log.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(handler)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


# ---------- labels JSON loading and validation ----------

def _load_labels(path: Path, log: logging.Logger) -> dict:
    if not path.exists():
        fail(f"ball_labels.json not found: {path}", FileNotFoundError)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        fail(f"ball_labels.json is not valid JSON ({path}): {e}", ValueError)

    sv = data.get("schema_version")
    if sv != LABELS_SCHEMA_VERSION:
        fail(f"{path}: schema_version={sv}, expected {LABELS_SCHEMA_VERSION}",
             ValueError)

    required = ("video_path", "video_frame_count", "video_width",
                "video_height", "labels")
    for k in required:
        if k not in data:
            fail(f"{path}: missing required field '{k}'", ValueError)

    labels = data["labels"]
    if not isinstance(labels, list):
        fail(f"{path}: 'labels' must be a list", ValueError)

    vw = int(data["video_width"])
    vh = int(data["video_height"])
    vfc = int(data["video_frame_count"])

    # Validate each label entry
    prev_idx = -1
    for i, lab in enumerate(labels):
        for k in ("frame_idx", "ball_visible", "pixel_x", "pixel_y"):
            if k not in lab:
                fail(f"{path}: labels[{i}] missing '{k}'", ValueError)
        fi = lab["frame_idx"]
        if not isinstance(fi, int) or fi < 0:
            fail(f"{path}: labels[{i}].frame_idx must be non-negative int, "
                 f"got {fi!r}", ValueError)
        if fi <= prev_idx:
            fail(f"{path}: labels not strictly sorted by frame_idx at "
                 f"index {i} (frame_idx={fi}, previous={prev_idx})",
                 ValueError)
        if fi >= vfc:
            fail(f"{path}: labels[{i}].frame_idx={fi} >= "
                 f"video_frame_count={vfc}", ValueError)
        bv = lab["ball_visible"]
        if bv not in (True, False):
            fail(f"{path}: labels[{i}].ball_visible must be bool, got {bv!r}",
                 ValueError)
        px, py = lab["pixel_x"], lab["pixel_y"]
        if bv:
            if px is None or py is None:
                fail(f"{path}: labels[{i}] ball_visible=true but pixel "
                     f"coords are null", ValueError)
            if not (0 <= float(px) < vw) or not (0 <= float(py) < vh):
                log.warning(f"{path}: labels[{i}].pixel=({px},{py}) outside "
                            f"video bounds {vw}x{vh}; will clip on resize")
        else:
            if px is not None or py is not None:
                fail(f"{path}: labels[{i}] ball_visible=false but pixel "
                     f"coords are not null", ValueError)
        prev_idx = fi

    if len(labels) < PER_VIDEO_MIN_LABELS:
        log.warning(f"{path}: {len(labels)} labels (< "
                    f"{PER_VIDEO_MIN_LABELS} per-video minimum); training "
                    f"may be unreliable. Stage 4.5 smoke test will fail.")

    log.info(f"loaded {path}: {len(labels)} labels for {data['video_path']}")
    return data


def _resolve_video_path(labels_json_path: Path, video_path_str: str) -> Path:
    """The video_path field in ball_labels.json is recorded as it was
    passed to label_ball.py — most commonly relative to the project root.
    Resolve cwd-first, then absolute fallback."""
    p = Path(video_path_str)
    if p.exists():
        return p.resolve()
    if p.is_absolute():
        fail(f"video referenced in {labels_json_path} not found at "
             f"{video_path_str}", FileNotFoundError)
    fail(f"video referenced in {labels_json_path} ({video_path_str}) not "
         f"found relative to cwd ({Path.cwd()})", FileNotFoundError)


# ---------- target generation ----------

def _make_gaussian(x: float, y: float, h: int, w: int,
                   sigma: float) -> np.ndarray:
    """2D unnormalized Gaussian with peak=1 at (x, y) on an (h, w) grid.
    Returns float32. Bounded to 3*sigma for efficiency."""
    target = np.zeros((h, w), dtype=np.float32)
    if not (0 <= x < w and 0 <= y < h):
        return target  # peak outside grid; zero supervision
    r = int(3 * sigma) + 1
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return target
    yy, xx = np.ogrid[y0:y1, x0:x1]
    target[y0:y1, x0:x1] = np.exp(
        -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)
    )
    return target


# ---------- per-frame preprocessing (matches stages.track_ball) ----------

def _preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Mirror Stage 4's _preprocess_triple per-frame logic, but return uint8.
    Notebook casts to float32 and divides by 255 at training time.
    Output: (3, MODEL_H, MODEL_W) uint8, channel order RGB."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (MODEL_W, MODEL_H),
                         interpolation=cv2.INTER_LINEAR)
    return resized.transpose(2, 0, 1)  # HWC -> CHW


# ---------- per-video processing ----------

def _process_video(video_path: Path, labels: list,
                   video_width: int, video_height: int,
                   sigma: float, log: logging.Logger):
    """Walk the video sequentially with a 3-frame rolling buffer.
    Yields (label_dict, input_uint8 (9,H,W), target_f16 (3,H,W)) for each
    label that has 2 frames of history."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"OpenCV could not open video: {video_path}", RuntimeError)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sx = MODEL_W / video_width
    sy = MODEL_H / video_height

    # frame_idx -> label lookup, dropping labels with frame_idx < 2
    label_by_idx = {lab["frame_idx"]: lab for lab in labels
                    if lab["frame_idx"] >= 2}
    n_skipped_history = len(labels) - len(label_by_idx)
    if n_skipped_history > 0:
        log.info(f"  skipped {n_skipped_history} labels with frame_idx < 2 "
                 f"(insufficient history for triple)")

    if not label_by_idx:
        cap.release()
        log.warning(f"  no usable labels for {video_path}")
        return

    max_label_idx = max(label_by_idx.keys())
    last_frame_to_read = min(n_frames - 1, max_label_idx)

    buffer = []  # last 3 preprocessed frames as uint8 (3, H, W)
    n_emitted = 0
    n_buffer_misses = 0
    n_read_failures = 0

    for frame_idx in range(last_frame_to_read + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            log.warning(f"  frame {frame_idx}: read failed; clearing buffer")
            buffer = []
            n_read_failures += 1
            continue

        buffer.append(_preprocess_frame(frame))
        if len(buffer) > 3:
            buffer.pop(0)

        if frame_idx in label_by_idx:
            if len(buffer) < 3:
                n_buffer_misses += 1
                continue
            lab = label_by_idx[frame_idx]
            # 9-channel input: [frame_{i-2} RGB, frame_{i-1} RGB, frame_i RGB]
            # matches stages.track_ball._preprocess_triple channel order.
            input_stack = np.concatenate(buffer, axis=0)  # (9, H, W) uint8

            if lab["ball_visible"]:
                mx = float(lab["pixel_x"]) * sx
                my = float(lab["pixel_y"]) * sy
                mx = float(np.clip(mx, 0, MODEL_W - 1))
                my = float(np.clip(my, 0, MODEL_H - 1))
                peak = _make_gaussian(mx, my, MODEL_H, MODEL_W, sigma)
                # Option B: same peak on all 3 channels
                target = np.stack([peak, peak, peak], axis=0)
            else:
                target = np.zeros((3, MODEL_H, MODEL_W), dtype=np.float32)

            target_f16 = target.astype(np.float16)
            yield lab, input_stack, target_f16
            n_emitted += 1

    cap.release()
    if n_read_failures > 0:
        log.warning(f"  {n_read_failures} frame reads failed in {video_path}")
    if n_buffer_misses > 0:
        log.warning(f"  {n_buffer_misses} labels could not be emitted due "
                    f"to read failures in their preceding 2 frames")
    log.info(f"  emitted {n_emitted} samples from {video_path}")


# ---------- main ----------

def run_stage(args, log: logging.Logger) -> dict:
    t0 = time.time()
    label_paths = [Path(p) for p in args.labels]
    out_dir = Path(args.out_dir)
    val_substring = args.val_video
    sigma = args.heatmap_sigma

    if sigma <= 0:
        fail(f"--heatmap-sigma must be > 0, got {sigma}", ValueError)
    if not val_substring:
        fail("--val-video must be a non-empty substring", ValueError)
    if len(label_paths) < 2:
        fail(f"--labels must be passed at least twice (need train + val "
             f"source videos); got {len(label_paths)}", ValueError)

    if out_dir.exists():
        if not args.force:
            fail(f"output directory exists: {out_dir}. Use --force to "
                 f"overwrite.", FileExistsError)
        log.info(f"--force: removing existing {out_dir}")
        shutil.rmtree(out_dir)

    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    # Load all labels JSONs and resolve their videos
    sources = []
    for lp in label_paths:
        data = _load_labels(lp, log)
        video_path = _resolve_video_path(lp, data["video_path"])
        sources.append({
            "labels_path": lp,
            "video_path": video_path,
            "video_path_str": data["video_path"],
            "video_width": int(data["video_width"]),
            "video_height": int(data["video_height"]),
            "labels": data["labels"],
        })

    # Resolve --val-video to exactly one source
    matches = [s for s in sources if val_substring in s["video_path_str"]]
    if len(matches) == 0:
        fail(f"--val-video '{val_substring}' matched no input video. "
             f"Candidates: "
             f"{[s['video_path_str'] for s in sources]}", ValueError)
    if len(matches) > 1:
        fail(f"--val-video '{val_substring}' matched {len(matches)} "
             f"videos: {[s['video_path_str'] for s in matches]}. "
             f"Use a more specific substring.", ValueError)
    val_source = matches[0]
    train_sources = [s for s in sources if s is not val_source]
    log.info(f"validation video: {val_source['video_path_str']}")
    log.info(f"training videos:  "
             f"{[s['video_path_str'] for s in train_sources]}")

    # Process every source, write samples
    n_train = 0
    n_val = 0
    samples_meta = []

    for src in sources:
        is_val = src is val_source
        split = "val" if is_val else "train"
        log.info(f"processing {src['video_path']} -> {split}/")
        for lab, input_stack, target_f16 in _process_video(
                src["video_path"], src["labels"],
                src["video_width"], src["video_height"],
                sigma, log):
            if is_val:
                idx = n_val
                out_path = out_val / f"{idx:06d}.npz"
                n_val += 1
            else:
                idx = n_train
                out_path = out_train / f"{idx:06d}.npz"
                n_train += 1
            np.savez(out_path, input=input_stack, target=target_f16)
            samples_meta.append({
                "split": split,
                "idx": idx,
                "source_video": src["video_path_str"],
                "source_frame_idx": int(lab["frame_idx"]),
                "ball_visible": bool(lab["ball_visible"]),
                "pixel_x": (None if lab["pixel_x"] is None
                            else float(lab["pixel_x"])),
                "pixel_y": (None if lab["pixel_y"] is None
                            else float(lab["pixel_y"])),
            })

    if n_train == 0:
        fail("no training samples produced; check --val-video and label "
             "counts", ValueError)
    if n_val == 0:
        fail("no validation samples produced; check --val-video matches "
             "an input with labels", ValueError)

    # Write metadata
    meta = {
        "schema_version": SCHEMA_VERSION,
        "n_train": n_train,
        "n_val": n_val,
        "heatmap_sigma": sigma,
        "model_input_h": MODEL_H,
        "model_input_w": MODEL_W,
        "val_source_video": val_source["video_path_str"],
        "train_source_videos": [s["video_path_str"] for s in train_sources],
        "file_format": "npz",
        "input_array_name": "input",
        "input_dtype": "uint8",
        "input_shape": [9, MODEL_H, MODEL_W],
        "input_channel_order": ("RGB per frame; frames in time order "
                                "(i-2, i-1, i)"),
        "input_normalization_at_train_time": ("cast to float32 and divide "
                                              "by 255.0"),
        "target_array_name": "target",
        "target_dtype": "float16",
        "target_shape": [3, MODEL_H, MODEL_W],
        "target_semantics": ("same Gaussian peak on all 3 channels at the "
                             "labeled position when ball_visible=true; "
                             "all-zero when ball_visible=false"),
        "stage_version": STAGE_VERSION,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "samples": samples_meta,
    }
    meta_path = out_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    wall = time.time() - t0
    log.info(f"wrote {meta_path}")
    log.info(f"done: {n_train} train, {n_val} val ({wall:.1f}s)")
    return meta


# ---------- CLI ----------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 4.5 — prepare training data")
    p.add_argument("--labels", action="append", required=True,
                   help="ball_labels.json path; repeat per video")
    p.add_argument("--out-dir", type=Path, required=True, dest="out_dir")
    p.add_argument("--val-video", type=str, required=True, dest="val_video",
                   help="substring matching exactly one input video path; "
                        "that video's samples go to val/")
    p.add_argument("--heatmap-sigma", type=float, default=2.0,
                   dest="heatmap_sigma")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   dest="log_level")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    log = setup_logging(args.log_level)
    try:
        run_stage(args, log)
    except (FileNotFoundError, FileExistsError, ValueError,
            RuntimeError) as e:
        log.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
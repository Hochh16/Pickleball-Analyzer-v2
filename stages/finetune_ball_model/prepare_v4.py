"""Stage 4.5 v4 — prepare a training frame-cache for one clip.

Pre-extracts the frames needed for training (each labeled center frame +/- the
3-frame-stack neighbors) at the 720p processing resolution as JPEGs, plus a
v4_manifest.json describing each sample. Training then reads small JPEGs
(fast, fork-safe DataLoader, ~1-2 GB to upload to Colab) instead of seeking
through 15 GB of 4K video.

Reads frames SEQUENTIALLY over the needed range (4K random seeks are slow).

Usage:
    python -m stages.finetune_ball_model.prepare_v4 data/pb_2min --clip pb_2min
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from stages.finetune_ball_model._v4_data import (
    densify_labels, FRAMES_DIR, PROC_H, PROC_W, FRAME_STRIDE)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage 4.5 v4 — frame-cache prep")
    p.add_argument("folder", type=Path)
    p.add_argument("--clip", default=None, help="clip name (default folder name)")
    p.add_argument("--stride", type=int, default=FRAME_STRIDE)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--video", type=Path, default=None,
                   help="source video, when it does not sit in the clip folder. Five labelled "
                        "clips keep only their frame cache locally; their 4K sources live in "
                        "Dropbox, and they are 1-4 GB each, so this points at them in place "
                        "rather than copying.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    clip = args.clip or args.folder.name
    label_path = args.folder / "ball_labels.json"
    if not label_path.exists():
        print(f"no labels: {label_path}")
        return 1
    d = json.loads(label_path.read_text(encoding="utf-8"))
    video_path = (str(args.video) if args.video
                  else d.get("video_path") or str(args.folder / "video.mp4"))
    if not Path(video_path).exists():
        # fall back to the local copy if the recorded path isn't reachable
        local = args.folder / "video.mp4"
        if local.exists():
            video_path = str(local)
        else:
            print(f"video not found: {video_path}")
            return 1
    src_w = int(d.get("video_width", 0)) or 3840
    src_h = int(d.get("video_height", 0)) or 2160

    dens = densify_labels(d["labels"])
    # build samples: keep those whose full 3-frame window is in-range
    samples = []
    needed = set()
    for l in dens:
        c = int(l["frame_idx"])
        frames = [c - args.stride, c, c + args.stride]
        if frames[0] < 0:
            continue
        vis = bool(l.get("ball_visible")) and l.get("pixel_x") is not None
        samples.append({
            "center": c, "frames": frames, "visible": vis,
            "x_proc": (float(l["pixel_x"]) * PROC_W / src_w) if vis else None,
            "y_proc": (float(l["pixel_y"]) * PROC_H / src_h) if vis else None,
        })
        needed.update(frames)
    if not needed:
        print("no usable samples")
        return 1

    frames_dir = args.folder / FRAMES_DIR
    frames_dir.mkdir(exist_ok=True)
    lo, hi = min(needed), max(needed)
    print(f"{clip}: {len(samples)} samples, {len(needed)} unique frames "
          f"to extract over [{lo},{hi}] from {video_path}")

    # If every needed JPEG is already cached and we are not forcing, skip the video
    # entirely and just rewrite the manifest. The manifest carries x_proc/y_proc, which are
    # resolution-dependent, so it must be REGENERATED whenever PROC_H/PROC_W changes -- and
    # editing it by hand is how seven clips ended up with 1080p coordinates against 720p
    # frames, silently training the model to fire on empty court.
    if not args.force and all((frames_dir / f"{i}.jpg").exists() for i in needed):
        print(f"  all {len(needed)} frames already cached; rewriting manifest only")
        cap = None
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"cannot open {video_path}")
            return 1
    # Walk from frame 0 rather than cap.set(CAP_PROP_POS_FRAMES, lo). That seek is NOT
    # frame-accurate on long-GOP H.264: it lands on a nearby keyframe, and since the loop
    # then counts indices from `lo`, every extracted frame would be misfiled by the drift.
    # That exact bug already invalidated one labelling session (see the remap_note in
    # ball_labels.json). It has been harmless only because `lo` was always ~2; targeted
    # labelling now puts real ranges deep into the clip.
    #
    # grab() advances the decoder one frame without paying to decode a picture we discard,
    # so walking from 0 costs little even when `lo` is 17,000 frames in.
    written = 0
    for idx in ([] if cap is None else range(0, hi + 1)):
        if idx < lo or idx not in needed:
            if not cap.grab():
                break
            continue
        ok, fr = cap.read()
        if not ok:
            break
        out = frames_dir / f"{idx}.jpg"
        if args.force or not out.exists():
            small = cv2.resize(fr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), small,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        written += 1
        if written % 200 == 0:
            print(f"  {written}/{len(needed)} frames...", flush=True)
    if cap is not None:
        cap.release()

    manifest = {
        "schema_version": 1, "clip": clip, "proc_h": PROC_H, "proc_w": PROC_W,
        "stride": args.stride, "frames_dir": FRAMES_DIR,
        "src_w": src_w, "src_h": src_h, "n_samples": len(samples),
        "n_visible": sum(1 for s in samples if s["visible"]),
        "samples": samples,
    }
    (args.folder / "v4_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.folder/'v4_manifest.json'} "
          f"({len(samples)} samples, {manifest['n_visible']} visible); "
          f"extracted {written} JPEGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

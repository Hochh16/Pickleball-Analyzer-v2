"""
Verify rally-frame labels by sampling frames from each labeled rally
and writing them to disk as annotated thumbnails.

Usage:
    python tools/verify_rally_frames.py --clip data/test_clip
    python tools/verify_rally_frames.py --clip data/test_clip --samples-per-rally 5

Reads:
    <clip>/video.mp4
    <clip>/active_rally_frames.json

Writes:
    <clip>/rally_verification/rally_<N>_start.jpg
    <clip>/rally_verification/rally_<N>_mid.jpg
    <clip>/rally_verification/rally_<N>_end.jpg
    <clip>/rally_verification/summary.txt

Open the JPGs and confirm:
- rally_<N>_start.jpg shows the server about to contact the ball (or just contacted)
- rally_<N>_end.jpg shows the ball going dead (out, into net, etc.)
- rally_<N>_mid.jpg shows live play

Fails loudly if:
- video or json missing
- placeholder fields still present in the json
- no rallies labeled
- rally ranges overlap
- frame indices outside video length
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2

SCHEMA_VERSION = 1
PLACEHOLDER_FIELDS = ("_comment", "_labeling_instructions", "_example_entry_REMOVE_ME")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_rally_json(path: Path) -> dict:
    if not path.exists():
        fail(f"rally frames file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            fail(f"failed to parse {path}: {e}")

    if data.get("schema_version") != SCHEMA_VERSION:
        fail(
            f"schema_version in {path} is {data.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )

    leftover = [k for k in PLACEHOLDER_FIELDS if k in data]
    if leftover:
        fail(
            f"{path} still contains placeholder fields: {leftover}. "
            f"Fill in the rallies array and remove these fields before verifying."
        )

    rallies = data.get("rallies")
    if not isinstance(rallies, list):
        fail(f"{path}: 'rallies' must be a list")
    if len(rallies) == 0:
        fail(f"{path}: 'rallies' is empty — label at least one rally before verifying")

    for i, r in enumerate(rallies):
        if not isinstance(r, dict):
            fail(f"{path}: rally[{i}] is not an object")
        if "start_frame" not in r or "end_frame" not in r:
            fail(f"{path}: rally[{i}] is missing start_frame or end_frame")
        if not isinstance(r["start_frame"], int) or not isinstance(r["end_frame"], int):
            fail(f"{path}: rally[{i}] start_frame/end_frame must be integers")
        if r["start_frame"] < 0 or r["end_frame"] < r["start_frame"]:
            fail(
                f"{path}: rally[{i}] has invalid range "
                f"[{r['start_frame']}, {r['end_frame']}]"
            )

    sorted_rallies = sorted(rallies, key=lambda r: r["start_frame"])
    for a, b in zip(sorted_rallies, sorted_rallies[1:]):
        if b["start_frame"] <= a["end_frame"]:
            fail(
                f"{path}: rally ranges overlap — "
                f"[{a['start_frame']}, {a['end_frame']}] and "
                f"[{b['start_frame']}, {b['end_frame']}]"
            )

    return data


def annotate(frame, text_lines):
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    pad = 8
    line_h = 24
    box_h = pad * 2 + line_h * len(text_lines)
    box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
                for t in text_lines) + pad * 2
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for i, t in enumerate(text_lines):
        y = pad + line_h * (i + 1) - 6
        cv2.putText(
            frame, t, (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def grab_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True,
                    help="clip folder, e.g. data/test_clip")
    ap.add_argument("--samples-per-rally", type=int, default=3,
                    choices=[3, 5],
                    help="3 = start/mid/end; 5 = start/q1/mid/q3/end")
    args = ap.parse_args()

    clip = args.clip
    if not clip.is_dir():
        fail(f"clip folder not found: {clip}")

    video_path = clip / "video.mp4"
    json_path = clip / "active_rally_frames.json"
    out_dir = clip / "rally_verification"
    out_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        fail(f"video not found: {video_path}")

    data = load_rally_json(json_path)
    rallies = sorted(data["rallies"], key=lambda r: r["start_frame"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    for i, r in enumerate(rallies):
        if r["end_frame"] >= total_frames:
            cap.release()
            fail(
                f"rally[{i}] end_frame {r['end_frame']} >= total frames "
                f"{total_frames}"
            )

    if args.samples_per_rally == 3:
        sample_labels = ["start", "mid", "end"]
        sample_fracs = [0.0, 0.5, 1.0]
    else:
        sample_labels = ["start", "q1", "mid", "q3", "end"]
        sample_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]

    summary_lines = [
        f"video: {video_path}",
        f"total frames: {total_frames}, fps: {fps:.2f}",
        f"rallies labeled: {len(rallies)}",
        "",
    ]

    for n, r in enumerate(rallies, start=1):
        s, e = r["start_frame"], r["end_frame"]
        length = e - s + 1
        secs = length / fps if fps > 0 else 0.0
        summary_lines.append(
            f"rally {n}: frames [{s}, {e}]  length={length} "
            f"({secs:.2f}s)  note={r.get('note', '')!r}"
        )

        for label, frac in zip(sample_labels, sample_fracs):
            idx = int(round(s + frac * (e - s)))
            frame = grab_frame(cap, idx)
            if frame is None:
                summary_lines.append(
                    f"  WARN: could not read frame {idx} for {label}"
                )
                continue
            t_secs = idx / fps if fps > 0 else 0.0
            text = [
                f"rally {n} / {label}",
                f"frame {idx}  ({t_secs:.2f}s)",
                f"range [{s}, {e}]",
            ]
            annotate(frame, text)
            out_path = out_dir / f"rally_{n:02d}_{label}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        summary_lines.append("")

    cap.release()

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"verification thumbnails written to: {out_dir}")
    print(f"open {out_dir / 'summary.txt'} for the rally summary")
    print(f"inspect rally_NN_start.jpg / rally_NN_end.jpg per rally to confirm labels")


if __name__ == "__main__":
    main()
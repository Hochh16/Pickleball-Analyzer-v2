"""Build stages/track_ball/infer_v4.ipynb — the GPU (Colab) inference path
for Stage 4 v4.

Self-contained, mirroring stages/track_ball/track_ball_v4.py but with BATCHED
GPU inference (the real speedup over the per-frame CPU module — CPU is ~11
s/frame; an A100 batched does the whole 2-min clip in ~1-2 min). The TrackNet
model class is read from _tracknet_model.py at build time and embedded verbatim
so the notebook stays a true "Run All, nothing to edit" file with no bundle.

postprocess() is copied verbatim from track_ball_v4.py so trajectory behavior
is identical to the committed local module. If you change postprocess there,
re-run this builder.

Usage:
    python tools/build_infer_v4_nb.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC = (ROOT / "stages/track_ball/_tracknet_model.py").read_text(encoding="utf-8")

# Strip the model file's __main__ demo block so importing it is side-effect free.
_marker = '\nif __name__ == "__main__":'
if _marker in MODEL_SRC:
    MODEL_SRC = MODEL_SRC[:MODEL_SRC.index(_marker)].rstrip() + "\n"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(
"""# Stage 4 v4 — ball inference on GPU (self-contained)

**One file. Set `CLIP` + `RANGE`, Run All.** Loads the trained `ball_model_v4.pt`
and a clip's `video.mp4` from Drive, runs the v4 TrackNet on GPU (batched), and
writes a real `ball.parquet` + `ball.meta.json` (`synthetic=false`) back to
Drive — a drop-in for the synthetic placeholder.

**Setup on Drive (`MyDrive/`):**
- `ball_model_v4.pt`  ← from training (already there)
- `pb_infer/<CLIP>/video.mp4`  ← upload the clip you want to analyze

**Runtime → Change runtime type → GPU.** CPU works but is ~11 s/frame.

Mirrors `stages/track_ball/track_ball_v4.py`; `postprocess()` is identical, so
the trajectory output matches the local module. Built by
`tools/build_infer_v4_nb.py`.
"""))

CELLS.append(code(
"""import torch
print('CUDA:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow!)')
"""))

CELLS.append(code(
"""from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive')
"""))

CELLS.append(code(
"""try:
    import cv2
except Exception:
    !pip -q install opencv-python-headless
    import cv2
import numpy as np, pandas as pd
"""))

CELLS.append(code(
"""# ===== KNOBS =====
CLIP    = 'pb_2min'          # folder under MyDrive/pb_infer/<CLIP>/ with video.mp4
WEIGHTS = DRIVE/'ball_model_v4.pt'
START   = 0                  # first frame to infer
MAXF    = None               # None = whole clip; or an int for a sub-range
OVERLAY = False              # write a debug overlay mp4 (only for short ranges!)
OVERLAY_MAXF = 600           # guard: refuse overlay if range longer than this

CONF_THRESH      = 0.30      # heatmap peak >= this -> a detection
MAX_GAP_FRAMES   = 8         # interpolate confirmed-detection gaps up to this
OUTLIER_MAX_STEP_PX = 250.0  # source px/frame; det this far from BOTH neighbours dropped
PROC_H, PROC_W   = 720, 1280
BATCH = 16 if torch.cuda.is_available() else 1   # frames per forward pass
SCHEMA_VERSION = 1

CLIP_DIR = DRIVE/'pb_infer'/CLIP
VIDEO = CLIP_DIR/'video.mp4'
assert VIDEO.exists(), f'missing {VIDEO} — upload the clip video to Drive'
assert Path(WEIGHTS).exists(), f'missing {WEIGHTS}'
print('clip', VIDEO, '| weights', WEIGHTS)
"""))

CELLS.append(md("## TrackNet model (embedded verbatim from _tracknet_model.py)"))
CELLS.append(code(MODEL_SRC))

CELLS.append(md("## Trajectory post-processing (verbatim from track_ball_v4.py)"))
CELLS.append(code(
"""def postprocess(dets: dict, frames: list) -> list:
    \"\"\"dets: frame -> (x, y, conf) for frames whose peak >= CONF. frames: full
    ordered frame list to emit rows for. Returns per-frame row dicts.\"\"\"
    conf_frames = sorted(dets.keys())
    # 1) drop isolated velocity outliers (far from BOTH neighbors)
    kept = set(conf_frames)
    for i, f in enumerate(conf_frames):
        x, y, _ = dets[f]
        bad = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(conf_frames):
                g = conf_frames[j]
                px, py, _ = dets[g]
                if np.hypot(x - px, y - py) > OUTLIER_MAX_STEP_PX * max(1, abs(f - g)):
                    bad.append(True)
                else:
                    bad.append(False)
        if bad and all(bad):  # impossible jump from every neighbor present
            kept.discard(f)
    conf = [f for f in conf_frames if f in kept]
    confset = set(conf)

    # 2) interpolate short gaps between consecutive confirmed detections
    interp = {}
    for a, b in zip(conf, conf[1:]):
        gap = b - a
        if 1 < gap <= MAX_GAP_FRAMES:
            xa, ya, _ = dets[a]
            xb, yb, _ = dets[b]
            for k in range(1, gap):
                t = k / gap
                interp[a + k] = (xa + t * (xb - xa), ya + t * (yb - ya))

    rows = []
    for f in frames:
        if f in confset:
            x, y, c = dets[f]
            rows.append({"frame_idx": f, "pixel_x": float(x), "pixel_y": float(y),
                         "visible": True, "confidence": float(c), "interpolated": False})
        elif f in interp:
            x, y = interp[f]
            rows.append({"frame_idx": f, "pixel_x": float(x), "pixel_y": float(y),
                         "visible": False, "confidence": np.nan, "interpolated": True})
        else:
            rows.append({"frame_idx": f, "pixel_x": np.nan, "pixel_y": np.nan,
                         "visible": False, "confidence": np.nan, "interpolated": False})
    return rows
"""))

CELLS.append(md("## Run inference (batched on GPU)"))
CELLS.append(code(
"""import datetime as dt, json, time
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(str(WEIGHTS), map_location=dev)
ishape = tuple(ck.get('input_shape', (PROC_H, PROC_W)))
model = TrackNet(in_channels=ck.get('in_channels', 9),
                 out_channels=ck.get('out_channels', 1), input_shape=ishape).to(dev)
model.load_state_dict(ck['state_dict']); model.eval()
print('model @', ishape, 'on', dev)

cap = cv2.VideoCapture(str(VIDEO))
assert cap.isOpened(), VIDEO
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
sx, sy = sw / PROC_W, sh / PROC_H
start = max(0, START); end = n_total if MAXF is None else min(n_total, start + MAXF)
frames = list(range(start, end))
print(f'video {sw}x{sh}@{fps:.1f}, {n_total} frames; inferring [{start},{end})  batch {BATCH}')

want_overlay = OVERLAY and (end - start) <= OVERLAY_MAXF
if OVERLAY and not want_overlay:
    print(f'OVERLAY skipped: range {end-start} > OVERLAY_MAXF {OVERLAY_MAXF}')
src_cache = {} if want_overlay else None

def to_proc(fr):
    rgb = cv2.cvtColor(cv2.resize(fr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)

@torch.no_grad()
def flush(batch, dets):
    if not batch:
        return
    t = torch.from_numpy(np.stack([s for _, s in batch])).to(dev)
    with torch.amp.autocast('cuda', enabled=dev == 'cuda'):
        hm = model(t)[:, 0]
    hm = hm.float().cpu().numpy()
    for k, (cf, _) in enumerate(batch):
        h = hm[k]; iy, ix = np.unravel_index(int(h.argmax()), h.shape)
        c = float(h[iy, ix])
        if c >= CONF_THRESH:
            dets[cf] = (ix * sx, iy * sy, c)

cap.set(cv2.CAP_PROP_POS_FRAMES, start)
buf, batch, dets = [], [], {}
fidx = start; t0 = time.time()
while fidx < end:
    ok, fr = cap.read()
    if not ok:
        break
    if src_cache is not None:
        src_cache[fidx] = fr
    buf.append(to_proc(fr))
    if len(buf) > 3:
        buf.pop(0)
    if len(buf) == 3:
        batch.append((fidx - 1, np.concatenate(buf, axis=0)))
        if len(batch) >= BATCH:
            flush(batch, dets); batch = []
    fidx += 1
    if (fidx - start) % 500 == 0:
        el = time.time() - t0
        print(f'  {fidx-start}/{len(frames)}  ({(fidx-start)/max(el,1e-9):.1f} fps)')
flush(batch, dets)
cap.release()
print(f'detections: {len(dets)} / {len(frames)} frames in {time.time()-t0:.1f}s')
"""))

CELLS.append(md("## Write ball.parquet + meta (+ optional overlay), back to Drive"))
CELLS.append(code(
"""rows = postprocess(dets, frames)
df = pd.DataFrame(rows)
df.insert(0, 'schema_version', SCHEMA_VERSION)
df['visible'] = df['visible'].astype(bool)
df['interpolated'] = df['interpolated'].astype(bool)
df['confidence'] = df['confidence'].astype('float32')
out_parquet = CLIP_DIR/'ball.parquet'; out_meta = CLIP_DIR/'ball.meta.json'
df.to_parquet(out_parquet, index=False)

n_vis = int(df['visible'].sum()); n_interp = int(df['interpolated'].sum()); n = len(df)
meta = {'schema_version': SCHEMA_VERSION, 'synthetic': False, 'video_path': str(VIDEO),
        'video_frame_count': n_total, 'video_fps': float(fps),
        'video_width': sw, 'video_height': sh,
        'detector': {'tool': 'stages/track_ball/infer_v4.ipynb', 'weights': str(WEIGHTS),
                     'proc_hw': [PROC_H, PROC_W], 'conf_thresh': CONF_THRESH,
                     'max_gap_frames': MAX_GAP_FRAMES, 'batch': BATCH},
        'range': [start, end],
        'stats': {'frames': n, 'frames_visible': n_vis, 'frames_interpolated': n_interp,
                  'frames_not_visible': n - n_vis - n_interp,
                  'visible_frac': round(n_vis / max(n, 1), 4),
                  'detect_frac': round((n_vis + n_interp) / max(n, 1), 4)},
        'completed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
Path(out_meta).write_text(json.dumps(meta, indent=1) + '\\n', encoding='utf-8')
print(f'wrote {out_parquet}')
print(f'  {n_vis} visible, {n_interp} interp, {n-n_vis-n_interp} not-visible '
      f"(detect_frac {meta['stats']['detect_frac']})")

if src_cache is not None:
    op = CLIP_DIR/'_ball_check.mp4'
    rmap = {int(r.frame_idx): r for r in df.itertuples(index=False)}
    w = cv2.VideoWriter(str(op), cv2.VideoWriter_fourcc(*'mp4v'), fps, (sw, sh))
    for f in sorted(src_cache):
        fr = src_cache[f]; r = rmap.get(f)
        if r is not None and not (isinstance(r.pixel_x, float) and np.isnan(r.pixel_x)):
            col = (0, 255, 0) if r.visible else (0, 255, 255)
            cv2.circle(fr, (int(r.pixel_x), int(r.pixel_y)), 12, col, 3, cv2.LINE_AA)
        w.write(fr)
    w.release(); print('wrote overlay', op)
"""))

CELLS.append(md(
"""## Done
- Download `MyDrive/pb_infer/<CLIP>/ball.parquet` + `ball.meta.json` → the clip
  folder under `data/` locally. It is a drop-in for the synthetic ball.
- `detect_frac` is the share of frames with a ball (visible + interpolated).
  Active rallies should be high; dead time between points is legitimately low.
- If `OVERLAY=True` on a short range, eyeball `_ball_check.mp4`: green = detected,
  yellow = interpolated. Dots should sit on the ball.
"""))

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU",
      "colab": {"provenance": []}, "kernelspec": {"name": "python3",
      "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = ROOT / "stages/track_ball/infer_v4.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")

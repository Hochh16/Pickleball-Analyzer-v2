"""Manifest coordinates must land ON the ball, at the manifest's own resolution.

`x_proc`/`y_proc` are the label position expressed in PROCESSING pixels, so they are
resolution-dependent: at 720p they are source/3, at 1080p source/2. A manifest is therefore
only valid for the frame cache it was generated against.

This exists because editing a manifest by hand broke seven of eight clips. Reverting the
1080p experiment, `proc_w`/`proc_h`/`frames_dir` were patched in place while `x_proc`/`y_proc`
were left at 1080p scale — so every target heatmap sat ~1.5x off the ball and the model was
trained to fire on empty court. It failed silently: training ran to completion reporting
recall 0.000, which looked like the same warm-start failure as the real 1080p bug.

Nothing about the numbers alone catches it — 1080p coordinates still look plausible in a
1280x720 frame whenever the ball stays left of x=1280. The only reliable check is against
PIXELS: at a labelled position there must actually be something bright, because a pickleball
is the brightest small object on a court.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

CLIPS = ['pb_2min', 'pb_3min', 'pb_4min', 'pb_5min',
         'pb_3min_court2', 'pb_3min_indoor', 'indoor_B1_3min', 'indoor_C1_3min']
MIN_BRIGHTNESS_OVER_MEDIAN = 20     # measured 29-85 when correct
SAMPLES = 8


def _clip_dirs():
    return [Path('data') / c for c in CLIPS if (Path('data') / c / 'v4_manifest.json').exists()]


@pytest.mark.skipif(not _clip_dirs(), reason="no training caches present (data/ is gitignored)")
@pytest.mark.parametrize("d", _clip_dirs(), ids=lambda d: d.name)
def test_manifest_coords_land_on_the_ball(d):
    m = json.loads((d / 'v4_manifest.json').read_text(encoding='utf-8'))
    vis = [s for s in m['samples'] if s['visible'] and s['x_proc'] is not None]
    assert vis, f"{d.name}: no visible samples"

    # cheap check first: a coordinate outside the frame is unambiguously wrong
    for s in vis:
        assert 0 <= s['x_proc'] <= m['proc_w'], f"{d.name}: x_proc {s['x_proc']} > {m['proc_w']}"
        assert 0 <= s['y_proc'] <= m['proc_h'], f"{d.name}: y_proc {s['y_proc']} > {m['proc_h']}"

    rng = random.Random(0)
    scores = []
    for s in rng.sample(vis, min(SAMPLES, len(vis))):
        p = d / m['frames_dir'] / f"{s['center']}.jpg"
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        assert img is not None, f"{d.name}: unreadable {p}"
        assert img.shape[:2] == (m['proc_h'], m['proc_w']), (
            f"{d.name}: frame is {img.shape[1]}x{img.shape[0]} but manifest says "
            f"{m['proc_w']}x{m['proc_h']}")
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = int(s['x_proc']), int(s['y_proc'])
        if not (4 <= x < g.shape[1] - 4 and 4 <= y < g.shape[0] - 4):
            continue
        scores.append(float(g[y - 3:y + 4, x - 3:x + 4].max()) - float(np.median(g)))

    assert scores, f"{d.name}: no frames available to check"
    med = float(np.median(scores))
    assert med >= MIN_BRIGHTNESS_OVER_MEDIAN, (
        f"{d.name}: labelled positions are not on anything bright "
        f"(median {med:.0f} over frame median, expected >= {MIN_BRIGHTNESS_OVER_MEDIAN}). "
        f"The manifest coordinates probably do not match {m['frames_dir']}.")

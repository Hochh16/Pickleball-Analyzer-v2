# Ball-detector data-collection plan (court2 / indoor / new venues)

_Created 2026-07-07 after the Run-2b reality check. Goal: get cross-venue ball
detection to the ~0.80 recall bar so Stages 5-11 can be trusted on new venues._

## Why (what the reality check showed)

Run-2b (warm-start, all-venue) effective coverage after trajectory post-proc:
`home 0.935 | court2 0.691 | indoor 0.608`. court2 & indoor are below the 0.80
bar. Gap structure told us *why* each falls short:

- **court2**: half its misses are **long gaps (>8 frames)** = the ball vanishes for
  many consecutive frames. Root cause = **motion blur from hard hits** (blur length
  ~ ball_speed x shutter time). This is largely **capture-side**, not data volume.
- **indoor**: misses are **mostly short isolated gaps**, fp 0 -> its low recall is
  isolated misses that respond well to **more training data**.

## Lever 1 (highest leverage for hard-hit venues): faster shutter

- In the phone camera's manual/pro mode, set the **fastest shutter the light allows**
  -- target **1/1000 s or faster**. This freezes the ball and kills the blur streaks
  that gap-fill can't recover.
- Tradeoff: faster shutter = darker image. Fine in bright daylight (outdoor court2).
  **Indoor** is dimmer -> use the best-lit court you can; accept a bit more ISO noise
  before accepting blur.
- Keep **4K + 60fps** (60fps already halves inter-frame motion). If the phone offers
  **120fps at acceptable resolution**, it helps fast balls further -- but not if it
  forces a slower shutter or a big resolution drop.

## Lever 2: more clips per venue (diversity)

One clip per venue overfits to that clip's quirks. For each venue we want the model
to learn the *invariant* (ball appearance + motion), so:

- **2-3 clips per venue**, each **~3-5 min**, deliberately VARIED: different
  time-of-day / lighting, different players + clothing colors, slightly different
  (still corner-mounted) camera placement.
- **Prioritize indoor** (responds best): 2-3 indoor clips.
- **court2 / hard-hit outdoor**: 2-3 clips shot **with the faster shutter**, and
  deliberately include **hard-hit rallies** (the failure mode) so we can verify the
  faster shutter actually recovers them.

## Capture constraints (pipeline assumptions - keep these)

- Camera in a **far corner, ~6 ft high, ENTIRE court in frame**; no pan / zoom / cut.
- **Avoid adjacent-court play in the frame** (causes phantom-ball contamination).
- One continuous match per video file.

## Labeling (per clip, ~400-600 labels)

    python tools/label_ball.py --video data/<clip>/video.mp4 --out data/<clip>/ball_labels.json
    # default --sample-every 3

- Emphasize **mid-flight** frames (hard + most valuable), **including the blurred
  hard-hit ones** -- click the **center of the blur streak**.
- Mark **not-visible honestly** (Spacebar / right-click) -- negatives control false
  positives.
- Controls: left-click = ball here + advance; Spacebar = not visible; Backspace /
  left-arrow = undo a misclick; Esc = save + quit.

## Held-out discipline

With 2-3 clips per venue, **hold one whole clip per venue out** as a true unseen-venue
test (cleaner than the within-clip 12% slices Run-2b used).

## Retrain loop (you capture + label; I run the rest)

1. `python -m stages.finetune_ball_model.prepare_v4 data/<clip> --clip <clip>`
   -> 720p frame cache + `v4_manifest.json`.
2. Add the new clip name(s) to `CLIPS` in `tools/build_v4_train_bundle.py`, then
   `python tools/build_v4_train_bundle.py` -> rebuild `data/pb_v4_upload.zip`.
3. Upload the bundle to Drive; add the new clips to `TRAIN_CLIPS` (or the held-out
   test) in `finetune_v4.ipynb`; re-run the warm-start fine-tune.
4. Run `reality_check_v4.ipynb` -> per-venue raw + effective coverage. Target: each
   venue >= ~0.80 effective, home not regressed.

## Note on expectations

- **indoor** should reach the bar with more data (isolated misses).
- **court2** needs the faster shutter as much as more data; if hard-hit blur persists
  even with a fast shutter, that's a documented fundamental limit (C1) -- we then
  flag those shots as an undercount rather than chase them.

# Session Handoff: Stage 4 v4 (real ball detection) — WORKING; first real full-clip `ball.parquet` landed

State of Pickleball-Analyzer-v2 at the end of the **June 11–12 2026** session. The
full 13-stage pipeline was implemented + smoke-tested previously on a **synthetic**
placeholder ball. Stage 4/4.5 v4 (real ball detection) is now **working**: the
detector is trained and validated, the inference path is built, and the **first
real full-clip `ball.parquet` (`synthetic: false`) has been produced for
`data/pb_2min/`**. The remaining work is to run Stages 1–3 on pb_2min and then
re-run Stages 5–11 on the real ball.

## Where the project is

- **Stages 1–11:** implemented, end-to-end runnable. Last run on the **synthetic**
  placeholder ball — every ball-derived output is a validated scaffold until
  re-run on the real ball.
- **Stage 4/4.5:** **v4 WORKING.**
  - Trained detector `data/models/ball_model_v4.pt` (720p TrackNet, 3-frame/9-ch):
    **val recall 0.90 same-court / 0.54 cross-court**, fp 0.02.
  - Inference: `stages/track_ball/track_ball_v4.py` (720p + trajectory
    post-processing) + smoke `test_track_ball_v4.py`. Validated vs ground truth on
    pb_2min frames 300–420: 39/40 balls, **median 4.9px at 4K**, 100% within 25px.
  - Production (full-clip) inference: `stages/track_ball/infer_v4.ipynb` (GPU/Colab,
    built by `tools/build_infer_v4_nb.py`). **Real full-clip `ball.parquet`
    produced for `data/pb_2min/`**: 7164 frames, 4418 visible + 426 interp + 2320
    not-visible, detect_frac 0.676, coords in-bounds, conf mean 0.78. Visually
    spot-checked on the longest rally — looks good.

## What was done this session (2026-06-11/12)

- **Drove `infer_v4.ipynb` on Colab end-to-end via the Claude-in-Chrome browser
  MCP** (Claude *can* drive Colab once the Chrome extension is connected) and
  produced the first real full-clip `ball.parquet` + `ball.meta.json` for pb_2min;
  downloaded + validated them locally; rendered a local overlay spot-check.
- **Fixed a T4 OOM:** the notebook hardcoded `BATCH=16`, which OOMs a 15GB T4 at
  720×1280. The builder now scales BATCH to GPU memory (T4→4, >20GB→8, >32GB→16).
  Committed **`1621541`** ("Stage 4 v4: T4-safe GPU batch size").
- **Logged two now-first-class requirements** (product reality: many ≥5-min videos,
  varied courts) → see `KNOWN_ISSUES.md`:
  - **Throughput:** full-clip inference is **CPU-decode-bound at ~2.9 fps**
    (~40 min for 2 min of 4K/60; ~100 min for 5 min). A background task to switch
    to GPU/hardware decode (NVDEC) was spawned.
  - **Cross-court generalization:** 0.90 same-court vs **0.54 cross-court** — must
    close before relying on the detector across indoor/outdoor venues.
- **Updated docs** (this commit): ARCHITECTURE.md (Stage 4/4.5 + pipeline status no
  longer "paused"), KNOWN_ISSUES.md (v4-landed update, synthetic-caveat-still-applies
  note, two new issues), this handoff.

## NEXT STEPS (me, next session)

The real ball exists for pb_2min but the pipeline hasn't consumed it yet. pb_2min
has only `video.mp4` + `ball.parquet`; it lacks `court.json`, `players.parquet`,
`track_roles.json`, `poses.parquet` (only the old `data/test_clip/` has those).

1. **Stages 1–3 on pb_2min** (calibrate → track_players → classify_tracks → pose).
   - Stage 1 (calibrate) needs **operator court-corner clicks**; Stage 2.5 needs
     **`user_clicks.json` + `roster.json`** (operator). Flag these inputs before
     coding (ask, don't guess — see design-review prefs).
2. **Re-run Stages 5→11 on the real ball + real players**; re-validate each;
   re-tune Stage 5–10 thresholds for real (noisy, gappy) trajectories vs synthetic.
   Lift the synthetic caveat **per-stage** as each is re-validated.
3. **Calibrate Stages 9/10** against real rallies (uncalibrated until now).
4. The Stage 11 synthetic-ball watermark drops automatically once
   `ball_source != synthetic`.

Parallel / larger efforts (tracked in KNOWN_ISSUES + a spawned task):
- **Inference speedup** (GPU decode) — required for the real ≥5-min workload.
- **Cross-court training diversity** — add more indoor/outdoor courts.

## Key facts / gotchas

- **Production inference is GPU-only.** Local CPU torch is ~11 s/frame (~23h for a
  2-min clip). Use `infer_v4.ipynb` on Colab. Claude can drive it via Chrome MCP
  once the extension is connected.
- **`data/` is gitignored.** `ball.parquet`, `ball.meta.json`, `video.mp4`,
  `frames_720/`, the `ball.val_300-419.*` backups — all local, regenerable, NOT
  committed. The 120-frame validation slice is preserved as
  `data/pb_2min/ball.val_300-419.parquet`.
- **Colab upload gotcha:** uploading the whole `data\pb_2min` folder into
  `MyDrive/pb_infer/pb_2min/` doubles the path → `.../pb_2min/pb_2min/video.mp4`.
  Either upload just `video.mp4`, or set `CLIP='pb_2min/pb_2min'` in the notebook.
- **Chrome downloads** land in `C:\Users\hochh\Dropbox\My PC (DESKTOP-94DNBCT)\Downloads`
  (symlinked from `~/Downloads`); Dropbox briefly renames in-flight files to
  `<guid>.tmp` then restores the real name.
- **The 512×288 trap** (still true): never let inference silently downscale 4K to
  512×288 — it reshrinks the ball to ~2px. v4 runs at 1280×720.

## Things to NOT touch

- Don't re-attempt ball-detection v1/v2/v3; failures well understood (KNOWN_ISSUES).
- v1/v2 weights on Drive retained for reference.

## Bring this to the next session

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, stages/finetune_ball_model/contract_v4.md
    before proposing anything.

    Stage 4 v4 real ball detection is WORKING — data/pb_2min/ball.parquet is real
    (synthetic:false) and validated. Next: run Stages 1-3 on pb_2min (needs my
    court-calibration clicks + user_clicks.json + roster.json), then re-run Stages
    5-11 on the real ball and lift the synthetic caveat per-stage. Also open:
    inference throughput (GPU decode) and cross-court generalization.

---

Generated at session end on June 12, 2026.

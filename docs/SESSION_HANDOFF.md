# Session Handoff: real ball through Stages 1–5.5 done for pb_2min — Stage 6 next

State of Pickleball-Analyzer-v2 at the end of the **June 13–14 2026** session. The
full 13-stage pipeline was implemented + smoke-tested previously on a **synthetic**
placeholder ball. Real ball detection (v4) works, and pb_2min has now been pushed
through **Stages 1, 2, 2.5, 3, 5, and 5.5 on the REAL ball + real players**, each
operator-validated and committed. **Next: Stage 6 (classify shots)**, then 7→11.

**Real-vs-synthetic adaptation pattern (applies to every remaining stage 6–11):**
1. **`is_user` from `track_roles.json`** (role 'user'), NOT players.parquet's
   click-only flag (empty in the no-clicks flow). Every stage reading is_user.
2. **Resolution scaling**: px thresholds × `frame_width/1920` (4K = 2×).
3. **fps scaling**: frame-count windows × `fps/30` (60fps = 2×).
4. **Real-world-phenomenon filters gated to real ball** (`ball_source=="real"`):
   the synthetic placeholder lacks the noise/handling, so gating keeps the
   synthetic smoke bars valid (e.g. Stage 5 net-side ball-handling rejection;
   Stage 5.5 y-flip-for-all + apex filter).
5. **Validation = operator spot-check overlays** (render markers on the video,
   David confirms) — there is no real-data ground truth to auto-grade against.
6. **Gotcha:** `(x > thresh)` on numpy floats yields a **numpy bool**; `if b is
   not True` then rejects everything (numpy True ≠ Python True). Wrap in `bool()`.

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

## DONE 2026-06-13: Stages 1–3 on pb_2min (no clicking) + user-tracking fixes

pb_2min now has the full Stage 1–3 set: `court.json`/`court_zones.json` (operator
court clicks via `tools/mark_court.py`), `players.parquet`, `track_roles.json`,
`poses.parquet`. Three improvements landed, driven by "track the user extremely
well" + "no user-clicking":
- **No-click user ID** (`4b8e4b8`): operator only sets handedness/baseline/
  starting-corner; Stage 2.5 seeds the user geometrically from
  `user_starting_corner`. `user_clicks.json` is now an optional override (Stage 2
  + 2.5). `tools/mark_user.py` is the override clicker.
- **Appearance re-id** (`b348d98`): Stage 2.5 follows the user across ByteTrack
  ID swaps / gaps / side-switches by clothing-color + height. pb_2min user
  coverage 68% -> **85.5%** (visually verified on a role-overlay clip).
- **Role-aware pose** (`f349141`): Stage 3 takes `is_user` from the role `user`
  and poses every user track (incl. behind-baseline 1663). pb_2min: user pose
  6125 rows @ 99.1% detection, all 3 user tracks `[1, 1554, 1663]`.

**Product note (David):** the final app's input/setup flow must let the user
select their own handedness; the court/user inputs are product UI, not dev
fixtures (see memory `project_product_requirements`).

## DONE 2026-06-14: Stages 5 + 5.5 on the real ball (pb_2min)

Both operator-validated via spot-check overlays and committed (real-ball
adaptations per the pattern above):
- **Stage 5 (detect shots)** `8aa9164`, v0.2.0: 304 → 45 real shots (all real
  over-net strikes by David's eye). Adaptations: teleport-drop (don't crash on
  outliers), 4K resolution + 60fps scaling (the fps scaling collapsed 2–3
  duplicate detections/strike), is_user-from-roles, and a **net-side
  ball-handling filter** (real players catch/bounce/hold the ball between points
  = a sharp dir-change at a hand; every rally shot crosses the net, so consecutive
  same-side impacts = handling — keep the LAST of each run).
- **Stage 5.5 (detect bounces)** `740fac9`, v0.2.0: 135 → 16 bounces (4/4
  validated). Adaptations: scaling, **apex/off-court filter** (reject bounces
  projecting far off-court = ball in the air, not on the ground),
  **ground-contact refinement** (snap to lowest pixel_y for accurate far-court
  zones), and **y-flip-for-all on real ball** (a real bounce reverses vertical
  down→up; impulse-with-no-reversal = mid-air wobble). Deferred: bounce occluded
  behind the net is missed (ball-quality cap); is_at_feet edge case.

## NEXT STEPS (me, next session)

1. **Stage 6 (classify shots)** — unblocked now (its `bounces.json` exists). Reads
   shots/players/poses/ball/court/roster + bounces. Apply the adaptation pattern
   (is_user-from-roles, 4K scaling); validate shot labels (forehand/backhand via
   handedness, drive/dink/lob, volley) via a spot-check overlay with David.
2. **Then Stages 7 → 8 → 9 → 10 → 11** on the real ball, same per-stage approach.
3. **Calibrate Stages 9/10** against real rallies (uncalibrated until now).
4. Stage 11 synthetic-ball watermark drops automatically once `ball_source != synthetic`.

Notes carried forward: Stage 2.5 user coverage 85.5% (rest is genuine off-frame
time); partner/opponent role-awareness + opp L/R continuity still geometric
heuristics (KNOWN_ISSUES) — revisit if downstream opponent stats look off.

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

    pb_2min has real ball (synthetic:false) run through Stages 1,2,2.5,3,5,5.5
    (court.json, players.parquet, track_roles.json, poses.parquet, shots.json,
    bounces.json), each operator-validated. Next: Stage 6 (classify shots) on the
    real ball, then 7-11. Follow the real-vs-synthetic adaptation pattern at the
    top of this handoff (is_user-from-roles, 4K/fps scaling, real-only filter
    gating, spot-check validation, numpy-bool gotcha). Then calibrate Stages 9/10.
    Also open: inference throughput (GPU decode), cross-court generalization,
    partner/opponent role-awareness.

---

Generated at session end on June 14, 2026.

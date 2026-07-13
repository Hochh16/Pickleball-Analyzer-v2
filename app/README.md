# Setup Wizard (app/) — Phase 1

A local web app that replaces the Tkinter `mark_court.py` / `mark_user.py` tools.
It guides a user through setting up an analysis: pick a video, mark the court,
set up players, optionally point themselves out, and review — writing the exact
input JSONs the pipeline already consumes. Data contracts are unchanged.

See `docs/UI_PLAN.md` for the scoped plan and milestones.

## Run

```
python -m app
```

Opens `http://127.0.0.1:8000` in your browser. Options:

- `--port 8000` / `--host 127.0.0.1` (or env `PB_APP_PORT` / `PB_APP_HOST`)
- `--no-browser` — don't auto-open the browser
- `--reload` — dev auto-reload

Analyses are written under the data root (default `./data`, override with
`PB_DATA_DIR`), one folder per video: `data/<name>/`. Videos are picked from a
single designated drop folder (default `./videos`, override with
`PB_VIDEOS_DIR`) — the user copies a clip there and selects it (no filesystem
browsing). The Video step also lists the on-screen recording requirements
(camera position/height/shutter/fps/resolution/ISO/length).

## What it produces (Phase 1)

For each session folder it writes the same files the pipeline reads:

| File | Written by | Consumed by |
|---|---|---|
| `markers.json` | Court step | Stage 1 (calibrate) |
| `court.json`, `court_zones.json` | Stage 1, run in-process | Stages 2+ |
| `roster.json` | Players step | Stage 6 (handedness) |
| `session.json` | app | the app (video path, meta, progress) |

The **You** step is a visual left/right side pick (which side you start on) that
patches `user_starting_corner` into `markers.json` + `court.json` — Stage 2.5's
geometric user seed. (The optional per-frame `user_clicks.json` override endpoint
still exists in the backend but the wizard no longer needs it — the no-click
geometric seed is the default flow.)

## Phase 2 — run & progress

After setup, **Start Analysis** kicks off the pipeline as a background job and
opens the **Run** view (live per-stage progress + activity log via SSE):

- Materializes `video.mp4` into the session folder (hardlink; copy fallback).
- Runs Stages 2 → 2.5 → 3 locally (`track_players`, `classify_tracks`, `pose`).
- **Pauses at Stage 4 (ball detection = GPU)** and waits for `ball.parquet`.
  Since there's no local GPU, this step runs on Colab; the run **auto-resumes**
  the instant `ball.parquet` is uploaded back (decoupling ball production from
  the rest of the run).
- Resumes Stages 5–11 + `compress_video` + `build_report`, then serves
  `report.html` + the annotated video in-app.

`PB_FAKE_STAGES=1` simulates the stages quickly (no GPU/long wait) to preview
the Run UI end-to-end. `pipeline.py` owns the runner; endpoints: `POST …/run`,
`GET …/run` (+ `…/run/stream` SSE), `POST …/ball`, `GET …/report`, `…/annotated`.

> Real full runs are long: Stages 2/3 on a 4K/2-min clip are minutes-to-hours on
> CPU (see KNOWN_ISSUES C8 throughput). The run is a background job — leave it going.

## Layout

- `server.py` — FastAPI routes (sessions, upload, frame, calibrate, roster,
  user-clicks, summary, browse) + static SPA mount.
- `sessions.py` — per-video folders; writes the input JSONs; calls
  `stages.calibrate.calibrate()` in-process.
- `video.py` — exact source-frame JPEG serving (OpenCV), the same frame
  indexing the pipeline uses.
- `browse.py` — lists videos in the designated drop folder.
- `static/` — the vanilla-JS single-page wizard (`index.html`, `styles.css`,
  `app.js`).
- `test_app.py` — backend smoke tests (`pytest app/test_app.py -q`).

## Notes

- **Frame indexing is backend-side** (OpenCV), so every marked coordinate maps
  to an exact original-video pixel — critical because all downstream geometry
  depends on it.
- **Picking a video:** copy the clip into the designated `PB_VIDEOS_DIR` folder
  (shown in the UI), then select it. No filesystem browsing or multi-GB upload.

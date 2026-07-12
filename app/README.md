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
`PB_DATA_DIR`), one folder per video: `data/<name>/`.

## What it produces (Phase 1)

For each session folder it writes the same files the pipeline reads:

| File | Written by | Consumed by |
|---|---|---|
| `markers.json` | Court step | Stage 1 (calibrate) |
| `court.json`, `court_zones.json` | Stage 1, run in-process | Stages 2+ |
| `roster.json` | Players step | Stage 6 (handedness) |
| `user_clicks.json` | You step (optional) | Stage 2.5 (user seed override) |
| `session.json` | app | the app (video path, meta, progress) |

The GPU ball step (Stage 4) and full run orchestration are **Phase 2** (see the
plan). Phase 1 is the setup wizard only.

## Layout

- `server.py` — FastAPI routes (sessions, upload, frame, calibrate, roster,
  user-clicks, summary, browse) + static SPA mount.
- `sessions.py` — per-video folders; writes the input JSONs; calls
  `stages.calibrate.calibrate()` in-process.
- `video.py` — exact source-frame JPEG serving (OpenCV), the same frame
  indexing the pipeline uses.
- `browse.py` — server-side local video file browser.
- `static/` — the vanilla-JS single-page wizard (`index.html`, `styles.css`,
  `app.js`).
- `test_app.py` — backend smoke tests (`pytest app/test_app.py -q`).

## Notes

- **Frame indexing is backend-side** (OpenCV), so every marked coordinate maps
  to an exact original-video pixel — critical because all downstream geometry
  depends on it.
- **Two ways to pick a video:** browse the local machine (the "server" is your
  laptop) for a clip you've already copied over, or upload one via the file
  picker (works off a camera/SD card, but a 4K clip is multi-GB and slow).

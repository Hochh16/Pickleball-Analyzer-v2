# Input / Setup UI — scoped plan (2026-07-11)

The build program's **step 4 (COMPLETE the UI)**. The consumer *report* is done
(`tools/build_report.py`); this plan covers the front end a user drives to *run*
the app: pick a video, set up the court + players, launch processing, and land on
the report. Scoped with the operator; decisions locked below. Build against this.

## Operator decisions (2026-07-11)

1. **Deployment: local guided web UI + Colab hand-off.** A browser-based UI served
   by a local backend replaces the Tkinter tools and orchestrates the local
   pipeline stages; the GPU ball-detection step (Stage 4) stays a guided Colab
   hand-off (semi-automated, as today). Reuses the whole pipeline, honest about the
   GPU limit, evolves toward cloud later.
2. **Audience: early outside users.** Setup + report must be usable by a tester with
   no instructions — higher bar on guidance, validation, and error handling.

**Named tension + resolution.** "Local + Colab hand-off" vs "outside users" partly
conflict: an outside tester can't easily run a Colab notebook. v1 resolves this by
polishing the **setup wizard + report** to an outside-user bar, while the **GPU
processing step stays operator-assisted** (the operator runs the ball step, or a
shared Colab/GPU is used) — the one non-self-serve part. A cloud GPU service (later)
makes it fully self-serve.

## Stack

A **local web app** (matches the pipeline's Python + the Stage-1 contract's original
FastAPI + web-frontend intent, which the Tkinter tools were a stopgap for):

- **Backend: FastAPI** (Python) — serves the UI, serves video frames for marking,
  writes the input JSONs, runs the pipeline stages as background jobs with progress,
  invokes `build_report.py` + `compress_video.py`, serves the finished report/video.
  Calls the existing stage `main()` functions directly (no reimplementation).
- **Frontend: a single-page browser UI** (vanilla or a light framework; no heavy
  build chain) — the setup wizard (canvas court-marking on served frames), a run/
  progress view, and the report view. Run locally: `python -m app` opens
  `localhost:PORT`.

## The three surfaces

### 1. Setup wizard (highest-value first build — replaces the Tkinter tools)
Guided, validated, outside-user-ready:
1. **Video** — select/upload a local file; backend registers it + probes fps/res/
   duration.
2. **Court marking** — scrub to a clear frame; click the 8 points (4 court corners +
   2 user-kitchen + 2 opponent-kitchen) on a canvas with live dots/lines, a
   zoom-loupe, undo, and a homography-sanity check (reproject + show error) before
   accept. Writes `markers.json` → runs Stage 1 → `court.json` + `court_zones.json`.
3. **Player setup** — dominant hand, which baseline you're on, starting corner
   (the 3 dropdowns today), plus optional partner/opponent handedness → `roster.json`.
4. **Identify yourself (optional)** — "tap yourself in a few frames" across ~5
   spread frames → `user_clicks.json`; skippable (geometric seed fallback).
5. **Review & confirm** — summary of everything, then launch.

Data contracts are unchanged — the wizard produces the exact files the pipeline
already consumes (`markers.json`, `court.json`, `roster.json`, `user_clicks.json`),
so nothing downstream changes.

### 2. Run & progress
- Backend runs Stages 1–3 (calibrate, track, roles, pose) locally as a job, streams
  per-stage progress (SSE/websocket), handles failures with clear messages.
- **Stage 4 (ball) hand-off** — the pipeline pauses; the UI shows a guided step:
  "your video needs GPU ball detection." v1 path = **guided Colab** (download the
  bundle, a walkthrough, upload the resulting `ball.parquet` back) OR, if a local
  CUDA GPU is present, run `infer_v4` locally. This is the one assisted step.
- On `ball.parquet` present, backend resumes Stages 5–11, then `build_report.py` +
  `compress_video.py`.

### 3. Report
- Serve the finished `report.html` + `annotated_web.mp4` in-app.
- A **library** view: list past analyses (per-folder), re-open reports, compare.

## Milestones

1. **Phase 1 — Setup wizard** ✅ **DONE (2026-07-12, branch `feat/setup-ui-phase1`,
   commits `6cf935f` backend + `69b42b2` front end).** New `app/` package: FastAPI
   backend (OpenCV exact-frame serving, per-video folder management, Stage 1 called
   in-process, server-side file browser) + a vanilla-JS 5-step SPA (Video → Court →
   Players → You → Review). Writes `markers.json`/`court.json`/`court_zones.json`/
   `roster.json`/`user_clicks.json` — contracts unchanged. Validated end-to-end in a
   real browser on the 4K pb_2min clip + 8 pytest smoke tests (`app/test_app.py`).
   Run: `python -m app`. See `app/README.md`.
2. **Phase 2 — Run & progress + the Stage-4 hand-off.** Job orchestration, progress
   streaming, the guided GPU step, resume-and-finish, report generation.
3. **Phase 3 — Library + polish.** Past-analysis list, error states, empty states,
   responsive/theme, first-run guidance.

## Risks / open items

- **GPU step UX** is the hard part for outside users (the tension above) — the
  guided-Colab flow needs care, or a shared GPU. Cloud GPU is the real fix (later).
- **Throughput** — even on GPU, inference is decode-bound (~40 min / 2-min clip;
  KNOWN_ISSUES C8); progress UI must set expectations. GPU-decode (NVDEC) is the
  standing speedup.
- **Video upload size** — 4K clips are large; local file access avoids upload, but a
  cloud version needs chunked upload + storage.
- **Multi-clip / cross-venue** — the operator's real workload; the library should
  support multiple clips per player and (later) cross-video identity (F28).

"""Pipeline job runner for the setup wizard (Phase 2).

Runs the analysis pipeline for a session as a background job, each stage an
isolated subprocess (`python -m stages.<x>.<x> <folder>`) so a crash can't take
down the server and stdout/stderr stream into a live log.

The HEAVY VISION stages (2 track_players -> 2.5 classify_tracks -> 3 pose ->
4 track_ball) are the throughput wall on a CPU machine (~1 fps for tracking ->
hours per 5-min clip). So the run:
  - prepare : materialize video.mp4 (fast, local)
  - vision  : if a local CUDA GPU is present, run the 4 vision stages locally;
              otherwise PAUSE as a GPU hand-off — the operator runs the combined
              vision pass on Colab (stages/infer_vision.ipynb) and uploads the
              outputs (players/track_roles/poses/ball + metas) back, which
              AUTO-RESUMES the run (same decouple mechanism the ball step used).
  - post    : 5 detect_shots -> 5.5 detect_bounces -> 5.7 ball_trajectory ->
              6 classify_shots -> 7 segment_rallies -> 8 compute_metrics -> 9 rate ->
              10 plan_improvement -> build_report (all light/local). The Stage-11
              annotated-overlay render is intentionally skipped: the box overlay
              added little value, so the report links the original clip instead.
              (Revisit later with a real ball-trail / shot-label render — see
              docs/USABILITY_BACKLOG.md.)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_TAIL = 240  # lines kept per job for the live log

# The vision outputs the post stages need — their presence AUTO-RESUMES the run.
VISION_OUTPUTS = ("players.parquet", "track_roles.json", "poses.parquet",
                  "ball.parquet", "ball.meta.json")

# Colab backs up each vision stage's output to Drive AS IT FINISHES, so the
# watcher can reflect per-stage progress live in the UI while Colab runs.
VISION_FILE_TO_STEP = {"players.parquet": "track", "track_roles.json": "roles",
                       "poses.parquet": "pose", "ball.parquet": "ball"}


class Step:
    def __init__(self, key: str, label: str, module: Optional[str] = None,
                 func: Optional[Callable[[Path], None]] = None,
                 args: Optional[List[str]] = None):
        self.key = key
        self.label = label
        self.module = module
        self.func = func
        self.args = args or []   # extra CLI args after the folder


PREPARE_STEP = Step("video", "Prepare video")
# The heavy vision stages — run locally only with a GPU, else produced on Colab.
VISION_STEPS = [
    Step("track", "Track players", module="stages.track_players.track"),
    Step("roles", "Identify players", module="stages.classify_tracks.classify_tracks", args=["--force"]),
    Step("pose", "Body pose", module="stages.pose.pose"),
    Step("ball", "Ball detection", module="stages.track_ball.track_ball_v4",
         args=["--weights", "data/models/ball_model_v4.pt", "--force"]),
]
# --force where supported so re-runs overwrite prior outputs. The Stage-11
# annotated render + compress are intentionally omitted (see module docstring);
# the report links the original clip.
POST_STEPS = [
    Step("shots", "Detect shots", module="stages.detect_shots.detect_shots", args=["--force"]),
    Step("bounces", "Detect bounces", module="stages.detect_bounces.detect_bounces", args=["--force"]),
    Step("trajectory", "Ball trajectory", module="stages.ball_trajectory.ball_trajectory", args=["--force"]),
    Step("classify", "Classify shots", module="stages.classify_shots.classify_shots", args=["--force"]),
    Step("rallies", "Segment rallies", module="stages.segment_rallies.segment_rallies", args=["--force"]),
    Step("metrics", "Compute metrics", module="stages.compute_metrics.compute_metrics", args=["--force"]),
    Step("rate", "USAPA rating", module="stages.rate.rate", args=["--force"]),
    Step("plan", "Improvement plan", module="stages.plan_improvement.plan_improvement", args=["--force"]),
    Step("report", "Build report", module="tools.build_report", args=["--force"]),
]

ALL_STEPS = [PREPARE_STEP] + VISION_STEPS + POST_STEPS
_VISION_KEYS = [s.key for s in VISION_STEPS]


def _cuda_available() -> bool:
    if os.environ.get("PB_FAKE_STAGES"):
        return True  # preview mode runs the (fake) vision stages locally
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def has_vision_outputs(folder: Path) -> bool:
    return all((folder / f).exists() for f in VISION_OUTPUTS)


class Job:
    """Live state of one session's pipeline run."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # status: pending | running | done | failed | skipped | waiting
        self.steps: List[Dict] = [
            {"key": s.key, "label": s.label, "status": "pending",
             "started_at": None, "ended_at": None, "returncode": None}
            for s in ALL_STEPS
        ]
        self.phase = "idle"     # idle | prepare | vision | post | done | failed
        self.error: Optional[str] = None
        self.log: Deque[str] = deque(maxlen=LOG_TAIL)
        self.version = 0
        self._proc: Optional[subprocess.Popen] = None
        self._cancel = False

    def _step(self, key: str) -> Dict:
        return next(s for s in self.steps if s["key"] == key)

    def snapshot(self) -> Dict:
        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "error": self.error,
            "version": self.version,
            "steps": [dict(s) for s in self.steps],
            "log": list(self.log),
        }


class PipelineRunner:
    def __init__(self, store, drivesync=None):
        self.store = store
        self.drivesync = drivesync   # optional Drive-for-Desktop auto-sync
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()

    # ---- public API ----

    def get(self, session_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(session_id)

    def start(self, session_id: str) -> Job:
        """Begin the run (prepare -> vision) in a background thread."""
        folder = self.store.folder(session_id)
        if not (folder / "court.json").exists():
            raise RuntimeError("Court isn't set up yet — finish setup first.")
        with self.lock:
            job = self.jobs.get(session_id)
            if job and job.phase in ("prepare", "vision", "post"):
                return job  # already running
            job = Job(session_id)
            self.jobs[session_id] = job
        threading.Thread(target=self._run_prepare_and_vision, args=(job,), daemon=True).start()
        return job

    def resume_post(self, session_id: str) -> Optional[Job]:
        """Start the post phase — called when the vision outputs are present."""
        folder = self.store.folder(session_id)
        if not has_vision_outputs(folder):
            return None
        with self.lock:
            job = self.jobs.get(session_id)
            if job is None:
                job = Job(session_id)
                self.jobs[session_id] = job
            if job.phase == "post":
                return job
            for key in ["video"] + _VISION_KEYS:  # vision done out of band
                self._set(job, key, "done", bump=False)
        threading.Thread(target=self._run_post, args=(job,), daemon=True).start()
        return job

    def cancel(self, session_id: str) -> None:
        with self.lock:
            job = self.jobs.get(session_id)
            if not job:
                return
            job._cancel = True
            if job._proc and job._proc.poll() is None:
                job._proc.terminate()

    # ---- Drive-for-Desktop auto-sync (optional) ----

    def _maybe_drive_sync(self, job: Job) -> None:
        """If a synced Drive folder is configured, push the clip bundle into it and
        start a background watcher that auto-resumes when the vision outputs land
        back — so the operator only runs the Colab notebook (no download/upload)."""
        ds = self.drivesync
        if not (ds and ds.enabled()):
            return
        sid = job.session_id
        folder = self.store.folder(sid)
        try:
            bundle = folder / f"{sid}_vision_input.zip"
            self.store.build_vision_input_zip(sid, bundle)
            dest = ds.push_bundle(sid, bundle)
        except Exception as e:  # noqa: BLE001
            self._log(job, f"Drive auto-sync unavailable ({e}); use the manual buttons.")
            return
        self._log(job, f"Clip synced to Google Drive as {dest.name}. Open the Colab "
                       "notebook and Run all — the results import automatically.")
        threading.Thread(target=self._watch_drive_outputs, args=(job,), daemon=True).start()

    def _sync_vision_progress(self, job: Job) -> None:
        """Flip vision steps to done as Colab's per-stage backups land on Drive —
        live progress in the UI while the notebook runs."""
        d = self.drivesync.outputs_dir(job.session_id)
        for fname, key in VISION_FILE_TO_STEP.items():
            step = job._step(key)
            if step["status"] != "done" and (d / fname).exists():
                self._log(job, f"{step['label']}: finished on Colab.")
                self._set(job, key, "done")

    def _watch_drive_outputs(self, job: Job) -> None:
        ds = self.drivesync
        sid = job.session_id
        folder = self.store.folder(sid)
        poll = float(os.environ.get("PB_DRIVE_POLL", "5"))
        settle = float(os.environ.get("PB_DRIVE_SETTLE", "3"))
        while not job._cancel and job.phase == "vision":
            self._sync_vision_progress(job)
            if ds.outputs_ready(sid):
                time.sleep(settle)          # let Drive finish syncing every file
                if not ds.outputs_ready(sid):
                    continue
                got = ds.ingest_outputs(sid, folder)
                self._log(job, f"Vision results synced from Drive ({len(got)} files). Resuming.")
                self.store.ensure_ball_meta(sid)
                self.resume_post(sid)
                return
            time.sleep(poll)

    # ---- internals ----

    def _bump(self, job: Job) -> None:
        job.version += 1

    def _set(self, job: Job, key: str, status: str, rc: Optional[int] = None, bump: bool = True) -> None:
        s = job._step(key)
        s["status"] = status
        if status == "running":
            s["started_at"] = time.time()
        if status in ("done", "failed", "skipped"):
            s["ended_at"] = time.time()
        if rc is not None:
            s["returncode"] = rc
        if bump:
            self._bump(job)

    def _log(self, job: Job, line: str) -> None:
        job.log.append(line.rstrip("\n"))
        self._bump(job)

    def _run_steps(self, job: Job, steps: List[Step]) -> bool:
        folder = self.store.folder(job.session_id)
        for step in steps:
            if job._cancel:
                self._set(job, step.key, "skipped")
                continue
            self._set(job, step.key, "running")
            self._log(job, f"── {step.label} ──")
            try:
                if step.key == "video":
                    self._materialize_video(job)
                    ok, rc = True, 0
                else:
                    rc = self._run_module(job, step.module, folder, list(step.args))
                    ok = (rc == 0)
            except Exception as e:  # noqa: BLE001
                self._log(job, f"ERROR: {type(e).__name__}: {e}")
                ok, rc = False, -1
            if ok:
                self._set(job, step.key, "done", rc=rc)
            else:
                self._set(job, step.key, "failed", rc=rc)
                job.error = f"{step.label} failed"
                job.phase = "failed"
                self._bump(job)
                return False
        return True

    def _run_prepare_and_vision(self, job: Job) -> None:
        job.phase = "prepare"
        self._bump(job)
        if not self._run_steps(job, [PREPARE_STEP]):
            return
        folder = self.store.folder(job.session_id)
        if has_vision_outputs(folder):
            # already produced (re-run) — go straight to post
            for key in _VISION_KEYS:
                self._set(job, key, "done")
            self._run_post(job)
            return
        job.phase = "vision"
        self._bump(job)
        if _cuda_available():
            self._log(job, "Local GPU detected — running vision stages locally.")
            if not self._run_steps(job, VISION_STEPS):
                return
            self._run_post(job)
        else:
            for key in _VISION_KEYS:
                self._set(job, key, "waiting", bump=False)
            self._log(job, "Vision (tracking, pose, ball) needs a GPU. Run the Colab "
                           "notebook (Run all) to produce the outputs.")
            self._bump(job)
            self._maybe_drive_sync(job)

    def _run_post(self, job: Job) -> None:
        job.phase = "post"
        self._bump(job)
        if not self._run_steps(job, POST_STEPS):
            return
        job.phase = "done"
        self._log(job, "Analysis complete. Report ready.")
        self._bump(job)

    def _run_module(self, job: Job, module: str, folder: Path,
                    extra_args: Optional[List[str]] = None) -> int:
        if os.environ.get("PB_FAKE_STAGES"):
            return self._fake_module(job, module, folder)
        cmd = [sys.executable, "-u", "-m", module, str(folder), *(extra_args or [])]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        job._proc = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            self._log(job, line)
        proc.wait()
        job._proc = None
        return proc.returncode

    def _fake_module(self, job: Job, module: str, folder: Path) -> int:
        """Simulate a stage quickly for UI preview (PB_FAKE_STAGES)."""
        import json
        for i in range(3):
            self._log(job, f"[preview] {module} … {(i + 1) * 33}%")
            time.sleep(0.3)
        if os.environ.get("PB_FAKE_FAIL", "") and os.environ["PB_FAKE_FAIL"] in module:
            self._log(job, f"[preview] simulated failure in {module}")
            return 1
        # produce stub artifacts so downstream 'file exists' checks pass
        stubs = {
            "track_players.track": [("players.parquet", b"PAR1")],
            "classify_tracks.classify_tracks": [("track_roles.json", b'{"roles":{}}')],
            "pose.pose": [("poses.parquet", b"PAR1")],
            "track_ball.track_ball_v4": [("ball.parquet", b"PAR1"),
                                         ("ball.meta.json", b'{"synthetic": false}')],
            "render.render": [("annotated.mp4", b"\x00")],
            "compress_video": [("annotated_web.mp4", b"\x00")],
            "build_report": [("report.html", b"<h1>Preview report</h1>")],
        }
        for suffix, files in stubs.items():
            if module.endswith(suffix):
                for name, data in files:
                    (folder / name).write_bytes(data)
        return 0

    def _materialize_video(self, job: Job) -> None:
        """Ensure data/<id>/video.mp4 exists (hardlink; copy fallback)."""
        folder = self.store.folder(job.session_id)
        dest = folder / "video.mp4"
        if dest.exists():
            self._log(job, "video.mp4 already in place")
            return
        src = self.store.video_path(job.session_id)
        if not src.exists():
            raise FileNotFoundError(f"Source video missing: {src}")
        try:
            os.link(src, dest)
            self._log(job, f"Linked video.mp4 from {src.name}")
        except OSError:
            self._log(job, f"Copying video into the analysis folder ({src.name})…")
            shutil.copy2(src, dest)
            self._log(job, "Copy complete")

"""Pipeline job runner for the setup wizard (Phase 2).

Runs the analysis pipeline for a session as a background job, one stage at a
time, each as an isolated subprocess (`python -m stages.<x>.<x> <folder>`) so a
crash can't take down the server and stdout/stderr stream cleanly into a live
log. The chain pauses at Stage 4 (ball detection = GPU/Colab hand-off) and
AUTO-RESUMES the moment `ball.parquet` appears in the folder — decoupling ball
production (guided Colab / cloud GPU / operator) from the rest of the run.

Stages (folder-based CLIs, called with the per-video folder as the sole arg):
  pre-ball:  2 track_players -> 2.5 classify_tracks -> 3 pose
  [ball]:    4 track_ball  (GPU; produced out-of-app, uploaded back)
  post-ball: 5 detect_shots -> 5.5 detect_bounces -> 6 classify_shots ->
             7 segment_rallies -> 8 compute_metrics -> 9 rate ->
             10 plan_improvement -> 11 render -> compress_video -> build_report
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


# A step is either a module CLI (module + folder arg) or a python callable.
class Step:
    def __init__(self, key: str, label: str, module: Optional[str] = None,
                 func: Optional[Callable[[Path], None]] = None,
                 args: Optional[List[str]] = None):
        self.key = key
        self.label = label
        self.module = module
        self.func = func
        # extra CLI args after the folder (e.g. --force so re-runs overwrite)
        self.args = args or []


# pre-ball (local), then the ball marker, then post-ball (local)
PRE_STEPS = [
    Step("video", "Prepare video", func=None),  # materialize video.mp4 (special)
    Step("track", "Track players", module="stages.track_players.track"),
    Step("roles", "Identify players", module="stages.classify_tracks.classify_tracks"),
    Step("pose", "Body pose", module="stages.pose.pose"),
]
BALL_STEP = Step("ball", "Ball detection (GPU)", func=None)
# --force where supported so a re-run overwrites prior outputs (render fails
# outright on an existing annotated.mp4 without it).
POST_STEPS = [
    Step("shots", "Detect shots", module="stages.detect_shots.detect_shots", args=["--force"]),
    Step("bounces", "Detect bounces", module="stages.detect_bounces.detect_bounces", args=["--force"]),
    Step("classify", "Classify shots", module="stages.classify_shots.classify_shots", args=["--force"]),
    Step("rallies", "Segment rallies", module="stages.segment_rallies.segment_rallies", args=["--force"]),
    Step("metrics", "Compute metrics", module="stages.compute_metrics.compute_metrics", args=["--force"]),
    Step("rate", "USAPA rating", module="stages.rate.rate", args=["--force"]),
    Step("plan", "Improvement plan", module="stages.plan_improvement.plan_improvement", args=["--force"]),
    Step("render", "Annotated video", module="stages.render.render", args=["--force"]),
    Step("compress", "Compress video", module="tools.compress_video"),
    Step("report", "Build report", module="tools.build_report", args=["--force"]),
]

ALL_STEPS = PRE_STEPS + [BALL_STEP] + POST_STEPS


class Job:
    """Live state of one session's pipeline run (thread-safe via the runner lock)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # status: pending | running | done | failed | skipped | waiting
        self.steps: List[Dict] = [
            {"key": s.key, "label": s.label, "status": "pending",
             "started_at": None, "ended_at": None, "returncode": None}
            for s in ALL_STEPS
        ]
        self.phase = "idle"     # idle | pre | ball | post | done | failed
        self.error: Optional[str] = None
        self.log: Deque[str] = deque(maxlen=LOG_TAIL)
        self.version = 0        # bumped on any change (for SSE polling)
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
    def __init__(self, store):
        self.store = store          # SessionStore (for folder + video_path)
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()

    # ---- public API ----

    def get(self, session_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(session_id)

    def start(self, session_id: str) -> Job:
        """Begin (or restart) the pre-ball phase in a background thread."""
        folder = self.store.folder(session_id)
        if not (folder / "court.json").exists():
            raise RuntimeError("Court isn't set up yet — finish setup first.")
        with self.lock:
            job = self.jobs.get(session_id)
            if job and job.phase in ("pre", "post"):
                return job  # already running
            job = Job(session_id)
            self.jobs[session_id] = job
        threading.Thread(target=self._run_pre, args=(job,), daemon=True).start()
        return job

    def resume_post(self, session_id: str) -> Optional[Job]:
        """Start the post-ball phase (called when ball.parquet is present)."""
        folder = self.store.folder(session_id)
        if not (folder / "ball.parquet").exists():
            return None
        with self.lock:
            job = self.jobs.get(session_id)
            if job is None:
                job = Job(session_id)
                # pre-ball already done out of band — mark those done
                for key in ("video", "track", "roles", "pose", "ball"):
                    self._set(job, key, "done", bump=False)
                self.jobs[session_id] = job
            if job.phase == "post":
                return job  # already running
            # mark the ball step done
            self._set(job, "ball", "done")
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
        """Run a list of steps sequentially. Returns False on failure/cancel."""
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
                    extra = list(step.args)
                    # optional preview cap for the slow full render
                    cap = os.environ.get("PB_RENDER_MAX_SECONDS")
                    if step.key == "render" and cap:
                        extra += ["--max-seconds", str(cap)]
                    rc = self._run_module(job, step.module, folder, extra)
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

    def _run_pre(self, job: Job) -> None:
        job.phase = "pre"
        self._bump(job)
        if not self._run_steps(job, PRE_STEPS):
            return
        # pause for the ball hand-off
        self._set(job, "ball", "waiting")
        job.phase = "ball"
        self._log(job, "Waiting for ball detection (GPU / Colab). Upload ball.parquet to continue.")
        self._bump(job)
        # auto-resume if ball.parquet already exists (e.g. re-run)
        if (self.store.folder(job.session_id) / "ball.parquet").exists():
            self.resume_post(job.session_id)

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
        # Preview/dev hook: simulate stages fast (no GPU/long wait). Lets the run
        # UI be exercised end-to-end. Never set in a real run.
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
        for i in range(3):
            self._log(job, f"[preview] {module} … {(i + 1) * 33}%")
            time.sleep(0.4)
        fail = os.environ.get("PB_FAKE_FAIL", "")
        if fail and fail in module:
            self._log(job, f"[preview] simulated failure in {module}")
            return 1
        # produce stub artifacts so downstream 'file exists' checks & report links work
        if module.endswith("render"):
            (folder / "annotated.mp4").write_bytes(b"\x00")
        if module.endswith("build_report"):
            (folder / "report.html").write_text("<h1>Preview report</h1>", encoding="utf-8")
        return 0

    def _materialize_video(self, job: Job) -> None:
        """Ensure data/<id>/video.mp4 exists (hardlink the picked clip; copy fallback)."""
        folder = self.store.folder(job.session_id)
        dest = folder / "video.mp4"
        if dest.exists():
            self._log(job, "video.mp4 already in place")
            return
        src = self.store.video_path(job.session_id)
        if not src.exists():
            raise FileNotFoundError(f"Source video missing: {src}")
        try:
            os.link(src, dest)  # hardlink: instant, no extra space (same volume)
            self._log(job, f"Linked video.mp4 from {src.name}")
        except OSError:
            self._log(job, f"Copying video into the analysis folder ({src.name})…")
            shutil.copy2(src, dest)
            self._log(job, "Copy complete")

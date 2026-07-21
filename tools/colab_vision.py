"""Colab vision-pass orchestration — Stages 2 -> 2.5 -> 3 -> 4 on GPU.

Imported from the repo that `stages/infer_vision.ipynb` clones on Colab, so the
notebook is a dead-simple "Run All" and code changes need only a `git pull` (no
bundle to rebuild/upload). This logic is a real, testable module.

Reliability goals (docs/USABILITY_BACKLOG.md, B-block):
  - **B1** auto-derive CLIP from the single `<clip>_vision_input.zip` on Drive —
    nothing to type or hunt for.
  - **B3** pull the clip + weights to LOCAL disk once, robustly (remount retries) —
    never random-read the multi-GB bundle over Drive FUSE.
  - **B4** back up each stage's outputs to Drive as it finishes, and RESUME from
    there: a runtime reset -> re-run skips the stages already done (and skips the
    whole bundle copy if everything's done) instead of starting over.
  - **B2** GPU self-management: free CUDA memory between stages + ball OOM
    auto-fallback down a batch ladder.

The heavy stages run as SUBPROCESSES (`python -m stages.<x>`), so a crash can't
kill the kernel and their GPU memory is released on exit. This module only
orchestrates; it imports nothing from `stages`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# The 5 outputs the local post-stages require; presence of `required` means a
# stage is done. Sidecars (players_pending / pose_summary) are backed up too.
REQUIRED_OUTPUTS = ["players.parquet", "track_roles.json", "poses.parquet",
                    "ball.parquet", "ball.meta.json"]
ALL_OUTPUTS = REQUIRED_OUTPUTS + ["players_pending.json", "pose_summary.json"]

# Stage table (order matters: 2 -> 2.5 -> 3 -> 4).
STAGES = [
    {"name": "track_players", "module": "stages.track_players.track", "args": [],
     "required": "players.parquet", "outputs": ["players.parquet", "players_pending.json"]},
    {"name": "classify_tracks", "module": "stages.classify_tracks.classify_tracks",
     "args": ["--force"], "required": "track_roles.json", "outputs": ["track_roles.json"]},
    {"name": "pose", "module": "stages.pose.pose", "args": [],
     "required": "poses.parquet", "outputs": ["poses.parquet", "pose_summary.json"]},
    {"name": "ball", "module": "stages.track_ball.track_ball_v4", "args": ["--force"],
     "required": "ball.parquet", "outputs": ["ball.parquet", "ball.meta.json"],
     "gpu_batch": True},
]
BALL_BATCHES = [8, 4, 2, 1]   # ball OOM auto-fallback ladder (largest first)


# ---------------------------------------------------------------- CLIP / Drive

def derive_clip(drive_dir, forced: Optional[str] = None) -> str:
    """Clip name from the single `<clip>_vision_input.zip` on Drive (or `forced`).
    Clear, actionable errors on zero / multiple bundles."""
    if forced:
        return forced
    suffix = "_vision_input.zip"
    zips = sorted(p for p in Path(drive_dir).glob(f"*{suffix}"))
    if not zips:
        raise SystemExit(
            f"No *{suffix} found on Drive root ({drive_dir}). Download the clip "
            "bundle from the app's run screen and upload it to My Drive.")
    if len(zips) > 1:
        names = ", ".join(z.name for z in zips)
        raise SystemExit(
            f"Multiple clip bundles on Drive ({names}). Set CLIP='<name>' to pick "
            "one, or remove the ones you don't want.")
    return zips[0].name[: -len(suffix)]


def _remount() -> None:
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=True)
    except Exception:  # noqa: BLE001  (only runs on Colab; best-effort)
        pass


def robust_copy(src, dst, tries: int = 4) -> None:
    """Copy a (possibly multi-GB) file off Drive FUSE, retrying with a force
    remount — sequential copies survive the FUSE drops that random reads don't."""
    src, dst = str(src), str(dst)
    for i in range(tries):
        try:
            shutil.copyfile(src, dst)
            return
        except OSError as e:
            print(f"  copy retry {i + 1}/{tries}: {e}", flush=True)
            _remount()
            time.sleep(3)
    raise RuntimeError(f"could not copy {src} -> {dst} after {tries} tries")


def free_gpu() -> None:
    try:
        import gc
        import torch  # type: ignore
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- resume / inputs

def wait_for_complete_zip(path, tries: int = 120, wait: float = 30.0) -> None:
    """Block until `path` is a complete zip archive. A bundle freshly dropped into
    the synced Drive folder may still be UPLOADING when the notebook starts —
    reading it too early yields a partial file (`BadZipFile`). The end-of-archive
    check reads only the file tail, so polling is cheap even over Drive FUSE."""
    for i in range(tries):
        try:
            if zipfile.is_zipfile(str(path)):
                return
        except OSError:
            _remount()
        print(f"  {Path(path).name} isn't fully uploaded to Drive yet — "
              f"retrying in {wait:.0f}s ({i + 1}/{tries})", flush=True)
        time.sleep(wait)
    raise SystemExit(
        f"{path} never became a complete zip. Check that Google Drive finished "
        "uploading it (tray icon), then Run all again — the run will resume.")


def restore_outputs(backup_dir, clip_dir) -> List[str]:
    """Copy any already-computed outputs from the Drive backup into the local clip
    dir (the resume mechanism). Returns the names restored."""
    backup_dir, clip_dir = Path(backup_dir), Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    restored: List[str] = []
    if not backup_dir.is_dir():
        return restored
    for name in ALL_OUTPUTS:
        bp = backup_dir / name
        if bp.exists():
            robust_copy(bp, clip_dir / name)
            restored.append(name)
    return restored


def have_all_required(clip_dir) -> bool:
    return all((Path(clip_dir) / f).exists() for f in REQUIRED_OUTPUTS)


def copy_inputs(drive_dir, clip: str, content="/content") -> Tuple[Path, Path]:
    """Copy the clip bundle (video + setup) and the ball weights to local disk and
    unzip the clip. Returns (clip_dir, weights_path)."""
    drive_dir, content = Path(drive_dir), Path(content)
    clip_dir = content / clip
    clip_dir.mkdir(parents=True, exist_ok=True)
    src = drive_dir / f"{clip}_vision_input.zip"
    zip_local = content / f"{clip}_vision_input.zip"
    wait_for_complete_zip(src)   # Drive may still be uploading the bundle
    print("copying clip bundle to local disk (may be several GB)...", flush=True)
    for attempt in (1, 2):
        robust_copy(src, zip_local)
        try:
            with zipfile.ZipFile(zip_local) as z:
                z.extractall(clip_dir)
            break
        except zipfile.BadZipFile:
            if attempt == 2:
                raise
            print("  copied bundle was incomplete — retrying the copy once...", flush=True)
    if not (clip_dir / "video.mp4").exists():
        raise RuntimeError(f"clip bundle for {clip} has no video.mp4")
    weights = content / "ball_model_v4.pt"
    if not weights.exists():
        print("copying ball weights to local disk...", flush=True)
        robust_copy(drive_dir / "ball_model_v4.pt", weights)
    return clip_dir, weights


# ---------------------------------------------------------------- stage runner

def _run(cmd, cwd, env=None) -> Tuple[int, str]:
    """Run a subprocess (cwd = the cloned repo so `stages` imports), streaming its
    output live; return (rc, captured_output)."""
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def _run_ball_with_fallback(base, args, cwd) -> Tuple[int, str]:
    """Run the ball stage, auto-falling back down the batch ladder on CUDA OOM.
    A non-OOM failure stops immediately (no point retrying)."""
    last: Tuple[int, str] = (1, "")
    for b in BALL_BATCHES:
        free_gpu()
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        print(f"  [ball] batch {b}", flush=True)
        rc, out = _run(base + args + ["--batch", str(b)], cwd, env=env)
        if rc == 0:
            return rc, out
        if "out of memory" not in out.lower():
            return rc, out
        print(f"  [ball] CUDA OOM at batch {b} — falling back", flush=True)
        last = (rc, out)
    return last


def run_stage(stage, clip_dir, weights, backup_dir, repo_dir) -> str:
    """Run one stage unless its output is already present. Backs up outputs to
    Drive on success. Returns 'skipped' | 'done'."""
    clip_dir = Path(clip_dir)
    if (clip_dir / stage["required"]).exists():
        print(f"[skip] {stage['name']}: {stage['required']} already present", flush=True)
        return "skipped"
    print(f"\n===== {stage['name']} =====", flush=True)
    free_gpu()
    base = [sys.executable, "-u", "-m", stage["module"], str(clip_dir)]
    args = list(stage["args"])
    t0 = time.time()
    if stage.get("gpu_batch"):
        args = ["--weights", str(weights)] + args
        rc, _ = _run_ball_with_fallback(base, args, repo_dir)
    else:
        rc, _ = _run(base + args, repo_dir)
    if rc != 0:
        raise RuntimeError(f"{stage['name']} failed (rc={rc})")
    print(f"  -> {stage['name']} done in {time.time() - t0:.0f}s", flush=True)
    _backup_stage(stage, clip_dir, backup_dir)
    return "done"


def _backup_stage(stage, clip_dir, backup_dir) -> None:
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in stage["outputs"]:
        p = Path(clip_dir) / name
        if p.exists():
            robust_copy(p, backup_dir / name)
    print(f"  backed up {stage['name']} -> {backup_dir}", flush=True)


# ---------------------------------------------------------------- entry point

def run_all(repo_dir, drive_dir="/content/drive/MyDrive", clip=None, content="/content",
            rerun=None):
    """Full reset-proof vision pass. `repo_dir` is the cloned repo (stages run with
    it as cwd). Auto-detects the clip, resumes from the Drive backup, runs only the
    outstanding stages, and backs each up as it finishes.

    `rerun`: stage name(s) to FORCE re-running even though their outputs already
    exist, e.g. rerun="ball" after a detector change. Their outputs are cleared
    locally first (the Drive backup is overwritten when the stage finishes), so the
    other, expensive stages are still skipped."""
    clip = derive_clip(drive_dir, clip)
    print(f"CLIP = {clip}\n", flush=True)

    clip_dir = Path(content) / clip
    backup_dir = Path(drive_dir) / f"{clip}_outputs"
    restored = restore_outputs(backup_dir, clip_dir)
    if restored:
        print("resumed from Drive backup:", ", ".join(restored), flush=True)

    force = {rerun} if isinstance(rerun, str) else set(rerun or ())
    if force:
        unknown = force - {s["name"] for s in STAGES}
        if unknown:
            raise ValueError(f"unknown stage(s) to rerun: {sorted(unknown)}; "
                             f"valid: {[s['name'] for s in STAGES]}")
        for stage in STAGES:
            if stage["name"] in force:
                for out in stage["outputs"]:
                    p = clip_dir / out
                    if p.exists():
                        p.unlink()
                print(f"[rerun] {stage['name']}: cleared {stage['outputs']}", flush=True)

    if have_all_required(clip_dir):
        print("all outputs already computed — nothing to run.", flush=True)
    else:
        clip_dir, weights = copy_inputs(drive_dir, clip, content)
        restore_outputs(backup_dir, clip_dir)   # re-place after unzip (belt+braces)
        for stage in STAGES:
            run_stage(stage, clip_dir, weights, backup_dir, repo_dir)

    missing = [f for f in REQUIRED_OUTPUTS if not (clip_dir / f).exists()]
    print("\n" + ("=" * 52), flush=True)
    if missing:
        print("INCOMPLETE — missing:", ", ".join(missing), flush=True)
    else:
        print("ALL 7 OUTPUTS READY.", flush=True)
        print(f"Download them from EITHER:\n"
              f"  - Files panel:  content/{clip}/\n"
              f"  - or Drive:     {backup_dir}   (survives a runtime reset)\n"
              f"Then click 'Upload vision outputs' on the app run screen.", flush=True)
    return clip_dir

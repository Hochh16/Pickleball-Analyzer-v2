"""FastAPI app for the setup wizard (Phase 1).

Serves the single-page vanilla-JS UI and the JSON/frame endpoints it drives.
All heavy lifting (calibration, file writing) lives in sessions.py; this module
is just HTTP glue + error mapping.

Run:  python -m app     (see app/__main__.py)
      or: uvicorn app.server:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import browse as browse_mod
from . import video as video_mod
from .drivesync import DriveSync, detect_drive_dir
from .pipeline import PipelineRunner, has_vision_outputs
from .collections import SAME_PERSON_REMINDER, CollectionError, CollectionStore
from .sessions import SessionError, SessionStore

# Only these (known pipeline outputs from the vision GPU pass) may be written via
# the vision upload — a guard so the endpoint can't drop arbitrary files.
ALLOWED_VISION_FILES = {
    "players.parquet", "players_pending.json", "track_roles.json",
    "poses.parquet", "pose_summary.json", "ball.parquet", "ball.meta.json",
}

DATA_ROOT = Path(os.environ.get("PB_DATA_DIR", "data")).resolve()
# One designated drop folder for videos — the user copies a clip here and picks
# it, instead of browsing the whole filesystem.
VIDEOS_DIR = Path(os.environ.get("PB_VIDEOS_DIR", "videos")).resolve()
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"

store = SessionStore(DATA_ROOT)
collections = CollectionStore(DATA_ROOT)
drivesync = DriveSync(detect_drive_dir())
runner = PipelineRunner(store, drivesync=drivesync)

app = FastAPI(title="Pickleball Analyzer v2 — Setup Wizard")


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class CreateLocalRequest(BaseModel):
    path: str
    name: Optional[str] = None


class CalibrateRequest(BaseModel):
    court_corners_image: List[List[float]]
    kitchen_line_user_image: List[List[float]]
    kitchen_line_opponent_image: List[List[float]]
    user_baseline: str
    dominant_hand: str
    user_starting_corner: str
    frame_used_for_calibration: int = 0


class StartingCornerRequest(BaseModel):
    corner: str


class RosterRequest(BaseModel):
    user: str = "unknown"
    partner: str = "unknown"
    opp_a: str = "unknown"
    opp_b: str = "unknown"


class ClickModel(BaseModel):
    frame: int
    x: int
    y: int


class UserClicksRequest(BaseModel):
    clicks: List[ClickModel] = []


# --------------------------------------------------------------------------
# Health + file browser
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "data_root": str(DATA_ROOT)}


@app.get("/api/config")
def config() -> dict:
    """Paths the UI should always be able to show (even if listing fails)."""
    return {
        "videos_dir": str(VIDEOS_DIR),
        "data_root": str(DATA_ROOT),
        "drive_sync": drivesync.enabled(),
        "drive_dir": str(drivesync.drive_dir) if drivesync.drive_dir else None,
    }


@app.get("/api/videos")
def videos() -> dict:
    """List videos in the single designated drop folder. Always includes the
    folder path so the UI can show it even when the folder is empty; the folder
    is (re)created defensively so a missing folder never errors."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = browse_mod.listing(VIDEOS_DIR)
        vids = data["videos"]
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        vids = []
    return {"dir": str(VIDEOS_DIR), "videos": vids}


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

@app.get("/api/sessions")
def list_sessions(all: bool = False) -> dict:
    """One entry per VIDEO, newest setup first.

    Setting a video up more than once creates a new folder each time, so the raw list
    grows a row per attempt — 14 rows for 3 actual videos in testing, which makes both
    the "continue a previous setup" list and the collection picker unusable. Folders
    beginning with "_" are internal (collections, scratch) and are never offered.

    ?all=true returns the raw list for debugging.
    """
    def _is_real(s: dict) -> bool:
        if str(s.get("id", "")).startswith("_"):
            return False
        # A session pointing at a file INSIDE the data root is one of the pipeline's own
        # debug clips (_bounces_check2.mp4 and friends) that got opened by accident, not
        # a match someone wants to analyse.
        vp = str(s.get("video_path") or "")
        try:
            inside = Path(vp).resolve().is_relative_to(DATA_ROOT)
        except (OSError, ValueError):
            inside = False
        return not (inside and Path(vp).name != "video.mp4")

    raw = [s for s in store.list() if _is_real(s)]
    if all:
        return {"sessions": raw}
    best: dict = {}
    for s in raw:
        key = str(s.get("video_path") or s.get("id"))
        prev = best.get(key)
        # keep the most recent setup for a video, but report how many exist so nothing
        # is silently hidden
        if prev is None or str(s.get("created_at", "")) > str(prev.get("created_at", "")):
            s = dict(s)
            s["duplicate_setups"] = (prev or {}).get("duplicate_setups", 0) + 1
            best[key] = s
        else:
            prev["duplicate_setups"] = prev.get("duplicate_setups", 0) + 1
    out = sorted(best.values(), key=lambda s: str(s.get("created_at", "")), reverse=True)
    return {"sessions": out}


@app.post("/api/sessions")
def create_session_local(req: CreateLocalRequest) -> dict:
    try:
        return store.create_from_path(Path(req.path), name=req.name)
    except (SessionError, video_mod.VideoError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sessions/upload")
async def create_session_upload(video: UploadFile = File(...)) -> dict:
    try:
        session = store.create_from_upload(video.filename or "upload.mp4", video.file)
    except (SessionError, video_mod.VideoError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        await video.close()
    return session


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    try:
        return store.get(session_id)
    except SessionError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/sessions/{session_id}/frame/{frame_idx}")
def get_frame(session_id: str, frame_idx: int, maxw: int = 1600) -> Response:
    try:
        video_path = store.video_path(session_id)
    except SessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        jpeg = video_mod.frame_server.frame_jpeg(video_path, frame_idx, max_w=maxw)
    except video_mod.VideoError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/sessions/{session_id}/calibrate")
def calibrate_session(session_id: str, req: CalibrateRequest) -> dict:
    try:
        return store.calibrate(session_id, req.model_dump())
    except SessionError as e:
        raise HTTPException(status_code=400, detail=str(e))


class SessionCollectionRequest(BaseModel):
    collection_id: Optional[str] = None      # None = standalone
    player_name: Optional[str] = None


@app.post("/api/sessions/{session_id}/collection")
def set_session_collection(session_id: str, req: SessionCollectionRequest) -> dict:
    """Choose the cumulative report for a video BEFORE it is analysed."""
    try:
        return store.set_collection(session_id, req.collection_id, req.player_name)
    except SessionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sessions/{session_id}/finalize-collection")
def finalize_session_collection(session_id: str) -> dict:
    """Fold a finished video into the cumulative report chosen when it was set up.

    Idempotent: re-running after a rebuild, or on a video already in the collection,
    reports the existing membership instead of failing. The UI calls this when a run
    completes, so the operator never has to remember a second step.
    """
    try:
        session = store.get(session_id)
    except SessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    cid = session.get("collection_id")
    if not cid:
        return {"collection_id": None, "added": False, "reason": "standalone"}
    try:
        doc = collections.get_doc(cid)
    except CollectionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if any(m["session_id"] == session_id for m in doc.get("members", [])):
        return {"collection_id": cid, "added": False, "reason": "already a member",
                "collection": doc}
    try:
        doc = collections.add(cid, store.folder(session_id),
                              venue_ok=_venue_ok(session_id))
    except CollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"collection_id": cid, "added": True, "collection": doc}


@app.post("/api/sessions/{session_id}/starting-corner")
def starting_corner_session(session_id: str, req: StartingCornerRequest) -> dict:
    try:
        return store.set_starting_corner(session_id, req.corner)
    except SessionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sessions/{session_id}/roster")
def roster_session(session_id: str, req: RosterRequest) -> dict:
    try:
        return store.write_roster(session_id, req.model_dump())
    except SessionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sessions/{session_id}/user-clicks")
def user_clicks_session(session_id: str, req: UserClicksRequest) -> dict:
    try:
        return store.write_user_clicks(session_id, [c.model_dump() for c in req.clicks])
    except SessionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sessions/{session_id}/summary")
def summary_session(session_id: str) -> dict:
    try:
        return store.summary(session_id)
    except SessionError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --------------------------------------------------------------------------
# Pipeline run (Phase 2)
# --------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/run")
def start_run(session_id: str) -> dict:
    try:
        store.get(session_id)  # 404 if unknown
        job = runner.start(session_id)
    except SessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return job.snapshot()


@app.get("/api/sessions/{session_id}/run")
def get_run(session_id: str) -> dict:
    job = runner.get(session_id)
    if job is None:
        return {"phase": "idle", "steps": [], "log": [], "version": 0}
    return job.snapshot()


@app.post("/api/sessions/{session_id}/run/cancel")
def cancel_run(session_id: str) -> dict:
    runner.cancel(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/run/stream")
async def stream_run(session_id: str) -> StreamingResponse:
    async def gen():
        last = -1
        idle_ticks = 0
        while True:
            job = runner.get(session_id)
            snap = job.snapshot() if job else {"phase": "idle", "steps": [], "log": [], "version": 0}
            if snap["version"] != last:
                last = snap["version"]
                idle_ticks = 0
                yield f"data: {json.dumps(snap)}\n\n"
            else:
                idle_ticks += 1
                if idle_ticks % 30 == 0:      # keepalive ~ every 15s
                    yield ": keepalive\n\n"
            if snap["phase"] in ("done", "failed"):
                yield f"data: {json.dumps(snap)}\n\n"
                return
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/sessions/{session_id}/vision")
async def upload_vision(session_id: str, files: List[UploadFile] = File(...)) -> dict:
    """Receive the combined vision-pass outputs (players.parquet, track_roles.json,
    poses.parquet, ball.parquet, ball.meta.json, + optional sidecars) produced on
    Colab, and AUTO-RESUME Stages 5+ once the required set is present."""
    try:
        folder = store.folder(session_id)
    except SessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Unknown session")
    saved, skipped = [], []
    try:
        for f in files:
            name = Path(f.filename or "").name
            # A .zip (e.g. what Google Drive's web download hands you) — extract the
            # allowed members so the operator never has to unzip by hand.
            if name.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(f.file) as z:
                        for member in z.namelist():
                            base = Path(member).name
                            if member.endswith("/") or base not in ALLOWED_VISION_FILES:
                                continue
                            with z.open(member) as src, (folder / base).open("wb") as out:
                                shutil.copyfileobj(src, out)
                            saved.append(base)
                except zipfile.BadZipFile:
                    skipped.append(name)
                continue
            if name not in ALLOWED_VISION_FILES:
                skipped.append(name)
                continue
            with (folder / name).open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(name)
    finally:
        for f in files:
            await f.close()
    store.ensure_ball_meta(session_id)  # synthesize the sidecar if not uploaded
    job = runner.resume_post(session_id)
    return {
        "ok": True,
        "saved": saved,
        "skipped": skipped,
        "have_all_outputs": has_vision_outputs(folder),
        "resumed": job is not None,
    }


@app.get("/api/sessions/{session_id}/files/{file_path:path}")
def get_session_file(session_id: str, file_path: str) -> FileResponse:
    """Serve a file from the session folder (report.html + its sibling assets like
    annotated_web.mp4, so the report's relative video src resolves in-app). Guarded
    to the session folder."""
    folder = store.folder(session_id).resolve()
    target = (folder / file_path).resolve()
    if folder not in target.parents and target != folder:
        raise HTTPException(status_code=403, detail="Path outside session")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(target)


@app.get("/api/sessions/{session_id}/vision-input.zip")
def get_vision_input_bundle(session_id: str) -> FileResponse:
    """One per-clip bundle (video + court/roster) to upload to Drive for the GPU
    vision pass — so the hand-off is a single download/upload."""
    try:
        store.get(session_id)
    except SessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    dest = store.folder(session_id) / f"{session_id}_vision_input.zip"
    try:
        store.build_vision_input_zip(session_id, dest)
    except SessionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return FileResponse(dest, media_type="application/zip",
                        filename=f"{session_id}_vision_input.zip")


@app.get("/api/sessions/{session_id}/report")
def get_report(session_id: str) -> FileResponse:
    path = store.folder(session_id) / "report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not built yet")
    return FileResponse(path, media_type="text/html")


@app.get("/api/sessions/{session_id}/annotated")
def get_annotated(session_id: str) -> FileResponse:
    folder = store.folder(session_id)
    for name in ("annotated_web.mp4", "annotated.mp4"):
        p = folder / name
        if p.exists():
            return FileResponse(p, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Annotated video not ready")


# --------------------------------------------------------------------------
# Collections — cumulative analysis across videos
# --------------------------------------------------------------------------

class CreateCollectionRequest(BaseModel):
    name: str


class AddMemberRequest(BaseModel):
    session_id: str


def _venue_ok(session_id: str) -> Optional[bool]:
    """The stored venue verdict for a session, if one has been measured.

    None means "not measured", which is NOT the same as "supported" — the caller must not
    turn an absent verdict into a refusal, or every video would be blocked before the
    check has ever been run.
    """
    try:
        v = store.get(session_id).get("venue", {}).get("verdict")
    except SessionError:
        return None
    return None if v is None else v != "not_supported"


@app.get("/api/models/ball_model_v4.pt")
def get_ball_model() -> FileResponse:
    """Hand the operator the CURRENTLY DEPLOYED ball model to put on Drive.

    The Colab vision pass uses whatever ball_model_v4.pt sits in My Drive. When the app's
    model is updated and Drive's is not, clips are silently analysed by an older detector
    — no error, just worse numbers, and a collection that mixes two accuracy regimes.
    Serving the exact deployed file removes the guesswork about which one is current.
    """
    p = (Path("data/models") / "ball_model_v4.pt").resolve()
    if not p.is_file():
        raise HTTPException(status_code=404, detail="No ball model is deployed")
    return FileResponse(p, media_type="application/octet-stream",
                        filename="ball_model_v4.pt")


@app.get("/api/collections")
def list_collections() -> dict:
    active = collections.active()
    return {"collections": collections.list(),
            "active_id": active["id"] if active else None,
            # Nothing in the data can verify a collection holds one person, so the
            # reminder travels with the list rather than living only in the docs.
            "reminder": SAME_PERSON_REMINDER}


@app.post("/api/collections")
def create_collection(req: CreateCollectionRequest) -> dict:
    return collections.create(req.name)


@app.get("/api/collections/{cid}")
def get_collection(cid: str) -> dict:
    try:
        return collections.get_doc(cid)
    except CollectionError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/collections/{cid}/activate")
def activate_collection(cid: str) -> dict:
    try:
        return collections.set_active(cid)
    except CollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/collections/{cid}/close")
def close_collection(cid: str) -> dict:
    try:
        return collections.close(cid)
    except CollectionError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/collections/{cid}/reopen")
def reopen_collection(cid: str) -> dict:
    try:
        return collections.reopen(cid)
    except CollectionError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/collections/{cid}/members")
def add_collection_member(cid: str, req: AddMemberRequest) -> dict:
    try:
        return collections.add(cid, store.folder(req.session_id),
                               venue_ok=_venue_ok(req.session_id))
    except CollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/collections/{cid}/members/{session_id}")
def remove_collection_member(cid: str, session_id: str) -> dict:
    try:
        return collections.remove(cid, session_id)
    except CollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/collections/{cid}/rebuild")
def rebuild_collection(cid: str) -> dict:
    try:
        return collections.rebuild(cid)
    except CollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/collections/{cid}/files/{file_path:path}")
def get_collection_file(cid: str, file_path: str) -> FileResponse:
    """Serve the cumulative report and its sibling assets, guarded to the folder."""
    folder = collections.folder(cid).resolve()
    target = (folder / file_path).resolve()
    if folder not in target.parents and target != folder:
        raise HTTPException(status_code=403, detail="Path outside collection")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(target)


# --------------------------------------------------------------------------
# Static SPA (mounted last so /api/* wins)
# --------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

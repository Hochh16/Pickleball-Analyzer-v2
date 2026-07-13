"""FastAPI app for the setup wizard (Phase 1).

Serves the single-page vanilla-JS UI and the JSON/frame endpoints it drives.
All heavy lifting (calibration, file writing) lives in sessions.py; this module
is just HTTP glue + error mapping.

Run:  python -m app     (see app/__main__.py)
      or: uvicorn app.server:app --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import browse as browse_mod
from . import video as video_mod
from .sessions import SessionError, SessionStore

DATA_ROOT = Path(os.environ.get("PB_DATA_DIR", "data")).resolve()
# One designated drop folder for videos — the user copies a clip here and picks
# it, instead of browsing the whole filesystem.
VIDEOS_DIR = Path(os.environ.get("PB_VIDEOS_DIR", "videos")).resolve()
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"

store = SessionStore(DATA_ROOT)

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


@app.get("/api/videos")
def videos() -> dict:
    """List videos in the single designated drop folder."""
    try:
        data = browse_mod.listing(VIDEOS_DIR)
    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"dir": str(VIDEOS_DIR), "videos": data["videos"]}


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

@app.get("/api/sessions")
def list_sessions() -> dict:
    return {"sessions": store.list()}


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
# Static SPA (mounted last so /api/* wins)
# --------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

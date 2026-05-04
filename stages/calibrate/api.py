"""Stage 1 - FastAPI wrapper around calibrate.py.

Single endpoint POST /calibrate accepts a video upload + markers JSON, runs
the calibration, writes court.json and court_zones.json to a per-video data
folder, and returns the calibration result + a top-down preview as a base64
JPEG.

Run locally:
    uvicorn stages.calibrate.api:app --port 8765
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .calibrate import (
    MarkersError,
    VideoError,
    calibrate,
    load_markers,
    render_top_down_preview,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

app = FastAPI(title="Pickleball Analyzer v2 - Stage 1: Calibrate")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class CalibrateResponse(BaseModel):
    job_id:    str
    court:     dict
    zones:     dict
    preview_jpeg_base64: str
    warnings:  list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "stage": 1, "data_dir": str(DATA_DIR)}


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate_endpoint(
    video:   UploadFile = File(...),
    markers: str        = Form(...),
) -> CalibrateResponse:
    """Run Stage 1 calibration on the uploaded video."""
    job_id = str(uuid.uuid4())
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / "video.mp4"
    try:
        with video_path.open("wb") as out:
            shutil.copyfileobj(video.file, out)
    finally:
        await video.close()

    markers_path = job_dir / "markers.json"
    try:
        markers_dict = json.loads(markers)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid markers JSON: {e}")
    with markers_path.open("w", encoding="utf-8") as f:
        json.dump(markers_dict, f, indent=2)

    try:
        loaded = load_markers(markers_path)
        court_json, zones_json = calibrate(video_path, loaded)
    except MarkersError as e:
        raise HTTPException(status_code=422, detail=f"Markers invalid: {e}")
    except VideoError as e:
        raise HTTPException(status_code=400, detail=f"Video error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Calibration failed: {e}")

    with (job_dir / "court.json").open("w", encoding="utf-8") as f:
        json.dump(court_json, f, indent=2)
    with (job_dir / "court_zones.json").open("w", encoding="utf-8") as f:
        json.dump(zones_json, f, indent=2)

    image_to_court = np.asarray(court_json["homography"]["image_to_court"], dtype=np.float64)
    preview_bgr = render_top_down_preview(
        video_path,
        court_json["video"]["frame_used_for_calibration"],
        image_to_court,
    )
    ok, jpeg = cv2.imencode(".jpg", preview_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode preview JPEG")
    preview_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

    return CalibrateResponse(
        job_id=job_id,
        court=court_json,
        zones=zones_json,
        preview_jpeg_base64=preview_b64,
        warnings=court_json["validation"]["warnings"],
    )
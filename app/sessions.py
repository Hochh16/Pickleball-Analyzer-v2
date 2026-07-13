"""Session + per-video folder management for the setup wizard.

A "session" is one analysis = one per-video folder under the data root (the
pipeline's existing convention, `data/<name>/`). This module owns creating those
folders, resolving the video handle, and writing the exact input JSONs the
pipeline consumes:

  - markers.json      (8 court points + 3 form answers)  -> Stage 1 calibrate
  - court.json        (written by Stage 1)
  - court_zones.json  (written by Stage 1)
  - roster.json       (handedness per role)
  - user_clicks.json  (optional self-identification)

It calls `stages.calibrate.calibrate()` in-process — no reimplementation of the
calibration math. Data contracts are unchanged from the Tkinter tools.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from stages.calibrate.calibrate import (
    MarkersError,
    calibrate,
    load_markers,
    render_top_down_preview,
)

from . import video as video_mod

SESSION_FILE = "session.json"
ROSTER_SCHEMA_VERSION = 1
ROLE_KEYS = ("user", "partner", "opp_a", "opp_b")
HAND_VALUES = ("left", "right", "unknown")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Generic filenames that don't make a good session name — fall back to the
# parent folder (this pipeline names most clips data/<clip>/video.mp4).
_GENERIC_STEMS = {"video", "clip", "output", "movie", "render", "input", "match"}


def _default_name(video_path: Path) -> str:
    stem = video_path.stem
    if stem.lower() in _GENERIC_STEMS:
        parent = video_path.parent.name
        if parent:
            return parent
    return stem


def slugify(name: str) -> str:
    """Filesystem-safe folder slug from an arbitrary name."""
    stem = Path(name).stem
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_").lower()
    return slug or "session"


class SessionError(RuntimeError):
    """User-facing session problem (bad input, missing file, etc.)."""


class SessionStore:
    """Owns the data root and the per-video session folders."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ----- paths -----

    def folder(self, session_id: str) -> Path:
        return self.data_root / session_id

    def _session_path(self, session_id: str) -> Path:
        return self.folder(session_id) / SESSION_FILE

    def _unique_id(self, base: str) -> str:
        """A folder slug not already taken (append -2, -3, ... if needed)."""
        with self._lock:
            candidate = base
            i = 2
            while (self.data_root / candidate).exists():
                candidate = f"{base}-{i}"
                i += 1
            (self.data_root / candidate).mkdir(parents=True, exist_ok=False)
            return candidate

    # ----- creation -----

    def create_from_path(self, video_path: Path, name: Optional[str] = None) -> Dict:
        """Register an existing local video in place (no copy)."""
        video_path = Path(video_path).expanduser()
        if not video_path.exists() or not video_path.is_file():
            raise SessionError(f"Video file not found: {video_path}")
        meta = video_mod.probe(video_path)  # validates it's a readable video
        display_name = name or _default_name(video_path)
        session_id = self._unique_id(slugify(display_name))
        return self._write_new_session(
            session_id, video_path.resolve(), meta, source="local",
            display_name=display_name,
        )

    def create_from_upload(self, filename: str, fileobj) -> Dict:
        """Save an uploaded video into a new folder as video.mp4."""
        session_id = self._unique_id(slugify(filename))
        dest = self.folder(session_id) / "video.mp4"
        with dest.open("wb") as out:
            shutil.copyfileobj(fileobj, out)
        try:
            meta = video_mod.probe(dest)
        except video_mod.VideoError as e:
            # Clean up the half-baked folder so it doesn't litter the library.
            shutil.rmtree(self.folder(session_id), ignore_errors=True)
            raise SessionError(f"Uploaded file is not a readable video: {e}")
        return self._write_new_session(
            session_id, dest.resolve(), meta, source="upload",
            display_name=Path(filename).stem,
        )

    def _write_new_session(
        self, session_id: str, video_path: Path, meta: Dict,
        source: str, display_name: str,
    ) -> Dict:
        session = {
            "schema_version": 1,
            "id": session_id,
            "name": display_name,
            "source": source,
            "video_path": str(video_path),
            "video": meta,
            "created_at": _now_iso(),
            "steps": {"calibration": False, "roster": False, "user_clicks": False},
        }
        self._save_session(session)
        return session

    # ----- load / list -----

    def _save_session(self, session: Dict) -> None:
        path = self._session_path(session["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        tmp.replace(path)

    def get(self, session_id: str) -> Dict:
        path = self._session_path(session_id)
        if not path.exists():
            raise SessionError(f"Unknown session: {session_id}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def video_path(self, session_id: str) -> Path:
        return Path(self.get(session_id)["video_path"])

    def list(self) -> List[Dict]:
        out: List[Dict] = []
        if not self.data_root.exists():
            return out
        for child in sorted(self.data_root.iterdir()):
            sp = child / SESSION_FILE
            if sp.exists():
                try:
                    with sp.open("r", encoding="utf-8") as f:
                        out.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    continue
        out.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return out

    def _mark_step(self, session_id: str, step: str, done: bool = True) -> Dict:
        session = self.get(session_id)
        session.setdefault("steps", {})[step] = done
        self._save_session(session)
        return session

    # ----- Stage 1: markers + calibrate -----

    def calibrate(self, session_id: str, markers_payload: Dict) -> Dict:
        """Write markers.json, run Stage 1 in-process, write court.json /
        court_zones.json, and return a summary + top-down preview (base64 JPEG)."""
        folder = self.folder(session_id)
        if not folder.exists():
            raise SessionError(f"Unknown session: {session_id}")
        video_path = self.video_path(session_id)

        markers = _build_markers(markers_payload)
        markers_path = folder / "markers.json"
        with markers_path.open("w", encoding="utf-8") as f:
            json.dump(markers, f, indent=2)

        try:
            loaded = load_markers(markers_path)
            court_json, zones_json = calibrate(video_path, loaded)
        except MarkersError as e:
            raise SessionError(f"Court points are invalid: {e}")
        except ValueError as e:
            raise SessionError(f"Calibration failed: {e}")

        with (folder / "court.json").open("w", encoding="utf-8") as f:
            json.dump(court_json, f, indent=2)
        with (folder / "court_zones.json").open("w", encoding="utf-8") as f:
            json.dump(zones_json, f, indent=2)

        image_to_court = np.asarray(
            court_json["homography"]["image_to_court"], dtype=np.float64
        )
        preview_bgr = render_top_down_preview(
            video_path,
            court_json["video"]["frame_used_for_calibration"],
            image_to_court,
        )
        ok, jpeg = cv2.imencode(".jpg", preview_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise SessionError("Failed to render the top-down preview")
        preview_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

        self._mark_step(session_id, "calibration", True)
        return {
            "validation": court_json["validation"],
            "preview_jpeg_base64": preview_b64,
            "frame_used_for_calibration": court_json["video"]["frame_used_for_calibration"],
        }

    def set_starting_corner(self, session_id: str, corner: str) -> Dict:
        """Patch the user's starting corner into markers.json + court.json.

        `user_starting_corner` isn't used in the homography (only `user_baseline`
        is), so it can be set visually AFTER calibration without recomputing —
        we just keep markers.json and court.json.user_inputs consistent for the
        downstream stages (Stage 2/2.5 read it from court.json)."""
        if corner not in ("left", "right"):
            raise SessionError(f"starting corner must be 'left' or 'right'; got {corner!r}")
        folder = self.folder(session_id)
        markers_path = folder / "markers.json"
        court_path = folder / "court.json"
        if not court_path.exists():
            raise SessionError("Mark the court first")
        for path, patch in ((markers_path, "flat"), (court_path, "nested")):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if patch == "flat":
                data["user_starting_corner"] = corner
            else:
                data.setdefault("user_inputs", {})["user_starting_corner"] = corner
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(path)
        return {"user_starting_corner": corner}

    # ----- roster -----

    def write_roster(self, session_id: str, handedness: Dict[str, str]) -> Dict:
        folder = self.folder(session_id)
        if not folder.exists():
            raise SessionError(f"Unknown session: {session_id}")
        clean: Dict[str, str] = {}
        for key in ROLE_KEYS:
            val = (handedness or {}).get(key, "unknown")
            if val not in HAND_VALUES:
                raise SessionError(f"handedness.{key} must be one of {HAND_VALUES}; got {val!r}")
            clean[key] = val
        roster = {"schema_version": ROSTER_SCHEMA_VERSION, "handedness": clean}
        with (folder / "roster.json").open("w", encoding="utf-8") as f:
            json.dump(roster, f, indent=2)
        self._mark_step(session_id, "roster", True)
        return roster

    # ----- user clicks (optional self-identification) -----

    def write_user_clicks(self, session_id: str, clicks: List[Dict]) -> Dict:
        folder = self.folder(session_id)
        if not folder.exists():
            raise SessionError(f"Unknown session: {session_id}")
        clean: List[Dict] = []
        for c in clicks or []:
            try:
                clean.append({"frame": int(c["frame"]), "x": int(c["x"]), "y": int(c["y"])})
            except (KeyError, TypeError, ValueError):
                raise SessionError(f"Malformed click entry: {c!r}")
        clean.sort(key=lambda c: c["frame"])
        payload = {"clicks": clean}
        path = folder / "user_clicks.json"
        if clean:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._mark_step(session_id, "user_clicks", True)
        else:
            # Empty = skipped; remove any prior file so the geometric seed is used.
            if path.exists():
                path.unlink()
            self._mark_step(session_id, "user_clicks", False)
        return payload

    def summary(self, session_id: str) -> Dict:
        """Everything the Review step needs, read back from disk."""
        session = self.get(session_id)
        folder = self.folder(session_id)

        def _read(name: str) -> Optional[Dict]:
            p = folder / name
            if not p.exists():
                return None
            try:
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None

        court = _read("court.json")
        roster = _read("roster.json")
        clicks = _read("user_clicks.json")
        return {
            "session": session,
            "calibration": (
                {
                    "validation": court["validation"],
                    "user_inputs": court["user_inputs"],
                    "frame_used_for_calibration": court["video"]["frame_used_for_calibration"],
                }
                if court else None
            ),
            "roster": roster,
            "user_clicks_count": len(clicks["clicks"]) if clicks else 0,
        }


def _build_markers(payload: Dict) -> Dict:
    """Validate + normalize the frontend payload into markers.json schema.

    All coordinates are in ORIGINAL-video pixel space (the frontend converts
    display clicks back to source pixels before sending).
    """
    def _pts(key: str, n: int) -> List[List[int]]:
        raw = payload.get(key)
        if not isinstance(raw, list) or len(raw) != n:
            raise SessionError(f"{key} must be {n} [x, y] points")
        out = []
        for p in raw:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise SessionError(f"{key} entries must be [x, y]")
            out.append([int(round(p[0])), int(round(p[1]))])
        return out

    def _enum(key: str, allowed) -> str:
        val = payload.get(key)
        if val not in allowed:
            raise SessionError(f"{key} must be one of {allowed}; got {val!r}")
        return val

    return {
        "court_corners_image": _pts("court_corners_image", 4),
        "kitchen_line_user_image": _pts("kitchen_line_user_image", 2),
        "kitchen_line_opponent_image": _pts("kitchen_line_opponent_image", 2),
        "user_baseline": _enum("user_baseline", ("near", "far")),
        "dominant_hand": _enum("dominant_hand", ("right", "left")),
        "user_starting_corner": _enum("user_starting_corner", ("left", "right")),
        "frame_used_for_calibration": int(payload.get("frame_used_for_calibration", 0)),
    }

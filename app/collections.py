"""Collections — the cumulative-analysis layer over per-video sessions.

Contract: stages/aggregate/contract.md (decisions D1-D5). The operator's requirements:

  - a video either JOINS the running cumulative analysis or stands alone;
  - a fresh cumulative analysis can be started at any point, leaving the old one intact;
  - a collection is one PERSON (D5) — several people can be analysed, so collections are
    selectable and the UI must remind the operator not to mix people;
  - a venue the detector cannot measure is EXCLUDED rather than merged (D4).

This module owns membership and rebuilds. It computes no statistics: a rebuild unions the
members (Stage 7.9) and then runs Stages 8-11 over the union, unchanged.

Design notes that are easy to get wrong later:

  STANDALONE IS THE DEFAULT. A video is never silently absorbed into a collection — the
  operator has to say so. Analysing one video must never quietly change another report.

  ADDING REBUILDS EVERYTHING. Aggregation is arithmetic over a few thousand rows, so a
  full rebuild costs a second or two and keeps the "as if one video" invariant true by
  construction. Incremental accumulation would drift the first time a member was removed
  or re-run, and drift silently.

  CLOSING IS NOT DELETING. A closed collection stays readable and rebuildable forever.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

COLLECTIONS_DIRNAME = "_collections"
COLLECTION_FILE = "collection.json"
INDEX_FILE = "_collections/index.json"
SCHEMA_VERSION = 1

# Stages that run over the union. Aggregation is Stage 7.9; everything after it is the
# ordinary per-video pipeline, which is the entire point of unioning below Stage 8.
POST_STAGES = ["stages.compute_metrics.compute_metrics",
               "stages.rate.rate",
               "stages.plan_improvement.plan_improvement"]

# Shown wherever collections are listed. Nothing in the data can verify that a collection
# holds one person — it is an operator promise (D5), so it has to be visible at the moment
# of choosing, not buried in documentation.
SAME_PERSON_REMINDER = ("Each collection must be for the SAME person. "
                        "Start a new collection for a different player.")


class CollectionError(RuntimeError):
    """User-facing problem (unknown collection, duplicate video, unsupported venue)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_").lower()
    return s or "collection"


def file_sha256(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    """Hash of the first `limit` bytes. A full hash of a 5 GB video costs minutes for no
    extra safety here — the job is catching the same file added twice, not resisting an
    adversary."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()


class CollectionStore:
    def __init__(self, data_root: Path):
        self.root = Path(data_root)
        self.dir = self.root / COLLECTIONS_DIRNAME
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---------- paths / io ----------

    def folder(self, cid: str) -> Path:
        return self.dir / cid

    def _index_path(self) -> Path:
        return self.root / INDEX_FILE

    def _read_index(self) -> Dict:
        p = self._index_path()
        if not p.exists():
            return {"schema_version": SCHEMA_VERSION, "active": None}
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_index(self, idx: Dict) -> None:
        p = self._index_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        tmp.replace(p)

    def _read(self, cid: str) -> Dict:
        p = self.folder(cid) / COLLECTION_FILE
        if not p.exists():
            raise CollectionError(f"Unknown collection: {cid}")
        return json.loads(p.read_text(encoding="utf-8"))

    def _write(self, doc: Dict) -> None:
        f = self.folder(doc["id"])
        f.mkdir(parents=True, exist_ok=True)
        tmp = f / (COLLECTION_FILE + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(f / COLLECTION_FILE)

    # ---------- lifecycle ----------

    def create(self, name: str, make_active: bool = True) -> Dict:
        """Start a new cumulative analysis. Any currently active collection is CLOSED,
        not deleted — that is what "start a new one from this point forward" means."""
        with self._lock:
            base = _slug(name)
            cid, i = base, 2
            while (self.dir / cid).exists():
                cid, i = f"{base}-{i}", i + 1
            idx = self._read_index()
            if make_active and idx.get("active"):
                try:
                    self.close(idx["active"], _locked=True)
                except CollectionError:
                    pass
            doc = {"schema_version": SCHEMA_VERSION, "id": cid, "name": name,
                   "created_at": _now(), "closed_at": None, "members": [],
                   "built_at": None, "warnings": []}
            self._write(doc)
            if make_active:
                idx["active"] = cid
                self._write_index(idx)
            return doc

    def close(self, cid: str, _locked: bool = False) -> Dict:
        doc = self._read(cid)
        doc["closed_at"] = doc["closed_at"] or _now()
        self._write(doc)
        idx = self._read_index()
        if idx.get("active") == cid:
            idx["active"] = None
            self._write_index(idx)
        return doc

    def active(self) -> Optional[Dict]:
        cid = self._read_index().get("active")
        if not cid:
            return None
        try:
            return self._read(cid)
        except CollectionError:
            return None

    def set_active(self, cid: str) -> Dict:
        doc = self._read(cid)
        if doc.get("closed_at"):
            raise CollectionError(f"{cid} is closed; start a new collection instead")
        idx = self._read_index()
        idx["active"] = cid
        self._write_index(idx)
        return doc

    def list(self) -> List[Dict]:
        out = []
        active = self._read_index().get("active")
        for child in sorted(self.dir.iterdir()) if self.dir.exists() else []:
            p = child / COLLECTION_FILE
            if not p.exists():
                continue
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            doc["is_active"] = doc["id"] == active
            out.append(doc)
        out.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return out

    # ---------- membership ----------

    def add(self, cid: str, session_folder: Path, captured_at: Optional[str] = None,
            venue_ok: Optional[bool] = None, rebuild: bool = True) -> Dict:
        """Add one analysed session. Refuses duplicates and unsupported venues."""
        doc = self._read(cid)
        if doc.get("closed_at"):
            raise CollectionError(f"{cid} is closed; reopen or start a new collection")
        session_folder = Path(session_folder)
        if not (session_folder / "classified.json").exists():
            raise CollectionError(f"{session_folder.name} has not been analysed yet")

        # D4: a venue the detector cannot measure is excluded, not merged. A wrong number
        # is worse than a missing one, so this refuses rather than warning.
        if venue_ok is False:
            raise CollectionError(
                f"{session_folder.name}: this venue is not supported yet, so it is not "
                f"added to the collection. The ball is not reliably visible to the "
                f"detector here — the video itself is fine.")

        video = session_folder / "video.mp4"
        sha = file_sha256(video) if video.exists() else None
        for m in doc["members"]:
            if m["session_id"] == session_folder.name:
                raise CollectionError(f"{session_folder.name} is already in {cid}")
            if sha and m.get("video_sha256") == sha:
                raise CollectionError(
                    f"{session_folder.name} is the same video as {m['session_id']}, "
                    f"already in {cid} — adding it twice would double-count everything")

        doc["members"].append({
            "session_id": session_folder.name,
            "path": str(session_folder),
            "captured_at": captured_at or self._captured_at(session_folder),
            "video_sha256": sha,
            "added_at": _now(),
            "pipeline": self._fingerprint(session_folder)})
        # Chronological, so ids and time offsets are deterministic across rebuilds.
        doc["members"].sort(key=lambda m: (m.get("captured_at") or "", m["session_id"]))
        self._write(doc)
        return self.rebuild(cid) if rebuild else doc

    def remove(self, cid: str, session_id: str, rebuild: bool = True) -> Dict:
        doc = self._read(cid)
        n = len(doc["members"])
        doc["members"] = [m for m in doc["members"] if m["session_id"] != session_id]
        if len(doc["members"]) == n:
            raise CollectionError(f"{session_id} is not in {cid}")
        self._write(doc)
        return self.rebuild(cid) if rebuild else doc

    # ---------- rebuild ----------

    def rebuild(self, cid: str) -> Dict:
        """Union the members and run Stages 8-10 over the result. Full rebuild every
        time — see the module docstring."""
        doc = self._read(cid)
        out = self.folder(cid)
        members = [Path(m["path"]) for m in doc["members"]]
        missing = [m for m in members if not m.exists()]
        if missing:
            raise CollectionError(f"member folders missing: {[str(m) for m in missing]}")

        warnings: List[str] = []
        if not members:
            doc["warnings"] = ["collection is empty"]
            doc["built_at"] = _now()
            self._write(doc)
            return doc

        for stale in ("metrics.json", "rating.json", "improvement_plan.json"):
            (out / stale).unlink(missing_ok=True)

        cmd = [sys.executable, "-m", "stages.aggregate.aggregate", "--out", str(out),
               "--log-level", "ERROR"]
        for m in members:
            cmd += ["--member", str(m)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise CollectionError(f"aggregate failed: {r.stderr.strip()[-400:]}")

        for mod in POST_STAGES:
            r = subprocess.run([sys.executable, "-m", mod, str(out), "--force",
                                "--log-level", "ERROR"], capture_output=True, text=True)
            if r.returncode != 0:
                warnings.append(f"{mod.split('.')[-1]} failed: {r.stderr.strip()[-200:]}")

        # D3: mixed pipeline versions are a report FOOTNOTE, never a warning and never a
        # block — too technical to put in a player's face. The detail stays here, where
        # the operator can see which members are stale.
        fps = {json.dumps(m.get("pipeline"), sort_keys=True) for m in doc["members"]}
        if len(fps) > 1:
            doc["footnotes"] = [
                "Videos in this analysis were processed with different versions of the "
                "ball detector, so accuracy is not identical across them."]
        else:
            doc.pop("footnotes", None)

        # The union writes its own collection.json describing the build; keep the
        # membership record as the authority and fold the build summary into it.
        built = out / COLLECTION_FILE
        if built.exists():
            b = json.loads(built.read_text(encoding="utf-8"))
            doc["build"] = {k: b.get(k) for k in
                            ("total_span_sec", "ball_source", "synthetic_gated",
                             "pooled_opponents", "members")}
        doc["warnings"] = warnings
        doc["built_at"] = _now()
        self._write(doc)
        return doc

    # ---------- helpers ----------

    @staticmethod
    def _captured_at(folder: Path) -> Optional[str]:
        """Capture time, best effort: the session record, else the video's mtime. mtime is
        unreliable (copying a file resets it) — order affects ids and time offsets, never
        any statistic, but it must be STABLE or two rebuilds would renumber."""
        s = folder / "session.json"
        if s.exists():
            try:
                d = json.loads(s.read_text(encoding="utf-8"))
                if d.get("captured_at"):
                    return d["captured_at"]
                if d.get("created_at"):
                    return d["created_at"]
            except (json.JSONDecodeError, OSError):
                pass
        v = folder / "video.mp4"
        if v.exists():
            return datetime.fromtimestamp(v.stat().st_mtime,
                                          tz=timezone.utc).isoformat(timespec="seconds")
        return None

    @staticmethod
    def _fingerprint(folder: Path) -> Dict:
        """What produced this member's ball, so mixed-version members can be footnoted."""
        out: Dict = {}
        meta = folder / "ball.meta.json"
        if meta.exists():
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
                det = d.get("detector", {})
                out["weights"] = Path(str(det.get("weights", ""))).name or None
                out["ball_source"] = "synthetic" if d.get("synthetic") else "real"
            except (json.JSONDecodeError, OSError):
                pass
        cl = folder / "classified.json"
        if cl.exists():
            try:
                out["classify_version"] = json.loads(
                    cl.read_text(encoding="utf-8")).get("stage_version")
            except (json.JSONDecodeError, OSError):
                pass
        return out

    def delete(self, cid: str) -> None:
        """Remove a collection's derived data. Members are untouched — a collection is
        derived data and can always be rebuilt from the videos it lists."""
        shutil.rmtree(self.folder(cid), ignore_errors=True)
        idx = self._read_index()
        if idx.get("active") == cid:
            idx["active"] = None
            self._write_index(idx)

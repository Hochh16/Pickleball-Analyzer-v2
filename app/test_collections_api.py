"""Route-level test for the collection endpoints.

Calls the route functions directly rather than over HTTP. That is the convention the
existing app/test_app.py uses, and it avoids adding httpx purely for tests — while still
catching what actually breaks in wiring: a route referencing a store method that does not
exist, an unmapped exception reaching the client as a 500, or the file route escaping its
folder.

Runs against a temporary data root, so it never touches the real library.

Usage:
    python -m app.test_collections_api
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

REAL_DATA = Path("data").resolve()
SRC = [REAL_DATA / "pb_2min", REAL_DATA / "pb_outdoor2_excerpt"]


def _fail(m): print(f"  FAIL: {m}"); return False
def _pass(m): print(f"  PASS: {m}"); return True


def main() -> int:
    print("Collection route test")
    print()
    missing = [s for s in SRC if not (s / "classified.json").exists()]
    if missing:
        print(f"missing analysed clips: {[m.name for m in missing]}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="pb_routes_"))
    # server.py resolves its data root at import time, so this must be set first.
    os.environ["PB_DATA_DIR"] = str(tmp)
    for s in SRC:
        shutil.copytree(s, tmp / s.name,
                        ignore=shutil.ignore_patterns("*.mp4", "frames_720", "_*"))

    from fastapi import HTTPException

    from app import server as srv

    results = []
    try:
        body = srv.list_collections()
        results.append(_pass("list: empty, nothing active, reminder carried")
                       if body["collections"] == [] and body["active_id"] is None
                       and "same" in body["reminder"].lower()
                       else _fail(f"list wrong: {body}"))

        cid = srv.create_collection(srv.CreateCollectionRequest(name="David"))["id"]
        results.append(_pass(f"created {cid}") if cid else _fail("create failed"))

        srv.add_collection_member(cid, srv.AddMemberRequest(session_id=SRC[0].name))
        doc = srv.add_collection_member(cid, srv.AddMemberRequest(session_id=SRC[1].name))
        results.append(_pass(f"added 2 members, built {bool(doc['built_at'])}")
                       if len(doc["members"]) == 2 else _fail(f"add wrong: {doc}"))

        try:
            srv.add_collection_member(cid, srv.AddMemberRequest(session_id=SRC[0].name))
            results.append(_fail("duplicate accepted"))
        except HTTPException as e:
            results.append(_pass("duplicate -> 400") if e.status_code == 400
                           else _fail(f"duplicate -> {e.status_code}"))

        d = srv.get_collection(cid)
        results.append(_pass("detail: is_active + build summary present")
                       if d["is_active"] and d.get("build") else _fail(f"detail: {d}"))

        cid2 = srv.create_collection(srv.CreateCollectionRequest(name="Sam"))["id"]
        results.append(_pass("second person created; first left open")
                       if srv.get_collection(cid)["closed_at"] is None
                       else _fail("creating one closed another"))

        srv.activate_collection(cid)
        results.append(_pass("switched the default target")
                       if srv.list_collections()["active_id"] == cid
                       else _fail("activate failed"))

        d = srv.remove_collection_member(cid, SRC[1].name)
        results.append(_pass("removed a member and rebuilt")
                       if len(d["members"]) == 1 else _fail(f"remove wrong: {d}"))

        srv.close_collection(cid2)
        try:
            srv.add_collection_member(cid2, srv.AddMemberRequest(session_id=SRC[0].name))
            results.append(_fail("closed collection accepted a member"))
        except HTTPException as e:
            results.append(_pass("closed -> 400") if e.status_code == 400
                           else _fail(f"closed -> {e.status_code}"))

        try:
            srv.get_collection_file(cid, "../../../etc/passwd")
            results.append(_fail("traversal was served"))
        except HTTPException as e:
            results.append(_pass(f"traversal blocked ({e.status_code})")
                           if e.status_code in (403, 404) else _fail(str(e)))

        resp = srv.get_collection_file(cid, "metrics.json")
        results.append(_pass("serves the built metrics.json")
                       if Path(resp.path).name == "metrics.json"
                       else _fail("metrics not served"))

        # an unknown collection must 404, not 500
        try:
            srv.get_collection("does-not-exist")
            results.append(_fail("unknown collection did not raise"))
        except HTTPException as e:
            results.append(_pass("unknown collection -> 404") if e.status_code == 404
                           else _fail(f"unknown -> {e.status_code}"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

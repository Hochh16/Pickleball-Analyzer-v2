"""Smoke test for the collection layer.

Exercises the operator's stated requirements rather than the code's surface: a video
joins or stands alone, a new collection can be started at any point without disturbing
the old one, the same video cannot be counted twice, an unsupported venue is refused, and
a rebuild reproduces the sum of its members.

Usage:
    python -m app.test_collections
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.collections import CollectionError, CollectionStore

ROOT = Path("data/_coll_test")
A, B = Path("data/pb_2min"), Path("data/pb_outdoor2_excerpt")


def _fail(m): print(f"  FAIL: {m}"); return False
def _pass(m): print(f"  PASS: {m}"); return True


def counts(folder: Path) -> dict:
    m = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["match"]
    return {k: m[k]["value"] for k in ("n_shots", "n_rallies", "n_bounces")}


def main() -> int:
    print("Collection layer smoke test")
    print()
    for src in (A, B):
        if not (src / "classified.json").exists():
            print(f"missing {src} — run the pipeline on it first")
            return 1
    if ROOT.exists():
        shutil.rmtree(ROOT)
    store = CollectionStore(ROOT)
    results = []

    # standalone is the default: a fresh store has no active collection, so nothing can
    # be absorbed by accident
    results.append(_pass("no active collection by default")
                   if store.active() is None else _fail("something was active at startup"))

    c = store.create("David outdoor")
    results.append(_pass(f"created + activated {c['id']}")
                   if store.active() and store.active()["id"] == c["id"]
                   else _fail("create did not activate"))

    store.add(c["id"], A, rebuild=False)
    doc = store.add(c["id"], B)
    results.append(_pass(f"2 members, built at {doc['built_at']}")
                   if len(doc["members"]) == 2 and doc["built_at"]
                   else _fail(f"membership/build wrong: {doc}"))

    # the union must equal the sum of its members
    got = counts(store.folder(c["id"]))
    want = {k: counts(A)[k] + counts(B)[k] for k in got}
    results.append(_pass(f"union counts = members summed {got}")
                   if got == want else _fail(f"union {got} != members {want}"))

    # opponents pooled (D1), partner kept separate
    tr = json.loads((store.folder(c["id"]) / "track_roles.json").read_text(encoding="utf-8"))
    pooled = (len(tr["roles"]["opp_b"]["track_ids"]) == 0
              and len(tr["roles"]["opp_a"]["track_ids"]) > 0
              and len(tr["roles"]["partner"]["track_ids"]) > 0)
    results.append(_pass("opponents pooled, partner kept separate") if pooled
                   else _fail("role pooling wrong"))

    # the same video cannot be counted twice
    try:
        store.add(c["id"], A)
        results.append(_fail("duplicate member was accepted"))
    except CollectionError:
        results.append(_pass("duplicate member refused"))

    # an unsupported venue is refused, not merged (D4)
    try:
        store.add(c["id"], B, venue_ok=False)
        results.append(_fail("unsupported venue was accepted"))
    except CollectionError:
        results.append(_pass("unsupported venue refused"))

    # removing a member rebuilds to the remaining member alone
    doc = store.remove(c["id"], B.name)
    results.append(_pass("remove rebuilt to the remaining member")
                   if counts(store.folder(c["id"])) == counts(A)
                   else _fail(f"after remove: {counts(store.folder(c['id']))} != {counts(A)}"))

    # MULTI-PERSON: starting a collection for another player must NOT stop the first
    # player's from accepting videos. Creating selects; it does not close.
    c2 = store.create("Sam outdoor")
    first = [x for x in store.list() if x["id"] == c["id"]][0]
    ok = (first["closed_at"] is None and store.active()["id"] == c2["id"])
    results.append(_pass("second person's collection created; first left open") if ok
                   else _fail(f"create closed another collection: {first.get('closed_at')}"))

    doc = store.add(c["id"], B)
    results.append(_pass("first person's collection still accepts videos while another "
                         "is active")
                   if len(doc["members"]) == 2 else _fail("add to non-active failed"))

    store.set_active(c["id"])
    results.append(_pass("can switch which collection new videos default to")
                   if store.active()["id"] == c["id"] else _fail("set_active failed"))

    # closing is an explicit, separate action -- and it is not deletion
    store.close(c2["id"])
    closed = [x for x in store.list() if x["id"] == c2["id"]][0]
    results.append(_pass("close is explicit; collection remains listed")
                   if closed["closed_at"] else _fail("close did not record"))

    try:
        store.add(c2["id"], B)
        results.append(_fail("closed collection accepted a member"))
    except CollectionError:
        results.append(_pass("closed collection refuses new members"))

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    shutil.rmtree(ROOT, ignore_errors=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

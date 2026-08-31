"""Re-running a video must be able to supersede its old analysis in a collection.

Re-analysing a clip -- for a better calibration, a corrected player click -- produces a
NEW session id for the SAME footage. add() rightly refuses it, because adding both would
count every shot twice. But refusing was the whole story: the operator finished a run
specifically to fix a video already in the cumulative report and had no way to get it in.

These pin the swap: the member count must not grow, the new analysis must be the one that
counts, and a video that is NOT already a member must not be silently swapped in.
"""
from __future__ import annotations

import json

import pytest

from app.collections import CollectionError, CollectionStore, DuplicateVideoError

FOOTAGE = b"\x00\x01the same footage, byte for byte\x02\x03"


def _session(root, name, footage=FOOTAGE):
    """A folder that looks analysed enough for membership bookkeeping."""
    d = root / name
    d.mkdir(parents=True)
    (d / "video.mp4").write_bytes(footage)
    (d / "classified.json").write_text(json.dumps({"shots": []}), encoding="utf-8")
    return d


@pytest.fixture()
def store(tmp_path):
    return CollectionStore(tmp_path / "data")


def test_a_rerun_replaces_the_old_analysis_instead_of_doubling_it(store, tmp_path):
    c = store.create("David2")
    first = _session(tmp_path, "clip-2")
    store.add(c["id"], first, rebuild=False)

    rerun = _session(tmp_path, "clip-11")          # same bytes, new session id
    doc = store.replace(c["id"], rerun, rebuild=False)

    assert len(doc["members"]) == 1, "replacing must not grow the collection"
    m = doc["members"][0]
    assert m["session_id"] == "clip-11", "the re-run must be the analysis that counts"
    assert m["replaced"] == ["clip-2"], "the superseded run should be recorded"


def test_adding_the_rerun_is_still_refused_and_says_what_it_duplicates(store, tmp_path):
    c = store.create("David2")
    store.add(c["id"], _session(tmp_path, "clip-2"), rebuild=False)

    with pytest.raises(DuplicateVideoError) as e:
        store.add(c["id"], _session(tmp_path, "clip-11"), rebuild=False)
    # The UI turns this into a "Replace clip-2" button, so the id has to be on the error.
    assert e.value.replaces == "clip-2"
    assert e.value.session_id == "clip-11"
    assert len(store.get_doc(c["id"])["members"]) == 1, "a refused add must change nothing"


def test_replace_refuses_a_video_that_is_not_already_a_member(store, tmp_path):
    """Otherwise "replace" would be a way to drop a member and add an unrelated clip."""
    c = store.create("David2")
    store.add(c["id"], _session(tmp_path, "clip-2"), rebuild=False)

    other = _session(tmp_path, "different-clip", footage=b"entirely different footage")
    with pytest.raises(CollectionError, match="not a re-run"):
        store.replace(c["id"], other, rebuild=False)
    assert len(store.get_doc(c["id"])["members"]) == 1


def test_replacing_leaves_the_other_members_alone(store, tmp_path):
    c = store.create("David2")
    store.add(c["id"], _session(tmp_path, "clip-2"), rebuild=False)
    store.add(c["id"], _session(tmp_path, "other", footage=b"a second video"),
              rebuild=False)

    doc = store.replace(c["id"], _session(tmp_path, "clip-11"), rebuild=False)
    assert sorted(m["session_id"] for m in doc["members"]) == ["clip-11", "other"]

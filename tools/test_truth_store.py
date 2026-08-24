"""Tests for the accumulating truth store.

The operator's complaint this exists to answer: "the info I provide on shots should be saved
as truths and built upon so I don't have to keep reviewing the same info." So the failures
that matter are losing a fact, or letting a weaker source overwrite a stronger one.
"""
from __future__ import annotations

from tools import truth_store as ts


def test_a_better_source_supersedes_and_the_change_is_kept():
    """Import order used to decide the winner, so a 2026-08-17 spreadsheet overruled a review
    done today. Authority decides, and what was replaced stays visible."""
    doc = ts.empty("v.mp4")
    ts.add_shot(doc, 10.0, type_="drive", source="old.csv", kind="labels_csv")
    v = ts.add_shot(doc, 10.0, type_="drop", source="review", kind="review")
    assert v == "updated"
    s = doc["shots"][0]
    assert s["type"] == "drop"
    assert s["superseded"][0] == {"field": "type", "was": "drive", "now": "drop",
                                  "by": "review"}


def test_a_weaker_source_cannot_overwrite():
    doc = ts.empty("v.mp4")
    ts.add_shot(doc, 10.0, type_="drop", source="review", kind="review")
    ts.add_shot(doc, 10.0, type_="drive", source="old.csv", kind="labels_csv")
    assert doc["shots"][0]["type"] == "drop"
    assert "superseded" not in doc["shots"][0]


def test_equal_authority_disagreement_is_a_conflict_not_a_silent_pick():
    doc = ts.empty("v.mp4")
    ts.add_shot(doc, 10.0, type_="drop", source="a", kind="review")
    v = ts.add_shot(doc, 10.0, type_="lob", source="b", kind="review")
    assert v == "CONFLICT"
    assert doc["shots"][0]["type"] == "drop"          # first kept, both recorded
    assert doc["shots"][0]["conflicts"][0]["rejected"] == "lob"


def test_one_stored_shot_cannot_absorb_several_distinct_ones():
    """A +/-1 s window is right for hand-typed times and wide enough to span three shots of a
    kitchen exchange. Without one-to-one claiming, one row swallowed them all and then
    'conflicted' with every one -- 117 false conflicts on the first import."""
    doc = ts.empty("v.mp4")
    claimed: set = set()
    for t, ty in ((10.0, "dink"), (10.4, "dink"), (10.8, "drive")):
        ts.add_shot(doc, t, type_=ty, source="review", kind="review", claimed=claimed)
    assert len(doc["shots"]) == 3
    assert not any(s.get("conflicts") for s in doc["shots"])


def test_hitter_naming_is_normalised():
    """The operator writes 'opponent'; track_roles says opp_a. Comparing those raw invents a
    disagreement out of a naming difference."""
    assert ts.norm_hitter("opp_a") == ts.norm_hitter("opponent") == "opponent"
    assert ts.norm_hitter("Partner") == "partner"
    assert ts.norm_hitter("") is None
    doc = ts.empty("v.mp4")
    ts.add_shot(doc, 5.0, hitter="opp_b", source="a", kind="labels_csv")
    v = ts.add_shot(doc, 5.0, hitter="opponent", source="b", kind="review")
    assert v == "agreed"


def test_enrichment_adds_without_conflicting():
    doc = ts.empty("v.mp4")
    ts.add_shot(doc, 5.0, type_="dink", source="a", kind="labels_csv")
    v = ts.add_shot(doc, 5.0, hitter="user", side="near", source="b", kind="labels_csv")
    assert v == "enriched"
    s = doc["shots"][0]
    assert s["type"] == "dink" and s["hitter"] == "user" and s["side"] == "near"


def test_store_is_keyed_by_video_not_by_folder():
    """Eight analysed folders exist for one video. Truth belongs to the video, or it is
    re-collected every time a clip is re-analysed."""
    a = ts.store_path("PB 5 minute outdoor.mp4")
    b = ts.store_path("PB 5 minute outdoor.mp4")
    assert a == b and a.suffix == ".json"
    assert ts.store_path("PB 3 min indoor 1 court B.mp4") != a

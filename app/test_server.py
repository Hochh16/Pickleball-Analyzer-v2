

def test_the_session_list_says_which_videos_have_a_report(tmp_path):
    """Three things pull on which setup a video's row should be, and they conflict:

      * finish a video, re-run court setup a few times, then want the REPORT -- ranking by
        "newest" hides the finished session behind empty attempts;
      * deliberately re-do setup to fix the player identity and want to RUN THAT -- ranking
        by "has a report" hides the new setup behind the old finished one;
      * open the wizard and back out -- that leaves an empty stub which is newest of all.

    So the ROW is the newest CONFIGURED setup (a live job first), and the report link
    follows whichever setup actually produced a report, even an earlier one.
    """
    from app import server
    sessions = server.list_sessions()["sessions"]
    assert sessions, "no sessions to check"
    for s in sessions:
        assert "has_report" in s and "report_session_id" in s

        # the row must be a setup someone can actually continue, whenever one exists
        steps = s.get("steps") or {}
        if not (steps.get("calibration") and steps.get("roster")):
            same = [o for o in server.list_sessions(all=True)["sessions"]
                    if str(o.get("video_path")) == str(s.get("video_path"))]
            assert not any((o.get("steps") or {}).get("calibration")
                           and (o.get("steps") or {}).get("roster") for o in same), (
                f"{s['id']} is an unconfigured stub shown over a configured setup")

        # and the report link must point at a session that really has one
        if s["has_report"]:
            rep = server.store.folder(str(s["report_session_id"])) / "report.html"
            assert rep.exists(), f"{s['id']} links to a report that is not there"
        else:
            assert s["report_session_id"] is None


def test_every_finished_report_is_listable_from_one_place():
    """The operator: "HOW DO I SEE existing reports? SHOULD BE VERY CLEAR!!!"

    Every finished report -- cumulative and per video -- has to be reachable from the start
    screen without loading a session first. Before this, the only link was bound to the
    session currently loaded, so finishing a second video made the first unreachable.
    """
    from app import server
    cols = server.list_collections()["collections"]
    for c in cols:
        assert "has_report" in c and "n_members" in c
        if c["has_report"]:
            assert (server.collections.folder(str(c["id"])) / "report.html").exists()
    sessions = server.list_sessions()["sessions"]
    listable = [s for s in sessions if s["has_report"]]
    # nothing with a report on disk may be missing from the list
    for s in sessions:
        if not s["has_report"]:
            same = [o for o in server.list_sessions(all=True)["sessions"]
                    if str(o.get("video_path")) == str(s.get("video_path"))]
            assert not any((server.store.folder(str(o["id"])) / "report.html").exists()
                           for o in same), f"{s['id']}: a report exists but is not listed"
    assert listable, "no per-video reports listed"


def test_the_ui_is_not_served_from_a_stale_browser_cache():
    """A fix to the UI was deployed, the app restarted, and the operator still saw the old
    screen -- the browser was holding app.js. Static assets and index.html must revalidate,
    or every UI change needs the operator to know about hard-refresh."""
    from app import server
    resp = server.index()
    assert "no-cache" in resp.headers.get("cache-control", "")
    mounted = [r for r in server.app.routes if getattr(r, "name", "") == "static"]
    assert mounted, "static mount missing"
    assert isinstance(mounted[0].app, server._NoCacheStatic)

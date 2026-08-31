

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

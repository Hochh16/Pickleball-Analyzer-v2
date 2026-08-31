

def test_the_session_list_says_which_videos_have_a_report(tmp_path):
    """The UI's only "View report" link was bound to the session currently loaded, so once
    the operator moved on there was no route back to an earlier video's report -- the file
    sat on disk, reachable only by typing the path. The list now says which have one."""
    # Called directly rather than over HTTP: httpx (and so starlette's TestClient) is not
    # a dependency of this project.
    from app import server
    sessions = server.list_sessions()["sessions"]
    assert sessions, "no sessions to check"
    assert all("has_report" in s for s in sessions), "every row must say whether it has one"
    for s in sessions:
        on_disk = (server.store.folder(str(s["id"])) / "report.html").exists()
        assert s["has_report"] is on_disk, f"{s['id']}: has_report disagrees with disk"

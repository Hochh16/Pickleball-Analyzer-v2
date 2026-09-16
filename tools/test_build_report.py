

def test_serves_and_returns_are_counted_the_same_way():
    """The report put "47 serves" beside "56 returns", which reads as nine missing serves.
    They were different questions: serves came from the shot MIX (shots we typed "serve")
    and returns from the STRUCTURAL count (the shot after the serve). Every rally has a
    serve whether or not we could type it, and a serve we retype as the return it really is
    still opened a rally. Structural against structural gives 58 vs 56 -- a real, small gap.
    """
    import inspect
    from tools import build_report
    src = inspect.getsource(build_report)
    block = src[src.index("match_counts = {"):]
    i = block.index('"n_serves":')
    j = block.index('"n_returns":')
    seg = block[i:j]
    assert 'get("serve"' in seg and "match" in seg, (
        'n_serves must come from match.serve (structural), not from the shot-type mix')
    assert 'by_shot_type' not in seg, "n_serves must not be read from the shot-type mix"


def test_the_third_shot_line_counts_third_shots_not_typed_decisions():
    """It rendered n_third_decisions -- the subset we could type from a landing -- and showed
    "0 of 3" beside a rating card reading n=9 for the same player in the same report."""
    from tools.build_report import third_shot_line
    drivers = {"n_third_shots": 9, "n_third_shot_drops": 1,
               "n_third_decisions": 3, "third_shot_by_type": {"drive": 3},
               "n_third_unmeasurable": 2}
    line = third_shot_line(drivers, 49, n_videos=5)
    assert "1 of 9" in line, line
    assert "49 third shot" in line, line
    # and it still degrades to the old counts for metrics.json written before the split
    old = third_shot_line({"n_third_decisions": 3, "third_shot_by_type": {"drop": 1}}, 16)
    assert "1 of 3" in old, old


def test_the_moments_list_uses_the_same_third_shot_rule_as_the_metrics():
    """The report had its OWN copy of the index rule, taking rally position 3, and listed
    "Third shots (7)" beside a metrics-derived 9 for the same player in the same report.
    One definition, imported, so they cannot drift again."""
    from tools.build_report import user_shot_groups
    # a rally that does NOT run serve-return-third: the serve was missed, so the rally opens
    # on the return and position 3 is the FOURTH ball.
    classified = {"shots": [
        {"shot_id": 0, "track_id": 1, "t_sec": 1.0, "shot_type": "return"},
        {"shot_id": 1, "track_id": 9, "t_sec": 2.0, "shot_type": "drive"},   # the third shot
        {"shot_id": 2, "track_id": 9, "t_sec": 3.0, "shot_type": "dink"},
    ]}
    rallies = [{"shot_ids": [0, 1, 2]}]
    roles = {"track_roles": {"9": {"role": "user"}}}
    groups = user_shot_groups(classified, rallies, roles)
    assert groups["Third shots"] == [2.0], (
        "must take the shot after the RETURN, not rally position 3")


def test_the_identity_warning_names_the_videos_that_were_guessed(tmp_path):
    """A cumulative report inherits the LOWEST seed confidence of its members, so ONE
    unclicked video made the banner read "nobody marked which player to analyse" across six
    videos of which five were clicked -- wrong, and unactionable, because it does not say
    which one to re-do."""
    import json
    from tools.build_report import unclicked_members
    def _member(name, basis, conf):
        d = tmp_path / name
        d.mkdir()
        (d / "track_roles.json").write_text(json.dumps(
            {"track_roles": {"1": {"role": "user", "basis": basis, "confidence": conf}}}),
            encoding="utf-8")
        return {"session_id": name, "path": str(d)}
    coll = {"members": [_member("clicked", "click", 0.95),
                        _member("guessed", "starting-corner", 0.5),
                        _member("appearance", "appearance+height", 0.95)]}
    got = unclicked_members(coll)
    assert [n for n, _b, _c in got] == ["guessed", "appearance"]
    assert unclicked_members({"members": []}) == []
    assert unclicked_members(None) == []


# --- the redesign (docs/REPORT_REDESIGN.md) ----------------------------------

def test_worth_column_shows_a_number_when_it_scores_and_a_bucket_when_it_cannot():
    """The three no-gain cases must not read alike. An earlier draft showed one dash for
    all of them, which told a player that unforced errors -- which USAPA grades -- were
    simply not worth working on."""
    from tools.build_report import worth_html
    assert "+0.07" in worth_html({"driver": "Time at the kitchen line", "d_rating": 0.0667,
                                  "verdict": "scores"})
    floor = worth_html({"driver": "Returns landing in", "d_rating": 0.0, "verdict": "below_floor",
                        "d_rating_max": 0.06})
    assert "+0.06" in floor and "under where the scale starts" in floor
    gap = worth_html({"driver": "Unforced errors", "d_rating": 0.0, "verdict": "inert"})
    assert "🔴" in gap and "14 times in 23" in gap
    assert "🟠" in worth_html({"driver": "Court covered", "d_rating": 0.0, "verdict": "inert"})
    assert "🟢" in worth_html({"driver": "Serves landing in", "d_rating": 0.0, "verdict": "inert"})


def test_category_rows_are_ordered_by_measured_worth_and_counts_are_context():
    from tools.build_report import category_rows
    drivers = {"dink_count": 23, "popup": {"n_measured": 13, "n_popped": 4, "popup_frac": 0.31},
               "mean_rally_length": 6.5}
    lev = {"dink": [
        {"driver": "Pop-ups given up", "d_rating": 0.0246, "verdict": "scores"},
        {"driver": "Average rally length", "d_rating": 0.0129, "verdict": "scores"},
        {"driver": "Share of your shots that are dinks", "d_rating": 0.0158, "verdict": "scores"},
    ]}
    rows = category_rows("dink", drivers, lev, {"dink_count": 94}, 6)
    worths = [w for w, _l, _v, _h in rows]
    assert worths == sorted(worths, reverse=True), rows
    assert [l for _w, l, _v, _h in rows][0] == "Dinks the opponent took above the waist"
    # a count still shows the match total beside the user's share
    count_row = next(r for r in rows if r[1] == "Dinks detected")
    assert "94" in count_row[2] and "23" in count_row[2]


def test_zone_rows_refuse_to_publish_a_split_the_tracking_has_not_earned():
    """Far-side roles fail two ways on David2: opp_b has no tracked frames at all, and
    opp_a is tracked but reads 3% of rallies at the kitchen line, which no pickleball
    player does. Publishing either as a percentage would invent a fact."""
    from tools.build_report import zone_rows
    good = {"players": {"user": {"position": {"confidence": 0.98, "value": {
        "n_frames": 26320, "zone_time_frac": {"kitchen": 0.43, "transition": 0.19,
                                              "baseline": 0.38}}}}}}
    assert [round(f, 2) for _l, f in zone_rows(good, "user")] == [0.43, 0.19, 0.38]
    assert zone_rows({"players": {"opp_b": {"position": {"confidence": 0.0,
                                                         "value": {"n_frames": 0}}}}},
                     "opp_b") is None
    low = {"players": {"opp_a": {"position": {"confidence": 0.774, "value": {
        "n_frames": 33339, "zone_time_frac": {"kitchen": 0.03, "transition": 0.45,
                                              "baseline": 0.52}}}}}}
    assert zone_rows(low, "opp_a") is None
    contaminated = {"players": {"partner": {"role_contaminated": True, "position": {
        "confidence": 0.99, "value": {"n_frames": 100,
                                      "zone_time_frac": {"kitchen": 0.5}}}}}}
    assert zone_rows(contaminated, "partner") is None


def test_ball_views_colour_by_what_happened_and_mark_the_weak_ones():
    from tools.build_report import ball_views
    classified = {"shots": [
        {"shot_id": 0, "track_id": 1, "t_sec": 1.0, "shot_type": "serve",
         "hitter_court_xy_ft": [5.0, 2.0]},
        {"shot_id": 1, "track_id": 1, "t_sec": 2.0, "shot_type": "volley", "is_volley": True,
         "hitter_court_xy_ft": [9.0, 17.0]},
        {"shot_id": 2, "track_id": 2, "t_sec": 3.0, "shot_type": "drive",
         "hitter_court_xy_ft": [11.0, 30.0]},
    ]}
    rallies = [{"shot_ids": [0, 1, 2], "end_reason": "ball-out"}]
    bounces = [{"bounce_id": 0, "frame": 10, "between_shots": [0, 1],
                "court_xy_ft": [10.0, 33.0], "is_in_court": True},
               {"bounce_id": 1, "frame": 30, "between_shots": [2, None],
                "court_xy_ft": [10.0, -3.0], "is_in_court": False}]
    roles = {"track_roles": {"1": {"role": "user"}, "2": {"role": "opp_a"}}}
    v = ball_views(classified, rallies, bounces, roles)
    assert v["Serves & returns"]["points"] == [(10.0, 33.0, "bounce")]
    assert v["Your volleys"]["points"] == [(9.0, 17.0, "volley")]
    # the opponent hit the last shot and it went out -> their mistake, not the user's
    assert v["Opponent mistakes"]["points"] == [(10.0, -3.0, "out")]
    assert v["Your errors"]["points"] == [] and v["Your winners"]["points"] == []
    assert v["Opponent mistakes"]["weak"] and not v["Serves & returns"]["weak"]


def test_the_evidence_strip_needs_at_least_two_sessions():
    from tools.build_report import evidence_svg, session_day
    assert evidence_svg([("29 Aug", 28)]) == ""
    svg = evidence_svg([("29 Aug", 28), ("31 Aug", 12)])
    assert "polyline" in svg and "29 Aug" in svg and ">40<" in svg   # cumulative top label
    assert session_day({"captured_at": "2026-08-29T00:30:40+00:00"}) == "29 Aug"
    assert session_day({}) is None


def test_ball_view_layers_switch_by_class_because_hidden_does_not_work_on_svg():
    """The buttons switched and the picture did not: `hidden` is an HTML attribute and
    browsers ignore it on SVG elements, so all five views stayed drawn at once."""
    import re
    from tools.build_report import ball_views_svg
    svg = ball_views_svg({
        "A": {"points": [(1.0, 2.0, "bounce")], "weak": False, "note": "a"},
        "B": {"points": [(3.0, 4.0, "volley")], "weak": True, "note": "b"},
    })
    groups = re.findall(r'<g class="([^"]+)" data-view="(\d+)">', svg)
    assert groups == [("bv-layer on", "0"), ("bv-layer", "1")], groups
    assert "<g class=\"bv-layer\" data-view=\"1\" hidden" not in svg
    assert ".bv-layer{display:none" not in svg          # the rule lives in the stylesheet
    assert 'o.classList.toggle("on"' in svg             # layers toggle by class
    assert 'o.hidden=(o.dataset.view!==v)' in svg       # notes are HTML, so hidden is fine

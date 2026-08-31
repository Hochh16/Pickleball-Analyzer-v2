

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

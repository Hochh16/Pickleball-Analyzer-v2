"""The report must say so when it does not know which player is the user.

Everything in a player report is per-player. If Stage 2.5 picked the wrong near-side
player, every number belongs to the partner and NOTHING ELSE in the report looks wrong --
there is no second signal. That is why the low-confidence case gets a banner instead of a
log line, and why it is asserted here.
"""
from tools.build_report import user_seed_basis


def test_click_seed_is_trusted():
    roles = {"track_roles": {"7": {"role": "user", "basis": "click", "confidence": 0.95}}}
    assert user_seed_basis(roles) == ("click", 0.95)


def test_corner_seed_is_reported_as_a_guess():
    roles = {"track_roles": {"2": {"role": "user", "basis": "starting-corner",
                                   "confidence": 0.5}}}
    basis, conf = user_seed_basis(roles)
    assert basis == "starting-corner" and conf == 0.5


def test_user_role_found_among_other_roles():
    roles = {"track_roles": {
        "1": {"role": "partner", "basis": "simultaneous-with-user", "confidence": 0.8},
        "3": {"role": "opp_a", "basis": "appearance+height", "confidence": 0.75},
        "2": {"role": "user", "basis": "starting-corner", "confidence": 0.5}}}
    assert user_seed_basis(roles)[0] == "starting-corner"


def test_no_user_role_is_not_an_error():
    assert user_seed_basis({"track_roles": {}}) == (None, None)
    assert user_seed_basis({}) == (None, None)


def test_a_click_anywhere_among_the_user_fragments_counts_as_clicked():
    """The tracker fragments the user into many tracks; the click seeds ONE of them and the
    rest are linked to it by appearance+height. Reading whichever landed first in the file
    made the report accuse the operator of not marking themselves in a video they HAD
    marked -- pb_3_min_indoor_2_court_b-2, 8 user tracks, the click on track 40."""
    roles = {"track_roles": {
        "21":  {"role": "user", "basis": "appearance+height", "confidence": 0.95},
        "40":  {"role": "user", "basis": "click",             "confidence": 0.95},
        "134": {"role": "user", "basis": "appearance+height", "confidence": 0.516}}}
    assert user_seed_basis(roles) == ("click", 0.95)


def test_no_click_among_the_fragments_is_still_a_guess():
    """The other half of the same report: pb_3_min_indoor_1_court_b-2 has 11 user tracks
    and no click among them, so it must stay flagged."""
    roles = {"track_roles": {
        "2":   {"role": "user", "basis": "starting-corner",   "confidence": 0.679},
        "169": {"role": "user", "basis": "appearance+height", "confidence": 0.747}}}
    assert user_seed_basis(roles) == ("starting-corner", 0.679)

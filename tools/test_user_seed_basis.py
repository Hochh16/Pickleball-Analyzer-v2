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

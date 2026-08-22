from pickleball_vision.match_roster import parse_match_roster


def test_parses_compact_youtube_doubles_title() -> None:
    roster = parse_match_roster("8/17/26 ChhayDiana vs JohnNana")

    assert roster is not None
    assert roster.teams == (("Chhay", "Diana"), ("John", "Nana"))
    assert roster.as_dict()["playerNames"] == ["Chhay", "Diana", "John", "Nana"]


def test_parses_explicit_team_separators() -> None:
    roster = parse_match_roster("John & Denny vs Diana + Oksana")

    assert roster is not None
    assert roster.teams == (("John", "Denny"), ("Diana", "Oksana"))


def test_ambiguous_title_does_not_invent_roster() -> None:
    assert parse_match_roster("Thursday night pickleball") is None
    assert parse_match_roster("John vs Diana") is None
    assert parse_match_roster("JohnJohn vs DianaOksana") is None

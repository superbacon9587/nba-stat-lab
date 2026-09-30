"""Tests for the app's pure logic: the query <-> form mapping and the confidence badge."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from nbalab.query import schema as s  # noqa: E402
from ui.formmodel import default_form, form_to_query, query_to_form  # noqa: E402
from ui.logic import confidence_badge, format_value, single_filter_queries  # noqa: E402

LO, HI = 1996, 2025
CURRY, LEBRON, SPURS, SIXERS, WARRIORS = 201939, 2544, 1610612759, 1610612755, 1610612744


def roundtrip(q: s.StatQuery) -> s.StatQuery:
    return form_to_query(query_to_form(q, LO, HI), q, LO, HI)


def test_split_query_roundtrips() -> None:
    q = s.StatQuery(subject_ids=[CURRY], stats=["assists"], filters=[
        s.OpponentTeamFilter(team_ids=[SPURS]), s.HomeAwayFilter(value="away"),
        s.DayOfWeekFilter(days=["Sunday"]), s.SeasonRangeFilter(start=2015, end=2020),
        s.DefenderHeightRangeFilter(min_inches=78, max_inches=80), s.BackToBackFilter(value=False)])
    out = roundtrip(q)
    assert out.mode == "split"
    assert {f.type for f in out.filters} == {f.type for f in q.filters}
    assert out == roundtrip(out)


def test_unhandled_filters_pass_through() -> None:
    q = s.StatQuery(subject_ids=[CURRY], stats=["points"], filters=[
        s.TeammateFilter(person_id=LEBRON, status="out"), s.MonthFilter(months=[1, 2])])
    out = roundtrip(q)
    assert [f.type for f in out.filters] == ["teammate", "month"]


def test_last_n_seasons_becomes_a_season_range() -> None:
    q = s.StatQuery(subject_type="team", subject_ids=[WARRIORS], stats=["team_score"],
                    filters=[s.LastNSeasonsFilter(n=5)])
    f = query_to_form(q, LO, HI)
    assert f["season_range"] == (2021, 2025)


def test_projection_context_roundtrips() -> None:
    q = s.StatQuery(subject_ids=[CURRY], stats=["points", "assists"], mode="projection",
                    projection=s.Projection(lines={"points": 27.5, "assists": 5.5}, context=s.ProjectionContext(
                        opponent_team_id=SIXERS, venue_team_id=WARRIORS, home_away="home",
                        defender_person_id=LEBRON)))
    f = query_to_form(q, LO, HI)
    assert f["line_on"] and f["line"] == 27.5 and f["opponents"] == [SIXERS] and f["defender_id"] == LEBRON
    out = roundtrip(q)
    assert out.mode == "projection"
    assert out.projection.lines == {"points": 27.5, "assists": 5.5}  # secondary line kept from the base
    assert out.projection.context == q.projection.context


def test_editing_the_line_changes_only_the_first_stat() -> None:
    q = s.StatQuery(subject_ids=[CURRY], stats=["points", "assists"], mode="projection",
                    projection=s.Projection(lines={"points": 27.5, "assists": 5.5}))
    f = query_to_form(q, LO, HI)
    f["line"] = 30.5
    out = form_to_query(f, q, LO, HI)
    assert out.projection.lines == {"points": 30.5, "assists": 5.5}


def test_turning_off_the_line_moves_context_into_filters() -> None:
    q = s.StatQuery(subject_ids=[CURRY], stats=["points"], mode="projection",
                    projection=s.Projection(lines={"points": 27.5},
                                            context=s.ProjectionContext(opponent_team_id=SIXERS)))
    f = query_to_form(q, LO, HI)
    f["line_on"] = False
    out = form_to_query(f, q, LO, HI)
    assert out.mode == "split" and out.projection is None
    assert out.filters == [s.OpponentTeamFilter(team_ids=[SIXERS])]


def test_league_wide_query() -> None:
    q = s.StatQuery(stats=["points"], mode="variable_effect", effect_variable="back_to_back",
                    filters=[s.PlayerPositionFilter(positions=["G"])])
    f = query_to_form(q, LO, HI)
    assert f["subject_type"] == "league" and f["positions"] == ["G"]
    assert roundtrip(q) == q


def test_team_form_drops_player_only_filters() -> None:
    f = default_form(LO, HI)
    f.update(subject_type="team", subject_id=WARRIORS, stats=["team_score"], height_on=True, defender_id=LEBRON)
    out = form_to_query(f, None, LO, HI)
    assert out.filters == []


def test_invalid_form_raises() -> None:
    f = default_form(LO, HI)
    f.update(subject_id=CURRY, stats=[])
    with pytest.raises(ValueError):
        form_to_query(f, None, LO, HI)


@pytest.mark.parametrize("n, p, level", [
    (5, 0.001, "low"), (40, math.nan, "low"), (200, 0.6, "low"), (40, 0.01, "high"),
    (15, 0.01, "medium"), (40, 0.1, "medium"), (25, 0.049, "high"), (24, 0.049, "medium"), (10, 0.19, "medium"),
])
def test_confidence_badge(n: int, p: float, level: str) -> None:
    assert confidence_badge(n, p).level == level


def test_single_filter_queries_keep_scope() -> None:
    q = s.StatQuery(subject_ids=[CURRY], stats=["points"], filters=[
        s.LastNSeasonsFilter(n=3), s.OpponentTeamFilter(team_ids=[SPURS]), s.HomeAwayFilter(value="home")])
    singles = single_filter_queries(q)
    assert [f.type for f, _ in singles] == ["opponent_team", "home_away"]
    for f, sq in singles:
        assert [x.type for x in sq.filters] == ["last_n_seasons", f.type]


def test_single_filter_queries_use_projection_context() -> None:
    q = s.StatQuery(subject_ids=[CURRY], stats=["points"], mode="projection",
                    projection=s.Projection(lines={"points": 27.5},
                                            context=s.ProjectionContext(opponent_team_id=SIXERS, home_away="home")))
    assert {f.type for f, _ in single_filter_queries(q)} == {"opponent_team", "home_away"}


def test_format_value() -> None:
    assert format_value(27.44) == "27.4"
    assert format_value(48.123, is_ratio=True) == "48.1%"
    assert format_value(math.nan) == "–"


# ---------------------------------------------------------------- league-wide team queries (bug)


def test_league_team_query_keeps_team_subject_type() -> None:
    """Regression: the form used to force subject_type="player" for league-wide queries,
    so a team stat raised "unknown player stat 'team_score'"."""
    q = s.StatQuery(subject_type="team", stats=["team_score"], mode="variable_effect", effect_variable="month")
    out = roundtrip(q)
    assert out.subject_type == "team" and out.stats == ["team_score"] and out.mode == "variable_effect"


def test_league_subject_type_follows_the_stats() -> None:
    from ui.formmodel import league_subject_type

    assert league_subject_type(["team_score"]) == "team"
    assert league_subject_type(["points"]) == "player"
    assert league_subject_type(["assists"]) == "player"  # shared stat, no context: player
    team_q = s.StatQuery(subject_type="team", stats=["assists"], mode="variable_effect", effect_variable="month")
    assert league_subject_type(["assists"], team_q) == "team"  # shared stat keeps the query's type

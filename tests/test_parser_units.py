"""Unit tests for the parser's building blocks: seasons, measures, names, defaults, builder."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from nbalab.data.config import DATA_DIR
from nbalab.nlp.build import BuildSettings, QueryBuilder
from nbalab.nlp.defaults import half_point_line, recent_value
from nbalab.nlp.entities import EntityIndex, clean_name
from nbalab.nlp.measures import find_height, height_range, parse_height_inches, parse_weight_lbs, weight_range
from nbalab.nlp.rules import parse_rules
from nbalab.nlp.seasons import current_season, last_completed_season, last_n_completed, season_label
from nbalab.nlp.types import QueryDraft
from tests.nlp_fixture import fixture_index

TODAY = date(2026, 9, 29)


@pytest.fixture(scope="module")
def index() -> EntityIndex:
    return fixture_index()


# ----------------------------------------------------------------------- seasons


@pytest.mark.parametrize("today,current,completed", [
    (date(2026, 9, 29), 2025, 2025),  # offseason: 2025-26 is both
    (date(2026, 11, 1), 2026, 2025),  # 2026-27 under way
    (date(2027, 3, 1), 2026, 2025),
    (date(2027, 7, 1), 2026, 2026),  # Finals over
])
def test_current_and_completed_season(today: date, current: int, completed: int) -> None:
    assert current_season(today) == current
    assert last_completed_season(today) == completed


def test_last_n_completed_and_label() -> None:
    assert last_n_completed(5, TODAY) == (2021, 2025)
    assert last_n_completed(1, date(2027, 1, 15)) == (2025, 2025)  # the half-played season is left out
    assert season_label(1999) == "1999-00"


# ---------------------------------------------------------------------- measures


@pytest.mark.parametrize("text,inches", [
    ("6'7", 79), ("6'7\"", 79), ("6’ 7”", 79), ("6 foot 7", 79), ("6 feet 7 inches", 79), ("6ft 7in", 79),
    ("6-7", 79), ("79 inches", 79), ("79", 79), ("201 cm", 79.1), ("7 foot", 84), ("seven footer", 84),
    ("6'10", 82), ("6'7.5", 79.5),
])
def test_parse_height(text: str, inches: float) -> None:
    assert parse_height_inches(text) == pytest.approx(inches)


def test_heights_ignore_seasons_and_records() -> None:
    assert find_height("since 2020-21") is None
    assert find_height("a 6-7 record on the road") is None
    assert find_height("a 6-7 defender")[0] == 79


def test_height_and_weight_ranges() -> None:
    assert height_range(79) == (78, 80)
    assert height_range(79, tolerance=2) == (77, 81)
    assert height_range(82, "at_least") == (82, 96)
    assert height_range(75, "at_most") == (60, 75)
    assert parse_weight_lbs("250 lbs") == 250 and weight_range(250) == (240, 260)
    assert parse_weight_lbs("heavy") is None


# ------------------------------------------------------------------------- names


def test_clean_name() -> None:
    assert clean_name("the Spurs") == "spurs"
    assert clean_name("Curry's") == "curry"


def test_last_name_with_clear_winner(index: EntityIndex) -> None:
    res = index.resolve_player("Curry")
    assert res.status == "clear" and res.id == 201939
    assert {c.name for c in res.candidates} >= {"Seth Curry", "Dell Curry"}


def test_full_name_collision_is_ambiguous(index: EntityIndex) -> None:
    res = index.resolve_player("Mike James")
    assert res.status == "ambiguous" and [c.id for c in res.candidates] == [2229, 1628455]


def test_seasons_narrow_a_collision(index: EntityIndex) -> None:
    assert index.resolve_player("Glen Rice", (2013, 2013)).status == "unique"
    assert index.resolve_player("Glen Rice", (2013, 2013)).id == 203318


@pytest.mark.parametrize("text,person_id", [
    ("LeBron", 2544), ("King James", 2544), ("Lebrom James", 2544), ("Steph", 201939), ("SGA", None),
])
def test_player_lookup(index: EntityIndex, text: str, person_id: int | None) -> None:
    assert index.resolve_player(text).id == person_id


@pytest.mark.parametrize("text,team_id", [
    ("the Spurs", 1610612759), ("Philly", 1610612755), ("Sixers", 1610612755), ("Dubs", 1610612744),
    ("celtcs", 1610612738), ("Hornets", 1610612766),
])
def test_team_lookup(index: EntityIndex, text: str, team_id: int) -> None:
    assert index.resolve_team(text).id == team_id


def test_team_names_are_dated(index: EntityIndex) -> None:
    assert index.resolve_team("Hornets", (2004, 2004)).id == 1610612740  # New Orleans then
    assert index.resolve_team("Sonics").id == 1610612760
    assert index.resolve_team("LA").status == "ambiguous"


def test_venues(index: EntityIndex) -> None:
    assert index.resolve_venue("San Francisco", 2025) == [1610612744]
    assert index.resolve_venue("MSG") == [1610612752]
    assert index.resolve_venue("Boston") == [1610612738]
    assert index.resolve_venue("Los Angeles") == [1610612746, 1610612747]
    assert index.resolve_venue("Atlantis") == []


def test_rosters_by_season(index: EntityIndex) -> None:
    assert index.teams_in_season(2544, 2025) == {1610612747}
    assert index.teams_in_season(2544, 2012) == {1610612748}
    assert index.ever_played_for(201935, 1610612755) == (2021, 2022)
    assert index.ever_played_for(2544, 1610612755) is None


def test_common_words_are_not_names(index: EntityIndex) -> None:
    d = parse_rules("how does day of week affect green's rebounds on the road", index)
    assert d.subjects == []  # "day" is not Todd Day, lowercase "green" is not Draymond


# ---------------------------------------------------------------------- defaults


def test_half_point_line() -> None:
    assert half_point_line(26.7) == 26.5
    assert half_point_line(26.2) == 26.5
    assert half_point_line(25.9) == 25.5


def test_recent_value_uses_most_recent_games() -> None:
    games = pd.DataFrame({"game_date": pd.date_range("2026-01-01", periods=5), "points": [10, 10, 30, 30, 30],
                          "fieldGoalsMade": [1, 1, 5, 5, 5], "fieldGoalsAttempted": [10, 10, 10, 10, 0]})
    assert recent_value(games, "player", "points", 3) == 30
    assert recent_value(games, "player", "fg_pct", 3) == pytest.approx(75.0)  # pooled 15/20
    assert recent_value(games.iloc[0:0], "player", "points", 3) is None


# ----------------------------------------------------------------------- builder


def build(index: EntityIndex, draft: QueryDraft, today: date = TODAY, settings: BuildSettings = BuildSettings()):
    b = QueryBuilder(index, today, settings)
    return b.build(draft), b.assumptions, b.unresolved


def test_relative_seasons(index: EntityIndex) -> None:
    q, a, _ = build(index, QueryDraft(subjects=["Tatum"], stats=["points"], relative_season="this"),
                    today=date(2026, 12, 1))
    assert q.scope_filters[0].start == 2026
    q, _, _ = build(index, QueryDraft(subjects=["Tatum"], stats=["points"], relative_season="last"),
                    today=date(2026, 12, 1))
    assert q.scope_filters[0].start == 2025


def test_unknown_stat_is_flagged(index: EntityIndex) -> None:
    q, _, u = build(index, QueryDraft(subjects=["Tatum"], stats=["points", "vibes"]))
    assert q.stats == ["points"] and [(x.kind, x.text) for x in u] == [("unsupported", "vibes")]


def test_team_subject_mislabeled_as_player(index: EntityIndex) -> None:
    q, a, _ = build(index, QueryDraft(subjects=["the Knicks"], stats=["points"]))
    assert q.subject_type == "team" and q.stats == ["team_score"]


def test_defender_filters_rejected_for_teams(index: EntityIndex) -> None:
    q, _, u = build(index, QueryDraft(subject_type="team", subjects=["Knicks"], stats=["points"],
                                      defenders=["LeBron"]))
    assert [f.type for f in q.filters] == ["season_range"]  # only the team default seasons, no defender
    assert u[0].kind == "unsupported"


def test_projection_without_line_provider_notes_missing_line(index: EntityIndex) -> None:
    q, a, _ = build(index, QueryDraft(subjects=["Tatum"], stats=["points"], mode="projection"))
    assert q.projection.lines == {}
    assert any(x.field == "lines.points" for x in a)


def test_split_defender_never_on_opponent_is_flagged(index: EntityIndex) -> None:
    _, _, u = build(index, QueryDraft(subjects=["Tatum"], stats=["points"], opponent_teams=["76ers"],
                                      defenders=["LeBron"]))
    assert [(x.kind, x.field) for x in u] == [("conflict", "defender")]
    _, _, u = build(index, QueryDraft(subjects=["Tatum"], stats=["points"], opponent_teams=["76ers"],
                                      defenders=["Harden"]))
    assert u == []  # Harden did play for Philadelphia


def test_retired_player_gets_history_not_projection(index: EntityIndex) -> None:
    q, a, u = build(index, QueryDraft(subjects=["Todd Day"], stats=["points"], mode="projection",
                                      opponent_teams=["Suns"], lines=[{"stat": "points", "line": 14.5}]))
    assert q.mode == "split" and q.projection is None
    assert [f.type for f in q.filters] == ["opponent_team"]
    [note] = [x for x in a if x.field == "mode"]
    assert "2000-01" in note.message and "points 14.5" in note.message
    q, _, _ = build(index, QueryDraft(subjects=["Tatum"], stats=["points"], mode="projection"))
    assert q.mode == "projection"


def test_home_away_contradicting_venue_is_flagged(index: EntityIndex) -> None:
    _, _, u = build(index, QueryDraft(subjects=["Curry"], stats=["points"], mode="projection",
                                      venue="San Francisco", home_away="away"))
    assert [(x.kind, x.field) for x in u] == [("conflict", "home_away")]


# ------------------------------------------------------------ real processed data


REQUIRED = [
    ("what is Steph Curry's ppg against LeBron as a defender", "player", [201939]),
    ("what is Steph Curry's apg against the spurs", "player", [201939]),
    ("what is the avg ppg for the Boston Celtics in the last 5 years playing against the Lakers", "team", [1610612738]),
    ("how might Steph Curry perform against a 6'7 defender", "player", [201939]),
    ("Curry is playing against Philly in San Francisco, his defender is LeBron. What is the expected points and "
     "assists for this game? What are the chances he exceeds 27.5 points and 5.5 assists?", "player", [201939]),
    ("how does day of week affect Jayson Tatum's rebounds", "player", [1628369]),
    ("how do back to backs affect scoring for all guards", "player", []),
    ("Knicks points at home vs away on Sundays since 2020", "team", [1610612752]),
]


@pytest.mark.parametrize("prompt,kind,ids", REQUIRED)
def test_required_examples_on_processed_data(prompt: str, kind: str, ids: list[int]) -> None:
    processed = DATA_DIR / "processed"
    if not (processed / "players.parquet").exists():
        pytest.skip("processed tables not built")
    from nbalab.nlp.parser import PromptParser

    result = PromptParser(index=EntityIndex.from_processed(processed)).parse(prompt, backend="rules", today=TODAY)
    assert result.query is not None
    assert (result.query.subject_type, result.query.subject_ids) == (kind, ids)

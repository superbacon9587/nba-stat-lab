"""Example prompts and the exact StatQuery each must parse to (rule-based backend).

The first eight are the required examples. All run on the in-memory fixture
index with today fixed at 2026-09-29, so "last 5 years" is 2021-22..2025-26
and the current season is 2025-26.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from nbalab.nlp.parser import PromptParser
from nbalab.query.schema import StatQuery
from tests.nlp_fixture import fixture_index

TODAY = date(2026, 9, 29)
DEFAULT_LINES = {"points": 24.5, "rebounds": 4.5, "assists": 6.5}

# ids
CURRY, LEBRON, TATUM, JOKIC, AD, EMBIID, GIANNIS = 201939, 2544, 1628369, 203999, 203076, 203954, 203507
BRUNSON, WEMBY, KLAY, DURANT, HARDEN, DRAYMOND, LUKA = 1628973, 1641705, 202691, 201142, 201935, 203110, 1629029
JAYLEN, MIKE_JAMES_OLD, GLEN_RICE_JR = 1627759, 2229, 203318
BOS, DAL, GSW, LAC, LAL, NYK, NOP, PHI, SAS = (
    1610612738, 1610612742, 1610612744, 1610612746, 1610612747, 1610612752, 1610612740, 1610612755, 1610612759,
)


def stub_lines(subject_type: str, subject_id: int, stat: str, today: date) -> float | None:
    return DEFAULT_LINES.get(stat)


@pytest.fixture(scope="module")
def parser() -> PromptParser:
    return PromptParser(index=fixture_index(), line_provider=stub_lines)


def canon(q: StatQuery) -> dict[str, Any]:
    """Query as a dict with filters in a stable order (filter order carries no meaning)."""
    d = q.model_dump()
    d["filters"] = sorted(d["filters"], key=lambda f: json.dumps(f, sort_keys=True))
    return d


def player(ids: list[int], stats: list[str], filters: list[dict] | None = None, **kw: Any) -> dict[str, Any]:
    return {"subject_type": "player", "subject_ids": ids, "stats": stats, "filters": filters or [], **kw}


def team(ids: list[int], stats: list[str], filters: list[dict] | None = None, **kw: Any) -> dict[str, Any]:
    return {"subject_type": "team", "subject_ids": ids, "stats": stats, "filters": filters or [], **kw}


def proj(lines: dict[str, float], **context: Any) -> dict[str, Any]:
    return {"lines": lines, "context": context}


def seasons(start: int, end: int) -> dict[str, Any]:
    return {"type": "season_range", "start": start, "end": end}


# (prompt, expected StatQuery kwargs, expected unresolved as {(kind, field)})
CASES: list[tuple[str, dict[str, Any], set[tuple[str, str]]]] = [
    # ---- the eight required examples
    ("what is Steph Curry's ppg against LeBron as a defender",
     player([CURRY], ["points"], [{"type": "defender_player", "person_id": LEBRON}]), set()),
    ("what is Steph Curry's apg against the spurs",
     player([CURRY], ["assists"], [{"type": "opponent_team", "team_ids": [SAS]}]), set()),
    ("what is the avg ppg for the Boston Celtics in the last 5 years playing against the Lakers",
     team([BOS], ["team_score"], [{"type": "opponent_team", "team_ids": [LAL]}, seasons(2021, 2025)]), set()),
    ("how might Steph Curry perform against a 6'7 defender",
     player([CURRY], ["points", "rebounds", "assists"],
            [{"type": "defender_height_range", "min_inches": 78, "max_inches": 80}],
            mode="projection", projection=proj(DEFAULT_LINES)), set()),
    ("Curry is playing against Philly in San Francisco, his defender is LeBron. What is the expected points and "
     "assists for this game? What are the chances he exceeds 27.5 points and 5.5 assists?",
     player([CURRY], ["points", "assists"], mode="projection",
            projection=proj({"points": 27.5, "assists": 5.5}, opponent_team_id=PHI, venue_team_id=GSW,
                            defender_person_id=LEBRON, home_away="home")),
     {("conflict", "defender")}),
    ("how does day of week affect Jayson Tatum's rebounds",
     player([TATUM], ["rebounds"], group_by="day_of_week"), set()),
    ("how do back to backs affect scoring for all guards",
     player([], ["points"], [{"type": "player_position", "positions": ["G"]}],
            mode="variable_effect", effect_variable="back_to_back"), set()),
    ("Knicks points at home vs away on Sundays since 2020",
     team([NYK], ["team_score"], [{"type": "day_of_week", "days": ["Sunday"]}, seasons(2020, 2025)],
          group_by="home_away"), set()),
    # ---- filters
    ("Jokic rebounds in the playoffs since 2019",
     player([JOKIC], ["rebounds"], [{"type": "playoffs", "value": True}, seasons(2019, 2025)]), set()),
    ("Anthony Davis points without LeBron James",
     player([AD], ["points"], [{"type": "teammate", "person_id": LEBRON, "status": "out"}]), set()),
    ("Tatum points with Jaylen Brown",
     player([TATUM], ["points"], [{"type": "teammate", "person_id": JAYLEN, "status": "in"}]), set()),
    ("Embiid points against centers",
     player([EMBIID], ["points"], [{"type": "defender_position", "positions": ["C"]}]), set()),
    ("Giannis points against defenders taller than 6'10",
     player([GIANNIS], ["points"], [{"type": "defender_height_range", "min_inches": 82, "max_inches": 96}]), set()),
    ("Jalen Brunson points on the second night of a back-to-back",
     player([BRUNSON], ["points"], [{"type": "back_to_back", "value": True}]), set()),
    ("Wembanyama blocks last 10 games",
     player([WEMBY], ["blocks"], [{"type": "last_n_games", "n": 10}]), set()),
    ("Klay Thompson threes this season",
     player([KLAY], ["threes"], [seasons(2025, 2025)]), set()),
    ("Durant points last season",
     player([DURANT], ["points"], [seasons(2025, 2025)]), set()),
    ("Harden assists in 2019-20",
     player([HARDEN], ["assists"], [seasons(2019, 2019)]), set()),
    ("LeBron points on Mondays in March",
     player([LEBRON], ["points"], [{"type": "day_of_week", "days": ["Monday"]}, {"type": "month", "months": [3]}]),
     set()),
    ("Curry 3P% at home",
     player([CURRY], ["three_pct"], [{"type": "home_away", "value": "home"}]), set()),
    ("Lakers points allowed on the road",
     team([LAL], ["opponent_score"], [{"type": "home_away", "value": "away"}, {"type": "season_range", "start": 2023, "end": 2025}]), set()),
    # ---- grouping and effects
    ("Celtics pace by month", team([BOS], ["pace"], [{"type": "season_range", "start": 2023, "end": 2025}], group_by="month"), set()),
    ("how does rest affect Nikola Jokic assists", player([JOKIC], ["assists"], group_by="rest_days"), set()),
    ("how do back to backs affect scoring",
     player([], ["points"], mode="variable_effect", effect_variable="back_to_back"), set()),
    # ---- projections
    ("Will Tatum score 30+ points against the Knicks at MSG?",
     player([TATUM], ["points"], mode="projection",
            projection=proj({"points": 29.5}, opponent_team_id=NYK, venue_team_id=NYK, home_away="away")), set()),
    ("Doncic points vs Dallas in Dallas next game",
     player([LUKA], ["points"], mode="projection",
            projection=proj({"points": 24.5}, opponent_team_id=DAL, venue_team_id=DAL, home_away="away")), set()),
    ("Will Curry score over 25.5 points against the Warriors?",
     player([CURRY], ["points"], mode="projection", projection=proj({"points": 25.5}, opponent_team_id=GSW)),
     {("conflict", "opponent")}),
    # ---- names: collisions, eras, misspellings, team hints
    ("Mike James points", player([MIKE_JAMES_OLD], ["points"]), {("ambiguous", "subject")}),
    ("Glen Rice points in 2014", player([GLEN_RICE_JR], ["points"], [seasons(2013, 2013)]), set()),
    ("Stef Curry points vs Philly",
     player([CURRY], ["points"], [{"type": "opponent_team", "team_ids": [PHI]}]), set()),
    ("Lebrom James points against the Warriors",
     player([LEBRON], ["points"], [{"type": "opponent_team", "team_ids": [GSW]}]), set()),
    ("Hornets points in 2005", team([NOP], ["team_score"], [seasons(2004, 2004)]), set()),
    ("Tatum points vs LA",
     player([TATUM], ["points"], [{"type": "opponent_team", "team_ids": [LAC]}]), {("ambiguous", "opponent")}),
    ("Warriors Curry points", player([CURRY], ["points"]), set()),
    ("Draymond Green rebounds vs Boston with Curry out",
     player([DRAYMOND], ["rebounds"], [{"type": "opponent_team", "team_ids": [BOS]},
                                      {"type": "teammate", "person_id": CURRY, "status": "out"}]), set()),
]


def test_at_least_25_cases() -> None:
    assert len(CASES) >= 25


@pytest.mark.parametrize("prompt,expected,unresolved", CASES, ids=[c[0][:50] for c in CASES])
def test_prompt_parses_to_expected_query(
    parser: PromptParser, prompt: str, expected: dict[str, Any], unresolved: set[tuple[str, str]]
) -> None:
    result = parser.parse(prompt, backend="rules", today=TODAY)
    assert result.query is not None, [u.message for u in result.unresolved]
    assert canon(result.query) == canon(StatQuery.model_validate(expected))
    assert {(u.kind, u.field) for u in result.unresolved} == unresolved


def test_example_5_flags_lebron_and_explains(parser: PromptParser) -> None:
    prompt = CASES[4][0]
    query, assumptions, unresolved = parser.parse(prompt, backend="rules", today=TODAY)
    [conflict] = unresolved
    assert conflict.chosen_id == LEBRON
    assert "Los Angeles Lakers" in conflict.message and "Philadelphia 76ers" in conflict.message
    fields = {a.field for a in assumptions}
    assert {"subject", "opponent", "home_away"} <= fields  # Curry -> Stephen, Philly -> 76ers, SF -> home
    subject = next(a for a in assumptions if a.field == "subject")
    assert "Stephen Curry" in subject.message and "Seth Curry" in subject.message


def test_example_4_lists_default_line_assumptions(parser: PromptParser) -> None:
    _, assumptions, _ = parser.parse(CASES[3][0], backend="rules", today=TODAY)
    fields = {a.field for a in assumptions}
    assert {"stats", "lines.points", "lines.rebounds", "lines.assists", "defender_height"} <= fields


def test_example_3_explains_last_5_years(parser: PromptParser) -> None:
    _, assumptions, _ = parser.parse(CASES[2][0], backend="rules", today=TODAY)
    [seasons_note] = [a for a in assumptions if a.field == "seasons"]
    assert seasons_note.value == [2021, 2025] and "2021-22" in seasons_note.message


def test_ambiguous_name_offers_candidates(parser: PromptParser) -> None:
    _, _, [item] = parser.parse("Mike James points", backend="rules", today=TODAY)
    assert {c.id for c in item.candidates} == {2229, 1628455}
    assert item.chosen_id == 2229


def test_unknown_name_leaves_no_query(parser: PromptParser) -> None:
    query, _, unresolved = parser.parse("average points", backend="rules", today=TODAY)
    assert query is None
    assert [u.field for u in unresolved] == ["subject"]


@pytest.mark.parametrize("prompt, stat, line", [
    ("Victor Wembanyama blocks vs the Warriors, line 3.5", "blocks", 3.5),
    ("Lakers team threes at home, line 13", "threes", 13.0),
    ("Curry points o/u 26.5 vs the Lakers", "points", 26.5),
    ("Jokic line of 12 rebounds", "rebounds", 12.0),
])
def test_bare_line_makes_a_projection(parser: PromptParser, prompt: str, stat: str, line: float) -> None:
    q = parser.parse(prompt, backend="rules", today=TODAY).query
    assert q is not None and q.mode == "projection"
    assert q.projection.lines[stat] == line


def test_team_questions_default_to_last_three_seasons(parser: PromptParser) -> None:
    res = parser.parse("Celtics points per game", backend="rules", today=TODAY)
    assert [f.model_dump() for f in res.query.filters] == [{"type": "season_range", "start": 2023, "end": 2025}]
    assert any("last 3 seasons" in a.message for a in res.assumptions)
    explicit = parser.parse("Celtics points per game since 2015", backend="rules", today=TODAY).query
    assert explicit.filters[0].start == 2015


# ---------------------------------------------------------------- period comparisons


@pytest.mark.parametrize("prompt, kind, subject_type, subjects, stats, split", [
    ("before vs after the all star break", "team", "team", [], ["team_score"], {}),
    ("Celtics after the all star break", "team", "team", [BOS], ["team_score"], {}),
    ("how do teams change after the all star break", "team", "team", [], ["team_score"], {}),
    ("Tatum's scoring leading up to the playoffs", "player", "player", [TATUM], ["points"],
     {"kind": "last_n_before_playoffs"}),
    ("compare team ppg from months 10,11,12,1 to months 2,3,4,5,6", "team", "team", [], ["team_score"],
     {"kind": "month_groups", "before_months": [10, 11, 12, 1], "after_months": [2, 3, 4, 5, 6]}),
    ("who improves most after the break", "player", "player", [], ["points"], {}),
    ("Lakers net rating last 20 games before playoffs", "team", "team", [LAL], ["net_rating"],
     {"kind": "last_n_before_playoffs"}),
    ("Curry points after Feb 20", "player", "player", [CURRY], ["points"], {"kind": "custom_date", "date": "02-20"}),
])
def test_period_phrasings(parser: PromptParser, prompt: str, kind: str, subject_type: str, subjects: list[int],
                          stats: list[str], split: dict) -> None:
    res = parser.parse(prompt, backend="rules", today=TODAY)
    q = res.query
    assert q is not None, [u.message for u in res.unresolved]
    assert q.mode == "period" and q.subject_type == subject_type and q.subject_ids == subjects and q.stats == stats
    assert q.period_split.model_dump(exclude_defaults=True) == split
    assert all(f.type in ("season_range", "last_n_seasons") for f in q.filters)  # only the season scope applies


def test_period_bug_prompt_is_a_valid_team_query(parser: PromptParser) -> None:
    """The prompt from the bug report: team stat, no team named -> every team, month groups."""
    q = parser.parse("compare team ppg from months 10,11,12,1 to months 2,3,4,5,6", backend="rules", today=TODAY).query
    assert q.subject_type == "team" and q.stats == ["team_score"]
    assert StatQuery.model_validate_json(q.model_dump_json()) == q


def test_which_players_is_a_player_ranking(parser: PromptParser) -> None:
    q = parser.parse("which players get better after the all star break in rebounds", backend="rules", today=TODAY).query
    assert q.mode == "period" and q.subject_type == "player" and q.subject_ids == [] and q.stats == ["rebounds"]

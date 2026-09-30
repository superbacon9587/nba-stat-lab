"""CLI parsing helpers, plus the five requested examples on the real processed data (skipped if not built)."""

from __future__ import annotations

import pytest

from nbalab.query.cli import build_parser, main, query_from_args
from nbalab.query.data import PROCESSED_DIR, QueryData
from nbalab.query.resolve import parse_height, parse_height_range, resolve_player


@pytest.mark.parametrize("text, inches", [("6'7", 79), ("6-7", 79), ("6 7", 79), ("6 ft 7", 79), ("79", 79),
                                          ("6'7\"", 79), ("79in", 79)])
def test_parse_height(text: str, inches: float) -> None:
    assert parse_height(text) == inches


def test_parse_height_range() -> None:
    assert parse_height_range("6-6:6-8") == (78, 80)
    assert parse_height_range("6'8\" to 6'6\"") == (78, 80)
    assert parse_height_range("6'7") == (78, 80)


def test_query_from_args_on_fixture(data: QueryData) -> None:
    args = build_parser().parse_args(
        ["--player", "Player 10", "--stat", "PRA", "--opponent", "bees", "--defender-height", "6-5:6-7",
         "--last-n-seasons", "2", "--line", "pts=25.5"]
    )
    q, _ = query_from_args(args, data)
    assert q.subject_ids == [10] and q.stats == ["pra"]
    types = [f.type for f in q.filters]
    assert types == ["opponent_team", "defender_height_range", "last_n_seasons"]
    assert q.filters[0].team_ids == [2]
    assert q.projection.lines == {"points": 25.5}


def test_resolver_prefers_exact_then_most_games(data: QueryData) -> None:
    assert resolve_player(data.players, "Player 20").id == 20
    with pytest.raises(LookupError):
        resolve_player(data.players, "nobody at all")


needs_data = pytest.mark.skipif(not (PROCESSED_DIR / "player_games.parquet").exists(),
                                reason="processed data not built")

EXAMPLES = [
    ["--player", "Stephen Curry", "--stat", "assists", "--opponent", "San Antonio Spurs"],
    ["--team", "Boston Celtics", "--stat", "points", "--opponent", "Lakers", "--last-n-seasons", "5"],
    ["--player", "Stephen Curry", "--stat", "points", "--defender", "LeBron James"],
    ["--player", "Stephen Curry", "--stat", "points", "--defender-height", "6-6:6-8"],
    ["--player", "Stephen Curry", "--stat", "points", "--group-by", "day_of_week"],
]


@needs_data
@pytest.mark.parametrize("argv", EXAMPLES)
def test_requested_examples_run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "baseline" in out or "overall" in out

"""Stat definitions, aliases, buckets, and StatQuery validation."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from pydantic import ValidationError

from nbalab.query.schema import (
    DefenderPlayerFilter,
    HomeAwayFilter,
    OpponentTeamFilter,
    StatQuery,
)
from nbalab.query.stats import (
    PLAYER_STATS,
    TEAM_STATS,
    canonical_stat,
    height_bucket,
    rest_bucket,
    weight_bucket,
)

GAMES = pd.DataFrame({
    "points": [30.0, 10.0], "reboundsTotal": [5.0, 5.0], "assists": [5.0, 3.0], "minutes": [36.0, 18.0],
    "fieldGoalsMade": [10.0, 0.0], "fieldGoalsAttempted": [20.0, 0.0],
    "freeThrowsAttempted": [10.0, 0.0], "threePointersMade": [2.0, 0.0],
})


def test_pra_is_the_sum() -> None:
    assert PLAYER_STATS["pra"].per_game(GAMES).tolist() == [40.0, 18.0]


def test_ratio_per_game_is_nan_without_attempts_and_pooled_uses_totals() -> None:
    fg = PLAYER_STATS["fg_pct"]
    assert fg.per_game(GAMES).iloc[0] == pytest.approx(50.0)
    assert math.isnan(fg.per_game(GAMES).iloc[1])
    assert fg.pooled(GAMES) == pytest.approx(10 / 20 * 100)


def test_true_shooting_hand_value() -> None:
    # TS = 30 / (2 * (20 + 0.44*10)) = 30 / 48.8
    assert PLAYER_STATS["ts_pct"].per_game(GAMES).iloc[0] == pytest.approx(30 / 48.8 * 100)
    # pooled over both games: 40 / 48.8
    assert PLAYER_STATS["ts_pct"].pooled(GAMES) == pytest.approx(40 / 48.8 * 100)


def test_efg_counts_threes_extra() -> None:
    assert PLAYER_STATS["efg_pct"].per_game(GAMES).iloc[0] == pytest.approx((10 + 1) / 20 * 100)


def test_team_margin_and_total() -> None:
    tg = pd.DataFrame({"teamScore": [110.0], "opponentScore": [100.0]})
    assert TEAM_STATS["margin"].per_game(tg).iloc[0] == 10
    assert TEAM_STATS["total_points"].per_game(tg).iloc[0] == 210


@pytest.mark.parametrize("raw, kind, expected", [
    ("PRA", "player", "pra"), ("FG%", "player", "fg_pct"), ("TS%", "player", "ts_pct"),
    ("ppg", "player", "points"), ("apg", "player", "assists"), ("3PM", "player", "threes"),
    ("Rebounds", "player", "rebounds"), ("points", "team", "team_score"), ("ppg", "team", "team_score"),
    ("team score", "team", "team_score"),
])
def test_canonical_stat_aliases(raw: str, kind: str, expected: str) -> None:
    assert canonical_stat(raw, kind) == expected


def test_canonical_stat_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        canonical_stat("dunks")


def test_buckets() -> None:
    assert rest_bucket(pd.Series([0.0, 1.0, 5.0, None])).tolist() == ["0", "1", "3+", None]
    assert height_bucket(pd.Series([78.0, 79.0, 80.0])).tolist() == ["6'6\"-6'7\"", "6'6\"-6'7\"", "6'8\"-6'9\""]
    assert weight_bucket(pd.Series([205.0])).tolist() == ["200-219 lb"]


def test_query_parses_discriminated_filters_from_json() -> None:
    q = StatQuery.model_validate({
        "subject_ids": [1], "stats": ["PRA"],
        "filters": [{"type": "opponent_team", "team_ids": [5]}, {"type": "home_away", "value": "home"}],
    })
    assert q.stats == ["pra"]
    assert isinstance(q.filters[0], OpponentTeamFilter) and isinstance(q.filters[1], HomeAwayFilter)


def test_scope_vs_split_filters() -> None:
    q = StatQuery(subject_ids=[1], stats=["points"],
                  filters=[{"type": "last_n_seasons", "n": 5}, {"type": "home_away", "value": "away"}])
    assert [f.type for f in q.scope_filters] == ["last_n_seasons"]
    assert [f.type for f in q.split_filters] == ["home_away"]


@pytest.mark.parametrize("bad", [
    {"subject_ids": [1], "stats": ["dunks"]},
    {"subject_ids": [], "stats": ["points"]},
    {"subject_type": "team", "subject_ids": [1], "stats": ["points"],
     "filters": [{"type": "defender_player", "person_id": 2}]},
    {"subject_ids": [1], "stats": ["points"], "group_by": "zodiac_sign"},
    {"subject_ids": [1], "stats": ["points"], "filters": [{"type": "defender_height_range", "min_inches": 80, "max_inches": 70}]},
    {"mode": "variable_effect", "stats": ["points"]},
    {"mode": "projection", "subject_ids": [1], "stats": ["points"]},
])
def test_invalid_queries_are_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        StatQuery.model_validate(bad)


def test_projection_context_becomes_filters() -> None:
    q = StatQuery(subject_ids=[1], stats=["points"], mode="projection",
                  projection={"lines": {"pts": 27.5},
                              "context": {"opponent_team_id": 9, "home_away": "home", "defender_person_id": 3}})
    assert q.projection.lines == {"points": 27.5}
    types = {f.type for f in q.with_projection_context().filters}
    assert types == {"opponent_team", "home_away", "defender_player"}
    assert any(isinstance(f, DefenderPlayerFilter) for f in q.with_projection_context().filters)

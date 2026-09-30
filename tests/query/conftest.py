"""Fixtures for the query engine tests. The league itself is in query_fixture_league.py."""

from __future__ import annotations

import pytest
from query_fixture_league import build_data

from nbalab.query.data import QueryData


@pytest.fixture
def data() -> QueryData:
    return build_data()


@pytest.fixture
def box_only(data: QueryData) -> QueryData:
    """Same league without lineup or matchup tables."""
    return QueryData(data.player_games, data.team_games, data.players, data.teams)

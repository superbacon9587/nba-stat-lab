"""Tests for rolling form, rest, calendar and leakage-free opponent ratings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nbalab.data.features import (
    add_lagged_rolling_means,
    calendar_features,
    cumulative_team_ratings,
    lagged_rolling_mean,
    pregame_team_ratings,
    rest_features,
)
from tests.conftest import load_processed


def two_player_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "personId": [1, 1, 1, 1, 2, 2],
            "points": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0],
        }
    )


def test_lagged_rolling_mean_excludes_current_game() -> None:
    out = lagged_rolling_mean(two_player_frame(), "personId", "points", 2)
    expected = [np.nan, 10.0, 15.0, 25.0, np.nan, 100.0]
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_lagged_rolling_mean_does_not_leak_across_players() -> None:
    out = lagged_rolling_mean(two_player_frame(), "personId", "points", 5)
    assert np.isnan(out.iloc[4])  # player 2's first game ignores player 1's history


def test_changing_current_game_does_not_change_its_feature() -> None:
    df = two_player_frame()
    before = lagged_rolling_mean(df, "personId", "points", 3)
    df.loc[2, "points"] = 9999.0
    after = lagged_rolling_mean(df, "personId", "points", 3)
    assert before.iloc[2] == after.iloc[2]
    assert before.iloc[3] != after.iloc[3]  # but it does feed the *next* game


def test_add_lagged_rolling_means_names_columns() -> None:
    out = add_lagged_rolling_means(two_player_frame(), "personId", ("points",), (5, 10))
    assert {"points_last5", "points_last10"} <= set(out.columns)


def test_rest_features_back_to_back() -> None:
    out = rest_features(pd.Series([np.nan, 1.0, 2.0, 4.0]), "team_")
    assert out["team_rest_days"].tolist()[1:] == [0.0, 1.0, 3.0]
    assert out["team_is_back_to_back"].tolist() == [False, True, False, False]


def test_calendar_features_week_of_season() -> None:
    dates = pd.Series(pd.to_datetime(["2024-10-22", "2024-10-28", "2024-10-29", "2025-01-06"]))
    opener = pd.Series(pd.to_datetime(["2024-10-22"] * 4))
    cal = calendar_features(dates, opener)
    assert cal["week_of_season"].tolist() == [1, 1, 2, 11]
    assert cal["day_of_week"].tolist()[0] == "Tuesday"
    assert cal["month"].tolist() == [10, 10, 10, 1]


def ratings_frame() -> pd.DataFrame:
    """Team 1 plays two games in 2023 and three in 2024; team 2 only appears in 2024."""
    return pd.DataFrame(
        {
            "gameId": [1, 2, 3, 4, 5, 6],
            "teamId": [1, 1, 1, 1, 1, 2],
            "season": [2023, 2023, 2024, 2024, 2024, 2024],
            "gameDateTimeEst": pd.to_datetime(
                ["2023-11-01", "2023-11-03", "2024-10-25", "2024-10-27", "2024-10-29", "2024-10-25"]
            ),
            "teamScore": [100.0, 120.0, 110.0, 90.0, 130.0, 100.0],
            "opponentScore": [90.0, 110.0, 100.0, 120.0, 99.0, 100.0],
            "possessions": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "pace": [98.0, 102.0, 100.0, 96.0, 104.0, 100.0],
        }
    )


def test_pregame_ratings_use_only_earlier_games() -> None:
    df = ratings_frame()
    pre = pregame_team_ratings(df, cumulative_team_ratings(df)).set_index("gameId")
    # Game 4 sees only game 3: 100 x 100 points allowed / 100 possessions.
    assert pre.loc[4, "std_def_rating"] == pytest.approx(100.0)
    # Game 5 sees games 3-4: (100 + 120) / 200 possessions.
    assert pre.loc[5, "std_def_rating"] == pytest.approx(110.0)
    assert pre.loc[5, "std_pace"] == pytest.approx(98.0)
    assert pre.loc[5, "std_games"] == 2


def test_pregame_ratings_first_game_falls_back_to_previous_season() -> None:
    df = ratings_frame()
    pre = pregame_team_ratings(df, cumulative_team_ratings(df)).set_index("gameId")
    # Game 3 is team 1's 2024 opener: use full 2023 = (90 + 110) / 200.
    assert pre.loc[3, "std_def_rating"] == pytest.approx(100.0)
    assert pre.loc[3, "std_games"] == 0
    # Team 2 has no history at all.
    assert np.isnan(pre.loc[6, "std_def_rating"])
    assert np.isnan(pre.loc[1, "std_def_rating"])


def test_pregame_ratings_ignore_current_game_result() -> None:
    df = ratings_frame()
    before = pregame_team_ratings(df, cumulative_team_ratings(df)).set_index("gameId")
    df.loc[df["gameId"] == 5, "opponentScore"] = 500.0
    after = pregame_team_ratings(df, cumulative_team_ratings(df)).set_index("gameId")
    assert before.loc[5, "std_def_rating"] == after.loc[5, "std_def_rating"]


def recompute_prior_mean(games: pd.DataFrame, stat: str, window: int) -> pd.Series:
    """Independent reference: plain Python loop over each player's earlier games."""
    out = []
    for _, g in games.groupby("personId", sort=False):
        vals = g[stat].tolist()
        out += [np.mean(vals[max(0, i - window):i]) if i else np.nan for i in range(len(vals))]
    return pd.Series(out, index=games.index)


@pytest.mark.parametrize("window", [5, 10, 20])
def test_sample_rolling_features_match_reference_and_skip_current_game(
    sample_tables: dict[str, pd.DataFrame], window: int
) -> None:
    pg = sample_tables["player_games"].sort_values(["personId", "gameDateTimeEst"])
    for stat in ("points", "assists", "reboundsTotal", "minutes"):
        ref = recompute_prior_mean(pg, stat, window)
        np.testing.assert_allclose(pg[f"{stat}_last{window}"], ref, rtol=1e-5)


def test_processed_rolling_features_never_include_current_game() -> None:
    pg = load_processed("player_games")
    rng = np.random.default_rng(0)
    ids = rng.choice(pg["personId"].unique(), size=40, replace=False)
    sub = pg[pg["personId"].isin(ids)].sort_values(["personId", "gameDateTimeEst", "gameId"])
    for window in (5, 10, 20):
        for stat in ("points", "minutes"):
            ref = recompute_prior_mean(sub, stat, window)
            np.testing.assert_allclose(sub[f"{stat}_last{window}"], ref, rtol=1e-4)
    first_games = sub.groupby("personId").head(1)
    assert first_games["points_last5"].isna().all()

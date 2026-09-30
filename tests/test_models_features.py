"""Model features: lags, positional defense, and no leakage from the game being predicted."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nbalab.models import features as F


def test_lagged_ewm_excludes_current_and_skips_missing() -> None:
    df = pd.DataFrame({"p": [1, 1, 1, 1], "x": [10.0, 20.0, np.nan, 40.0]})
    out = F.lagged_ewm_mean(df, "p", ["x"], halflife=1)["x"]
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 10.0
    # game 3 sees games 1-2: weights 0.5 (older) and 1.0 (newer)
    assert out.iloc[2] == pytest.approx((0.5 * 10 + 20) / 1.5)
    # game 3's missing value is skipped, not treated as zero
    assert out.iloc[3] == pytest.approx(out.iloc[2])


def test_effective_sample_size() -> None:
    assert F.ewm_effective_n(1, 10) == pytest.approx(1.0)
    assert F.ewm_effective_n(1000, 10) < 40  # recency weighting caps the information
    assert F.ewm_effective_n(1000, 10) > F.ewm_effective_n(1000, 5)


def _toy_games() -> pd.DataFrame:
    """Two defenses, guards scoring against them over three dates."""
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    rows = []
    for d, pts_vs_a, pts_vs_b in zip(dates, [30.0, 30.0, 30.0], [10.0, 10.0, 10.0]):
        rows.append({"opponentTeamId": 1, "season": 2019, "gameDateTimeEst": d, "position_group": "G",
                     "minutes": 100.0, "points": pts_vs_a})
        rows.append({"opponentTeamId": 2, "season": 2019, "gameDateTimeEst": d, "position_group": "G",
                     "minutes": 100.0, "points": pts_vs_b})
    return pd.DataFrame(rows)


def test_positional_defense_ranks_bad_defense_higher_and_is_lagged() -> None:
    games = _toy_games()
    cum = F.positional_defense_cumulative(F.positional_defense_totals(games, ("points",)), ("points",))
    keys = games[["opponentTeamId", "season", "position_group", "gameDateTimeEst"]]
    out = F.positional_defense_asof(keys, cum, ("points",))["points_opp_pos"]
    first_day = games["gameDateTimeEst"].eq(games["gameDateTimeEst"].min()).to_numpy()
    assert (out[first_day] == 1.0).all()  # no earlier games -> league average
    later = ~first_day
    vs_a = out[later & games["opponentTeamId"].eq(1).to_numpy()]
    vs_b = out[later & games["opponentTeamId"].eq(2).to_numpy()]
    assert (vs_a > 1).all() and (vs_b < 1).all()
    # league rate 0.2/min; defense A allowed 0.3/min over 100 min, shrunk with 500 pseudo-minutes
    expected = ((30 + F.POS_PSEUDO_MINUTES * 0.2) / (100 + F.POS_PSEUDO_MINUTES)) / 0.2
    assert vs_a.iloc[0] == pytest.approx(expected)


def test_player_frame_has_no_leakage(sample_tables: dict[str, pd.DataFrame]) -> None:
    """Changing one game's box score must not change any feature of that game."""
    pg, tg = sample_tables["player_games"], sample_tables["team_games"]
    before = F.build_player_frame(pg, tg, 2007)
    target = before.iloc[len(before) // 2]
    edited = pg.copy()
    hit = edited["personId"].eq(target["personId"]) & edited["gameId"].eq(target["gameId"])
    for c in ("points", "assists", "reboundsTotal", "minutes", "threePointersMade", "steals"):
        edited.loc[hit, c] = edited.loc[hit, c] * 3 + 7
    after = F.build_player_frame(edited, tg, 2007)
    key = ["personId", "gameId"]
    b = before.set_index(key).loc[[(target["personId"], target["gameId"])], F.player_feature_names()]
    a = after.set_index(key).loc[[(target["personId"], target["gameId"])], F.player_feature_names()]
    pd.testing.assert_frame_equal(a, b)


def test_player_frame_features_exist(sample_tables: dict[str, pd.DataFrame]) -> None:
    frame = F.build_player_frame(sample_tables["player_games"], sample_tables["team_games"], 2007)
    missing = set(F.player_feature_names()) - set(frame.columns)
    assert not missing
    assert frame["minutes"].gt(0).all()


def test_future_row_features_match_a_played_row(sample_tables: dict[str, pd.DataFrame]) -> None:
    """A placeholder game with unknown (NaN) stats must get exactly the lagged features
    it would get if its stats were known: train and serve must see the same inputs."""
    pg, tg = sample_tables["player_games"], sample_tables["team_games"]
    base = F.prepare_player_games(pg, 2007)
    cum = F.positional_defense_cumulative(F.positional_defense_totals(base))
    sizes = F.opponent_starter_size(base)
    pid = base["personId"].value_counts().idxmax()
    hist = base[base["personId"].eq(pid)]
    nxt = hist.iloc[[-1]].assign(gameId=-1, gameDateTimeEst=hist["gameDateTimeEst"].max() + pd.Timedelta(days=2))
    known = F.player_frame_from(pd.concat([hist, nxt], ignore_index=True), cum, sizes, tg).iloc[-1]
    unknown_row = nxt.assign(**{c: np.nan for c in ("minutes", *F.PLAYER_COUNT_COLS)})
    unknown = F.player_frame_from(pd.concat([hist, unknown_row], ignore_index=True), cum, sizes, tg).iloc[-1]
    cols = F.player_feature_names()
    pd.testing.assert_series_equal(known[cols].astype(float), unknown[cols].astype(float), check_names=False)
    assert np.isfinite(unknown[["points_rate_std", "minutes_std"]].astype(float)).all()


def test_team_future_row_keeps_season_to_date() -> None:
    t = pd.DataFrame({"teamId": [1, 1, 1], "season": [2020] * 3, "teamScore": [100.0, 110.0, np.nan]})
    sums = F.lagged_group_sum(t, ["teamId", "season"], ["teamScore"])
    assert sums["teamScore"].tolist() == [0.0, 100.0, 210.0]

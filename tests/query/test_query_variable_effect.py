"""The fixed-effects regression recovers effects planted in synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from nbalab.query.data import QueryData
from nbalab.query.schema import StatQuery
from nbalab.query.variable_effect import fit_fixed_effects, run_variable_effect, within

HOME_EFFECT = 3.0


def synthetic_league(seed: int = 7, n_games: int = 60) -> QueryData:
    """Two teams of 10 players play each other ``n_games`` times.

    points = player talent (10..40) + 3 x home + 0.5 x minutes + noise(sd 2).
    Talent differs a lot between players, so a model without fixed effects
    would be dominated by who is playing.
    """
    rng = np.random.default_rng(seed)
    talent = {p: 10 + 3 * (p % 10) for p in range(20)}
    rows = []
    for g in range(n_games):
        date = pd.Timestamp("2024-11-01") + pd.Timedelta(days=g)
        a_home = g % 2 == 0
        for p in range(20):
            team = 1 if p < 10 else 2
            home = int((team == 1) == a_home)
            minutes = float(rng.uniform(15, 38))
            rows.append({
                "personId": p, "gameId": g, "teamId": team, "opponentTeamId": 3 - team,
                "gameDateTimeEst": date, "season": 2024, "game_type": "regular", "home": home,
                "team_is_back_to_back": g % 7 == 1, "team_rest_days": 0.0 if g % 7 == 1 else 1.0,
                "month": date.month, "week_of_season": g // 7 + 1, "day_of_week": date.day_name(),
                "venue_city": "A" if a_home else "B", "opp_pre_def_rating": float(rng.normal(110, 3)),
                "opp_pre_pace": 99.0, "minutes": minutes, "starter": p % 10 < 5,
                "startingPosition": "G" if p % 10 < 5 else None, "age": 25.0,
                "heightInches": 70.0 + p % 10, "bodyWeightLbs": 200.0, "position_group": "G",
                "points": talent[p] + HOME_EFFECT * home + 0.5 * minutes + rng.normal(0, 2),
            })
    pg = pd.DataFrame(rows)
    players = pd.DataFrame({"personId": range(20), "full_name": [f"P{p}" for p in range(20)],
                            "heightInches": [70.0 + p % 10 for p in range(20)],
                            "bodyWeightLbs": 200.0, "position_group": "G"})
    teams = pd.DataFrame({"teamId": [1, 2], "full_name": ["A", "B"]})
    return QueryData(pg, pg.iloc[0:0], players, teams)


def test_recovers_planted_home_effect() -> None:
    data = synthetic_league()
    q = StatQuery(mode="variable_effect", stats=["points"], effect_variable="home_away")
    r = run_variable_effect(q, data)
    t = r.per_stat["points"].coefficients.set_index("term")
    assert t.loc["home_away", "ci_low"] < HOME_EFFECT < t.loc["home_away", "ci_high"]
    assert t.loc["home_away", "coef"] == pytest.approx(HOME_EFFECT, abs=0.3)
    assert t.loc["home_away", "p_value"] < 1e-6
    assert t.loc["minutes", "coef"] == pytest.approx(0.5, abs=0.05)
    assert r.per_stat["points"].n_subjects == 20
    assert t.loc["home_away", "is_variable"] and not t.loc["minutes", "is_variable"]


def test_null_variable_is_not_significant() -> None:
    data = synthetic_league()
    q = StatQuery(mode="variable_effect", stats=["points"], effect_variable="back_to_back")
    t = run_variable_effect(q, data).per_stat["points"].coefficients.set_index("term")
    assert t.loc["back_to_back", "ci_low"] < 0 < t.loc["back_to_back", "ci_high"]


def test_within_transform_equals_player_dummies() -> None:
    """Demeaning by player gives exactly the same slope as one dummy per player."""
    data = synthetic_league(n_games=12)
    pg = data.player_games
    X = pd.DataFrame({"home": pg["home"].astype(float), "minutes": pg["minutes"]})
    table, *_ = fit_fixed_effects(pg["points"], X, pg["personId"])
    dummies = pd.get_dummies(pg["personId"], prefix="p", dtype=float)
    ols = sm.OLS(pg["points"].to_numpy(), pd.concat([X, dummies], axis=1).to_numpy()).fit()
    assert table.set_index("term").loc["home", "coef"] == pytest.approx(ols.params[0], rel=1e-8)
    assert table.set_index("term").loc["minutes", "coef"] == pytest.approx(ols.params[1], rel=1e-8)


def test_within_subtracts_group_means() -> None:
    df = pd.DataFrame({"x": [1.0, 3.0, 10.0, 20.0]})
    out = within(df, pd.Series([1, 1, 2, 2]))
    assert out["x"].tolist() == [-1.0, 1.0, -5.0, 5.0]


def test_categorical_deviation_coding_sums_to_zero() -> None:
    data = synthetic_league()
    q = StatQuery(mode="variable_effect", stats=["points"], effect_variable="venue")
    t = run_variable_effect(q, data).per_stat["points"].coefficients
    venue = t[t["is_variable"]]
    assert set(venue["term"]) == {"venue=A", "venue=B"}
    assert venue["coef"].sum() == pytest.approx(0.0, abs=1e-9)


def test_filters_restrict_population() -> None:
    data = synthetic_league()
    q = StatQuery(mode="variable_effect", stats=["points"], effect_variable="home_away",
                  filters=[{"type": "min_minutes", "minutes": 30}])
    r = run_variable_effect(q, data)
    assert r.per_stat["points"].n_obs < len(data.player_games) / 2


def test_subject_specific_filters_rejected() -> None:
    q = StatQuery(mode="variable_effect", stats=["points"], effect_variable="home_away",
                  filters=[{"type": "defender_player", "person_id": 1}])
    with pytest.raises(ValueError):
        run_variable_effect(q, synthetic_league(n_games=4))

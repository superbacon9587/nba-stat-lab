"""Baseline, rate/minutes offsets, copula, Shapley contributions, trend bands, defender proxy, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from nbalab.models import copula, metrics as M
from nbalab.models.baseline import Baseline, form
from nbalab.models.contributions import all_coalitions, shapley
from nbalab.models.distributions import count_pmf
from nbalab.models.defender import effects_from_sums, on_off_delta, pair_sums
from nbalab.models.minutes import MinutesModel, residual_table
from nbalab.models.rate import GlmRate
from nbalab.models.trend import trend_bands


# ------------------------------------------------------------------ baseline


def _baseline_frame(n_players: int = 60, n_games: int = 80, seed: int = 0) -> pd.DataFrame:
    """Players whose points drop against good defenses (low DRtg) and rise at home."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        level = rng.uniform(8, 25)
        for g in range(n_games):
            drtg = rng.uniform(0.94, 1.06)
            home = float(rng.integers(0, 2))
            mean = level * drtg**3 * np.exp(0.05 * home)
            rows.append({"personId": p, "g": g, "opp_drtg_rel": drtg, "home": home, "points": rng.poisson(mean)})
    return pd.DataFrame(rows).astype({"points": float})


def test_baseline_form_is_lagged() -> None:
    df = pd.DataFrame({"personId": [1, 1, 1], "points": [10.0, 20.0, 30.0]})
    out = form(df, "personId", "points", 5)
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 10.0
    assert 10.0 < out.iloc[2] < 20.0  # never includes the current game (30)


def test_baseline_adjustments_have_the_right_sign() -> None:
    df = _baseline_frame()
    sb = Baseline.fit_stat(df, "personId", "points", 10)
    assert sb.params["log_opp_drtg"] > 1.0  # worse defense (higher DRtg) -> more points
    assert sb.params["home"] > 0
    mean, alpha = Baseline("personId", {"points": sb}).predict(df, "points")
    first_games = df.groupby("personId").cumcount().eq(0).to_numpy()
    assert np.isnan(mean[first_games]).all() and np.isfinite(mean[~first_games]).all()
    assert (alpha >= 0).all()


# ------------------------------------------------------------------ rate and minutes


def _rate_frame(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "points_rate_ewm5": rng.uniform(0.3, 0.9, n), "points_rate_ewm20": rng.uniform(0.3, 0.9, n),
        "points_rate_std": rng.uniform(0.3, 0.9, n), "games_std": 20, "points_opp_pos": rng.uniform(0.9, 1.1, n),
        "opp_drtg_rel": rng.uniform(0.95, 1.05, n), "opp_pace_rel": 1.0, "team_pace_rel": 1.0,
        "home": rng.integers(0, 2, n).astype(float), "team_is_back_to_back": 0.0, "team_rest_days": 1.0,
        "is_playoff": 0.0, "starter_share10": 1.0, "age": 27.0, "pos_code": 0, "minutes": rng.uniform(10, 40, n),
    })
    f["points"] = rng.poisson(f["minutes"] * f["points_rate_ewm20"]).astype(float)
    return f


def test_rate_offset_scales_linearly_with_minutes() -> None:
    f = _rate_frame()
    glm = GlmRate.fit(f, "points", "player")
    rate = glm.predict_rate(f.head(5))
    doubled = f.head(5).assign(minutes=f["minutes"].head(5) * 2)
    assert np.allclose(glm.predict_rate(doubled), rate)  # rate is per minute
    mean_30 = rate * 30.0
    assert np.allclose(rate * 15.0 * 2, mean_30)


def test_minutes_override_is_a_point_mass() -> None:
    model = MinutesModel(booster=None, residuals=np.zeros((8, 25)))  # type: ignore[arg-type]
    nodes, w = model.distribution(pd.DataFrame(index=[0, 1]), override=31.5)
    assert nodes.shape == (2, 1) and (nodes == 31.5).all() and w.tolist() == [1.0]


def test_minutes_residual_table_is_sorted_and_pools_small_buckets() -> None:
    rng = np.random.default_rng(2)
    pred = rng.uniform(20, 36, 5000)
    table = residual_table(pred, pred + rng.normal(0, 4, 5000))
    assert table.shape == (8, 25)
    assert (np.diff(table, axis=1) >= 0).all()
    np.testing.assert_allclose(table[0], table[-1])  # both buckets empty -> pooled


# ------------------------------------------------------------------ copula


def test_copula_recovers_gaussian_correlation() -> None:
    rng = np.random.default_rng(3)
    true = np.array([[1, 0.5, 0.2], [0.5, 1, -0.3], [0.2, -0.3, 1]])
    z = rng.multivariate_normal(np.zeros(3), true, size=20_000)
    u = copula.normal_scores(sps.norm.cdf(z))
    assert np.abs(copula.correlation(u) - true).max() < 0.03


def test_independent_copula_joint_equals_product() -> None:
    rng = np.random.default_rng(4)
    support = np.arange(60)
    tables = {"a": np.cumsum(count_pmf(np.array([20.0]), 0.1, support), axis=1),
              "b": np.cumsum(count_pmf(np.array([6.0]), 0.0, support), axis=1)}
    draws = copula.simulate(np.ones(1), tables, np.eye(2), 200_000, rng)
    pa = (draws["a"] > 20.5).mean()
    pb = (draws["b"] > 5.5).mean()
    joint = copula.joint_probability(draws, {"a": (">", 20.5), "b": (">", 5.5)})
    assert joint == pytest.approx(pa * pb, abs=0.005)


def test_shrunk_correlation_moves_toward_league() -> None:
    own, league = np.array([[1, 0.9], [0.9, 1]]), np.array([[1, 0.1], [0.1, 1]])
    few = copula.shrink_correlation(own, league, 10)[0, 1]
    many = copula.shrink_correlation(own, league, 10_000)[0, 1]
    assert 0.1 < few < many < 0.9 + 1e-9


# ------------------------------------------------------------------ contributions


def test_shapley_sums_to_total_and_ignores_null_group() -> None:
    effects = {"opponent": 1.5, "home": 0.8, "rest": -0.4}

    def value(c: frozenset[str]) -> float:
        v = 20.0 + sum(effects[g] for g in c if g in effects)
        if {"opponent", "rest"} <= c:
            v += 0.6  # an interaction, split between the two groups
        return v

    groups = ["opponent", "home", "rest", "nothing"]
    phi = shapley(groups, value)
    assert sum(phi.values()) == pytest.approx(value(frozenset(groups)) - value(frozenset()))
    assert phi["nothing"] == pytest.approx(0.0)
    assert phi["home"] == pytest.approx(0.8)
    assert phi["opponent"] == pytest.approx(1.5 + 0.3)


def test_all_coalitions_count() -> None:
    assert len(all_coalitions("abcdef")) == 64


# ------------------------------------------------------------------ trend


def test_trend_perfect_line_has_zero_width_bands() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="7D")
    t = np.arange(10) * 7 / 365.25
    bands = trend_bands(pd.Series(dates), pd.Series(5 + 2 * t))
    assert np.allclose(bands["ci_hi"] - bands["ci_lo"], 0, atol=1e-8)


def test_trend_prediction_band_is_wider_than_confidence_band() -> None:
    rng = np.random.default_rng(5)
    dates = pd.Series(pd.date_range("2024-01-01", periods=60, freq="3D"))
    bands = trend_bands(dates, pd.Series(20 + rng.normal(0, 5, 60)), horizon_days=30, n_future=5)
    assert ((bands["pi_hi"] - bands["pi_lo"]) > (bands["ci_hi"] - bands["ci_lo"])).all()
    assert bands["is_future"].sum() == 5
    assert trend_bands(dates.head(2), pd.Series([1.0, 2.0])) is None


# ------------------------------------------------------------------ defender proxy


def _pair_rows(on_rate: float, off_rate: float, games: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "personId": 1, "opponentPersonId": 2, "gameDateTimeEst": pd.date_range("2020-01-01", periods=games),
        "on_min": 20.0, "off_min": 15.0, "on_points": on_rate * 20.0, "off_points": off_rate * 15.0,
    })


def test_on_off_delta_and_shrinkage() -> None:
    sums = pair_sums(_pair_rows(0.5, 1.0), ["points"])
    d, v = on_off_delta(sums, "points")
    assert d[0] == pytest.approx(-0.5)
    assert v[0] == pytest.approx(0.0)  # identical games -> no sampling noise
    no_prior = effects_from_sums(sums, {"points": 0.0})
    assert no_prior["points_mult"].iloc[0] == pytest.approx(1.0)  # tau^2 = 0 -> shrunk all the way
    strong = effects_from_sums(sums, {"points": 1.0})
    share = 20 / 35
    assert strong["points_mult"].iloc[0] == pytest.approx(1 - 0.5 * share)


# ------------------------------------------------------------------ metrics


def test_brier_log_loss_and_coverage_by_hand() -> None:
    p, o = np.array([0.8, 0.3]), np.array([1, 0])
    assert M.brier(p, o) == pytest.approx((0.04 + 0.09) / 2)
    assert M.log_loss(p, o) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    assert M.coverage(np.array([0, 5]), np.array([10, 6]), np.array([10, 7])) == 0.5


def test_calibration_bins() -> None:
    cal = M.calibration_bins(np.array([0.05, 0.08, 0.55, 0.95]), np.array([0, 1, 1, 1]))
    assert cal["n"].tolist() == [2, 1, 1]
    assert cal["bin_actual"].tolist() == [0.5, 1.0, 1.0]


def test_climatology_uses_only_earlier_games() -> None:
    y = np.array([10, 30, 30, 10.0])
    lines = np.full(4, 20.5)
    out = M.climatology(y, lines, np.array(["a"] * 4))
    assert out.tolist() == [0.5, 0.0, 0.5, pytest.approx(2 / 3)]


def test_synthetic_lines() -> None:
    half, whole = M.synthetic_lines(np.array([22.4, 22.4]), np.array([18.0, 18.0]), np.array([10, 2]))
    assert half.tolist() == [22.5, 18.5] and whole.tolist() == [22.0, 18.0]


def test_paired_bootstrap_detects_a_real_gap() -> None:
    rng = np.random.default_rng(6)
    a = rng.uniform(0, 1, 2000)
    diff, lo, hi = M.paired_bootstrap_diff(a, a + 0.2, np.repeat(np.arange(100), 20))
    assert diff == pytest.approx(-0.2) and hi < 0


def test_time_split_guard() -> None:
    from nbalab.models.evaluate import assert_time_split

    class Fake:
        manifest = {"train_seasons": [2012, 2024]}

    with pytest.raises(ValueError):
        assert_time_split(Fake(), (2024, 2025))  # type: ignore[arg-type]
    assert_time_split(type("Ok", (), {"manifest": {"train_seasons": [2012, 2023]}})(), (2024, 2025))  # type: ignore

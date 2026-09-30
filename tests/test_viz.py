"""Tests for the chart helpers: the trend fit math, color helpers, and figure smoke tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from nbalab.viz import charts, teams, theme
from nbalab.viz.trend import linear_trend, rolling_mean, years_since_start


def _dates(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2020-01-01", periods=n, freq="365.25D"))


def test_perfect_line_recovers_slope_and_zero_band() -> None:
    d = _dates(5)
    y = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])  # +2 per year
    fit = linear_trend(d, y)
    assert fit is not None
    assert fit.slope_per_year == pytest.approx(2.0, rel=1e-6)
    np.testing.assert_allclose(fit.fitted, y.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(fit.upper - fit.lower, 0.0, atol=1e-6)


def test_trend_matches_statsmodels_band() -> None:
    rng = np.random.default_rng(0)
    d = _dates(40)
    y = pd.Series(20 + 0.5 * np.arange(40) + rng.normal(0, 3, 40))
    fit = linear_trend(d, y)
    X = sm.add_constant(years_since_start(d))
    res = sm.OLS(y.to_numpy(), X).fit()
    band = res.get_prediction(X).conf_int(alpha=0.05)
    np.testing.assert_allclose(fit.fitted, res.fittedvalues, rtol=1e-8)
    np.testing.assert_allclose(fit.lower, band[:, 0], rtol=1e-8)
    np.testing.assert_allclose(fit.upper, band[:, 1], rtol=1e-8)
    assert fit.slope_p_value == pytest.approx(res.pvalues[1], rel=1e-6)


def test_trend_needs_three_games_and_two_dates() -> None:
    assert linear_trend(_dates(2), pd.Series([1.0, 2.0])) is None
    same = pd.Series([pd.Timestamp("2021-01-01")] * 4)
    assert linear_trend(same, pd.Series([1.0, 2.0, 3.0, 4.0])) is None


def test_trend_drops_nan_values() -> None:
    fit = linear_trend(_dates(5), pd.Series([1.0, np.nan, 3.0, 4.0, 5.0]))
    assert fit is not None and fit.n == 4


def test_rolling_mean_is_trailing() -> None:
    out = rolling_mean(pd.Series([2.0, 4.0, 6.0, 8.0]), window=2)
    assert out.tolist() == [2.0, 3.0, 5.0, 7.0]


def test_contrast_ratio_known_values() -> None:
    assert teams.contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert teams.contrast("#777777", "#777777") == pytest.approx(1.0)


def test_accent_skips_dark_team_colors() -> None:
    # Charlotte's primary (#1D1160) is nearly invisible on the dark surface
    assert teams.accent("CHA") == "#00788C"
    assert teams.accent("XXX") == teams.DEFAULT_ACCENT
    for abbrev in teams.TEAM_COLORS:
        assert teams.contrast(teams.accent(abbrev), theme.SURFACE) >= 3.0


def test_signed_color() -> None:
    assert theme.signed_color(1.2) == theme.UP
    assert theme.signed_color(-0.1) == theme.DOWN
    assert theme.signed_color(0.0) == theme.MUTED
    assert theme.signed_color(math.nan) == theme.MUTED


def test_effect_row_significance() -> None:
    assert charts.EffectRow("a", 2.0, 0.5, 3.5, 30).significant
    assert not charts.EffectRow("b", 1.0, -0.5, 2.5, 30).significant
    assert not charts.EffectRow("c", 1.0, math.nan, math.nan, 1).significant


def test_figures_build() -> None:
    rows = [charts.EffectRow("vs Spurs", 1.5, 0.2, 2.8, 30, 0.03),
            charts.EffectRow("Away", -0.4, -1.0, 0.2, 400),
            charts.EffectRow("Combined", 1.1, -0.5, 2.7, 15, combined=True)]
    assert len(charts.effect_chart(rows, 6.4, "Assists").data) == 1
    s = pd.Series([5, 7, 9, 6]); a = pd.Series(range(0, 15))
    assert len(charts.distribution_chart(s, a, "Assists").data) == 2
    d = _dates(10); v = pd.Series(np.arange(10.0))
    fig = charts.trend_chart(d, v, linear_trend(d, v), "Points", in_split=pd.Series([True, False] * 5),
                             rolling=rolling_mean(v))
    assert len(fig.data) == 5
    ints = {0.90: (20.0, 34.0), 0.95: (18.0, 36.0), 0.99: (14.0, 40.0)}
    assert charts.interval_chart(27.0, ints, 27.5, "Points", 26.5).data
    lines = np.arange(10, 40, 0.5)
    assert charts.probability_curve(lines, np.linspace(1, 0, lines.size), 27.5, "Points").data
    coef = pd.DataFrame({"label": ["Mon", "Tue"], "estimate": [0.1, -0.2], "lo": [-0.1, -0.4], "hi": [0.3, 0.0]})
    assert charts.coefficient_chart(coef, "Points", reference="Wednesday").data
    assert len(charts.calibration_chart([0.1, 0.5, 0.9], [0.12, 0.48, 0.85], [10, 20, 10]).data) == 2


def test_break_at_gaps_inserts_blank_between_seasons() -> None:
    d = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-03", "2020-10-20", "2020-10-22"]))
    xs, ys = charts.break_at_gaps(d, pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert ys == [1.0, 2.0, None, 3.0, 4.0]
    assert len(xs) == 5


def test_projection_figures_build() -> None:
    import numpy as np
    import pandas as pd

    from nbalab.models.trend import trend_bands
    from nbalab.viz import projection as P

    support = np.arange(40)
    pmf = np.exp(-0.5 * ((support - 20) / 5) ** 2)
    pmf /= pmf.sum()
    assert P.outcome_chart(support, pmf, {90: (12, 28), 99: (7, 33)}, 20.5, "Points", 20.0).data
    fig = P.contribution_waterfall(["Opponent", "Home"], [1.2, -0.4], 19.0, "Points")
    assert list(fig.data[0].y)[-1] == pytest.approx(19.8)
    dates = pd.Series(pd.date_range("2025-01-01", periods=30, freq="3D"))
    values = pd.Series(np.linspace(15, 25, 30))
    assert P.trend_chart(dates, values, trend_bands(dates, values, horizon_days=20), "Points", (24, 12, 36)).data

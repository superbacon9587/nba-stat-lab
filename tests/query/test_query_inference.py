"""Hand-computed checks for every function in nbalab.query.inference."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from nbalab.query import inference as inf

SPLIT = np.array([10, 20, 30, 20.0])            # mean 20, var 200/3
REST = np.array([25, 25, 30, 20, 30, 20, 35, 35.0])  # mean 27.5, var 250/7
BASE = np.concatenate([SPLIT, REST])             # mean 25, var 600/11


def test_summarize_hand_values() -> None:
    s = inf.summarize(SPLIT)
    assert (s.n, s.mean, s.median) == (4, 20.0, 20.0)
    assert s.std == pytest.approx(math.sqrt(200 / 3))


def test_summarize_ignores_nan_and_handles_tiny_samples() -> None:
    assert inf.summarize(np.array([np.nan, 4.0])).n == 1
    assert math.isnan(inf.summarize(np.array([4.0])).std)
    assert inf.summarize(np.array([])).n == 0


def test_per36_uses_totals() -> None:
    assert inf.per36(20.0, 30.0) == pytest.approx(24.0)
    # a 40-min 20-pt game plus a 4-min 4-pt cameo: 24 pts / 44 min * 36
    assert inf.per36(24.0, 44.0) == pytest.approx(24 / 44 * 36)
    assert math.isnan(inf.per36(5.0, 0.0))


def test_difference_raw_and_percent() -> None:
    assert inf.difference(20.0, 25.0) == pytest.approx((-5.0, -20.0))
    assert math.isnan(inf.difference(1.0, 0.0)[1])


def test_z_score_hand_value() -> None:
    # (20 - 25) / (sqrt(600/11) / sqrt(4))
    expected = -5 / (math.sqrt(600 / 11) / 2)
    assert inf.z_score(20.0, 25.0, math.sqrt(600 / 11), 4) == pytest.approx(expected)
    assert expected == pytest.approx(-1.354006, rel=1e-5)


def test_welch_statistic_hand_formula() -> None:
    va, vb = (200 / 3) / 4, (250 / 7) / 8
    t_hand = (20 - 27.5) / math.sqrt(va + vb)
    df_hand = (va + vb) ** 2 / (va**2 / 3 + vb**2 / 7)
    t, df = inf.welch_statistic(SPLIT, REST)
    assert t == pytest.approx(t_hand) and t == pytest.approx(-1.631555, rel=1e-5)
    assert df == pytest.approx(df_hand) and df == pytest.approx(4.678526, rel=1e-5)


def test_welch_p_value_matches_t_distribution() -> None:
    t, df = inf.welch_statistic(SPLIT, REST)
    assert inf.welch_p_value(SPLIT, REST) == pytest.approx(2 * sps.t.sf(abs(t), df))
    assert inf.welch_p_value(SPLIT, REST) == pytest.approx(0.16770, abs=1e-4)


def test_welch_p_value_needs_two_per_side() -> None:
    assert math.isnan(inf.welch_p_value(np.array([1.0]), REST))


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate() -> None:
    lo, hi = inf.bootstrap_diff_ci(SPLIT, REST)
    assert (lo, hi) == inf.bootstrap_diff_ci(SPLIT, REST)
    assert lo < 20 - 25 < hi


def test_bootstrap_ci_zero_width_for_constant_data() -> None:
    lo, hi = inf.bootstrap_diff_ci(np.full(5, 3.0), np.full(5, 1.0))
    # split mean 3, baseline mean 2 -> difference exactly 1 in every replicate
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_bootstrap_ci_for_a_ratio_uses_pooled_totals() -> None:
    # split: 10/20 made every game -> 50%; rest: 5/20 -> 25%; pooled baseline 15/40 = 37.5%
    lo, hi = inf.bootstrap_diff_ci(np.full(4, 10.0), np.full(4, 5.0),
                                   split_den=np.full(4, 20.0), rest_den=np.full(4, 20.0))
    assert lo == pytest.approx(0.5 - 0.375) and hi == pytest.approx(0.5 - 0.375)


def test_prior_variance_method_of_moments() -> None:
    # group means 20, 27.5.. use the fixture's three opponents: means 20, 25, 35 with n 4, 6, 2
    within = 600 / 11
    observed = np.var([20, 25, 35], ddof=1)          # 58.333
    luck = within * np.mean([1 / 4, 1 / 6, 1 / 2])   # 16.667
    assert inf.prior_variance(np.array([20, 25, 35.0]), np.array([4, 6, 2]), within) == pytest.approx(observed - luck)
    assert observed - luck == pytest.approx(41.6667, rel=1e-4)


def test_prior_variance_floors_at_zero_and_needs_three_groups() -> None:
    assert inf.prior_variance(np.array([10, 10.1, 10.0]), np.array([5, 5, 5]), 100.0) == 0.0
    assert math.isnan(inf.prior_variance(np.array([1, 2.0]), np.array([5, 5]), 1.0))


def test_shrinkage_weight_and_estimate() -> None:
    tau2, within = 125 / 3, 600 / 11
    w = inf.shrinkage_weight(4, within, tau2)
    assert w == pytest.approx(tau2 / (tau2 + within / 4)) and w == pytest.approx(0.753425, rel=1e-5)
    assert inf.shrink(20.0, 25.0, w) == pytest.approx(21.232877, rel=1e-6)


def test_shrinkage_limits() -> None:
    assert inf.shrinkage_weight(10, 50.0, 0.0) == 0.0            # no real differences -> all baseline
    assert inf.shrinkage_weight(10_000, 50.0, 10.0) > 0.99        # huge sample -> trust the split
    assert inf.shrinkage_weight(3, 50.0, 10.0) < inf.shrinkage_weight(30, 50.0, 10.0)


def test_holm_adjust_hand_values() -> None:
    # sorted 0.01, 0.02, 0.04 -> x3, x2, x1 = 0.03, 0.04, 0.04 (monotone)
    out = inf.holm_adjust(np.array([0.04, 0.01, np.nan, 0.02]))
    assert out[1] == pytest.approx(0.03)
    assert out[3] == pytest.approx(0.04)
    assert out[0] == pytest.approx(0.04)
    assert math.isnan(out[2])

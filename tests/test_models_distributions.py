import numpy as np
import pytest

from nbalab.models import distributions as d


def test_mixture_sums_to_one_and_mean_matches():
    support, pmf = d.mixture_pmf(np.array([20.0, 25.0, 30.0]), np.array([0.25, 0.5, 0.25]), 0.1)
    assert pmf.sum() == pytest.approx(1.0)
    assert (support * pmf).sum() == pytest.approx(25.0, rel=1e-3)


def test_nb_variance_matches_alpha():
    support, pmf = d.mixture_pmf(np.array([10.0]), np.array([1.0]), 0.2)
    mean = (support * pmf).sum()
    var = ((support - mean) ** 2 * pmf).sum()
    assert var == pytest.approx(10 + 0.2 * 100, rel=1e-3)


def test_over_under_push_half_and_integer_lines():
    support, pmf = d.mixture_pmf(np.array([6.0]), np.array([1.0]), 0.0)
    o, u, p = d.over_under_push(support, pmf, 6.5)
    assert p == 0.0 and o + u == pytest.approx(1.0)
    o, u, p = d.over_under_push(support, pmf, 6)
    assert p > 0 and o + u + p == pytest.approx(1.0)


def test_prediction_interval_covers_at_least_nominal():
    rng = np.random.default_rng(0)
    support, pmf = d.mixture_pmf(np.array([8.0]), np.array([1.0]), 0.3)
    lo, hi = d.prediction_interval(support, pmf, 0.9)
    n, p = d.nb_params(8.0, 0.3)
    draws = rng.negative_binomial(n, p, size=50_000)
    assert ((draws >= lo) & (draws <= hi)).mean() >= 0.895


def test_fit_alpha_recovers_truth_and_poisson_detected():
    rng = np.random.default_rng(1)
    mean = rng.uniform(5, 30, size=20_000)
    n, p = d.nb_params(mean, 0.15)
    y = rng.negative_binomial(n, p)
    assert d.fit_nb_alpha(y, mean) == pytest.approx(0.15, abs=0.02)
    assert d.prefer_negative_binomial(y, mean)
    y_pois = rng.poisson(mean)
    assert d.fit_nb_alpha(y_pois, mean) < 0.01

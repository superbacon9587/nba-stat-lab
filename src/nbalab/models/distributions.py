"""Count distributions for single-game stat projections.

A player's points in one game are a count. The Poisson distribution assumes the
variance equals the mean; real scoring is *overdispersed* (variance > mean)
because shot volume and hot/cold nights vary. The negative binomial (NB) adds
one dispersion parameter ``alpha``: variance = mean + alpha x mean^2. With
alpha = 0 it is exactly Poisson.

Minutes are uncertain too, so the projected distribution is a *mixture*: an NB
for each plausible minutes value, weighted by how likely that minutes value is.
All probabilities (over/under/push, intervals) are read off the mixed pmf.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats


def nb_params(mean: np.ndarray | float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert (mean, alpha) to scipy's ``nbinom(n, p)``: n = 1/alpha, p = n / (n + mean)."""
    n = 1.0 / alpha
    mean = np.asarray(mean, dtype=float)
    return np.full_like(mean, n), n / (n + mean)


def count_pmf(mean: np.ndarray | float, alpha: float, support: np.ndarray) -> np.ndarray:
    """P(X = k) for each k in ``support``; one row per mean. Poisson when alpha <= 1e-8."""
    mean = np.atleast_1d(np.asarray(mean, dtype=float))[:, None]
    k = np.asarray(support)[None, :]
    if alpha <= 1e-8:
        return stats.poisson.pmf(k, mean)
    n, p = nb_params(mean, alpha)
    return stats.nbinom.pmf(k, n, p)


def mixture_pmf(
    means: np.ndarray, weights: np.ndarray, alpha: float, max_value: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mix count distributions over minutes scenarios.

    ``means[i]`` is the expected stat if minutes scenario ``i`` happens, and
    ``weights[i]`` its probability. Returns ``(support, pmf)`` with the pmf
    renormalized so tail truncation never loses probability mass.
    """
    means = np.asarray(means, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    if max_value is None:
        top = float(means.max())
        max_value = int(math.ceil(top + 12 * math.sqrt(top + alpha * top**2) + 10))
    support = np.arange(max_value + 1)
    pmf = (w[:, None] * count_pmf(means, alpha, support)).sum(axis=0)
    return support, pmf / pmf.sum()


def quantile(support: np.ndarray, pmf: np.ndarray, q: float) -> float:
    """Smallest value whose cumulative probability reaches ``q``."""
    cdf = np.cumsum(pmf)
    return float(support[min(np.searchsorted(cdf, q - 1e-12), len(support) - 1)])


def prediction_interval(support: np.ndarray, pmf: np.ndarray, level: float) -> tuple[float, float]:
    """Equal-tailed interval holding at least ``level`` (e.g. 0.9) of the probability.

    Counts are discrete, so actual coverage is usually a bit *above* nominal.
    """
    tail = (1.0 - level) / 2.0
    return quantile(support, pmf, tail), quantile(support, pmf, 1.0 - tail)


def over_under_push(support: np.ndarray, pmf: np.ndarray, line: float) -> tuple[float, float, float]:
    """P(stat > line), P(stat < line), P(stat == line). Push is 0 unless the line is an integer."""
    over = float(pmf[support > line].sum())
    under = float(pmf[support < line].sum())
    push = float(pmf[support == line].sum()) if float(line).is_integer() else 0.0
    return over, under, push


def nb_loglik(y: np.ndarray, mean: np.ndarray, alpha: float) -> float:
    """Total log-likelihood of observed counts ``y`` under NB(mean, alpha) (Poisson at alpha=0)."""
    y = np.asarray(y, dtype=float)
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-9)
    if alpha <= 1e-8:
        return float(stats.poisson.logpmf(y, mean).sum())
    n, p = nb_params(mean, alpha)
    return float(stats.nbinom.logpmf(y, n, p).sum())


def fit_nb_alpha(y: np.ndarray, mean: np.ndarray) -> float:
    """Maximum-likelihood dispersion alpha, given each game's predicted mean.

    Returns 0.0 (Poisson) when extra dispersion does not improve the fit.
    """
    res = optimize.minimize_scalar(
        lambda la: -nb_loglik(y, mean, math.exp(la)), bounds=(-12.0, 3.0), method="bounded"
    )
    alpha = math.exp(res.x)
    return alpha if nb_loglik(y, mean, alpha) > nb_loglik(y, mean, 0.0) else 0.0


def prefer_negative_binomial(y: np.ndarray, mean: np.ndarray, p_threshold: float = 0.01) -> bool:
    """Likelihood-ratio test of NB vs Poisson.

    alpha = 0 sits on the boundary, so the LR statistic's null distribution is
    a 50:50 mix of 0 and chi-square(1); the p-value is half the chi-square tail.
    """
    alpha = fit_nb_alpha(y, mean)
    if alpha == 0.0:
        return False
    lr = 2.0 * (nb_loglik(y, mean, alpha) - nb_loglik(y, mean, 0.0))
    return 0.5 * stats.chi2.sf(lr, df=1) < p_threshold


# ------------------------------------------------------------------ many games at once


def support_for(means: np.ndarray, alpha: float) -> np.ndarray:
    """Integer support 0..K wide enough for every mean in ``means`` (tail mass < 1e-9)."""
    top = float(np.nanmax(means)) if np.size(means) else 0.0
    return np.arange(int(math.ceil(top + 12 * math.sqrt(top + alpha * top**2) + 10)) + 1)


def pmf_matrix(means: np.ndarray, weights: np.ndarray, alpha: float, support: np.ndarray,
               chunk: int = 4000) -> np.ndarray:
    """Mixture pmf for many games: ``means`` is (games, nodes), ``weights`` (nodes,) or (games, nodes).

    Row i is the distribution of game i's stat, mixed over its minutes scenarios.
    """
    means = np.atleast_2d(np.asarray(means, dtype=float))
    w = np.broadcast_to(np.asarray(weights, dtype=float), means.shape)
    w = w / w.sum(axis=1, keepdims=True)
    out = np.empty((means.shape[0], len(support)))
    for a in range(0, means.shape[0], chunk):
        m, ww = means[a:a + chunk], w[a:a + chunk]
        pk = count_pmf(m.ravel(), alpha, support).reshape(m.shape[0], m.shape[1], len(support))
        out[a:a + chunk] = np.einsum("gn,gns->gs", ww, pk)
    return out / out.sum(axis=1, keepdims=True)


def over_under_push_rows(support: np.ndarray, pmf: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise P(over), P(under), P(push) for one line per row of ``pmf``."""
    lines = np.asarray(lines, dtype=float)[:, None]
    s = np.asarray(support)[None, :]
    over = (pmf * (s > lines)).sum(axis=1)
    under = (pmf * (s < lines)).sum(axis=1)
    return over, under, 1.0 - over - under


def interval_rows(support: np.ndarray, pmf: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise equal-tailed prediction interval of a pmf matrix."""
    cdf = np.cumsum(pmf, axis=1)
    tail = (1.0 - level) / 2.0
    lo = (cdf < tail - 1e-12).sum(axis=1)
    hi = (cdf < 1.0 - tail - 1e-12).sum(axis=1)
    top = len(support) - 1
    return support[np.minimum(lo, top)].astype(float), support[np.minimum(hi, top)].astype(float)


def cdf_from_quantiles(levels: np.ndarray, values: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Turn predicted quantiles (games, levels) into a pmf matrix on integer ``support``.

    Quantile predictions are sorted first (crossing fix). The CDF is linear
    between the predicted quantiles and is anchored at -0.5 (probability 0) and
    past the top quantile (probability 1). Integer k gets the mass between
    k - 0.5 and k + 0.5.
    """
    v = np.sort(np.atleast_2d(values), axis=1)
    levels = np.asarray(levels, dtype=float)
    span = np.maximum(v[:, -1] - v[:, v.shape[1] // 2], 1.0)
    edges = np.asarray(support, dtype=float) + 0.5
    pmf = np.empty((v.shape[0], len(support)))
    for i in range(v.shape[0]):
        xs = np.concatenate([[-0.5], np.maximum(v[i], -0.5), [v[i, -1] + 2 * span[i] + 1]])
        xs = np.maximum.accumulate(xs + np.arange(len(xs)) * 1e-9)
        ps = np.concatenate([[0.0], levels, [1.0]])
        cdf = np.interp(edges, xs, ps)
        pmf[i] = np.diff(np.concatenate([[0.0], cdf]))
    pmf = np.clip(pmf, 0, None)
    return pmf / pmf.sum(axis=1, keepdims=True)


def nb_cdf(y: np.ndarray, mean: np.ndarray, alpha: float) -> np.ndarray:
    """P(X <= y) under NB(mean, alpha); Poisson when alpha <= 1e-8. Negative y gives 0."""
    y = np.asarray(y, dtype=float)
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-9)
    if alpha <= 1e-8:
        return stats.poisson.cdf(y, mean)
    n, p = nb_params(mean, alpha)
    return stats.nbinom.cdf(y, n, p)


def pmf_moments(support: np.ndarray, pmf: np.ndarray) -> tuple[float, float]:
    """Mean and variance of a discrete distribution."""
    mean = float((support * pmf).sum())
    return mean, float(((support - mean) ** 2 * pmf).sum())

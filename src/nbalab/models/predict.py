"""Predictive distributions for many games at once (training selection and backtests).

Every candidate model is turned into the same thing: a pmf over integer
outcomes for each game. That makes every score (log score, MAE of the mean,
P(over), interval coverage) comparable across model families.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nbalab.models.copula import randomized_pit
from nbalab.models.distributions import nb_cdf, pmf_matrix
from nbalab.models.quantile import MINUTES_FEATURE, QuantileModel


def outcome_support(y: np.ndarray, pad: float = 1.4) -> np.ndarray:
    """0..K covering every observed outcome with room for the upper tail."""
    return np.arange(int(np.nanmax(y) * pad) + 15)


def model_pmfs(model, frame: pd.DataFrame, support: np.ndarray, nodes: np.ndarray | None = None,
               weights: np.ndarray | None = None, multiplier: np.ndarray | None = None) -> np.ndarray:
    """(games, support) pmfs from any candidate model.

    Player models mix over the minutes scenarios ``nodes`` (games, K); team
    models have none. ``multiplier`` scales each game's rate (defender proxy).
    """
    if isinstance(model, QuantileModel):
        f = frame
        if model.use_minutes:
            f = frame.assign(**{MINUTES_FEATURE: np.average(nodes, axis=1, weights=weights)})
        return model.pmf(f, support)
    rate = model.predict_rate(frame)
    if multiplier is not None:
        rate = rate * multiplier
    if model.use_minutes:
        means, w = nodes * rate[:, None], weights
    else:
        means, w = rate[:, None], np.ones(1)
    return pmf_matrix(means, w, model.alpha, support)


def pmf_mean(pmf: np.ndarray, support: np.ndarray) -> np.ndarray:
    return pmf @ np.asarray(support, dtype=float)


def log_score(pmf: np.ndarray, support: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log P(Y = observed y) for each game (higher is better); floored at log(1e-12)."""
    idx = np.clip(np.asarray(y, dtype=int), 0, len(support) - 1)
    return np.log(np.maximum(pmf[np.arange(len(idx)), idx], 1e-12))


def conditional_pit(model, frame: pd.DataFrame, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized PIT of each observed outcome under the model *at the minutes actually played*.

    Conditioning on real minutes removes the shared-minutes part of the
    correlation between stats, which the simulation handles separately.
    """
    y = np.asarray(y, dtype=float)
    if isinstance(model, QuantileModel):
        f = frame.assign(**{MINUTES_FEATURE: frame["minutes"].to_numpy(float)}) if model.use_minutes else frame
        support = outcome_support(y)
        cdf = np.cumsum(model.pmf(f, support), axis=1)
        idx = np.clip(y.astype(int), 0, len(support) - 1)
        at = cdf[np.arange(len(y)), idx]
        below = np.where(idx > 0, cdf[np.arange(len(y)), np.maximum(idx - 1, 0)], 0.0)
        return randomized_pit(below, at, rng)
    mean = model.predict_rate(frame) * (frame["minutes"].to_numpy(float) if model.use_minutes else 1.0)
    return randomized_pit(nb_cdf(y - 1, mean, model.alpha), nb_cdf(y, mean, model.alpha), rng)

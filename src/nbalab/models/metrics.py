"""Scores for the backtest, in plain terms.

- **MAE / RMSE**: average miss of the expected value (RMSE punishes big misses more).
- **Brier score**: mean squared error of a probability, (p - outcome)^2.
  Always guessing 50% scores 0.25; lower is better.
- **Log loss**: -log of the probability given to what happened; punishes
  confident mistakes hard. Always guessing 50% scores 0.693.
- **Coverage**: share of games whose result landed inside the interval.
  A good 90% interval covers about 90%.
- **Calibration**: group predictions into buckets (e.g. 60-70%) and check
  the actual hit rate in each bucket.
- **Paired bootstrap**: resample *players* (a player's games are not
  independent) and recompute the MAE difference between two models, to see
  whether "model A beats model B" survives resampling or is noise.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(pred, float) - np.asarray(y, float)) ** 2)))


def brier(p: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(outcome, float)) ** 2))


def log_loss(p: np.ndarray, outcome: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    o = np.asarray(outcome, float)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def calibration_bins(p: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability buckets: mean prediction, actual rate and count per non-empty bucket."""
    p = np.asarray(p, float)
    o = np.asarray(outcome, float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"b": b, "p": p, "o": o}).groupby("b").agg(bin_pred=("p", "mean"), bin_actual=("o", "mean"),
                                                                n=("o", "size"))
    return df.reset_index(drop=True)


def paired_bootstrap_diff(err_a: np.ndarray, err_b: np.ndarray, groups: np.ndarray, n_boot: int = 1000,
                          seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Mean of (err_a - err_b) and a cluster-bootstrap CI that resamples whole groups (players).

    Negative means A has the smaller error. If the CI excludes 0, the gap is
    unlikely to be resampling noise.
    """
    diff = np.asarray(err_a, float) - np.asarray(err_b, float)
    codes, uniq = pd.factorize(np.asarray(groups))
    sums = np.bincount(codes, weights=diff)
    counts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    tail = (1 - level) / 2
    return float(diff.mean()), float(np.quantile(boots, tail)), float(np.quantile(boots, 1 - tail))


def synthetic_lines(season_avg: np.ndarray, fallback: np.ndarray, games_so_far: np.ndarray,
                    min_games: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Stand-in betting lines (the dataset has no sportsbook lines).

    The pre-game season-to-date average (or ``fallback``, e.g. an EWMA, before
    ``min_games`` games) rounded to x.5, and the same rounded to a whole
    number (which can push).
    """
    base = np.where(np.asarray(games_so_far) >= min_games, season_avg, fallback)
    return np.floor(base) + 0.5, np.round(base)


def climatology(y: np.ndarray, lines: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each game's historical hit rate: share of the group's *earlier* games above this game's line.

    Rows must be in time order within each group. 0.5 when there is no earlier game.
    """
    y = np.asarray(y, float)
    lines = np.asarray(lines, float)
    out = np.full(len(y), 0.5)
    order = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(list)
    for idx in order:
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        yy, ll = y[idx], lines[idx]
        hits = (yy[None, :] > ll[:, None])
        mask = np.tri(len(idx), k=-1, dtype=bool)
        n = mask.sum(axis=1)
        rate = (hits & mask).sum(axis=1) / np.maximum(n, 1)
        out[idx] = np.where(n > 0, rate, 0.5)
    return out

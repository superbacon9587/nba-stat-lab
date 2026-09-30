"""Joint outcomes: how likely are several overs to hit *together*?

Points, rebounds and assists move together partly because they share
minutes. The simulation draws one minutes scenario per simulated game and
uses it for every stat, which captures that part exactly.

What's left is the correlation *given* minutes (a player who shoots a lot
tonight may pass less). It is measured with a Gaussian copula:

1. For each past game and stat, the probability integral transform (PIT)
   u = P(X < y) + V x P(X = y), with V uniform on [0, 1], where X is the model's
   distribution for that game at the minutes actually played. Randomizing
   within the observed count makes u exactly uniform for a correct discrete
   model.
2. z = Phi^-1(u) turns each stat into a standard normal score, and the
   correlation matrix of those scores is the dependence between stats.
3. With few games the matrix is noisy, so it is shrunk toward the
   league-average matrix: R = w x R_player + (1 - w) x R_league,
   w = n / (n + n0).

Simulation: correlated normals -> uniforms -> each stat's own inverse CDF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

SHRINK_GAMES = 100.0
Condition = tuple[str, float]  # (">", 28.5), ("<", 6.5), (">=", 7) ...


def randomized_pit(cdf_below: np.ndarray, cdf_at: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """u = F(y - 1) + V x (F(y) - F(y - 1)), uniform on [0, 1] when the model is right."""
    lo = np.asarray(cdf_below, dtype=float)
    return lo + rng.uniform(size=lo.shape) * (np.asarray(cdf_at, dtype=float) - lo)


def normal_scores(u: np.ndarray) -> np.ndarray:
    return sps.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))


def correlation(z: np.ndarray) -> np.ndarray:
    """Correlation of the columns of ``z`` (rows with any NaN dropped); identity if < 3 rows."""
    z = z[~np.isnan(z).any(axis=1)]
    k = z.shape[1]
    if len(z) < 3:
        return np.eye(k)
    r = np.corrcoef(z, rowvar=False)
    return np.nan_to_num(np.atleast_2d(r), nan=0.0) * (1 - np.eye(k)) + np.eye(k)


def shrink_correlation(own: np.ndarray, league: np.ndarray, n: int, n0: float = SHRINK_GAMES) -> np.ndarray:
    """Weighted average of the player's and the league's matrix, weight n / (n + n0) on the player."""
    w = n / (n + n0)
    return nearest_correlation(w * own + (1 - w) * league)


def nearest_correlation(r: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so the matrix is a valid correlation matrix."""
    vals, vecs = np.linalg.eigh((r + r.T) / 2)
    fixed = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def simulate(node_weights: np.ndarray, cdf_tables: dict[str, np.ndarray], corr: np.ndarray,
             n_draws: int, rng: np.random.Generator, node_values: np.ndarray | None = None) -> pd.DataFrame:
    """Draw ``n_draws`` joint games.

    ``cdf_tables[stat]`` is (nodes, support): the stat's CDF under each minutes
    scenario, in the same order as ``corr``. One scenario is drawn per game and
    shared by every stat.
    """
    stats = list(cdf_tables)
    w = np.asarray(node_weights, dtype=float)
    node = rng.choice(len(w), size=n_draws, p=w / w.sum())
    z = rng.multivariate_normal(np.zeros(len(stats)), corr, size=n_draws, method="cholesky")
    u = sps.norm.cdf(z)
    out = {}
    for j, s in enumerate(stats):
        table = cdf_tables[s]
        vals = np.empty(n_draws)
        for k in np.unique(node):
            idx = node == k
            vals[idx] = np.minimum(np.searchsorted(table[k], u[idx, j]), table.shape[1] - 1)
        out[s] = vals
    df = pd.DataFrame(out)
    if node_values is not None:
        df["minutes"] = np.asarray(node_values)[node]
    return df


def joint_probability(draws: pd.DataFrame, conditions: dict[str, Condition]) -> float:
    """Share of simulated games where every condition holds, e.g. {"points": (">", 28.5)}."""
    ops = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal, "==": np.equal}
    mask = np.ones(len(draws), dtype=bool)
    for stat, (op, value) in conditions.items():
        mask &= ops[op](draws[stat].to_numpy(), value)
    return float(mask.mean())

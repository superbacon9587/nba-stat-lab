"""The statistics behind every split: summaries, significance tests, and shrinkage.

Every function here is pure (arrays in, numbers out) and tested against
hand-computed answers in ``tests/query/test_query_inference.py``.

Vocabulary
----------
- **Split**: the games that pass the user's filters ("Curry vs the Spurs").
- **Baseline**: all of the subject's games in the same seasons, with no split filters.
- **Rest**: the baseline games that are *not* in the split. Significance tests
  compare split vs. rest, because the split is part of the baseline and a test
  of a sample against a set that contains it is not a valid two-sample test.
  "Split differs from baseline" and "split differs from rest" are the same
  hypothesis: split - baseline = (1 - f) x (split - rest), where f is the
  split's share of baseline games.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_SEED = 20260929


@dataclass(frozen=True)
class Summary:
    """Plain description of a sample of per-game values."""

    n: int
    mean: float
    median: float
    std: float


def summarize(values: np.ndarray) -> Summary:
    """Count, mean, median and sample standard deviation (n - 1 denominator).

    NaNs (e.g. a shooting % in a game with no attempts) are ignored. The
    standard deviation of a single game is undefined and returned as NaN.
    """
    v = _clean(values)
    if v.size == 0:
        return Summary(0, math.nan, math.nan, math.nan)
    std = float(np.std(v, ddof=1)) if v.size > 1 else math.nan
    return Summary(int(v.size), float(np.mean(v)), float(np.median(v)), std)


def per36(stat_total: float, minutes_total: float) -> float:
    """Rate per 36 minutes: total stat / total minutes x 36.

    Totals (not the mean of per-game rates) are used so a 4-minute cameo does
    not count as much as a 40-minute start.
    """
    return stat_total / minutes_total * 36.0 if minutes_total > 0 else math.nan


def difference(split_mean: float, baseline_mean: float) -> tuple[float, float]:
    """Raw difference (split - baseline) and percent difference relative to baseline."""
    raw = split_mean - baseline_mean
    if baseline_mean == 0 or math.isnan(baseline_mean):
        return raw, math.nan
    return raw, raw / abs(baseline_mean) * 100.0


def z_score(split_mean: float, baseline_mean: float, baseline_std: float, n_split: int) -> float:
    """How many standard errors the split mean sits from the baseline mean.

    If the split games were just a random draw of ``n_split`` baseline games,
    their mean would scatter around the baseline mean with standard error
    baseline_std / sqrt(n_split). z = (split mean - baseline mean) / that
    standard error. |z| > 1.96 corresponds roughly to p < 0.05.
    """
    if n_split < 1 or not baseline_std or math.isnan(baseline_std):
        return math.nan
    return (split_mean - baseline_mean) / (baseline_std / math.sqrt(n_split))


def welch_p_value(split: np.ndarray, rest: np.ndarray) -> float:
    """Two-sided Welch t-test p-value for "split and rest have the same mean".

    Welch's version does not assume the two groups have equal variance, which
    matters because splits are often small and noisy. Needs at least two games
    on each side. Returns NaN otherwise.
    """
    a, b = _clean(split), _clean(rest)
    if a.size < 2 or b.size < 2:
        return math.nan
    if np.var(a) == 0 and np.var(b) == 0:
        return 0.0 if a.mean() != b.mean() else 1.0
    with warnings.catch_warnings():  # scipy warns when one side is constant; the test is still defined
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(sps.ttest_ind(a, b, equal_var=False).pvalue)


def welch_statistic(split: np.ndarray, rest: np.ndarray) -> tuple[float, float]:
    """Welch t statistic and Welch-Satterthwaite degrees of freedom (for tests and display).

    t = (mean_a - mean_b) / sqrt(s_a^2/n_a + s_b^2/n_b)
    df = (s_a^2/n_a + s_b^2/n_b)^2 / [(s_a^2/n_a)^2/(n_a-1) + (s_b^2/n_b)^2/(n_b-1)]
    """
    a, b = _clean(split), _clean(rest)
    va, vb = np.var(a, ddof=1) / a.size, np.var(b, ddof=1) / b.size
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    return float(t), float(df)


def bootstrap_diff_ci(
    split: np.ndarray,
    rest: np.ndarray,
    level: float = 0.95,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
    split_den: np.ndarray | None = None,
    rest_den: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for (split value - baseline value).

    Each replicate resamples the split games and the rest games separately,
    with replacement, rebuilds the baseline as their union, and records
    split value - baseline value. The middle ``level`` share of those replicate
    differences is the interval. Resampling within each group keeps the group
    sizes fixed, as they are in the real data. The seed is fixed so answers are
    reproducible.

    For counting stats the value is the mean. For ratio stats (FG%, TS%) pass
    the per-game numerators as ``split``/``rest`` and the attempts as
    ``split_den``/``rest_den``. The value is then total made / total attempted,
    so games are resampled whole, keeping each game's makes with its attempts.
    """
    a_num, a_den = _paired(split, split_den)
    b_num, b_den = _paired(rest, rest_den)
    if a_num.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a_num.size, size=(reps, a_num.size))
    a_n, a_d = a_num[ia].sum(axis=1), a_den[ia].sum(axis=1)
    if b_num.size:
        ib = rng.integers(0, b_num.size, size=(reps, b_num.size))
        b_n, b_d = b_num[ib].sum(axis=1), b_den[ib].sum(axis=1)
    else:
        b_n, b_d = np.zeros(reps), np.zeros(reps)
    with np.errstate(invalid="ignore", divide="ignore"):
        diffs = a_n / a_d - (a_n + b_n) / (a_d + b_d)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return math.nan, math.nan
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(diffs, [tail, 100.0 - tail])
    return float(lo), float(hi)


def _paired(num: np.ndarray, den: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Numerator/denominator arrays with NaN rows dropped (denominator 1 for plain means)."""
    n = np.asarray(num, dtype=float)
    d = np.ones_like(n) if den is None else np.asarray(den, dtype=float)
    ok = ~(np.isnan(n) | np.isnan(d))
    return n[ok], d[ok]


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values for testing several splits at once.

    Looking at 7 weekdays and flagging any with p < 0.05 finds a "significant"
    day by luck far more often than 5% of the time. Holm's method sorts the
    p-values, multiplies the k-th smallest by (m - k + 1), and keeps the
    results non-decreasing. It controls the chance of *any* false alarm across
    the family. NaNs are left out of the count and stay NaN.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out[order] = adjusted
    return out


# ------------------------------------------------------------------- shrinkage


def prior_variance(group_means: np.ndarray, group_sizes: np.ndarray, within_var: float) -> float:
    """Empirical Bayes estimate of how much true split averages really differ (tau^2).

    Observed group averages (e.g. a player's average vs each of 29 opponents)
    vary for two reasons: real differences between the groups (tau^2) and
    luck, because each average is built from only n_g games (within_var / n_g).
    Method of moments: tau^2 = variance of the group averages - average luck
    variance. It is floored at 0: if the averages spread no more than luck
    alone would predict, there is no evidence of any real difference.
    """
    m = np.asarray(group_means, dtype=float)
    n = np.asarray(group_sizes, dtype=float)
    ok = ~np.isnan(m) & (n > 0)
    m, n = m[ok], n[ok]
    if m.size < 3 or not within_var or math.isnan(within_var):
        return math.nan
    observed = float(np.var(m, ddof=1))
    luck = float(np.mean(within_var / n))
    return max(observed - luck, 0.0)


def shrinkage_weight(n: int, within_var: float, tau2: float) -> float:
    """Weight B on the split's own average; 1 - B goes to the baseline.

    B = tau^2 / (tau^2 + within_var / n). This is equivalent to
    n / (n + k) with k = within_var / tau^2, the number of games it takes
    before the split's own data counts as much as the baseline. With few games,
    or when real differences between splits are small (tau^2 near 0), B is
    near 0 and the estimate is pulled toward the baseline.
    """
    if n < 1 or math.isnan(tau2) or math.isnan(within_var):
        return math.nan
    if within_var == 0:
        return 1.0
    return tau2 / (tau2 + within_var / n)


def shrink(split_mean: float, baseline_mean: float, weight: float) -> float:
    """Empirical Bayes estimate: weight x split mean + (1 - weight) x baseline mean."""
    if math.isnan(weight):
        return math.nan
    return weight * split_mean + (1.0 - weight) * baseline_mean


def _clean(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[~np.isnan(v)]

"""Numbers behind the trend chart: a straight-line fit of a stat over time.

The fit is ordinary least squares, value = a + b x t, where t is years since
the first game. The shaded band is the 95% **confidence** band for the fitted
line: where the true average trend plausibly runs. It is not a prediction
band for single games, which would be much wider, since one game scatters far
more than an average does.

Band half-width at time t:
    t_crit x s x sqrt(1/n + (t - mean t)^2 / sum((t_i - mean t)^2))
where s is the residual standard deviation and t_crit the Student-t quantile
with n - 2 degrees of freedom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class TrendFit:
    """A fitted line plus its confidence band, evaluated at every input date."""

    dates: pd.Series
    fitted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    slope_per_year: float
    slope_p_value: float
    n: int


def years_since_start(dates: pd.Series) -> np.ndarray:
    """Convert dates to fractional years since the earliest date."""
    d = pd.to_datetime(dates)
    return ((d - d.min()).dt.total_seconds() / (365.25 * 86400)).to_numpy(dtype=float)


def linear_trend(dates: pd.Series, values: pd.Series, level: float = 0.95) -> TrendFit | None:
    """OLS line of ``values`` against time, with a confidence band.

    Returns None when there are fewer than 3 usable games, or when all games
    fall on one date, because a line with a band needs at least that much.
    """
    df = pd.DataFrame({"d": pd.to_datetime(dates).to_numpy(), "v": pd.to_numeric(values, errors="coerce").to_numpy()})
    df = df.dropna().sort_values("d").reset_index(drop=True)
    n = len(df)
    if n < 3:
        return None
    t = years_since_start(df["d"])
    y = df["v"].to_numpy(dtype=float)
    sxx = float(np.sum((t - t.mean()) ** 2))
    if sxx == 0:
        return None
    b = float(np.sum((t - t.mean()) * (y - y.mean())) / sxx)
    a = float(y.mean() - b * t.mean())
    fitted = a + b * t
    s = math.sqrt(float(np.sum((y - fitted) ** 2)) / (n - 2))
    t_crit = float(sps.t.ppf(0.5 + level / 2, n - 2))
    half = t_crit * s * np.sqrt(1.0 / n + (t - t.mean()) ** 2 / sxx)
    se_b = s / math.sqrt(sxx)
    p = float(2 * sps.t.sf(abs(b / se_b), n - 2)) if se_b > 0 else (0.0 if b != 0 else 1.0)
    return TrendFit(df["d"], fitted, fitted - half, fitted + half, b, p, n)


def rolling_mean(values: pd.Series, window: int = 10) -> pd.Series:
    """Trailing average over the last ``window`` games (fewer at the start)."""
    return pd.to_numeric(values, errors="coerce").rolling(window, min_periods=1).mean()

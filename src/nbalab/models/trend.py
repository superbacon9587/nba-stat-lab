"""Straight-line trend of a stat over time, with both kinds of band.

Ordinary least squares, stat = a + b x years. statsmodels' ``get_prediction``
gives two bands at every date:

- the **confidence band** for the fitted line: where the true average trend
  plausibly runs (narrow, shrinks with more games);
- the **prediction band** for a single game: where one game's box score is
  likely to land (wide, never narrower than the game-to-game scatter).

The band can be extended past the last game to show possible future outcomes.
It is descriptive: a straight line cannot know about injuries or role changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

DAY_YEARS = 1.0 / 365.25


def trend_bands(dates: pd.Series, values: pd.Series, level: float = 0.95,
                horizon_days: int = 0, n_future: int = 10) -> pd.DataFrame | None:
    """Fitted line, CI band and PI band at each game date, plus ``n_future`` points ahead.

    Returns None with fewer than 3 games or a single date. Columns: date,
    fitted, ci_lo, ci_hi, pi_lo, pi_hi, is_future.
    """
    df = pd.DataFrame({"d": pd.to_datetime(dates).to_numpy(), "v": pd.to_numeric(values, errors="coerce").to_numpy()})
    df = df.dropna().sort_values("d")
    if len(df) < 3 or df["d"].nunique() < 2:
        return None
    t0 = df["d"].min()
    t = ((df["d"] - t0).dt.total_seconds() / 86400.0 * DAY_YEARS).to_numpy()
    res = sm.OLS(df["v"].to_numpy(float), sm.add_constant(t)).fit()
    out_dates = df["d"]
    if horizon_days > 0:
        future = pd.date_range(df["d"].max(), df["d"].max() + pd.Timedelta(days=horizon_days), periods=n_future + 1)[1:]
        out_dates = pd.concat([out_dates, pd.Series(future)], ignore_index=True)
    tt = ((pd.to_datetime(out_dates) - t0).dt.total_seconds() / 86400.0 * DAY_YEARS).to_numpy()
    frame = res.get_prediction(sm.add_constant(tt, has_constant="add")).summary_frame(alpha=1 - level)
    return pd.DataFrame({
        "date": pd.to_datetime(out_dates).to_numpy(),
        "fitted": frame["mean"].to_numpy(),
        "ci_lo": frame["mean_ci_lower"].to_numpy(), "ci_hi": frame["mean_ci_upper"].to_numpy(),
        "pi_lo": frame["obs_ci_lower"].to_numpy(), "pi_hi": frame["obs_ci_upper"].to_numpy(),
        "is_future": np.arange(len(out_dates)) >= len(df),
    })

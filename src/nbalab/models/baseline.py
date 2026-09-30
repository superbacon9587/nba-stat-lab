"""The benchmark every model has to beat.

Expected stat = recent form x opponent adjustment x home adjustment:

    mean = EWMA(stat) x (opponent DRtg / league DRtg)^beta x exp(gamma x home) x c

- EWMA(stat) is the player's (or team's) exponentially weighted average of
  previous games. The half-life (5, 10 or 20 games) is chosen on the
  validation season.
- beta, gamma and the scale c are fitted once, league-wide, by Poisson
  regression with log(EWMA) as an offset.
- The distribution is negative binomial. Its dispersion comes from the
  subject's own last 40 games (method of moments: alpha = (var - mean) / mean^2),
  shrunk toward the league alpha: alpha_i = (n x own + 40 x league) / (n + 40).

It deliberately uses no minutes model, no positional defense and no trees,
so a gap between it and the main model measures what that machinery adds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from nbalab.data.features import lagged_rolling_mean
from nbalab.models.distributions import fit_nb_alpha
from nbalab.models.features import lagged_ewm_mean

HALFLIVES: tuple[int, ...] = (5, 10, 20)
FORM_FLOOR = 0.1  # per game; keeps log(form) finite for players who (almost) never record the stat
VAR_WINDOW = 40
ALPHA_PRIOR_GAMES = 40.0


def form(frame: pd.DataFrame, by: str, stat: str, halflife: int) -> pd.Series:
    """Lagged EWMA of the per-game stat, floored at ``FORM_FLOOR``. ``frame`` must be sorted by ``by`` then time."""
    return lagged_ewm_mean(frame, by, [stat], halflife)[stat].clip(lower=FORM_FLOOR)


def own_alpha(frame: pd.DataFrame, by: str, stat: str, league_alpha: float) -> np.ndarray:
    """Per-game dispersion from the subject's previous games, shrunk toward the league value."""
    f = frame.assign(_y=frame[stat], _y2=frame[stat] ** 2)
    m = lagged_rolling_mean(f, by, "_y", VAR_WINDOW)
    m2 = lagged_rolling_mean(f, by, "_y2", VAR_WINDOW)
    n = f.groupby(by, sort=False).cumcount().clip(upper=VAR_WINDOW).astype(float)
    var = (m2 - m**2) * n / (n - 1).where(n > 1)
    own = ((var - m) / (m**2).where(m > 0)).clip(lower=0).fillna(league_alpha)
    return ((n * own + ALPHA_PRIOR_GAMES * league_alpha) / (n + ALPHA_PRIOR_GAMES)).to_numpy()


def design(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "const": 1.0,
        "log_opp_drtg": np.log(frame["opp_drtg_rel"].astype(float)).fillna(0.0),
        "home": frame["home"].astype(float).fillna(0.5),
    }, index=frame.index)


@dataclass
class StatBaseline:
    """Fitted coefficients for one stat."""

    halflife: int
    params: dict[str, float]
    league_alpha: float


@dataclass
class Baseline:
    """Benchmark for every stat of one subject type (``by`` = personId or teamId)."""

    by: str
    stats: dict[str, StatBaseline]

    @staticmethod
    def fit_stat(train: pd.DataFrame, by: str, stat: str, halflife: int) -> StatBaseline:
        ewm = form(train, by, stat, halflife)
        ok = ewm.notna() & train["opp_drtg_rel"].notna()
        X = design(train[ok])
        res = sm.GLM(train.loc[ok, stat].to_numpy(float), X, family=sm.families.Poisson(),
                     offset=np.log(ewm[ok].to_numpy(float))).fit()
        mean = res.predict(X, offset=np.log(ewm[ok].to_numpy(float)))
        return StatBaseline(halflife, {k: float(v) for k, v in res.params.items()},
                            fit_nb_alpha(train.loc[ok, stat].to_numpy(float), np.asarray(mean)))

    def predict(self, frame: pd.DataFrame, stat: str) -> tuple[np.ndarray, np.ndarray]:
        """(mean, alpha) per row. Rows with no history get NaN means."""
        sb = self.stats[stat]
        ewm = form(frame, self.by, stat, sb.halflife).to_numpy(float)
        X = design(frame)
        eta = X.to_numpy(float) @ np.array([sb.params[c] for c in X.columns])
        mean = ewm * np.exp(eta)  # NaN when there is no earlier game
        return mean, own_alpha(frame, self.by, stat, sb.league_alpha)

    def save(self, path: Path, name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / f"baseline_{name}.json").write_text(json.dumps({
            "by": self.by, "stats": {s: vars(b) for s, b in self.stats.items()}}))

    @classmethod
    def load(cls, path: Path, name: str) -> "Baseline":
        d = json.loads((path / f"baseline_{name}.json").read_text())
        return cls(d["by"], {s: StatBaseline(**b) for s, b in d["stats"].items()})


def pmf_rows(means: np.ndarray, alphas: np.ndarray, support: np.ndarray) -> np.ndarray:
    """NB pmf with a different mean *and* alpha per row."""
    mu = np.maximum(np.asarray(means, dtype=float), 1e-9)[:, None]
    a = np.maximum(np.asarray(alphas, dtype=float), 1e-6)[:, None]
    n = 1.0 / a
    pmf = sps.nbinom.pmf(np.asarray(support)[None, :], n, n / (n + mu))
    return pmf / pmf.sum(axis=1, keepdims=True)

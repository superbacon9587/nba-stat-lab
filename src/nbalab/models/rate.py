"""Per-minute rate models: expected stat per minute given the game context.

Both candidates use a log link with log(minutes) as an *offset*:

    log E[stat] = log(minutes) + f(features)

so the expected stat is minutes x rate, and the rate is exp(f). The offset
makes a projection scale exactly with minutes, which is what lets the app's
minutes override work without retraining.

- **GLM**: Poisson regression (statsmodels) on a small, readable set of
  log-rates and context terms. Poisson estimation of the mean is consistent
  even when the counts are overdispersed; dispersion is fitted afterwards.
- **LightGBM**: gradient-boosted trees with the Poisson objective and
  ``init_score = log(minutes)``, early-stopped on the validation season.

Team models are the same with no offset (a team always plays ~240 minutes).
Dispersion alpha (negative binomial) is fitted per stat on held-out games; if
a likelihood-ratio test cannot tell it from 0, the stat is modeled as Poisson.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from nbalab.models.config import SEED
from nbalab.models.distributions import fit_nb_alpha, prefer_negative_binomial

LGBM_PARAMS: dict = {
    "objective": "poisson", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 300,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 2.0,
    "verbose": -1, "seed": SEED,
}
EPS = 1e-3


def offset(frame: pd.DataFrame, use_minutes: bool) -> np.ndarray:
    """log(minutes) for player models, 0 for team models."""
    return np.log(frame["minutes"].to_numpy(dtype=float)) if use_minutes else np.zeros(len(frame))


def fit_dispersion(y: np.ndarray, mean: np.ndarray) -> float:
    """NB alpha from held-out games, or 0.0 (Poisson) if the extra parameter isn't justified."""
    return fit_nb_alpha(y, mean) if prefer_negative_binomial(y, mean) else 0.0


# ------------------------------------------------------------------ GLM


def _log(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float).clip(lower=0) + EPS)


def player_glm_design(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Compact GLM design for a player stat: log-rates, opponent, schedule and role terms."""
    f = frame
    std = f[f"{stat}_rate_std"].where(f["games_std"].ge(5), f[f"{stat}_rate_ewm20"])
    age = f["age"].astype(float) - 27.0
    return pd.DataFrame({
        "log_rate_ewm5": _log(f[f"{stat}_rate_ewm5"]),
        "log_rate_ewm20": _log(f[f"{stat}_rate_ewm20"]),
        "log_rate_std": _log(std),
        "log_opp_pos": np.log(f[f"{stat}_opp_pos"].astype(float)),
        "log_opp_drtg": np.log(f["opp_drtg_rel"].astype(float)),
        "log_opp_pace": np.log(f["opp_pace_rel"].astype(float)),
        "log_team_pace": np.log(f["team_pace_rel"].astype(float)),
        "home": f["home"].astype(float),
        "b2b": f["team_is_back_to_back"].astype(float),
        "rest": f["team_rest_days"].astype(float).clip(0, 3),
        "playoff": f["is_playoff"].astype(float),
        "starter_share": f["starter_share10"].astype(float),
        "age": age, "age2": age**2 / 10.0,
        "guard": f["pos_code"].eq(0).astype(float), "center": f["pos_code"].eq(2).astype(float),
    }, index=f.index)


def team_glm_design(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Compact GLM design for a team stat."""
    f = frame
    return pd.DataFrame({
        "log_ewm5": _log(f[f"{stat}_ewm5"]),
        "log_ewm20": _log(f[f"{stat}_ewm20"]),
        "log_opp_allowed20": _log(f[f"opp_allowed_{stat}_ewm20"]),
        "log_opp_drtg": np.log(f["opp_drtg_rel"].astype(float)),
        "log_opp_pace": np.log(f["opp_pace_rel"].astype(float)),
        "log_team_pace": np.log(f["team_pace_rel"].astype(float)),
        "log_team_ortg": np.log(f["team_ortg_rel"].astype(float)),
        "home": f["home"].astype(float),
        "b2b": f["team_is_back_to_back"].astype(float),
        "opp_b2b": f["opp_is_back_to_back"].astype(float),
        "playoff": f["is_playoff"].astype(float),
    }, index=f.index)


DESIGNS = {"player": player_glm_design, "team": team_glm_design}


@dataclass
class GlmRate:
    """Poisson GLM on a log scale; ``predict_rate`` is per minute (player) or per game (team)."""

    stat: str
    subject: str
    params: dict[str, float]
    fill: dict[str, float]
    alpha: float = 0.0
    kind: str = "glm"

    @property
    def use_minutes(self) -> bool:
        return self.subject == "player"

    def design(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = DESIGNS[self.subject](frame, self.stat).replace([np.inf, -np.inf], np.nan)
        return X.fillna(self.fill) if self.fill else X

    @classmethod
    def fit(cls, train: pd.DataFrame, stat: str, subject: str) -> "GlmRate":
        X = DESIGNS[subject](train, stat).replace([np.inf, -np.inf], np.nan)
        fill = X.median().to_dict()
        X = sm.add_constant(X.fillna(fill), has_constant="add")
        use_minutes = subject == "player"
        res = sm.GLM(train[stat].to_numpy(float), X, family=sm.families.Poisson(),
                     offset=offset(train, use_minutes)).fit()
        return cls(stat, subject, {k: float(v) for k, v in res.params.items()}, fill)

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(self.design(frame), has_constant="add")
        beta = np.array([self.params[c] for c in X.columns])
        return np.exp(X.to_numpy(float) @ beta)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{self.subject}_{self.stat}_glm.json").write_text(json.dumps({
            "stat": self.stat, "subject": self.subject, "params": self.params, "fill": self.fill, "alpha": self.alpha,
        }))

    @classmethod
    def load(cls, path: Path, subject: str, stat: str) -> "GlmRate":
        d = json.loads((path / f"{subject}_{stat}_glm.json").read_text())
        return cls(d["stat"], d["subject"], d["params"], d["fill"], d["alpha"])


# ------------------------------------------------------------------ LightGBM


@dataclass
class LgbmRate:
    """LightGBM Poisson with a log-minutes offset."""

    stat: str
    subject: str
    booster: lgb.Booster
    features: list[str] = field(default_factory=list)
    alpha: float = 0.0
    kind: str = "lgbm"

    @property
    def use_minutes(self) -> bool:
        return self.subject == "player"

    @classmethod
    def fit(cls, train: pd.DataFrame, stat: str, subject: str, features: list[str],
            valid: pd.DataFrame | None = None, num_rounds: int | None = None) -> tuple["LgbmRate", int]:
        """Early-stop on ``valid`` when given, else train ``num_rounds`` rounds."""
        use_minutes = subject == "player"
        dtrain = lgb.Dataset(train[features], train[stat].to_numpy(float),
                             init_score=offset(train, use_minutes), free_raw_data=False)
        if valid is not None:
            dvalid = lgb.Dataset(valid[features], valid[stat].to_numpy(float),
                                 init_score=offset(valid, use_minutes), reference=dtrain)
            booster = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            rounds = booster.best_iteration
        else:
            rounds = int(num_rounds or 500)
            booster = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=rounds)
        return cls(stat, subject, booster, list(features)), rounds

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        """exp(raw tree score): the rate *without* the minutes offset."""
        return np.exp(self.booster.predict(frame[self.features], raw_score=True,
                                           num_iteration=self.booster.best_iteration or None))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / f"{self.subject}_{self.stat}_lgbm.txt"))
        (path / f"{self.subject}_{self.stat}_lgbm.json").write_text(json.dumps({
            "features": self.features, "alpha": self.alpha}))

    @classmethod
    def load(cls, path: Path, subject: str, stat: str) -> "LgbmRate":
        d = json.loads((path / f"{subject}_{stat}_lgbm.json").read_text())
        booster = lgb.Booster(model_file=str(path / f"{subject}_{stat}_lgbm.txt"))
        return cls(stat, subject, booster, d["features"], d["alpha"])


def expected_counts(model: GlmRate | LgbmRate, frame: pd.DataFrame, minutes: np.ndarray | None = None) -> np.ndarray:
    """Expected stat for each row given actual (or scenario) minutes; team models ignore minutes."""
    rate = model.predict_rate(frame)
    if not model.use_minutes:
        return rate
    m = frame["minutes"].to_numpy(float) if minutes is None else np.asarray(minutes, dtype=float)
    return rate * m

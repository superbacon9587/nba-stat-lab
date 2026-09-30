"""The comparison model: predict the outcome's quantiles directly.

Instead of assuming a count distribution, LightGBM fits one model per
quantile level (the 0.5% ... 99.5% points of the outcome) plus one for the
mean, with projected minutes as an ordinary feature. The predicted quantiles
are sorted (independently fitted quantile models can "cross") and joined into
a piecewise-linear CDF, from which probabilities and intervals are read.

It is trained on a random subsample of the training rows to keep training
time reasonable. Its scores are reported side by side with the count models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from nbalab.models.config import QUANTILES, SEED
from nbalab.models.distributions import cdf_from_quantiles

PARAMS: dict = {
    "learning_rate": 0.08, "num_leaves": 31, "min_data_in_leaf": 400, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": SEED,
}
ROUNDS = 300
MAX_TRAIN_ROWS = 150_000
MINUTES_FEATURE = "proj_minutes"


@dataclass
class QuantileModel:
    """One LightGBM booster per quantile level plus a mean booster."""

    stat: str
    subject: str
    boosters: dict[str, lgb.Booster]
    features: list[str] = field(default_factory=list)
    levels: tuple[float, ...] = QUANTILES
    alpha: float = 0.0  # unused; keeps the interface of the count models
    kind: str = "quantile"

    @property
    def use_minutes(self) -> bool:
        return self.subject == "player"

    @classmethod
    def fit(cls, train: pd.DataFrame, stat: str, subject: str, features: list[str]) -> "QuantileModel":
        """``train`` must already carry ``proj_minutes`` for player models."""
        feats = [*features, MINUTES_FEATURE] if subject == "player" else list(features)
        t = train.sample(min(len(train), MAX_TRAIN_ROWS), random_state=SEED)
        data = lgb.Dataset(t[feats], t[stat].to_numpy(float), free_raw_data=False)
        boosters = {"mean": lgb.train({**PARAMS, "objective": "regression"}, data, ROUNDS)}
        for q in QUANTILES:
            boosters[f"{q:g}"] = lgb.train({**PARAMS, "objective": "quantile", "alpha": q}, data, ROUNDS)
        return cls(stat, subject, boosters, feats)

    def predict_quantiles(self, frame: pd.DataFrame) -> np.ndarray:
        """(games, levels) sorted quantile predictions, floored at 0."""
        X = frame[self.features]
        q = np.column_stack([self.boosters[f"{lvl:g}"].predict(X) for lvl in self.levels])
        return np.sort(np.clip(q, 0, None), axis=1)

    def predict_mean(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(self.boosters["mean"].predict(frame[self.features]), 0, None)

    def pmf(self, frame: pd.DataFrame, support: np.ndarray) -> np.ndarray:
        return cdf_from_quantiles(np.asarray(self.levels), self.predict_quantiles(frame), support)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for name, b in self.boosters.items():
            b.save_model(str(path / f"{self.subject}_{self.stat}_q_{name}.txt"))
        (path / f"{self.subject}_{self.stat}_quantile.json").write_text(json.dumps({
            "features": self.features, "levels": list(self.levels), "names": list(self.boosters)}))

    @classmethod
    def load(cls, path: Path, subject: str, stat: str) -> "QuantileModel":
        d = json.loads((path / f"{subject}_{stat}_quantile.json").read_text())
        boosters = {n: lgb.Booster(model_file=str(path / f"{subject}_{stat}_q_{n}.txt")) for n in d["names"]}
        return cls(stat, subject, boosters, d["features"], tuple(d["levels"]))


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, levels: tuple[float, ...] = QUANTILES) -> dict[float, float]:
    """Average pinball (quantile) loss per level; lower is better.

    For level tau, a miss above the prediction costs tau per unit and a miss
    below costs (1 - tau): the loss that the tau-quantile minimizes.
    """
    y = np.asarray(y, dtype=float)[:, None]
    diff = y - q_pred
    lv = np.asarray(levels)[None, :]
    loss = np.maximum(lv * diff, (lv - 1) * diff).mean(axis=0)
    return {lvl: float(v) for lvl, v in zip(levels, loss)}

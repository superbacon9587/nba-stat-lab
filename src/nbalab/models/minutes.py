"""Minutes model: how long will the player be on the floor?

Minutes drive almost every counting stat, and they are the most uncertain part
of a projection (blowouts, foul trouble, early exits). The model has two parts:

1. A LightGBM regression predicts *expected* minutes from recent minutes,
   role (starter share), rest, schedule and team strength.
2. The uncertainty around that prediction is *empirical*: the model's own
   out-of-sample errors, grouped by predicted-minutes bucket (a 34-minute
   starter misses differently from a 12-minute bench player). The errors are
   summarized as K equally likely scenarios ("nodes"), so the stat
   distribution can be mixed over them exactly.

Projections are conditional on the player playing, so minutes are kept in
[1, 53]. A user override replaces the scenarios with that single value.
The spread can later be widened by split conformal calibration
(``nbalab.models.calibrate``), stored as per-bucket ``scales``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from nbalab.models.calibration import spread_nodes
from nbalab.models.config import SEED
from nbalab.models.features import MINUTES_FEATURES

N_NODES = 25
BUCKET_EDGES: tuple[float, ...] = (0, 10, 15, 20, 25, 30, 34, 38, 60)
MIN_MINUTES, MAX_MINUTES = 1.0, 53.0
FEATURES: tuple[str, ...] = (*MINUTES_FEATURES, "pos_code", "home")
PARAMS: dict = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbose": -1, "seed": SEED,
}


def node_levels(k: int = N_NODES) -> np.ndarray:
    """Midpoints of k equal-probability slices: (i + 0.5) / k."""
    return (np.arange(k) + 0.5) / k


def residual_table(pred: np.ndarray, actual: np.ndarray, edges: tuple[float, ...] = BUCKET_EDGES,
                   k: int = N_NODES) -> np.ndarray:
    """Quantiles of (actual - predicted) at the k node levels, one row per prediction bucket.

    Buckets with fewer than 200 games borrow the pooled quantiles.
    """
    resid = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    pooled = np.quantile(resid, node_levels(k))
    bucket = np.digitize(pred, edges[1:-1])
    rows = []
    for b in range(len(edges) - 1):
        r = resid[bucket == b]
        rows.append(np.quantile(r, node_levels(k)) if len(r) >= 200 else pooled)
    return np.vstack(rows)


@dataclass
class MinutesModel:
    """Expected minutes (LightGBM) plus empirical error scenarios."""

    booster: lgb.Booster
    residuals: np.ndarray  # (buckets, nodes)
    features: tuple[str, ...] = FEATURES
    edges: tuple[float, ...] = BUCKET_EDGES
    scales: np.ndarray | None = None  # conformal spread multipliers per bucket (None = 1)

    def with_scales(self, scales: np.ndarray | None) -> "MinutesModel":
        """Same model with a different scenario spread (``None`` = the raw 2023-24 errors)."""
        return MinutesModel(self.booster, self.residuals, self.features, self.edges, scales)

    @classmethod
    def fit(cls, train: pd.DataFrame, valid: pd.DataFrame | None = None, num_rounds: int | None = None
            ) -> tuple["MinutesModel", int]:
        """Fit on ``train``; early-stop and build the error table on ``valid`` if given.

        Returns the model and the number of boosting rounds used.
        """
        dtrain = lgb.Dataset(train[list(FEATURES)], train["minutes"], free_raw_data=False)
        if valid is not None:
            dvalid = lgb.Dataset(valid[list(FEATURES)], valid["minutes"], reference=dtrain)
            booster = lgb.train(PARAMS, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            rounds = booster.best_iteration
            pred = booster.predict(valid[list(FEATURES)], num_iteration=rounds)
            table = residual_table(pred, valid["minutes"].to_numpy())
        else:
            rounds = int(num_rounds or 500)
            booster = lgb.train(PARAMS, dtrain, num_boost_round=rounds)
            table = np.zeros((len(BUCKET_EDGES) - 1, N_NODES))
        return cls(booster, table), rounds

    def predict_mean(self, frame: pd.DataFrame) -> np.ndarray:
        """Point prediction of minutes (before the error scenarios)."""
        return self.booster.predict(frame[list(self.features)])

    def nodes(self, frame: pd.DataFrame) -> np.ndarray:
        """Minutes scenarios, shape (games, nodes), each equally likely."""
        scales = np.ones(len(self.residuals)) if self.scales is None else np.asarray(self.scales)
        return spread_nodes(self.predict_mean(frame), self.residuals, self.edges, scales, MIN_MINUTES, MAX_MINUTES)

    def distribution(self, frame: pd.DataFrame, override: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(nodes, weights). With ``override`` the minutes are known: a single node at that value."""
        if override is not None:
            return np.full((len(frame), 1), float(override)), np.ones(1)
        n = self.nodes(frame)
        return n, np.full(n.shape[1], 1.0 / n.shape[1])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "minutes.txt"))
        (path / "minutes.json").write_text(json.dumps({
            "residuals": self.residuals.tolist(), "features": list(self.features), "edges": list(self.edges),
            "scales": None if self.scales is None else list(map(float, self.scales)),
        }))

    @classmethod
    def load(cls, path: Path) -> "MinutesModel":
        meta = json.loads((path / "minutes.json").read_text())
        scales = meta.get("scales")
        return cls(lgb.Booster(model_file=str(path / "minutes.txt")), np.asarray(meta["residuals"]),
                   tuple(meta["features"]), tuple(meta["edges"]), None if scales is None else np.asarray(scales))

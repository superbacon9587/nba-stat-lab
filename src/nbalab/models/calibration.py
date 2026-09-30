"""Post-hoc calibration, fitted on one held-out season and checked on the next.

**Over/under probabilities.** If the model says 90% but such games go over
80% of the time, the raw probabilities are overconfident. A calibrator g is a
monotone map from raw to calibrated probability, fitted on raw P(over) vs.
what actually happened in the calibration season:

- *Platt scaling*: logistic regression on logit(p), a smooth S-shaped fix
  with two parameters.
- *Isotonic regression*: the best non-decreasing step function; more
  flexible, needs more data.
- *Identity*: no change, when neither helps.

The kind is picked per stat by 5-fold cross-validation **within** the
calibration season (folds split by player, so one player's games never sit
on both sides), then refitted on the whole season.

g is applied to the whole survival curve, S'(k) = g(P(X > k)). Because g is
non-decreasing, the calibrated P(over), P(under) and P(push) stay
non-negative and sum to 1 for every line.

**Minutes intervals (split conformal).** The minutes scenarios come from
2023-24 errors. On a later season their 90% range can cover less than 90%.
Within each predicted-minutes bucket, the scenario spread around its median
is multiplied by the factor lambda that makes the 90% range cover exactly
90% of calibration-season games.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from nbalab.models import metrics as M

KINDS: tuple[str, ...] = ("identity", "platt", "isotonic")
EPS = 1e-4


@dataclass
class ProbabilityCalibrator:
    """A monotone map from raw to calibrated probability."""

    kind: str = "identity"
    a: float = 1.0  # platt: calibrated = expit(a * logit(p) + b)
    b: float = 0.0
    x: tuple[float, ...] = ()  # isotonic knots, interpolated linearly
    y: tuple[float, ...] = ()
    raw_push: bool = False  # keep the model's own P(push) on whole-number lines (see calibrated_over_under_push)
    note: str = ""

    def __call__(self, p: np.ndarray | float) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        if self.kind == "platt":
            out = expit(self.a * logit(np.clip(p, EPS, 1 - EPS)) + self.b)
            return np.where(p <= 0, 0.0, np.where(p >= 1, 1.0, out))
        if self.kind == "isotonic":
            return np.interp(p, self.x, self.y)
        return p

    def to_dict(self) -> dict:
        return {"kind": self.kind, "a": self.a, "b": self.b, "x": list(self.x), "y": list(self.y),
                "raw_push": self.raw_push, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ProbabilityCalibrator":
        if not d:
            return cls()
        return cls(d["kind"], d.get("a", 1.0), d.get("b", 0.0), tuple(d.get("x", ())), tuple(d.get("y", ())),
                   bool(d.get("raw_push", False)), d.get("note", ""))


def fit_platt(p: np.ndarray, outcome: np.ndarray) -> ProbabilityCalibrator:
    z = logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))[:, None]
    lr = LogisticRegression(C=1e6).fit(z, np.asarray(outcome, int))
    return ProbabilityCalibrator("platt", float(lr.coef_[0, 0]), float(lr.intercept_[0]))


def fit_isotonic(p: np.ndarray, outcome: np.ndarray) -> ProbabilityCalibrator:
    """Isotonic fit, anchored at (0, 0) and (1, 1) so certain events stay certain."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, outcome)
    x = np.concatenate([[0.0], iso.X_thresholds_, [1.0]])
    y = np.concatenate([[0.0], iso.y_thresholds_, [1.0]])
    x, keep = np.unique(x, return_index=True)
    return ProbabilityCalibrator("isotonic", x=tuple(x.tolist()), y=tuple(np.maximum.accumulate(y[keep]).tolist()))


FITTERS = {"identity": lambda p, o: ProbabilityCalibrator(), "platt": fit_platt, "isotonic": fit_isotonic}


def cross_validated_brier(p: np.ndarray, outcome: np.ndarray, groups: np.ndarray, kind: str,
                          folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold Brier score of a calibrator kind; folds are whole groups (players)."""
    uniq = pd.unique(np.asarray(groups))
    fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % folds))
    fold = np.array([fold_of[g] for g in groups])
    pred = np.empty(len(p))
    for f in range(folds):
        test = fold == f
        pred[test] = FITTERS[kind](p[~test], outcome[~test])(p[test])
    return M.brier(pred, outcome)


def choose_calibrator(p: np.ndarray, outcome: np.ndarray, groups: np.ndarray) -> tuple[ProbabilityCalibrator, dict]:
    """Pick the kind with the best cross-validated Brier, refit it on everything."""
    p, outcome = np.asarray(p, float), np.asarray(outcome, float)
    scores = {k: cross_validated_brier(p, outcome, groups, k) for k in KINDS}
    best = min(scores, key=scores.get)
    return FITTERS[best](p, outcome), scores


def calibrated_over_under_push(cal: ProbabilityCalibrator, s_line: np.ndarray, s_below: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrated P(over), P(under), P(push) from raw survival values.

    ``s_line`` = raw P(X > floor(line)), ``s_below`` = raw P(X > ceil(line) - 1).
    For a half-line the two are equal and push is 0.

    With ``cal.raw_push`` the push probability is the raw model's
    (s_below - s_line), and the calibrated over and under are rescaled to share
    the remaining 1 - push in their calibrated ratio. Half-lines are unaffected.
    """
    s_line, s_below = np.asarray(s_line, dtype=float), np.asarray(s_below, dtype=float)
    over = cal(s_line)
    at_least = cal(s_below)
    under = 1.0 - at_least
    if not cal.raw_push:
        return over, under, at_least - over
    push = np.clip(s_below - s_line, 0.0, 1.0)
    total = over + under
    scale = np.where(total > 0, (1.0 - push) / np.where(total > 0, total, 1.0), 0.0)
    return over * scale, under * scale, push


# ------------------------------------------------------------------ minutes


def spread_nodes(pred: np.ndarray, residuals: np.ndarray, edges: tuple[float, ...], scales: np.ndarray,
                 lo: float, hi: float) -> np.ndarray:
    """Minutes scenarios with each bucket's spread around its median multiplied by ``scales[bucket]``."""
    bucket = np.digitize(pred, edges[1:-1])
    r = residuals[bucket]
    med = np.median(r, axis=1, keepdims=True)
    return np.clip(pred[:, None] + med + scales[bucket][:, None] * (r - med), lo, hi)


def node_interval(nodes: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed interval of equally likely scenarios (linear interpolation between them)."""
    tail = (1 - level) / 2
    return np.quantile(nodes, tail, axis=1), np.quantile(nodes, 1 - tail, axis=1)


def conformal_scales(pred: np.ndarray, actual: np.ndarray, residuals: np.ndarray, edges: tuple[float, ...],
                     lo: float, hi: float, level: float = 0.9, min_games: int = 500) -> np.ndarray:
    """Per-bucket spread multipliers giving ``level`` coverage on the calibration games.

    Coverage only grows as the spread widens, so each factor is found by
    bisection. Buckets with fewer than ``min_games`` games use the pooled factor.
    """
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    bucket = np.digitize(pred, edges[1:-1])
    n_buckets = residuals.shape[0]

    def solve(mask: np.ndarray) -> float:
        a, b = 0.25, 6.0
        for _ in range(40):
            mid = (a + b) / 2
            nodes = spread_nodes(pred[mask], residuals, edges, np.full(n_buckets, mid), lo, hi)
            if M.coverage(*node_interval(nodes, level), actual[mask]) < level:
                a = mid
            else:
                b = mid
        return b

    pooled = solve(np.ones(len(pred), dtype=bool))
    return np.array([solve(bucket == k) if (bucket == k).sum() >= min_games else pooled for k in range(n_buckets)])

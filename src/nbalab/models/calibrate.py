"""Fit the post-hoc calibration on one held-out season and store it in the bundle.

    python -m nbalab.models.calibrate

Uses only CALIBRATION_SEASON games (2024-25). The bundle was trained through
2023-24, and CALIBRATION_TEST_SEASON (2025-26) is left untouched for the
before/after comparison in ``nbalab.models.evaluate``. Steps:

1. Minutes: per-bucket conformal spread factors so the 90% minutes range
   covers 90% of 2024-25 games.
2. Over/under: raw P(over) on synthetic x.5 lines (with the widened minutes),
   then per stat the calibrator kind with the best player-grouped
   cross-validated Brier score (identity, Platt or isotonic), refitted on
   all 2024-25 games.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from nbalab.data.features import lagged_rolling_mean
from nbalab.models import config as C
from nbalab.models import metrics as M
from nbalab.models.calibration import ProbabilityCalibrator, choose_calibrator, conformal_scales, node_interval
from nbalab.models.minutes import MAX_MINUTES, MIN_MINUTES
from nbalab.models.predict import model_pmfs, outcome_support
from nbalab.models.registry import Bundle, SubjectModels, load_bundle, save_calibration
from nbalab.models.train import load_frames

log = logging.getLogger("nbalab.models.calibrate")
MIN_PRIOR_GAMES = 5
MINUTES_LEVEL = 0.9


def check_seasons(bundle: Bundle) -> None:
    train_end = int(bundle.manifest["train_seasons"][1])
    if not train_end < C.CALIBRATION_SEASON < C.CALIBRATION_TEST_SEASON:
        raise ValueError(f"need train end {train_end} < calibration {C.CALIBRATION_SEASON} "
                         f"< held-out {C.CALIBRATION_TEST_SEASON}")


def season_mask(frame: pd.DataFrame, by: str, season: int) -> np.ndarray:
    """Games of ``season`` with at least MIN_PRIOR_GAMES earlier games for the subject."""
    prior = frame.groupby(by, sort=False).cumcount()
    return (frame["season"].eq(season) & prior.ge(MIN_PRIOR_GAMES)).to_numpy()


def half_lines(frame: pd.DataFrame, stat: str, subject: str, by: str) -> np.ndarray:
    """Synthetic x.5 lines for every row (same rule as the backtest)."""
    last10 = lagged_rolling_mean(frame, by, stat, 10).to_numpy()
    if subject == "player":
        avg, n = (frame[f"{stat}_rate_std"] * frame["minutes_std"]).to_numpy(), frame["games_std"].to_numpy()
    else:
        avg, n = frame[f"{stat}_std"].to_numpy(), frame.groupby(["teamId", "season"], sort=False).cumcount().to_numpy()
    return M.synthetic_lines(avg, last10, n)[0]


def raw_over_probabilities(frame: pd.DataFrame, mask: np.ndarray, stat: str, sm: SubjectModels, by: str,
                           nodes: np.ndarray | None, weights: np.ndarray | None) -> pd.DataFrame:
    """Raw P(over the x.5 line) from the chosen model, with the outcome and the player/team id."""
    rows = frame[mask].reset_index(drop=True)
    y = rows[stat].to_numpy(float)
    support = outcome_support(y)
    pmf = model_pmfs(sm.model(stat), rows, support, nodes, weights)
    line = half_lines(frame, stat, sm.subject, by)[mask]
    over = (pmf * (support[None, :] > line[:, None])).sum(axis=1)
    return pd.DataFrame({"group": rows[by].to_numpy(), "p": over, "outcome": (y > line).astype(float)})


def fit_minutes(bundle: Bundle, pf: pd.DataFrame) -> dict:
    rows = pf[season_mask(pf, "personId", C.CALIBRATION_SEASON)].reset_index(drop=True)
    raw = bundle.minutes.with_scales(None)
    pred = raw.predict_mean(rows)
    actual = rows["minutes"].to_numpy(float)
    before = M.coverage(*node_interval(raw.nodes(rows), MINUTES_LEVEL), actual)
    scales = conformal_scales(pred, actual, raw.residuals, raw.edges, MIN_MINUTES, MAX_MINUTES, MINUTES_LEVEL)
    bundle.minutes.scales = scales
    after = M.coverage(*node_interval(bundle.minutes.nodes(rows), MINUTES_LEVEL), actual)
    log.info("minutes 90%% coverage on %d: %.3f -> %.3f (scales %s)", C.CALIBRATION_SEASON, before, after,
             np.round(scales, 2).tolist())
    return {"level": MINUTES_LEVEL, "scales": scales.tolist(), "coverage_before": before, "coverage_after": after,
            "n": int(len(rows))}


def fit_probabilities(bundle: Bundle, frame: pd.DataFrame, subject: str, by: str) -> dict:
    sm = bundle.subject(subject)
    mask = season_mask(frame, by, C.CALIBRATION_SEASON)
    nodes = weights = None
    if subject == "player":
        nodes, weights = bundle.minutes.distribution(frame[mask].reset_index(drop=True))
    out = {}
    for stat in sm.stats:
        d = raw_over_probabilities(frame, mask, stat, sm, by, nodes, weights)
        cal, scores = choose_calibrator(d["p"].to_numpy(), d["outcome"].to_numpy(), d["group"].to_numpy())
        if (subject, stat) in C.CALIBRATION_FORCE_IDENTITY:
            cal = ProbabilityCalibrator(note=f"forced to identity (CV picked {cal.kind}; it hurt on the 2025-26 check)")
        if (subject, stat) in C.CALIBRATION_RAW_PUSH:
            cal.raw_push = True
            cal.note = "calibrated over/under, raw push on whole-number lines, rescaled to sum to 1"
        out[stat] = {**cal.to_dict(), "cv_brier": scores, "n": int(len(d))}
        log.info("%s %s: %s (cv Brier %s)", subject, stat, cal.kind, {k: round(v, 5) for k, v in scores.items()})
    return out


def calibrate(bundle: Bundle | None = None) -> Bundle:
    bundle = bundle or load_bundle()
    check_seasons(bundle)
    _, _, pf, tf = load_frames()
    minutes = fit_minutes(bundle, pf)
    bundle.calibration = {
        "season": C.CALIBRATION_SEASON, "held_out_season": C.CALIBRATION_TEST_SEASON, "minutes": minutes,
        "player": fit_probabilities(bundle, pf, "player", "personId"),
        "team": fit_probabilities(bundle, tf, "team", "teamId"),
    }
    bundle.minutes.save(bundle.path)
    save_calibration(bundle)
    bundle.manifest["calibration"] = {
        "season": C.CALIBRATION_SEASON, "minutes_scales": minutes["scales"],
        "kinds": {s: {k: v["kind"] for k, v in bundle.calibration[s].items()} for s in ("player", "team")},
    }
    (bundle.path / "manifest.json").write_text(json.dumps(bundle.manifest, indent=2, default=str))
    log.info("saved calibration to %s", bundle.path)
    return bundle


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    calibrate()


if __name__ == "__main__":
    main()

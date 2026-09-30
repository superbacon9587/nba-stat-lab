"""Train the projection models and save a bundle.

    python -m nbalab.models.train [--version 1.0.0]

Steps (time-ordered, never shuffled):

1. **Selection**: fit every candidate on seasons TRAIN_START..VALIDATION-1 and
   score it on the VALIDATION season. The score is the average log
   probability the model gave to what actually happened, which rewards the
   whole distribution, not just the mean. It picks, per stat, GLM vs LightGBM
   vs quantile, NB vs Poisson, and the baseline's EWMA half-life.
2. **Refit**: refit every candidate on TRAIN_START..TRAIN_END (validation
   included) with the settings chosen in step 1, and save them all. The test
   seasons are never touched here.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nbalab.data.config import DATA_DIR
from nbalab.models import config as C
from nbalab.models.baseline import HALFLIVES, Baseline, pmf_rows
from nbalab.models.copula import correlation, normal_scores
from nbalab.models.defender import ON_COURT_STATS, league_tau2, pair_game_rows, pair_sums
from nbalab.models.features import build_player_frame, build_team_frame, player_feature_names, team_feature_names
from nbalab.models.minutes import MinutesModel
from nbalab.models.predict import conditional_pit, log_score, model_pmfs, outcome_support
from nbalab.models.quantile import MINUTES_FEATURE, QuantileModel
from nbalab.models.rate import GlmRate, LgbmRate, expected_counts, fit_dispersion
from nbalab.models.registry import Bundle, SubjectModels, save_bundle, version_tag
from nbalab.query.data import normalize_on_court

log = logging.getLogger("nbalab.models.train")
PROCESSED = DATA_DIR / "processed"
HISTORY_SEASONS = 2  # feature warm-up seasons before TRAIN_START


@dataclass(frozen=True)
class Split:
    """Seasons for selection (train_start..validation-1 -> validation) and the final refit (..train_end)."""

    train_start: int = C.TRAIN_START_SEASON
    validation: int = C.VALIDATION_SEASON
    train_end: int = C.TRAIN_END_SEASON


def model_frames(pg: pd.DataFrame, tg: pd.DataFrame, start: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_player_frame(pg, tg, start), build_team_frame(tg, start)


def load_frames(processed: Path = PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw player/team games plus the player and team model frames."""
    pg = pd.read_parquet(processed / "player_games.parquet")
    tg = pd.read_parquet(processed / "team_games.parquet")
    return pg, tg, *model_frames(pg, tg, C.TRAIN_START_SEASON - HISTORY_SEASONS)


def season_slice(frame: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    return frame[frame["season"].between(first, last)].reset_index(drop=True)


# ------------------------------------------------------------------ candidates


def fit_candidates(train: pd.DataFrame, valid: pd.DataFrame | None, stat: str, subject: str, features: list[str],
                   rounds: dict[str, int] | None = None) -> tuple[dict, dict[str, int]]:
    """GLM, LightGBM and quantile candidates for one stat. With ``valid``, LightGBM early-stops."""
    glm = GlmRate.fit(train, stat, subject)
    lgbm, n = LgbmRate.fit(train, stat, subject, features, valid=valid,
                           num_rounds=None if rounds is None else rounds["lgbm"])
    quant = QuantileModel.fit(train, stat, subject, features)
    return {"glm": glm, "lgbm": lgbm, "quantile": quant}, {"lgbm": n}


def score_candidates(models: dict, valid: pd.DataFrame, stat: str, nodes: np.ndarray | None,
                     weights: np.ndarray | None) -> dict[str, dict[str, float]]:
    """Fit alpha on validation (count models) and return log score and MAE per candidate."""
    y = valid[stat].to_numpy(float)
    support = outcome_support(y)
    out = {}
    for kind, m in models.items():
        if kind != "quantile":
            m.alpha = fit_dispersion(y, expected_counts(m, valid))
        pmf = model_pmfs(m, valid, support, nodes, weights)
        mean = pmf @ support
        out[kind] = {"log_score": float(log_score(pmf, support, y).mean()),
                     "mae": float(np.abs(mean - y).mean()), "alpha": float(m.alpha)}
    return out


def select_baseline(train: pd.DataFrame, valid_with_history: pd.DataFrame, valid_mask: np.ndarray,
                    by: str, stat: str) -> tuple[int, dict[int, float]]:
    """EWMA half-life with the best validation log score."""
    scores = {}
    for h in HALFLIVES:
        b = Baseline(by, {stat: Baseline.fit_stat(train, by, stat, h)})
        mean, alpha = b.predict(valid_with_history, stat)
        y = valid_with_history[stat].to_numpy(float)
        ok = valid_mask & np.isfinite(mean)
        support = outcome_support(y[ok])
        scores[h] = float(log_score(pmf_rows(mean[ok], alpha[ok], support), support, y[ok]).mean())
    return max(scores, key=scores.get), scores


# ------------------------------------------------------------------ subject pipelines


def add_projected_minutes(frame: pd.DataFrame, minutes: MinutesModel) -> pd.DataFrame:
    nodes, w = minutes.distribution(frame)
    return frame.assign(**{MINUTES_FEATURE: np.average(nodes, axis=1, weights=w)})


def train_subject(frame: pd.DataFrame, subject: str, stats: tuple[str, ...], features: list[str],
                  minutes_sel: MinutesModel | None, minutes_final: MinutesModel | None, by: str,
                  rng: np.random.Generator, split: Split = Split()) -> tuple[SubjectModels, dict]:
    """Select on the validation season, refit on everything up to ``split.train_end``."""
    sel_train = season_slice(frame, split.train_start, split.validation - 1)
    valid = season_slice(frame, split.validation, split.validation)
    final_train = season_slice(frame, split.train_start, split.train_end)
    upto_valid = frame[frame["season"].le(split.validation)].reset_index(drop=True)
    valid_mask = upto_valid["season"].eq(split.validation).to_numpy()
    nodes = weights = None
    if minutes_sel is not None:
        sel_train, final_train = add_projected_minutes(sel_train, minutes_sel), add_projected_minutes(final_train, minutes_final)
        valid = add_projected_minutes(valid, minutes_sel)
        nodes, weights = minutes_sel.distribution(valid)
    candidates, chosen, report, half = {}, {}, {}, {}
    pits = []
    for stat in stats:
        t0 = time.time()
        sel_models, rounds = fit_candidates(sel_train, valid, stat, subject, features)
        scores = score_candidates(sel_models, valid, stat, nodes, weights)
        best = max(scores, key=lambda k: scores[k]["log_score"])
        pits.append(normal_scores(conditional_pit(sel_models[best], valid, valid[stat].to_numpy(float), rng)))
        final, _ = fit_candidates(final_train, None, stat, subject, features,
                                  rounds={"lgbm": max(int(rounds["lgbm"] * 1.1), 50)})
        for kind, m in final.items():
            m.alpha = sel_models[kind].alpha
        h, h_scores = select_baseline(sel_train, upto_valid, valid_mask, by, stat)
        candidates[stat], chosen[stat], half[stat] = final, best, h
        report[stat] = {"validation": scores, "chosen": best, "lgbm_rounds": rounds["lgbm"],
                        "baseline_halflife": h, "baseline_validation_log_score": h_scores}
        log.info("%s %s: chose %s (%s) in %.0fs", subject, stat, best,
                 {k: round(v["log_score"], 4) for k, v in scores.items()}, time.time() - t0)
    baseline = Baseline(by, {s: Baseline.fit_stat(final_train, by, s, half[s]) for s in stats})
    league_corr = correlation(np.column_stack(pits))
    sm = SubjectModels(subject, candidates, chosen, baseline, features, list(stats), league_corr)
    return sm, report


def train_minutes(frame: pd.DataFrame, split: Split = Split()) -> tuple[MinutesModel, MinutesModel, dict]:
    sel_train = season_slice(frame, split.train_start, split.validation - 1)
    valid = season_slice(frame, split.validation, split.validation)
    final_train = season_slice(frame, split.train_start, split.train_end)
    sel, rounds = MinutesModel.fit(sel_train, valid)
    final, _ = MinutesModel.fit(final_train, num_rounds=int(rounds * 1.1))
    final.residuals = sel.residuals  # out-of-sample errors describe real uncertainty
    pred = sel.predict_mean(valid)
    mae = float(np.abs(pred - valid["minutes"]).mean())
    mae_last10 = float(np.abs(valid["minutes_last10"] - valid["minutes"]).mean())
    log.info("minutes: validation MAE %.2f (last-10 average %.2f), %d rounds", mae, mae_last10, rounds)
    return sel, final, {"validation_mae": mae, "validation_mae_last10": mae_last10, "rounds": rounds}


def defender_prior(pg: pd.DataFrame, on_court: pd.DataFrame | None, train_end: int) -> dict[str, float]:
    """League-wide spread of true on/off effects (tau^2) from training seasons only."""
    if on_court is None:
        return {}
    oc = normalize_on_court(on_court[on_court["season"].le(train_end)])
    box = pg[pg["season"].le(train_end) & pg["minutes"].gt(0)]
    rows = pair_game_rows(oc, box)
    return league_tau2(pair_sums(rows, list(ON_COURT_STATS)), list(ON_COURT_STATS))


def neutral_values(frame: pd.DataFrame, split: Split = Split()) -> dict:
    """League-average opponent starter size per position, for the 'neutral game'."""
    t = season_slice(frame, split.train_start, split.train_end)
    g = t.groupby("pos_code")[["opp_starter_height", "opp_starter_weight"]].mean()
    return {"starter_size": {str(int(k)): v.to_dict() for k, v in g.iterrows()}}


def data_fingerprint(processed: Path = PROCESSED) -> dict:
    out = {}
    for name in ("player_games", "team_games", "on_court"):
        p = processed / f"{name}.parquet"
        if p.exists():
            import pyarrow.parquet as pq

            out[name] = {"mtime": p.stat().st_mtime, "rows": pq.ParquetFile(p).metadata.num_rows}
    return out


def build_bundle(pg: pd.DataFrame, tg: pd.DataFrame, semver: str = "1.0.0", on_court: pd.DataFrame | None = None,
                 split: Split = Split(), data_info: dict | None = None,
                 player_stats: tuple[str, ...] = C.PLAYER_STATS, team_stats: tuple[str, ...] = C.TEAM_STATS) -> Bundle:
    """Fit everything from in-memory tables (no files written)."""
    assert split.train_end < min(C.TEST_SEASONS), "training must end before the test seasons"
    rng = np.random.default_rng(C.SEED)
    pf, tf = model_frames(pg, tg, split.train_start - HISTORY_SEASONS)
    minutes_sel, minutes_final, minutes_report = train_minutes(pf, split)
    player, player_report = train_subject(pf, "player", player_stats, player_feature_names(),
                                          minutes_sel, minutes_final, "personId", rng, split)
    team, team_report = train_subject(tf, "team", team_stats, team_feature_names(), None, None, "teamId", rng, split)
    tau2 = defender_prior(pg, on_court, split.train_end)
    log.info("defender prior tau^2: %s", tau2)
    today = date.today().isoformat()
    version = version_tag(semver, today)
    manifest = {
        "version": version, "created": today,
        "train_seasons": [split.train_start, split.train_end],
        "selection": {"train": [split.train_start, split.validation - 1], "validation": split.validation},
        "test_seasons": list(C.TEST_SEASONS),
        "player_features": player.features, "team_features": team.features,
        "chosen": {"player": player.chosen, "team": team.chosen},
        "minutes": minutes_report, "player": player_report, "team": team_report,
        "defender_tau2": tau2, "data": data_info or {},
    }
    return Bundle(version, manifest, minutes_final, player, team, tau2, neutral_values(pf, split))


def train(semver: str = "1.0.0", processed: Path = PROCESSED, root: Path = C.MODEL_DIR) -> Bundle:
    """Train on the processed parquet files and save the bundle under ``root``."""
    pg = pd.read_parquet(processed / "player_games.parquet")
    tg = pd.read_parquet(processed / "team_games.parquet")
    oc_path = processed / "on_court.parquet"
    on_court = pd.read_parquet(oc_path) if oc_path.exists() else None
    bundle = build_bundle(pg, tg, semver, on_court, data_info=data_fingerprint(processed))
    save_bundle(bundle, root)
    log.info("saved %s", bundle.path)
    return bundle


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", default="1.0.0", help="semantic version; the date is appended")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    train(args.version)


if __name__ == "__main__":
    main()

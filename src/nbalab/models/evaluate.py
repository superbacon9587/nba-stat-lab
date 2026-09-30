"""Backtest the saved bundle on the held-out test seasons and write the report.

    python -m nbalab.models.evaluate

Outputs: ``docs/model_report.md``, ``docs/img/*.png``, ``docs/model_metrics.json``
and ``docs/model_report_calibration.csv`` (read by the app's Methodology page).

The bundle was trained on seasons up to TRAIN_END and never saw the test
seasons; the script refuses to run otherwise. Test games need at least 5
earlier games for the player (or team) in the data so every method has
some history to work with.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nbalab.data.config import PROJECT_ROOT
from nbalab.data.features import lagged_rolling_mean
from nbalab.models import config as C
from nbalab.models import copula
from nbalab.models import metrics as M
from nbalab.models.baseline import pmf_rows
from nbalab.models.calibration import ProbabilityCalibrator, calibrated_over_under_push, node_interval
from nbalab.models.defender import ON_COURT_STATS, asof_multipliers, pair_game_rows
from nbalab.models.distributions import count_pmf, interval_rows, over_under_push_rows
from nbalab.models.predict import conditional_pit, log_score, model_pmfs, outcome_support
from nbalab.models.quantile import MINUTES_FEATURE, pinball_loss
from nbalab.models.registry import Bundle, SubjectModels, load_bundle
from nbalab.models.train import PROCESSED, load_frames
from nbalab.query.data import normalize_on_court
from nbalab.query.defenders import from_same_position_starter

log = logging.getLogger("nbalab.models.evaluate")
DOCS = PROJECT_ROOT / "docs"
IMG = DOCS / "img"
MIN_PRIOR_GAMES = 5
JOINT_ROWS = 3000
JOINT_DRAWS = 4000
JOINT_PAIRS: tuple[tuple[str, str], ...] = (("points", "assists"), ("points", "reboundsTotal"),
                                            ("reboundsTotal", "assists"))
LABELS = {"points": "Points", "assists": "Assists", "reboundsTotal": "Rebounds", "threePointersMade": "Threes",
          "steals": "Steals", "blocks": "Blocks", "turnovers": "Turnovers", "teamScore": "Team points"}


def assert_time_split(bundle: Bundle, test_seasons: tuple[int, ...] = C.TEST_SEASONS) -> None:
    """Refuse to backtest on any season the bundle was trained on."""
    train_end = int(bundle.manifest["train_seasons"][1])
    if not train_end < min(test_seasons):
        raise ValueError(f"bundle trained through {train_end}, which overlaps the test seasons {test_seasons}")


def test_mask(frame: pd.DataFrame, by: str) -> np.ndarray:
    prior = frame.groupby(by, sort=False).cumcount()
    return (frame["season"].isin(C.TEST_SEASONS) & prior.ge(MIN_PRIOR_GAMES)).to_numpy()


# ------------------------------------------------------------------ per-game scoring


@dataclass
class StatScores:
    stat: str
    games: pd.DataFrame  # one row per test game
    pmf_main: np.ndarray
    support: np.ndarray


def season_average(frame: pd.DataFrame, stat: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
    """Pre-game season-to-date average per game, and games played so far this season."""
    if subject == "player":
        return (frame[f"{stat}_rate_std"] * frame["minutes_std"]).to_numpy(), frame["games_std"].to_numpy()
    n = frame.groupby(["teamId", "season"], sort=False).cumcount().to_numpy()
    return frame[f"{stat}_std"].to_numpy(), n


def score_stat(frame: pd.DataFrame, mask: np.ndarray, stat: str, sm: SubjectModels, by: str,
               nodes: np.ndarray | None, weights: np.ndarray | None) -> StatScores:
    """Every model's predictions for one stat on the test games."""
    test = frame[mask].reset_index(drop=True)
    y = test[stat].to_numpy(float)
    support = outcome_support(y)
    last10 = lagged_rolling_mean(frame, by, stat, 10).to_numpy()
    avg, n_so_far = season_average(frame, stat, sm.subject)
    half_all, int_all = M.synthetic_lines(avg, last10, n_so_far)
    clim = M.climatology(frame[stat].to_numpy(float), half_all, frame[by].astype(str) + "_" + frame["season"].astype(str))
    g = pd.DataFrame({"season": test["season"], by: test[by], "y": y, "mean_last10": last10[mask],
                      "line_half": half_all[mask], "line_int": int_all[mask], "p_over_clim": clim[mask]})
    pmfs = {}
    for kind, model in sm.candidates[stat].items():
        pmf = model_pmfs(model, test, support, nodes, weights)
        g[f"mean_{kind}"] = pmf @ support
        if kind == sm.chosen[stat] or kind == "quantile":
            pmfs[kind] = pmf
    b_mean, b_alpha = sm.baseline.predict(frame, stat)
    pmfs["baseline"] = pmf_rows(np.nan_to_num(b_mean[mask], nan=np.nanmean(y)), b_alpha[mask], support)
    pmfs["main"] = pmfs[sm.chosen[stat]]
    g["mean_main"] = g[f"mean_{sm.chosen[stat]}"]
    g["mean_baseline"] = pmfs["baseline"] @ support
    for name in ("main", "baseline", "quantile"):
        pmf = pmfs[name]
        g[f"p_over_{name}"], _, _ = over_under_push_rows(support, pmf, g["line_half"].to_numpy())
        o, u, p = over_under_push_rows(support, pmf, g["line_int"].to_numpy())
        g[f"p_over_int_{name}"], g[f"p_push_int_{name}"] = o, p
        g[f"logscore_{name}"] = log_score(pmf, support, y)
        for lvl in C.INTERVAL_LEVELS:
            g[f"pi{lvl}_lo_{name}"], g[f"pi{lvl}_hi_{name}"] = interval_rows(support, pmf, lvl / 100)
    g["s_below_int_main"] = g["p_over_int_main"] + g["p_push_int_main"]  # raw P(X > line - 1)
    q = sm.candidates[stat]["quantile"]
    qf = test.assign(**{MINUTES_FEATURE: np.average(nodes, axis=1, weights=weights)}) if q.use_minutes else test
    qp = q.predict_quantiles(qf)
    for i, lvl in enumerate(q.levels):
        g[f"q{lvl:g}"] = qp[:, i]
    return StatScores(stat, g, pmfs["main"], support)


# ------------------------------------------------------------------ summaries


def point_summary(g: pd.DataFrame, by: str, kinds: list[str]) -> dict:
    out = {"n": int(len(g))}
    for name in ["main", "baseline", "last10", *kinds]:
        out[f"mae_{name}"] = M.mae(g[f"mean_{name}"], g["y"])
        out[f"rmse_{name}"] = M.rmse(g[f"mean_{name}"], g["y"])
    err = {k: np.abs(g[f"mean_{k}"] - g["y"]) for k in ("main", "baseline", "last10")}
    out["mae_diff_vs_baseline"] = M.paired_bootstrap_diff(err["main"], err["baseline"], g[by])
    out["mae_diff_vs_last10"] = M.paired_bootstrap_diff(err["main"], err["last10"], g[by])
    return out


def probability_summary(g: pd.DataFrame) -> dict:
    over = (g["y"] > g["line_half"]).astype(float)
    out = {"base_rate_over": float(over.mean())}
    for name in ("main", "baseline", "quantile", "clim"):
        out[f"brier_{name}"] = M.brier(g[f"p_over_{name}"], over)
        out[f"logloss_{name}"] = M.log_loss(g[f"p_over_{name}"], over)
    out["logscore_main"] = float(g["logscore_main"].mean())
    out["logscore_baseline"] = float(g["logscore_baseline"].mean())
    push = (g["y"] == g["line_int"]).astype(float)
    out["push_actual"] = float(push.mean())
    out["push_pred_main"] = float(g["p_push_int_main"].mean())
    out["push_pred_baseline"] = float(g["p_push_int_baseline"].mean())
    return out


def coverage_summary(g: pd.DataFrame) -> dict:
    out = {}
    for name in ("main", "baseline", "quantile"):
        for lvl in C.INTERVAL_LEVELS:
            lo, hi = g[f"pi{lvl}_lo_{name}"], g[f"pi{lvl}_hi_{name}"]
            out[f"cov{lvl}_{name}"] = M.coverage(lo, hi, g["y"])
            out[f"width{lvl}_{name}"] = float((hi - lo).mean())
    return out


def pinball_summary(g: pd.DataFrame, levels: tuple[float, ...]) -> dict[str, float]:
    q = np.column_stack([g[f"q{lvl:g}"] for lvl in levels])
    return {f"{k:g}": v for k, v in pinball_loss(g["y"].to_numpy(), q, levels).items()}


def summarize(scores: StatScores, sm: SubjectModels, by: str) -> dict:
    g = scores.games
    kinds = list(sm.candidates[scores.stat])
    per_season = {int(s): {**point_summary(d, by, kinds), **probability_summary(d), **coverage_summary(d)}
                  for s, d in g.groupby("season")}
    return {
        "chosen": sm.chosen[scores.stat],
        "all": {**point_summary(g, by, kinds), **probability_summary(g), **coverage_summary(g)},
        "by_season": per_season,
        "pinball_quantile": pinball_summary(g, sm.candidates[scores.stat]["quantile"].levels),
        "calibration": M.calibration_bins(g["p_over_main"], (g["y"] > g["line_half"]).astype(float)).to_dict("list"),
    }


# ------------------------------------------------------------------ joint probabilities


def pit_scores(frame: pd.DataFrame, sm: SubjectModels, stats: list[str], rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([copula.normal_scores(conditional_pit(sm.model(s), frame, frame[s].to_numpy(float), rng))
                            for s in stats])


def joint_backtest(frame: pd.DataFrame, mask: np.ndarray, bundle: Bundle, rng: np.random.Generator) -> dict:
    """Brier of P(both overs) with the copula vs. treating stats as independent (minutes still shared)."""
    sm = bundle.player
    stats = sorted({s for pair in JOINT_PAIRS for s in pair}, key=list(sm.corr_stats).index)
    idx_league = [sm.corr_stats.index(s) for s in stats]
    league = sm.league_corr[np.ix_(idx_league, idx_league)]
    z = pit_scores(frame, sm, stats, rng)
    test_idx = np.flatnonzero(mask)
    pick = np.sort(rng.choice(test_idx, size=min(JOINT_ROWS, len(test_idx)), replace=False))
    rows = frame.iloc[pick].reset_index(drop=True)
    nodes, w = bundle.minutes.distribution(rows)
    rates = {s: sm.model(s).predict_rate(rows) for s in stats}
    corr_cache: dict[tuple[int, int], np.ndarray] = {}
    person = frame["personId"].to_numpy()
    season = frame["season"].to_numpy()
    probs = {("copula", a, b): [] for a, b in JOINT_PAIRS} | {("independent", a, b): [] for a, b in JOINT_PAIRS}
    outcomes = {(a, b): [] for a, b in JOINT_PAIRS}
    for i, r in rows.iterrows():
        key = (int(r["personId"]), int(r["season"]))
        if key not in corr_cache:
            hist = (person == key[0]) & (season < key[1]) & (season >= key[1] - 3)
            corr_cache[key] = copula.shrink_correlation(copula.correlation(z[hist]), league, int(hist.sum()))
        tables = {}
        lines = {}
        for s in stats:
            means = nodes[i] * rates[s][i]
            alpha = sm.model(s).alpha
            support = np.arange(int(means.max() * 2 + 30))
            tables[s] = np.cumsum(count_pmf(means, alpha, support), axis=1)
            avg = r[f"{s}_rate_std"] * r["minutes_std"] if r["games_std"] >= MIN_PRIOR_GAMES else r[f"{s}_rate_ewm20"] * r["minutes_ewm20"]
            lines[s] = np.floor(avg) + 0.5 if np.isfinite(avg) else 0.5
        seed = int(rng.integers(1 << 31))
        for kind, corr in (("copula", corr_cache[key]), ("independent", np.eye(len(stats)))):
            draws = copula.simulate(w, tables, corr, JOINT_DRAWS, np.random.default_rng(seed))
            for a, b in JOINT_PAIRS:
                probs[(kind, a, b)].append(copula.joint_probability(draws, {a: (">", lines[a]), b: (">", lines[b])}))
        for a, b in JOINT_PAIRS:
            outcomes[(a, b)].append(float(r[a] > lines[a] and r[b] > lines[b]))
    out = {"n_games": len(rows), "draws": JOINT_DRAWS}
    for a, b in JOINT_PAIRS:
        o = np.asarray(outcomes[(a, b)])
        out[f"{a}+{b}"] = {"actual_rate": float(o.mean()),
                           **{f"brier_{k}": M.brier(np.asarray(probs[(k, a, b)]), o) for k in ("copula", "independent")},
                           **{f"mean_pred_{k}": float(np.mean(probs[(k, a, b)])) for k in ("copula", "independent")}}
    return out


# ------------------------------------------------------------------ defender ablation


def defender_ablation(frame: pd.DataFrame, mask: np.ndarray, pg: pd.DataFrame, bundle: Bundle,
                      nodes: np.ndarray, weights: np.ndarray, processed: Path = PROCESSED) -> dict:
    """Does the on/off defender proxy improve predictions when Y = opponent's same-position starter?"""
    path = processed / "on_court.parquet"
    if not path.exists() or not bundle.defender_tau2:
        return {"skipped": "no on_court table or defender prior"}
    test = frame[mask].reset_index(drop=True)
    starters = test[test["starter"].astype(bool)].copy()
    opp = pg.merge(starters[["gameId", "opponentTeamId"]].drop_duplicates().rename(columns={"opponentTeamId": "teamId"}),
                   on=["gameId", "teamId"])
    ydef = from_same_position_starter(starters, opp)[["gameId", "personId", "def_person_id"]]
    keys = starters.reset_index().merge(ydef, on=["gameId", "personId"])
    keys = keys.rename(columns={"def_person_id": "opponentPersonId"}).astype({"opponentPersonId": "int64"})
    oc = pd.read_parquet(path)
    oc = normalize_on_court(oc.merge(keys[["personId", "opponentPersonId"]].drop_duplicates(),
                                     on=["personId", "opponentPersonId"]))
    rows = pair_game_rows(oc, pg[pg["minutes"].gt(0)])
    mult = asof_multipliers(keys[["personId", "opponentPersonId", "gameDateTimeEst"]], rows, bundle.defender_tau2)
    sub = test.iloc[keys["index"].to_numpy()].reset_index(drop=True)
    sub_nodes = nodes[keys["index"].to_numpy()]
    out = {"n_games": int(len(sub)), "share_with_history": float((mult.iloc[:, 0] != 1.0).mean())}
    for stat in ON_COURT_STATS:
        model = bundle.player.model(stat)
        y = sub[stat].to_numpy(float)
        support = outcome_support(y)
        with_ = model_pmfs(model, sub, support, sub_nodes, weights, multiplier=mult[f"{stat}_mult"].to_numpy())
        without = model_pmfs(model, sub, support, sub_nodes, weights)
        out[stat] = {
            "mae_without": M.mae(without @ support, y), "mae_with": M.mae(with_ @ support, y),
            "logscore_without": float(log_score(without, support, y).mean()),
            "logscore_with": float(log_score(with_, support, y).mean()),
            "mean_abs_adjustment_pct": float(np.abs(mult[f"{stat}_mult"] - 1).mean() * 100),
        }
    return out


# ------------------------------------------------------------------ charts and report


def calibration_png(before: dict, after: dict | None, stat: str, path: Path, season: int) -> None:
    """Reliability diagram on the held-out season: raw probabilities and, if given, calibrated ones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=120)
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, label="perfect calibration")
    series = [("raw model", before, "#d95926")] + ([("calibrated", after, "#3987e5")] if after is not None else [])
    for label, cal, color in series:
        n = np.asarray(cal["n"], float)
        ax.plot(cal["bin_pred"], cal["bin_actual"], color=color, lw=2, marker="o", label=label)
        ax.scatter(cal["bin_pred"], cal["bin_actual"], s=20 + 160 * n / n.max(), color=color, alpha=0.3)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(over)"), ax.set_ylabel("Actual over rate")
    ax.set_title(f"{LABELS.get(stat, stat)}: {season_label(season)} (held out)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stat_label(subject: str, stat: str) -> str:
    label = LABELS.get(stat, stat)
    return label if subject == "player" or label.startswith("Team") else f"Team {label.lower()}"


def season_label(season: int) -> str:
    return f"{season}-{str(season + 1)[-2:]}"


def fmt(x: float, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci_text(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def verdict_rows(results: dict) -> list[str]:
    """One line per stat: did the main model beat the baseline on MAE, Brier and log score?"""
    rows = ["| Subject | Stat | Model | MAE vs baseline (95% CI) | Brier model / baseline | Log score model / baseline | Verdict |",
            "|---|---|---|---|---|---|---|"]
    for subject in ("player", "team"):
        for stat, r in results[subject].items():
            a = r["all"]
            d, lo, hi = a["mae_diff_vs_baseline"]
            wins = [hi < 0, a["brier_main"] < a["brier_baseline"], a["logscore_main"] > a["logscore_baseline"]]
            verdict = ("beats baseline" if all(wins) else "mixed" if any(wins) else "does not beat baseline")
            if lo <= 0 <= hi:
                verdict += " (MAE gap within noise)"
            rows.append(f"| {subject} | {LABELS.get(stat, stat)} | {r['chosen']} | {ci_text(a['mae_diff_vs_baseline'])} | "
                        f"{fmt(a['brier_main'], 4)} / {fmt(a['brier_baseline'], 4)} | "
                        f"{fmt(a['logscore_main'])} / {fmt(a['logscore_baseline'])} | {verdict} |")
    return rows


def write_report(results: dict, bundle: Bundle) -> str:
    L: list[str] = []
    m = bundle.manifest
    sel = m["selection"]
    L += [f"# Projection model report ({bundle.version})", "",
          f"Trained on {season_label(m['train_seasons'][0])} through {season_label(m['train_seasons'][1])}. Model "
          f"choices were made on {season_label(sel['validation'])}, held out from a fit on "
          f"{season_label(sel['train'][0])} through {season_label(sel['train'][1])}. Tested on "
          + " and ".join(season_label(s) for s in C.TEST_SEASONS) + ", which the models never saw.", "",
          "**No sportsbook lines exist in this dataset.** Every line below is synthetic: the player's pre-game "
          "season-to-date average (or last-10 average before 5 games) rounded to x.5, plus a whole-number version "
          "to test pushes. Real market lines are sharper, so these probability scores are a proxy for how the "
          "model would do against a market.", "",
          "## Verdict", "",
          "Main model = the model chosen per stat on the validation season. Baseline = recency-weighted average x "
          "opponent defensive rating x home factor, negative binomial. MAE difference is model minus baseline, "
          "with a 95% bootstrap CI that resamples whole players (negative = model better).", "",
          *verdict_rows(results), ""]
    L += ["## Point accuracy (MAE of the expected value)", "",
          "| Stat | Season | Games | Main | Baseline | Last-10 avg | GLM | LightGBM | Quantile | Main - last-10 (95% CI) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for subject in ("player", "team"):
        for stat, r in results[subject].items():
            for season, s in [*r["by_season"].items(), ("all", r["all"])]:
                L.append(f"| {LABELS.get(stat, stat)} | {season} | {s['n']} | {fmt(s['mae_main'])} | "
                         f"{fmt(s['mae_baseline'])} | {fmt(s['mae_last10'])} | {fmt(s['mae_glm'])} | "
                         f"{fmt(s['mae_lgbm'])} | {fmt(s['mae_quantile'])} | {ci_text(s['mae_diff_vs_last10'])} |")
    L += ["", "## Over/under probabilities (synthetic x.5 lines, raw model)", "",
          "Raw model probabilities, before the calibration map described further down.", "",
          "Brier: lower is better (0.25 = always saying 50%). Climatology = the player's hit rate on this line in "
          "earlier games this season.", "",
          "| Stat | Over rate | Brier main | Brier baseline | Brier quantile | Brier climatology | Log loss main | Log loss baseline |",
          "|---|---|---|---|---|---|---|---|"]
    for subject in ("player", "team"):
        for stat, r in results[subject].items():
            a = r["all"]
            L.append(f"| {LABELS.get(stat, stat)} | {fmt(a['base_rate_over'])} | {fmt(a['brier_main'], 4)} | "
                     f"{fmt(a['brier_baseline'], 4)} | {fmt(a['brier_quantile'], 4)} | {fmt(a['brier_clim'], 4)} | "
                     f"{fmt(a['logloss_main'], 4)} | {fmt(a['logloss_baseline'], 4)} |")
    L += ["", "### Pushes on whole-number lines", "", "| Stat | Actual push rate | Predicted (main) | Predicted (baseline) |",
          "|---|---|---|---|"]
    for subject in ("player", "team"):
        for stat, r in results[subject].items():
            a = r["all"]
            L.append(f"| {LABELS.get(stat, stat)} | {fmt(a['push_actual'])} | {fmt(a['push_pred_main'])} | "
                     f"{fmt(a['push_pred_baseline'])} |")
    L += ["", "## Prediction interval coverage and width", "",
          "Count stats are whole numbers, so an interval with integer ends covers at least its nominal level; "
          "coverage well above nominal means the interval is wider than it needs to be.", "",
          "| Stat | Model | 90% cover | 95% cover | 99% cover | 90% width | 95% width | 99% width |",
          "|---|---|---|---|---|---|---|---|"]
    for subject in ("player", "team"):
        for stat, r in results[subject].items():
            a = r["all"]
            for name in ("main", "baseline", "quantile"):
                L.append(f"| {LABELS.get(stat, stat)} | {name} | "
                         + " | ".join(fmt(a[f"cov{lvl}_{name}"]) for lvl in C.INTERVAL_LEVELS) + " | "
                         + " | ".join(fmt(a[f"width{lvl}_{name}"], 1) for lvl in C.INTERVAL_LEVELS) + " |")
    L += ["", "### Quantile model pinball loss (lower is better)", "", "| Stat | " +
          " | ".join(f"q{q:g}" for q in C.QUANTILES) + " |", "|---|" + "---|" * len(C.QUANTILES)]
    for subject in ("player", "team"):
        for stat, r in results[subject].items():
            L.append(f"| {LABELS.get(stat, stat)} | " + " | ".join(fmt(v) for v in r["pinball_quantile"].values()) + " |")
    L += calibration_section(results)
    L += minutes_section(results["minutes"])
    j = results["joint"]
    L += ["", "## Joint probabilities (both overs hit)", "",
          f"{j['n_games']} random test games, {j['draws']} simulated games each. Both methods share the minutes "
          "draw; 'independent' drops the copula correlation.", "",
          "| Pair | Actual rate | Mean pred (copula) | Mean pred (independent) | Brier copula | Brier independent |",
          "|---|---|---|---|---|---|"]
    for a, b in JOINT_PAIRS:
        r = j[f"{a}+{b}"]
        L.append(f"| {LABELS[a]} + {LABELS[b]} | {fmt(r['actual_rate'])} | {fmt(r['mean_pred_copula'])} | "
                 f"{fmt(r['mean_pred_independent'])} | {fmt(r['brier_copula'], 4)} | {fmt(r['brier_independent'], 4)} |")
    d = results["defender"]
    L += ["", "## Defender proxy ablation", ""]
    if "skipped" in d:
        L.append(f"Skipped: {d['skipped']}.")
    else:
        L += [f"Y = the opponent's starter at the subject's starting position, in {d['n_games']} test games where the "
              f"subject started ({fmt(d['share_with_history'])} had earlier shared floor time with Y). The adjustment "
              "uses only games before each test game.", "",
              "| Stat | MAE without | MAE with | Log score without | Log score with | Mean adjustment |",
              "|---|---|---|---|---|---|"]
        helped = []
        for stat in ON_COURT_STATS:
            r = d[stat]
            helped.append(r["logscore_with"] > r["logscore_without"] and r["mae_with"] < r["mae_without"])
            L.append(f"| {LABELS[stat]} | {fmt(r['mae_without'])} | {fmt(r['mae_with'])} | {fmt(r['logscore_without'], 4)} "
                     f"| {fmt(r['logscore_with'], 4)} | {fmt(r['mean_abs_adjustment_pct'], 2)}% |")
        L += ["", "The proxy improves both scores for every tracked stat." if all(helped) else
              "The proxy does **not** consistently improve predictions, so the app shows it as a descriptive, "
              "clearly labeled adjustment and not as a validated effect."]
    L += ["", "## Validation-season model choice", "", "| Subject | Stat | GLM | LightGBM | Quantile | Chosen | α (NB) |",
          "|---|---|---|---|---|---|---|"]
    for subject in ("player", "team"):
        for stat, r in m[subject].items():
            v = r["validation"]
            L.append(f"| {subject} | {LABELS.get(stat, stat)} | {fmt(v['glm']['log_score'], 4)} | "
                     f"{fmt(v['lgbm']['log_score'], 4)} | {fmt(v['quantile']['log_score'], 4)} | {r['chosen']} | "
                     f"{fmt(v[r['chosen']]['alpha'], 3)} |")
    L += ["", "Scores are mean log probability of the observed count (higher is better). α = 0 means the likelihood-"
          "ratio test preferred Poisson.", "",
          "## Method notes", "",
          "- Projections are conditional on the player playing (a DNP voids a prop).",
          "- Minutes: LightGBM for the expected value, empirical out-of-sample errors (by predicted-minutes bucket) "
          "for the 25 scenarios.",
          "- Rates: log link with log(minutes) offset, so projections scale exactly with minutes.",
          "- Opponent same-position starters are treated as known before tip-off (lineups are announced).",
          "- Joint probabilities use a Gaussian copula on randomized PIT residuals, shrunk toward the league matrix "
          f"with weight n/(n+{int(copula.SHRINK_GAMES)}).",
          "- Named-defender effects are an on/off floor-time **proxy**, not 'guarded by' data.",
          "- This is a statistical estimate for learning and exploration, not betting advice."]
    return "\n".join(L) + "\n"


def calibration_section(results: dict) -> list[str]:
    fit, test = season_label(C.CALIBRATION_SEASON), season_label(C.CALIBRATION_TEST_SEASON)
    L = ["", "## Probability calibration", "",
         f"The raw over/under probabilities pass through a calibration map fitted on **{fit} only** and judged "
         f"on **{test} only**, so the comparison below is out of sample. For each stat the map is Platt scaling, "
         f"isotonic regression, or no change (identity), whichever had the best Brier score in 5-fold "
         f"cross-validation within {fit} (folds split by player). The map is applied to the whole survival curve, "
         "so over, under and push stay consistent for every line. Composite stats (PRA etc.) are not recalibrated.", "",
         f"| Stat | Map | Games ({test}) | Brier raw | Brier calibrated | Log loss raw | Log loss calibrated | "
         "ECE raw | ECE calibrated | Push actual / raw / calibrated |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in results["calibration"].items():
        subject, stat = key.split(":")
        name = stat_label(subject, stat)
        kind = c["kind"] + (" + raw push" if c.get("raw_push") else "") + (" (forced)" if "forced" in c.get("note", "") else "")
        L.append(f"| {name} | {kind} | {c['n']} | {fmt(c['brier_raw'], 4)} | {fmt(c['brier_cal'], 4)} | "
                 f"{fmt(c['logloss_raw'], 4)} | {fmt(c['logloss_cal'], 4)} | {fmt(c['ece_raw'], 4)} | "
                 f"{fmt(c['ece_cal'], 4)} | {fmt(c['push_actual'])} / {fmt(c['push_raw'])} / {fmt(c['push_cal'])} |")
    notes = [f"{stat_label(*k.split(':'))}: {c['note']}" for k, c in results["calibration"].items() if c.get("note")]
    if notes:
        L += ["", "Manual overrides (chosen after seeing the " + test + " check, so for these two decisions " + test +
              " is not untouched): " + "; ".join(notes) + "."]
    helped = [c["brier_cal"] < c["brier_raw"] for c in results["calibration"].values() if c["kind"] != "identity"]
    changed = sum(c["kind"] != "identity" for c in results["calibration"].values())
    L += ["", f"ECE = expected calibration error, the size-weighted average gap between predicted and actual rates "
          f"across 10 buckets. Of the {changed} stats with a non-identity map, {sum(helped)} improved on {test}.", "",
          f"Reliability diagrams on {test} (orange = raw model, blue = after calibration; stats where cross-validation "
          "kept the identity map show only the raw line):", ""]
    for key in results["calibration"]:
        subject, stat = key.split(":")
        if subject == "player" or stat == "teamScore":
            L.append(f"![{LABELS.get(stat, stat)} calibration](img/calibration_{stat}.png)")
    return L


def minutes_section(mn: dict) -> list[str]:
    fit, test = season_label(C.CALIBRATION_SEASON), season_label(C.CALIBRATION_TEST_SEASON)
    L = ["", "## Minutes model", "", f"MAE of projected minutes: **{fmt(mn['mae_model'], 2)}** vs "
         f"{fmt(mn['mae_last10'], 2)} for the last-10 average ({mn['n']} test games).", "",
         "The minutes scenarios originally came from 2023-24 errors and covered too little on later seasons. A split "
         f"conformal adjustment fitted on **{fit}** multiplies each predicted-minutes bucket's spread by a factor "
         "(" + ", ".join(f"{x:.2f}" for x in (mn["scales"] or [])) + f") so the 90% range covers 90% there. {test} "
         "is the honest check.", "",
         "| Season | Games | 90% coverage before | 90% coverage after | Mean width before | Mean width after |",
         "|---|---|---|---|---|---|"]
    for season, r in mn["by_season"].items():
        tag = " (fit)" if season == C.CALIBRATION_SEASON else " (held out)"
        L.append(f"| {season_label(season)}{tag} | {r['n']} | {fmt(r['cov90_before'])} | {fmt(r['cov90_after'])} | "
                 f"{fmt(r['width90_before'], 1)} min | {fmt(r['width90_after'], 1)} min |")
    L += ["", f"Every stat distribution in this report is mixed over the widened minutes scenarios, so {fit} stat "
          f"results use a spread tuned on that same season; {test} is fully out of sample."]
    return L


def app_metrics(results: dict) -> dict:
    """Compact headline numbers for the Methodology page."""
    out = {}
    for subject in ("player", "team"):
        for stat, r in results[subject].items():
            a = r["all"]
            out[f"{subject} {LABELS.get(stat, stat)}"] = {
                "model": r["chosen"], "MAE model": round(a["mae_main"], 3), "MAE baseline": round(a["mae_baseline"], 3),
                "Brier model": round(a["brier_main"], 4), "Brier baseline": round(a["brier_baseline"], 4),
                "95% PI coverage": round(a["cov95_main"], 3),
            }
            c = results["calibration"][f"{subject}:{stat}"]
            out[f"{subject} {LABELS.get(stat, stat)}"] |= {
                "calibration": c["kind"], f"Brier raw {season_label(C.CALIBRATION_TEST_SEASON)}": round(c["brier_raw"], 4),
                f"Brier calibrated {season_label(C.CALIBRATION_TEST_SEASON)}": round(c["brier_cal"], 4)}
    out["minutes MAE"] = {"model": round(results["minutes"]["mae_model"], 2),
                          "last-10 average": round(results["minutes"]["mae_last10"], 2)}
    out["minutes 90% coverage"] = {season_label(s): round(r["cov90_after"], 3)
                                   for s, r in results["minutes"]["by_season"].items()}
    return out


def minutes_backtest(pf: pd.DataFrame, mask: np.ndarray, bundle: Bundle) -> dict:
    """Minutes MAE, and 90% range coverage per test season before/after the conformal widening."""
    test = pf[mask].reset_index(drop=True)
    pred = bundle.minutes.predict_mean(test)
    actual = test["minutes"].to_numpy()
    raw = bundle.minutes.with_scales(None)
    before = node_interval(raw.nodes(test), 0.9)
    after = node_interval(bundle.minutes.nodes(test), 0.9)
    by_season = {}
    for season in C.TEST_SEASONS:
        m = test["season"].eq(season).to_numpy()
        by_season[int(season)] = {
            "n": int(m.sum()),
            "cov90_before": M.coverage(before[0][m], before[1][m], actual[m]),
            "cov90_after": M.coverage(after[0][m], after[1][m], actual[m]),
            "width90_before": float((before[1] - before[0])[m].mean()),
            "width90_after": float((after[1] - after[0])[m].mean()),
        }
    return {"n": int(len(test)), "mae_model": M.mae(pred, actual),
            "mae_last10": M.mae(test["minutes_last10"].fillna(test["minutes_ewm20"]), actual),
            "cov90": M.coverage(*after, actual), "by_season": by_season,
            "scales": list(map(float, bundle.minutes.scales)) if bundle.minutes.scales is not None else None}


def calibration_backtest(sc: StatScores, cal_dict: dict | None) -> dict:
    """Raw vs calibrated over/under probabilities on the held-out calibration test season."""
    g = sc.games[sc.games["season"].eq(C.CALIBRATION_TEST_SEASON)]
    cal = ProbabilityCalibrator.from_dict(cal_dict)
    over = (g["y"] > g["line_half"]).astype(float).to_numpy()
    raw = g["p_over_main"].to_numpy()
    fixed = cal(raw)
    o_int, _, push_int = calibrated_over_under_push(cal, g["p_over_int_main"].to_numpy(), g["s_below_int_main"].to_numpy())
    return {
        "kind": cal.kind, "raw_push": cal.raw_push, "note": cal.note, "n": int(len(g)), "cv_brier_2024": (cal_dict or {}).get("cv_brier"),
        "brier_raw": M.brier(raw, over), "brier_cal": M.brier(fixed, over),
        "logloss_raw": M.log_loss(raw, over), "logloss_cal": M.log_loss(fixed, over),
        "ece_raw": expected_calibration_error(raw, over), "ece_cal": expected_calibration_error(fixed, over),
        "push_actual": float((g["y"] == g["line_int"]).mean()), "push_raw": float(g["p_push_int_main"].mean()),
        "push_cal": float(push_int.mean()),
        "bins_raw": M.calibration_bins(raw, over).to_dict("list"),
        "bins_cal": M.calibration_bins(fixed, over).to_dict("list"),
    }


def expected_calibration_error(p: np.ndarray, outcome: np.ndarray) -> float:
    """Average |predicted - actual| over 10 probability buckets, weighted by bucket size."""
    b = M.calibration_bins(p, outcome)
    return float((b["n"] * (b["bin_pred"] - b["bin_actual"]).abs()).sum() / b["n"].sum())


def evaluate(bundle: Bundle | None = None) -> dict:
    bundle = bundle or load_bundle(include_archived=True)
    assert_time_split(bundle)
    if not (C.CALIBRATION_SEASON in C.TEST_SEASONS and C.CALIBRATION_SEASON < C.CALIBRATION_TEST_SEASON):
        raise ValueError("calibration must be fitted on an earlier test season than the one it is judged on")
    missing = [s for s, k in bundle.player.candidates.items() if "quantile" not in k]
    if missing:
        raise FileNotFoundError("archived quantile models not found in models_archive/; retrain to regenerate them")
    rng = np.random.default_rng(C.SEED)
    pg, _, pf, tf = load_frames()
    IMG.mkdir(parents=True, exist_ok=True)
    pmask = test_mask(pf, "personId")
    ptest = pf[pmask].reset_index(drop=True)
    nodes, w = bundle.minutes.distribution(ptest)
    results: dict = {"player": {}, "team": {}, "minutes": minutes_backtest(pf, pmask, bundle), "calibration": {}}
    cal_rows = []
    tmask = test_mask(tf, "teamId")
    jobs = [("player", stat, pf, pmask, "personId", nodes, w) for stat in bundle.player.stats]
    jobs += [("team", stat, tf, tmask, "teamId", None, None) for stat in bundle.team.stats]
    for subject, stat, frame, mask, by, nd, ww in jobs:
        log.info("scoring %s %s", subject, stat)
        sm = bundle.subject(subject)
        sc = score_stat(frame, mask, stat, sm, by, nd, ww)
        results[subject][stat] = summarize(sc, sm, by)
        cb = calibration_backtest(sc, bundle.calibration.get(subject, {}).get(stat))
        results["calibration"][f"{subject}:{stat}"] = cb
        if subject == "player" or stat == "teamScore":
            after = cb["bins_cal"] if cb["kind"] != "identity" else None
            calibration_png(cb["bins_raw"], after, stat, IMG / f"calibration_{stat}.png", C.CALIBRATION_TEST_SEASON)
        label = stat_label(subject, stat)
        cal_rows.append(pd.DataFrame(cb["bins_raw"]).assign(stat=f"{label} (raw)"))
        cal_rows.append(pd.DataFrame(cb["bins_cal"]).assign(stat=f"{label} (calibrated)"))
    log.info("joint probabilities")
    results["joint"] = joint_backtest(pf, pmask, bundle, rng)
    log.info("defender ablation")
    results["defender"] = defender_ablation(pf, pmask, pg, bundle, nodes, w)
    (DOCS / "model_report.md").write_text(write_report(results, bundle))
    (DOCS / "model_metrics.json").write_text(json.dumps(app_metrics(results), indent=1))
    pd.concat(cal_rows).to_csv(DOCS / "model_report_calibration.csv", index=False)
    (DOCS / "model_backtest.json").write_text(json.dumps(results, indent=1, default=float))
    log.info("wrote %s", DOCS / "model_report.md")
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    evaluate()


if __name__ == "__main__":
    main()

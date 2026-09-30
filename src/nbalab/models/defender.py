"""Named-defender adjustment. **This is a proxy**, and the output labels it so.

No box score says who guarded whom. What the play-by-play does give is who
was on the floor together. For a subject X and an opponent player Y, over
X's earlier games against Y:

    on-rate  = X's stat per minute while Y was on the floor
    off-rate = X's stat per minute, same games, while Y was off the floor
    delta    = on-rate / off-rate - 1

Comparing within the same games cancels the opposing team's overall quality.
Most of delta is noise: a few hundred minutes of floor time say little. The
noise is measured per pair (a cluster-robust variance that treats each game
as one unit), and delta is shrunk toward 0 by empirical Bayes:
B = tau^2 / (tau^2 + noise variance), where tau^2 is how much true
on/off effects really vary across all pairs in the training seasons
(``nbalab.query.inference.prior_variance``).

The adjustment to the rate is 1 + share x B x delta, where share is the
typical fraction of X's minutes against Y's team with Y on the floor.

Only points, threes and assists are tracked in the on-court table, so other
stats are not adjusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nbalab.query.inference import prior_variance

ON_COURT_STATS: dict[str, str] = {"points": "pts", "threePointersMade": "fg3m", "assists": "ast"}
MIN_PAIR_MINUTES = 100.0
LABEL = "PROXY: on/off floor time with this defender (not 'guarded by')"


def pair_game_rows(on_court: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """Per (subject, opponent player, game): minutes and stats with the opponent on and off the floor.

    Off = the subject's box-score totals minus the on-floor part.
    """
    box = player_games[["gameId", "personId", "gameDateTimeEst", "minutes", *ON_COURT_STATS]]
    cols = ["gameId", "personId", "opponentPersonId", "shared_minutes", *ON_COURT_STATS.values()]
    rows = on_court[[c for c in cols if c in on_court.columns]].merge(box, on=["gameId", "personId"], how="inner")
    out = pd.DataFrame({
        "personId": rows["personId"].to_numpy(), "opponentPersonId": rows["opponentPersonId"].to_numpy(),
        "gameDateTimeEst": rows["gameDateTimeEst"].to_numpy(),
        "on_min": rows["shared_minutes"].to_numpy(float),
        "off_min": (rows["minutes"].astype(float) - rows["shared_minutes"]).clip(lower=0).to_numpy(),
    })
    for stat, oc in ON_COURT_STATS.items():
        if oc not in rows:
            continue
        on = rows[oc].astype(float).to_numpy()
        out[f"on_{stat}"] = on
        out[f"off_{stat}"] = np.clip(rows[stat].astype(float).to_numpy() - on, 0, None)
    return out


def _sum_columns(stats: list[str]) -> list[str]:
    cols = ["on_min", "off_min", "on_min2", "off_min2"]
    for s in stats:
        cols += [f"{p}_{s}{suf}" for p in ("on", "off") for suf in ("", "2", "_x_min")]
    return cols


def with_products(rows: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Add the squares and cross-products the cluster-robust variance needs."""
    r = rows.copy()
    for p in ("on", "off"):
        r[f"{p}_min2"] = r[f"{p}_min"] ** 2
        for s in stats:
            r[f"{p}_{s}2"] = r[f"{p}_{s}"] ** 2
            r[f"{p}_{s}_x_min"] = r[f"{p}_{s}"] * r[f"{p}_min"]
    return r


def pair_sums(rows: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Totals per (subject, opponent player) over all rows given."""
    r = with_products(rows, stats)
    g = r.groupby(["personId", "opponentPersonId"])[_sum_columns(stats)].sum()
    g["n_games"] = r.groupby(["personId", "opponentPersonId"]).size()
    return g.reset_index()


def _ratio_var(s: np.ndarray, s2: np.ndarray, sm: np.ndarray, m: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rate = sum s / sum m and its cluster-robust variance sum((s_j - r m_j)^2) / (sum m)^2."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = s / m
        var = (s2 - 2 * r * sm + r**2 * m2) / m**2
    return r, np.clip(var, 0, None)


def on_off_delta(sums: pd.DataFrame, stat: str) -> tuple[np.ndarray, np.ndarray]:
    """delta = on-rate / off-rate - 1 and its approximate sampling variance (delta method)."""
    def part(p: str) -> tuple[np.ndarray, np.ndarray]:
        return _ratio_var(*(sums[c].to_numpy(float) for c in
                            (f"{p}_{stat}", f"{p}_{stat}2", f"{p}_{stat}_x_min", f"{p}_min", f"{p}_min2")))
    r_on, v_on = part("on")
    r_off, v_off = part("off")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r_on / r_off
        var = ratio**2 * (v_on / r_on**2 + v_off / r_off**2)
    ok = (r_on > 0) & (r_off > 0) & np.isfinite(var)
    return np.where(ok, ratio - 1, np.nan), np.where(ok, var, np.nan)


def league_tau2(sums: pd.DataFrame, stats: list[str], min_minutes: float = MIN_PAIR_MINUTES) -> dict[str, float]:
    """How much true on/off effects vary across all pairs with enough minutes both ways (tau^2)."""
    big = sums[(sums["on_min"] >= min_minutes) & (sums["off_min"] >= min_minutes)]
    out = {}
    for s in stats:
        d, v = on_off_delta(big, s)
        ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
        # prior_variance(means, sizes, within_var): size 1/v with within_var 1 gives luck = mean(v).
        tau2 = prior_variance(d[ok], 1.0 / v[ok], 1.0)
        out[s] = 0.0 if math.isnan(tau2) else float(tau2)
    return out


@dataclass(frozen=True)
class DefenderEffect:
    """The proxy adjustment for one stat."""

    stat: str
    raw_delta: float
    shrink_weight: float
    share: float
    on_minutes: float
    n_games: int

    @property
    def multiplier(self) -> float:
        if not math.isfinite(self.raw_delta) or not math.isfinite(self.shrink_weight):
            return 1.0
        return max(1.0 + self.share * self.shrink_weight * self.raw_delta, 0.05)


def effects_from_sums(sums: pd.DataFrame, tau2: dict[str, float]) -> pd.DataFrame:
    """Per-pair shrunk multipliers for every tracked stat (vectorized)."""
    out = sums[["personId", "opponentPersonId", "on_min", "off_min", "n_games"]].copy()
    total = (sums["on_min"] + sums["off_min"]).to_numpy(float)
    out["share"] = np.where(total > 0, sums["on_min"].to_numpy(float) / np.where(total > 0, total, 1), 0.0)
    for s, t2 in tau2.items():
        d, v = on_off_delta(sums, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(np.isfinite(v), t2 / (t2 + v), 0.0) if t2 > 0 else np.zeros(len(d))
        out[f"{s}_delta"] = d
        out[f"{s}_weight"] = b
        out[f"{s}_mult"] = np.clip(1.0 + out["share"].to_numpy() * b * np.nan_to_num(d), 0.05, None)
    return out


def defender_effects(on_court: pd.DataFrame, player_games: pd.DataFrame, subject_id: int, defender_id: int,
                     before: pd.Timestamp | None, tau2: dict[str, float]) -> dict[str, DefenderEffect]:
    """Proxy effects of one named defender on one subject, from games before ``before``."""
    oc = on_court[(on_court["personId"] == subject_id) & (on_court["opponentPersonId"] == defender_id)]
    pg = player_games[player_games["personId"] == subject_id]
    rows = pair_game_rows(oc, pg)
    if before is not None:
        rows = rows[rows["gameDateTimeEst"] < before]
    stats = [s for s in tau2 if f"on_{s}" in rows]
    if rows.empty or not stats:
        return {}
    sums = pair_sums(rows, stats)
    eff = effects_from_sums(sums, {s: tau2[s] for s in stats}).iloc[0]
    return {
        s: DefenderEffect(s, float(eff[f"{s}_delta"]), float(eff[f"{s}_weight"]), float(eff["share"]),
                          float(eff["on_min"]), int(eff["n_games"]))
        for s in stats
    }


def asof_multipliers(keys: pd.DataFrame, rows: pd.DataFrame, tau2: dict[str, float]) -> pd.DataFrame:
    """For backtests: each key (personId, opponentPersonId, gameDateTimeEst) gets multipliers
    computed from that pair's games strictly before the key's date."""
    stats = [s for s in tau2 if f"on_{s}" in rows]
    r = with_products(rows, stats).sort_values("gameDateTimeEst")
    cols = _sum_columns(stats)
    cum = r.groupby(["personId", "opponentPersonId"], sort=False)[cols].cumsum()
    cum["n_games"] = r.groupby(["personId", "opponentPersonId"], sort=False).cumcount() + 1
    cum[["personId", "opponentPersonId", "gameDateTimeEst"]] = r[["personId", "opponentPersonId", "gameDateTimeEst"]]
    left = keys.reset_index().sort_values("gameDateTimeEst")
    merged = pd.merge_asof(left, cum.sort_values("gameDateTimeEst"), on="gameDateTimeEst",
                           by=["personId", "opponentPersonId"], allow_exact_matches=False, direction="backward")
    have = merged["on_min"].notna()
    eff = effects_from_sums(merged[have].reset_index(drop=True), tau2)
    out = pd.DataFrame(1.0, index=keys.index, columns=[f"{s}_mult" for s in stats])
    out.loc[merged.loc[have, "index"].to_numpy(), out.columns] = eff[out.columns].to_numpy()
    return out

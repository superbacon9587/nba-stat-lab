"""Period comparisons: the same stat before vs after a split point in the season.

Each season is cut into "before" and "after" (by default at the All-Star
break, see :class:`~nbalab.query.schema.PeriodSplit`). Regular-season games
only unless asked otherwise. The statistics, in plain terms:

- **Value of a period**: the average per game for counting stats; for
  percentages, total made / total attempted (pooled), so high-volume games
  count more.
- **Standard error (SE)**: for averages, sd / sqrt(games). For pooled ratios,
  the usual linearization: sqrt(sum((made_i - R x att_i)^2)) / sum(att_i).
- **Change** = after - before, with SE = sqrt(SE_before^2 + SE_after^2), a
  95% interval of change +/- 1.96 SE, and a two-sided p-value from the normal
  distribution (games per period are in the dozens, where t and normal agree).
- **Averaging seasons**: each season's change counts equally (a team's
  2016 change and 2024 change are separate experiments); the SE of the mean
  is sqrt(sum SE_s^2) / k.
- **League-adjusted change**: team change - league change over the same
  periods. If every team scores 1.5 more after the break (fresh legs, rule
  emphasis, pace), a team at +1.5 did not really change relative to the league.
- **30 teams at once**: with 30 tests, about 1.5 look "significant" at 5% by
  luck. Benjamini-Hochberg q-values correct for that; q < 0.10 is labeled
  "likely real", everything else "could be noise".
- **Players league-wide**: each player's change is shrunk toward the league
  average change by empirical Bayes, so a 12-game hot streak can't top the list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from nbalab.query import inference as inf
from nbalab.query import schema as s
from nbalab.query.data import QueryData
from nbalab.query.filters import scope_mask
from nbalab.query.stats import StatDef, stat_catalog

REGULAR = frozenset({"regular", "nba_cup"})
POSTSEASON = frozenset({"playoffs", "play_in"})
LIKELY_REAL_Q = 0.10
ROSTER_CHURN_FLAG = 0.25
FLAG_MIN_GAMES_LEFT = 5
MONTH_ORDER: tuple[int, ...] = (10, 11, 12, 1, 2, 3, 4)
MONTH_NAMES = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun"}


# ------------------------------------------------------------------ assigning periods


def cutoff_date(season: int, mm_dd: str) -> pd.Timestamp:
    """Calendar date of "MM-DD" inside a season: Oct-Dec fall in the starting year, Jan-Sep in the next."""
    m, d = (int(x) for x in mm_dd.split("-"))
    return pd.Timestamp(year=season if m >= 10 else season + 1, month=m, day=d)


def assign_period(games: pd.DataFrame, calendar: pd.DataFrame, split: s.PeriodSplit, by: str) -> pd.Series:
    """"before" / "after" / NaN for each game row.

    ``by`` is the subject key (teamId or personId), needed for "last N games".
    """
    gtype = games["game_type"].astype(str)
    regular = gtype.isin(REGULAR)
    post = gtype.isin(POSTSEASON)
    date = pd.to_datetime(games["game_date"]).dt.normalize()
    out = pd.Series(np.nan, index=games.index, dtype=object)
    if split.kind == "all_star_break":
        cal = calendar.set_index("season")
        last_before = games["season"].map(cal["break_last_before"])
        first_after = games["season"].map(cal["break_first_after"])
        out[regular & (date <= last_before)] = "before"
        out[regular & (date >= first_after)] = "after"
    elif split.kind == "custom_date":
        cut = games["season"].map(lambda yr: cutoff_date(int(yr), split.date))
        out[regular & (date < cut)] = "before"
        out[regular & (date >= cut)] = "after"
    elif split.kind == "month_groups":
        month = date.dt.month
        pool = regular | post if split.include_playoffs else regular
        out[pool & month.isin(split.before_months)] = "before"
        out[pool & month.isin(split.after_months)] = "after"
        return out
    else:  # last_n_before_playoffs
        reg = games[regular]
        from_end = reg.sort_values("game_date").groupby([by, "season"]).cumcount(ascending=False)
        out[from_end.index[from_end < split.n_games]] = "after"
        out[from_end.index[from_end >= split.n_games]] = "before"
    if split.include_playoffs:
        out[post] = "after"
    return out


# ------------------------------------------------------------------ one comparison


@dataclass(frozen=True)
class PeriodValue:
    value: float
    se: float
    n: int


@dataclass(frozen=True)
class PeriodChange:
    """Before vs after for one subject (one season, or several averaged)."""

    before: float
    after: float
    diff: float
    se: float
    n_before: int
    n_after: int
    seasons: int = 1
    seasons_improved: int = 0

    @property
    def pct(self) -> float:
        return 100.0 * self.diff / abs(self.before) if self.before else math.nan

    @property
    def ci(self) -> tuple[float, float]:
        return self.diff - 1.96 * self.se, self.diff + 1.96 * self.se

    @property
    def p_value(self) -> float:
        if not self.se or math.isnan(self.se):
            return math.nan
        return float(2 * sps.norm.sf(abs(self.diff) / self.se))


def period_value(stat: StatDef, games: pd.DataFrame) -> PeriodValue:
    """Value of one period with its standard error (see module docstring)."""
    n = len(games)
    if n == 0:
        return PeriodValue(math.nan, math.nan, 0)
    if not stat.is_ratio:
        v = stat.per_game(games).to_numpy(float)
        v = v[~np.isnan(v)]
        se = float(np.std(v, ddof=1) / math.sqrt(v.size)) if v.size > 1 else math.nan
        return PeriodValue(float(v.mean()) if v.size else math.nan, se, n)
    num = stat.value(games).to_numpy(float)
    den = stat.denominator(games).to_numpy(float)
    ok = ~(np.isnan(num) | np.isnan(den))
    num, den = num[ok], den[ok]
    if den.sum() <= 0:
        return PeriodValue(math.nan, math.nan, n)
    r = num.sum() / den.sum()
    se = math.sqrt(float(((num - r * den) ** 2).sum())) / den.sum() if num.size > 1 else math.nan
    return PeriodValue(r * stat.scale, se * stat.scale, n)


def per36_value(stat: StatDef, games: pd.DataFrame) -> PeriodValue:
    """Per-36-minute rate of a player counting stat: total / total minutes x 36."""
    if stat.is_ratio or not stat.per36 or "minutes" not in games or games.empty:
        return PeriodValue(math.nan, math.nan, len(games))
    num = stat.per_game(games).to_numpy(float)
    mins = games["minutes"].to_numpy(float)
    ok = ~(np.isnan(num) | np.isnan(mins))
    num, mins = num[ok], mins[ok]
    if mins.sum() <= 0:
        return PeriodValue(math.nan, math.nan, len(games))
    r = num.sum() / mins.sum()
    se = math.sqrt(float(((num - r * mins) ** 2).sum())) / mins.sum() if num.size > 1 else math.nan
    return PeriodValue(36 * r, 36 * se, len(games))


def change(before: PeriodValue, after: PeriodValue, higher_is_better: bool = True) -> PeriodChange:
    diff = after.value - before.value
    improved = int((diff > 0) == higher_is_better and diff != 0)
    return PeriodChange(before.value, after.value, diff, math.hypot(before.se, after.se),
                        before.n, after.n, 1, improved)


def average(changes: list[PeriodChange], precision_weighted: bool = False,
            min_games: int = 1) -> PeriodChange | None:
    """Average of seasonal changes, counting seasons with at least ``min_games`` on each side.

    Equal weights suit teams (every season has ~55 games before and ~27 after).
    ``precision_weighted`` weights each season by 1 / SE^2 instead, which suits
    players: an injury season with 4 games after the break should not count as
    much as a full one. The SE of the weighted mean is 1 / sqrt(sum of weights).
    """
    ok = [c for c in changes if c.n_before >= min_games and c.n_after >= min_games and not math.isnan(c.diff)]
    if precision_weighted:
        ok = [c for c in ok if c.se and not math.isnan(c.se)]
    if not ok:
        return None
    k = len(ok)
    if precision_weighted:
        w = np.array([1.0 / c.se**2 for c in ok])
        w = w / w.sum()
        se = 1.0 / math.sqrt(sum(1.0 / c.se**2 for c in ok))
    else:
        w = np.full(k, 1.0 / k)
        se = math.sqrt(sum((c.se if not math.isnan(c.se) else 0.0) ** 2 for c in ok)) / k
    return PeriodChange(
        before=float(w @ [c.before for c in ok]), after=float(w @ [c.after for c in ok]),
        diff=float(w @ [c.diff for c in ok]), se=se,
        n_before=sum(c.n_before for c in ok), n_after=sum(c.n_after for c in ok),
        seasons=k, seasons_improved=sum(c.seasons_improved for c in ok),
    )


PLAYER_MIN_GAMES_PER_SIDE = 5


# ------------------------------------------------------------------ shared frame prep


def scoped_frame(q: s.StatQuery, data: QueryData) -> pd.DataFrame:
    """Subject-type games inside the query's season scope, with a ``period`` column."""
    frame = data.team_games if q.subject_type == "team" else data.player_games
    frame = frame[scope_mask(q.scope_filters, frame, data.latest_season)]
    by = "teamId" if q.subject_type == "team" else "personId"
    if q.subject_ids:
        frame = frame[frame[by].isin(q.subject_ids)]
    frame = frame.assign(period=assign_period(frame, data.calendar(), q.period_split, by))
    return frame[frame["period"].notna()]


def seasonal_changes(frame: pd.DataFrame, stat: StatDef, by: str, per36: bool = False) -> pd.DataFrame:
    """One row per (subject, season) with games on both sides: before, after, change, se, n."""
    rows = []
    fn = per36_value if per36 else period_value
    for (key, season), g in frame.groupby([by, "season"], sort=False):
        b, a = g[g["period"] == "before"], g[g["period"] == "after"]
        if b.empty or a.empty:
            continue
        c = change(fn(stat, b), fn(stat, a), stat.higher_is_better)
        rows.append({by: key, "season": int(season), "before": c.before, "after": c.after, "diff": c.diff,
                     "se": c.se, "n_before": c.n_before, "n_after": c.n_after, "improved": c.seasons_improved})
    return pd.DataFrame(rows)


def league_changes(frame: pd.DataFrame, stat: StatDef) -> dict[int, PeriodChange]:
    """League-wide change per season (every subject's games pooled)."""
    out = {}
    for season, g in frame.groupby("season"):
        b, a = g[g["period"] == "before"], g[g["period"] == "after"]
        if len(b) and len(a):
            out[int(season)] = change(period_value(stat, b), period_value(stat, a), stat.higher_is_better)
    return out


# ------------------------------------------------------------------ all teams


@dataclass
class TeamTable:
    table: pd.DataFrame  # one row per team
    league: PeriodChange | None
    seasons: list[int]
    notes: list[str] = field(default_factory=list)
    split_sources: dict[int, str] = field(default_factory=dict)


def team_table(q: s.StatQuery, data: QueryData, season: int | None = None) -> TeamTable:
    """Every team's before/after change: one season (``season``) or the scope averaged."""
    stat = stat_catalog("team")[q.stats[0]]
    frame = scoped_frame(q.model_copy(update={"subject_ids": []}), data)
    if season is not None:
        frame = frame[frame["season"] == season]
    seasons = sorted(int(x) for x in frame["season"].unique())
    per = seasonal_changes(frame, stat, "teamId")
    league_by_season = league_changes(frame, stat)
    if per.empty:
        return TeamTable(pd.DataFrame(), None, seasons)
    per["league_diff"] = per["season"].map({k: v.diff for k, v in league_by_season.items()})
    per["adj_diff"] = per["diff"] - per["league_diff"]
    rows = []
    for tid, g in per.groupby("teamId"):
        changes = [PeriodChange(r.before, r.after, r.diff, r.se, r.n_before, r.n_after, 1, r.improved)
                   for r in g.itertuples()]
        avg = average(changes)
        adj = float(g["adj_diff"].mean())
        adj_improved = int(((g["adj_diff"] > 0) == stat.higher_is_better).sum())
        rows.append({
            "teamId": int(tid), "team": data.team_name(int(tid), max(seasons)),
            "before": avg.before, "after": avg.after, "change": avg.diff, "pct_change": avg.pct,
            "ci_low": avg.ci[0], "ci_high": avg.ci[1], "p_value": avg.p_value,
            "adj_change": adj, "adj_ci_low": adj - 1.96 * avg.se, "adj_ci_high": adj + 1.96 * avg.se,
            "adj_p_value": float(2 * sps.norm.sf(abs(adj) / avg.se)) if avg.se else math.nan,
            "games_before": avg.n_before, "games_after": avg.n_after,
            "seasons": avg.seasons, "seasons_improved": avg.seasons_improved,
            "seasons_improved_vs_league": adj_improved,
        })
    table = pd.DataFrame(rows)
    table["q_value"] = inf.benjamini_hochberg(table["p_value"].to_numpy())
    table["adj_q_value"] = inf.benjamini_hochberg(table["adj_p_value"].to_numpy())
    table["verdict"] = np.where(table["q_value"] < LIKELY_REAL_Q, "likely real", "could be noise")
    table["adj_verdict"] = np.where(table["adj_q_value"] < LIKELY_REAL_Q, "likely real", "could be noise")
    table["better"] = np.where((table["change"] > 0) == stat.higher_is_better, "better", "worse")
    league = average(list(league_by_season.values()))
    cal = data.calendar().set_index("season")
    sources = {yr: str(cal.at[yr, "source"]) for yr in seasons if yr in cal.index}
    t = TeamTable(table.sort_values("change", ascending=not stat.higher_is_better).reset_index(drop=True),
                  league, seasons, split_sources=sources)
    t.notes = team_context_notes(q, data, frame, seasons, league, stat)
    return t


def team_detail(q: s.StatQuery, data: QueryData, team_id: int) -> dict:
    """One team: its seasonal changes (consistency) and monthly trend."""
    stat = stat_catalog("team")[q.stats[0]]
    frame = scoped_frame(q.model_copy(update={"subject_ids": []}), data)
    per = seasonal_changes(frame[frame["teamId"] == team_id], stat, "teamId")
    league = league_changes(frame, stat)
    if not per.empty:
        per["league_diff"] = per["season"].map({k: v.diff for k, v in league.items()})
    return {"seasonal": per, "monthly": monthly_trend(q, data, "teamId", team_id)}


# ------------------------------------------------------------------ one player


@dataclass
class PlayerPeriods:
    raw: PeriodChange | None
    per36: PeriodChange | None
    seasonal: pd.DataFrame
    monthly: pd.DataFrame
    last_n: dict[int, PeriodChange]
    minutes: PeriodChange | None
    notes: list[str] = field(default_factory=list)


def player_detail(q: s.StatQuery, data: QueryData) -> PlayerPeriods:
    """One player's before/after (raw and per 36), per season, by month, and last-10/20 splits."""
    stat = stat_catalog("player")[q.stats[0]]
    pid = q.subject_ids[0]
    frame = scoped_frame(q, data)
    per = seasonal_changes(frame, stat, "personId")
    per36 = seasonal_changes(frame, stat, "personId", per36=True)
    if not per36.empty:
        per = per.merge(per36[["season", "before", "after", "diff"]].rename(
            columns={"before": "before36", "after": "after36", "diff": "diff36"}), on="season", how="left")
    mins = seasonal_changes(frame, stat_catalog("player")["minutes"], "personId")

    def avg_of(df: pd.DataFrame) -> PeriodChange | None:
        if df.empty:
            return None
        return average([PeriodChange(r.before, r.after, r.diff, r.se, r.n_before, r.n_after, 1, r.improved)
                        for r in df.itertuples()], precision_weighted=True, min_games=PLAYER_MIN_GAMES_PER_SIDE)

    last_n = {}
    for n in (10, 20):
        qn = q.model_copy(update={"period_split": s.PeriodSplit(kind="last_n_before_playoffs", n_games=n)})
        ch = avg_of(seasonal_changes(scoped_frame(qn, data), stat, "personId"))
        if ch is not None:
            last_n[n] = ch
    notes = []
    thin = per[(per["n_before"] < PLAYER_MIN_GAMES_PER_SIDE) | (per["n_after"] < PLAYER_MIN_GAMES_PER_SIDE)] \
        if not per.empty else per
    if len(thin):
        notes.append("Left out of the career average (fewer than 5 games on one side, e.g. injuries): "
                     + ", ".join(f"{r.season}-{str(r.season + 1)[-2:]} ({r.n_before}/{r.n_after} games)"
                                 for r in thin.itertuples()) + ". Seasons are weighted by how precise they are.")
    if not mins.empty and (avg := avg_of(mins)) is not None and abs(avg.diff) >= 1.5:
        notes.append(f"Minutes changed by {avg.diff:+.1f} per game after the split, so compare the per-36 "
                     "numbers as well as the raw ones.")
    return PlayerPeriods(avg_of(per), avg_of(per36), per, monthly_trend(q, data, "personId", pid),
                         last_n, avg_of(mins), notes)


def monthly_trend(q: s.StatQuery, data: QueryData, by: str, key: int) -> pd.DataFrame:
    """Per-month average (Oct-Apr regular season, then playoffs) with a 95% CI, plus the
    league-average monthly curve rescaled to this subject's overall level."""
    stat = stat_catalog(q.subject_type)[q.stats[0]]
    frame = data.team_games if q.subject_type == "team" else data.player_games
    frame = frame[scope_mask(q.scope_filters, frame, data.latest_season)]
    gtype = frame["game_type"].astype(str)
    month = pd.to_datetime(frame["game_date"]).dt.month
    bucket = pd.Series(np.where(gtype.eq("playoffs"), "Playoffs", month.map(MONTH_NAMES)), index=frame.index)
    bucket[~(gtype.isin(REGULAR) | gtype.eq("playoffs"))] = None
    frame = frame.assign(bucket=bucket).dropna(subset=["bucket"])
    mine = frame[frame[by] == key]
    if mine.empty:
        return pd.DataFrame()
    overall = period_value(stat, mine[mine["game_type"].astype(str).isin(REGULAR)]).value
    league_overall = period_value(stat, frame[frame["game_type"].astype(str).isin(REGULAR)]).value
    order = [MONTH_NAMES[m] for m in MONTH_ORDER] + ["Playoffs"]
    rows = []
    for b in order:
        g = mine[mine["bucket"] == b]
        lg = frame[frame["bucket"] == b]
        if g.empty and lg.empty:
            continue
        pv, lv = period_value(stat, g), period_value(stat, lg)
        rows.append({"month": b, "value": pv.value, "ci_low": pv.value - 1.96 * pv.se, "ci_high": pv.value + 1.96 * pv.se,
                     "games": pv.n, "league_scaled": lv.value / league_overall * overall if league_overall else math.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ league-wide players


def player_leaderboard(q: s.StatQuery, data: QueryData, min_games: int = 10, per36: bool = False,
                       top: int = 25) -> pd.DataFrame:
    """Players whose stat changed most after the split, shrunk toward the league-average change.

    Each player's change (averaged over the seasons in scope) has an SE. The
    spread of true changes across players (tau^2) is estimated from all
    players at once, and each change is pulled toward the league mean with
    weight B = tau^2 / (tau^2 + SE^2): a noisy 12-game swing is pulled most.
    """
    stat = stat_catalog("player")[q.stats[0]]
    frame = scoped_frame(q.model_copy(update={"subject_ids": []}), data)
    per = seasonal_changes(frame, stat, "personId", per36=per36)
    if per.empty:
        return pd.DataFrame()
    per = per[(per["n_before"] >= min_games) & (per["n_after"] >= min_games)]
    g = per.groupby("personId")
    agg = pd.DataFrame({
        "before": g["before"].mean(), "after": g["after"].mean(), "change": g["diff"].mean(),
        "se": g["se"].apply(lambda x: math.sqrt(float((x.fillna(0) ** 2).sum())) / len(x)),
        "games_before": g["n_before"].sum(), "games_after": g["n_after"].sum(),
        "seasons": g.size(), "seasons_improved": g["improved"].sum(),
    }).reset_index()
    agg = agg[agg["se"] > 0]
    center = float(agg["change"].mean())
    tau2 = inf.prior_variance(agg["change"].to_numpy(), 1.0 / agg["se"].to_numpy() ** 2, 1.0)
    tau2 = 0.0 if math.isnan(tau2) else tau2
    weight = tau2 / (tau2 + agg["se"] ** 2) if tau2 > 0 else 0.0 * agg["se"]
    agg["shrunk_change"] = center + weight * (agg["change"] - center)
    agg["player"] = agg["personId"].map(lambda p: data.player_name(int(p)))
    ascending = not stat.higher_is_better
    return agg.sort_values("shrunk_change", ascending=ascending).head(top).reset_index(drop=True)


# ------------------------------------------------------------------ context notes


CHURN_LOOKBACK_DAYS = 21


def roster_churn(data: QueryData, frame_periods: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Share of each team's "after" minutes played by players who were not on it well before the split.

    "Well before" = games at least three weeks before the team's last "before"
    game. The trade deadline usually falls one to two weeks *before* the
    All-Star break, so players acquired at the deadline already appear in the
    last "before" games; comparing with the earlier roster catches them.
    """
    pg = data.player_games[data.player_games["season"].isin(seasons)]
    labels = frame_periods.set_index(["gameId", "teamId"])["period"]
    pg = pg.join(labels, on=["gameId", "teamId"], how="inner")
    pg = pg.assign(day=pd.to_datetime(pg["game_date"]).dt.normalize())
    rows = []
    for (tid, season), g in pg.groupby(["teamId", "season"]):
        before = g[g["period"] == "before"]
        if before.empty:
            continue
        early = before[before["day"] <= before["day"].max() - pd.Timedelta(days=CHURN_LOOKBACK_DAYS)]
        before_players = set((early if len(early) else before)["personId"])
        after = g[g["period"] == "after"]
        total = float(after["minutes"].sum())
        if total <= 0:
            continue
        new = float(after.loc[~after["personId"].isin(before_players), "minutes"].sum())
        rows.append({"teamId": int(tid), "season": int(season), "new_minutes_share": new / total})
    return pd.DataFrame(rows)


def standings_flags(data: QueryData, season: int, from_date: pd.Timestamp) -> pd.DataFrame:
    """Approximate date each team was eliminated from / clinched a postseason spot.

    A team is out when its wins plus remaining games can't reach the current wins
    of the conference's last postseason seed (8th before 2020-21, 10th with the
    play-in). It is in when fewer than that many rivals can still reach its wins.
    Tie-breakers and exact remaining schedules are ignored, so dates are approximate.
    """
    tg = data.team_games
    tg = tg[(tg["season"] == season) & tg["game_type"].astype(str).isin(REGULAR)].sort_values("game_date")
    if tg.empty:
        return pd.DataFrame()
    conf = data.conference_table()
    conf = conf[conf["season"] == season].set_index("teamId")["conference"]
    cutoff = 10 if season >= 2020 else 8
    total_games = tg.groupby("teamId").size()
    dates = sorted(d for d in pd.to_datetime(tg["game_date"]).dt.normalize().unique() if d >= from_date)
    eliminated: dict[int, tuple[pd.Timestamp, int]] = {}
    clinched: dict[int, tuple[pd.Timestamp, int]] = {}
    gd = pd.to_datetime(tg["game_date"]).dt.normalize()
    for d in dates:
        played = tg[gd <= d]
        w = played.groupby("teamId")["win"].sum().reindex(total_games.index, fill_value=0).astype(float)
        left = total_games - played.groupby("teamId").size().reindex(total_games.index, fill_value=0)
        for c in ("East", "West"):
            teams = [t for t in total_games.index if conf.get(t) == c]
            if len(teams) <= cutoff:
                continue
            cur = w[teams].sort_values(ascending=False)
            for t in teams:
                others = [o for o in teams if o != t]
                if t not in eliminated and w[t] + left[t] < cur.drop(t).iloc[cutoff - 1]:
                    eliminated[t] = (d, int(left[t]))
                can_catch = sum(1 for o in others if w[o] + left[o] >= w[t])
                if t not in clinched and can_catch <= cutoff - 1:
                    clinched[t] = (d, int(left[t]))
    rows = [{"teamId": int(t), "status": "eliminated", "date": d, "games_left": n} for t, (d, n) in eliminated.items()]
    rows += [{"teamId": int(t), "status": "clinched", "date": d, "games_left": n} for t, (d, n) in clinched.items()]
    out = pd.DataFrame(rows)
    return out[out["games_left"] >= FLAG_MIN_GAMES_LEFT] if not out.empty else out


def team_context_notes(q: s.StatQuery, data: QueryData, frame: pd.DataFrame, seasons: list[int],
                       league: PeriodChange | None, stat: StatDef) -> list[str]:
    """Automatic caveats: roster churn, tanking/resting candidates, league-wide drift, split source."""
    notes = []
    cal = data.calendar().set_index("season")
    inferred = [yr for yr in seasons if yr in cal.index and cal.at[yr, "source"] == "schedule_gap"]
    fallback = [yr for yr in seasons if yr in cal.index and cal.at[yr, "source"] == "midseason_fallback"]
    if q.period_split.kind == "all_star_break" and inferred:
        notes.append(f"All-Star break dates for {len(inferred)} of {len(seasons)} season(s) are inferred from the "
                     "league-wide schedule gap (the data has All-Star Game rows only for 2025-26).")
    if q.period_split.kind == "all_star_break" and fallback:
        notes.append("1998-99 had no All-Star break (lockout season); it is split at mid-season by games played.")
    if league is not None:
        notes.append(f"League-wide, {stat.label.lower()} moved {league.diff:+.2f} per game after the split "
                     f"(average over {league.seasons} season(s)). The league-adjusted column subtracts this.")
    churn = roster_churn(data, frame, seasons)
    if not churn.empty:
        big = churn[churn["new_minutes_share"] >= ROSTER_CHURN_FLAG]
        if len(big):
            names = [f"{data.team_name(int(r.teamId), int(r.season))} {r.season}-{str(r.season + 1)[-2:]} "
                     f"({r.new_minutes_share:.0%})" for r in big.sort_values("new_minutes_share", ascending=False).head(8).itertuples()]
            notes.append("Big roster turnover around the split (share of 'after' minutes by players who weren't playing "
                         "for the team 3+ weeks earlier: deadline trades, call-ups, or returns from injury), so these "
                         "teams' 'after' numbers partly describe a different roster: " + ", ".join(names) + ".")
    if q.period_split.kind in ("all_star_break", "custom_date", "last_n_before_playoffs"):
        flagged = []
        for yr in seasons[-3:]:  # the approximation is costly; the recent seasons are what readers check
            if yr == 2019 or yr not in cal.index:
                continue
            fl = standings_flags(data, yr, pd.Timestamp(cal.at[yr, "break_first_after"]))
            for r in fl.itertuples():
                flagged.append(f"{data.team_name(r.teamId, yr)} {yr}-{str(yr + 1)[-2:]} {r.status} with {r.games_left} left")
        if flagged:
            notes.append("Teams out of the race (possible tanking) or already safe (possible resting of stars) with "
                         "games still to play (approximate, ignores tie-breakers): " + "; ".join(flagged[:12]) + ".")
    notes.append("Trade-deadline splits are not offered: the data has no transaction records, so the deadline "
                 "can't be told apart from waiver claims and buyout signings.")
    return notes

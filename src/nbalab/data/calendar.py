"""Season calendar: the All-Star break and season boundaries for every season.

The raw data has All-Star Game rows (gameId prefix ``3``) for only one season,
so for most seasons the break is *inferred* from the schedule: the All-Star
break is the only multi-day stretch in mid-season when no NBA games are
played at all. Rules, per season:

1. ``all_star_game``: an All-Star Game in the data. The break is the no-game
   stretch around it.
2. ``schedule_gap``: otherwise, the longest league-wide gap of 3 to 8 days
   between Jan 15 and Mar 20. (The 3-8 day window skips the 2019-20 COVID
   shutdown, which was 141 days.)
3. ``midseason_fallback``: seasons with no break at all (the 1998-99 lockout)
   are split at the date by which half of the season's games were played.

Also here: conference membership inferred from the schedule (conference
rivals meet 3-4 times a season, other teams at most twice), which the
period comparisons use for approximate eliminated/clinched flags.

    python -m nbalab.data.calendar   # writes data/processed/season_calendar.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nbalab.data.config import DATA_DIR

REGULAR_TYPES: tuple[str, ...] = ("regular", "nba_cup")
GAP_MIN_DAYS, GAP_MAX_DAYS = 3, 8
WINDOW = ((1, 15), (3, 20))  # (month, day) bounds for where the break can start
BOSTON = 1610612738  # names the Eastern component when inferring conferences
SOURCE_LABELS: dict[str, str] = {
    "all_star_game": "All-Star Game date from the data",
    "schedule_gap": "inferred from the league-wide schedule gap (no All-Star Game rows in the data)",
    "midseason_fallback": "no All-Star break this season (1998-99 lockout); split at mid-season by games played",
}


def season_of_game_id(game_id: int) -> int:
    """Starting year of the season from the gameId (``T YY NNNNN``): 32500011 -> 2025."""
    yy = int(game_id) // 100_000 % 100
    return 1900 + yy if yy >= 46 else 2000 + yy


def all_star_game_dates(games_raw: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """Season -> date of the main All-Star Game (the last All-Star event of that season)."""
    ids = pd.to_numeric(games_raw["gameId"], errors="coerce")
    asg = games_raw.loc[ids.astype("Int64").astype(str).str.startswith("3"), ["gameId", "gameDateTimeEst"]].dropna()
    if asg.empty:
        return {}
    asg = asg.assign(season=asg["gameId"].map(season_of_game_id),
                     date=pd.to_datetime(asg["gameDateTimeEst"]).dt.normalize())
    return {int(s): d for s, d in asg.groupby("season")["date"].max().items()}


def _in_window(d: pd.Timestamp) -> bool:
    (m0, d0), (m1, d1) = WINDOW
    return (d.month, d.day) >= (m0, d0) and (d.month, d.day) <= (m1, d1)


def break_from_gap(dates: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(last game date before, first game date after) of the longest 3-8 day mid-season gap."""
    days = pd.Series(sorted(pd.to_datetime(dates).dt.normalize().unique()))
    if len(days) < 2:
        return None
    frame = pd.DataFrame({"before": days.shift(1), "after": days, "gap": days.diff().dt.days}).dropna()
    frame = frame[frame["gap"].between(GAP_MIN_DAYS, GAP_MAX_DAYS) & frame["before"].map(_in_window)]
    if frame.empty:
        return None
    best = frame.loc[frame["gap"].idxmax()]
    return pd.Timestamp(best["before"]), pd.Timestamp(best["after"])


def break_around(dates: pd.Series, asg: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Last game date on or before the All-Star Game and first date after it."""
    days = pd.Series(sorted(pd.to_datetime(dates).dt.normalize().unique()))
    before, after = days[days <= asg], days[days > asg]
    if before.empty or after.empty:
        return None
    return pd.Timestamp(before.iloc[-1]), pd.Timestamp(after.iloc[0])


def midseason_split(game_dates: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Split where half of the season's games (not dates) have been played."""
    d = pd.to_datetime(game_dates).dt.normalize().sort_values().reset_index(drop=True)
    half = d.iloc[len(d) // 2]
    later = d[d > half]
    return pd.Timestamp(half), pd.Timestamp(later.iloc[0] if len(later) else half)


def build_season_calendar(games_raw: pd.DataFrame | None, team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per season: the break (last date before / first date after), its source,
    the All-Star Game date if known, and regular-season / playoff boundaries."""
    asg_dates = all_star_game_dates(games_raw) if games_raw is not None else {}
    rows = []
    for season, g in team_games.groupby("season"):
        season = int(season)
        dates = pd.to_datetime(g["game_date"]).dt.normalize()
        reg = dates[g["game_type"].astype(str).isin(REGULAR_TYPES)]
        post = dates[g["game_type"].astype(str).eq("playoffs")]
        asg = asg_dates.get(season)
        brk, source = (break_around(reg, asg), "all_star_game") if asg is not None else (None, "")
        if brk is None:
            brk, source = break_from_gap(reg), "schedule_gap"
        if brk is None:
            brk, source = midseason_split(reg), "midseason_fallback"
        rows.append({
            "season": season, "break_last_before": brk[0], "break_first_after": brk[1],
            "all_star_date": asg if asg is not None else (brk[0] + (brk[1] - brk[0]) / 2).normalize(),
            "all_star_date_known": asg is not None, "source": source,
            "regular_season_start": reg.min(), "regular_season_end": reg.max(),
            "playoffs_start": post.min() if len(post) else pd.NaT,
        })
    return pd.DataFrame(rows)


def infer_conferences(team_games: pd.DataFrame) -> pd.DataFrame:
    """(season, teamId, conference) from how often teams met in the regular season.

    Teams that met at least 3 times are linked; the two connected groups are the
    conferences, and the one containing Boston is the East. If that fails for a
    season (unusual schedule), each team joins the group it played more games against.
    """
    reg = team_games[team_games["game_type"].astype(str).isin(REGULAR_TYPES)]
    out = []
    for season, g in reg.groupby("season"):
        pairs = g.groupby(["teamId", "opponentTeamId"]).size()
        teams = sorted(set(g["teamId"].astype(int)))
        east = _component(pairs, BOSTON, threshold=3)
        if not (0.35 * len(teams) <= len(east) <= 0.65 * len(teams)):
            east = _majority_split(pairs, teams)
        out += [{"season": int(season), "teamId": t, "conference": "East" if t in east else "West"} for t in teams]
    return pd.DataFrame(out)


def _component(pairs: pd.Series, start: int, threshold: int) -> set[int]:
    linked = pairs[pairs >= threshold].reset_index()[["teamId", "opponentTeamId"]].astype(int)
    adj: dict[int, set[int]] = {}
    for a, b in linked.itertuples(index=False):
        adj.setdefault(a, set()).add(b)
    seen, stack = {start}, [start]
    while stack:
        for nxt in adj.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _majority_split(pairs: pd.Series, teams: list[int]) -> set[int]:
    """Fallback: grow the East from Boston by adding the team with most games against it."""
    counts = pairs.unstack(fill_value=0).reindex(index=teams, columns=teams, fill_value=0)
    east = {BOSTON}
    while len(east) < len(teams) // 2:
        rest = [t for t in teams if t not in east]
        east.add(max(rest, key=lambda t: counts.loc[t, list(east)].mean()))
    return east


def main(argv: list[str] | None = None) -> None:
    from nbalab.data.load import GAMES_DTYPES, read_typed_csv

    ap = argparse.ArgumentParser(description="Build data/processed/season_calendar.parquet")
    ap.add_argument("--raw-dir", type=Path, default=DATA_DIR / "raw")
    ap.add_argument("--processed-dir", type=Path, default=DATA_DIR / "processed")
    args = ap.parse_args(argv)
    games = read_typed_csv(args.raw_dir / "Games.csv", GAMES_DTYPES)
    tg = pd.read_parquet(args.processed_dir / "team_games.parquet", columns=["season", "game_date", "game_type",
                                                                             "teamId", "opponentTeamId"])
    cal = build_season_calendar(games, tg)
    cal.to_parquet(args.processed_dir / "season_calendar.parquet", index=False)
    conf = infer_conferences(tg)
    conf.to_parquet(args.processed_dir / "conferences.parquet", index=False)
    with pd.option_context("display.width", 160):
        print(cal[["season", "break_last_before", "break_first_after", "all_star_date", "source"]].to_string(index=False))
    sizes = conf.groupby(["season", "conference"]).size().unstack()
    print("\nconference sizes (min/max):", sizes.min().to_dict(), sizes.max().to_dict())


if __name__ == "__main__":
    main()

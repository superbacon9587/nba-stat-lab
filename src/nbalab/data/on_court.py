"""Shared floor time between opposing players, from reconstructed lineups.

For every player-game this produces one row per opponent who played in that
game:

``sharedFloorSeconds``
    How long the two were **on the floor at the same time**.
``pts, fgm, fga, fg3m, fg3a, ftm, fta, ast``
    The player's own counting stats while that opponent was on the floor.

This is **shared floor time, not "guarded by"**. Being on the court together
does not mean one defended the other. It supports honest questions like "how
does X score while Y is on the floor" (an on/off split), not "when Y guards
X". Official defensive matchups come from :mod:`nbalab.data.matchups`.

A second table, ``on_court_players.parquet``, holds one row per player-game
with reconstructed seconds and play-by-play stat totals next to the box score.
It is the quality check (reconstructed vs box-score minutes) and the "off the
floor" denominator (a player's time and stats while Y was off the floor =
game total minus the shared amount).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from nbalab.data.clean import parse_minutes, resolve_game_type, season_from_game_id
from nbalab.data.lineups import GameReconstruction, reconstruct_game
from nbalab.data.pbp import PBP_COLUMNS, STAT_COLUMNS, normalize_events

# Reconstruction is flagged as unreliable for a player-game when reconstructed
# minutes are off from box-score minutes by more than this.
MINUTES_TOLERANCE = 2.0


def load_rosters(raw_dir: Path, game_ids: list[int] | None = None) -> pd.DataFrame:
    """Box-score roster per game: ``gameId, personId, teamId, firstName, lastName, starter, minutes, points, assists``.

    ``playerteamId`` is missing on ~6.5% of rows (21% in the 2020s). It is
    recovered by matching the team name to the home/away team of the same game
    in ``Games.csv``, which is 100% reliable (see ``docs/data_profile.md``).
    """
    con = duckdb.connect()
    ps = (raw_dir / "PlayerStatistics.csv").as_posix()
    gm = (raw_dir / "Games.csv").as_posix()
    where = ""
    if game_ids is not None:
        con.register("wanted", pd.DataFrame({"gameId": game_ids}))
        where = "where p.gameId in (select gameId from wanted)"
    df = con.sql(
        f"""
        select p.gameId, p.personId,
               coalesce(p.playerteamId,
                        case when p.playerteamName = g.hometeamName then g.hometeamId
                             when p.playerteamName = g.awayteamName then g.awayteamId end) as teamId,
               p.firstName, p.lastName, p.startingPosition, p.numMinutes, p.points,
               p.assists
        from read_csv('{ps}', quote='"', escape='"', types={{'numMinutes': 'VARCHAR'}}) p
        left join read_csv('{gm}', quote='"', escape='"') g on g.gameId = p.gameId
        {where}
        """
    ).df()
    df["minutes"] = parse_minutes(df.pop("numMinutes"))
    df["starter"] = df["startingPosition"].notna() & (df["startingPosition"].astype("string").str.strip() != "")
    df["teamId"] = df["teamId"].astype("Int64")
    return df.drop(columns="startingPosition")


def pair_rows(rec: GameReconstruction) -> pd.DataFrame:
    """Aggregate one game's segments into one row per (player, opponent) pair.

    Both directions are emitted: the row for (X, Y) carries X's stats while Y
    was on the floor, the row for (Y, X) carries Y's stats while X was on.
    ``sharedFloorSeconds`` is the same in both.
    """
    shared: dict[tuple[int, int], float] = defaultdict(float)
    stats: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(len(STAT_COLUMNS)))
    team_of: dict[int, int] = {}
    a, b = rec.teams
    for seg in rec.segments:
        d = seg.end - seg.start
        for x in seg.lineups[a]:
            team_of[x] = a
        for y in seg.lineups[b]:
            team_of[y] = b
        if d <= 0:
            continue
        for x in seg.lineups[a]:
            for y in seg.lineups[b]:
                shared[(x, y)] += d
    for ev in rec.stat_events:
        opp = b if ev.teamId == a else a
        team_of.setdefault(ev.personId, ev.teamId)
        for y in rec.segments[ev.segment].lineups[opp]:
            stats[(ev.personId, y)] += ev.stats

    keys = set()
    for x, y in shared:
        keys.add((x, y))
        keys.add((y, x))
    keys |= set(stats)
    if not keys:
        return pd.DataFrame()
    records = []
    for x, y in keys:
        secs = shared.get((x, y), shared.get((y, x), 0.0))
        records.append((x, team_of.get(x), y, team_of.get(y), secs, *stats.get((x, y), np.zeros(len(STAT_COLUMNS)))))
    out = pd.DataFrame(
        records,
        columns=["personId", "teamId", "opponentPersonId", "opponentTeamId", "sharedFloorSeconds", *STAT_COLUMNS],
    )
    out.insert(0, "gameId", rec.gameId)
    for c in STAT_COLUMNS:
        out[c] = out[c].astype("int16")
    return out


def player_rows(rec: GameReconstruction) -> pd.DataFrame:
    """Reconstructed seconds on the floor and play-by-play stat totals per player."""
    secs: dict[int, float] = defaultdict(float)
    team_of: dict[int, int] = {}
    for seg in rec.segments:
        d = seg.end - seg.start
        for team, unit in seg.lineups.items():
            for p in unit:
                secs[p] += d
                team_of[p] = team
    totals: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(len(STAT_COLUMNS)))
    for ev in rec.stat_events:
        totals[ev.personId] += ev.stats
        team_of.setdefault(ev.personId, ev.teamId)
    pids = set(secs) | set(totals)
    out = pd.DataFrame(
        [(p, team_of[p], secs.get(p, 0.0), *totals.get(p, np.zeros(len(STAT_COLUMNS)))) for p in pids],
        columns=["personId", "teamId", "reconSeconds", *(f"pbp_{c}" for c in STAT_COLUMNS)],
    )
    out.insert(0, "gameId", rec.gameId)
    return out


def lineup_quality(rec: GameReconstruction) -> dict[str, float]:
    """Share of game time where each team had exactly five players tracked, plus repair counts."""
    total = sum(s.end - s.start for s in rec.segments)
    good = sum(
        s.end - s.start for s in rec.segments if all(len(u) == 5 for u in s.lineups.values())
    )
    return {"gameId": rec.gameId, "gameSeconds": total, "fiveOnFiveShare": good / total if total else np.nan, **rec.repairs}


def build_games(events: pd.DataFrame, rosters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run reconstruction for every game in ``events``. Returns (pairs, players, quality)."""
    pairs, players, quality = [], [], []
    roster_by_game = dict(tuple(rosters.groupby("gameId")))
    for gid, ev in events.groupby("gameId", sort=False):
        roster = roster_by_game.get(gid)
        if roster is None:
            continue
        rec = reconstruct_game(ev, roster)
        if rec is None:
            continue
        pairs.append(pair_rows(rec))
        players.append(player_rows(rec))
        quality.append(lineup_quality(rec))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return cat(pairs), cat(players), pd.DataFrame(quality)


def read_pbp(raw_dir: Path, season_prefixes: list[str]) -> pd.DataFrame:
    """Read the play-by-play columns needed for games whose gameId starts with any prefix.

    Prefixes are gameId strings without the leading zeros, e.g. ``"215"`` for
    2015-16 regular season, ``"415"`` for its playoffs.
    """
    path = (raw_dir / "PlayByPlay.parquet").as_posix()
    cols = ", ".join(PBP_COLUMNS)
    cond = " or ".join(f"gameId like '{p}%'" for p in season_prefixes)
    return duckdb.sql(f"select {cols} from '{path}' where {cond}").df()


def attach_game_info(df: pd.DataFrame, box_players: pd.DataFrame) -> pd.DataFrame:
    """Add season and normalized game type (both from the gameId)."""
    df = df.copy()
    gid = df["gameId"].astype("int64")
    df.insert(1, "season", season_from_game_id(gid).astype("int16"))
    df.insert(2, "gameType", resolve_game_type(gid).astype("string"))
    return df


def compare_to_box(players: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Join reconstructed totals to box-score minutes and points, with error columns.

    Outer join: a player the box score says played but the reconstruction never
    put on the floor shows up with ``reconSeconds = 0`` (a full-size error).
    """
    box = rosters.loc[rosters["minutes"].fillna(0) > 0, ["gameId", "personId", "teamId", "minutes", "points", "assists"]]
    box = box[box["gameId"].isin(players["gameId"].unique())].rename(
        columns={
            "minutes": "boxMinutes", "points": "boxPoints", "assists": "boxAssists", "teamId": "boxTeamId",
        }
    )
    out = players.merge(box, on=["gameId", "personId"], how="outer")
    out["teamId"] = out["teamId"].fillna(out.pop("boxTeamId"))
    out["reconSeconds"] = out["reconSeconds"].fillna(0.0)
    out["reconMinutes"] = out["reconSeconds"] / 60
    out["minutesError"] = out["reconMinutes"] - out["boxMinutes"]
    out["reliable"] = out["minutesError"].abs() <= MINUTES_TOLERANCE
    return out

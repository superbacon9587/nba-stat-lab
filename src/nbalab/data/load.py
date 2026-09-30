"""Load raw CSVs with explicit dtypes.

Default type inference breaks on ``PlayerStatistics.csv`` (``numMinutes`` mixes
decimals and ``"MM:SS"`` strings; team ids are null on some rows), so every
column we use is typed here. Only the columns the pipeline needs are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ID = "Int64"  # nullable integer
NUM = "float64"
STR = "string"

PLAYER_STATS_DTYPES: dict[str, str] = {
    "firstName": STR,
    "lastName": STR,
    "personId": ID,
    "gameId": ID,
    "gameDateTimeEst": STR,
    "playerteamCity": STR,
    "playerteamName": STR,
    "opponentteamCity": STR,
    "opponentteamName": STR,
    "gameType": STR,
    "win": ID,
    "home": ID,
    "numMinutes": STR,
    "points": NUM,
    "assists": NUM,
    "blocks": NUM,
    "steals": NUM,
    "fieldGoalsAttempted": NUM,
    "fieldGoalsMade": NUM,
    "threePointersAttempted": NUM,
    "threePointersMade": NUM,
    "freeThrowsAttempted": NUM,
    "freeThrowsMade": NUM,
    "reboundsDefensive": NUM,
    "reboundsOffensive": NUM,
    "reboundsTotal": NUM,
    "foulsPersonal": NUM,
    "turnovers": NUM,
    "plusMinusPoints": NUM,
    "playerteamId": ID,
    "opponentteamId": ID,
    "startingPosition": STR,
}

TEAM_STATS_DTYPES: dict[str, str] = {
    "gameId": ID,
    "gameDateTimeEst": STR,
    "teamId": ID,
    "opponentTeamId": ID,
    "home": ID,
    "win": ID,
    "teamScore": NUM,
    "opponentScore": NUM,
    "assists": NUM,
    "blocks": NUM,
    "steals": NUM,
    "fieldGoalsAttempted": NUM,
    "fieldGoalsMade": NUM,
    "threePointersAttempted": NUM,
    "threePointersMade": NUM,
    "freeThrowsAttempted": NUM,
    "freeThrowsMade": NUM,
    "reboundsDefensive": NUM,
    "reboundsOffensive": NUM,
    "reboundsTotal": NUM,
    "foulsPersonal": NUM,
    "turnovers": NUM,
    "numMinutes": NUM,
    "gameType": STR,
}

TEAM_ADVANCED_DTYPES: dict[str, str] = {
    "gameId": ID,
    "teamId": ID,
    "offensiveRating": NUM,
    "defensiveRating": NUM,
    "netRating": NUM,
    "pace": NUM,
    "possessions": NUM,
    "effectiveFieldGoalPercentage": NUM,
    "trueShootingPercentage": NUM,
    "teamTurnoverPercentage": NUM,
    "offensiveReboundPercentage": NUM,
    "freeThrowAttemptRate": NUM,
    "opponentEffectiveFieldGoalPercentage": NUM,
    "opponentTurnoverPercentage": NUM,
    "opponentOffensiveReboundPercentage": NUM,
    "opponentFreeThrowAttemptRate": NUM,
}

GAMES_DTYPES: dict[str, str] = {
    "gameId": ID,
    "gameDateTimeEst": STR,
    "hometeamCity": STR,
    "hometeamName": STR,
    "hometeamId": ID,
    "awayteamCity": STR,
    "awayteamName": STR,
    "awayteamId": ID,
    "homeScore": NUM,
    "awayScore": NUM,
    "gameType": STR,
    "arenaName": STR,
    "arenaCity": STR,
}

PLAYERS_DTYPES: dict[str, str] = {
    "personId": ID,
    "firstName": STR,
    "lastName": STR,
    "birthDate": STR,
    "heightInches": NUM,
    "bodyWeightLbs": NUM,
    "guard": ID,
    "forward": ID,
    "center": ID,
    "draftYear": ID,
    "draftRound": ID,
    "draftNumber": ID,
}

TEAM_HISTORIES_DTYPES: dict[str, str] = {
    "teamId": ID,
    "teamCity": STR,
    "teamName": STR,
    "teamAbbrev": STR,
    "seasonFounded": ID,
    "seasonActiveTill": ID,
    "league": STR,
}

SCHEDULE_DTYPES: dict[str, str] = {
    "gameId": ID,
    "arenaName": STR,
    "arenaCity": STR,
}


@dataclass
class RawTables:
    """The raw inputs the pipeline needs, already typed but otherwise untouched."""

    player_stats: pd.DataFrame
    team_stats: pd.DataFrame
    team_advanced: pd.DataFrame
    games: pd.DataFrame
    players: pd.DataFrame
    team_histories: pd.DataFrame
    schedules: pd.DataFrame


def read_typed_csv(path: Path, dtypes: dict[str, str]) -> pd.DataFrame:
    """Read only the typed columns of a CSV; parse ``gameDateTimeEst`` as a naive EST timestamp."""
    df = pd.read_csv(path, usecols=list(dtypes), dtype=dtypes, keep_default_na=True)
    if "gameDateTimeEst" in df:
        df["gameDateTimeEst"] = parse_est_timestamp(df["gameDateTimeEst"])
    return df


def parse_est_timestamp(raw: pd.Series) -> pd.Series:
    """Parse EST wall-clock timestamps; drops any ``+00:00`` suffix (the values are EST already)."""
    text = raw.astype("string").str.slice(0, 19)
    return pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def read_schedules(raw_dir: Path, files: tuple[str, ...]) -> pd.DataFrame:
    """Stack arena info from the per-season schedule files (their other columns differ)."""
    frames = [
        read_typed_csv(raw_dir / name, SCHEDULE_DTYPES)
        for name in files
        if (raw_dir / name).exists()
    ]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEDULE_DTYPES.items()})
    return pd.concat(frames, ignore_index=True).drop_duplicates("gameId", keep="last")


def load_raw(raw_dir: Path, schedule_files: tuple[str, ...] = ()) -> RawTables:
    """Load every raw table the pipeline uses from ``raw_dir``."""
    return RawTables(
        player_stats=read_typed_csv(raw_dir / "PlayerStatistics.csv", PLAYER_STATS_DTYPES),
        team_stats=read_typed_csv(raw_dir / "TeamStatistics.csv", TEAM_STATS_DTYPES),
        team_advanced=read_typed_csv(raw_dir / "TeamStatisticsExtended.csv", TEAM_ADVANCED_DTYPES),
        games=read_typed_csv(raw_dir / "Games.csv", GAMES_DTYPES),
        players=read_typed_csv(raw_dir / "Players.csv", PLAYERS_DTYPES),
        team_histories=read_typed_csv(raw_dir / "TeamHistories.csv", TEAM_HISTORIES_DTYPES),
        schedules=read_schedules(raw_dir, schedule_files),
    )

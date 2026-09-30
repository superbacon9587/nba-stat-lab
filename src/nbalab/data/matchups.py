"""Official NBA defensive matchups (BoxScoreMatchupsV3) -> one tidy table.

The NBA's player tracking assigns each offensive possession, in fractions, to
the defender(s) guarding the ball-handler or player. For every (offensive
player, defender) pair in a game the endpoint reports:

- ``partialPossessions``: how many possessions (fractional) the defender was
  the primary defender on this player. **This is the right denominator for
  "points per possession when Y guards X".**
- ``matchupSeconds``: time the defender was matched up on him.
- ``playerPoints`` / ``fgm`` / ``fga`` / ``fg3m`` / ``fg3a`` / ``ftm`` / ``fta``:
  the offensive player's own production against that defender.
- ``assists``, ``turnovers``, ``blocks``: likewise, during the matchup.

This is the only source in the project that genuinely answers "when Y guards
X". It exists from 2017-18 on. Earlier questions need a proxy, see
:mod:`nbalab.query.defender`.

Raw responses are cached by ``scripts/fetch_matchups.py`` in
``data/raw/matchups/<season>/<gameId>.json.gz``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from nbalab.data.clean import resolve_game_type, season_from_game_id

FIRST_MATCHUP_SEASON = 2017  # 2017-18: first season with tracking matchup data

# Raw statistics key -> output column.
STAT_FIELDS: dict[str, str] = {
    "partialPossessions": "partialPossessions",
    "playerPoints": "playerPoints",
    "teamPoints": "teamPoints",
    "matchupAssists": "assists",
    "matchupPotentialAssists": "potentialAssists",
    "matchupTurnovers": "turnovers",
    "matchupBlocks": "blocks",
    "matchupFieldGoalsMade": "fgm",
    "matchupFieldGoalsAttempted": "fga",
    "matchupThreePointersMade": "fg3m",
    "matchupThreePointersAttempted": "fg3a",
    "matchupFreeThrowsMade": "ftm",
    "matchupFreeThrowsAttempted": "fta",
    "helpBlocks": "helpBlocks",
    "helpFieldGoalsMade": "helpFgm",
    "helpFieldGoalsAttempted": "helpFga",
    "shootingFouls": "shootingFouls",
    "switchesOn": "switchesOn",
    "percentageDefenderTotalTime": "pctDefenderTotalTime",
    "percentageOffensiveTotalTime": "pctOffensiveTotalTime",
    "percentageTotalTimeBothOn": "pctTotalTimeBothOn",
}

BIO_COLUMNS = ["personId", "heightInches", "bodyWeightLbs", "position_group", "position"]


def cache_path(cache_dir: Path, game_id: str, season: int) -> Path:
    """Where the raw response for a 10-digit ``game_id`` is cached."""
    return cache_dir / str(season) / f"{game_id}.json.gz"


def clock_to_seconds(text: object) -> float:
    """``"M:SS"`` -> seconds (``"0:42"`` -> 42.0). Returns NaN if unparseable."""
    try:
        minutes, seconds = str(text).split(":")
        return int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return float("nan")


def parse_game(data: dict) -> pd.DataFrame:
    """Flatten one raw response into one row per (offensive player, defender).

    In the response, ``homeTeam.players`` are the home team's players **on
    offense**, and each one's ``matchups`` list is the away defenders who guarded
    him (and the reverse for ``awayTeam``).
    """
    box = data.get("boxScoreMatchups") or {}
    game_id = int(box.get("gameId") or 0)
    home, away = box.get("homeTeam") or {}, box.get("awayTeam") or {}
    rows = []
    for off_team, def_team in ((home, away), (away, home)):
        for player in off_team.get("players") or []:
            for m in player.get("matchups") or []:
                stats = m.get("statistics") or {}
                row = {
                    "gameId": game_id,
                    "offTeamId": off_team.get("teamId"),
                    "offPersonId": player.get("personId"),
                    "offStartingPosition": player.get("position") or None,
                    "defTeamId": def_team.get("teamId"),
                    "defPersonId": m.get("personId"),
                    "matchupSeconds": clock_to_seconds(stats.get("matchupMinutes")),
                }
                row.update({out: stats.get(raw) for raw, out in STAT_FIELDS.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def read_cached_game(path: Path) -> pd.DataFrame:
    """Parse one cached ``.json.gz`` response."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return parse_game(json.load(f))


def load_player_bio(processed_dir: Path, raw_dir: Path) -> pd.DataFrame:
    """Height, weight and position per personId.

    Prefers ``data/processed/players.parquet`` (built from ``Players.csv``, with
    dual-position players collapsed to one ``position_group``). Falls back to
    ``Players.csv`` directly, using the bigger of the flagged positions.
    """
    processed = processed_dir / "players.parquet"
    if processed.exists():
        return pd.read_parquet(processed, columns=BIO_COLUMNS)
    from nbalab.data.clean import position_group

    raw = pd.read_csv(raw_dir / "Players.csv")
    flags = raw[["guard", "forward", "center"]].fillna(0).astype(bool)
    raw["position"] = flags.apply(lambda r: "-".join(p for p, on in zip("GFC", r) if on) or None, axis=1)
    raw["position_group"] = position_group(raw["guard"], raw["forward"], raw["center"])
    return raw[BIO_COLUMNS]


def attach_defender_bio(matchups: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """Add ``defHeightInches``, ``defWeightLbs``, ``defPosition`` (G/F/C) and ``defPositionDetail``."""
    b = bio.drop_duplicates("personId").rename(
        columns={
            "personId": "defPersonId",
            "heightInches": "defHeightInches",
            "bodyWeightLbs": "defWeightLbs",
            "position_group": "defPosition",
            "position": "defPositionDetail",
        }
    )
    return matchups.merge(b, on="defPersonId", how="left")


def build_matchups(cache_dir: Path, bio: pd.DataFrame) -> pd.DataFrame:
    """Parse every cached game and attach defender bio, season and game type."""
    files = sorted(cache_dir.glob("*/*.json.gz"))
    frames = [read_cached_game(f) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    int_cols = ["gameId", "offTeamId", "offPersonId", "defTeamId", "defPersonId"]
    df[int_cols] = df[int_cols].astype("int64")
    gid = df["gameId"]
    df.insert(1, "season", season_from_game_id(gid).astype("int16"))
    df.insert(2, "gameType", resolve_game_type(gid).astype("string"))
    return attach_defender_bio(df, bio)

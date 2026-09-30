"""Load the processed tables the query engines read.

:class:`QueryData` bundles the tables so tests can build one from small
in-memory fixtures and the app can build one from ``data/processed/``.

Two optional tables come from the defender-matchup pipeline (build prompt 2).
They are used when present and skipped when absent:

- ``on_court.parquet``: shared floor time between each player and each
  opponent player per game. This is *not* "guarded by".
- ``matchups.parquet``: official NBA tracking matchups (who defended whom,
  in partial possessions), 2017-18 onward.

Their column names are matched loosely (see ``ON_COURT_COLUMNS`` /
``MATCHUP_COLUMNS``). That keeps this module working while the pipeline that
writes them settles, and it fails loudly if no candidate name matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from nbalab.data.config import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"

# canonical name -> accepted source column names, first match wins
ON_COURT_COLUMNS: dict[str, tuple[str, ...]] = {
    "gameId": ("gameId", "game_id"),
    "personId": ("personId", "person_id", "offPersonId", "playerId"),
    "opponentPersonId": (
        "opponentPersonId", "opponent_person_id", "oppPersonId", "opp_person_id",
        "opponentId", "otherPersonId",
    ),
    "shared_minutes": (
        "sharedFloorMinutes", "shared_minutes", "sharedMinutes", "minutes_shared", "shared_floor_minutes", "minutes",
    ),
}
# Optional on_court columns kept when present: a reliability flag (reconstructed
# minutes agree with the box score) and the player's own stats while the
# opponent was on the floor.
ON_COURT_OPTIONAL: tuple[str, ...] = ("reliable", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "ast")
MATCHUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "gameId": ("gameId", "game_id"),
    "offPersonId": ("offPersonId", "personIdOff", "offensivePlayerId", "off_person_id", "personId"),
    "defPersonId": ("defPersonId", "personIdDef", "defensivePlayerId", "def_person_id", "defenderId"),
    "partial_possessions": (
        "partial_possessions", "partialPossessions", "matchupPossessions", "possessions",
    ),
}


@dataclass
class QueryData:
    """All tables the query layer needs. Optional tables are ``None`` when not built."""

    player_games: pd.DataFrame
    team_games: pd.DataFrame
    players: pd.DataFrame
    teams: pd.DataFrame
    on_court: pd.DataFrame | None = None
    matchups: pd.DataFrame | None = None
    _team_names: dict[tuple[int, int], str] = field(default_factory=dict, repr=False)

    @property
    def latest_season(self) -> int:
        return int(max(self.player_games["season"].max(), self.team_games["season"].max()))

    def player_name(self, person_id: int) -> str:
        row = self.players.loc[self.players["personId"] == person_id, "full_name"]
        return str(row.iloc[0]) if len(row) else f"player {person_id}"

    def team_name(self, team_id: int, season: int | None = None) -> str:
        """Team name, season-appropriate when ``season`` is given (Sonics vs Thunder)."""
        if season is not None and (team_id, season) in self.team_names_by_season:
            return self.team_names_by_season[(team_id, season)]
        row = self.teams.loc[self.teams["teamId"] == team_id, "full_name"]
        return str(row.iloc[0]) if len(row) else f"team {team_id}"

    @property
    def team_names_by_season(self) -> dict[tuple[int, int], str]:
        if not self._team_names and "name_history" in self.teams:
            for tid, hist in zip(self.teams["teamId"], self.teams["name_history"]):
                for era in hist if hist is not None else []:
                    last = era["last_season"] if era["last_season"] is not None else self.latest_season
                    for s in range(int(era["first_season"]), int(last) + 1):
                        self._team_names[(int(tid), s)] = f"{era['city']} {era['name']}"
        return self._team_names


def pick_columns(df: pd.DataFrame, spec: dict[str, tuple[str, ...]], table: str) -> pd.DataFrame:
    """Rename the first matching source column for each canonical name, keep only those."""
    rename: dict[str, str] = {}
    for canonical, candidates in spec.items():
        found = next((c for c in candidates if c in df.columns and c not in rename), None)
        if found is None:
            raise ValueError(
                f"{table}: none of {candidates} found for '{canonical}'. Columns: {list(df.columns)}"
            )
        rename[found] = canonical
    return df[list(rename)].rename(columns=rename)


def normalize_on_court(df: pd.DataFrame) -> pd.DataFrame:
    """Shared-floor-time table as ``gameId, personId, opponentPersonId, shared_minutes``
    plus ``reliable`` (all True if the source has no such flag) and any stat columns.

    Rows flagged unreliable are dropped: for those player-games the lineup
    reconstruction disagreed with box-score minutes, so the engine treats the
    game as uncovered and falls back to the next defender source.
    """
    out = pick_columns(df, ON_COURT_COLUMNS, "on_court")
    extra = [c for c in ON_COURT_OPTIONAL if c in df.columns]
    out = pd.concat([out, df[extra]], axis=1)
    if "reliable" not in out:
        out["reliable"] = True
    out = out[out["reliable"].astype(bool)]
    return out.astype({"gameId": "int64", "personId": "int64", "opponentPersonId": "int64",
                       "shared_minutes": "float64"}).reset_index(drop=True)


def normalize_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Official matchups as ``gameId, offPersonId, defPersonId, partial_possessions``."""
    out = pick_columns(df, MATCHUP_COLUMNS, "matchups")
    return out.astype({"gameId": "int64", "offPersonId": "int64", "defPersonId": "int64",
                       "partial_possessions": "float64"})


def _read_optional(path: Path, normalize, wanted: tuple[str, ...] = ()) -> pd.DataFrame | None:
    """Read and normalize an optional table, only the ``wanted`` columns if they exist."""
    if not path.exists():
        return None
    if wanted:
        import pyarrow.parquet as pq

        present = set(pq.read_schema(path).names)
        return normalize(pd.read_parquet(path, columns=[c for c in wanted if c in present]))
    return normalize(pd.read_parquet(path))


def _all_candidates(spec: dict[str, tuple[str, ...]], extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    return tuple(dict.fromkeys([c for cands in spec.values() for c in cands] + list(extra)))


@lru_cache(maxsize=2)
def load_query_data(processed_dir: Path = PROCESSED_DIR) -> QueryData:
    """Read the processed parquet files once per process."""
    d = Path(processed_dir)
    pg = pd.read_parquet(d / "player_games.parquet")
    tg = pd.read_parquet(d / "team_games.parquet")
    pg["gameId"] = pg["gameId"].astype("int64")
    for c in ("teamId", "gameId", "opponentTeamId"):
        tg[c] = tg[c].astype("int64")
    return QueryData(
        player_games=pg,
        team_games=tg,
        players=pd.read_parquet(d / "players.parquet"),
        teams=pd.read_parquet(d / "teams.parquet"),
        on_court=_read_optional(d / "on_court.parquet", normalize_on_court,
                                _all_candidates(ON_COURT_COLUMNS, ON_COURT_OPTIONAL)),
        matchups=_read_optional(d / "matchups.parquet", normalize_matchups, _all_candidates(MATCHUP_COLUMNS)),
    )

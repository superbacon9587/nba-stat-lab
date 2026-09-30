"""Resolve player, team, and venue names to ids.

The LLM (or the rule parser) hands over names exactly as written: "Steph",
"the Spurs", "Philly", "LeBron". This module matches them against the aliases
built into ``players.parquet`` and ``teams.parquet``:

1. **Exact alias match** after normalizing ("Curry's" -> "curry").
2. **Fuzzy match** with rapidfuzz when nothing matches exactly ("Lebrom",
   "Stef Curry"). A score of 0-100 says how close the spelling is.

A name can fit several players ("Curry" fits Stephen, Seth, Dell, ...).
Candidates are ranked by how well the spelling matches, then by whether their
career overlaps the seasons asked about, then by games played. The top pick is
a **clear winner** when it has played far more games than the runner-up
(Stephen vs Seth Curry). Two players who share the *full* name typed
("Mike James") are never a clear winner, so the user is always asked.

Players change teams, so "which team is LeBron on" depends on the season.
``stints`` (built from ``player_games``) records each player's seasons with
each team, so the builder can check "his defender is LeBron" against the
opponent for the right season.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from rapidfuzz import fuzz, process

from nbalab.data.aliases import normalize_alias, team_aliases
from nbalab.data.config import DATA_DIR, REFERENCE_DIR
from nbalab.nlp.types import Candidate

PROCESSED_DIR = DATA_DIR / "processed"
VENUE_ALIASES_CSV = REFERENCE_DIR / "venue_aliases.csv"

EXACT_SCORE = 100.0
SeasonSpan = tuple[int, int]


@dataclass(frozen=True)
class MatchSettings:
    """Tunable matching thresholds.

    ``fuzzy_cutoff``: lowest rapidfuzz score (0-100) accepted as a match.
    ``clear_winner_ratio``: the top candidate is picked without asking when it
    has at least this many times the runner-up's games (and they do not share
    the full name typed).
    ``score_gap``: or when its spelling score beats the runner-up by this much.
    """

    fuzzy_cutoff: float = 85.0
    clear_winner_ratio: float = 1.8
    score_gap: float = 8.0


@dataclass
class Resolution:
    """Outcome of resolving one name.

    ``status`` is ``unique`` (one match), ``clear`` (several, but one clear
    winner), ``ambiguous`` (the user must pick), or ``not_found``.
    """

    text: str
    kind: Literal["player", "team"]
    status: Literal["unique", "clear", "ambiguous", "not_found"]
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def chosen(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def id(self) -> int | None:
        return self.chosen.id if self.chosen else None


def clean_name(text: str) -> str:
    """Normalize a name as written: drop "the", possessives, and punctuation.

    ``"the Spurs"`` -> ``"spurs"``, ``"Curry's"`` -> ``"curry"``.
    """
    text = re.sub(r"(?:'|’)s\b", "", text.strip())
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    return normalize_alias(text)


def overlaps(span: SeasonSpan | None, first: int, last: int) -> bool:
    return span is None or (first <= span[1] and last >= span[0])


def _alias_map(ids: Iterable[int], alias_lists: Iterable[Iterable[str]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, aliases in zip(ids, alias_lists):
        for a in aliases if aliases is not None else []:
            out.setdefault(str(a), [])
            if int(i) not in out[a]:
                out[a].append(int(i))
    return out


def build_stints(player_games: pd.DataFrame) -> pd.DataFrame:
    """Each player's seasons with each team: ``personId, teamId, first_season, last_season, games``."""
    g = player_games.groupby(["personId", "teamId"], observed=True)["season"]
    out = g.agg(first_season="min", last_season="max", games="size").reset_index()
    return out.astype({"personId": "int64", "teamId": "int64"})


def load_venue_aliases(path: Path = VENUE_ALIASES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", dtype={"teamId": "int64"})
    return df.assign(alias=df["alias"].map(normalize_alias))


class EntityIndex:
    """Name lookup over the players and teams tables.

    ``players`` needs ``personId, full_name, aliases, first_season,
    last_season, games, last_team_id`` (``position_group`` and
    ``heightInches`` are shown when present). ``teams`` needs ``teamId,
    city, name, abbrev, full_name, aliases`` (``name_history`` is used when
    present to date historical names like the Sonics).
    """

    def __init__(
        self,
        players: pd.DataFrame,
        teams: pd.DataFrame,
        stints: pd.DataFrame | None = None,
        venues: pd.DataFrame | None = None,
        settings: MatchSettings = MatchSettings(),
    ) -> None:
        self.players = players.set_index("personId", drop=False)
        self.teams = teams.set_index("teamId", drop=False)
        self.stints = stints
        self.venues = venues if venues is not None else load_venue_aliases()
        self.settings = settings
        self.player_aliases = _alias_map(players["personId"], players["aliases"])
        self.team_aliases = _alias_map(teams["teamId"], teams["aliases"])
        self.current_team_aliases = _alias_map(
            teams["teamId"],
            (team_aliases(c, n, a) for c, n, a in zip(teams["city"], teams["name"], teams["abbrev"])),
        )
        self.venue_aliases = _alias_map(self.venues["teamId"], ([a] for a in self.venues["alias"]))
        self._player_keys = list(self.player_aliases)
        self._team_keys = list(self.team_aliases)
        self._full_names = {int(i): normalize_alias(str(n)) for i, n in zip(players["personId"], players["full_name"])}

    @classmethod
    def from_processed(cls, processed_dir: Path = PROCESSED_DIR) -> "EntityIndex":
        return _cached_index(Path(processed_dir))

    # ------------------------------------------------------------------ lookups

    def is_player_alias(self, norm: str) -> bool:
        return norm in self.player_aliases

    def is_team_alias(self, norm: str) -> bool:
        return norm in self.team_aliases

    def is_venue_alias(self, norm: str) -> bool:
        return norm in self.venue_aliases

    def player_name(self, person_id: int) -> str:
        return str(self.players.at[person_id, "full_name"]) if person_id in self.players.index else f"player {person_id}"

    def team_name(self, team_id: int) -> str:
        return str(self.teams.at[team_id, "full_name"]) if team_id in self.teams.index else f"team {team_id}"

    def team_abbrev(self, team_id: int) -> str:
        return str(self.teams.at[team_id, "abbrev"]).strip() if team_id in self.teams.index else "?"

    def position_group(self, person_id: int) -> str | None:
        if person_id not in self.players.index or "position_group" not in self.players:
            return None
        v = self.players.at[person_id, "position_group"]
        return None if pd.isna(v) else str(v)

    # --------------------------------------------------------------- candidates

    def player_candidate(self, person_id: int, score: float) -> Candidate:
        r = self.players.loc[person_id]
        pos = r.get("position_group")
        bits = [str(pos)] if pos is not None and not pd.isna(pos) else []
        bits.append(f"{int(r['first_season'])}-{int(r['last_season'])}")
        bits.append(f"{int(r['games'])} games")
        bits.append(f"last team {self.team_abbrev(int(r['last_team_id']))}")
        return Candidate(id=int(person_id), kind="player", name=str(r["full_name"]), detail=", ".join(bits), score=score)

    def team_candidate(self, team_id: int, score: float) -> Candidate:
        return Candidate(id=int(team_id), kind="team", name=self.team_name(team_id),
                         detail=self.team_abbrev(team_id), score=score)

    def _scores(self, norm: str, alias_ids: dict[str, list[int]], keys: list[str]) -> dict[int, float]:
        """Best spelling score per id: exact alias hits score 100, else fuzzy.

        Fuzzy candidates come from rapidfuzz's WRatio (forgiving: "curry" inside
        "stef curry" scores 90). Their score is the mean of WRatio and the plain
        ratio, so the alias that is closest as a whole ranks first."""
        if norm in alias_ids:
            return {i: EXACT_SCORE for i in alias_ids[norm]}
        if len(norm) < 3:
            return {}
        out: dict[int, float] = {}
        hits = process.extract(norm, keys, scorer=fuzz.WRatio, score_cutoff=self.settings.fuzzy_cutoff, limit=25)
        for key, wscore, _ in hits:
            score = (float(wscore) + fuzz.ratio(norm, key)) / 2  # whole-name similarity breaks WRatio ties
            for i in alias_ids[key]:
                out[i] = max(out.get(i, 0.0), score)
        return out

    # ---------------------------------------------------------------- resolving

    def resolve_player(self, text: str, seasons: SeasonSpan | None = None) -> Resolution:
        """Rank the players a name could mean and decide whether one clearly wins."""
        norm = clean_name(text)
        scores = self._scores(norm, self.player_aliases, self._player_keys)
        if not scores:
            return Resolution(text, "player", "not_found")
        ids = list(scores)
        in_span = [i for i in ids if overlaps(seasons, *self._career(i))]
        ids = in_span or ids
        games = {i: int(self.players.at[i, "games"]) for i in ids}
        ids.sort(key=lambda i: (-round(scores[i]), -games[i], -self._career(i)[1]))
        cands = [self.player_candidate(i, scores[i]) for i in ids]
        if len(ids) == 1:
            return Resolution(text, "player", "unique", cands)
        top, second = ids[0], ids[1]
        same_full_name = sum(self._full_names.get(i) == norm for i in ids) > 1
        clear = not same_full_name and (
            scores[top] - scores[second] >= self.settings.score_gap
            or games[top] >= self.settings.clear_winner_ratio * max(games[second], 1)
        )
        return Resolution(text, "player", "clear" if clear else "ambiguous", cands)

    def resolve_team(self, text: str, seasons: SeasonSpan | None = None) -> Resolution:
        """Rank the teams a name could mean. Current names beat historical ones
        ("Hornets" is Charlotte today, New Orleans in 2005)."""
        norm = clean_name(text)
        scores = self._scores(norm, self.team_aliases, self._team_keys)
        if not scores:
            return Resolution(text, "team", "not_found")
        ids = list(scores)
        if seasons is not None:
            dated = [i for i in ids if self._team_alias_in_span(i, norm, seasons)]
            ids = dated or ids
        current = [i for i in ids if i in self.current_team_aliases.get(norm, [])]
        preferred = current if seasons is None and current else ids
        ids.sort(key=lambda i: (i not in preferred, -round(scores[i]), i))
        cands = [self.team_candidate(i, scores[i]) for i in ids]
        if len(ids) == 1:
            return Resolution(text, "team", "unique", cands)
        clear = len(preferred) == 1 or scores[ids[0]] - scores[ids[1]] >= self.settings.score_gap
        return Resolution(text, "team", "clear" if clear else "ambiguous", cands)

    def resolve_venue(self, text: str, season: int | None = None) -> list[int]:
        """Team ids whose building is at this place ("San Francisco" -> Warriors).

        Tries arena/city aliases first, then team aliases that are cities
        ("in Boston"). Several ids (Los Angeles) means use the city instead.
        """
        norm = clean_name(text)
        if norm in self.venue_aliases:
            rows = self.venues[self.venues["alias"] == norm]
            if season is not None:
                lo = rows["first_season"].fillna(0)
                hi = rows["last_season"].fillna(9999)
                dated = rows[(lo <= season) & (hi >= season)]
                rows = dated if len(dated) else rows
            return sorted(set(int(i) for i in rows["teamId"]))
        cities = {int(t): normalize_alias(str(c)) for t, c in zip(self.teams["teamId"], self.teams["city"])}
        by_city = [t for t, c in cities.items() if c == norm]
        if by_city:
            return by_city
        res = self.resolve_team(text)
        return [res.id] if res.status in ("unique", "clear") and res.id is not None else []

    # ------------------------------------------------------------------- rosters

    def _career(self, person_id: int) -> tuple[int, int]:
        r = self.players.loc[person_id]
        return int(r["first_season"]), int(r["last_season"])

    def _team_alias_in_span(self, team_id: int, norm: str, seasons: SeasonSpan) -> bool:
        if "name_history" not in self.teams:
            return True
        hist = self.teams.at[team_id, "name_history"]
        for era in hist if hist is not None else []:
            names = team_aliases(era["city"], era["name"], era["abbrev"])
            last = era["last_season"] if era["last_season"] is not None and not pd.isna(era["last_season"]) else 9999
            if norm in names and overlaps(seasons, int(era["first_season"]), int(last)):
                return True
        return False

    def teams_in_season(self, person_id: int, season: int) -> set[int]:
        """Teams the player played for in ``season``. Falls back to ``last_team_id``
        when no roster stints are loaded or the season is past the player's last."""
        if self.stints is not None:
            s = self.stints[self.stints["personId"] == person_id]
            hit = s[(s["first_season"] <= season) & (s["last_season"] >= season)]
            if len(hit):
                return set(int(t) for t in hit["teamId"])
            if len(s) and season < int(s["first_season"].min()):
                return set()
        if person_id in self.players.index and season >= int(self.players.at[person_id, "last_season"]):
            return {int(self.players.at[person_id, "last_team_id"])}
        return set()

    def ever_played_for(self, person_id: int, team_id: int) -> tuple[int, int] | None:
        """(first, last) seasons the player spent with the team, or ``None``."""
        if self.stints is None:
            if person_id in self.players.index and int(self.players.at[person_id, "last_team_id"]) == team_id:
                return self._career(person_id)
            return None
        s = self.stints[(self.stints["personId"] == person_id) & (self.stints["teamId"] == team_id)]
        return (int(s["first_season"].min()), int(s["last_season"].max())) if len(s) else None


@lru_cache(maxsize=2)
def _cached_index(processed_dir: Path) -> EntityIndex:
    players = pd.read_parquet(processed_dir / "players.parquet")
    teams = pd.read_parquet(processed_dir / "teams.parquet")
    pg_path = processed_dir / "player_games.parquet"
    stints = build_stints(pd.read_parquet(pg_path, columns=["personId", "teamId", "season"])) if pg_path.exists() else None
    return EntityIndex(players, teams, stints)

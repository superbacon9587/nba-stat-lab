"""Minimal name -> id resolution for the CLI (the full fuzzy resolver lives in ``nbalab.nlp``).

Matches normalized text against the alias lists in ``players.parquet`` /
``teams.parquet``. An exact full-name match wins. Otherwise, among players
sharing an alias ("Curry"), the one with the most games is picked and the
others are returned as alternatives so the caller can say what it assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from nbalab.data.aliases import normalize_alias


@dataclass
class Resolved:
    id: int
    name: str
    alternatives: list[str] = field(default_factory=list)


def _alias_hits(table: pd.DataFrame, key: str) -> pd.DataFrame:
    hits = table["aliases"].map(lambda a: key in set(a) if a is not None else False)
    return table[hits]


def resolve_player(players: pd.DataFrame, text: str) -> Resolved:
    """Best player for ``text``: exact full name, else the alias match with the most games."""
    key = normalize_alias(text)
    exact = players[players["full_name"].map(lambda n: normalize_alias(str(n))) == key]
    hits = exact if len(exact) else _alias_hits(players, key)
    if hits.empty:
        raise LookupError(f"no player matches {text!r}")
    hits = hits.sort_values(["games", "last_season"], ascending=False)
    top = hits.iloc[0]
    alts = [f"{r.full_name} ({r.first_season}-{r.last_season})" for r in hits.iloc[1:6].itertuples()]
    return Resolved(int(top["personId"]), str(top["full_name"]), alts)


def resolve_team(teams: pd.DataFrame, text: str) -> Resolved:
    """Team whose aliases (any era's city, name, abbreviation, nickname) include ``text``."""
    key = normalize_alias(text)
    hits = _alias_hits(teams, key)
    if hits.empty:
        hits = teams[teams["full_name"].map(lambda n: key in normalize_alias(str(n)))]
    if hits.empty:
        raise LookupError(f"no team matches {text!r}")
    top = hits.iloc[0]
    return Resolved(int(top["teamId"]), str(top["full_name"]), list(hits["full_name"].iloc[1:]))


_FEET_INCHES = re.compile(r"^\s*(\d)\s*(?:'|ft|foot|feet|-|\s)\s*(\d{1,2})\s*(?:\"|in|inches)?\s*$")


def parse_height(text: str) -> float:
    """Height in inches from "6'7", "6-7", "6 7", "6 ft 7", or plain inches "79"."""
    t = text.strip().replace("’", "'").replace("”", '"')
    m = _FEET_INCHES.match(t)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    t = re.sub(r"(in|inches|\")$", "", t).strip()
    return float(t)


def parse_height_range(text: str) -> tuple[float, float]:
    """"6-6:6-8", "6'6\" to 6'8\"", "78:80" -> (78, 80). A single height h means h-1..h+1."""
    parts = re.split(r"\s*(?::|\bto\b|\.\.)\s*", text.strip())
    if len(parts) == 1:
        h = parse_height(parts[0])
        return h - 1, h + 1
    lo, hi = parse_height(parts[0]), parse_height(parts[1])
    return min(lo, hi), max(lo, hi)

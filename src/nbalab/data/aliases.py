"""Build lookup aliases so plain-English names resolve to player and team ids.

Aliases are generated from the names in the data using generic rules (full
name, last name, initials, common first-name diminutives, city shorthands).
Famous nicknames that no rule can produce ("King James", "Dubs") come from
the curated CSVs in ``reference/``, keyed by id. Aliases are stored
normalized; use :func:`normalize_alias` on user text before matching.

Aliases are deliberately ambiguous-tolerant: "curry" maps to several players.
The resolver (not this module) picks among candidates, e.g. by recency.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

FIRST_NAME_DIMINUTIVES: dict[str, list[str]] = {
    "alexander": ["alex"],
    "andrew": ["andy", "drew"],
    "anthony": ["tony"],
    "benjamin": ["ben"],
    "bradley": ["brad"],
    "cameron": ["cam"],
    "christopher": ["chris"],
    "daniel": ["dan", "danny"],
    "donald": ["don"],
    "douglas": ["doug"],
    "edward": ["ed", "eddie"],
    "frederick": ["fred"],
    "gerald": ["jerry"],
    "gregory": ["greg"],
    "jacob": ["jake"],
    "james": ["jim", "jimmy"],
    "jeffrey": ["jeff"],
    "jonathan": ["jon"],
    "joseph": ["joe"],
    "joshua": ["josh"],
    "kenneth": ["ken", "kenny"],
    "lawrence": ["larry"],
    "matthew": ["matt"],
    "michael": ["mike"],
    "nathan": ["nate"],
    "nathaniel": ["nate"],
    "nicholas": ["nick"],
    "nicolas": ["nick"],
    "patrick": ["pat"],
    "raymond": ["ray"],
    "richard": ["rich", "rick"],
    "robert": ["rob", "bob", "bobby"],
    "ronald": ["ron"],
    "samuel": ["sam"],
    "stephen": ["steph", "steve"],
    "steven": ["steve"],
    "terrence": ["terry"],
    "thomas": ["tom", "tommy"],
    "timothy": ["tim"],
    "william": ["will", "bill"],
    "zachary": ["zach"],
}

CITY_SHORTHANDS: dict[str, list[str]] = {
    "los angeles": ["la"],
    "new york": ["ny"],
    "golden state": ["gs"],
    "oklahoma city": ["okc"],
    "new orleans": ["nola"],
    "san antonio": ["sa"],
    "new jersey": ["nj"],
    "philadelphia": ["philly"],
}


def normalize_alias(text: str) -> str:
    """Lowercase, turn hyphens into spaces, drop dots/apostrophes, collapse whitespace.

    ``"Shai Gilgeous-Alexander"`` -> ``"shai gilgeous alexander"``,
    ``"D'Angelo"`` -> ``"dangelo"``, ``"P.J."`` -> ``"pj"``.
    """
    text = text.lower().replace("-", " ")
    text = re.sub(r"[.'`’,]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_suffix(tokens: list[str]) -> list[str]:
    """Drop generational suffixes (Jr., III, ...) from the end of a normalized name."""
    while len(tokens) > 1 and tokens[-1] in NAME_SUFFIXES:
        tokens = tokens[:-1]
    return tokens


def player_aliases(first_name: str, last_name: str) -> list[str]:
    """Generate normalized aliases for one player from their name alone.

    Produces: full name, full name without suffix, last name, first name,
    first-initial + last name, diminutive + last name (Stephen -> Steph Curry),
    and initials for names of three or more parts (Shai Gilgeous-Alexander -> SGA).
    """
    first = normalize_alias(first_name or "")
    last_tokens = strip_suffix(normalize_alias(last_name or "").split())
    last = " ".join(last_tokens)
    full = normalize_alias(f"{first_name or ''} {last_name or ''}")
    aliases = {full, f"{first} {last}".strip(), last, first}
    if first and last:
        aliases.add(f"{first[0]} {last}")
        for short in FIRST_NAME_DIMINUTIVES.get(first, []):
            aliases.add(f"{short} {last}")
        tokens = first.split() + last_tokens
        if len(tokens) >= 3:
            aliases.add("".join(t[0] for t in tokens))
    return sorted(a for a in aliases if a)


def team_aliases(city: str, name: str, abbrev: str) -> list[str]:
    """Generate normalized aliases for one team name era (city, nickname, abbreviation)."""
    city_n, name_n = normalize_alias(city), normalize_alias(name)
    aliases = {city_n, name_n, f"{city_n} {name_n}", normalize_alias(abbrev)}
    for short in CITY_SHORTHANDS.get(city_n, []):
        aliases.update({short, f"{short} {name_n}"})
    return sorted(a for a in aliases if a)


def load_nicknames(path: Path, id_col: str) -> pd.DataFrame:
    """Read a curated nickname CSV (``<id_col>,alias``) and normalize the aliases."""
    df = pd.read_csv(path, dtype={id_col: "int64", "alias": "string"}, comment="#")
    return df.assign(alias=df["alias"].map(normalize_alias))


def attach_nicknames(
    aliases: pd.Series, nicknames: pd.DataFrame, ids: pd.Series, id_col: str
) -> pd.Series:
    """Merge curated nicknames into each id's generated alias list (sorted, deduplicated)."""
    extra = nicknames.groupby(id_col)["alias"].agg(list)
    merged = [sorted(set(base) | set(extra.get(i, []))) for base, i in zip(aliases, ids)]
    return pd.Series(merged, index=aliases.index)

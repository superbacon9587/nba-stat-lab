"""Normalize ``PlayByPlay.parquet`` into one uniform event stream.

The raw file mixes two schemas (see ``docs/data_profile.md``):

- **Legacy** (1996 to early 2019): capitalized ``actionType``. A substitution
  is one row whose ``personId`` is the player *leaving*; the player entering is
  only a name in ``description`` (``"SUB: Odom FOR Jones"``). Substitutions
  made between periods are not recorded at all.
- **Modern** (2019-20 partial, all games from 2020-21): lowercase
  ``actionType``, separate ``in`` / ``out`` substitution rows with ids, and
  between-period substitutions recorded at the start of the period. Early
  2019-20 modern games have only ``out`` rows, with the entering player in
  ``subsInPersonId``.

Every event becomes one row of a small, uniform table:

``kind``
    ``"sub_out"``, ``"sub_in"`` or ``"ref"``. A ``ref`` is any event that proves
    the player was on the floor at that moment (shot, rebound, turnover,
    personal foul, steal, block, assist...). Technical fouls are *not* refs,
    because bench players can draw them.
``personId`` / ``playerName`` / ``inName`` / ``assistName``
    The player. Legacy sub-ins carry only ``inName`` and legacy assists only
    ``assistName``; both are resolved later against the game's roster (see
    :mod:`nbalab.data.lineups`). ``playerName`` is the actor's name as written
    in the play-by-play, used to build that name lookup.
``pts, fgm, fga, fg3m, fg3a, ftm, fta, ast``
    The player's box-score counting stats from this event.

All functions are pure so they can be tested without the raw file.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

REGULATION_PERIOD_SECONDS = 12 * 60
OVERTIME_PERIOD_SECONDS = 5 * 60
STAT_COLUMNS: tuple[str, ...] = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "ast")

# Raw columns the normalizer needs; read only these from the parquet file.
PBP_COLUMNS: tuple[str, ...] = (
    "gameId",
    "period",
    "clock",
    "actionNumber",
    "orderNumber",
    "actionType",
    "subType",
    "description",
    "personId",
    "playerName",
    "teamId",
    "shotResult",
    "shotValue",
    "assistPersonId",
    "subsInPersonId",
)

_CLOCK_RE = re.compile(r"^PT(\d+)M([\d.]+)S$")
_LEGACY_SUB_RE = r"^SUB:\s*(?P<inName>.*?)\s+FOR\s*(?P<outName>.*)$"
_LEGACY_ASSIST_RE = r"\(([^()]+?) \d+ AST\)"

# Action types (lowercased, stripped) whose actor must be on the floor.
_ON_FLOOR_ACTIONS = frozenset(
    {
        "made shot", "missed shot", "free throw", "rebound", "turnover", "foul",
        "jump ball", "violation",
        "2pt", "3pt", "freethrow", "steal", "block", "jumpball", "heave",
    }
)


def period_length(period: pd.Series) -> pd.Series:
    """Length of each period in seconds: 12 minutes for quarters 1-4, 5 for overtime."""
    return pd.Series(
        np.where(period <= 4, REGULATION_PERIOD_SECONDS, OVERTIME_PERIOD_SECONDS),
        index=period.index,
    )


def period_start(period: pd.Series) -> pd.Series:
    """Game seconds elapsed when each period begins (OT periods are 5 minutes)."""
    p = period.astype("int64")
    regulation = np.minimum(p - 1, 4) * REGULATION_PERIOD_SECONDS
    overtime = np.maximum(p - 5, 0) * OVERTIME_PERIOD_SECONDS
    return pd.Series(regulation + overtime, index=period.index, dtype="float64")


def parse_clock(clock: pd.Series) -> pd.Series:
    """Seconds remaining in the period from an ISO-8601 clock (``"PT06M40.00S"`` -> 400.0)."""
    parts = clock.astype("string").str.strip().str.extract(_CLOCK_RE)
    return parts[0].astype("float64") * 60 + parts[1].astype("float64")


def game_seconds(period: pd.Series, clock: pd.Series) -> pd.Series:
    """Seconds elapsed since tip-off for each event."""
    remaining = parse_clock(clock).clip(lower=0)
    remaining = np.minimum(remaining, period_length(period))
    return period_start(period) + period_length(period) - remaining


def _to_int_id(raw: pd.Series) -> pd.Series:
    """Parse a string id column to nullable Int64 (blank / non-numeric -> NA)."""
    return pd.to_numeric(raw.astype("string").str.strip(), errors="coerce").astype("Int64")


def _is_technical(sub_type: pd.Series) -> pd.Series:
    """True for technical-type fouls, which bench players can also be charged with."""
    s = sub_type.str.lower()
    return s.str.contains("technical", na=False) | s.str.contains("taunting", na=False)


def normalize_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw play-by-play rows (either schema) into the uniform event table.

    Output columns: ``gameId, period, t, seq, kind, personId, teamId, inName,
    assistName`` plus :data:`STAT_COLUMNS`. ``t`` is game seconds elapsed.
    Rows are sorted in game order. Events that are neither substitutions nor
    evidence of a player being on the floor are dropped. Period markers are
    dropped too; the period boundaries come from the period number.
    """
    df = raw.copy()
    action = df["actionType"].astype("string").str.strip()
    action_l = action.str.lower()
    sub_type = df["subType"].astype("string").str.strip().fillna("")
    desc = df["description"].astype("string").fillna("")
    is_modern = action.notna() & (action == action_l) & action.str.len().gt(0)

    base = pd.DataFrame(
        {
            "gameId": pd.to_numeric(df["gameId"], errors="coerce").astype("int64"),
            "period": df["period"].astype("int64"),
            "t": game_seconds(df["period"], df["clock"]),
            "seq": df["orderNumber"].astype("Float64").fillna(df["actionNumber"].astype("float64")),
            "personId": _to_int_id(df["personId"]),
            "teamId": _to_int_id(df["teamId"]),
            "playerName": df["playerName"].astype("string").str.strip(),
        },
        index=df.index,
    )

    stats = _event_stats(action_l, sub_type, desc, df["shotResult"], df["shotValue"])
    base = pd.concat([base, stats], axis=1)

    # Modern substitutions: one row each for in and out, both with ids.
    modern_sub = is_modern & (action_l == "substitution")
    modern_kind = pd.Series(
        np.where(sub_type.str.lower() == "in", "sub_in", "sub_out"), index=df.index
    )

    # Legacy substitutions: personId leaves, the entering player is a name.
    legacy_sub = ~is_modern & (action_l == "substitution")
    names = desc.str.extract(_LEGACY_SUB_RE)

    # On-floor evidence from the event's actor.
    on_floor = action_l.isin(_ON_FLOOR_ACTIONS) & ~(action_l.eq("foul") & _is_technical(sub_type))
    on_floor &= ~(action_l.eq("violation") & sub_type.str.lower().str.contains("delay", na=False))

    parts: list[pd.DataFrame] = []

    refs = base[on_floor].assign(kind="ref", inName=pd.NA)
    parts.append(refs)

    mod = base[modern_sub].assign(kind=modern_kind[modern_sub], inName=pd.NA)
    parts.append(mod.assign(**{c: 0 for c in STAT_COLUMNS}))

    # Early 2019-20 modern games have only "out" rows; the entering player's id
    # is in ``subsInPersonId`` on the same row ("O'Neale replaced by Ingles").
    subs_in_id = _to_int_id(df["subsInPersonId"])
    paired = modern_sub & subs_in_id.notna()
    if paired.any():
        extra = base[paired].assign(
            kind="sub_in", personId=subs_in_id[paired], playerName=pd.NA, inName=pd.NA,
            seq=base.loc[paired, "seq"] + 0.5,
        )
        parts.append(extra.assign(**{c: 0 for c in STAT_COLUMNS}))

    leg = base[legacy_sub]
    leg_out = leg.assign(kind="sub_out", inName=pd.NA, seq=leg["seq"])
    leg_in = leg.assign(
        kind="sub_in",
        personId=pd.array([pd.NA] * len(leg), dtype="Int64"),
        inName=names.loc[legacy_sub, "inName"].str.strip().replace("", pd.NA),
        playerName=pd.NA,
        seq=leg["seq"] + 0.5,
    )
    for part in (leg_out, leg_in):
        parts.append(part.assign(**{c: 0 for c in STAT_COLUMNS}))

    # Assists: modern rows carry the passer's id; legacy only the passer's name.
    made_fg = (base["fgm"] == 1) & on_floor
    modern_ast = made_fg & is_modern & _to_int_id(df["assistPersonId"]).notna()
    if modern_ast.any():
        a = base[modern_ast].assign(
            personId=_to_int_id(df.loc[modern_ast, "assistPersonId"]),
            kind="ref",
            inName=pd.NA,
            playerName=pd.NA,
        )
        parts.append(a.assign(**{c: 0 for c in STAT_COLUMNS}).assign(ast=1, seq=a["seq"] + 0.25))
    legacy_ast_name = desc.str.extract(_LEGACY_ASSIST_RE)[0].str.strip()
    legacy_ast = made_fg & ~is_modern & legacy_ast_name.notna()
    if legacy_ast.any():
        a = base[legacy_ast].assign(
            personId=pd.array([pd.NA] * int(legacy_ast.sum()), dtype="Int64"),
            kind="ref",
            inName=pd.NA,
            playerName=pd.NA,
            assistName=legacy_ast_name[legacy_ast],
        )
        parts.append(a.assign(**{c: 0 for c in STAT_COLUMNS}).assign(ast=1, seq=a["seq"] + 0.25))

    out = pd.concat(parts, ignore_index=True)
    if "assistName" not in out:
        out["assistName"] = pd.NA
    out = out.dropna(subset=["t"])
    out = out.sort_values(["gameId", "period", "t", "seq"], kind="stable").reset_index(drop=True)
    cols = [
        "gameId", "period", "t", "seq", "kind", "personId", "teamId", "playerName", "inName",
        "assistName",
    ]
    return out[cols + list(STAT_COLUMNS)]


def _event_stats(
    action_l: pd.Series,
    sub_type: pd.Series,
    desc: pd.Series,
    shot_result: pd.Series,
    shot_value: pd.Series,
) -> pd.DataFrame:
    """Box-score counting stats credited to the event's actor.

    Field goals: legacy ``Made Shot`` / ``Missed Shot`` or modern ``2pt`` / ``3pt``.
    Three-pointers come from ``shotValue`` (legacy) or the action type (modern).
    Free throws: legacy rows mark misses only with a leading ``MISS`` in the
    description; modern rows use ``shotResult``.
    """
    result = shot_result.astype("string").fillna("")
    made_text = result.str.contains("Made", na=False)
    value = pd.to_numeric(shot_value, errors="coerce").fillna(0)

    legacy_fg = action_l.isin(["made shot", "missed shot"])
    modern_fg = action_l.isin(["2pt", "3pt"])
    fga = legacy_fg | modern_fg
    fgm = (action_l == "made shot") | (modern_fg & made_text)
    three = (legacy_fg & (value == 3)) | (action_l == "3pt")

    legacy_ft = action_l == "free throw"
    modern_ft = action_l == "freethrow"
    fta = legacy_ft | modern_ft
    ftm = (legacy_ft & ~desc.str.upper().str.startswith("MISS")) | (modern_ft & made_text)

    as_int = lambda s: s.fillna(False).astype("int64")  # noqa: E731
    return pd.DataFrame(
        {
            "pts": as_int(fgm) * np.where(three, 3, 2) + as_int(ftm),
            "fgm": as_int(fgm),
            "fga": as_int(fga),
            "fg3m": as_int(fgm & three),
            "fg3a": as_int(fga & three),
            "ftm": as_int(ftm),
            "fta": as_int(fta),
            "ast": 0,
        },
        index=action_l.index,
    )

"""Reconstruct which ten players were on the floor at every moment of a game.

Input is one game's normalized event stream (:func:`nbalab.data.pbp.normalize_events`)
plus its box-score roster. Output is a list of *segments*: stretches of game
time during which neither team's five-man unit changed.

How the unit is tracked
-----------------------
1. **Period 1 starters** come from the box score (``startingPosition`` is set).
2. **Later periods**: the play-by-play does not always record substitutions
   made between periods (legacy games never do). A player is inferred to have
   started the period if his first event in it is anything other than a
   sub-in (he shot, fouled, or was subbed *out* before being subbed in). If
   fewer than five are found, players still on the floor at the end of the
   previous period who have no events at all in this period fill the gaps
   (a quiet player can play a whole quarter without an event).
3. **Substitutions** are applied in order.
4. **Repairs.** The raw feed sometimes drops a substitution. When a player
   records an event while not in the tracked unit, he is added. If the unit
   was short a player, he is assumed to have entered when the slot opened
   (earlier segments are back-filled). If that makes six, the player whose
   next event is a sub-in (so he must be on the bench) is removed; otherwise
   the player not seen for the longest time is removed.

Every repair is counted so the build can report reconstruction quality, and
reconstructed minutes are checked against box-score minutes downstream.

Important: this says who **shared the floor**. It says nothing about who
guarded whom.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import pandas as pd

from nbalab.data.pbp import STAT_COLUMNS, period_length, period_start

LINEUP_SIZE = 5


@dataclass
class Segment:
    """A stretch of game time with fixed lineups (``lineups[teamId]`` = set of personIds)."""

    period: int
    start: float
    end: float
    lineups: dict[int, set[int]]


@dataclass
class StatEvent:
    """A counting-stat event by ``personId`` that happened during ``segments[segment]``."""

    personId: int
    teamId: int
    segment: int
    stats: tuple[int, ...]


@dataclass
class GameReconstruction:
    """Segments, stat events and repair counters for one game."""

    gameId: int
    teams: tuple[int, int]
    segments: list[Segment] = field(default_factory=list)
    stat_events: list[StatEvent] = field(default_factory=list)
    repairs: dict[str, int] = field(
        default_factory=lambda: {
            "unresolved_sub_in": 0,
            "unresolved_assist": 0,
            "added_from_event": 0,
            "backfilled": 0,
            "evicted": 0,
            "short_period_start": 0,
        }
    )


# --------------------------------------------------------------------------- names


def _norm_name(name: object) -> str:
    return " ".join(str(name).lower().replace(".", "").split()) if isinstance(name, str) else ""


def build_name_index(roster: pd.DataFrame, events: pd.DataFrame) -> dict[int, dict[str, set[int]]]:
    """Map ``teamId -> normalized name -> personIds`` for resolving legacy name-only fields.

    Name variants per player: every ``playerName`` spelling he has in this game's
    play-by-play, his box-score last name, and ``"first last"``.
    """
    team_of = dict(zip(roster["personId"], roster["teamId"]))
    index: dict[int, dict[str, set[int]]] = {}

    def add(pid: int, name: object) -> None:
        key = _norm_name(name)
        if key and pid in team_of:
            index.setdefault(team_of[pid], {}).setdefault(key, set()).add(pid)

    for pid, first, last in zip(roster["personId"], roster["firstName"], roster["lastName"]):
        add(pid, last)
        add(pid, f"{first} {last}")
    named = events.dropna(subset=["personId", "playerName"])
    for pid, name in zip(named["personId"], named["playerName"]):
        add(int(pid), name)
    return index


def resolve_name(
    name: object,
    team: int,
    index: dict[int, dict[str, set[int]]],
    prefer: set[int] | None = None,
    avoid: set[int] | None = None,
) -> int | None:
    """Resolve a play-by-play name to a personId on ``team``.

    Exact (case/period-insensitive) match first, then a match on the last word
    of the name (``"Marc Morris"`` -> ``"morris"``). Ties are broken by
    preferring players in ``prefer`` and not in ``avoid`` (e.g. a sub-in should
    be someone currently on the bench). Returns ``None`` if still ambiguous or
    unknown.
    """
    key = _norm_name(name)
    names = index.get(team, {})
    if not key:
        return None
    candidates = set(names.get(key, set()))
    if not candidates:
        last = key.split()[-1]
        candidates = {p for n, ids in names.items() if n.split()[-1] == last for p in ids}
    if len(candidates) > 1 and prefer is not None:
        preferred = candidates & prefer
        candidates = preferred or candidates
    if len(candidates) > 1 and avoid is not None:
        candidates = (candidates - avoid) or candidates
    return next(iter(candidates)) if len(candidates) == 1 else None


# ------------------------------------------------------------------ reconstruction


def reconstruct_game(events: pd.DataFrame, roster: pd.DataFrame) -> GameReconstruction | None:
    """Rebuild on-floor lineups for one game.

    ``events``: normalized events for this game only.
    ``roster``: box-score rows for this game with ``personId, teamId, firstName,
    lastName, starter`` (bool) and ``minutes``. Players with no minutes are
    ignored. Returns ``None`` if the roster does not have exactly two teams.
    """
    played = roster[roster["minutes"].fillna(0) > 0]
    teams = tuple(sorted(played["teamId"].dropna().astype("int64").unique()))
    if len(teams) != 2:
        return None
    team_of: dict[int, int] = dict(zip(played["personId"].astype("int64"), played["teamId"].astype("int64")))
    minutes = dict(zip(played["personId"].astype("int64"), played["minutes"].astype("float64")))
    game_id = int(events["gameId"].iloc[0]) if len(events) else int(roster["gameId"].iloc[0])
    rec = GameReconstruction(gameId=game_id, teams=teams)  # type: ignore[arg-type]
    names = build_name_index(played, events)

    box_starters = {
        t: set(played.loc[played["starter"] & (played["teamId"] == t), "personId"].astype("int64"))
        for t in teams
    }
    rows = _prepare_rows(events, team_of, teams, names, rec)
    end_lineups: dict[int, set[int]] = {t: set() for t in teams}

    periods = sorted({r["period"] for r in rows}) if rows else []
    for period in periods:
        prows = [r for r in rows if r["period"] == period]
        if period == 1 and all(len(box_starters[t]) == LINEUP_SIZE for t in teams):
            starters = {t: set(box_starters[t]) for t in teams}
        else:
            starters = {
                t: _infer_period_starters(prows, t, end_lineups[t], minutes, team_of) for t in teams
            }
        end_lineups = _run_period(period, prows, starters, rec, names, team_of)
    return rec


def _prepare_rows(
    events: pd.DataFrame,
    team_of: dict[int, int],
    teams: tuple[int, ...],
    names: dict[int, dict[str, set[int]]],
    rec: GameReconstruction,
) -> list[dict]:
    """Convert events to dicts, attach each player's team, pre-resolve unambiguous names."""
    rows: list[dict] = []
    for e in events.to_dict("records"):
        pid = e["personId"]
        pid = int(pid) if pd.notna(pid) else None
        team = int(e["teamId"]) if pd.notna(e["teamId"]) else None
        if e["kind"] == "sub_in" and pid is None:
            if team not in teams:
                continue
            pid = resolve_name(e["inName"], team, names)  # may stay None; retried in-game
        elif pid is None and isinstance(e.get("assistName"), str):
            if team not in teams:
                continue
            pid = resolve_name(e["assistName"], team, names)
            if pid is None:
                rec.repairs["unresolved_assist"] += 1
                continue
        if pid is not None:
            if pid not in team_of:
                continue  # team-level event (team rebound) or player missing from box score
            team = team_of[pid]
        e = {**e, "personId": pid, "teamId": team}
        rows.append(e)
    return rows


def _infer_period_starters(
    prows: list[dict],
    team: int,
    prev_end: set[int],
    minutes: dict[int, float],
    team_of: dict[int, int],
) -> set[int]:
    """Players on ``team`` who began the period on the floor (see module docstring)."""
    first_kind: dict[int, str] = {}
    for r in prows:
        pid = r["personId"]
        if pid is not None and r["teamId"] == team and pid not in first_kind:
            first_kind[pid] = r["kind"]
    starters = [p for p, k in first_kind.items() if k != "sub_in"][:LINEUP_SIZE]
    if len(starters) < LINEUP_SIZE:
        silent = sorted(
            (p for p in prev_end if p not in first_kind and team_of.get(p) == team),
            key=lambda p: -minutes.get(p, 0.0),
        )
        starters += silent[: LINEUP_SIZE - len(starters)]
    return set(starters)


def _run_period(
    period: int,
    prows: list[dict],
    starters: dict[int, set[int]],
    rec: GameReconstruction,
    names: dict[int, dict[str, set[int]]],
    team_of: dict[int, int],
) -> dict[int, set[int]]:
    """Walk one period's events, emitting segments and stat events. Returns end lineups."""
    p = pd.Series([period])
    t0 = float(period_start(p).iloc[0])
    t_end = t0 + float(period_length(p).iloc[0])

    lineups = {t: set(s) for t, s in starters.items()}
    last_seen: dict[int, float] = {pid: t0 for s in lineups.values() for pid in s}
    # Open vacancies per team: segment indices at which the unit dropped below five.
    vacancies: dict[int, list[int]] = {t: [] for t in lineups}
    for t, s in lineups.items():
        if len(s) < LINEUP_SIZE:
            rec.repairs["short_period_start"] += 1
    # Future-event lookup for the eviction rule.
    future: dict[int, list[tuple[int, str]]] = {}
    for i, r in enumerate(prows):
        if r["personId"] is not None:
            future.setdefault(r["personId"], []).append((i, r["kind"]))

    def open_segment(t: float) -> None:
        rec.segments.append(Segment(period, t, t, {k: set(v) for k, v in lineups.items()}))

    open_segment(t0)
    first_seg_of_period = len(rec.segments) - 1
    for t, s in lineups.items():
        vacancies[t] = [first_seg_of_period] * (LINEUP_SIZE - len(s))

    def next_kind(pid: int, i: int) -> str | None:
        evs = future.get(pid, [])
        j = bisect.bisect_right(evs, (i, "~"))
        return evs[j][1] if j < len(evs) else None

    def changed() -> None:
        """Close the current segment at the current time and open a new one."""
        seg = rec.segments[-1]
        if seg.end == seg.start:
            seg.lineups = {k: set(v) for k, v in lineups.items()}  # zero-length: update in place
        else:
            open_segment(seg.end)

    def add(pid: int, team: int, i: int, via_event: bool) -> None:
        unit = lineups[team]
        if pid in unit:
            return
        unit.add(pid)
        if via_event:
            rec.repairs["added_from_event"] += 1
        if len(unit) <= LINEUP_SIZE and vacancies[team]:
            since = vacancies[team].pop(0)
            if via_event:
                for seg in rec.segments[since:]:
                    seg.lineups[team].add(pid)
                rec.repairs["backfilled"] += 1
        if len(unit) > LINEUP_SIZE:
            _evict(unit - {pid}, i, unit)

    def _evict(candidates: set[int], i: int, unit: set[int]) -> None:
        benched = [c for c in candidates if next_kind(c, i) == "sub_in"]
        pool = benched or list(candidates)
        victim = min(pool, key=lambda c: (last_seen.get(c, t0), c))
        unit.discard(victim)
        rec.repairs["evicted"] += 1

    for i, r in enumerate(prows):
        t = min(max(float(r["t"]), rec.segments[-1].end), t_end)
        rec.segments[-1].end = t
        pid, team, kind = r["personId"], r["teamId"], r["kind"]
        if team not in lineups:
            continue
        if kind == "sub_in" and pid is None:
            pid = resolve_name(r["inName"], team, names, avoid=lineups[team])
            if pid is None:
                rec.repairs["unresolved_sub_in"] += 1
                continue
        if pid is None:
            continue
        if kind == "sub_out":
            if pid in lineups[team]:
                lineups[team].discard(pid)
                changed()
                vacancies[team].append(len(rec.segments) - 1)
        elif kind == "sub_in":
            if pid not in lineups[team]:
                add(pid, team, i, via_event=False)
                changed()
        else:  # ref
            if pid not in lineups[team]:
                add(pid, team, i, via_event=True)
                changed()
            stats = tuple(int(r[c]) for c in STAT_COLUMNS)
            if any(stats):
                rec.stat_events.append(StatEvent(pid, team, len(rec.segments) - 1, stats))
        last_seen[pid] = t
    rec.segments[-1].end = t_end
    return {t: set(s) for t, s in lineups.items()}

"""Command-line front end for the query engines.

Examples::

    python -m nbalab.query.cli --player "Stephen Curry" --stat assists --opponent "San Antonio Spurs"
    python -m nbalab.query.cli --team "Boston Celtics" --stat points --opponent Lakers --last-n-seasons 5
    python -m nbalab.query.cli --player "Stephen Curry" --stat points --defender "LeBron James"
    python -m nbalab.query.cli --player "Stephen Curry" --stat points --defender-height "6-6:6-8"
    python -m nbalab.query.cli --player "Stephen Curry" --stat points --group-by day_of_week
    python -m nbalab.query.cli --effect back_to_back --stat points --last-n-seasons 5
    python -m nbalab.query.cli --json query.json

Names are resolved to ids here only for convenience. The engines see ids only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nbalab.query.data import PROCESSED_DIR, QueryData, load_query_data
from nbalab.query.engine import run_query
from nbalab.query.format import render_effect, render_split
from nbalab.query.resolve import parse_height_range, resolve_player, resolve_team
from nbalab.query.schema import StatQuery


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NBA Stat Lab split engine")
    who = p.add_mutually_exclusive_group()
    who.add_argument("--player", action="append", help="player name (repeatable)")
    who.add_argument("--team", action="append", help="team name (repeatable)")
    p.add_argument("--json", type=Path, help="read a StatQuery JSON file instead of flags")
    p.add_argument("--stat", action="append", help="stat (repeatable): points, assists, PRA, FG%%, TS%%, ...")
    p.add_argument("--opponent", action="append", help="opponent team (repeatable)")
    p.add_argument("--defender", help="named defender")
    p.add_argument("--opponent-on-court", help="opponent player on the floor")
    p.add_argument("--defender-height", help="range like 6-6:6-8 or 78:80, or one height (+/-1 inch)")
    p.add_argument("--defender-weight", help="range in lb like 200:230")
    p.add_argument("--defender-position", choices=["G", "F", "C"], action="append")
    p.add_argument("--teammate-in", help="teammate who played")
    p.add_argument("--teammate-out", help="teammate who did not play")
    p.add_argument("--venue", help="home team of the arena, or a city")
    p.add_argument("--home-away", choices=["home", "away"])
    p.add_argument("--day", action="append", help="day of week (repeatable)")
    p.add_argument("--month", type=int, action="append")
    p.add_argument("--seasons", help="season start years, e.g. 2019:2024")
    p.add_argument("--last-n-seasons", type=int)
    p.add_argument("--last-n-games", type=int)
    p.add_argument("--rest-days", help="range like 0:1")
    p.add_argument("--back-to-back", choices=["yes", "no"])
    p.add_argument("--playoffs", choices=["yes", "no"])
    p.add_argument("--min-minutes", type=float)
    p.add_argument("--position", choices=["G", "F", "C"], action="append", help="player position (variable_effect)")
    p.add_argument("--group-by", help="variable to split by, e.g. day_of_week, opponent_team, defender_height_bucket")
    p.add_argument("--effect", help="variable_effect mode: variable whose league-wide effect to estimate")
    p.add_argument("--no-minutes-control", action="store_true", help="variable_effect: do not hold minutes fixed")
    p.add_argument("--line", action="append", help="betting line stat=value, e.g. points=27.5")
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--as-json", action="store_true", help="print results as JSON")
    p.add_argument("--show-games", type=int, default=0, help="print the N most recent split games")
    return p


def query_from_args(a: argparse.Namespace, data: QueryData) -> tuple[StatQuery, list[str]]:
    """Build a StatQuery from CLI flags. Returns (query, assumptions made while resolving names)."""
    notes: list[str] = []

    def player(name: str) -> int:
        r = resolve_player(data.players, name)
        if r.alternatives:
            notes.append(f"'{name}' -> {r.name} (also matched: {', '.join(r.alternatives[:3])})")
        return r.id

    def team(name: str) -> int:
        r = resolve_team(data.teams, name)
        if r.alternatives:
            notes.append(f"'{name}' -> {r.name} (also matched: {', '.join(r.alternatives[:3])})")
        return r.id

    def rng(text: str) -> tuple[float, float]:
        lo, _, hi = text.partition(":")
        return float(lo), float(hi or lo)

    subject_type = "team" if a.team else "player"
    ids = [team(t) for t in a.team] if a.team else [player(n) for n in (a.player or [])]
    f: list[dict] = []
    if a.opponent:
        f.append({"type": "opponent_team", "team_ids": [team(t) for t in a.opponent]})
    if a.defender:
        f.append({"type": "defender_player", "person_id": player(a.defender)})
    if a.opponent_on_court:
        f.append({"type": "opponent_player_on_court", "person_id": player(a.opponent_on_court)})
    if a.defender_height:
        lo, hi = parse_height_range(a.defender_height)
        f.append({"type": "defender_height_range", "min_inches": lo, "max_inches": hi})
    if a.defender_weight:
        lo, hi = rng(a.defender_weight)
        f.append({"type": "defender_weight_range", "min_lbs": lo, "max_lbs": hi})
    if a.defender_position:
        f.append({"type": "defender_position", "positions": a.defender_position})
    if a.teammate_in:
        f.append({"type": "teammate", "person_id": player(a.teammate_in), "status": "in"})
    if a.teammate_out:
        f.append({"type": "teammate", "person_id": player(a.teammate_out), "status": "out"})
    if a.venue:
        try:
            f.append({"type": "venue", "team_id": team(a.venue)})
        except LookupError:
            f.append({"type": "venue", "city": a.venue})
    if a.home_away:
        f.append({"type": "home_away", "value": a.home_away})
    if a.day:
        f.append({"type": "day_of_week", "days": [d.capitalize() for d in a.day]})
    if a.month:
        f.append({"type": "month", "months": a.month})
    if a.seasons:
        lo, hi = rng(a.seasons)
        f.append({"type": "season_range", "start": int(lo), "end": int(hi)})
    if a.last_n_seasons:
        f.append({"type": "last_n_seasons", "n": a.last_n_seasons})
    if a.last_n_games:
        f.append({"type": "last_n_games", "n": a.last_n_games})
    if a.rest_days:
        lo, hi = rng(a.rest_days)
        f.append({"type": "rest_days", "min_days": int(lo), "max_days": int(hi)})
    if a.back_to_back:
        f.append({"type": "back_to_back", "value": a.back_to_back == "yes"})
    if a.playoffs:
        f.append({"type": "playoffs", "value": a.playoffs == "yes"})
    if a.min_minutes is not None:
        f.append({"type": "min_minutes", "minutes": a.min_minutes})
    if a.position:
        f.append({"type": "player_position", "positions": a.position})

    q: dict = {"subject_type": subject_type, "subject_ids": ids, "stats": a.stat or ["points"],
               "filters": f, "group_by": a.group_by}
    if a.effect:
        q.update(mode="variable_effect", effect_variable=a.effect, group_by=None)
    if a.line:
        lines = {k: float(v) for k, _, v in (ln.partition("=") for ln in a.line)}
        q["projection"] = {"lines": lines}
    return StatQuery.model_validate(q), notes


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    data = load_query_data(a.data_dir)
    if a.json:
        query, notes = StatQuery.model_validate_json(a.json.read_text()), []
    else:
        query, notes = query_from_args(a, data)
    for n in notes:
        print(f"(assumed) {n}")

    if query.mode == "variable_effect":
        from nbalab.query.variable_effect import run_variable_effect

        result = run_variable_effect(query, data, control_minutes=not a.no_minutes_control)
        print(json.dumps(result.to_dict(), indent=2, default=str) if a.as_json else render_effect(result))
        return 0

    for r in run_query(query, data):
        if a.as_json:
            print(json.dumps(r.to_dict(), indent=2, default=str))
            continue
        print(render_split(r))
        if a.show_games:
            cols = ["game_date", "opponent_name", "home_away", "minutes", *_stat_cols(r)]
            print(r.games[[c for c in cols if c in r.games]].tail(a.show_games).to_string(index=False))
        print()
    return 0


def _stat_cols(r) -> list[str]:
    return ["points", "assists", "reboundsTotal", "teamScore", "opponentScore", "def_height", "def_source"]


if __name__ == "__main__":
    sys.exit(main())

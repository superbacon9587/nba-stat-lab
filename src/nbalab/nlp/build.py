"""Turn a :class:`QueryDraft` (names and phrases as written) into a :class:`StatQuery`.

Everything here is deterministic: name -> id lookups, relative dates -> seasons,
heights -> inch ranges, default stats and lines, and consistency checks. Each
interpretation is recorded as an :class:`Assumption`, and anything the user
must settle is an :class:`UnresolvedItem`. The app shows both before running.

Consistency checks (flagged, never silently fixed):

- A defender must play for the opponent. "Curry vs Philly, his defender is
  LeBron" is flagged because LeBron plays for the Lakers.
- A player cannot face their own team, and a defender cannot be a teammate.
- "In San Francisco" means the Warriors' building. If the subject is a
  Warrior that makes it a home game, if the opponent is, an away game.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import ValidationError

from nbalab.data.config import BuildConfig
from nbalab.nlp.defaults import LineProvider
from nbalab.nlp.entities import EntityIndex, Resolution, SeasonSpan, clean_name
from nbalab.nlp.measures import (
    format_height, height_range, parse_height_inches, parse_weight_lbs, weight_range,
)
from nbalab.nlp.seasons import current_season, last_completed_season, last_n_completed, season_label
from nbalab.nlp.types import Assumption, QueryDraft, UnresolvedItem
from nbalab.query.schema import StatQuery
from nbalab.query.stats import GROUPABLE_VARIABLES, canonical_stat

FIRST_DATA_SEASON = BuildConfig().start_season
MAX_CANDIDATES = 6

# Everyday names for groupable variables -> catalog key.
VARIABLE_SYNONYMS: dict[str, str] = {
    "day": "day_of_week", "weekday": "day_of_week", "days_of_week": "day_of_week",
    "back_to_backs": "back_to_back", "b2b": "back_to_back", "rest": "rest_days",
    "home": "home_away", "home_vs_away": "home_away", "location": "home_away",
    "opponent": "opponent_team", "opponents": "opponent_team", "arena": "venue", "city": "venue",
    "season_type": "playoffs", "playoff": "playoffs", "starting": "starter",
    "defender_height": "defender_height_bucket", "defender_weight": "defender_weight_bucket",
    "seasons": "season", "year": "season", "months": "month", "week": "week_of_season",
}
# Filters that would contradict grouping by the same variable.
GROUPING_DROPS: dict[str, str] = {
    "back_to_back": "back_to_back", "home_away": "home_away", "day_of_week": "day_of_week",
    "month": "month", "rest_days": "rest_days", "playoffs": "playoffs",
}


TEAM_DEFAULT_SEASONS = 3  # team questions with no seasons use the last N completed seasons


@dataclass(frozen=True)
class BuildSettings:
    """Tunable defaults. ``height_tolerance_in`` turns "a 6'7 defender" into 78-80.

    ``active_within_seasons``: a player counts as active (and can be projected)
    if they played in one of this many most recent seasons.
    """

    height_tolerance_in: float = 1.0
    weight_tolerance_lb: float = 10.0
    default_stats: tuple[str, ...] = ("points",)
    default_projection_stats: tuple[str, ...] = ("points", "rebounds", "assists")
    active_within_seasons: int = 2


def canonical_variable(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = VARIABLE_SYNONYMS.get(key, key)
    return key


class QueryBuilder:
    """Builds one query. Create a new builder per prompt; it collects notes as it goes."""

    def __init__(
        self,
        index: EntityIndex,
        today: date,
        settings: BuildSettings = BuildSettings(),
        line_provider: LineProvider | None = None,
    ) -> None:
        self.index = index
        self.today = today
        self.settings = settings
        self.line_provider = line_provider
        self.assumptions: list[Assumption] = []
        self.unresolved: list[UnresolvedItem] = []

    # --------------------------------------------------------------- notes

    def assume(self, field: str, message: str, value: Any = None) -> None:
        self.assumptions.append(Assumption(field=field, message=message, value=value))

    def flag(self, kind: str, field: str, text: str, message: str, res: Resolution | None = None) -> None:
        self.unresolved.append(UnresolvedItem(
            kind=kind, field=field, text=text, message=message,  # type: ignore[arg-type]
            candidates=res.candidates[:MAX_CANDIDATES] if res else [],
            chosen_id=res.id if res and kind != "not_found" else None,
        ))

    # ------------------------------------------------------------ entities

    def resolve(self, text: str, kind: Literal["player", "team"], field: str,
                seasons: SeasonSpan | None) -> Resolution:
        """Look a name up and record how it was read (assumption or unresolved chip)."""
        res = (self.index.resolve_player if kind == "player" else self.index.resolve_team)(text, seasons)
        if res.status == "not_found":
            self.flag("not_found", field, text, f"No {kind} matches {text!r}.")
        elif res.status == "ambiguous":
            names = ", ".join(c.name for c in res.candidates[1:MAX_CANDIDATES])
            self.flag("ambiguous", field, text,
                      f"{text!r} could be several {kind}s. Using {res.chosen.name}; also: {names}.", res)
        elif clean_name(text) != clean_name(res.chosen.name):
            msg = f"Read {text!r} as {res.chosen.name}."
            norm = clean_name(text)
            if (kind == "team" and res.id in self.index.team_aliases.get(norm, [])
                    and res.id not in self.index.current_team_aliases.get(norm, [])):
                msg = (f"Read {text!r} as the franchise now called the {res.chosen.name} (same franchise id). "
                       "Results show the name the team had in each season.")
            others = self.close_alternatives(res, text) if res.status == "clear" else []
            if others:
                msg += f" (Also matches {', '.join(others)}.)"
            self.assume(field, msg, res.id)
        return res

    def historical_team_named(self, d: QueryDraft) -> bool:
        """True if the subject was named by a former franchise name ("SuperSonics"): keep all seasons then."""
        for text in d.subjects:
            norm = clean_name(text)
            if norm in self.index.team_aliases and norm not in self.index.current_team_aliases:
                return True
        return False

    def close_alternatives(self, res: Resolution, text: str) -> list[str]:
        """Runner-up names worth mentioning: only when they are a genuinely close call.

        Spelling scores tie on exact aliases ("Curry", "OKC"), so closeness also
        needs a real-world signal: for teams the alternative must be a *current*
        name for this text (OKC once hosted the Hornets, which does not count);
        for players it must have played at least 40% as many games.
        """
        chosen, norm = res.candidates[0], clean_name(text)
        out = []
        for c in res.candidates[1:4]:
            if chosen.score - c.score >= self.index.settings.score_gap:
                continue
            if c.kind == "team" and c.id not in self.index.current_team_aliases.get(norm, []):
                continue
            if c.kind == "player" and self.player_games(c.id) < 0.4 * self.player_games(chosen.id):
                continue
            out.append(c.name)
        return out

    def player_games(self, pid: int) -> float:
        players = self.index.players
        return float(players.at[pid, "games"]) if "games" in players and pid in players.index else 0.0

    def resolve_ids(self, names: list[str], kind: Literal["player", "team"], field: str,
                    seasons: SeasonSpan | None) -> list[int]:
        ids = [self.resolve(n, kind, field, seasons).id for n in names]
        return [i for i in dict.fromkeys(ids) if i is not None]

    # --------------------------------------------------------------- dates

    def season_span(self, d: QueryDraft) -> SeasonSpan | None:
        """Season range (starting years, inclusive) from the draft's time fields."""
        cur, done = current_season(self.today), last_completed_season(self.today)
        if d.last_n_seasons:
            start, end = last_n_completed(d.last_n_seasons, self.today)
            self.assume("seasons", f"'Last {d.last_n_seasons}' read as the last {d.last_n_seasons} completed "
                        f"seasons: {season_label(start)} to {season_label(end)}.", [start, end])
            return start, end
        if d.relative_season == "this":
            note = "" if cur > done else " (the next season has not started)"
            self.assume("seasons", f"'This season' is {season_label(cur)}{note}.", [cur, cur])
            return cur, cur
        if d.relative_season == "last":
            self.assume("seasons", f"'Last season' is {season_label(done)}.", [done, done])
            return done, done
        if d.season_start is None and d.season_end is None:
            return None
        start = d.season_start if d.season_start is not None else FIRST_DATA_SEASON
        end = d.season_end if d.season_end is not None else cur
        if start > end:
            start, end = end, start
        if d.season_end is None:
            self.assume("seasons", f"Seasons {season_label(start)} through {season_label(end)} (the latest).",
                        [start, end])
        return start, end

    # ------------------------------------------------------------- stats

    def canonical_stats(self, names: list[str], kind: str) -> list[str]:
        out: list[str] = []
        for s in names:
            try:
                out.append(canonical_stat(s, kind))
            except ValueError:
                self.flag("unsupported", "stats", s, f"{s!r} is not a {kind} stat this app tracks.")
        return list(dict.fromkeys(out))

    def lines(self, d: QueryDraft, kind: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for ln in d.lines:
            try:
                out[canonical_stat(ln.stat, kind)] = float(ln.line)
            except ValueError:
                self.flag("unsupported", "lines", ln.stat, f"No stat {ln.stat!r} to set a line on.")
        return out

    # --------------------------------------------------------------- build

    def build(self, d: QueryDraft) -> StatQuery | None:
        for note in d.notes:
            self.assume("parser_note", note)
        seasons = self.season_span(d)
        kind, subject_ids = self.subjects(d, seasons)
        mode, group_by, effect = self.mode(d, bool(subject_ids))
        if kind == "team" and seasons is None and mode != "variable_effect" and not self.historical_team_named(d):
            seasons = last_n_completed(TEAM_DEFAULT_SEASONS, self.today)
            self.assume("seasons", f"No seasons given; team questions default to the last {TEAM_DEFAULT_SEASONS} "
                        f"seasons ({season_label(seasons[0])} to {season_label(seasons[1])}) because team styles change "
                        "fast. Edit the seasons chip to widen it.", list(seasons))
        if mode == "projection" and kind == "player" and self.history_only(subject_ids, d):
            mode = "split"

        stats = self.canonical_stats(d.stats, kind)
        lines = self.lines(d, kind) if mode == "projection" else {}
        stats = list(dict.fromkeys([*stats, *lines]))
        if not stats:
            stats = self.default_stats(kind, mode)

        filters, ctx = self.filters(d, kind, subject_ids, seasons, mode)
        drop = GROUPING_DROPS.get(group_by or effect or "")
        filters = [f for f in filters if f["type"] != drop]

        projection = None
        if mode == "projection":
            projection = {"lines": self.fill_lines(lines, stats, kind, subject_ids), "context": ctx}

        if mode != "variable_effect" and not subject_ids:
            if not any(u.field == "subject" for u in self.unresolved):
                self.flag("not_found", "subject", "", "No player or team found in the question.")
            return None
        try:
            return StatQuery(
                subject_type=kind, subject_ids=subject_ids, stats=stats, filters=filters,
                group_by=group_by, effect_variable=effect, projection=projection, mode=mode,
            )
        except ValidationError as e:
            msg = "; ".join(err["msg"] for err in e.errors())
            self.flag("unsupported", "query", "", f"Could not build a valid query: {msg}")
            return None

    def subjects(self, d: QueryDraft, seasons: SeasonSpan | None) -> tuple[str, list[int]]:
        """Resolve subjects. If a "player" is really a team name (or vice versa), switch."""
        kind: str = d.subject_type
        if d.subjects and kind == "player" and all(
            self.index.resolve_player(n, seasons).status == "not_found"
            and self.index.resolve_team(n, seasons).status != "not_found" for n in d.subjects
        ):
            kind = "team"
            self.assume("subject_type", "The subject is a team, so team stats are used.", "team")
        return kind, self.resolve_ids(d.subjects, kind, "subject", seasons)  # type: ignore[arg-type]

    def mode(self, d: QueryDraft, has_subject: bool) -> tuple[str, str | None, str | None]:
        """Pick split / variable_effect / projection and the grouping variable.

        "How does X affect <one player>" is a split grouped by X for that player.
        "How does X affect <everyone / all guards>" is a league-wide variable_effect.
        """
        mode = d.mode
        group_by = canonical_variable(d.group_by)
        effect = canonical_variable(d.effect_variable)
        for name in (group_by, effect):
            if name is not None and name not in GROUPABLE_VARIABLES:
                self.flag("unsupported", "group_by", name,
                          f"Cannot break results down by {name!r}; options: {sorted(GROUPABLE_VARIABLES)}.")
        group_by = group_by if group_by in GROUPABLE_VARIABLES else None
        effect = effect if effect in GROUPABLE_VARIABLES else None
        if mode == "variable_effect" and has_subject:
            group_by, effect, mode = group_by or effect, None, "split"
            if group_by:
                self.assume("mode", f"For a named subject, the effect is shown as a split by {group_by}.", group_by)
        elif mode != "projection" and not has_subject and (effect or group_by):
            effect, group_by, mode = effect or group_by, None, "variable_effect"
            self.assume("mode", f"League-wide effect of {effect} on a typical player.", effect)
        elif mode == "variable_effect":
            group_by = None
        return mode, group_by, effect

    def history_only(self, subject_ids: list[int], d: QueryDraft) -> bool:
        """A projection needs an active player. For a retired one, show history instead."""
        cutoff = current_season(self.today) - self.settings.active_within_seasons + 1
        retired = [p for p in subject_ids if int(self.index.players.at[p, "last_season"]) < cutoff]
        for p in retired:
            last = int(self.index.players.at[p, "last_season"])
            lines = ", ".join(f"{ln.stat} {ln.line:g}" for ln in d.lines)
            note = f" The line ({lines}) is not used." if lines else ""
            self.assume("mode", f"{self.index.player_name(p)} last played in {season_label(last)}, so this "
                        f"shows the history in these conditions, not a forecast.{note}", "split")
        return bool(retired)

    def default_stats(self, kind: str, mode: str) -> list[str]:
        names = self.settings.default_projection_stats if mode == "projection" else self.settings.default_stats
        stats = [canonical_stat(s, kind) for s in names]
        self.assume("stats", f"No stat named; using {', '.join(stats)}.", stats)
        return stats

    def fill_lines(self, lines: dict[str, float], stats: list[str], kind: str, subject_ids: list[int]) -> dict[str, float]:
        """Give every projected stat a line: the user's, else a default from recent games."""
        out = dict(lines)
        for stat in stats:
            if stat in out:
                continue
            value = None
            if self.line_provider is not None and len(subject_ids) == 1:
                value = self.line_provider(kind, subject_ids[0], stat, self.today)
            if value is None:
                self.assume(f"lines.{stat}", f"No {stat} line given and no recent games to set one; "
                            "the projection shows the median and intervals only.")
            else:
                out[stat] = value
                self.assume(f"lines.{stat}", f"No {stat} line given; using {value} "
                            "(recent 20-game average, rounded to a .5 line).", value)
        return out

    # ------------------------------------------------------------- filters

    def filters(self, d: QueryDraft, kind: str, subject_ids: list[int], seasons: SeasonSpan | None,
                mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Split filters, plus the projection context (its fields are not repeated as filters)."""
        f: list[dict[str, Any]] = []
        ctx: dict[str, Any] = {}
        proj = mode == "projection"
        is_player = kind == "player"

        opp = self.resolve_ids(d.opponent_teams, "team", "opponent", seasons)
        if proj and len(opp) == 1:
            ctx["opponent_team_id"] = opp[0]
        elif opp:
            f.append({"type": "opponent_team", "team_ids": opp})

        for pid in self.resolve_ids(d.opponent_players, "player", "opponent_player", seasons):
            f.append({"type": "opponent_player_on_court", "person_id": pid})

        defenders = self.resolve_ids(d.defenders, "player", "defender", seasons)
        if defenders and not is_player:
            self.flag("unsupported", "defender", ", ".join(d.defenders),
                      "Defender filters apply to a player, not a team. Ignored.")
            defenders = []
        for i, pid in enumerate(defenders):
            if proj and i == 0:
                ctx["defender_person_id"] = pid
            else:
                f.append({"type": "defender_player", "person_id": pid})

        f += self.body_filters(d, is_player)
        for pid in self.resolve_ids(d.teammates_in, "player", "teammate", seasons):
            f.append({"type": "teammate", "person_id": pid, "status": "in"})
        for pid in self.resolve_ids(d.teammates_out, "player", "teammate", seasons):
            f.append({"type": "teammate", "person_id": pid, "status": "out"})
        if d.subject_positions:
            if is_player:
                f.append({"type": "player_position", "positions": list(dict.fromkeys(d.subject_positions))})

        home_away = self.venue(d, f, ctx, kind, subject_ids, opp, proj)
        if home_away:
            if proj:
                ctx["home_away"] = home_away
            else:
                f.append({"type": "home_away", "value": home_away})
        f += self.when_filters(d, is_player, ctx, proj)
        if seasons is not None:
            f.append({"type": "season_range", "start": seasons[0], "end": seasons[1]})

        self.check_conflicts(kind, subject_ids, opp, defenders, proj, seasons)
        return f, ctx

    def body_filters(self, d: QueryDraft, is_player: bool) -> list[dict[str, Any]]:
        """Defender height / weight / position filters."""
        f: list[dict[str, Any]] = []
        wants = d.defender_height or d.defender_weight or d.defender_positions
        if wants and not is_player:
            self.flag("unsupported", "defender", d.defender_height or d.defender_weight or "",
                      "Defender filters apply to a player, not a team. Ignored.")
            return f
        if d.defender_height:
            inches = parse_height_inches(d.defender_height)
            if inches is None:
                self.flag("unsupported", "defender_height", d.defender_height,
                          f"Could not read {d.defender_height!r} as a height.")
            else:
                comp = d.defender_height_comparison or "about"
                lo, hi = height_range(inches, comp, self.settings.height_tolerance_in)
                f.append({"type": "defender_height_range", "min_inches": lo, "max_inches": hi})
                self.assume("defender_height", f"{d.defender_height!r} = {inches:g} in; defenders "
                            f"{format_height(lo)} to {format_height(hi)} ({lo:g}-{hi:g} in).", [lo, hi])
        if d.defender_weight:
            lbs = parse_weight_lbs(d.defender_weight)
            if lbs is None:
                self.flag("unsupported", "defender_weight", d.defender_weight,
                          f"Could not read {d.defender_weight!r} as a weight.")
            else:
                lo, hi = weight_range(lbs, d.defender_weight_comparison or "about", self.settings.weight_tolerance_lb)
                f.append({"type": "defender_weight_range", "min_lbs": lo, "max_lbs": hi})
                self.assume("defender_weight", f"Defenders {lo:g}-{hi:g} lb.", [lo, hi])
        if d.defender_positions:
            f.append({"type": "defender_position", "positions": list(dict.fromkeys(d.defender_positions))})
        return f

    def when_filters(self, d: QueryDraft, is_player: bool, ctx: dict[str, Any], proj: bool) -> list[dict[str, Any]]:
        """Day, month, rest, playoffs, recent games, minutes."""
        f: list[dict[str, Any]] = []
        if d.days_of_week:
            f.append({"type": "day_of_week", "days": list(dict.fromkeys(d.days_of_week))})
        months = [m for m in dict.fromkeys(d.months) if 1 <= m <= 12]
        if months:
            f.append({"type": "month", "months": months})
        if d.back_to_back is not None:
            if proj:
                ctx["back_to_back"] = d.back_to_back
            else:
                f.append({"type": "back_to_back", "value": d.back_to_back})
        if d.rest_days_min is not None or d.rest_days_max is not None:
            lo = d.rest_days_min if d.rest_days_min is not None else 0
            hi = d.rest_days_max if d.rest_days_max is not None else 99
            if proj and lo == hi:
                ctx["rest_days"] = lo
            else:
                f.append({"type": "rest_days", "min_days": lo, "max_days": hi})
        if d.playoffs is not None:
            f.append({"type": "playoffs", "value": d.playoffs})
        if d.last_n_games:
            f.append({"type": "last_n_games", "n": d.last_n_games})
        if d.min_minutes is not None and is_player:
            f.append({"type": "min_minutes", "minutes": d.min_minutes})
        return f

    # --------------------------------------------------------------- venue

    def subject_teams(self, kind: str, subject_ids: list[int]) -> set[int]:
        """The subject's current team(s); for a team subject, itself."""
        if kind == "team":
            return set(subject_ids)
        season = current_season(self.today)
        return set().union(*(self.index.teams_in_season(p, season) for p in subject_ids)) if subject_ids else set()

    def venue(self, d: QueryDraft, f: list[dict[str, Any]], ctx: dict[str, Any], kind: str,
              subject_ids: list[int], opp: list[int], proj: bool) -> str | None:
        """Add the venue and infer home/away from it. Returns the home/away value to use."""
        home_away = d.home_away
        if not d.venue:
            return home_away
        teams = self.index.resolve_venue(d.venue, current_season(self.today) if proj else None)
        if not teams:
            self.flag("not_found", "venue", d.venue, f"No NBA arena found for {d.venue!r}.")
            return home_away
        if len(teams) > 1:
            city = self.index.teams.at[teams[0], "city"]
            f.append({"type": "venue", "city": str(city)})
            self.assume("venue", f"{d.venue!r} has several NBA teams; filtering on the city {city}.", str(city))
            return home_away
        venue_team = teams[0]
        if proj:
            ctx["venue_team_id"] = venue_team
        else:
            f.append({"type": "venue", "team_id": venue_team})
        mine = self.subject_teams(kind, subject_ids)
        inferred = "home" if venue_team in mine else "away" if venue_team in opp else None
        where = f"{d.venue!r} is the {self.index.team_name(venue_team)} home arena"
        if inferred is None:
            if proj:
                self.assume("venue", f"{where}; neither team plays there, so treated as a neutral site.", venue_team)
            return home_away
        if home_away and home_away != inferred:
            self.flag("conflict", "home_away", d.venue,
                      f"{where}, which makes this a {inferred} game, but the question says {home_away}.")
            return home_away
        if proj or home_away is None:
            article = "an" if inferred == "away" else "a"
            self.assume("home_away", f"{where}, so this is {article} {inferred} game.", inferred)
        return inferred if proj else home_away

    # ----------------------------------------------------------- conflicts

    def check_conflicts(self, kind: str, subject_ids: list[int], opp: list[int], defenders: list[int],
                        proj: bool, seasons: SeasonSpan | None) -> None:
        """Flag combinations that cannot happen (they would give an empty or wrong split)."""
        mine = self.subject_teams(kind, subject_ids)
        season = current_season(self.today)
        for t in opp:
            if proj and t in mine:
                self.flag("conflict", "opponent", self.index.team_name(t),
                          f"The subject plays for the {self.index.team_name(t)}, so they cannot be the opponent.")
            elif kind == "team" and t in subject_ids:
                self.flag("conflict", "opponent", self.index.team_name(t), "A team cannot play itself.")
        for pid in defenders:
            name = self.index.player_name(pid)
            now = self.index.teams_in_season(pid, min(season, int(self.index.players.at[pid, "last_season"])))
            if proj and now & mine:
                self.flag("conflict", "defender", name, f"{name} is a teammate of the subject, not a defender.")
                self.unresolved[-1].chosen_id = pid
            for t in opp:
                self.defender_team_conflict(pid, t, now, proj, seasons)

    def defender_team_conflict(self, pid: int, team_id: int, now: set[int], proj: bool,
                               seasons: SeasonSpan | None) -> None:
        name, team = self.index.player_name(pid), self.index.team_name(team_id)
        stint = self.index.ever_played_for(pid, team_id)
        if proj and team_id not in now:
            current = " and ".join(self.index.team_name(t) for t in sorted(now)) or "no current team"
            past = f" (With them {season_label(stint[0])} to {season_label(stint[1])}.)" if stint else ""
            self.flag("conflict", "defender", name,
                      f"{name} plays for the {current}, not the {team}, so cannot defend in this game.{past} "
                      "Pick a different defender or opponent.")
            self.unresolved[-1].chosen_id = pid
        elif not proj and (stint is None or (seasons and not (stint[0] <= seasons[1] and stint[1] >= seasons[0]))):
            self.flag("conflict", "defender", name,
                      f"{name} never played for the {team} in the seasons asked about, so no games match both.")
            self.unresolved[-1].chosen_id = pid

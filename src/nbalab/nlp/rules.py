"""Rule-based prompt parser: regular expressions plus alias matching, no API key.

It exists so the public demo still works when the Anthropic key is missing or
out of credit. It fills the same :class:`QueryDraft` as the Claude parser, so
everything after parsing (name lookup, dates, defaults, checks) is shared.

How it reads a prompt, in order (each step marks the text it used so later
steps skip it):

1. Betting lines ("exceeds 27.5 points"), heights and weights.
2. Time: "last 5 years", "since 2020", "2023-24", "this season", "last 10 games".
3. Grouping and effects: "how does day of week affect ...", "home vs away".
4. Days, months, home/away, back-to-backs, rest, playoffs, positions, minutes.
5. Stats ("ppg", "scoring", "3P%").
6. Names: every run of 1-4 words that is a known player/team/venue alias,
   then a fuzzy pass over leftover capitalized words for misspellings.
   Each name gets a role from the words around it: "against X" (opponent),
   "his defender is X" / "X as a defender" (defender), "without X" (teammate
   out), "in X" (venue), otherwise the first name is the subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from rapidfuzz import fuzz

from nbalab.data.aliases import normalize_alias
from nbalab.nlp.entities import EntityIndex, clean_name
from nbalab.nlp.measures import comparison_before, find_height, find_weight
from nbalab.nlp.types import DraftLine, QueryDraft

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
}
NUM = r"(\d+|" + "|".join(NUMBER_WORDS) + r")"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

# Stat phrases -> a name the stat catalog understands. Longest phrases are tried first.
STAT_PHRASES: dict[str, str] = {
    "points per game": "points", "points rebounds and assists": "pra", "points rebounds assists": "pra",
    "pts+reb+ast": "pra", "p+r+a": "pra", "pra": "pra", "points and rebounds": "pr",
    "points allowed": "opponent_score", "total points": "total_points",
    "points": "points", "point": "points", "pts": "points", "ppg": "points",
    "scoring": "points", "score": "points", "scores": "points", "scored": "points",
    "assists": "assists", "assist": "assists", "apg": "assists", "ast": "assists", "dimes": "assists",
    "offensive rebounds": "offensive_rebounds", "defensive rebounds": "defensive_rebounds",
    "rebounds": "rebounds", "rebound": "rebounds", "rebounding": "rebounds", "rpg": "rebounds",
    "reb": "rebounds", "boards": "rebounds",
    "three point attempts": "threes_attempted", "3 point attempts": "threes_attempted", "3pa": "threes_attempted",
    "three point percentage": "three_pct", "3 point percentage": "three_pct", "3p%": "three_pct",
    "3pt%": "three_pct", "three pointers": "threes", "three-pointers": "threes", "3 pointers": "threes",
    "3-pointers": "threes", "threes": "threes", "3pm": "threes", "3s": "threes",
    "steals": "steals", "spg": "steals", "blocks": "blocks", "bpg": "blocks", "stocks": "stocks",
    "turnovers": "turnovers", "tov": "turnovers", "minutes": "minutes", "mpg": "minutes",
    "field goal percentage": "fg_pct", "shooting percentage": "fg_pct", "fg%": "fg_pct",
    "free throw percentage": "ft_pct", "ft%": "ft_pct", "free throws": "ftm",
    "true shooting": "ts_pct", "ts%": "ts_pct", "efg%": "efg_pct", "effective field goal": "efg_pct",
    "plus minus": "plus_minus", "plus-minus": "plus_minus", "+/-": "plus_minus",
    "fouls": "fouls", "pace": "pace", "offensive rating": "off_rating", "defensive rating": "def_rating",
    "net rating": "net_rating", "margin": "margin", "point differential": "margin",
}
_STAT_ALT = "|".join(re.escape(p) for p in sorted(STAT_PHRASES, key=len, reverse=True))
STAT_RE = re.compile(rf"(?<![\w%+/])(?:{_STAT_ALT})(?![\w%])", re.I)

LINE_RE = re.compile(
    rf"(?P<cue>over|exceeds?|exceeding|more than|at least|above|under|below|fewer than|less than|hits?|gets?|scores?)?"
    rf"\s*(?P<n>\d+(?:\.\d+)?)\s*(?P<plus>\+)?\s*(?P<stat>{_STAT_ALT})(?![\w%])", re.I,
)
PROJECTION_RE = re.compile(
    r"\b(?:expect(?:ed|ation)?|project(?:ed|ion)?|predict(?:ed|ion)?|forecast|chances?|probability|odds|"
    r"likel(?:y|ihood)|over/under|o/u|how might|how would|how will|will he|will they|going to|next game|"
    r"tonight|tomorrow|this game|upcoming|how often|hit rate|prop)\b|(?:^|[.?!]\s+)will\b", re.I,
)
BARE_LINE_RE = re.compile(
    r"\b(?:(?:prop\s+)?line|o/u|over/under)\s*(?:of|at|is|=|:)?\s*(?P<n>\d+(?:\.\d+)?)(?![\d.])(?!\s*%)"
    r"|\b(?P<n2>\d+(?:\.\d+)?)\s*(?:point\s+)?line\b", re.I,
)
EFFECT_RE = re.compile(r"\b(?:affects?|impacts?|effects?|influences?|matters?|changes?|depend(?:s)? on)\b", re.I)

# Variable phrases for "how does X affect ..." / "by X". Order matters (longest first).
VARIABLE_PHRASES: list[tuple[str, str]] = [
    (r"days? of (?:the )?week|weekdays?|which day", "day_of_week"),
    (r"back[\s-]to[\s-]backs?|b2bs?|second nights?", "back_to_back"),
    (r"days? of rest|rest days?|rest", "rest_days"),
    (r"home (?:vs\.?|versus|v\.?|or|and|/|compared to) (?:the )?(?:away|road)|home[\s/-]+away|home court", "home_away"),
    (r"playoffs? (?:vs\.?|versus|or|and) (?:the )?regular season|regular season (?:vs\.?|versus|or|and) (?:the )?playoffs?", "playoffs"),
    (r"defender height|height of (?:the |his )?defenders?", "defender_height_bucket"),
    (r"defender weight", "defender_weight_bucket"),
    (r"defender position", "defender_position"),
    (r"starting|starter|coming off the bench|bench", "starter"),
    (r"months?", "month"),
    (r"week of (?:the )?season", "week_of_season"),
    (r"seasons?|years?", "season"),
    (r"opponents?|each team|every team", "opponent_team"),
    (r"venues?|arenas?|cit(?:y|ies)", "venue"),
]
BY_RE = re.compile(r"\b(?:by|per|each|every|across|split by|broken down by)\s+(?:the\s+)?$", re.I)
ALWAYS_GROUP = {"home_away", "playoffs"}  # "home vs away" is a comparison even without "affect"

POSITION_WORDS = {
    "point guards": "G", "shooting guards": "G", "guards": "G", "guard": "G",
    "small forwards": "F", "power forwards": "F", "forwards": "F", "forward": "F", "wings": "F",
    "centers": "C", "center": "C", "bigs": "C", "big men": "C",
}
_POS_ALT = "|".join(sorted(POSITION_WORDS, key=len, reverse=True))
SUBJECT_POSITION_RE = re.compile(rf"\b(?:all|for|among|league[\s-]wide|every|nba)\s+(?:the\s+)?({_POS_ALT})\b", re.I)
DEFENDER_POSITION_RE = re.compile(
    rf"\b(?:against|vs\.?|versus|guarded by|defended by)\s+(?:an?\s+|opposing\s+)?({_POS_ALT})(?:\s+defenders?)?\b", re.I)

# Words that must not be read as names when written in lowercase or at the start of a sentence.
BLOCKED_WORDS = set("""
a an the and or but if of to in on at by for from with without vs versus against as is are was were be been
what whats how who whom which when where why does do did doing will would could should can may might must
his her their he she they him them it its this that these those my your our me we you i
avg average mean total per game games season seasons year years day days week weeks month months
home away road rest back all every each league nba team teams player players guard guards forward forwards
center centers big bigs wing wings defender defenders defense matchup chance chances expected expect over under
more less most least much many than then also just only even still since last past next this previous
good bad best worst better worse high low long short young old early late new big little great
rose love green brown white black king price bell wall hill house rice ball young smart hardaway
day week time night morning points point score scoring shot shots head field court floor bench line
""".split())

DEFENDER_AFTER = re.compile(
    r"^\s*(?:(?:'|’)s)?\s*(?:as|being)\s+(?:a|an|his|her|the)?\s*(?:primary\s+|main\s+)?defender|"
    r"^\s+(?:is\s+)?(?:guarding|defending|guards|defends|on the ball)\b", re.I)
DEFENDER_BEFORE = re.compile(
    r"(?:defender\s+(?:is|was|being|will be)|defended by|guarded by|guarding him is|matched up (?:against|with|on)|"
    r"match(?:ed)?[\s-]?up (?:with|against|vs\.?)|primary defender|defender)\s*(?:the\s+)?$", re.I)
TEAMMATE_OUT_BEFORE = re.compile(r"\bwithout\s+$", re.I)
TEAMMATE_OUT_AFTER = re.compile(r"^\s+(?:is\s+|was\s+)?(?:out|sits|sitting|injured|inactive|resting|off)\b", re.I)
TEAMMATE_IN_BEFORE = re.compile(r"\b(?:with|alongside|next to|playing with)\s+$", re.I)
OPPONENT_BEFORE = re.compile(
    r"(?:against|vs\.?|versus|v\.?|facing|faces|face|playing|play|plays|played|opponent|opponents|@)\s+(?:the\s+)?$", re.I)
AWAY_AT_BEFORE = re.compile(r"(?:\bat|@)\s+(?:the\s+)?$", re.I)
VENUE_BEFORE = re.compile(r"(?:\b(?:in|inside|at)|@)\s+(?:the\s+)?$", re.I)
FIRST_NAME_TYPO_RATIO = 60.0  # how close a misspelled first name must be ("Stef" ~ "Steph" is 67)
SENTENCE_START = re.compile(r"(?:^|[.?!]\s*)$")


@dataclass
class Mention:
    """A name found in the prompt, with the kinds of entity it can be."""

    text: str
    start: int
    end: int
    kinds: set[str]
    role: str | None = None


@dataclass
class Scanner:
    """The prompt plus a mask of characters already used by an earlier rule."""

    text: str
    used: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.used = [False] * len(self.text)

    def free(self, start: int, end: int) -> bool:
        return not any(self.used[start:end])

    def take(self, start: int, end: int) -> None:
        for i in range(start, end):
            self.used[i] = True

    def matches(self, pattern: re.Pattern[str], take: bool = True) -> Iterator[re.Match[str]]:
        """Matches that do not overlap text used earlier; marks them used."""
        for m in pattern.finditer(self.text):
            if m.end() > m.start() and self.free(*m.span()):
                if take:
                    self.take(*m.span())
                yield m


def to_int(word: str) -> int:
    return int(word) if word.isdigit() else NUMBER_WORDS[word.lower()]


# --------------------------------------------------------------------------- pieces


def read_lines(sc: Scanner, projection: bool) -> list[DraftLine]:
    """"exceeds 27.5 points", "30+ points", "over 8.5 assists". Whole numbers count
    only with a cue word; "at least 30" and "30+" mean a 29.5 line."""
    out: list[DraftLine] = []
    for m in LINE_RE.finditer(sc.text):
        n, cue = float(m["n"]), (m["cue"] or "").lower()
        is_half = n != int(n)
        if not (is_half or m["plus"] or cue) or not (projection or is_half):
            continue
        if not sc.free(m.start("n"), m.end()):
            continue
        sc.take(m.start("n"), m.end())
        if (m["plus"] or cue == "at least") and not is_half:
            n -= 0.5
        out.append(DraftLine(stat=STAT_PHRASES[m["stat"].lower()], line=n))
    return out


def read_bare_lines(sc: Scanner, stats: list[str]) -> list[DraftLine]:
    """"line 28.5", "line of 13", "o/u 3.5", "a 28.5 line": a line with no stat next to it.

    It belongs to the first stat named anywhere in the question (points if none).
    Any number is accepted because the word "line" already says it is a betting line.
    """
    out: list[DraftLine] = []
    for m in BARE_LINE_RE.finditer(sc.text):
        group = "n" if m["n"] is not None else "n2"
        if not sc.free(m.start(group), m.end(group)):
            continue
        sc.take(m.start(), m.end())
        out.append(DraftLine(stat=stats[0] if stats else "points", line=float(m[group])))
        break  # one bare line per question; a second one would have no stat to attach to
    return out


def read_measures(sc: Scanner, d: QueryDraft) -> None:
    """Defender height and weight ("a 6'7 defender", "taller than 6-9", "250 lb")."""
    h = find_height(sc.text)
    if h and sc.free(*h[1]):
        s, e = h[1]
        d.defender_height = sc.text[s:e].strip()
        d.defender_height_comparison = comparison_before(sc.text, s)
        sc.take(s, e)
        if not re.search(r"defend|guard|matchup|match up|against", sc.text, re.I):
            d.notes.append("The height was read as the defender's height.")
    w = find_weight(sc.text)
    if w and sc.free(*w[1]):
        s, e = w[1]
        d.defender_weight = sc.text[s:e].strip()
        d.defender_weight_comparison = comparison_before(sc.text, s)
        sc.take(s, e)


def read_time(sc: Scanner, d: QueryDraft) -> None:
    """Relative and absolute seasons, and "last N games"."""
    for m in sc.matches(re.compile(rf"\b(?:last|past|previous)\s+{NUM}\s+(?:years?|seasons?)\b", re.I)):
        d.last_n_seasons = to_int(m.group(1))
    for m in sc.matches(re.compile(rf"\b(?:last|past|previous)\s+{NUM}\s+games?\b", re.I)):
        d.last_n_games = to_int(m.group(1))
    for m in sc.matches(re.compile(r"\bsince\s+(?:the\s+)?((?:19|20)\d{2})(?:\s*-\s*\d{2,4})?(?:\s+season)?\b", re.I)):
        d.season_start = int(m.group(1))
    for m in sc.matches(re.compile(r"\b(?:from|between)\s+((?:19|20)\d{2})\s*(?:to|and|through|until|-)\s*((?:19|20)\d{2})\b", re.I)):
        d.season_start, d.season_end = int(m.group(1)), int(m.group(2))
    for m in sc.matches(re.compile(r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)?\d{2})\b")):
        d.season_start = d.season_end = int(m.group(1))
    for m in sc.matches(re.compile(r"\b(?:in|during)\s+((?:19|20)\d{2})\b", re.I)):
        y = int(m.group(1))
        d.season_start = d.season_end = y - 1
        d.notes.append(f"'{y}' read as the {y - 1}-{y % 100:02d} season (the one ending in {y}).")
    if list(sc.matches(re.compile(r"\b(?:this|current)\s+(?:season|year)\b", re.I))):
        d.relative_season = "this"
    if list(sc.matches(re.compile(r"\blast\s+(?:season|year)\b", re.I))):
        d.relative_season = "last"


MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_WORD = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_MONTH_LIST = rf"((?:(?:\d{{1,2}}|{_MONTH_WORD})\s*(?:,|and|&|-|to|through)?\s*)+)"
MONTH_GROUPS_RE = re.compile(
    rf"\bmonths?\s+{_MONTH_LIST}\s*(?:to|vs\.?|versus|against|compared (?:to|with)|and)\s+(?:months?\s+)?{_MONTH_LIST}",
    re.I)
LAST_N_BEFORE_PLAYOFFS_RE = re.compile(
    rf"\b(?:last|final)\s+{NUM}\s+(?:regular[\s-]season\s+)?games?\s+(?:before|heading into|going into|leading up to)"
    r"\s+(?:the\s+)?(?:playoffs?|postseason)\b", re.I)
RUN_UP_RE = re.compile(r"\b(?:leading (?:up )?(?:in)?to|heading into|going into|run[\s-]up to)\s+(?:the\s+)?"
                       r"(?:playoffs?|postseason)\b|\b(?:down the stretch|stretch run)\b", re.I)
ALL_STAR_RE = re.compile(
    r"\b(?:(?:before\s+(?:vs\.?|versus|and)\s+after|after|before|since|post|pre)[\s-]+(?:the\s+)?)?"
    r"(?:all[\s-]?star\s+(?:break|game|weekend)|the\s+break)\b|\bpost[\s-]?(?:all[\s-]?star|break)\b|"
    r"\bsecond half of the season\b", re.I)
CUSTOM_DATE_RE = re.compile(rf"\b(?:before|after|since)\s+({_MONTH_WORD})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\d)", re.I)
LEADERBOARD_RE = re.compile(r"\b(?:who|which players?)\b.{0,40}?\b(?:improves?|gets? better|improved|rises?|"
                            r"declines?|drops? off|gets? worse|changes?)\b.{0,15}?\b(?:most|the most)\b"
                            r"|\bwhich players?\b|\bplayers? (?:who|that)\b|\b(?:rank|ranking of) (?:all )?players\b", re.I)
TEAMS_WORD_RE = re.compile(r"\b(?:teams|every team|all teams|team (?:ppg|points|scoring|pace|ratings?))\b", re.I)


def _month_numbers(text: str) -> list[int]:
    """"10,11,12,1" or "Oct-Jan" -> month numbers (ranges wrap across the new year)."""
    parts = re.findall(rf"\d{{1,2}}|{_MONTH_WORD}", text, re.I)
    nums = [int(p) if p.isdigit() else MONTH_ABBR[p[:3].lower()] for p in parts]
    if re.search(r"-|\bto\b|through", text) and len(nums) == 2:
        a, b = nums
        return [((a - 1 + i) % 12) + 1 for i in range((b - a) % 12 + 1)]
    return [n for n in dict.fromkeys(nums) if 1 <= n <= 12]


def read_period(sc: Scanner, d: QueryDraft) -> bool:
    """Before/after comparisons. Runs before the time and calendar rules so that
    "last 20 games before the playoffs" and "months 10,11,12,1" are not read as filters."""
    if (m := next(sc.matches(MONTH_GROUPS_RE), None)) is not None:
        d.period_kind = "month_groups"
        d.before_months, d.after_months = _month_numbers(m.group(1)), _month_numbers(m.group(2))
    elif (m := next(sc.matches(LAST_N_BEFORE_PLAYOFFS_RE), None)) is not None:
        d.period_kind, d.period_n_games = "last_n_before_playoffs", to_int(m.group(1))
    elif next(sc.matches(ALL_STAR_RE), None) is not None:
        d.period_kind = "all_star_break"
    elif next(sc.matches(RUN_UP_RE), None) is not None:
        d.period_kind, d.period_n_games = "last_n_before_playoffs", 20
    elif (m := next(sc.matches(CUSTOM_DATE_RE), None)) is not None:
        d.period_kind, d.period_date = "custom_date", f"{MONTH_ABBR[m.group(1)[:3].lower()]:02d}-{int(m.group(2)):02d}"
    if next(sc.matches(LEADERBOARD_RE), None) is not None:
        d.leaderboard = True
        if d.period_kind == "none":
            d.period_kind = "all_star_break"
    if d.period_kind != "none" and next(sc.matches(TEAMS_WORD_RE, take=False), None) is not None:
        d.subject_type = "team"
    return d.period_kind != "none"


def read_grouping(sc: Scanner, effect_question: bool) -> str | None:
    """The variable in "how does X affect ..." / "by X" / "home vs away"."""
    for pattern, var in VARIABLE_PHRASES:
        for m in re.finditer(rf"\b(?:{pattern})\b", sc.text, re.I):
            if not sc.free(*m.span()):
                continue
            by = BY_RE.search(sc.text[:m.start()])
            if var in ALWAYS_GROUP or by or (effect_question and var not in ("season", "month", "venue")):
                sc.take(*m.span())
                return var
    return None


def read_calendar(sc: Scanner, d: QueryDraft) -> None:
    """Days of week, months (only after "in"/"during"), weekends."""
    for m in sc.matches(re.compile(r"\b(" + "|".join(DAYS) + r")s?\b", re.I)):
        d.days_of_week.append(m.group(1).capitalize())  # type: ignore[arg-type]
    if list(sc.matches(re.compile(r"\bweekends?\b", re.I))):
        d.days_of_week += ["Saturday", "Sunday"]  # type: ignore[list-item]
    for m in sc.matches(re.compile(r"\b(?:in|during)\s+(" + "|".join(MONTHS) + r")\b", re.I)):
        d.months.append(MONTHS.index(m.group(1).lower()) + 1)


def read_situation(sc: Scanner, d: QueryDraft) -> None:
    """Home/away, back-to-backs, rest, playoffs, minutes."""
    if list(sc.matches(re.compile(r"\b(?:at home|home games?|in home games)\b", re.I))):
        d.home_away = "home"
    if list(sc.matches(re.compile(r"\b(?:on the road|away games?|road games?|away from home|on away)\b", re.I))):
        d.home_away = "away"
    if list(sc.matches(re.compile(r"\b(?:not|no|without)\s+(?:on\s+)?(?:a\s+)?back[\s-]to[\s-]backs?\b", re.I))):
        d.back_to_back = False
    if list(sc.matches(re.compile(r"\b(?:back[\s-]to[\s-]backs?|b2bs?|second night of a back[\s-]to[\s-]back)\b", re.I))):
        d.back_to_back = True
    for m in sc.matches(re.compile(rf"\b{NUM}\s+days?\s+(?:of\s+)?rest\b", re.I)):
        d.rest_days_min = d.rest_days_max = to_int(m.group(1))
    if list(sc.matches(re.compile(r"\bregular[\s-]season\b", re.I))):
        d.playoffs = False
    if list(sc.matches(re.compile(r"\b(?:playoffs?|postseason)\b", re.I))):
        d.playoffs = True
    for m in sc.matches(re.compile(r"\b(?:at least|more than|over)\s+(\d{1,2})\s*min(?:ute)?s\b|\b(\d{1,2})\+\s*min(?:ute)?s\b", re.I)):
        d.min_minutes = float(m.group(1) or m.group(2))


def read_positions(sc: Scanner, d: QueryDraft) -> None:
    for m in sc.matches(DEFENDER_POSITION_RE):
        d.defender_positions.append(POSITION_WORDS[m.group(1).lower()])  # type: ignore[arg-type]
    for m in sc.matches(SUBJECT_POSITION_RE):
        d.subject_positions.append(POSITION_WORDS[m.group(1).lower()])  # type: ignore[arg-type]


def read_stats(sc: Scanner) -> list[str]:
    return [STAT_PHRASES[m.group(0).lower()] for m in sc.matches(STAT_RE)]


# ---------------------------------------------------------------------------- names

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.\-]*[A-Za-z0-9]|[A-Za-z0-9]")


def blocked(token: str, sentence_start: bool) -> bool:
    """Common words cannot be names unless capitalized mid-sentence ("Green", not "green")."""
    word = clean_name(token)
    if token[:1].islower() or sentence_start:
        return word in BLOCKED_WORDS
    return False


def alias_kinds(index: EntityIndex, norm: str, raw: str) -> set[str]:
    kinds: set[str] = set()
    short_ok = len(norm) > 2 or raw.isupper()  # "LA", "AD" only in capitals
    if index.is_team_alias(norm) and short_ok:
        kinds.add("team")
    if index.is_player_alias(norm) and short_ok:
        kinds.add("player")
    if index.is_venue_alias(norm) and short_ok:
        kinds.add("venue")
    if "team" in kinds and norm in {normalize_alias(str(c)) for c in index.teams["city"]}:
        kinds.add("venue")
    return kinds


def find_mentions(sc: Scanner, index: EntityIndex) -> list[Mention]:
    """Longest alias n-grams first, then fuzzy runs of leftover capitalized words."""
    toks = [m for m in TOKEN_RE.finditer(sc.text) if sc.free(*m.span())]
    mentions: list[Mention] = []
    i = 0
    while i < len(toks):
        found = None
        for n in range(min(4, len(toks) - i), 0, -1):
            span = toks[i:i + n]
            if any(not sc.free(t.start(), t.end()) for t in span):
                continue
            if any(span[k + 1].start() - span[k].end() > 2 for k in range(n - 1)):
                continue
            s, e = span[0].start(), span[-1].end()
            raw = sc.text[s:e]
            if n == 1 and blocked(raw, bool(SENTENCE_START.search(sc.text[:s]))):
                continue
            if clean_name(raw) in BLOCKED_WORDS:
                continue
            kinds = alias_kinds(index, clean_name(raw), re.sub(r"(?:'|’)s$", "", raw))
            if kinds:
                found = Mention(raw, s, e, kinds)
                i += n
                break
        if found:
            sc.take(found.start, found.end)
            mentions.append(extend_left(sc, index, found))
        else:
            i += 1
    mentions += fuzzy_mentions(sc, index)
    return sorted(mentions, key=lambda m: m.start)


def extend_left(sc: Scanner, index: EntityIndex, m: Mention) -> Mention:
    """"Stef Curry" / "Lebrom James": the surname matched exactly, but a misspelled
    first name sits just before it. Take both when together they fuzzy-match a player."""
    if "player" not in m.kinds:
        return m
    prev = re.search(r"([A-Z][A-Za-z'’.\-]+)\s+$", sc.text[:m.start])
    if not prev or not sc.free(*prev.span(1)) or blocked(prev.group(1), bool(SENTENCE_START.search(sc.text[:prev.start(1)]))):
        return m
    first = clean_name(prev.group(1))
    if alias_kinds(index, first, prev.group(1)):
        return m
    raw = sc.text[prev.start(1):m.end]
    res = index.resolve_player(raw)
    if res.id is None or res.id not in index.player_aliases.get(clean_name(m.text), []):
        return m
    aliases = index.players.at[res.id, "aliases"]
    firsts = {a.split()[0] for a in aliases if " " in a}
    if max((fuzz.ratio(first, f) for f in firsts), default=0) < FIRST_NAME_TYPO_RATIO:
        return m
    sc.take(prev.start(1), m.start)
    return Mention(raw, prev.start(1), m.end, {"player"})


def fuzzy_mentions(sc: Scanner, index: EntityIndex) -> list[Mention]:
    """Runs of capitalized, unused, non-blocked words that fuzzily match a name ("Lebrom")."""
    out: list[Mention] = []
    run: list[re.Match[str]] = []
    toks = list(TOKEN_RE.finditer(sc.text)) + [None]
    for t in toks:
        ok = (t is not None and sc.free(*t.span()) and t.group(0)[:1].isupper()
              and not blocked(t.group(0), bool(SENTENCE_START.search(sc.text[:t.start()]))))
        if ok and (not run or t.start() - run[-1].end() <= 2):  # type: ignore[union-attr]
            run.append(t)  # type: ignore[arg-type]
            continue
        if run:
            s, e = run[0].start(), run[-1].end()
            raw = sc.text[s:e]
            if len(clean_name(raw)) >= 4:
                kinds = {k for k, res in (("team", index.resolve_team(raw)), ("player", index.resolve_player(raw)))
                         if res.status != "not_found"}
                if kinds:
                    out.append(Mention(raw, s, e, kinds))
                    sc.take(s, e)
        run = [t] if ok else []  # type: ignore[list-item]
    return out


def assign_roles(text: str, mentions: list[Mention]) -> None:
    """Give each mention a role from the words just before and after it."""
    for m in mentions:
        before, after = text[max(0, m.start - 40):m.start], text[m.end:m.end + 40]
        is_player = "player" in m.kinds
        if is_player and (DEFENDER_AFTER.search(after) or DEFENDER_BEFORE.search(before)):
            m.role = "defender"
        elif is_player and (TEAMMATE_OUT_BEFORE.search(before) or TEAMMATE_OUT_AFTER.search(after)):
            m.role = "teammate_out"
        elif is_player and TEAMMATE_IN_BEFORE.search(before) and m is not mentions[0]:
            m.role = "teammate_in"
        elif "venue" in m.kinds and VENUE_BEFORE.search(before):
            m.role = "venue"
        elif "team" in m.kinds and AWAY_AT_BEFORE.search(before):
            m.role = "at_opponent"
        elif OPPONENT_BEFORE.search(before):
            m.role = "opponent"


def drop_team_descriptors(text: str, mentions: list[Mention], d: QueryDraft) -> list[Mention]:
    """"Warriors Curry" / "the Celtics' Jayson Tatum": a team right before a player
    names the player's team. It is not a second subject or an opponent."""
    keep: list[Mention] = []
    for a, b in zip(mentions, [*mentions[1:], None]):
        between = text[a.end:b.start] if b else ""
        if (b is not None and a.role is None and a.kinds & {"team"} and "player" not in a.kinds
                and "player" in b.kinds and re.fullmatch(r"(?:'|’)?s?'?\s*", between)):
            d.notes.append(f"{a.text!r} read as {b.text}'s team.")
            continue
        keep.append(a)
    return keep


def fill_names(text: str, mentions: list[Mention], d: QueryDraft) -> None:
    """Put mentions into the draft. The first mention without a role is the subject."""
    assign_roles(text, mentions)
    for m in mentions:
        m.text = re.sub(r"(?:'|’)s$", "", m.text)
    mentions[:] = drop_team_descriptors(text, mentions, d)
    subject: Mention | None = None
    for m in mentions:
        if m.role is None and subject is None and (m.kinds & {"player", "team"}):
            subject, m.role = m, "subject"
            d.subject_type = "team" if "team" in m.kinds else "player"
            d.subjects.append(m.text)
        elif m.role is None and subject is not None:
            joined = re.fullmatch(r"\s*(?:,|and|&|\+)\s*", text[subject.end:m.start]) is not None
            same_kind = ("team" in m.kinds) == (d.subject_type == "team")
            if joined and same_kind:
                d.subjects.append(m.text)
                subject = m
                continue
            m.role = "opponent"
            d.notes.append(f"{m.text!r} was read as the opponent.")
        if m.role == "opponent" or m.role == "at_opponent":
            if "team" in m.kinds:
                d.opponent_teams.append(m.text)
                if m.role == "at_opponent" and d.venue is None:
                    d.venue = m.text
            elif "player" in m.kinds:
                d.opponent_players.append(m.text)
        elif m.role == "defender":
            d.defenders.append(m.text)
        elif m.role == "teammate_in":
            d.teammates_in.append(m.text)
        elif m.role == "teammate_out":
            d.teammates_out.append(m.text)
        elif m.role == "venue":
            d.venue = m.text


# ---------------------------------------------------------------------------- main


def parse_rules(prompt: str, index: EntityIndex) -> QueryDraft:
    """Read a prompt into a :class:`QueryDraft` using rules only."""
    sc = Scanner(prompt)
    d = QueryDraft()
    projection = bool(PROJECTION_RE.search(prompt))
    effect_question = bool(EFFECT_RE.search(prompt))

    d.lines = read_lines(sc, projection)
    period = read_period(sc, d)
    read_measures(sc, d)
    read_time(sc, d)
    variable = read_grouping(sc, effect_question)
    read_positions(sc, d)
    read_calendar(sc, d)
    read_situation(sc, d)
    stats = read_stats(sc)
    bare = [ln for ln in read_bare_lines(sc, [*stats, *(ln.stat for ln in d.lines)])
            if ln.stat not in {x.stat for x in d.lines}]
    if bare:
        projection = True
        d.lines += bare
    d.stats = list(dict.fromkeys([*stats, *(ln.stat for ln in d.lines)]))
    fill_names(prompt, find_mentions(sc, index), d)

    if period:
        d.mode = "split"  # the builder turns a period_kind into mode="period"
    elif variable and (effect_question or variable not in ALWAYS_GROUP) and not d.subjects:
        d.mode, d.effect_variable = "variable_effect", variable
    elif variable:
        d.mode, d.group_by = "split", variable
    elif projection:
        d.mode = "projection"
    if d.subject_positions and not d.subjects and d.mode != "variable_effect":
        d.notes.append("No player named; the question covers every player at that position.")
    return d



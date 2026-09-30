"""Plain-English question -> (StatQuery, assumptions, unresolved items).

Two parsers read the question into a :class:`~nbalab.nlp.types.QueryDraft`:

- **Claude** (:mod:`nbalab.nlp.llm`), when an Anthropic API key is available.
- **Rules** (:mod:`nbalab.nlp.rules`), regex plus alias matching, with no key.
  Used automatically when the key is missing, rejected, out of credit, or the
  call fails for any reason, so the public demo always works.

Either way, :class:`~nbalab.nlp.build.QueryBuilder` then resolves names to ids,
turns relative dates into seasons, fills defaults, and checks for conflicts.
The app shows the assumptions and unresolved items as editable chips before
running anything.

Usage::

    query, assumptions, unresolved = parse_prompt("Curry's apg vs the Spurs")

Identical prompts (same text, same day, same backend) are answered from an
in-process cache, so re-runs of a Streamlit script do not call the API again.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Literal

from nbalab.data.config import DATA_DIR
from nbalab.deploy.secrets import anthropic_api_key, get_secret
from nbalab.nlp.build import BuildSettings, QueryBuilder
from nbalab.nlp.defaults import LineProvider, RecentAverageLines
from nbalab.nlp.entities import EntityIndex
from nbalab.nlp.llm import LLMSettings, LLMUnavailable, MessagesClient, make_client, parse_with_claude
from nbalab.nlp.rules import parse_rules
from nbalab.nlp.types import Assumption, Candidate, QueryDraft, UnresolvedItem
from nbalab.query.schema import StatQuery

Backend = Literal["auto", "llm", "rules"]
Source = Literal["llm", "rules"]
CACHE_SIZE = 256

__all__ = [
    "Assumption", "Candidate", "ParseResult", "ParserConfig", "PromptParser", "UnresolvedItem",
    "parse_prompt", "swap_entity",
]


@dataclass(frozen=True)
class ParserConfig:
    """Everything tunable about parsing. Environment overrides (read by
    :meth:`from_env`): ``NBALAB_PARSER_MODEL``, ``NBALAB_PARSER_EFFORT``,
    ``NBALAB_PARSER_DATA_DIR`` (folder with players/teams/player_games parquet; the
    hosted app sets it to wherever it downloaded the data),
    ``NBALAB_HEIGHT_TOLERANCE`` (inches either side of a single height). Each is
    looked up in Streamlit secrets, then the environment, then ``.env``."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    build: BuildSettings = field(default_factory=BuildSettings)
    processed_dir: Path = DATA_DIR / "processed"

    @classmethod
    def from_env(cls) -> "ParserConfig":
        llm = LLMSettings()
        llm = replace(llm, model=get_secret("NBALAB_PARSER_MODEL", llm.model) or llm.model,
                      effort=get_secret("NBALAB_PARSER_EFFORT", llm.effort) or llm.effort)
        data_dir = get_secret("NBALAB_PARSER_DATA_DIR")  # hosted app: where the data was downloaded
        build = BuildSettings()
        if tol := get_secret("NBALAB_HEIGHT_TOLERANCE"):
            build = replace(build, height_tolerance_in=float(tol))
        if data_dir:
            return cls(llm=llm, build=build, processed_dir=Path(data_dir))
        return cls(llm=llm, build=build)


@dataclass(frozen=True)
class ParseResult:
    """The parsed query and the notes the app shows as chips.

    Unpacks to exactly three values: ``query, assumptions, unresolved = result``.
    ``query`` is ``None`` only when no valid query can be built (the reason is
    in ``unresolved``). ``source`` says which parser read the prompt.
    """

    query: StatQuery | None
    assumptions: list[Assumption]
    unresolved: list[UnresolvedItem]
    source: Source
    draft: QueryDraft

    def __iter__(self) -> Iterator[Any]:
        return iter((self.query, self.assumptions, self.unresolved))

    def copy(self) -> "ParseResult":
        return ParseResult(
            self.query.model_copy(deep=True) if self.query else None,
            [a.model_copy(deep=True) for a in self.assumptions],
            [u.model_copy(deep=True) for u in self.unresolved],
            self.source,
            self.draft.model_copy(deep=True),
        )


def load_api_key() -> str | None:
    """``ANTHROPIC_API_KEY`` from Streamlit secrets, the environment, or ``.env``.

    Placeholder values ("sk-ant-...") count as missing, so an unfilled template
    means the rule-based parser, not a failing API call.
    """
    return anthropic_api_key()


def cache_key(prompt: str) -> str:
    """Whitespace-insensitive but case-sensitive ("Green" the player vs "green")."""
    return re.sub(r"\s+", " ", prompt).strip()


class PromptParser:
    """A reusable parser holding the name index, the Claude client, and a cache.

    ``client`` and ``index`` can be injected (tests pass fakes). With neither,
    it loads ``players.parquet`` / ``teams.parquet`` and builds a client from
    the API key if one is set.
    """

    def __init__(
        self,
        config: ParserConfig | None = None,
        *,
        index: EntityIndex | None = None,
        client: MessagesClient | None = None,
        api_key: str | None = None,
        line_provider: LineProvider | None = None,
    ) -> None:
        self.config = config or ParserConfig.from_env()
        self._index = index
        self._client = client
        self._api_key = api_key
        self.line_provider = line_provider if line_provider is not None else RecentAverageLines(self.config.processed_dir)
        self._cache: OrderedDict[tuple[str, date, str], ParseResult] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def index(self) -> EntityIndex:
        if self._index is None:
            self._index = EntityIndex.from_processed(self.config.processed_dir)
        return self._index

    def client(self) -> MessagesClient | None:
        """The Claude client, or ``None`` when there is no API key."""
        if self._client is None:
            key = self._api_key or load_api_key()
            if key:
                self._client = make_client(key, self.config.llm)
        return self._client

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------ parse

    def parse(self, prompt: str, *, backend: Backend = "auto", today: date | None = None) -> ParseResult:
        """Parse one prompt. See the module docstring for the backends."""
        today = today or date.today()
        key = (cache_key(prompt), today, backend)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key].copy()
        result, cacheable = self._parse_uncached(prompt, backend, today)
        if cacheable:
            with self._lock:
                self._cache[key] = result
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
        return result.copy()

    def _parse_uncached(self, prompt: str, backend: Backend, today: date) -> tuple[ParseResult, bool]:
        """Returns the result and whether it may be cached (a transient API
        failure that fell back to rules is not cached, so the next try uses Claude)."""
        notes: list[Assumption] = []
        source: Source = "rules"
        draft: QueryDraft | None = None
        cacheable = True
        client = self.client() if backend != "rules" else None
        if backend == "llm" and client is None:
            raise LLMUnavailable("backend='llm' but no ANTHROPIC_API_KEY is set")
        if client is not None:
            try:
                draft, source = parse_with_claude(prompt, today, client, self.config.llm), "llm"
            except LLMUnavailable as e:
                if backend == "llm":
                    raise
                cacheable = False
                notes.append(Assumption(field="parser", message=f"Claude parser unavailable ({e}); "
                                        "read by the rule-based parser. Check the chips.", value="rules"))
        elif backend == "auto":
            notes.append(Assumption(field="parser", message="No Anthropic API key; read by the rule-based "
                                    "parser. Check the chips.", value="rules"))
        if draft is None:
            draft = parse_rules(prompt, self.index)
        builder = QueryBuilder(self.index, today, self.config.build, self.line_provider)
        query = builder.build(draft)
        return ParseResult(query, notes + builder.assumptions, builder.unresolved, source, draft), cacheable


_default_parser: PromptParser | None = None
_default_lock = threading.Lock()


def default_parser() -> PromptParser:
    global _default_parser
    with _default_lock:
        if _default_parser is None:
            _default_parser = PromptParser()
        return _default_parser


def parse_prompt(prompt: str, *, backend: Backend = "auto", today: date | None = None,
                 parser: PromptParser | None = None) -> ParseResult:
    """Parse a question with the shared default parser (or ``parser``).

    ``backend``: ``"auto"`` uses Claude when a key is set and falls back to
    rules on any failure; ``"rules"`` forces the no-key parser; ``"llm"``
    raises :class:`~nbalab.nlp.llm.LLMUnavailable` instead of falling back.
    """
    return (parser or default_parser()).parse(prompt, backend=backend, today=today)


# ------------------------------------------------------------------ chip edits

ID_KEYS = {"person_id", "opponent_team_id", "venue_team_id", "defender_person_id", "team_id"}
ID_LIST_KEYS = {"subject_ids", "team_ids"}


def _swap(obj: Any, old: int, new: int) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ID_KEYS and v == old:
                out[k] = new
            elif k in ID_LIST_KEYS and isinstance(v, list):
                out[k] = [new if x == old else x for x in v]
            else:
                out[k] = _swap(v, old, new)
        return out
    if isinstance(obj, list):
        return [_swap(x, old, new) for x in obj]
    return obj


def swap_entity(query: StatQuery, old_id: int, new_id: int) -> StatQuery:
    """Replace a player or team id everywhere in the query (the user picked another candidate)."""
    return StatQuery.model_validate(_swap(query.model_dump(), old_id, new_id))

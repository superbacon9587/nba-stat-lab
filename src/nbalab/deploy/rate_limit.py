"""Cap how many questions go to the LLM parser, so a shared link cannot run up the API bill.

Two sliding-window limits apply, and a question uses the LLM only when both have room:

- **per session** (default 20 per hour): one browser tab. Refreshing the page
  starts a new Streamlit session, so this limit alone is easy to get around.
- **per app** (default 200 per hour): shared by every visitor to this server
  process. This is the limit that actually bounds spending.

When either limit is hit, or no API key is configured, the question goes to
the rule-based parser instead. The app keeps working, with a weaker parser.

A "sliding window" counts calls in the last ``window_seconds``, measured from
now. Unlike a counter that resets on the hour, it never allows a burst of
2x the limit across an hour boundary.

As a hard backstop, also set a monthly spend limit in the Anthropic Console
(see ``docs/deploy.md``). Code limits reset when the app restarts; the
Console limit does not.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from nbalab.deploy.secrets import anthropic_api_key, get_int_setting

DEFAULT_PER_SESSION_PER_HOUR = 20
DEFAULT_PER_APP_PER_HOUR = 200
HOUR = 3600.0


@dataclass
class SlidingWindowLimiter:
    """Allow at most ``max_calls`` in any ``window_seconds`` span. Thread-safe."""

    max_calls: int
    window_seconds: float = HOUR
    clock: Callable[[], float] = time.monotonic
    _calls: deque[float] = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _expire(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self.window_seconds:
            self._calls.popleft()

    def remaining(self) -> int:
        """Calls still allowed in the current window."""
        with self._lock:
            self._expire(self.clock())
            return max(0, self.max_calls - len(self._calls))

    def retry_after(self) -> float:
        """Seconds until one more call is allowed (0 if allowed now)."""
        with self._lock:
            now = self.clock()
            self._expire(now)
            if len(self._calls) < self.max_calls:
                return 0.0
            return self._calls[0] + self.window_seconds - now

    def try_acquire(self) -> bool:
        """Record a call and return ``True`` if under the limit, else ``False`` (nothing recorded)."""
        with self._lock:
            now = self.clock()
            self._expire(now)
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


ParserKind = Literal["llm", "rules"]


@dataclass(frozen=True)
class ParserRoute:
    """Which parser to use for one question, and a user-facing reason when it is not the LLM."""

    parser: ParserKind
    reason: str = ""


def _minutes(seconds: float) -> int:
    return max(1, round(seconds / 60))


def choose_parser(
    has_api_key: bool,
    session: SlidingWindowLimiter,
    app: SlidingWindowLimiter,
) -> ParserRoute:
    """Decide LLM vs rule-based parsing for one question, consuming quota only for LLM calls.

    Call this only on a cache miss: a repeated question served from the parser
    cache costs nothing and should not use up quota.
    """
    if not has_api_key:
        return ParserRoute("rules", "No Anthropic API key is configured; using the rule-based parser.")
    if session.remaining() == 0:
        return ParserRoute(
            "rules",
            f"You've used your {session.max_calls} AI-parsed questions for this hour "
            f"(next one in ~{_minutes(session.retry_after())} min). Using the rule-based parser.",
        )
    if not app.try_acquire():
        return ParserRoute(
            "rules",
            "The demo's hourly AI budget is used up "
            f"(resets in ~{_minutes(app.retry_after())} min). Using the rule-based parser.",
        )
    session.try_acquire()
    return ParserRoute("llm")


# ------------------------------------------------------------------ Streamlit glue


def session_limiter() -> SlidingWindowLimiter:
    """This browser session's limiter, kept in ``st.session_state``."""
    import streamlit as st

    key = "_nbalab_llm_session_limiter"
    if key not in st.session_state:
        st.session_state[key] = SlidingWindowLimiter(
            get_int_setting("NBALAB_LLM_PER_SESSION_PER_HOUR", DEFAULT_PER_SESSION_PER_HOUR)
        )
    return st.session_state[key]


def app_limiter() -> SlidingWindowLimiter:
    """The process-wide limiter shared by all sessions (one object via ``st.cache_resource``)."""
    import streamlit as st

    @st.cache_resource
    def _make(max_calls: int) -> SlidingWindowLimiter:
        return SlidingWindowLimiter(max_calls)

    return _make(get_int_setting("NBALAB_LLM_PER_APP_PER_HOUR", DEFAULT_PER_APP_PER_HOUR))


def route_question() -> ParserRoute:
    """Pick the parser for a question in the running Streamlit app."""
    return choose_parser(anthropic_api_key() is not None, session_limiter(), app_limiter())

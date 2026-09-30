"""Read secrets and settings the same way locally and on every host.

Lookup order, first non-empty value wins:

1. ``st.secrets`` (Streamlit Community Cloud "Secrets" box, or a local
   ``.streamlit/secrets.toml``),
2. environment variables (Hugging Face Spaces, Docker, CI),
3. the project's ``.env`` file (local development).

Placeholder values copied from the example files (``"sk-ant-..."``, ``""``)
count as missing, so an unfilled template never looks like a real key.
"""

from __future__ import annotations

import os
from functools import lru_cache

from nbalab.data.config import PROJECT_ROOT

# Settings the app reads; export_secrets_to_env() copies these from st.secrets.
KNOWN_SETTINGS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY", "NBALAB_DATA_REPO", "NBALAB_DATA_REVISION", "HF_TOKEN",
    "NBALAB_LLM_PER_SESSION_PER_HOUR", "NBALAB_LLM_PER_APP_PER_HOUR", "NBALAB_SKIP_ON_COURT",
)
_PLACEHOLDERS: frozenset[str] = frozenset({"", "sk-ant-...", "changeme", "your-key-here", "hf_..."})


def _from_streamlit(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:  # no secrets.toml, or not running under Streamlit
        return None
    return None if value is None else str(value)


@lru_cache(maxsize=1)
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return None if value in _PLACEHOLDERS else value


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return the secret or setting ``name``, or ``default`` if no source has a real value."""
    value = _clean(_from_streamlit(name))
    if value is None:
        _load_dotenv()
        value = _clean(os.environ.get(name))
    return value if value is not None else default


def get_int_setting(name: str, default: int) -> int:
    """Integer setting (e.g. a rate limit). Unparseable values fall back to ``default``."""
    raw = get_secret(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def anthropic_api_key() -> str | None:
    """The Anthropic key, or ``None``. ``None`` means the app uses the rule-based parser."""
    return get_secret("ANTHROPIC_API_KEY")


def export_secrets_to_env(names: tuple[str, ...] = KNOWN_SETTINGS) -> list[str]:
    """Copy real values from ``st.secrets`` into ``os.environ`` (never overwriting).

    Call this at the very top of the Streamlit entry point, before importing
    code that reads ``os.environ`` directly (the Anthropic parser). Returns the
    names that were exported.
    """
    exported: list[str] = []
    for name in names:
        value = _clean(_from_streamlit(name))
        if value is not None and not os.environ.get(name):
            os.environ[name] = value
            exported.append(name)
    return exported

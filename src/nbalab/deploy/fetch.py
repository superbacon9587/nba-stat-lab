"""Find the app's data, downloading it from Hugging Face on first start if needed.

Resolution order in :func:`ensure_data`:

1. ``data/deploy/`` already holds every required table: use it (a warm
   restart, or a local ``python -m nbalab.deploy.slim`` run).
2. ``NBALAB_DATA_REPO`` is set (e.g. ``yourname/nbalab-data``): download the
   dataset repo into ``data/deploy/`` and use it. ``HF_TOKEN`` is needed only
   if the dataset is private.
3. ``data/processed/`` holds every required table: use it (local development).
4. Otherwise raise :class:`DataUnavailable` with instructions.

Set ``NBALAB_SKIP_ON_COURT=1`` to skip the largest optional table if the host
runs out of memory. The app then falls back to "both played in the game" for
on-court filters, as :mod:`nbalab.query.data` documents.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from nbalab.data.config import DATA_DIR
from nbalab.deploy.manifest import OPTIONAL_TABLES, TABLES
from nbalab.deploy.secrets import get_secret
from nbalab.deploy.slim import DEPLOY_DIR, MANIFEST_FILE

PROCESSED_DIR = DATA_DIR / "processed"
Downloader = Callable[..., str]


class DataUnavailable(RuntimeError):
    """No usable copy of the data was found or could be downloaded."""


def wanted_tables(skip_on_court: bool = False) -> list[str]:
    """Tables to fetch. ``on_court`` is optional and can be skipped to save memory."""
    return [t for t in TABLES if not (skip_on_court and t == "on_court")]


def required_tables() -> list[str]:
    return [t for t in TABLES if t not in OPTIONAL_TABLES]


def has_tables(directory: Path, tables: list[str]) -> bool:
    return all((directory / f"{t}.parquet").exists() for t in tables)


def _default_downloader() -> Downloader:
    from huggingface_hub import snapshot_download

    return snapshot_download


def download(
    repo_id: str,
    dest: Path,
    tables: list[str],
    revision: str | None = None,
    token: str | None = None,
    downloader: Downloader | None = None,
) -> Path:
    """Download the listed tables plus ``manifest.json`` from a HF dataset repo into ``dest``."""
    fetch = downloader or _default_downloader()
    dest.mkdir(parents=True, exist_ok=True)
    fetch(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=dest,
        allow_patterns=[f"{t}.parquet" for t in tables] + [MANIFEST_FILE],
        token=token,
    )
    if not has_tables(dest, required_tables()):
        raise DataUnavailable(
            f"Downloaded {repo_id} but required tables are missing: "
            f"{[t for t in required_tables() if not (dest / f'{t}.parquet').exists()]}"
        )
    return dest


def ensure_data(
    deploy_dir: Path = DEPLOY_DIR,
    processed_dir: Path = PROCESSED_DIR,
    downloader: Downloader | None = None,
) -> Path:
    """Return a directory holding the app's parquet files. See the module docstring for the order."""
    tables = wanted_tables(get_secret("NBALAB_SKIP_ON_COURT") in {"1", "true", "yes"})
    if has_tables(deploy_dir, tables):
        return deploy_dir
    repo_id = get_secret("NBALAB_DATA_REPO")
    if repo_id:
        return download(
            repo_id, deploy_dir, tables,
            revision=get_secret("NBALAB_DATA_REVISION"),
            token=get_secret("HF_TOKEN"),
            downloader=downloader,
        )
    if has_tables(processed_dir, required_tables()):
        return processed_dir
    raise DataUnavailable(
        "No data found. Locally, run `python -m nbalab.data.build`. On a host, set "
        "NBALAB_DATA_REPO (and HF_TOKEN for a private dataset); see docs/deploy.md."
    )


def data_manifest(directory: Path) -> dict:
    """The ``manifest.json`` written by :mod:`nbalab.deploy.slim`, or ``{}`` if absent."""
    path = directory / MANIFEST_FILE
    return json.loads(path.read_text()) if path.exists() else {}

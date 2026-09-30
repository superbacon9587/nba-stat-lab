"""Shared fixtures: the pipeline run on the committed sample, and the real processed files."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from nbalab.data.build import build_tables
from nbalab.data.config import DATA_DIR, BuildConfig
from nbalab.data.load import load_raw

SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"


@pytest.fixture(scope="session")
def sample_config() -> BuildConfig:
    return replace(BuildConfig(), raw_dir=SAMPLE_DIR, start_season=2007, extra_schedule_files=())


@pytest.fixture(scope="session")
def sample_tables(sample_config: BuildConfig) -> dict[str, pd.DataFrame]:
    """The four processed tables built from ``data/sample/`` (Seattle 2007-08 -> OKC 2008-09)."""
    raw = load_raw(sample_config.raw_dir)
    return build_tables(raw, sample_config)


def load_processed(name: str) -> pd.DataFrame:
    """Read a real processed table, skipping the test if the pipeline has not been run."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built; run `python -m nbalab.data.build`")
    return pd.read_parquet(path)

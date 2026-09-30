"""Tests for the hosting layer: data slimming, fetching, secrets, and LLM rate limits."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nbalab.deploy import fetch, secrets
from nbalab.deploy.manifest import TABLES, TableSpec
from nbalab.deploy.rate_limit import SlidingWindowLimiter, choose_parser
from nbalab.deploy.slim import slim_table, write_slim
from nbalab.query.data import MATCHUP_COLUMNS, ON_COURT_COLUMNS, ON_COURT_OPTIONAL

SRC = Path(__file__).resolve().parents[1] / "src" / "nbalab"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------------ slimming


def _table() -> pa.Table:
    return pa.table({
        "gameId": pa.array([3, 1, 2], pa.int64()),
        "season": pa.array([1995, 1996, 2020], pa.int16()),
        "minutes": pa.array([10.5, 20.25, 30.0], pa.float64()),
        "unused": ["a", "b", "c"],
    })


def test_slim_table_keeps_columns_filters_seasons_downcasts_and_sorts() -> None:
    spec = TableSpec(columns=("gameId", "season", "minutes"),
                     downcast={"minutes": pa.float32()}, sort_by=("gameId",))
    out = slim_table(_table(), spec, start_season=1996)
    assert out.column_names == ["gameId", "season", "minutes"]
    assert out["gameId"].to_pylist() == [1, 2]  # 1995 row dropped, sorted
    assert out.schema.field("minutes").type == pa.float32()


def test_slim_table_drops_rows_flagged_false_or_null() -> None:
    t = _table().append_column("reliable", pa.array([True, False, None]))
    out = slim_table(t, TableSpec(columns=("gameId", "reliable"), keep_if_true="reliable"), 0)
    assert out["gameId"].to_pylist() == [3]


def test_slim_table_missing_manifest_column_raises() -> None:
    with pytest.raises(KeyError, match="nope"):
        slim_table(_table(), TableSpec(columns=("gameId", "nope")), start_season=1996)


def test_write_slim_uses_zstd_and_writes_manifest(tmp_path: Path) -> None:
    src, out = tmp_path / "processed", tmp_path / "deploy"
    src.mkdir()
    teams = pa.table({c: pa.array([1], pa.int64()) if c == "teamId" else pa.array(["x"])
                      for c in TABLES["teams"].columns})
    pq.write_table(teams, src / "teams.parquet")
    stats = write_slim(src, out, start_season=1996)
    assert set(stats) == {"teams"}
    assert pq.ParquetFile(out / "teams.parquet").metadata.row_group(0).column(0).compression == "ZSTD"
    assert fetch.data_manifest(out)["tables"]["teams"]["rows"] == 1


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data not built")
@pytest.mark.parametrize("table", ["player_games", "team_games", "players", "teams"])
def test_manifest_covers_columns_named_in_query_code(table: str) -> None:
    """Any processed column the query code quotes by name must ship with the app."""
    path = PROCESSED / f"{table}.parquet"
    if not path.exists():
        pytest.skip(f"{table} not built")
    code = "".join(p.read_text() for p in (SRC / "query").glob("*.py"))
    quoted = set(re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']", code))
    used = quoted & set(pq.ParquetFile(path).schema_arrow.names)
    assert used - set(TABLES[table].columns) == set()


@pytest.mark.parametrize("table, required, optional", [
    ("on_court", ON_COURT_COLUMNS, ON_COURT_OPTIONAL),
    ("matchups", MATCHUP_COLUMNS, ()),
])
def test_manifest_satisfies_loader_column_specs(table: str, required: dict, optional: tuple) -> None:
    """The loader matches these tables by candidate names; the shipped columns must match too."""
    shipped = set(TABLES[table].columns)
    for canonical, candidates in required.items():
        assert shipped & set(candidates), f"{table}: nothing shipped for '{canonical}'"
    path = PROCESSED / f"{table}.parquet"
    if path.exists():
        present = set(pq.ParquetFile(path).schema_arrow.names)
        assert (set(optional) & present) - shipped == set()


# ------------------------------------------------------------------------- fetching


def _touch_tables(directory: Path, tables: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for t in tables:
        (directory / f"{t}.parquet").write_bytes(b"")


@pytest.fixture
def no_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NBALAB_DATA_REPO", "NBALAB_SKIP_ON_COURT", "HF_TOKEN", "NBALAB_DATA_REVISION"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(secrets, "_from_streamlit", lambda name: None)


def test_ensure_data_prefers_existing_deploy_dir(tmp_path: Path, no_settings: None) -> None:
    _touch_tables(tmp_path / "deploy", list(TABLES))
    assert fetch.ensure_data(tmp_path / "deploy", tmp_path / "processed") == tmp_path / "deploy"


def test_ensure_data_downloads_from_repo(tmp_path: Path, no_settings: None,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBALAB_DATA_REPO", "me/nbalab-data")
    monkeypatch.setenv("NBALAB_SKIP_ON_COURT", "1")
    calls: list[dict] = []

    def fake_download(repo_id: str, **kw) -> str:
        calls.append({"repo_id": repo_id, **kw})
        _touch_tables(Path(kw["local_dir"]), fetch.required_tables())
        return str(kw["local_dir"])

    out = fetch.ensure_data(tmp_path / "deploy", tmp_path / "processed", downloader=fake_download)
    assert out == tmp_path / "deploy"
    assert calls[0]["repo_id"] == "me/nbalab-data" and calls[0]["repo_type"] == "dataset"
    assert "on_court.parquet" not in calls[0]["allow_patterns"]


def test_download_missing_required_table_raises(tmp_path: Path) -> None:
    with pytest.raises(fetch.DataUnavailable, match="required tables are missing"):
        fetch.download("me/x", tmp_path, ["teams"], downloader=lambda *a, **k: "")


def test_ensure_data_falls_back_to_processed_then_raises(tmp_path: Path, no_settings: None) -> None:
    with pytest.raises(fetch.DataUnavailable):
        fetch.ensure_data(tmp_path / "deploy", tmp_path / "processed")
    _touch_tables(tmp_path / "processed", fetch.required_tables())
    assert fetch.ensure_data(tmp_path / "deploy", tmp_path / "processed") == tmp_path / "processed"


# -------------------------------------------------------------------------- secrets


def test_secret_lookup_order_and_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "_from_streamlit", lambda name: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    assert secrets.anthropic_api_key() is None  # template placeholder is not a key
    monkeypatch.setenv("ANTHROPIC_API_KEY", " real-env-key ")
    assert secrets.anthropic_api_key() == "real-env-key"
    monkeypatch.setattr(secrets, "_from_streamlit", lambda name: "from-st-secrets")
    assert secrets.anthropic_api_key() == "from-st-secrets"


def test_export_secrets_to_env_never_overwrites(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {"ANTHROPIC_API_KEY": "st-key", "HF_TOKEN": "st-token", "NBALAB_DATA_REPO": "hf_..."}
    monkeypatch.setattr(secrets, "_from_streamlit", fake.get)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.delenv("NBALAB_DATA_REPO", raising=False)
    assert secrets.export_secrets_to_env() == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "st-key" and os.environ["HF_TOKEN"] == "env-token"
    assert "NBALAB_DATA_REPO" not in os.environ  # placeholder not exported


def test_int_setting_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "_from_streamlit", lambda name: None)
    monkeypatch.setenv("NBALAB_LLM_PER_SESSION_PER_HOUR", "lots")
    assert secrets.get_int_setting("NBALAB_LLM_PER_SESSION_PER_HOUR", 20) == 20
    monkeypatch.setenv("NBALAB_LLM_PER_SESSION_PER_HOUR", "5")
    assert secrets.get_int_setting("NBALAB_LLM_PER_SESSION_PER_HOUR", 20) == 5


# ---------------------------------------------------------------------- rate limits


def test_sliding_window_allows_max_then_frees_oldest_slot() -> None:
    clock = FakeClock()
    lim = SlidingWindowLimiter(max_calls=2, window_seconds=60, clock=clock)
    assert lim.try_acquire() and not clock.__setattr__("t", 10) and lim.try_acquire()
    assert not lim.try_acquire()
    assert lim.retry_after() == pytest.approx(50)
    clock.t = 60  # first call (t=0) leaves the window
    assert lim.remaining() == 1 and lim.try_acquire()
    assert not lim.try_acquire()


def test_choose_parser_without_key_uses_rules_and_spends_nothing() -> None:
    session, app = SlidingWindowLimiter(1), SlidingWindowLimiter(1)
    route = choose_parser(False, session, app)
    assert route.parser == "rules" and "No Anthropic API key" in route.reason
    assert session.remaining() == 1 and app.remaining() == 1


def test_choose_parser_session_limit() -> None:
    clock = FakeClock()
    session = SlidingWindowLimiter(2, clock=clock)
    app = SlidingWindowLimiter(100, clock=clock)
    assert [choose_parser(True, session, app).parser for _ in range(3)] == ["llm", "llm", "rules"]
    assert app.remaining() == 98  # the refused question did not spend app quota
    clock.t = 3600
    assert choose_parser(True, session, app).parser == "llm"


def test_choose_parser_app_limit_spans_sessions() -> None:
    app = SlidingWindowLimiter(3)
    routes = [choose_parser(True, SlidingWindowLimiter(20), app) for _ in range(4)]  # 4 new tabs
    assert [r.parser for r in routes] == ["llm", "llm", "llm", "rules"]
    assert "hourly AI budget" in routes[-1].reason

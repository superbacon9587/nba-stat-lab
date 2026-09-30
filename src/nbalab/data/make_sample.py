"""Cut a small raw-format sample of ``data/raw/`` for tests.

Run with ``python -m nbalab.data.make_sample``. Keeps every game (all game
types, including preseason) involving one team in the chosen seasons, plus the
matching rows of every other raw table, in the same CSV format as the raw files.
The default (Seattle SuperSonics 2007-08 -> Oklahoma City Thunder 2008-09)
covers a franchise relocation and both season-boundary months.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from nbalab.data.config import DATA_DIR

DEFAULT_TEAM_ID = 1610612760
DEFAULT_SEASONS = (2007, 2008)
GAME_TABLES = (
    "Games.csv",
    "PlayerStatistics.csv",
    "TeamStatistics.csv",
    "TeamStatisticsExtended.csv",
)
FULL_COPY_TABLES = ("TeamHistories.csv",)


def make_sample(raw_dir: Path, out_dir: Path, team_id: int, seasons: tuple[int, ...]) -> None:
    """Write the sampled CSVs to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    yy = ",".join(f"'{s % 100:02d}'" for s in seasons)
    con.execute(
        f"""
        create temp table ids as
        select gameId from {csv_source(raw_dir / "Games.csv")}
        where (hometeamId = '{team_id}' or awayteamId = '{team_id}')
          and substr(gameId, 2, 2) in ({yy})
        """
    )
    for name in GAME_TABLES:
        copy_rows(con, raw_dir / name, out_dir / name, "gameId in (select gameId from ids)")
    copy_rows(
        con,
        raw_dir / "Players.csv",
        out_dir / "Players.csv",
        f"""personId in (select personId from {csv_source(out_dir / "PlayerStatistics.csv")})""",
    )
    for name in FULL_COPY_TABLES:
        copy_rows(con, raw_dir / name, out_dir / name, "true")


def csv_source(path: Path) -> str:
    """duckdb reader for a raw CSV as all-text; the quote char is explicit because sniffing misses it."""
    return f"read_csv('{path}', all_varchar=true, quote='\"', sample_size=-1)"


def copy_rows(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, where: str) -> None:
    """Copy matching rows as text so values round-trip exactly as in the raw file."""
    con.execute(
        f"copy (select * from {csv_source(src)} where {where}) "
        f"to '{dst}' (header, delimiter ',')"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team-id", type=int, default=DEFAULT_TEAM_ID)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "sample")
    args = parser.parse_args()
    make_sample(DATA_DIR / "raw", args.out_dir, args.team_id, tuple(args.seasons))
    for f in sorted(args.out_dir.glob("*.csv")):
        print(f"{f.name:<32}{f.stat().st_size / 1e3:>10.0f} KB")


if __name__ == "__main__":
    main()

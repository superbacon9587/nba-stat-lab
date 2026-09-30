"""Build ``data/processed/matchups.parquet`` from the cached BoxScoreMatchupsV3 responses.

One row per (game, offensive player, defender) with partial possessions,
points, assists, FGA/FGM etc., plus the defender's height, weight and
position from ``Players.csv`` (via ``players.parquet``).

Run ``scripts/fetch_matchups.py`` first. Safe to re-run at any time; it only
reads what is cached.

Usage:
    uv run python scripts/build_matchups.py
"""

from __future__ import annotations

from nbalab.data.config import BuildConfig
from nbalab.data.matchups import build_matchups, load_player_bio


def main() -> None:
    cfg = BuildConfig()
    cache_dir = cfg.raw_dir / "matchups"
    bio = load_player_bio(cfg.processed_dir, cfg.raw_dir)
    df = build_matchups(cache_dir, bio)
    if df.empty:
        print(f"No cached games with matchup rows in {cache_dir}. Run scripts/fetch_matchups.py first.")
        return
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.processed_dir / "matchups.parquet"
    df.to_parquet(out, index=False)
    print(
        f"Wrote {out}: {len(df):,} rows, {df['gameId'].nunique():,} games, "
        f"seasons {df['season'].min()}-{df['season'].max()}. "
        f"Defender height known on {df['defHeightInches'].notna().mean():.1%} of rows "
        f"({df.loc[df['defHeightInches'].notna(), 'partialPossessions'].sum() / df['partialPossessions'].sum():.1%} of possessions)."
    )


if __name__ == "__main__":
    main()

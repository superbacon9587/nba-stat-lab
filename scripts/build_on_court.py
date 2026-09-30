"""Build ``data/processed/on_court.parquet``: shared floor time between opposing players.

Reconstructs five-man units from ``PlayByPlay.parquet`` (substitutions plus
box-score starters), then writes, for every player-game, the seconds shared on
the floor with each opponent and the player's stats during that time.

This is SHARED FLOOR TIME, not "guarded by". See ``nbalab.data.on_court``.

Outputs (all in ``data/processed/``):
    on_court.parquet          one row per (game, player, opponent)
    on_court_players.parquet  one row per player-game: reconstructed vs box minutes/points
    on_court_quality.parquet  one row per game: 5-on-5 time share and repair counts

Usage:
    uv run python scripts/build_on_court.py                 # 1996-97 onward
    uv run python scripts/build_on_court.py --seasons 2015 2023 --workers 4
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from nbalab.data.config import BuildConfig
from nbalab.data.on_court import (
    MINUTES_TOLERANCE,
    attach_game_info,
    build_games,
    compare_to_box,
    load_rosters,
    read_pbp,
)
from nbalab.data.pbp import normalize_events

# gameId type digits kept: 2 regular (incl. NBA Cup group/knockout), 4 playoffs,
# 5 play-in, 6 NBA Cup final. 1 preseason and 3 All-Star are excluded.
KEPT_TYPE_DIGITS = ("2", "4", "5", "6")
# A (player, opponent) row is flagged reliable only if the game's lineups were
# 5-on-5 at least this share of the time and both players' minutes check out.
MIN_FIVE_ON_FIVE_SHARE = 0.95


def season_prefixes(season: int) -> list[str]:
    """gameId prefixes (without leading zeros) for one season's kept game types."""
    return [f"{d}{season % 100:02d}" for d in KEPT_TYPE_DIGITS]


def build_season(season: int, raw_dir: Path, rosters: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Reconstruct every kept game of one season."""
    pbp = read_pbp(raw_dir, season_prefixes(season))
    if pbp.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    events = normalize_events(pbp)
    season_rosters = rosters[rosters["gameId"].isin(events["gameId"].unique())]
    pairs, players, quality = build_games(events, season_rosters)
    players = compare_to_box(players, season_rosters)
    return pairs, players, quality


def finalize(pairs: pd.DataFrame, players: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Add game info, minutes, and the reliability flag to the pair table."""
    ok_player = players.set_index(["gameId", "personId"])["reliable"]
    ok_game = quality.set_index("gameId")["fiveOnFiveShare"] >= MIN_FIVE_ON_FIVE_SHARE
    pairs = attach_game_info(pairs, players)
    pairs["shared_floor_minutes"] = pairs["sharedFloorSeconds"] / 60  # name matches nbalab.query.data loader
    idx_x = pd.MultiIndex.from_arrays([pairs["gameId"], pairs["personId"]])
    idx_y = pd.MultiIndex.from_arrays([pairs["gameId"], pairs["opponentPersonId"]])
    pairs["reliable"] = (
        ok_player.reindex(idx_x).fillna(False).to_numpy()
        & ok_player.reindex(idx_y).fillna(False).to_numpy()
        & ok_game.reindex(pairs["gameId"]).fillna(False).to_numpy()
    )
    return pairs


def main() -> None:
    cfg = BuildConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"), default=(cfg.start_season, 2025))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=cfg.processed_dir)
    args = ap.parse_args()

    t0 = time.time()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    print("Loading box-score rosters ...", flush=True)
    rosters = load_rosters(cfg.raw_dir)
    parts: dict[int, tuple[pd.DataFrame, ...]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for s in seasons:
            yy = s % 100
            season_rosters = rosters[(rosters["gameId"].astype("int64") // 100_000) % 100 == yy]
            futures[pool.submit(build_season, s, cfg.raw_dir, season_rosters)] = s
        for fut in as_completed(futures):
            s = futures[fut]
            parts[s] = fut.result()
            q = parts[s][2]
            pl = parts[s][1]
            if len(q):
                print(
                    f"  {s}-{(s + 1) % 100:02d}: {len(q):5d} games, 5-on-5 share {q['fiveOnFiveShare'].mean():.3f}, "
                    f"player-games within {MINUTES_TOLERANCE:g} min {pl['reliable'].mean():.3f}  "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    ordered = [parts[s] for s in seasons if len(parts[s][0])]
    pairs = pd.concat([p[0] for p in ordered], ignore_index=True)
    players = pd.concat([p[1] for p in ordered], ignore_index=True)
    quality = pd.concat([p[2] for p in ordered], ignore_index=True)
    pairs = finalize(pairs, players, quality)
    players = attach_game_info(players, players)
    quality = attach_game_info(quality, players)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.out_dir / "on_court.parquet", index=False)
    players.to_parquet(args.out_dir / "on_court_players.parquet", index=False)
    quality.to_parquet(args.out_dir / "on_court_quality.parquet", index=False)
    print(
        f"Wrote {len(pairs):,} pair rows, {len(players):,} player-games, {len(quality):,} games "
        f"in {time.time() - t0:.0f}s. Reliable pair rows: {pairs['reliable'].mean():.1%}"
    )


if __name__ == "__main__":
    main()

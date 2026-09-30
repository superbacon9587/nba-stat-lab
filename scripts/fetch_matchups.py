"""Download official NBA defensive matchup data (BoxScoreMatchupsV3), one game at a time.

This is the NBA's player-tracking matchup data: for every offensive player, which
defenders guarded him and for how many partial possessions, with the points,
assists and shots that came in those possessions. It exists from 2017-18 on.

Run this on your own machine: stats.nba.com often blocks cloud servers.

Behavior
--------
- Every response is cached as ``data/raw/matchups/<season>/<gameId>.json.gz``
  and a game with a cache file is never requested again. Files are written
  atomically (temp file + rename), so a crash never leaves a half-written file.
- Resumable: re-run the same command after a crash or Ctrl-C and it continues
  where it stopped.
- Sleeps a random 0.6-1.0 s between requests. Failed requests are retried with
  exponential backoff (5 s, 10 s, 20 s, 40 s, 80 s, plus jitter). A game that
  still fails is logged to ``data/raw/matchups/_failures.jsonl`` and retried
  on the next run.
- Stops early after 8 consecutive failed games. That pattern usually means the
  NBA is blocking or rate-limiting you. Wait a while and re-run.

Usage
-----
    uv run python scripts/fetch_matchups.py --dry-run          # count + time estimate, no download
    uv run python scripts/fetch_matchups.py --limit 5          # small test
    uv run python scripts/fetch_matchups.py                    # everything 2017-18 onward
    uv run python scripts/fetch_matchups.py --seasons 2023 2023
    uv run python scripts/fetch_matchups.py --game-ids 0022300061 0042200405
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb

from nbalab.data.config import BuildConfig
from nbalab.data.matchups import FIRST_MATCHUP_SEASON, cache_path

SLEEP_RANGE = (0.6, 1.0)
MAX_ATTEMPTS = 6
BACKOFF_BASE = 5.0
MAX_CONSECUTIVE_FAILURES = 8
TIMEOUT = 30
# gameId type digit: 2 regular season (incl. NBA Cup group/knockout), 4 playoffs,
# 5 play-in, 6 NBA Cup final. Preseason (1) and All-Star (3) are skipped.
TYPE_DIGITS = (2, 4, 5, 6)


def list_games(raw_dir: Path, first: int, last: int) -> list[tuple[str, int]]:
    """``(10-digit gameId, season)`` for every kept game in ``Games.csv``, oldest first."""
    games = (raw_dir / "Games.csv").as_posix()
    rows = duckdb.sql(
        f"""
        select gameId, (gameId // 100000) % 100 as yy
        from read_csv('{games}', quote='"', escape='"')
        where gameId // 10000000 in {TYPE_DIGITS}
          and gameDateTimeEst < now()
        order by gameDateTimeEst, gameId
        """
    ).fetchall()
    out = []
    for gid, yy in rows:
        season = 2000 + int(yy) if yy < 46 else 1900 + int(yy)
        if first <= season <= last:
            out.append((f"{int(gid):010d}", season))
    return out


def fetch_one(game_id: str) -> dict:
    """One raw BoxScoreMatchupsV3 response as a dict. Raises on HTTP / parse errors."""
    from nba_api.stats.library.http import NBAStatsHTTP

    response = NBAStatsHTTP().send_api_request(
        endpoint="boxscorematchupsv3",
        parameters={"GameID": game_id},
        timeout=TIMEOUT,
    )
    status = getattr(response, "get_response_status_code", lambda: 200)()
    if status != 200:
        raise RuntimeError(f"HTTP {status}")
    data = response.get_dict()  # raises if the body is not JSON (e.g. a block page)
    if "boxScoreMatchups" not in data:
        raise RuntimeError(f"unexpected response keys: {sorted(data)[:5]}")
    return data


def fetch_with_retries(game_id: str) -> tuple[dict | None, str | None, float]:
    """Try up to ``MAX_ATTEMPTS`` times. Returns (data, last_error, seconds spent requesting)."""
    error = None
    spent = 0.0
    for attempt in range(MAX_ATTEMPTS):
        t = time.time()
        try:
            data = fetch_one(game_id)
            return data, None, spent + time.time() - t
        except Exception as exc:  # network errors, timeouts, HTTP errors, bad JSON
            spent += time.time() - t
            error = f"{type(exc).__name__}: {exc}"[:300]
            if attempt < MAX_ATTEMPTS - 1:
                wait = BACKOFF_BASE * 2**attempt + random.uniform(0, 2)
                print(f"    {game_id}: attempt {attempt + 1} failed ({error}); retrying in {wait:.0f}s", flush=True)
                time.sleep(wait)
    return None, error, spent


def write_atomic(path: Path, data: dict) -> None:
    """Write gzipped JSON via a temp file + rename so partial files never exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def log_failure(cache_dir: Path, game_id: str, error: str | None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "_failures.jsonl", "a") as f:
        f.write(json.dumps({"gameId": game_id, "error": error, "at": datetime.now().isoformat()}) + "\n")


def estimate(n: int, seconds_per_request: float) -> str:
    """Human-readable time for ``n`` requests: average sleep + request latency."""
    per_game = sum(SLEEP_RANGE) / 2 + seconds_per_request
    hours = n * per_game / 3600
    return f"{n:,} games x ~{per_game:.1f}s = ~{hours:.1f} h"


def main() -> None:
    cfg = BuildConfig()
    ap = argparse.ArgumentParser(description="Download NBA BoxScoreMatchupsV3 data with caching.")
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"), default=(FIRST_MATCHUP_SEASON, 2025))
    ap.add_argument("--game-ids", nargs="+", help="Specific 10-digit game ids (overrides --seasons).")
    ap.add_argument("--limit", type=int, help="Download at most this many uncached games.")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be downloaded and a time estimate.")
    ap.add_argument("--cache-dir", type=Path, default=cfg.raw_dir / "matchups")
    args = ap.parse_args()

    if args.game_ids:
        games = [(g.zfill(10), 2000 + int(g.zfill(10)[3:5])) for g in args.game_ids]
    else:
        games = list_games(cfg.raw_dir, *args.seasons)
    todo = [(g, s) for g, s in games if not cache_path(args.cache_dir, g, s).exists()]
    print(f"{len(games):,} games in scope, {len(games) - len(todo):,} already cached, {len(todo):,} to download.")
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"Estimate: {estimate(len(todo), 0.7)} (assumes ~0.7s per request; real speed shown as it runs).")
    if args.dry_run or not todo:
        return

    done = failed = consecutive = 0
    request_time = 0.0
    t0 = time.time()
    try:
        for i, (game_id, season) in enumerate(todo, 1):
            data, error, spent = fetch_with_retries(game_id)
            request_time += spent
            if data is None:
                failed += 1
                consecutive += 1
                log_failure(args.cache_dir, game_id, error)
                print(f"  FAILED {game_id}: {error}", flush=True)
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    print(
                        f"Stopping: {consecutive} games failed in a row. stats.nba.com is probably "
                        "blocking or rate-limiting this machine. Wait and re-run; progress is saved.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
            else:
                write_atomic(cache_path(args.cache_dir, game_id, season), data)
                done += 1
                consecutive = 0
            if i % 25 == 0 or i == len(todo):
                elapsed = time.time() - t0
                left = (len(todo) - i) * elapsed / i
                print(
                    f"  [{i:,}/{len(todo):,}] ok={done:,} failed={failed:,} "
                    f"avg request {request_time / i:.2f}s, elapsed {elapsed / 60:.1f} min, ~{left / 3600:.1f} h left",
                    flush=True,
                )
            if i < len(todo):
                time.sleep(random.uniform(*SLEEP_RANGE))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run the same command to resume.")
        sys.exit(130)
    print(f"Done: {done:,} downloaded, {failed:,} failed in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()

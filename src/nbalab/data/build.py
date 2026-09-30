"""Build the processed parquet tables from ``data/raw/``.

Run with ``python -m nbalab.data.build [--start-season 1996]``. Writes:

- ``player_games.parquet``: one row per player per game played
- ``team_games.parquet``: one row per team per game
- ``players.parquet`` / ``teams.parquet``: id lookup tables with name aliases

Raw files are only read, never modified.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from nbalab.data import aliases as al
from nbalab.data.clean import (
    age_in_years,
    parse_minutes,
    position_group,
    resolve_game_type,
    season_from_date,
    season_from_game_id,
    season_label,
)
from nbalab.data.calendar import build_season_calendar, infer_conferences
from nbalab.data.config import REFERENCE_DIR, BuildConfig
from nbalab.data.features import (
    add_lagged_rolling_means,
    calendar_features,
    cumulative_team_ratings,
    days_since_previous,
    pregame_team_ratings,
    rest_features,
)
from nbalab.data.load import RawTables, load_raw

log = logging.getLogger("nbalab.data.build")

PLAYER_BOX_STATS = [
    "points", "assists", "reboundsTotal", "reboundsOffensive", "reboundsDefensive",
    "steals", "blocks", "turnovers", "foulsPersonal",
    "fieldGoalsMade", "fieldGoalsAttempted", "threePointersMade", "threePointersAttempted",
    "freeThrowsMade", "freeThrowsAttempted", "plusMinusPoints",
]

TEAM_BOX_STATS = [
    "teamScore", "opponentScore", "assists", "blocks", "steals", "turnovers", "foulsPersonal",
    "reboundsTotal", "reboundsOffensive", "reboundsDefensive",
    "fieldGoalsMade", "fieldGoalsAttempted", "threePointersMade", "threePointersAttempted",
    "freeThrowsMade", "freeThrowsAttempted", "numMinutes",
]

# Context columns computed at team level and copied onto each player row.
SHARED_CONTEXT = [
    "season", "season_label", "game_date", "game_type", "is_playoff", "is_play_in",
    "day_of_week", "day_of_week_num", "month", "week_of_season",
    "venue_team_id", "venue_city", "arena_name", "arena_city",
    "team_game_num", "team_days_since_last_game", "team_rest_days", "team_is_back_to_back",
    "opp_days_since_last_game", "opp_rest_days", "opp_is_back_to_back",
    "opp_pre_off_rating", "opp_pre_def_rating", "opp_pre_pace", "opp_pre_games",
]


# --------------------------------------------------------------------------- games


def build_game_meta(raw: RawTables) -> pd.DataFrame:
    """One row per game: timestamp, home/away ids and names, raw label, arena.

    Arena names come from ``Games.csv`` and the schedule files where available
    (recent seasons only); elsewhere they stay null.
    """
    g = raw.games.drop_duplicates("gameId").copy()
    sched = raw.schedules.set_index("gameId")[["arenaName", "arenaCity"]]
    g = g.set_index("gameId")
    g[["arenaName", "arenaCity"]] = g[["arenaName", "arenaCity"]].combine_first(
        sched.reindex(g.index)
    )
    return g.reset_index().rename(columns={"gameType": "gameLabel"})


def attach_season_and_type(df: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    """Add ``season`` (from gameId) and normalized ``game_type`` to any per-game table.

    The label from ``Games.csv`` is preferred; the table's own ``gameType`` is the fallback.
    """
    labels = df[["gameId"]].merge(game_meta[["gameId", "gameLabel"]], on="gameId", how="left")
    out = df.copy()
    out["season"] = season_from_game_id(out["gameId"])
    out["game_type"] = resolve_game_type(
        out["gameId"], labels["gameLabel"].set_axis(out.index), out["gameType"]
    ).values
    return out


def log_season_mismatches(df: pd.DataFrame) -> None:
    """Log games whose gameId season disagrees with the October-cutoff date rule.

    The only expected disagreement is the 2020 bubble (Finals in October 2020,
    season 2019-20); the gameId is authoritative.
    """
    by_date = season_from_date(df["gameDateTimeEst"])
    bad = df.loc[by_date.ne(df["season"]), ["gameId", "gameDateTimeEst", "season"]].drop_duplicates("gameId")
    if len(bad):
        dates = bad["gameDateTimeEst"].dt.strftime("%Y-%m")
        log.info("season: %d games where gameId season != date rule (months: %s); using gameId",
                 len(bad), ", ".join(sorted(dates.unique())))


def keep_analysis_games(df: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Keep seasons from ``start_season`` on and only the included game types."""
    mask = df["season"].ge(config.start_season) & df["game_type"].isin(config.included_game_types)
    return df.loc[mask].copy()


def team_identity_by_season(histories: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """City, name and abbreviation a team used in a given season.

    ``keys`` has ``teamId`` and ``season``. Franchises that moved or renamed keep
    their teamId, so the season decides which city/name applies.
    """
    h = histories.assign(teamAbbrev=histories["teamAbbrev"].str.strip())
    pairs = keys[["teamId", "season"]].drop_duplicates()
    m = pairs.merge(h, on="teamId", how="left")
    m = m[(m["seasonFounded"] <= m["season"]) & (m["season"] <= m["seasonActiveTill"])]
    m = m.drop_duplicates(["teamId", "season"], keep="last")
    return m[["teamId", "season", "teamCity", "teamName", "teamAbbrev"]]


# ---------------------------------------------------------------------- team games


def build_team_games(raw: RawTables, game_meta: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """One row per team per game with box score, advanced stats, context and form."""
    t = drop_placeholder_team_rows(raw.team_stats, game_meta).drop_duplicates(["gameId", "teamId"])
    t = keep_analysis_games(attach_season_and_type(t, game_meta), config)
    log_season_mismatches(t)
    t = t.merge(raw.team_advanced.drop_duplicates(["gameId", "teamId"]), on=["gameId", "teamId"], how="left")
    t = t.sort_values(["teamId", "gameDateTimeEst", "gameId"]).reset_index(drop=True)

    t = add_game_context(t, game_meta, raw.team_histories)
    t = add_team_schedule_features(t)
    t = add_opponent_features(t)
    t = add_lagged_rolling_means(t, "teamId", config.team_rolling_stats, config.rolling_windows)
    return t


def drop_placeholder_team_rows(team_stats: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    """Keep only team rows whose teamId is one of the game's two teams in ``Games.csv``.

    Postponed games leave placeholder rows (teamId 0, original date, no stats)
    next to the real rows of the rescheduled game under the same gameId.
    """
    teams = game_meta[["gameId", "hometeamId", "awayteamId"]]
    m = team_stats.merge(teams, on="gameId", how="left")
    ok = (m["teamId"].eq(m["hometeamId"]) | m["teamId"].eq(m["awayteamId"])).fillna(False)
    dropped = int((~ok).sum())
    if dropped:
        log.info("team_games: dropped %d placeholder/unmatched team rows", dropped)
    return team_stats.loc[ok.to_numpy()]


def add_game_context(t: pd.DataFrame, game_meta: pd.DataFrame, histories: pd.DataFrame) -> pd.DataFrame:
    """Season label, game-type flags, calendar fields and venue for each team-game row.

    ``venue_team_id`` / ``venue_city`` identify the home team and its franchise
    city that season (consistent across all seasons). ``arena_name`` /
    ``arena_city`` are the physical arena, known only for recent seasons; they
    also reveal neutral-site games (e.g. a "home" game played in Mexico City).
    """
    t = t.copy()
    t["season_label"] = season_label(t["season"])
    t["game_date"] = t["gameDateTimeEst"].dt.normalize()
    t["is_playoff"] = t["game_type"].eq("playoffs")
    t["is_play_in"] = t["game_type"].eq("play_in")

    openers = (
        t[t["game_type"].isin(["regular", "nba_cup"])].groupby("season")["gameDateTimeEst"].min()
    )
    cal = calendar_features(t["gameDateTimeEst"], t["season"].map(openers))
    t[cal.columns] = cal

    t["venue_team_id"] = np.where(t["home"].eq(1).fillna(False), t["teamId"], t["opponentTeamId"]).astype("int64")
    ident = team_identity_by_season(
        histories, t[["venue_team_id", "season"]].rename(columns={"venue_team_id": "teamId"})
    ).rename(columns={"teamId": "venue_team_id", "teamCity": "venue_city"})
    t = t.merge(ident[["venue_team_id", "season", "venue_city"]], on=["venue_team_id", "season"], how="left")
    arena = game_meta.set_index("gameId")
    t["arena_name"] = t["gameId"].map(arena["arenaName"])
    t["arena_city"] = t["gameId"].map(arena["arenaCity"])
    return t


def add_team_schedule_features(t: pd.DataFrame) -> pd.DataFrame:
    """Game number within the season, days since the last game, rest days, back-to-back flag."""
    t = t.sort_values(["teamId", "gameDateTimeEst", "gameId"]).reset_index(drop=True)
    t["team_game_num"] = t.groupby(["teamId", "season"]).cumcount() + 1
    days = days_since_previous(t, ["teamId", "season"], "gameDateTimeEst")
    rest = rest_features(days, "team_")
    t[rest.columns] = rest
    return t


def add_opponent_features(t: pd.DataFrame) -> pd.DataFrame:
    """Own and opponent pre-game season-to-date ratings, plus opponent rest.

    Ratings come from :func:`pregame_team_ratings`, which only uses games played
    before this one, so no information from the game itself leaks in.
    """
    pre = pregame_team_ratings(t, cumulative_team_ratings(t)).rename(
        columns={
            "std_off_rating": "pre_off_rating",
            "std_def_rating": "pre_def_rating",
            "std_pace": "pre_pace",
            "std_games": "pre_games",
        }
    )
    t = t.merge(pre, on=["gameId", "teamId"], how="left")

    opp_cols = {
        "teamId": "opponentTeamId",
        "pre_off_rating": "opp_pre_off_rating",
        "pre_def_rating": "opp_pre_def_rating",
        "pre_pace": "opp_pre_pace",
        "pre_games": "opp_pre_games",
        "team_days_since_last_game": "opp_days_since_last_game",
        "team_rest_days": "opp_rest_days",
        "team_is_back_to_back": "opp_is_back_to_back",
    }
    opp = t[["gameId", *opp_cols]].rename(columns=opp_cols)
    return t.merge(opp, on=["gameId", "opponentTeamId"], how="left")


# -------------------------------------------------------------------- player games


def recover_team_ids(ps: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    """Fill null ``playerteamId`` / ``opponentteamId`` from the game's home/away teams.

    About 6.5% of player rows have null team ids but a team name. Within one
    game the two teams never share a name, so matching the player's team name to
    the game's home or away name identifies the id unambiguously.
    """
    cols = ["gameId", "hometeamId", "hometeamName", "awayteamId", "awayteamName"]
    m = ps.merge(game_meta[cols], on="gameId", how="left")
    is_home = m["playerteamName"].eq(m["hometeamName"]).fillna(False)
    is_away = m["playerteamName"].eq(m["awayteamName"]).fillna(False)
    team = pd.Series(pd.NA, index=m.index, dtype="Int64")
    team = team.mask(is_home, m["hometeamId"]).mask(is_away, m["awayteamId"])
    opp = pd.Series(pd.NA, index=m.index, dtype="Int64")
    opp = opp.mask(is_home, m["awayteamId"]).mask(is_away, m["hometeamId"])
    out = ps.copy()
    out["playerteamId"] = out["playerteamId"].fillna(team.set_axis(out.index))
    out["opponentteamId"] = out["opponentteamId"].fillna(opp.set_axis(out.index))
    return out


def clean_player_rows(raw: RawTables, game_meta: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Filter player box scores to analysis games actually played, with ids repaired.

    Drops seasons before the start season, preseason/All-Star, and rows where the
    player did not play (minutes null or zero). One row per (personId, gameId) is
    guaranteed; if the raw file ever repeats a pair, the row with most minutes wins.
    """
    ps = raw.player_stats
    ps = ps[season_from_game_id(ps["gameId"]) >= config.start_season]
    ps = keep_analysis_games(attach_season_and_type(ps, game_meta), config)
    ps = recover_team_ids(ps, game_meta)
    ps["minutes"] = parse_minutes(ps["numMinutes"])
    ps = ps[ps["minutes"].gt(0) & ps["playerteamId"].notna() & ps["opponentteamId"].notna()]
    ps = ps.sort_values("minutes", ascending=False).drop_duplicates(["personId", "gameId"])
    return ps.astype({"playerteamId": "int64", "opponentteamId": "int64", "personId": "int64"})


def build_player_games(
    ps: pd.DataFrame, bio: pd.DataFrame, team_games: pd.DataFrame, config: BuildConfig
) -> pd.DataFrame:
    """One row per player per game played, enriched with bio, opponent, context and form.

    ``ps`` is the output of :func:`clean_player_rows`; ``bio`` of :func:`player_bio_table`.
    """
    ps = ps.drop(columns=["season", "game_type"])
    context = team_games[["gameId", "teamId", "home", *SHARED_CONTEXT]].rename(
        columns={"teamId": "playerteamId"}
    )
    p = ps.merge(context, on=["gameId", "playerteamId"], how="inner", suffixes=("_raw", ""))
    if len(p) < len(ps):
        log.info("player_games: dropped %d rows with no matching team-game row", len(ps) - len(p))
    p["home"] = p["home"].fillna(p["home_raw"])
    p["gameDateTimeEst"] = p["gameDateTimeEst"].fillna(p["game_date"])
    p["starter"] = p["startingPosition"].notna()

    p = add_player_bio(p, bio)
    p = p.sort_values(["personId", "gameDateTimeEst", "gameId"]).reset_index(drop=True)
    p["player_game_num"] = p.groupby(["personId", "season"]).cumcount() + 1
    p["player_days_since_last_game"] = days_since_previous(p, ["personId"], "gameDateTimeEst")
    p = add_lagged_rolling_means(p, "personId", config.player_rolling_stats, config.rolling_windows)
    return p


def starting_position_mode(ps: pd.DataFrame) -> pd.Series:
    """Each player's most frequent starting position (G/F/C), indexed by personId."""
    starts = ps.dropna(subset=["startingPosition"])
    counts = starts.groupby(["personId", "startingPosition"]).size().rename("n").reset_index()
    top = counts.sort_values("n", ascending=False).drop_duplicates("personId")
    return top.set_index("personId")["startingPosition"].astype("object")


def player_bio_table(players: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    """Bio per personId for every player in ``Players.csv`` or ``ps``.

    Position group uses the G/F/C flags, with the player's most common starting
    position as tie-breaker/fallback (see :func:`position_group`). Players missing
    from ``Players.csv`` get their name from the box scores and null bio fields.
    """
    b = players.drop_duplicates("personId").astype({"personId": "int64"}).set_index("personId")
    box_names = ps.drop_duplicates("personId", keep="last").set_index("personId")[["firstName", "lastName"]]
    b = b.reindex(b.index.union(box_names.index))
    b[["firstName", "lastName"]] = b[["firstName", "lastName"]].combine_first(box_names)
    flags = b[["guard", "forward", "center"]].fillna(0).astype(bool)
    b["position"] = flags.apply(
        lambda r: "-".join(p for p, on in zip("GFC", r) if on) or pd.NA, axis=1
    )
    b["position_group"] = position_group(
        b["guard"], b["forward"], b["center"], starting_position_mode(ps)
    )
    return b


def add_player_bio(p: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """Attach height, weight, position group, and age on game day (null where unknown)."""
    cols = ["heightInches", "bodyWeightLbs", "birthDate", "position_group"]
    p = p.merge(bio[cols], left_on="personId", right_index=True, how="left")
    p["age"] = age_in_years(p["birthDate"], p["game_date"])
    return p.drop(columns=["birthDate"])


# ------------------------------------------------------------------ lookup tables


def build_players_lookup(player_games: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """One row per player in ``player_games`` with bio, career span and name aliases."""
    agg = player_games.groupby("personId").agg(
        first_season=("season", "min"),
        last_season=("season", "max"),
        games=("gameId", "size"),
    )
    last_team = (
        player_games.sort_values("gameDateTimeEst").groupby("personId")["playerteamId"].last()
    )
    df = agg.join(last_team.rename("last_team_id")).join(bio, how="left")
    df["full_name"] = (df["firstName"].fillna("") + " " + df["lastName"].fillna("")).str.strip()
    generated = pd.Series(
        [al.player_aliases(f, l) for f, l in zip(df["firstName"].fillna(""), df["lastName"].fillna(""))],
        index=df.index,
    )
    nick = al.load_nicknames(REFERENCE_DIR / "player_nicknames.csv", "personId")
    df["aliases"] = al.attach_nicknames(generated, nick, df.index.to_series(), "personId")
    keep = [
        "firstName", "lastName", "full_name", "aliases", "position", "position_group",
        "heightInches", "bodyWeightLbs", "birthDate", "draftYear", "draftRound", "draftNumber",
        "first_season", "last_season", "games", "last_team_id",
    ]
    return df.reset_index()[["personId", *keep]]


def build_teams_lookup(raw: RawTables, team_games: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """One row per team with its current name, every name it used since the start season, and aliases.

    Relocated/renamed franchises keep one teamId, so e.g. "Seattle SuperSonics"
    and "Oklahoma City Thunder" are both aliases of the same row. The
    ``name_history`` column says which seasons each name applies to, which the
    resolver needs for names shared across franchises ("Hornets").
    """
    team_ids = set(team_games["teamId"].unique())
    h = raw.team_histories.assign(teamAbbrev=raw.team_histories["teamAbbrev"].str.strip())
    h = h[h["teamId"].isin(team_ids) & h["seasonActiveTill"].ge(config.start_season)]
    h = h.sort_values(["teamId", "seasonFounded"])
    nick = al.load_nicknames(REFERENCE_DIR / "team_nicknames.csv", "teamId")

    rows = []
    for team_id, eras in h.groupby("teamId"):
        current = eras.iloc[-1]
        generated = sorted(
            {a for _, e in eras.iterrows() for a in al.team_aliases(e.teamCity, e.teamName, e.teamAbbrev)}
        )
        rows.append(
            {
                "teamId": int(team_id),
                "abbrev": current.teamAbbrev,
                "city": current.teamCity,
                "name": current.teamName,
                "full_name": f"{current.teamCity} {current.teamName}",
                "aliases": generated,
                "name_history": [
                    {
                        "city": e.teamCity,
                        "name": e.teamName,
                        "abbrev": e.teamAbbrev,
                        "first_season": int(e.seasonFounded),
                        "last_season": None if e.seasonActiveTill >= 2100 else int(e.seasonActiveTill),
                    }
                    for _, e in eras.iterrows()
                ],
            }
        )
    df = pd.DataFrame(rows)
    df["aliases"] = al.attach_nicknames(df["aliases"], nick, df["teamId"], "teamId")
    return df


# ------------------------------------------------------------------------ output


def shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce file size: float32 for measurements, categories for low-cardinality text."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_float_dtype(s):
            out[col] = s.astype("float32")
        elif pd.api.types.is_string_dtype(s) and is_low_cardinality(s):
            out[col] = s.astype("category")
    return out


def is_low_cardinality(s: pd.Series) -> bool:
    """True for text columns with few distinct values (list-valued columns are never)."""
    try:
        return s.nunique() < 0.05 * max(len(s), 1)
    except TypeError:
        return False


def select_player_columns(p: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Final column order for ``player_games.parquet``."""
    rolling = [f"{s}_last{w}" for s in config.player_rolling_stats for w in config.rolling_windows]
    cols = [
        "personId", "gameId", "playerteamId", "opponentteamId", "gameDateTimeEst", *SHARED_CONTEXT,
        "home", "win", "starter", "startingPosition", "minutes", *PLAYER_BOX_STATS,
        "heightInches", "bodyWeightLbs", "age", "position_group",
        "player_game_num", "player_days_since_last_game", *rolling,
    ]
    return p[cols].rename(columns={"playerteamId": "teamId", "opponentteamId": "opponentTeamId"})


def select_team_columns(t: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Final column order for ``team_games.parquet``."""
    advanced = [
        "offensiveRating", "defensiveRating", "netRating", "pace", "possessions",
        "effectiveFieldGoalPercentage", "trueShootingPercentage", "teamTurnoverPercentage",
        "offensiveReboundPercentage", "freeThrowAttemptRate", "opponentEffectiveFieldGoalPercentage",
        "opponentTurnoverPercentage", "opponentOffensiveReboundPercentage", "opponentFreeThrowAttemptRate",
    ]
    rolling = [f"{s}_last{w}" for s in config.team_rolling_stats for w in config.rolling_windows]
    cols = [
        "teamId", "gameId", "opponentTeamId", "gameDateTimeEst", *SHARED_CONTEXT,
        "home", "win", *TEAM_BOX_STATS, *advanced,
        "pre_off_rating", "pre_def_rating", "pre_pace", "pre_games", *rolling,
    ]
    return t[cols]


def build_tables(raw: RawTables, config: BuildConfig) -> dict[str, pd.DataFrame]:
    """Pure transformation from raw tables to the processed tables (games, lookups, season calendar)."""
    game_meta = build_game_meta(raw)
    team_games = build_team_games(raw, game_meta, config)
    ps = clean_player_rows(raw, game_meta, config)
    bio = player_bio_table(raw.players, ps)
    player_games = build_player_games(ps, bio, team_games, config)
    team_out = select_team_columns(team_games, config)
    return {
        "player_games": select_player_columns(player_games, config),
        "team_games": team_out,
        "players": build_players_lookup(player_games, bio),
        "teams": build_teams_lookup(raw, team_games, config),
        "season_calendar": build_season_calendar(raw.games, team_out),
        "conferences": infer_conferences(team_out),
    }


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    """Write each table to ``<out_dir>/<name>.parquet`` with zstd compression."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        shrink(df).to_parquet(path, index=False, compression="zstd")
        paths[name] = path
    return paths


def report(tables: dict[str, pd.DataFrame], paths: dict[str, Path]) -> None:
    """Print row counts and file sizes, and flag if the total exceeds the 150MB deploy target."""
    total = 0
    print(f"\n{'table':<14}{'rows':>12}{'cols':>6}{'size MB':>10}")
    for name, path in paths.items():
        size = path.stat().st_size
        total += size
        print(f"{name:<14}{len(tables[name]):>12,}{tables[name].shape[1]:>6}{size / 1e6:>10.1f}")
    status = "OK" if total < 150e6 else "OVER TARGET"
    print(f"{'total':<32}{total / 1e6:>10.1f}  (target < 150 MB: {status})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start-season", type=int, help="first season's start year, e.g. 1996")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    config = BuildConfig()
    overrides = {
        "start_season": args.start_season,
        "raw_dir": args.raw_dir,
        "processed_dir": args.out_dir,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})

    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    print(f"Loading raw tables from {config.raw_dir} ...", flush=True)
    raw = load_raw(config.raw_dir, config.extra_schedule_files)
    print(f"Building tables for {season_label(config.start_season)} onward ...", flush=True)
    tables = build_tables(raw, config)
    report(tables, write_tables(tables, config.processed_dir))


if __name__ == "__main__":
    main()

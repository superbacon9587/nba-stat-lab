"""Who defended the subject, answered as honestly as the data allows.

Box scores never record who guarded whom. Each player-game therefore gets a
defender from the best source available for that game, and the source is
recorded so the UI can say what the number really means.

**Defender attributes** (height, weight, position), best source first:

1. ``official_matchups``: NBA tracking matchups (2017-18 on, if downloaded).
   Height and weight are averaged over the subject's defenders, weighted by
   partial possessions. The position is the main defender's.
2. ``same_position_starter`` (proxy): the opponent's starter at the subject's
   starting position (G/F/C) in that game. If there are two (two starting
   guards), it's the one who played more minutes. Only available when the
   subject started.
3. ``opponent_team_average`` (weak proxy): the opponent's roster height and
   weight that game, weighted by minutes. There is no position.

**A named defender** ("when guarded by LeBron"), best source first:

1. ``official_matchups``: the named defender guarded the subject for at least
   ``min_partial_possessions`` in that game.
2. ``shared_floor_time`` (proxy): the two were on the floor together for at
   least ``min_shared_minutes`` (from play-by-play lineups).
3. ``same_game`` (weak proxy): the named player played for the opponent in
   that game. This says they were in the same game, not that he guarded the subject.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from nbalab.query.data import QueryData

SOURCE_LABELS: dict[str, str] = {
    "official_matchups": "Official NBA tracking matchups (who actually defended whom)",
    "shared_floor_time": "PROXY: minutes both players were on the floor together (not 'guarded by')",
    "same_game": "WEAK PROXY: the defender played for the opponent in that game (not 'guarded by')",
    "same_position_starter": "PROXY: opponent's starter at the same starting position",
    "opponent_team_average": "WEAK PROXY: opponent roster height/weight, weighted by minutes",
    "none": "No defender information for this game",
}
SOURCE_RANK: dict[str, int] = {
    "official_matchups": 1, "shared_floor_time": 2, "same_position_starter": 2,
    "same_game": 3, "opponent_team_average": 3, "none": 9,
}

ATTRIBUTE_COLUMNS = ["def_person_id", "def_height", "def_weight", "def_position", "def_source"]


def opponent_rows(subject: pd.DataFrame, data: QueryData) -> pd.DataFrame:
    """Every opponent player-row in the subject's games (one row per opponent per game)."""
    pg = data.player_games
    keys = subject[["gameId", "opponentTeamId"]].drop_duplicates().rename(columns={"opponentTeamId": "teamId"})
    opp = pg.merge(keys, on=["gameId", "teamId"])
    return opp[["gameId", "personId", "teamId", "minutes", "starter", "startingPosition",
                "heightInches", "bodyWeightLbs", "position_group"]]


# ------------------------------------------------------------------ attributes


def from_matchups(subject: pd.DataFrame, data: QueryData) -> pd.DataFrame:
    """Level 1: possession-weighted defender height/weight from official matchups."""
    empty = pd.DataFrame(columns=["gameId", "personId", *ATTRIBUTE_COLUMNS])
    if data.matchups is None:
        return empty
    m = data.matchups.merge(subject[["gameId", "personId"]].drop_duplicates(),
                            left_on=["gameId", "offPersonId"], right_on=["gameId", "personId"])
    if m.empty:
        return empty
    bio = data.players.set_index("personId")[["heightInches", "bodyWeightLbs", "position_group"]]
    m = m.join(bio, on="defPersonId")
    m["w"] = m["partial_possessions"].clip(lower=0)
    keys = ["gameId", "personId"]
    top = m.sort_values("w", ascending=False).drop_duplicates(keys).set_index(keys)
    out = pd.DataFrame({
        "def_person_id": top["defPersonId"],
        "def_height": _weighted_by_group(m, "heightInches", keys),
        "def_weight": _weighted_by_group(m, "bodyWeightLbs", keys),
        "def_position": top["position_group"],
    })
    out["def_source"] = "official_matchups"
    return out.reset_index()[["gameId", "personId", *ATTRIBUTE_COLUMNS]]


def from_same_position_starter(subject: pd.DataFrame, opp: pd.DataFrame) -> pd.DataFrame:
    """Level 2: the opponent's starter at the subject's starting position (most minutes if two)."""
    cols = ["gameId", "personId", "opponentTeamId", "startingPosition"]
    starters = subject.dropna(subset=["startingPosition"])[cols]
    opp_starters = opp[opp["starter"]].dropna(subset=["startingPosition"])
    cand = opp_starters.astype({"startingPosition": "object"}).merge(
        starters.astype({"startingPosition": "object"}),
        on=["gameId", "startingPosition"],
        suffixes=("_def", ""),
    )
    cand = cand[cand["teamId"] == cand["opponentTeamId"]]
    if cand.empty:
        return pd.DataFrame(columns=["gameId", "personId", *ATTRIBUTE_COLUMNS])
    cand = cand.sort_values("minutes", ascending=False).drop_duplicates(["gameId", "personId"])
    return pd.DataFrame({
        "gameId": cand["gameId"].values,
        "personId": cand["personId"].values,
        "def_person_id": cand["personId_def"].values,
        "def_height": cand["heightInches"].astype("float64").values,
        "def_weight": cand["bodyWeightLbs"].astype("float64").values,
        "def_position": cand["position_group"].values,
        "def_source": "same_position_starter",
    })


def from_team_average(subject: pd.DataFrame, opp: pd.DataFrame) -> pd.DataFrame:
    """Level 3: opponent height and weight that game, weighted by minutes played."""
    o = opp.assign(
        hw=opp["heightInches"].astype("float64") * opp["minutes"],
        hm=opp["minutes"].where(opp["heightInches"].notna(), 0.0),
        ww=opp["bodyWeightLbs"].astype("float64") * opp["minutes"],
        wm=opp["minutes"].where(opp["bodyWeightLbs"].notna(), 0.0),
    )
    agg = o.groupby(["gameId", "teamId"])[["hw", "hm", "ww", "wm"]].sum()
    avg = pd.DataFrame({
        "def_height": agg["hw"] / agg["hm"].where(agg["hm"] > 0),
        "def_weight": agg["ww"] / agg["wm"].where(agg["wm"] > 0),
    })
    avg.index = avg.index.set_names(["gameId", "opponentTeamId"])
    out = subject[["gameId", "personId", "opponentTeamId"]].drop_duplicates(["gameId", "personId"])
    out = out.join(avg, on=["gameId", "opponentTeamId"])
    out["def_person_id"] = pd.NA
    out["def_position"] = None
    out["def_source"] = "opponent_team_average"
    return out[["gameId", "personId", *ATTRIBUTE_COLUMNS]]


def defender_attributes(subject: pd.DataFrame, data: QueryData, opp: pd.DataFrame | None = None) -> pd.DataFrame:
    """One defender per subject row from the best available source (see module docstring).

    Returns a frame aligned to ``subject.index`` with columns ``def_person_id,
    def_height, def_weight, def_position, def_source``. A level is used only if
    it gives a height for that game. Otherwise the next level is tried.
    """
    opp = opponent_rows(subject, data) if opp is None else opp
    keys = subject[["gameId", "personId"]]
    result = pd.DataFrame(index=subject.index, columns=ATTRIBUTE_COLUMNS, dtype="object")
    for level in (from_matchups(subject, data), from_same_position_starter(subject, opp),
                  from_team_average(subject, opp)):
        if level.empty:
            continue
        level = level[level["def_height"].notna()].drop_duplicates(["gameId", "personId"])
        aligned = keys.merge(level, on=["gameId", "personId"], how="left").set_index(subject.index)
        todo = result["def_source"].isna() & aligned["def_source"].notna()
        result.loc[todo, ATTRIBUTE_COLUMNS] = aligned.loc[todo, ATTRIBUTE_COLUMNS]
    result["def_source"] = result["def_source"].fillna("none")
    result["def_height"] = pd.to_numeric(result["def_height"], errors="coerce")
    result["def_weight"] = pd.to_numeric(result["def_weight"], errors="coerce")
    return result


# ------------------------------------------------------------- named defender


def named_defender_mask(
    subject: pd.DataFrame,
    defender_id: int,
    data: QueryData,
    min_partial_possessions: float,
    min_shared_minutes: float,
) -> tuple[pd.Series, pd.Series]:
    """Which subject games had ``defender_id`` defending, and the source used for each game.

    Per game, the best source that *covers that game* decides (see module
    docstring). Returns (mask, source), both aligned to ``subject.index``.
    """
    source = pd.Series("same_game", index=subject.index, dtype="object")
    mask = same_game_mask(subject, defender_id, data)

    if data.on_court is not None:
        oc = data.on_court
        covered = subject["gameId"].isin(oc["gameId"].unique())
        shared = oc[oc["opponentPersonId"] == defender_id].groupby(["gameId", "personId"])["shared_minutes"].sum()
        mins = pd.Series(list(zip(subject["gameId"], subject["personId"])), index=subject.index).map(shared).fillna(0.0)
        mask = mask.where(~covered, mins >= min_shared_minutes)
        source = source.where(~covered, "shared_floor_time")

    if data.matchups is not None:
        mu = data.matchups
        covered = pd.Series(list(zip(subject["gameId"], subject["personId"])), index=subject.index).isin(
            set(zip(mu["gameId"], mu["offPersonId"]))
        )
        poss = mu[mu["defPersonId"] == defender_id].groupby(["gameId", "offPersonId"])["partial_possessions"].sum()
        pp = pd.Series(list(zip(subject["gameId"], subject["personId"])), index=subject.index).map(poss).fillna(0.0)
        mask = mask.where(~covered, pp >= min_partial_possessions)
        source = source.where(~covered, "official_matchups")
    return mask.astype(bool), source


def opponent_on_court_mask(
    subject: pd.DataFrame, opponent_id: int, data: QueryData, min_shared_minutes: float
) -> tuple[pd.Series, pd.Series]:
    """Games where ``opponent_id`` shared the floor with the subject (or, without lineups, played)."""
    without_matchups = replace(data, matchups=None)
    return named_defender_mask(subject, opponent_id, without_matchups, 0.0, min_shared_minutes)


def same_game_mask(subject: pd.DataFrame, other_id: int, data: QueryData) -> pd.Series:
    """True where ``other_id`` played for the subject's opponent in that game."""
    pg = data.player_games
    theirs = set(zip(pg.loc[pg["personId"] == other_id, "gameId"], pg.loc[pg["personId"] == other_id, "teamId"]))
    pairs = pd.Series(list(zip(subject["gameId"], subject["opponentTeamId"])), index=subject.index)
    return pairs.isin(theirs)


def shared_floor_summary(games: pd.DataFrame, other_id: int, data: QueryData) -> dict[str, float] | None:
    """The subject's production *only while* ``other_id`` was on the floor, per 36 minutes.

    Box-score splits count the whole game, including minutes when the other
    player sat. Lineup reconstruction lets us count only the shared minutes.
    That is the closest honest proxy to "when X defended him" without tracking
    data, but it is still on-floor time, not a defensive assignment. The same
    rates over the subject's full minutes in the same games are returned for
    comparison. Needs ``on_court`` with stat columns. Returns ``None`` otherwise.
    """
    oc = data.on_court
    if oc is None or "pts" not in oc or games.empty:
        return None
    rows = oc[(oc["opponentPersonId"] == other_id) & oc["personId"].isin(games["personId"].unique())]
    rows = rows.merge(games[["gameId", "personId"]], on=["gameId", "personId"])
    covered = games[games["gameId"].isin(rows["gameId"])]
    mins = float(rows["shared_minutes"].sum())
    if mins <= 0:
        return None
    full_mins = float(covered["minutes"].astype("float64").sum())

    def rate(total: float, minutes: float) -> float:
        return total / minutes * 36.0 if minutes > 0 else float("nan")

    def ts(pts: float, fga: float, fta: float) -> float:
        att = 2.0 * (fga + 0.44 * fta)
        return pts / att * 100.0 if att > 0 else float("nan")

    return {
        "games": float(rows["gameId"].nunique()),
        "shared_minutes": mins,
        "shared_minutes_per_game": mins / rows["gameId"].nunique(),
        "points_per36_shared": rate(float(rows["pts"].sum()), mins),
        "assists_per36_shared": rate(float(rows["ast"].sum()), mins),
        "threes_per36_shared": rate(float(rows["fg3m"].sum()), mins),
        "ts_pct_shared": ts(float(rows["pts"].sum()), float(rows["fga"].sum()), float(rows["fta"].sum())),
        "points_per36_same_games": rate(float(covered["points"].astype("float64").sum()), full_mins),
        "assists_per36_same_games": rate(float(covered["assists"].astype("float64").sum()), full_mins),
        "threes_per36_same_games": rate(float(covered["threePointersMade"].astype("float64").sum()), full_mins),
        "ts_pct_same_games": ts(float(covered["points"].astype("float64").sum()),
                                float(covered["fieldGoalsAttempted"].astype("float64").sum()),
                                float(covered["freeThrowsAttempted"].astype("float64").sum())),
    }


def summarize_sources(source: pd.Series) -> dict[str, int]:
    """Game counts per source, best source first."""
    counts = source.value_counts()
    return {k: int(counts[k]) for k in sorted(counts.index, key=lambda s: SOURCE_RANK.get(s, 9))}


def _weighted_by_group(m: pd.DataFrame, column: str, keys: list[str]) -> pd.Series:
    """Weighted mean of ``column`` by weight ``w`` within each group, ignoring missing values."""
    ok = m[column].notna() & (m["w"] > 0)
    num = (m[column].astype("float64") * m["w"]).where(ok, 0.0).groupby([m[k] for k in keys]).sum()
    den = m["w"].where(ok, 0.0).groupby([m[k] for k in keys]).sum()
    return num / den.where(den > 0)

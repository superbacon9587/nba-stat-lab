"""End-to-end tests of the pipeline on the sample, plus checks on the real processed files."""

from __future__ import annotations

import pandas as pd
import pytest

from nbalab.data.aliases import normalize_alias
from nbalab.data.build import recover_team_ids
from tests.conftest import load_processed

SONICS_THUNDER = 1610612760

# Franchise names from relocations/renames since 1996, each belonging to exactly one teamId.
RELOCATED_NAMES = {
    "seattle supersonics": 1610612760,
    "oklahoma city thunder": 1610612760,
    "new jersey nets": 1610612751,
    "brooklyn nets": 1610612751,
    "vancouver grizzlies": 1610612763,
    "memphis grizzlies": 1610612763,
    "washington bullets": 1610612764,
    "washington wizards": 1610612764,
    "charlotte bobcats": 1610612766,
    "new orleans pelicans": 1610612740,
}


# ------------------------------------------------------------------ duplicates


def test_sample_has_no_duplicate_player_games(sample_tables: dict[str, pd.DataFrame]) -> None:
    assert not sample_tables["player_games"].duplicated(["personId", "gameId"]).any()


def test_sample_has_no_duplicate_team_games(sample_tables: dict[str, pd.DataFrame]) -> None:
    assert not sample_tables["team_games"].duplicated(["teamId", "gameId"]).any()


def test_processed_has_no_duplicate_player_games() -> None:
    pg = load_processed("player_games")
    assert not pg.duplicated(["personId", "gameId"]).any()


def test_processed_has_no_duplicate_team_games() -> None:
    tg = load_processed("team_games")
    assert not tg.duplicated(["teamId", "gameId"]).any()
    assert tg.groupby("gameId").size().eq(2).all()


# --------------------------------------------------------------------- filters


def test_sample_excludes_preseason_and_dnp(sample_tables: dict[str, pd.DataFrame]) -> None:
    pg = sample_tables["player_games"]
    assert set(pg["game_type"]) <= {"regular", "nba_cup", "play_in", "playoffs"}
    assert pg["minutes"].gt(0).all()


def test_processed_excludes_preseason_all_star_and_old_seasons() -> None:
    pg = load_processed("player_games")
    assert not pg["game_type"].isin(["preseason", "all_star"]).any()
    assert pg["season"].min() >= 1996
    assert pg["minutes"].gt(0).all()


# --------------------------------------------------------------------- seasons


def test_sample_season_assignment_at_boundaries(sample_tables: dict[str, pd.DataFrame]) -> None:
    tg = sample_tables["team_games"]
    month = tg["gameDateTimeEst"].dt.month
    year = tg["gameDateTimeEst"].dt.year
    assert tg.loc[(year == 2007) & (month >= 10), "season"].eq(2007).all()
    assert tg.loc[(year == 2008) & (month <= 4), "season"].eq(2007).all()
    assert tg.loc[(year == 2008) & (month >= 10), "season"].eq(2008).all()
    assert tg.loc[(year == 2009), "season"].eq(2008).all()
    assert set(tg["season_label"]) == {"2007-08", "2008-09"}


# ---------------------------------------------------------------- relocations


def test_sample_relocation_keeps_one_team_id(sample_tables: dict[str, pd.DataFrame]) -> None:
    tg = sample_tables["team_games"]
    home = tg[tg["teamId"].eq(SONICS_THUNDER) & tg["home"].eq(1)]
    assert set(home.loc[home["season"] == 2007, "venue_city"]) == {"Seattle"}
    assert set(home.loc[home["season"] == 2008, "venue_city"]) == {"Oklahoma City"}
    assert home["venue_team_id"].eq(SONICS_THUNDER).all()


def test_sample_teams_lookup_maps_both_names_to_one_row(sample_tables: dict[str, pd.DataFrame]) -> None:
    teams = sample_tables["teams"]
    row = teams[teams["teamId"] == SONICS_THUNDER].iloc[0]
    assert {"seattle supersonics", "sonics", "oklahoma city thunder", "okc"} <= set(row["aliases"])


def alias_owners(teams: pd.DataFrame) -> pd.Series:
    """Map each alias to the set of teamIds that list it."""
    exploded = teams[["teamId", "aliases"]].explode("aliases")
    return exploded.groupby("aliases")["teamId"].agg(lambda s: set(s))


@pytest.mark.parametrize(("name", "team_id"), sorted(RELOCATED_NAMES.items()))
def test_processed_franchise_names_map_to_one_team_id(name: str, team_id: int) -> None:
    owners = alias_owners(load_processed("teams"))
    assert owners[normalize_alias(name)] == {team_id}


def test_processed_team_games_use_only_thirty_franchise_ids() -> None:
    tg = load_processed("team_games")
    assert tg["teamId"].nunique() == 30
    assert tg["teamId"].between(1610612737, 1610612766).all()


def test_processed_seattle_and_okc_seasons_share_team_id() -> None:
    tg = load_processed("team_games")
    home = tg[tg["home"].eq(1)]
    sea = home.loc[home["venue_city"].eq("Seattle"), "teamId"].unique()
    okc = home.loc[home["venue_city"].eq("Oklahoma City") & home["season"].ge(2008), "teamId"].unique()
    assert set(sea) == set(okc) == {SONICS_THUNDER}


# -------------------------------------------------------------- id recovery


def test_recover_team_ids_from_game_home_away_names() -> None:
    ps = pd.DataFrame(
        {
            "gameId": [1, 1, 1],
            "playerteamName": ["SuperSonics", "Lakers", "SuperSonics"],
            "playerteamId": pd.array([pd.NA, pd.NA, 1610612760], dtype="Int64"),
            "opponentteamId": pd.array([pd.NA, pd.NA, 1610612747], dtype="Int64"),
        }
    )
    games = pd.DataFrame(
        {
            "gameId": [1],
            "hometeamId": [1610612760],
            "hometeamName": ["SuperSonics"],
            "awayteamId": [1610612747],
            "awayteamName": ["Lakers"],
        }
    )
    out = recover_team_ids(ps, games)
    assert out["playerteamId"].tolist() == [1610612760, 1610612747, 1610612760]
    assert out["opponentteamId"].tolist() == [1610612747, 1610612760, 1610612747]


# --------------------------------------------------------- context columns


def test_sample_opponent_ratings_come_from_earlier_games(sample_tables: dict[str, pd.DataFrame]) -> None:
    tg = sample_tables["team_games"].sort_values("gameDateTimeEst")
    for _, row in tg.sample(25, random_state=1).iterrows():
        opp_prior = tg[
            tg["teamId"].eq(row["opponentTeamId"])
            & tg["season"].eq(row["season"])
            & tg["gameDateTimeEst"].lt(row["gameDateTimeEst"])
            & tg["possessions"].gt(0)
        ]
        if opp_prior.empty:
            continue
        expected = 100 * opp_prior["opponentScore"].sum() / opp_prior["possessions"].sum()
        assert row["opp_pre_def_rating"] == pytest.approx(expected, rel=1e-5)
        assert row["opp_pre_games"] == len(opp_prior)


def test_sample_player_rows_carry_bio_and_context(sample_tables: dict[str, pd.DataFrame]) -> None:
    pg = sample_tables["player_games"]
    assert pg["position_group"].isin(["G", "F", "C"]).mean() > 0.95
    assert pg["age"].between(17, 45).mean() > 0.95
    assert pg["day_of_week"].notna().all()
    assert pg["team_game_num"].between(1, 110).all()


def test_sample_players_lookup_covers_every_player(sample_tables: dict[str, pd.DataFrame]) -> None:
    players = sample_tables["players"]
    assert set(sample_tables["player_games"]["personId"]) == set(players["personId"])
    assert players["aliases"].map(len).gt(0).all()

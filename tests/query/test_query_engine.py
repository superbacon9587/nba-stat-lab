"""Split engine on the hand-built fixture league (see conftest.py for the known answers)."""

from __future__ import annotations

import math

import pytest

from nbalab.query.data import QueryData
from nbalab.query.engine import confidence_label, run_split
from nbalab.query.schema import StatQuery
from query_fixture_league import BEES, CATS, DOGS, HOME_TEAM, SUBJECT, TEAMMATE


def split(data: QueryData, filters: list[dict], stats: list[str] | None = None, **kw):
    q = StatQuery(subject_ids=[kw.pop("subject", SUBJECT)], stats=stats or ["points"], filters=filters, **kw)
    return run_split(q, data)[0]


def test_vs_opponent_known_answers(data: QueryData) -> None:
    r = split(data, [{"type": "opponent_team", "team_ids": [BEES]}])
    s = r.stats["points"]
    assert (r.n_games, r.n_baseline) == (4, 12)
    assert s.split.mean == 20 and s.split.median == 20 and s.baseline.mean == 25
    assert s.split.std == pytest.approx(math.sqrt(200 / 3))
    assert s.split.per36 == pytest.approx(20 / 30 * 36)      # 80 pts / 120 min * 36 = 24
    assert s.baseline.per36 == pytest.approx(30.0)            # 300 / 360 * 36
    assert s.diff == -5 and s.diff_pct == pytest.approx(-20.0)
    assert s.z_score == pytest.approx(-1.354006, rel=1e-5)
    assert s.p_value == pytest.approx(0.16770, abs=1e-4)      # Welch, split vs the other 8 games
    assert s.ci_low < -5 < s.ci_high


def test_shrinkage_known_answer(data: QueryData) -> None:
    # tau^2 from per-opponent means (20, 25, 35; n 4, 6, 2) = 125/3; weight = 0.7534
    s = split(data, [{"type": "opponent_team", "team_ids": [BEES]}]).stats["points"]
    assert s.shrink_weight == pytest.approx(0.753425, rel=1e-5)
    assert s.shrunk == pytest.approx(21.232877, rel=1e-6)
    assert s.split.mean < s.shrunk < s.baseline.mean          # pulled toward baseline, raw kept


def test_sample_size_labels(data: QueryData) -> None:
    r = split(data, [{"type": "opponent_team", "team_ids": [BEES]}])
    assert r.confidence == "very low"
    assert any("Sample size warning" in w for w in r.warnings)
    assert confidence_label(10) == "low" and confidence_label(24) == "low" and confidence_label(25) == "high"
    r = split(data, [{"type": "opponent_team", "team_ids": [CATS, BEES]}])   # 10 games
    assert r.confidence == "low" and any("Low confidence" in w for w in r.warnings)


def test_stacked_filters_flag_and_suggest_least_important(data: QueryData) -> None:
    r = split(data, [{"type": "opponent_team", "team_ids": [BEES]}, {"type": "home_away", "value": "home"},
                     {"type": "back_to_back", "value": False}])
    # Bees games 1-4, home = odd games (1, 3); game 2 was the back-to-back
    assert r.n_games == 2 and r.stacked_filter_flag
    # dropping b2b (least important) leaves the same 2 games and home leaves 3, so the only
    # useful drop is the opponent (6 games: Bees + home + not b2b is games 1, 3; without Bees 1,3,5,7,9,11)
    assert "vs Beeville Bees" in r.suggestion and "leave 6 games" in r.suggestion
    impacts = {i.filter_type: i for i in r.filter_impacts}
    assert impacts["back_to_back"].n_if_dropped == 2
    assert impacts["home_away"].n_if_dropped == 3              # Bees + not b2b: games 1, 3, 4
    assert impacts["opponent_team"].n_alone == 4


def test_suggestion_skips_filters_whose_removal_does_not_help(data: QueryData) -> None:
    # Cats (6 games) + defender 20 (never faced Cats) + home: 0 games. Dropping "home"
    # (least important) still leaves 0; only dropping the defender recovers games.
    r = split(data, [{"type": "opponent_team", "team_ids": [CATS]}, {"type": "defender_player", "person_id": 20},
                     {"type": "home_away", "value": "home"}])
    assert r.n_games == 0 and r.stacked_filter_flag
    assert "defended by Player 20" in r.suggestion and "leave 3 games" in r.suggestion
    assert "no single filter" in r.suggestion   # 3 < 5, so it says so


def test_suggestion_prefers_least_important_filter_that_reaches_five(data: QueryData) -> None:
    # Cats + home + back-to-back: 0 games. Dropping b2b -> 3, home -> 0, Cats -> 1... none reach 5
    # Cats + home + weeks 1-2 (games 1-6): Cats home games in weeks 1-2 = game 5 only
    r = split(data, [{"type": "opponent_team", "team_ids": [CATS, BEES]}, {"type": "home_away", "value": "home"},
                     {"type": "week_of_season", "min_week": 1, "max_week": 1}])
    # games 1-3 are week 1; home & Bees in week 1 = games 1, 3 -> n = 2
    impacts = {i.filter_type: i.n_if_dropped for i in r.filter_impacts}
    assert r.n_games == 2 and impacts["week_of_season"] == 5 and impacts["home_away"] == 3
    assert "season weeks 1-1" in r.suggestion and "no single filter" not in r.suggestion


def test_single_filter_below_five_is_not_a_stacking_flag(data: QueryData) -> None:
    r = split(data, [{"type": "opponent_team", "team_ids": [DOGS]}])
    assert r.n_games == 2 and not r.stacked_filter_flag


def test_empty_split_still_returns(data: QueryData) -> None:
    r = split(data, [{"type": "opponent_team", "team_ids": [DOGS]}, {"type": "back_to_back", "value": True}])
    assert r.n_games == 0 and "No games match these filters." in r.warnings
    assert math.isnan(r.stats["points"].split.mean)


def test_scope_filters_define_the_baseline(data: QueryData) -> None:
    r = split(data, [{"type": "last_n_seasons", "n": 1}, {"type": "opponent_team", "team_ids": [CATS]}])
    # 2024 season = games 7-12; Cats games in it = 7-10 (30, 20, 30, 20)
    assert r.n_baseline == 6 and r.n_games == 4
    assert r.stats["points"].baseline.mean == pytest.approx((30 + 20 + 30 + 20 + 35 + 35) / 6)
    assert r.stats["points"].split.mean == 25
    r = split(data, [{"type": "season_range", "start": 2023, "end": 2023}])
    assert r.n_baseline == 6 and r.n_games == 6


def test_last_n_games_applies_after_other_filters(data: QueryData) -> None:
    r = split(data, [{"type": "opponent_team", "team_ids": [CATS]}, {"type": "last_n_games", "n": 2}])
    assert sorted(r.games["gameId"]) == [1009, 1010]


def test_teammate_in_and_out(data: QueryData) -> None:
    r_in = split(data, [{"type": "teammate", "person_id": TEAMMATE, "status": "in"}])
    r_out = split(data, [{"type": "teammate", "person_id": TEAMMATE, "status": "out"}])
    assert r_in.n_games == 8
    assert sorted(r_out.games["gameId"]) == [1009, 1010, 1011, 1012]


def test_calendar_and_context_filters(data: QueryData) -> None:
    assert split(data, [{"type": "home_away", "value": "home"}]).n_games == 6
    assert split(data, [{"type": "back_to_back", "value": True}]).n_games == 1
    assert split(data, [{"type": "rest_days", "min_days": 1, "max_days": 3}]).n_games == 11
    assert split(data, [{"type": "min_minutes", "minutes": 31}]).n_games == 0
    assert split(data, [{"type": "venue", "city": "Beeville"}]).n_games == 2   # Bees home games 2 and 4
    assert split(data, [{"type": "venue", "team_id": HOME_TEAM}]).n_games == 6
    assert split(data, [{"type": "playoffs", "value": True}]).n_games == 0
    wd = data.player_games.loc[data.player_games["personId"] == SUBJECT, "day_of_week"].iloc[0]
    assert split(data, [{"type": "day_of_week", "days": [wd]}]).n_games >= 1


def test_group_by_opponent(data: QueryData) -> None:
    r = split(data, [], group_by="opponent_team")
    t = r.groups.table.set_index("level")
    assert t.loc["Beeville Bees", "value"] == 20 and t.loc["Beeville Bees", "n"] == 4
    assert t.loc["Catcity Cats", "value"] == 25
    assert t.loc["Dogtown Dogs", "value"] == 35 and t.loc["Dogtown Dogs", "diff"] == 10
    assert t.loc["Beeville Bees", "shrunk"] == pytest.approx(21.232877, rel=1e-6)
    assert (t["p_value_holm"] >= t["p_value"]).all()


def test_team_subject(data: QueryData) -> None:
    q = StatQuery(subject_type="team", subject_ids=[HOME_TEAM], stats=["points"],
                  filters=[{"type": "opponent_team", "team_ids": [BEES]}])
    r = run_split(q, data)[0]
    # team score = subject + teammate 12 + 70 -> 92, 102, 112, 102
    assert r.stats["team_score"].split.mean == 102
    assert r.stats["team_score"].split.per36 is None


def test_ratio_stat_headline_is_pooled(data: QueryData) -> None:
    s = split(data, [{"type": "opponent_team", "team_ids": [BEES]}], stats=["FG%"]).stats["fg_pct"]
    assert s.is_ratio and s.split.per36 is None
    assert s.split.value == pytest.approx((80 / 2.5) / 80 * 100)   # FGM = pts/2.5 on 20 FGA per game


def test_projection_lines_give_hit_rates(data: QueryData) -> None:
    q = StatQuery(subject_ids=[SUBJECT], stats=["points"], mode="projection",
                  projection={"lines": {"points": 20.0}, "context": {"opponent_team_id": BEES}})
    r = run_split(q, data)[0]
    h = r.stats["points"].hit_rates["split"]
    assert (h["over"], h["under"], h["push"]) == (0.25, 0.25, 0.5)       # 10, 20, 30, 20
    assert r.notes


# ------------------------------------------------------------------ defenders


def test_named_defender_uses_best_source_per_game(data: QueryData) -> None:
    r = split(data, [{"type": "defender_player", "person_id": 20}])
    # game 1: shared 20 min (>= 10) -> in; game 2: 5 min -> out;
    # game 3: official matchups, 2 partial possessions (< 5) -> out; game 4: he did not play
    assert list(r.games["gameId"]) == [1001]
    assert r.defender_sources == {"shared_floor_time": 1}
    assert "not a true 'guarded by'" in r.defender_note
    assert r.shared_floor["shared_minutes"] == 20
    assert r.shared_floor["points_per36_shared"] == pytest.approx(8 / 20 * 36)


def test_named_defender_without_lineups_falls_back_to_same_game(box_only: QueryData) -> None:
    r = split(box_only, [{"type": "defender_player", "person_id": 20}])
    assert sorted(r.games["gameId"]) == [1001, 1002, 1003]
    assert r.defender_sources == {"same_game": 3}
    assert "WEAK PROXY" in r.defender_note


def test_defender_attribute_hierarchy(data: QueryData) -> None:
    r = split(data, [])
    g = r.baseline_games.set_index("gameId")
    assert g.loc[1003, "def_source"] == "official_matchups"
    assert g.loc[1003, "def_height"] == pytest.approx((78 * 2 + 81 * 8) / 10)   # possession-weighted
    assert g.loc[1003, "def_person_id"] == 21
    assert g.loc[1001, "def_source"] == "same_position_starter" and g.loc[1001, "def_person_id"] == 20
    assert g.loc[1004, "def_person_id"] == 22                  # 20 missed game 4
    assert g.loc[1005, "def_height"] == 72
    assert g.loc[1011, "def_source"] == "opponent_team_average"  # subject came off the bench
    assert g.loc[1011, "def_height"] == pytest.approx(77.5)


def test_defender_height_range(data: QueryData) -> None:
    r = split(data, [{"type": "defender_height_range", "min_inches": 77, "max_inches": 79}])
    assert sorted(r.games["gameId"]) == [1001, 1002, 1011, 1012]
    assert r.defender_sources == {"same_position_starter": 2, "opponent_team_average": 2}


def test_defender_position(data: QueryData) -> None:
    r = split(data, [{"type": "defender_position", "positions": ["F"]}])
    assert list(r.games["gameId"]) == [1003]                    # the official matchup's main defender


def test_opponent_on_court(data: QueryData) -> None:
    r = split(data, [{"type": "opponent_player_on_court", "person_id": 20, "min_shared_minutes": 1}])
    # games 1-2 covered by lineups (both >= 1 min), game 3 falls back to same game
    assert sorted(r.games["gameId"]) == [1001, 1002, 1003]

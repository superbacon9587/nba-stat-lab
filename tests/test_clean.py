"""Tests for game type, season, minutes, position and age cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nbalab.data.clean import (
    age_in_years,
    game_type_from_id,
    normalize_game_type_label,
    parse_minutes,
    position_group,
    resolve_game_type,
    season_from_date,
    season_from_game_id,
    season_label,
)


@pytest.mark.parametrize(
    ("date", "season"),
    [
        ("2024-10-22", 2024),  # opening night
        ("2024-12-31", 2024),
        ("2025-01-01", 2024),  # January belongs to the season that started in October
        ("2025-01-15", 2024),
        ("2025-06-20", 2024),  # Finals
        ("2025-09-30", 2024),  # offseason still maps to the finished season
        ("2025-10-01", 2025),
    ],
)
def test_season_from_date_october_january_boundary(date: str, season: int) -> None:
    assert season_from_date(pd.Series(pd.to_datetime([date]))).iloc[0] == season


@pytest.mark.parametrize(
    ("game_id", "season"),
    [
        (22400123, 2024),
        (12400001, 2024),  # preseason
        (42500405, 2025),  # 2026 Finals
        (20000289, 2000),  # "00" -> 2000
        (29900010, 1999),
        (24600001, 1946),
        (41900406, 2019),  # 2020 bubble Finals, played October 2020
    ],
)
def test_season_from_game_id(game_id: int, season: int) -> None:
    assert season_from_game_id(pd.Series([game_id])).iloc[0] == season


def test_game_id_season_fixes_bubble_where_date_rule_fails() -> None:
    bubble_finals = pd.Series(pd.to_datetime(["2020-10-11"]))
    assert season_from_date(bubble_finals).iloc[0] == 2020  # the date rule is wrong here
    assert season_from_game_id(pd.Series([41900406])).iloc[0] == 2019  # gameId is right


def test_season_label() -> None:
    assert season_label(2024) == "2024-25"
    assert season_label(1999) == "1999-00"
    assert season_label(pd.Series([2009, 2010])).tolist() == ["2009-10", "2010-11"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Regular Season", "regular"),
        ("Preseason", "preseason"),
        ("Pre Season", "preseason"),
        (" pre season ", "preseason"),
        ("NBA Emirates Cup", "nba_cup"),
        ("Emirates NBA Cup", "nba_cup"),
        ("NBA Cup", "nba_cup"),
        ("in-season-knockout", "nba_cup"),
        ("Play-in Tournament", "play_in"),
        ("All-Star Game", "all_star"),
        ("Playoffs", "playoffs"),
    ],
)
def test_normalize_game_type_label(raw: str, expected: str) -> None:
    assert normalize_game_type_label(pd.Series([raw])).iloc[0] == expected


def test_normalize_game_type_label_unknown_is_nan() -> None:
    assert pd.isna(normalize_game_type_label(pd.Series([None, "Exhibition"]))).all()


def test_game_type_from_id() -> None:
    ids = pd.Series([12400001, 22400001, 32400001, 42400001, 52400001, 62400001])
    assert game_type_from_id(ids).tolist() == [
        "preseason", "regular", "all_star", "playoffs", "play_in", "nba_cup",
    ]


def test_resolve_game_type_uses_label_only_to_spot_cup_games() -> None:
    ids = pd.Series([22500100, 22500101, 42500001, 12500001, 22500102])
    games_label = pd.Series(["NBA Emirates Cup", "Regular Season", None, "Pre Season", None])
    own_label = pd.Series([None, "Regular Season", None, None, "Emirates NBA Cup"])
    assert resolve_game_type(ids, games_label, own_label).tolist() == [
        "nba_cup", "regular", "playoffs", "preseason", "nba_cup",
    ]


def test_resolve_game_type_label_cannot_override_prefix() -> None:
    # A playoff id stays a playoff game even if a label says otherwise.
    assert resolve_game_type(pd.Series([42400001]), pd.Series(["Regular Season"])).iloc[0] == "playoffs"


def test_parse_minutes_decimal_and_clock_formats() -> None:
    raw = pd.Series(["24.0", "19.983333333333334", "22:30", "7:04", None, "0.0", "abc"])
    out = parse_minutes(raw)
    np.testing.assert_allclose(out.iloc[:4], [24.0, 19.983333333333334, 22.5, 7 + 4 / 60])
    assert pd.isna(out.iloc[4]) and out.iloc[5] == 0.0 and pd.isna(out.iloc[6])


def test_position_group_single_and_dual_flags() -> None:
    idx = pd.Index([1, 2, 3, 4, 5, 6, 7], name="personId")
    guard = pd.Series([1, 0, 0, 1, 0, 1, 0], index=idx)
    forward = pd.Series([0, 1, 0, 1, 1, 1, 0], index=idx)
    center = pd.Series([0, 0, 1, 0, 1, 0, 0], index=idx)
    mode = pd.Series({6: "G", 7: "C"})
    out = position_group(guard, forward, center, mode)
    # 4: G-F no starts -> F; 5: F-C -> C; 6: G-F mostly starts at G -> G; 7: no flags -> starts
    assert out.tolist() == ["G", "F", "C", "F", "C", "G", "C"]


def test_position_group_ignores_mode_that_contradicts_flags() -> None:
    idx = pd.Index([1])
    out = position_group(pd.Series([1], idx), pd.Series([0], idx), pd.Series([0], idx), pd.Series({1: "C"}))
    assert out.iloc[0] == "G"


def test_age_in_years_and_placeholder_birthdate() -> None:
    birth = pd.Series(["2000-01-01", "1900-01-01", None])
    on = pd.Series(pd.to_datetime(["2025-01-01"] * 3))
    ages = age_in_years(birth, on)
    assert ages.iloc[0] == pytest.approx(25.0, abs=0.01)
    assert ages.iloc[1:].isna().all()

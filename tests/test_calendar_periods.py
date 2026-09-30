"""Season calendar (All-Star break inference), period assignment, period statistics and BH."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nbalab.data.calendar import (
    all_star_game_dates, break_from_gap, build_season_calendar, infer_conferences, season_of_game_id,
)
from nbalab.query import inference as inf
from nbalab.query import periods as P
from nbalab.query.schema import PeriodSplit, StatQuery
from nbalab.query.stats import PLAYER_STATS, TEAM_STATS


def schedule(start: str, end: str, skip: tuple[str, str] | None = None) -> pd.Series:
    """Daily game dates from start to end, with an optional no-game stretch (inclusive)."""
    days = pd.Series(pd.date_range(start, end, freq="D"))
    if skip:
        days = days[~days.between(pd.Timestamp(skip[0]), pd.Timestamp(skip[1]))]
    return days.reset_index(drop=True)


def team_games_for(dates: pd.Series, season: int, game_type: str = "regular") -> pd.DataFrame:
    return pd.DataFrame({"season": season, "game_date": dates, "game_type": game_type,
                         "teamId": 1, "opponentTeamId": 2})


# ------------------------------------------------------------------ calendar


def test_gap_inference_finds_the_break() -> None:
    dates = schedule("2024-10-22", "2025-04-13", skip=("2025-02-14", "2025-02-19"))
    assert break_from_gap(dates) == (pd.Timestamp("2025-02-13"), pd.Timestamp("2025-02-20"))


def test_gap_inference_ignores_a_shutdown_longer_than_eight_days() -> None:
    # a 7-day break in February and a months-long stoppage in March (2019-20 shape)
    dates = pd.concat([schedule("2019-10-22", "2020-02-13"), schedule("2020-02-20", "2020-03-11"),
                       schedule("2020-07-30", "2020-08-14")], ignore_index=True)
    assert break_from_gap(dates) == (pd.Timestamp("2020-02-13"), pd.Timestamp("2020-02-20"))


def test_no_break_falls_back_to_midseason() -> None:
    tg = team_games_for(schedule("1999-02-05", "1999-05-05"), 1998)  # lockout: games every day
    cal = build_season_calendar(None, tg).iloc[0]
    assert cal["source"] == "midseason_fallback"
    assert pd.Timestamp("1999-03-15") <= cal["break_last_before"] <= pd.Timestamp("1999-03-30")


def test_all_star_game_in_data_wins_over_gap() -> None:
    games = pd.DataFrame({"gameId": [32500041], "gameDateTimeEst": [pd.Timestamp("2026-02-15 19:00")]})
    assert season_of_game_id(32500041) == 2025 and season_of_game_id(39700001) == 1997
    assert all_star_game_dates(games) == {2025: pd.Timestamp("2026-02-15")}
    tg = team_games_for(schedule("2025-10-21", "2026-04-12", skip=("2026-02-13", "2026-02-18")), 2025)
    cal = build_season_calendar(games, tg).iloc[0]
    assert cal["source"] == "all_star_game" and cal["all_star_date_known"]
    assert (cal["break_last_before"], cal["break_first_after"]) == (pd.Timestamp("2026-02-12"), pd.Timestamp("2026-02-19"))


def test_conferences_from_schedule() -> None:
    east, west = [1610612738, 11, 12, 13], [21, 22, 23, 24]
    rows = []
    for conf in (east, west):  # conference rivals meet 4 times, others once
        for a in conf:
            for b in conf:
                if a != b:
                    rows += [{"season": 2020, "teamId": a, "opponentTeamId": b, "game_type": "regular"}] * 4
    for a in east:
        for b in west:
            rows += [{"season": 2020, "teamId": a, "opponentTeamId": b, "game_type": "regular"},
                     {"season": 2020, "teamId": b, "opponentTeamId": a, "game_type": "regular"}]
    conf = infer_conferences(pd.DataFrame(rows)).set_index("teamId")["conference"]
    assert set(conf[east]) == {"East"} and set(conf[west]) == {"West"}


# ------------------------------------------------------------------ assigning periods


@pytest.fixture()
def season_games() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.concat([schedule("2024-10-22", "2025-02-13"), schedule("2025-02-20", "2025-04-13")], ignore_index=True)
    games = pd.DataFrame({"season": 2024, "game_date": dates, "game_type": "regular", "teamId": 1})
    games.loc[len(games)] = {"season": 2024, "game_date": pd.Timestamp("2025-04-20"), "game_type": "playoffs", "teamId": 1}
    cal = pd.DataFrame({"season": [2024], "break_last_before": [pd.Timestamp("2025-02-13")],
                        "break_first_after": [pd.Timestamp("2025-02-20")]})
    return games, cal


def test_all_star_split_excludes_playoffs_by_default(season_games) -> None:
    games, cal = season_games
    p = P.assign_period(games, cal, PeriodSplit(), "teamId")
    assert p.iloc[-1] is np.nan or pd.isna(p.iloc[-1])  # the playoff game is in neither period
    d = pd.to_datetime(games["game_date"])
    assert (p[d <= "2025-02-13"] == "before").all() and (p[(d >= "2025-02-20") & (games["game_type"] == "regular")] == "after").all()
    p2 = P.assign_period(games, cal, PeriodSplit(include_playoffs=True), "teamId")
    assert p2.iloc[-1] == "after"


def test_custom_date_month_groups_and_last_n(season_games) -> None:
    games, cal = season_games
    d = pd.to_datetime(games["game_date"])
    p = P.assign_period(games, cal, PeriodSplit(kind="custom_date", date="01-01"), "teamId")
    assert (p[d < "2025-01-01"] == "before").all() and (p[(d >= "2025-01-01") & (games["game_type"] == "regular")] == "after").all()
    p = P.assign_period(games, cal, PeriodSplit(kind="month_groups", before_months=[10, 11, 12, 1],
                                                after_months=[2, 3, 4, 5, 6]), "teamId")
    assert (p[d.dt.month.isin([10, 11, 12, 1])] == "before").all()
    assert pd.isna(p.iloc[-1])  # April playoff game: regular season only
    p = P.assign_period(games, cal, PeriodSplit(kind="last_n_before_playoffs", n_games=20), "teamId")
    regular = games["game_type"] == "regular"
    assert (p == "after").sum() == 20 and (p[(d >= "2025-03-25") & regular] == "after").all()


def test_custom_date_crosses_the_new_year() -> None:
    assert P.cutoff_date(2024, "02-20") == pd.Timestamp("2025-02-20")
    assert P.cutoff_date(2024, "12-01") == pd.Timestamp("2024-12-01")


def test_period_split_validation() -> None:
    with pytest.raises(ValueError):
        PeriodSplit(kind="month_groups", before_months=[1, 2], after_months=[2, 3])
    with pytest.raises(ValueError):
        PeriodSplit(kind="custom_date")
    q = StatQuery(subject_type="team", stats=["team_score"], mode="period")
    assert q.period_split == PeriodSplit()  # All-Star break by default
    with pytest.raises(ValueError):
        StatQuery(subject_ids=[1], stats=["points"], period_split=PeriodSplit())


# ------------------------------------------------------------------ statistics


def test_period_change_and_ci() -> None:
    before = pd.DataFrame({"points": [10.0, 12.0, 14.0, 12.0]})
    after = pd.DataFrame({"points": [20.0, 22.0, 24.0, 22.0]})
    c = P.change(P.period_value(PLAYER_STATS["points"], before), P.period_value(PLAYER_STATS["points"], after))
    assert c.diff == pytest.approx(10.0) and c.pct == pytest.approx(10 / 12 * 100)
    se = math.hypot(np.std([10, 12, 14, 12], ddof=1) / 2, np.std([20, 22, 24, 22], ddof=1) / 2)
    assert c.se == pytest.approx(se) and c.ci == pytest.approx((10 - 1.96 * se, 10 + 1.96 * se))
    assert c.p_value < 0.001 and c.seasons_improved == 1


def test_direction_lower_is_better() -> None:
    before = pd.DataFrame({"defensiveRating": [115.0, 117.0]})
    after = pd.DataFrame({"defensiveRating": [105.0, 107.0]})
    stat = TEAM_STATS["def_rating"]
    c = P.change(P.period_value(stat, before), P.period_value(stat, after), stat.higher_is_better)
    assert c.diff < 0 and c.seasons_improved == 1  # a lower defensive rating is an improvement


def test_ratio_stat_is_pooled() -> None:
    g = pd.DataFrame({"fieldGoalsMade": [1.0, 9.0], "fieldGoalsAttempted": [10.0, 10.0]})
    assert P.period_value(PLAYER_STATS["fg_pct"], g).value == pytest.approx(50.0)


def test_per36_uses_total_minutes() -> None:
    g = pd.DataFrame({"points": [10.0, 30.0], "minutes": [18.0, 36.0]})
    assert P.per36_value(PLAYER_STATS["points"], g).value == pytest.approx(40 / 54 * 36)


def test_averages_equal_and_precision_weighted() -> None:
    a = P.PeriodChange(10, 12, 2.0, 1.0, 50, 25, 1, 1)
    b = P.PeriodChange(10, 16, 6.0, 3.0, 50, 25, 1, 1)
    eq = P.average([a, b])
    assert eq.diff == pytest.approx(4.0) and eq.se == pytest.approx(math.sqrt(1 + 9) / 2) and eq.seasons_improved == 2
    pw = P.average([a, b], precision_weighted=True)
    assert pw.diff == pytest.approx((2 * 1 + 6 / 9) / (1 + 1 / 9))  # the precise season counts ~9x as much
    thin = P.PeriodChange(10, 30, 20.0, 8.0, 40, 3, 1, 1)
    assert P.average([a, thin], min_games=5).seasons == 1


def test_benjamini_hochberg_by_hand() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.20, np.nan])
    q = inf.benjamini_hochberg(p)
    # sorted p: .01 .03 .04 .20 (m=4) -> .04 .06 .0533 .20, then running minimum from the top
    assert q[:4] == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], rel=1e-4)
    assert np.isnan(q[4])
    assert (q[:4] >= p[:4]).all()

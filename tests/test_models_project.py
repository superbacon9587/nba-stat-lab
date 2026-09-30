"""End to end on the sample fixture: train a tiny bundle, save/load it, project a player and a team."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbalab.models import quantile as Q
from nbalab.models.registry import Bundle, load_bundle, save_bundle
from nbalab.models.train import Split, build_bundle
from nbalab.models.project import project
from nbalab.query.data import QueryData
from nbalab.query.schema import StatQuery

LAKERS = 1610612747
SONICS_THUNDER = 1610612760


@pytest.fixture(scope="module")
def tiny_bundle(sample_tables: dict[str, pd.DataFrame]) -> Bundle:
    rounds = Q.ROUNDS
    Q.ROUNDS = 5
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return build_bundle(sample_tables["player_games"], sample_tables["team_games"],
                                split=Split(2007, 2008, 2008),
                                player_stats=("points", "assists", "reboundsTotal"), team_stats=("teamScore",))
    finally:
        Q.ROUNDS = rounds


@pytest.fixture(scope="module")
def data(sample_tables: dict[str, pd.DataFrame]) -> QueryData:
    t = sample_tables
    return QueryData(t["player_games"], t["team_games"], t["players"], t["teams"])


@pytest.fixture(scope="module")
def busiest_player(sample_tables: dict[str, pd.DataFrame]) -> int:
    return int(sample_tables["player_games"].groupby("personId").size().idxmax())


def _query(pid: int, **ctx) -> StatQuery:
    return StatQuery(subject_ids=[pid], stats=["points", "pra"], mode="projection",
                     projection={"lines": {"points": 18.5, "assists": 3}, "context": ctx})


def test_player_projection_is_coherent(tiny_bundle: Bundle, data: QueryData, busiest_player: int) -> None:
    res = project(_query(busiest_player, opponent_team_id=LAKERS, home_away="home"), data, tiny_bundle)
    assert {"points", "assists", "pra"} <= set(res.stats)
    pts = res.stats["points"]
    assert pts.pmf.sum() == pytest.approx(1.0)
    assert pts.p_over + pts.p_under == pytest.approx(1.0) and pts.p_push is None
    ast = res.stats["assists"]
    assert ast.p_over + ast.p_under + ast.p_push == pytest.approx(1.0)
    for lvl in (90, 95, 99):
        lo, hi = pts.pi[lvl]
        clo, chi = pts.ci_mean[lvl]
        assert lo <= pts.mean <= hi and clo <= pts.mean <= chi
        assert hi - lo > chi - clo  # one game is less certain than the average
    assert pts.pi[90][1] - pts.pi[90][0] <= pts.pi[99][1] - pts.pi[99][0]
    total = sum(c.value for c in pts.contributions)
    assert total == pytest.approx(pts.mean - pts.neutral_mean, abs=1e-3)
    recent = data.player_games.query("personId == @busiest_player")["points"].tail(20).mean()
    assert 0.5 * recent < pts.mean < 1.5 * recent


def test_joint_probability_and_combo(tiny_bundle: Bundle, data: QueryData, busiest_player: int) -> None:
    res = project(_query(busiest_player), data, tiny_bundle)
    p_joint = res.joint_probability({"points": (">", 18.5), "assists": (">", 3)})
    p_pts = (res.draws["points"] > 18.5).mean()
    assert 0 <= p_joint <= p_pts
    pra = res.stats["pra"]
    parts = res.stats["points"].mean + res.stats["assists"].mean
    assert pra.mean > parts  # rebounds add
    assert np.allclose(res.draws["pra"], res.draws[["points", "rebounds", "assists"]].sum(axis=1))


def test_minutes_override_scales_projection(tiny_bundle: Bundle, data: QueryData, busiest_player: int) -> None:
    q = _query(busiest_player, home_away="away")
    low = project(q, data, tiny_bundle, projected_minutes=20).stats["points"]
    high = project(q, data, tiny_bundle, projected_minutes=40).stats["points"]
    assert low.minutes_mean == 20 and high.minutes_mean == 40
    assert high.mean == pytest.approx(2 * low.mean, rel=1e-3)


def test_team_projection(tiny_bundle: Bundle, data: QueryData) -> None:
    q = StatQuery(subject_type="team", subject_ids=[SONICS_THUNDER], stats=["team_score"], mode="projection",
                  projection={"lines": {"team_score": 99.5}})
    sp = project(q, data, tiny_bundle).stats["team_score"]
    assert 80 < sp.mean < 120
    assert sp.p_over + sp.p_under == pytest.approx(1.0)


def test_save_load_round_trip(tiny_bundle: Bundle, data: QueryData, busiest_player: int, tmp_path: Path) -> None:
    save_bundle(tiny_bundle, tmp_path)
    loaded = load_bundle(root=tmp_path)
    assert loaded.version == tiny_bundle.version
    assert (tmp_path / "LATEST").read_text().strip() == tiny_bundle.version
    q = _query(busiest_player, opponent_team_id=LAKERS)
    a, b = project(q, data, tiny_bundle).stats["points"], project(q, data, loaded).stats["points"]
    assert a.mean == pytest.approx(b.mean) and a.pi == b.pi

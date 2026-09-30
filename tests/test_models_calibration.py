"""Probability calibrators, the survival-curve mapping, the minutes conformal widening, and model archiving."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nbalab.models import metrics as M
from nbalab.models.calibration import (
    ProbabilityCalibrator, calibrated_over_under_push, choose_calibrator, conformal_scales, fit_isotonic, fit_platt,
    node_interval, spread_nodes,
)
from nbalab.models.project import StatProjection


def _overconfident(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw probabilities that are too extreme: the truth is a squashed version of them."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.02, 0.98, n)
    true = 0.5 + 0.6 * (raw - 0.5)
    return raw, (rng.uniform(size=n) < true).astype(float), np.repeat(np.arange(n // 20), 20)


def test_platt_and_isotonic_fix_overconfidence() -> None:
    raw, y, _ = _overconfident()
    for cal in (fit_platt(raw, y), fit_isotonic(raw, y)):
        assert M.brier(cal(raw), y) < M.brier(raw, y)
        assert cal(0.95) < 0.9 and cal(0.05) > 0.1


def test_calibrators_are_monotone_and_keep_the_ends() -> None:
    raw, y, _ = _overconfident()
    grid = np.linspace(0, 1, 201)
    for cal in (fit_platt(raw, y), fit_isotonic(raw, y)):
        out = cal(grid)
        assert (np.diff(out) >= -1e-12).all()
        assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(1.0)


def test_choose_picks_identity_when_already_calibrated() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    cal, scores = choose_calibrator(p, y, np.repeat(np.arange(1000), 20))
    assert set(scores) == {"identity", "platt", "isotonic"}
    assert scores[cal.kind] == min(scores.values())
    assert np.abs(cal(p) - p).mean() < 0.02  # whatever was chosen barely moves calibrated input


def test_choose_picks_a_real_fix_when_needed() -> None:
    raw, y, groups = _overconfident()
    cal, _ = choose_calibrator(raw, y, groups)
    assert cal.kind in ("platt", "isotonic")


def test_calibrator_round_trip() -> None:
    raw, y, _ = _overconfident()
    for cal in (fit_platt(raw, y), fit_isotonic(raw, y), ProbabilityCalibrator()):
        back = ProbabilityCalibrator.from_dict(cal.to_dict())
        np.testing.assert_allclose(back(raw[:50]), cal(raw[:50]))


def test_calibrated_survival_keeps_over_under_push_consistent() -> None:
    support = np.arange(30)
    pmf = np.exp(-0.5 * ((support - 12) / 4) ** 2)
    sp = StatProjection("x", 12, 12, {}, {}, support, pmf / pmf.sum(), calibrator=ProbabilityCalibrator("platt", 0.7, 0.1))
    for line in (8, 11.5, 12, 15.5, 20):
        o, u, p = sp.p_over_at(line), sp.p_under_at(line), sp.p_push_at(line)
        assert o + u + p == pytest.approx(1.0) and min(o, u, p) >= 0
        if not float(line).is_integer():
            assert p == 0.0
    assert sp.p_over_at(12.5) != pytest.approx(sp.p_over_at(12.5, calibrated=False))
    o, u, p = calibrated_over_under_push(ProbabilityCalibrator(), np.array([0.3]), np.array([0.4]))
    assert (o[0], u[0], p[0]) == pytest.approx((0.3, 0.6, 0.1))


def test_conformal_scales_reach_target_coverage() -> None:
    rng = np.random.default_rng(2)
    edges = (0, 20, 60)
    k = 25
    levels = (np.arange(k) + 0.5) / k
    too_narrow = np.vstack([np.quantile(rng.normal(0, 3, 5000), levels)] * 2)  # true noise sd is 5
    pred = rng.uniform(15, 35, 20_000)
    actual = pred + rng.normal(0, 5, pred.size)
    before = M.coverage(*node_interval(spread_nodes(pred, too_narrow, edges, np.ones(2), -99, 99), 0.9), actual)
    scales = conformal_scales(pred, actual, too_narrow, edges, -99, 99, 0.9)
    after = M.coverage(*node_interval(spread_nodes(pred, too_narrow, edges, scales, -99, 99), 0.9), actual)
    assert before < 0.8
    assert after == pytest.approx(0.9, abs=0.005)
    assert (scales > 1.4).all()


def test_spread_nodes_with_unit_scale_is_unchanged() -> None:
    res = np.array([[-5.0, -1.0, 0.0, 2.0, 6.0]])
    out = spread_nodes(np.array([30.0]), res, (0, 60), np.ones(1), 1, 53)
    np.testing.assert_allclose(out[0], 30 + res[0])


def test_archive_moves_only_unchosen_quantile_models(tmp_path) -> None:
    import json

    from nbalab.models.registry import archive_unchosen, model_files

    root = tmp_path / "models"
    v = root / "v1"
    v.mkdir(parents=True)
    for name in ("player_points_glm.json", "player_points_lgbm.txt", "player_points_lgbm.json",
                 "player_points_q_0.5.txt", "player_points_quantile.json"):
        (v / name).write_text("x")
    extras = {"player": {"chosen": {"points": "lgbm"}, "candidates": {"points": ["glm", "lgbm", "quantile"]}},
              "team": {"chosen": {}, "candidates": {}}}
    (v / "extras.json").write_text(json.dumps(extras))
    moved = archive_unchosen("v1", root=root, archive_root=tmp_path / "arch")
    assert sorted(f.name for f in moved) == ["player_points_q_0.5.txt", "player_points_quantile.json"]
    assert not model_files(v, "player", "points", "quantile")
    meta = json.loads((v / "extras.json").read_text())["player"]
    assert meta["candidates"]["points"] == ["glm", "lgbm"] and meta["archived"]["points"] == ["quantile"]


def test_raw_push_mode_keeps_model_push_and_sums_to_one() -> None:
    cal = ProbabilityCalibrator("platt", 0.6, 0.2, raw_push=True)
    s_line, s_below = np.array([0.40, 0.55]), np.array([0.52, 0.55])  # integer line, then a half-line
    o, u, p = calibrated_over_under_push(cal, s_line, s_below)
    assert p[0] == pytest.approx(0.12) and p[1] == pytest.approx(0.0)
    np.testing.assert_allclose(o + u + p, 1.0)
    plain_o, plain_u, _ = calibrated_over_under_push(ProbabilityCalibrator("platt", 0.6, 0.2), s_line, s_below)
    assert o[0] / u[0] == pytest.approx(plain_o[0] / plain_u[0])  # over:under ratio stays calibrated
    assert o[1] == pytest.approx(plain_o[1])  # half-lines untouched

"""Save and load a trained model bundle.

A bundle is one folder, ``models/<version>/``, holding everything a
projection needs: the minutes model, every candidate rate model per stat
(the chosen one is recorded in the manifest), the baselines, dispersion
values, the league correlation matrices, the defender-proxy prior, and a
``manifest.json`` describing how and on what data it was trained.
``models/LATEST`` names the current version. Comparison models that were
never chosen (the quantile models) live in ``models_archive/<version>/``,
which is gitignored: the backtest reads them, the app never does. Version tags look like
``v1.0.0-20260929``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

from nbalab.models.baseline import Baseline
from nbalab.models.config import ARCHIVE_DIR, MODEL_DIR
from nbalab.models.minutes import MinutesModel
from nbalab.models.quantile import QuantileModel
from nbalab.models.rate import GlmRate, LgbmRate

StatModel = Union[GlmRate, LgbmRate, QuantileModel]
LOADERS = {"glm": GlmRate.load, "lgbm": LgbmRate.load, "quantile": QuantileModel.load}


@dataclass
class SubjectModels:
    """Everything for one subject type (``player`` or ``team``)."""

    subject: str
    candidates: dict[str, dict[str, StatModel]]  # stat -> kind -> model
    chosen: dict[str, str]  # stat -> kind
    baseline: Baseline
    features: list[str]
    corr_stats: list[str]
    league_corr: np.ndarray

    def model(self, stat: str, kind: str | None = None) -> StatModel:
        return self.candidates[stat][kind or self.chosen[stat]]

    @property
    def stats(self) -> list[str]:
        return list(self.chosen)


@dataclass
class Bundle:
    version: str
    manifest: dict
    minutes: MinutesModel
    player: SubjectModels
    team: SubjectModels
    defender_tau2: dict[str, float] = field(default_factory=dict)
    neutral: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)  # see nbalab.models.calibration
    path: Path | None = None

    def subject(self, kind: str) -> SubjectModels:
        return self.player if kind == "player" else self.team


def version_tag(semver: str, date: str) -> str:
    return f"v{semver}-{date.replace('-', '')}"


MODEL_FILE_PATTERNS: dict[str, tuple[str, ...]] = {
    "glm": ("{subject}_{stat}_glm.json",),
    "lgbm": ("{subject}_{stat}_lgbm.txt", "{subject}_{stat}_lgbm.json"),
    "quantile": ("{subject}_{stat}_q_*.txt", "{subject}_{stat}_quantile.json"),
}


def archive_dir(root: Path, archive_root: Path | None = None) -> Path:
    """``models_archive/`` for the real ``models/``; a sibling ``<root>_archive`` for any other root."""
    if archive_root is not None:
        return archive_root
    return ARCHIVE_DIR if root.resolve() == MODEL_DIR.resolve() else root.parent / f"{root.name}_archive"


def model_files(path: Path, subject: str, stat: str, kind: str) -> list[Path]:
    """Every file one saved candidate model consists of."""
    return [f for pat in MODEL_FILE_PATTERNS[kind] for f in path.glob(pat.format(subject=subject, stat=stat))]


def _save_subject(sm: SubjectModels, path: Path, archive: Path, archive_kinds: tuple[str, ...]) -> dict:
    """Save the subject's models. Unchosen candidates of ``archive_kinds`` go to ``archive``, not the bundle."""
    kept: dict[str, list[str]] = {}
    archived: dict[str, list[str]] = {}
    for stat, kinds in sm.candidates.items():
        for kind, model in kinds.items():
            to_archive = kind in archive_kinds and kind != sm.chosen[stat]
            model.save(archive if to_archive else path)
            (archived if to_archive else kept).setdefault(stat, []).append(kind)
    sm.baseline.save(path, sm.subject)
    return {
        "chosen": sm.chosen, "candidates": kept, "archived": archived,
        "features": sm.features, "corr_stats": sm.corr_stats, "league_corr": sm.league_corr.tolist(),
    }


def save_bundle(bundle: Bundle, root: Path = MODEL_DIR, make_latest: bool = True,
                archive_root: Path | None = None, archive_kinds: tuple[str, ...] = ("quantile",)) -> Path:
    """Write ``root/<version>/`` and point ``root/LATEST`` at it.

    Unchosen candidates of ``archive_kinds`` (kept only for the backtest
    report) are written to ``archive_root/<version>/`` instead, which is not
    shipped with the app.
    """
    path = root / bundle.version
    archive = archive_dir(root, archive_root) / bundle.version
    path.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    bundle.minutes.save(path)
    extras = {
        "player": _save_subject(bundle.player, path, archive, archive_kinds),
        "team": _save_subject(bundle.team, path, archive, archive_kinds),
        "defender_tau2": bundle.defender_tau2, "neutral": bundle.neutral,
    }
    (path / "extras.json").write_text(json.dumps(extras, indent=1))
    (path / "manifest.json").write_text(json.dumps(bundle.manifest, indent=2, default=str))
    save_calibration(bundle, path)
    if make_latest:
        (root / "LATEST").write_text(bundle.version + "\n")
    bundle.path = path
    return path


def save_calibration(bundle: Bundle, path: Path | None = None) -> None:
    """Write (or remove) ``calibration.json`` for the bundle."""
    target = (path or bundle.path) / "calibration.json"
    if bundle.calibration:
        target.write_text(json.dumps(bundle.calibration, indent=1))
    elif target.exists():
        target.unlink()


def archive_unchosen(version: str, kinds: tuple[str, ...] = ("quantile",), root: Path = MODEL_DIR,
                     archive_root: Path | None = None) -> list[Path]:
    """Move unchosen candidates of ``kinds`` out of an existing bundle into the archive."""
    path, archive = root / version, archive_dir(root, archive_root) / version
    archive.mkdir(parents=True, exist_ok=True)
    extras = json.loads((path / "extras.json").read_text())
    moved: list[Path] = []
    for subject in ("player", "team"):
        meta = extras[subject]
        meta.setdefault("archived", {})
        for stat, stat_kinds in meta["candidates"].items():
            for kind in [k for k in stat_kinds if k in kinds and k != meta["chosen"][stat]]:
                for f in model_files(path, subject, stat, kind):
                    moved.append(Path(shutil.move(str(f), archive / f.name)))
                stat_kinds.remove(kind)
                meta["archived"].setdefault(stat, []).append(kind)
    (path / "extras.json").write_text(json.dumps(extras, indent=1))
    return moved


def _load_subject(subject: str, meta: dict, path: Path, archive: Path | None) -> SubjectModels:
    candidates = {s: {k: LOADERS[k](path, subject, s) for k in kinds} for s, kinds in meta["candidates"].items()}
    if archive is not None and archive.exists():
        for s, kinds in meta.get("archived", {}).items():
            for k in kinds:
                candidates[s][k] = LOADERS[k](archive, subject, s)
    for s, kinds in candidates.items():  # alpha lives in each model file; quantile has none
        for m in kinds.values():
            m.stat = s
    return SubjectModels(subject, candidates, meta["chosen"], Baseline.load(path, subject), meta["features"],
                         meta["corr_stats"], np.asarray(meta["league_corr"]))


def load_bundle(version: str | None = None, root: Path = MODEL_DIR, include_archived: bool = False,
                archive_root: Path | None = None) -> Bundle:
    """Load ``root/<version>``, or the one named in ``root/LATEST``.

    ``include_archived`` also loads the archived comparison models (for the backtest).
    """
    if version is None:
        latest = root / "LATEST"
        if not latest.exists():
            raise FileNotFoundError(f"no trained models in {root}; run `python -m nbalab.models.train`")
        version = latest.read_text().strip()
    path = root / version
    archive = archive_dir(root, archive_root) / version if include_archived else None
    extras = json.loads((path / "extras.json").read_text())
    cal = path / "calibration.json"
    return Bundle(
        version=version,
        manifest=json.loads((path / "manifest.json").read_text()),
        minutes=MinutesModel.load(path),
        player=_load_subject("player", extras["player"], path, archive),
        team=_load_subject("team", extras["team"], path, archive),
        defender_tau2=extras["defender_tau2"],
        neutral=extras["neutral"],
        calibration=json.loads(cal.read_text()) if cal.exists() else {},
        path=path,
    )

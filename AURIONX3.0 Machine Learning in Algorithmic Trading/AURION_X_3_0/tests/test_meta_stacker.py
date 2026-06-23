"""AURION X 4.0 — Tests para MetaStacker."""

from __future__ import annotations

import numpy as np
import pytest

from ml.meta_stacker import MetaStacker


def _make_dataset(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dataset sintético: label depende sobre todo de prob_ensemble y patchtst."""
    rng = np.random.default_rng(seed)
    prob_ens = rng.uniform(0.2, 0.9, size=n)
    prob_pst = rng.uniform(0.2, 0.9, size=n)
    eff_conf = (prob_ens + prob_pst) / 2.0 + rng.normal(0, 0.05, size=n)
    grade = rng.uniform(0.4, 1.0, size=n)
    score = 0.55 * prob_ens + 0.35 * prob_pst + 0.05 * eff_conf + 0.05 * grade
    y = (score > 0.55).astype(np.int64)
    X = np.column_stack([prob_ens, prob_pst, eff_conf, grade])
    return X, y


def test_fit_and_predict(tmp_path) -> None:
    X, y = _make_dataset(400)
    ms = MetaStacker(cfg={
        "enabled": True,
        "model_path": tmp_path / "ms.pkl",
        "min_samples_to_fit": 100,
    })
    ms.fit(X, y)
    assert ms.is_fitted
    pred = ms.predict(
        prob_ensemble=0.75,
        prob_patchtst=0.70,
        effective_confidence=0.72,
        setup_grade="A",
    )
    assert 0.0 <= pred.prob_buy <= 1.0
    assert pytest.approx(pred.prob_buy + pred.prob_sell, abs=1e-9) == 1.0
    assert pred.used_patchtst is True


def test_fit_too_few_samples_raises(tmp_path) -> None:
    X, y = _make_dataset(50)
    ms = MetaStacker(cfg={
        "enabled": True,
        "model_path": tmp_path / "ms.pkl",
        "min_samples_to_fit": 200,
    })
    with pytest.raises(ValueError):
        ms.fit(X, y)


def test_nan_patchtst_imputed(tmp_path) -> None:
    X, y = _make_dataset(400)
    ms = MetaStacker(cfg={
        "enabled": True,
        "model_path": tmp_path / "ms.pkl",
        "min_samples_to_fit": 100,
    })
    ms.fit(X, y)
    pred = ms.predict(
        prob_ensemble=0.65,
        prob_patchtst=None,  # PatchTST no disponible
        effective_confidence=0.62,
        setup_grade="B",
    )
    assert 0.0 <= pred.prob_buy <= 1.0
    assert pred.used_patchtst is False


def test_fallback_when_not_fitted() -> None:
    ms = MetaStacker(cfg={
        "enabled": True,
        "model_path": "/no/existe.pkl",
        "fallback_to_ensemble": True,
    })
    pred = ms.predict(
        prob_ensemble=0.71,
        prob_patchtst=0.55,
        effective_confidence=0.63,
        setup_grade="A",
    )
    assert pred.prob_buy == pytest.approx(0.71)
    assert pred.used_patchtst is False  # no se usó porque no hay stacker


def test_save_and_reload(tmp_path) -> None:
    X, y = _make_dataset(400)
    path = tmp_path / "ms.pkl"
    ms = MetaStacker(cfg={"enabled": True, "model_path": path, "min_samples_to_fit": 100})
    ms.fit(X, y)
    ms.save()
    assert path.is_file()

    ms2 = MetaStacker(cfg={"enabled": True, "model_path": path})
    assert ms2.is_fitted
    pred1 = ms.predict(
        prob_ensemble=0.8, prob_patchtst=0.6, effective_confidence=0.7, setup_grade="A",
    )
    pred2 = ms2.predict(
        prob_ensemble=0.8, prob_patchtst=0.6, effective_confidence=0.7, setup_grade="A",
    )
    assert pred1.prob_buy == pytest.approx(pred2.prob_buy, abs=1e-9)

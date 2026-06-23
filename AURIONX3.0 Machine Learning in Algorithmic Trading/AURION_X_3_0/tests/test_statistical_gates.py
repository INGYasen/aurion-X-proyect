"""AURION X 4.0 — Tests para DSR y PBO (statistical_gates.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.statistical_gates import (
    deflated_sharpe_ratio,
    probability_backtest_overfitting,
    passes_statistical_gates,
)


# ─────────────────────────────────────────────────────────────────────────────
# DSR
# ─────────────────────────────────────────────────────────────────────────────


def _winning_series(n: int = 1000, seed: int = 42) -> np.ndarray:
    """Estrategia ganadora real: edge positivo claro + ruido moderado."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0015, scale=0.005, size=n)


def _random_series(n: int = 1000, seed: int = 123) -> np.ndarray:
    """Estrategia sin edge: media ≈ 0."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.005, size=n)


def test_dsr_winning_strategy_high_dsr() -> None:
    returns = _winning_series(2000)
    result = deflated_sharpe_ratio(returns, n_trials=10, bars_per_year=525_600)
    # Con un edge claro y pocos ensayos, DSR debe ser muy alto.
    assert result["dsr"] > 0.85, f"DSR={result['dsr']:.3f} esperado > 0.85"
    assert result["sr_annualized"] > 0.0
    assert result["n_obs"] == 2000.0


def test_dsr_random_strategy_low_dsr() -> None:
    returns = _random_series(2000)
    result = deflated_sharpe_ratio(returns, n_trials=100, bars_per_year=525_600)
    # Sin edge real y con muchos ensayos, DSR debe ser bajo.
    assert result["dsr"] < 0.50, f"DSR={result['dsr']:.3f} esperado < 0.50"


def test_dsr_penalizes_more_trials() -> None:
    returns = _winning_series(2000)
    low_trials = deflated_sharpe_ratio(returns, n_trials=1)
    high_trials = deflated_sharpe_ratio(returns, n_trials=1000)
    # A más ensayos, mayor expected_max_sr → menor DSR.
    assert high_trials["dsr"] <= low_trials["dsr"] + 1e-9, (
        f"DSR debería decrecer con más ensayos: {low_trials['dsr']:.3f} → {high_trials['dsr']:.3f}"
    )
    assert high_trials["expected_max_sr"] > low_trials["expected_max_sr"]


def test_dsr_insufficient_samples_returns_zero() -> None:
    returns = np.array([0.001, 0.002, -0.001])
    result = deflated_sharpe_ratio(returns, n_trials=10)
    assert result["dsr"] == 0.0
    assert result["n_obs"] == 3.0


# ─────────────────────────────────────────────────────────────────────────────
# PBO
# ─────────────────────────────────────────────────────────────────────────────


def test_pbo_independent_strategies_around_half() -> None:
    """Estrategias INDEPENDIENTES (sin señal común) → PBO esperado ≈ 0.5.

    Cuando N estrategias son retornos IID, su mejor IS debería estar
    distribuida uniformemente en OS (sin overfit estructural). El estimador
    de PBO con N=8 y s=8 sólo dispone de 70 combinaciones, por lo que es
    muy ruidoso para un único seed (std ~0.23). Para reducir varianza
    promediamos sobre 20 réplicas independientes; la media debe estar
    cerca de 0.5.
    """
    pbos: list[float] = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        R = rng.normal(loc=0.0005, scale=0.01, size=(2400, 8))
        info = probability_backtest_overfitting(R, s=8)
        pbos.append(info["pbo"])
    mean_pbo = float(np.mean(pbos))
    assert 0.35 <= mean_pbo <= 0.65, (
        f"PBO medio={mean_pbo:.3f} fuera de banda esperada [0.35, 0.65] "
        f"(pbos={[round(p,2) for p in pbos]})"
    )


def test_pbo_correlated_strategies_high_pbo() -> None:
    """Estrategias altamente correlacionadas con micro-ruido → PBO ALTO.

    Cuando N estrategias comparten señal (corr ~0.99), seleccionar la mejor
    IS por ruido implica que su ruido OS es opuesto (las particiones son
    complementarias). Es el caso clásico de overfitting de hiperparámetros
    sobre la misma idea base → CSCV lo detecta correctamente con PBO > 0.7.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(loc=0.0005, scale=0.01, size=2400)
    R = np.column_stack([base + rng.normal(0, 1e-3, size=2400) for _ in range(8)])
    info = probability_backtest_overfitting(R, s=8)
    assert info["pbo"] >= 0.6, f"PBO={info['pbo']:.3f} debería ser > 0.6 para estrategias correladas"


def test_pbo_dominant_strategy_low_pbo() -> None:
    """Una estrategia claramente superior debe dar PBO bajo."""
    rng = np.random.default_rng(13)
    T = 2400
    weak1 = rng.normal(loc=0.0, scale=0.01, size=T)
    weak2 = rng.normal(loc=-0.0001, scale=0.01, size=T)
    weak3 = rng.normal(loc=0.0, scale=0.012, size=T)
    weak4 = rng.normal(loc=-0.0002, scale=0.011, size=T)
    strong = rng.normal(loc=0.003, scale=0.008, size=T)
    R = np.column_stack([weak1, weak2, weak3, weak4, strong])
    info = probability_backtest_overfitting(R, s=8)
    assert info["pbo"] < 0.25, f"PBO={info['pbo']:.3f} esperado < 0.25"


def test_pbo_too_few_samples_returns_one() -> None:
    R = np.random.RandomState(0).randn(20, 3)
    info = probability_backtest_overfitting(R, s=8)
    assert info["pbo"] == 1.0
    assert info["n_combinations"] == 0.0


def test_pbo_single_strategy_returns_one() -> None:
    R = np.random.RandomState(0).randn(2000, 1)
    info = probability_backtest_overfitting(R, s=8)
    assert info["pbo"] == 1.0


def test_pbo_invalid_s_raises() -> None:
    R = np.random.RandomState(0).randn(2000, 3)
    with pytest.raises(ValueError):
        probability_backtest_overfitting(R, s=7)  # impar


# ─────────────────────────────────────────────────────────────────────────────
# passes_statistical_gates (combinado)
# ─────────────────────────────────────────────────────────────────────────────


def test_passes_gates_winning_pass() -> None:
    returns = _winning_series(2000)
    ok, reason, metrics = passes_statistical_gates(
        returns=returns,
        n_trials=10,
        returns_matrix=None,
        min_dsr=0.60,
        max_pbo=0.30,
    )
    assert ok is True, reason
    assert reason == "ok"
    assert metrics["dsr_dsr"] >= 0.60


def test_passes_gates_random_fails_on_dsr() -> None:
    returns = _random_series(2000)
    ok, reason, _ = passes_statistical_gates(
        returns=returns,
        n_trials=100,
        returns_matrix=None,
        min_dsr=0.60,
        max_pbo=0.30,
    )
    assert ok is False
    assert reason.startswith("dsr_below_threshold")


def test_passes_gates_with_pbo_matrix_when_overfit() -> None:
    """Si la matriz refleja overfitting, debe vetar."""
    rng = np.random.default_rng(99)
    T = 2400
    returns = _winning_series(T, seed=7)
    # Matriz con muchas estrategias casi-idénticas y ruido (caso típico overfit)
    R = np.column_stack([
        returns + rng.normal(0, 0.0001, size=T) for _ in range(10)
    ])
    ok, reason, metrics = passes_statistical_gates(
        returns=returns,
        n_trials=200,
        returns_matrix=R,
        min_dsr=0.30,   # bajamos DSR para que solo PBO sea decisivo
        max_pbo=0.10,
    )
    # Con muchas casi-idénticas el rank OS es esencialmente aleatorio → PBO ≈ 0.5
    assert ok is False
    assert "pbo" in reason
    assert metrics["pbo_pbo"] >= 0.10

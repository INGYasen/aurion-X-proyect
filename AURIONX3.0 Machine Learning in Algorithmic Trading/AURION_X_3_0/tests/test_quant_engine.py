"""Tests del motor cuantitativo."""

from __future__ import annotations

from ml.confidence_engine import MLConfidenceEngine
from ml.ensemble_model import EnsembleTradingModel
from ml.probability_engine import ProbabilityEngine
from ml.quant_features import FEATURE_VECTOR_LEN_V2, build_quant_features


def test_build_quant_features_len() -> None:
    ms = {
        "closes_m1": [1.0, 1.01, 1.02, 1.03, 1.04],
        "highs_m1": [1.01, 1.02, 1.03, 1.04, 1.05],
        "lows_m1": [0.99, 1.0, 1.01, 1.02, 1.03],
        "tick_volume_proxy": 120.0,
        "direction": 1,
        "drawdown": -0.01,
    }
    vec, d = build_quant_features(ms, context={"direction": "BUY", "spread_points": 12.0})
    assert len(vec) == FEATURE_VECTOR_LEN_V2
    assert d.get("feature_schema_version") == 2


def test_ensemble_fit_predict() -> None:
    import random

    random.seed(42)
    n = 80
    X = [[random.uniform(-1, 1) for _ in range(FEATURE_VECTOR_LEN_V2)] for _ in range(n)]
    y = [1 if sum(row) > 0 else 0 for row in X]
    m = EnsembleTradingModel()
    m.fit(X, y, feature_schema_version=2)
    p = m.predict_proba([X[0]])
    assert len(p[0]) == 2


def test_probability_and_confidence() -> None:
    pe = ProbabilityEngine()
    snap = pe.compute(
        direction="BUY",
        signal={"setup_type": "RANGE_BREAKOUT_MOMENTUM", "momentum_score": 0.6},
        regime="HIGH",
        mtf_trends={"M1": "UP", "M5": "UP", "M15": "RANGE"},
        structure_trend="UP",
        continuation_strength=0.7,
        spread_points=20.0,
        spread_cap=200.0,
    )
    assert 0.0 <= snap.continuation <= 1.0
    ce = MLConfidenceEngine()
    br = ce.compute(
        direction="BUY",
        prob_buy=0.62,
        prob_sell=0.38,
        ml_confidence=0.62,
        rule_score=55,
        probabilities=snap,
        signal={"setup_type": "RANGE_BREAKOUT_MOMENTUM"},
        spread_points=20.0,
        spread_cap=200.0,
    )
    assert br.total > 0.0

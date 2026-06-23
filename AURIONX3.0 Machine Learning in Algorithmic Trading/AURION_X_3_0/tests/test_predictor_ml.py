from __future__ import annotations

import numpy as np

from ml.model import TradingModel
from ml.predictor import Predictor


def test_predictor_wrong_feature_length_returns_neutral() -> None:
    model = TradingModel()
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5)).tolist()
    y = rng.integers(0, 2, size=30).tolist()
    model.fit(X, y)
    pred = Predictor(model)
    assert pred.expected_n_features() == 5
    out = pred.predict([0.1, 0.2])  # dimensión incorrecta
    assert out["prob_buy"] == out["prob_sell"] == 0.5
    assert out["confidence"] == 0.5


def test_predictor_correct_length() -> None:
    model = TradingModel()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 4)).tolist()
    y = rng.integers(0, 2, size=40).tolist()
    model.fit(X, y)
    pred = Predictor(model)
    out = pred.predict([0.1, 0.2, 0.3, 0.4])
    assert 0.0 <= out["prob_buy"] <= 1.0
    assert 0.0 <= out["prob_sell"] <= 1.0
    assert out["confidence"] == max(out["prob_buy"], out["prob_sell"])

from __future__ import annotations

import numpy as np

from ml.feature_engineering import FEATURE_ORDER, FEATURE_SCHEMA_VERSION, FEATURE_VECTOR_LEN
from ml.trainer import Trainer


def test_trainer_calibration_prefit_holdout() -> None:
    tr = Trainer(
        training_weight_by_r=False,
        calibration_method="sigmoid",
        calibration_holdout_ratio=0.2,
    )
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, FEATURE_VECTOR_LEN)).tolist()
    y = [i % 2 for i in range(80)]
    model = tr.train(X[:60], [float(v) for v in y[:60]])
    tr._calibrate_if_needed(model, X[60:], y[60:])
    assert model.metadata().get("calibrated") is True
    proba = model.estimator.predict_proba(np.asarray(X[:3], dtype=np.float64))
    assert proba.shape == (3, 2)


def test_trainer_train_attaches_feature_metadata() -> None:
    tr = Trainer(training_weight_by_r=False, walk_forward_embargo_ratio=0.02)
    X = [[float(i % 3) for _ in range(FEATURE_VECTOR_LEN)] for i in range(50)]
    y = [i % 2 for i in range(50)]
    m = tr.train(X, [float(v) for v in y])
    meta = m.metadata()
    assert meta.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
    assert meta.get("feature_names") == list(FEATURE_ORDER)
    assert meta.get("n_features_in") == FEATURE_VECTOR_LEN

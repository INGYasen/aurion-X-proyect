"""AURION X 4.0 — Tests para AnomalyAutoencoder y AnomalyDetector."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.anomaly_autoencoder import (
    AnomalyDetector,
    ANOMALY_FEATURE_NAMES,
    build_anomaly_input,
    is_available,
)


pytestmark = pytest.mark.skipif(not is_available(), reason="torch no instalado")


def _train_simple(model, X, epochs=80):
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    X_t = torch.from_numpy(X).float()
    for _ in range(epochs):
        model.train()
        xh = model(X_t, add_noise=True)
        loss = loss_fn(xh, X_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def test_input_shape() -> None:
    micro = {"intention_loss": 0.1, "absorption_pressure": 0.5,
             "directional_exhaustion": 0.3, "institutional_velocity": 0.7}
    ms = {"spread_points": 10, "spread_cap": 100, "volumes_m1": [100.0, 110.0, 105.0]}
    vec = build_anomaly_input(micro, ms)
    assert vec.shape == (len(ANOMALY_FEATURE_NAMES),)
    assert np.isfinite(vec).all()


def test_normal_reconstructs_well() -> None:
    """Tras entrenar con datos 'normales', un sample normal tiene error bajo."""
    import torch
    from ml.anomaly_autoencoder import AnomalyAutoencoder

    rng = np.random.default_rng(0)
    X_normal = rng.normal(0, 1, size=(2000, 8)).astype(np.float32)
    model = AnomalyAutoencoder(input_dim=8, encoder_dims=(32, 16, 8), dropout=0.0)
    _train_simple(model, X_normal, epochs=80)

    model.eval()
    with torch.no_grad():
        sample_normal = torch.from_numpy(rng.normal(0, 1, size=(1, 8)).astype(np.float32))
        sample_anomaly = torch.from_numpy(np.full((1, 8), 50.0, dtype=np.float32))
        e_normal = float(model.reconstruction_error(sample_normal).item())
        e_anomaly = float(model.reconstruction_error(sample_anomaly).item())

    assert e_normal < e_anomaly, f"Anomalía debe tener mayor error: {e_normal:.3f} vs {e_anomaly:.3f}"


def test_detector_loads_threshold(tmp_path) -> None:
    """Detector lee p99 desde JSON y normaliza correctamente."""
    import torch
    from ml.anomaly_autoencoder import AnomalyAutoencoder

    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(1000, 8)).astype(np.float32)
    model = AnomalyAutoencoder(input_dim=8, encoder_dims=(32, 16, 8), dropout=0.0)
    _train_simple(model, X, epochs=60)
    model_path = tmp_path / "ae.pt"
    threshold_path = tmp_path / "ae_thr.json"
    torch.save(model.state_dict(), model_path)
    with torch.no_grad():
        errs = model.reconstruction_error(torch.from_numpy(X)).cpu().numpy()
    p99 = float(np.percentile(errs, 99))
    threshold_path.write_text(json.dumps({"p99_error": p99}))

    detector = AnomalyDetector(cfg={
        "enabled": True,
        "model_path": str(model_path),
        "threshold_path": str(threshold_path),
        "block_threshold": 0.80,
        "input_features": 8,
    })

    micro_ok = {"intention_loss": 0.0, "absorption_pressure": 0.0,
                "directional_exhaustion": 0.0, "institutional_velocity": 0.0}
    ms_ok = {"spread_points": 5, "spread_cap": 100, "volumes_m1": [100.0, 102.0]}
    score_ok = detector.score(micro=micro_ok, market_state=ms_ok)
    assert score_ok is not None
    assert score_ok < 5.0  # debe ser razonable


def test_should_block_cooldown(tmp_path) -> None:
    """Una vez que bloquea, los siguientes calls también bloquean (cooldown)."""
    import torch
    from ml.anomaly_autoencoder import AnomalyAutoencoder

    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(500, 8)).astype(np.float32)
    model = AnomalyAutoencoder(input_dim=8, encoder_dims=(16, 8, 4), dropout=0.0)
    _train_simple(model, X, epochs=40)
    mp = tmp_path / "ae.pt"
    tp = tmp_path / "thr.json"
    torch.save(model.state_dict(), mp)
    tp.write_text(json.dumps({"p99_error": 0.01}))  # threshold artificialmente bajo

    detector = AnomalyDetector(cfg={
        "enabled": True,
        "model_path": str(mp),
        "threshold_path": str(tp),
        "block_threshold": 0.10,
        "block_cooldown_s": 90.0,
        "input_features": 8,
        "encoder_dims": [16, 8, 4],
        "dropout": 0.0,
    })
    micro = {"intention_loss": 5.0, "absorption_pressure": 5.0,
             "directional_exhaustion": 5.0, "institutional_velocity": 5.0}
    ms = {"spread_points": 999, "spread_cap": 1.0}
    b1, r1, _s1 = detector.should_block(now_mono=100.0, micro=micro, market_state=ms)
    b2, r2, _s2 = detector.should_block(now_mono=110.0, micro=micro, market_state=ms)
    assert b1 is True and "anomaly_autoencoder" in r1
    assert b2 is True, "Cooldown debe seguir bloqueando."

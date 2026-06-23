"""Tests sincronización retrain por símbolo y validación OOS."""

from __future__ import annotations

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.model_versioning import ModelVersionManager
from ml.oos_validator import OOSValidator, compute_oos_metrics
from ml.symbol_model_registry import SymbolModelRegistry


def test_oos_metrics_and_promotion_gate() -> None:
    m = compute_oos_metrics(
        y_true=[1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        y_pred=[1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        r_multiples=[0.5, -0.3, 0.8, 0.2, -0.4, 0.3, 0.6, -0.1, 0.4, -0.2],
    )
    assert "sharpe" in m
    assert "expectancy" in m
    v = OOSValidator({"max_overfitting": 0.5, "min_sharpe": -2.0})
    ok, reason = v.should_promote(m, None)
    assert ok, reason


def test_symbol_registry_paths() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        reg = SymbolModelRegistry(
            root,
            asset_profiles={"XAUUSD": {"ml_model_path": str(root / "model_XAUUSD.pkl")}},
            config_symbols=["XAUUSD", "NDX100"],
        )
        assert reg.deploy_path("XAUUSD") == root / "model_XAUUSD.pkl"
        assert reg.best_path("XAUUSD") == root / "XAUUSD" / "best_model.pkl"
        assert "XAUUSD" in reg.configured_symbols()


def test_version_manager_rejects_bad_oos() -> None:
    vm = ModelVersionManager(Path("."), symbol="XAUUSD")
    bad = {"expectancy": -0.5, "drawdown": 20.0, "sharpe": -2.0, "n_test": 20.0, "overfitting_score": 0.1}
    ok, reason = vm._oos.should_promote(bad, None)
    assert not ok
    assert reason in ("expectancy_too_low", "drawdown_too_high", "sharpe_too_low")


def _write_payload(path: Path, n_features: int) -> None:
    """Escribe un payload de modelo "fake" con sólo metadata de nº features."""
    payload = {"meta": {"n_features_in": n_features}, "estimator": None}
    path.write_bytes(pickle.dumps(payload))


def test_schema_upgrade_blocked_when_metrics_toxic() -> None:
    """Si el candidato sube esquema (v1→v2) pero métricas son catastróficas, NO promueve."""
    with TemporaryDirectory() as td:
        root = Path(td)
        vm = ModelVersionManager(root, symbol="BTCUSD")
        deploy = root / "model_BTCUSD.pkl"
        candidate = root / "candidate.pkl"
        _write_payload(deploy, n_features=10)
        _write_payload(candidate, n_features=25)
        toxic = {
            "expectancy": -1.85,
            "drawdown": 200.0,
            "sharpe": -87.0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "n_test": 50.0,
            "overfitting_score": 0.0,
        }
        promoted, _ = vm.register_candidate(
            candidate,
            metrics=toxic,
            reason="unit_test",
            deploy_path=deploy,
        )
        assert not promoted, "Modelo tóxico no debe promoverse ni con schema upgrade"


def test_schema_upgrade_allowed_when_metrics_reasonable() -> None:
    """Si las métricas son neutrales/aceptables y hay schema upgrade, sí promueve."""
    with TemporaryDirectory() as td:
        root = Path(td)
        vm = ModelVersionManager(root, symbol="XAUUSD")
        deploy = root / "model_XAUUSD.pkl"
        candidate = root / "candidate.pkl"
        _write_payload(deploy, n_features=10)
        _write_payload(candidate, n_features=25)
        ok = {
            "expectancy": -0.30,
            "drawdown": 8.0,
            "sharpe": -0.40,
            "winrate": 0.35,
            "profit_factor": 0.75,
            "n_test": 50.0,
            "overfitting_score": 0.10,
        }
        promoted, _ = vm.register_candidate(
            candidate,
            metrics=ok,
            reason="unit_test",
            deploy_path=deploy,
        )
        assert promoted, "Schema upgrade con métricas no-tóxicas debe forzar promoción"

"""Tests para ml.drift_detector."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ml.drift_detector import ModelDriftDetector


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestModelDriftDetector:
    def test_disabled_returns_no_op(self) -> None:
        det = ModelDriftDetector(enabled=False)
        ev = det.on_trade_closed(r_multiple=-2.0, win=False, symbol="XAUUSD")
        assert ev.detected is False
        assert ev.action == ""

    def test_under_min_trades_no_alert(self) -> None:
        det = ModelDriftDetector(
            enabled=True, min_trades_for_check=20, baseline_win_rate=0.55,
        )
        # Empuja 10 perdidas
        for _ in range(10):
            ev = det.on_trade_closed(r_multiple=-1.0, win=False)
        assert ev.detected is False

    def test_wr_drop_triggers_warning(self) -> None:
        det = ModelDriftDetector(
            enabled=True, window_size=20, min_trades_for_check=20,
            baseline_win_rate=0.55, baseline_avg_r=0.30,
            drift_wr_threshold=0.10, drift_r_threshold=0.20,
        )
        # 20 perdidas seguidas: WR=0 muy por debajo del baseline 0.55
        last_ev = None
        for _ in range(20):
            last_ev = det.on_trade_closed(r_multiple=-1.0, win=False)
        assert last_ev is not None
        assert last_ev.detected is True
        assert last_ev.severity in {"warning", "critical"}

    def test_consecutive_warnings_escalate_to_critical(self) -> None:
        det = ModelDriftDetector(
            enabled=True, window_size=20, min_trades_for_check=20,
            baseline_win_rate=0.55,
            drift_wr_threshold=0.10,
            consecutive_warnings_for_critical=2,
            action_on_critical="trigger_retrain",
            cooldown_between_retrains_hours=0,  # sin cooldown para test
        )
        # Llenar 20 trades perdedores - primer detect = warning, segundo = critical
        ev = None
        for _ in range(20):
            ev = det.on_trade_closed(r_multiple=-1.0, win=False)
        # Despues de 20 ya esta lleno. La primera deteccion fue tras el 20avo,
        # y la segunda tras el 21avo.
        ev2 = det.on_trade_closed(r_multiple=-1.0, win=False)
        assert ev2.detected is True
        assert ev2.action == "trigger_retrain"

    def test_recovery_resets_consecutive_counter(self) -> None:
        det = ModelDriftDetector(
            enabled=True, window_size=20, min_trades_for_check=20,
            baseline_win_rate=0.55, baseline_avg_r=0.30,
            drift_wr_threshold=0.10,
        )
        # 20 perdidas seguidas: alerta
        for _ in range(20):
            det.on_trade_closed(r_multiple=-1.0, win=False)
        # 20 ganancias seguidas - la ventana se vacia de perdidas
        last_ev = None
        for _ in range(20):
            last_ev = det.on_trade_closed(r_multiple=1.0, win=True)
        assert last_ev is not None
        # WR ahora 100% (todas ganadoras)
        assert last_ev.window_wr == 1.0
        assert last_ev.detected is False

    def test_cooldown_prevents_repeated_retrains(self) -> None:
        det = ModelDriftDetector(
            enabled=True, window_size=20, min_trades_for_check=20,
            baseline_win_rate=0.55, baseline_avg_r=0.30,
            drift_wr_threshold=0.10,
            consecutive_warnings_for_critical=1,
            cooldown_between_retrains_hours=24,
        )
        # Primer disparo
        for _ in range(20):
            ev1 = det.on_trade_closed(r_multiple=-1.0, win=False)
        assert ev1.action == "trigger_retrain"
        # Segundo intento dentro de cooldown
        ev2 = det.on_trade_closed(r_multiple=-1.0, win=False)
        # No debe disparar retrain de nuevo
        assert ev2.action != "trigger_retrain"

    def test_per_symbol_state_isolation(self) -> None:
        det = ModelDriftDetector(
            enabled=True, window_size=20, min_trades_for_check=20,
            baseline_win_rate=0.55,
        )
        # XAUUSD: perdedoras
        for _ in range(20):
            det.on_trade_closed(r_multiple=-1.0, win=False, symbol="XAUUSD")
        # NDX100: ganadoras
        last_ndx = None
        for _ in range(20):
            last_ndx = det.on_trade_closed(r_multiple=1.5, win=True, symbol="NDX100")
        # NDX100 no debe estar contaminado por XAUUSD
        assert last_ndx is not None
        assert last_ndx.detected is False

    def test_pf_below_1_with_positive_baseline_alerts(self) -> None:
        det = ModelDriftDetector(
            enabled=True, window_size=20, min_trades_for_check=20,
            baseline_win_rate=0.45,  # bajo, no triggea WR alone
            baseline_avg_r=0.10,
            drift_wr_threshold=0.20,
            drift_r_threshold=0.50,  # alto, no triggea avgR alone
        )
        # 10 wins de +0.3R, 10 losses de -1.0R => PF = 3.0 / 10.0 = 0.3 < 1.0
        for _ in range(10):
            det.on_trade_closed(r_multiple=0.3, win=True)
        last_ev = None
        for _ in range(10):
            last_ev = det.on_trade_closed(r_multiple=-1.0, win=False)
        assert last_ev is not None
        assert last_ev.window_pf < 1.0
        assert last_ev.detected is True

    def test_from_config_yaml_shape(self) -> None:
        cfg = {
            "enabled": True,
            "window_size": 30,
            "baseline_win_rate": 0.50,
            "baseline_avg_r": 0.25,
            "drift_wr_threshold": 0.08,
            "action_on_critical": "warn",
        }
        det = ModelDriftDetector.from_config(cfg)
        assert det.enabled is True
        assert det.window_size == 30
        assert det.baseline_win_rate == 0.50
        assert det.drift_wr_threshold == 0.08
        assert det.action_on_critical == "warn"

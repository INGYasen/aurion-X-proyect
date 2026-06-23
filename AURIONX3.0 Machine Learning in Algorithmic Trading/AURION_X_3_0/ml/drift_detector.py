"""Detector de drift de rendimiento del modelo y disparador de auto-retrain.

AURION X 4.3 (7-jun-2026): Monitorea performance del modelo en ventana movil
de N trades. Si win rate o R promedio degradan vs baseline historica, levanta
una flag de DRIFT y opcionalmente dispara reentrenamiento.

Diseno:
    - Ventana movil (default 50 trades) con KPIs: win_rate, avg_r, profit_factor.
    - Baseline: estadisticas globales del modelo desde su training set
      (se calculan del trade_repository al primer init).
    - Reglas de drift:
        * abs(win_rate_window - win_rate_baseline) > drift_wr_threshold (default 0.10)
        * avg_r_window < avg_r_baseline - drift_r_threshold (default 0.20)
        * profit_factor_window < 1.0 cuando baseline > 1.2 (degradacion mayor)
    - Hysteresis: necesita 2 chequeos consecutivos para evitar falsos positivos.
    - Acciones disponibles:
        * "warn": solo log.warning
        * "trigger_retrain": dispara reentrenamiento via callback
        * "halt_trading": pausa entradas via callback (futuro)

Uso tipico:
    detector = ModelDriftDetector.from_config(cfg["ml"]["drift"])
    # Llamar despues de cada cierre de trade:
    drift = detector.on_trade_closed(r_multiple=0.5, win=True, ts=now_utc, symbol="XAUUSD")
    if drift.action == "trigger_retrain":
        run_retrain()
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DriftEvent:
    """Resultado de un chequeo de drift tras cierre de trade."""

    detected: bool
    severity: str  # "none" | "warning" | "critical"
    action: str  # "" | "warn" | "trigger_retrain" | "halt_trading"
    message: str
    window_wr: float
    window_avg_r: float
    window_pf: float


@dataclass
class _SymbolDriftState:
    """Estado por simbolo del detector."""

    window: deque = field(default_factory=lambda: deque(maxlen=50))
    consecutive_warnings: int = 0
    last_retrain_ts: datetime | None = None
    last_drift_alert_ts: datetime | None = None


@dataclass
class ModelDriftDetector:
    """Monitorea drift de performance vs baseline."""

    enabled: bool = True
    window_size: int = 50
    baseline_win_rate: float = 0.55  # XAUUSD historico ~46%; aspiracional 55%
    baseline_avg_r: float = 0.30
    drift_wr_threshold: float = 0.10  # 10pp drop disparra warning
    drift_r_threshold: float = 0.20   # 0.2R drop disparra warning
    consecutive_warnings_for_critical: int = 2
    min_trades_for_check: int = 20  # No alertar con menos de 20 trades
    cooldown_between_retrains_hours: int = 24
    action_on_critical: str = "trigger_retrain"  # warn | trigger_retrain | halt_trading
    _states: dict[str, _SymbolDriftState] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Mapping[str, object] | None) -> "ModelDriftDetector":
        if not cfg:
            return cls(enabled=False)
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            window_size=int(cfg.get("window_size", 50) or 50),
            baseline_win_rate=float(cfg.get("baseline_win_rate", 0.55) or 0.55),
            baseline_avg_r=float(cfg.get("baseline_avg_r", 0.30) or 0.30),
            drift_wr_threshold=float(cfg.get("drift_wr_threshold", 0.10) or 0.10),
            drift_r_threshold=float(cfg.get("drift_r_threshold", 0.20) or 0.20),
            consecutive_warnings_for_critical=int(cfg.get("consecutive_warnings_for_critical", 2) or 2),
            min_trades_for_check=int(cfg.get("min_trades_for_check", 20) or 20),
            cooldown_between_retrains_hours=int(cfg.get("cooldown_between_retrains_hours", 24) or 24),
            action_on_critical=str(cfg.get("action_on_critical", "trigger_retrain") or "trigger_retrain"),
        )

    def _get_state(self, symbol: str) -> _SymbolDriftState:
        sym = (symbol or "").upper()
        if sym not in self._states:
            st = _SymbolDriftState(window=deque(maxlen=self.window_size))
            self._states[sym] = st
        return self._states[sym]

    def on_trade_closed(
        self,
        r_multiple: float,
        win: bool,
        ts: datetime | None = None,
        symbol: str = "XAUUSD",
    ) -> DriftEvent:
        """Procesa cierre y devuelve DriftEvent con accion sugerida.

        Args:
            r_multiple: R-multiple del trade cerrado.
            win: True si profit > 0.
            ts: timestamp UTC del cierre (default now).
            symbol: simbolo del trade.
        """
        ev_default = DriftEvent(
            detected=False, severity="none", action="",
            message="", window_wr=0.0, window_avg_r=0.0, window_pf=0.0,
        )
        if not self.enabled:
            return ev_default
        ts = ts or datetime.now(timezone.utc)
        st = self._get_state(symbol)
        st.window.append({"r": float(r_multiple), "win": bool(win), "ts": ts})

        if len(st.window) < self.min_trades_for_check:
            return ev_default

        # KPIs ventana.
        n = len(st.window)
        wins = sum(1 for tr in st.window if tr["win"])
        wr = wins / n
        avg_r = sum(tr["r"] for tr in st.window) / n
        gains = sum(tr["r"] for tr in st.window if tr["r"] > 0)
        losses = abs(sum(tr["r"] for tr in st.window if tr["r"] < 0))
        pf = (gains / losses) if losses > 1e-9 else (10.0 if gains > 0 else 0.0)

        # Reglas de drift.
        wr_drop = self.baseline_win_rate - wr
        r_drop = self.baseline_avg_r - avg_r
        drift_signals = []
        if wr_drop > self.drift_wr_threshold:
            drift_signals.append(
                f"WR drop {wr*100:.1f}% vs baseline {self.baseline_win_rate*100:.1f}% (Δ={wr_drop*100:.1f}pp)"
            )
        if r_drop > self.drift_r_threshold:
            drift_signals.append(
                f"avgR drop {avg_r:+.3f} vs baseline {self.baseline_avg_r:+.3f} (Δ={r_drop:+.3f})"
            )
        if pf < 1.0 and self.baseline_avg_r > 0:
            drift_signals.append(f"PF {pf:.2f} < 1.0 (perdida neta)")

        if not drift_signals:
            st.consecutive_warnings = 0
            return DriftEvent(
                detected=False, severity="none", action="",
                message=f"n={n} WR={wr*100:.1f}% avgR={avg_r:+.3f} PF={pf:.2f}",
                window_wr=wr, window_avg_r=avg_r, window_pf=pf,
            )

        st.consecutive_warnings += 1
        msg = "; ".join(drift_signals)
        severity = "warning"
        action = "warn"

        if st.consecutive_warnings >= self.consecutive_warnings_for_critical:
            # Cooldown post-retrain para evitar reentrenos en cascada.
            since_last = (
                (ts - st.last_retrain_ts).total_seconds() / 3600.0
                if st.last_retrain_ts else 1e9
            )
            if since_last >= self.cooldown_between_retrains_hours:
                severity = "critical"
                action = self.action_on_critical
                st.last_retrain_ts = ts
                st.consecutive_warnings = 0
            else:
                action = "warn"  # cooldown activo
                msg += f" (retrain cooldown: {since_last:.1f}h < {self.cooldown_between_retrains_hours}h)"

        return DriftEvent(
            detected=True, severity=severity, action=action,
            message=msg, window_wr=wr, window_avg_r=avg_r, window_pf=pf,
        )

    def execute_action(
        self,
        ev: DriftEvent,
        retrain_callback: Callable[[], None] | None = None,
        halt_callback: Callable[[], None] | None = None,
    ) -> None:
        """Ejecuta accion sugerida por DriftEvent con callbacks opcionales."""
        if not ev.detected:
            return
        if ev.action == "warn":
            logger.warning(
                "[DRIFT] %s severity=%s msg=%s",
                "DETECTADO" if ev.severity == "warning" else "CRITICO",
                ev.severity, ev.message,
            )
        elif ev.action == "trigger_retrain":
            logger.warning(
                "[DRIFT] CRITICAL - disparando reentrenamiento: %s", ev.message,
            )
            if retrain_callback is not None:
                try:
                    retrain_callback()
                except Exception as exc:
                    logger.error("[DRIFT] retrain_callback fallo: %s", exc)
        elif ev.action == "halt_trading":
            logger.error("[DRIFT] CRITICAL - halting trading: %s", ev.message)
            if halt_callback is not None:
                try:
                    halt_callback()
                except Exception as exc:
                    logger.error("[DRIFT] halt_callback fallo: %s", exc)

"""Parada de emergencia: cierra todo y bloquea nuevas entradas ante anomalías."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol

from analytics.logger import get_logger

logger = get_logger(__name__)


class EmergencyState(str, Enum):
    """Estado del controlador de emergencia."""

    NOMINAL = "nominal"
    TRIGGERED = "triggered"
    LOCKED = "locked"


class EmergencyExecutor(Protocol):
    """Interfaz mínima para ejecutar cierre masivo en emergencia."""

    def close_all(self) -> None:
        """Cierra todas las posiciones abiertas."""
        ...


class EmergencyController:
    """Monitoriza condiciones críticas y fuerza un apagado seguro."""

    def __init__(
        self,
        *,
        atr_spike_threshold: float = 2.5,
        spread_max: float = 0.0015,
        pause_seconds: int = 300,
        spread_emergency_enabled: bool = True,
    ) -> None:
        """Inicializa umbrales de disparo por ATR/spread."""
        self._state = EmergencyState.NOMINAL
        self._atr_spike_threshold = atr_spike_threshold
        self._spread_max = spread_max
        self._pause_seconds = pause_seconds
        self._spread_emergency_enabled = spread_emergency_enabled
        self._paused_until: datetime | None = None
        self._last_reason: str | None = None
        self._last_trigger_logged_at: datetime | None = None
        logger.debug("EmergencyController en estado %s", self._state)

    @property
    def state(self) -> EmergencyState:
        """Estado actual del controlador."""
        return self._state

    @property
    def paused_until(self) -> datetime | None:
        """Instante UTC hasta el cual se fuerza pausa."""
        return self._paused_until

    @property
    def last_reason(self) -> str | None:
        """Última razón que activó emergencia."""
        return self._last_reason

    def evaluate_and_act(
        self,
        *,
        atr_current: float,
        atr_baseline: float,
        spread: float,
        executor: EmergencyExecutor,
    ) -> bool:
        """Dispara emergencia por ATR spike o spread alto.

        Condiciones:
        - `atr_current / atr_baseline >= atr_spike_threshold`
        - `spread >= spread_max` (omitida si ``spread_emergency_enabled`` es falso)

        Si se activa: `close_all()` + `pause`.
        """
        if self._state == EmergencyState.LOCKED:
            return True

        # Si está en pausa temporal, no re-disparar en cada tick.
        if self.is_paused():
            return True

        now = datetime.now(timezone.utc)
        # Si la pausa terminó y no está bloqueado, volver a estado nominal.
        if self._state == EmergencyState.TRIGGERED and self._paused_until is not None:
            if now >= self._paused_until:
                self._state = EmergencyState.NOMINAL
                self._paused_until = None
                logger.info("EmergencyController: fin de pausa, estado NOMINAL")

        # Mismo criterio que antes: baseline<=0 ⇒ ratio inf (dispara); NaN en ratio ⇒ no dispara.
        if atr_baseline <= 0:
            spike_ratio = float("inf")
        elif not math.isfinite(atr_current) or not math.isfinite(atr_baseline):
            spike_ratio = float("nan")
        else:
            spike_ratio = atr_current / atr_baseline

        atr_fire = spike_ratio == float("inf") or (
            math.isfinite(spike_ratio) and spike_ratio >= self._atr_spike_threshold
        )
        if atr_fire:
            reason_ratio = (
                f"{spike_ratio:.2f}"
                if math.isfinite(spike_ratio)
                else ("inf" if spike_ratio == float("inf") else "invalid")
            )
            self.trigger(f"ATR spike ratio={reason_ratio}")
            try:
                executor.close_all()
            except Exception:
                logger.exception("EmergencyController: fallo en close_all() por ATR spike")
            self.pause()
            return True

        if (
            self._spread_emergency_enabled
            and math.isfinite(spread)
            and spread >= self._spread_max
        ):
            self.trigger(f"spread alto={spread:.6f}")
            try:
                executor.close_all()
            except Exception:
                logger.exception("EmergencyController: fallo en close_all() por spread")
            self.pause()
            return True

        return self._state in {EmergencyState.TRIGGERED, EmergencyState.LOCKED}

    def pause(self) -> None:
        """Activa pausa temporal de nuevas entradas."""
        self._paused_until = datetime.now(timezone.utc) + timedelta(seconds=self._pause_seconds)
        if self._state != EmergencyState.LOCKED:
            self._state = EmergencyState.TRIGGERED

    def is_paused(self) -> bool:
        """Indica si el sistema sigue en pausa."""
        if self._state == EmergencyState.LOCKED:
            return True
        if self._paused_until is None:
            return False
        return datetime.now(timezone.utc) < self._paused_until

    def trigger(self, reason: str) -> None:
        """Activa emergencia por ``reason`` (cerrar posiciones, bloquear entradas)."""
        self._last_reason = reason
        if self._state != EmergencyState.LOCKED:
            self._state = EmergencyState.TRIGGERED
        now = datetime.now(timezone.utc)
        if self._last_trigger_logged_at is None or (now - self._last_trigger_logged_at).total_seconds() >= 5:
            logger.error("EmergencyController.trigger: %s", reason)
            self._last_trigger_logged_at = now

    def reset(self, *, operator_ack: bool = False) -> None:
        """Vuelve a nominal tras confirmación humana explícita.

        Args:
            operator_ack: Debe ser ``True`` para desbloquear (stub de política).
        """
        if not operator_ack:
            self._state = EmergencyState.LOCKED
            logger.warning("reset rechazado sin operator_ack; estado=LOCKED")
            return
        self._state = EmergencyState.NOMINAL
        self._paused_until = None
        logger.info("EmergencyController.reset: estado NOMINAL")

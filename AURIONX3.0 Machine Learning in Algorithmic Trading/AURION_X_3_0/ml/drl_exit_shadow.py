"""AURION X 4.0 — Shadow Mode para el DRL Exit Agent.

Ejecuta el agente PPO **en paralelo** al DynamicExitManager del 3.0 sin enviar
órdenes al broker. Registra cada decisión en la tabla
``drl_shadow_decisions`` para análisis posterior.

Promoción a modo activo: tras N (default 200) trades en sombra, si:

    - ROI simulado del agente >= ROI real DynamicExitManager * 1.15,
    - Drawdown simulado del agente <= Drawdown real DynamicExitManager,

…se puede activar ``drl_exit.shadow_mode: false``.

NOTA: shadow_mode NUNCA llama a ``OrderManager.send_market`` ni
``MT5Connector.modify_position``. Es READ-ONLY salvo el INSERT a Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from analytics.logger import get_logger

from .drl_exit_agent import DRLExitAgent, is_available as _drl_available
from .envs.exit_env import ExitAction, STATE_DIM

logger = get_logger(__name__)


@dataclass
class ShadowDecision:
    ts: datetime
    symbol: str
    trade_ticket: str
    state_vec: list[float]
    drl_action: int
    actual_action: str
    notes: str | None = None


class DRLExitShadow:
    """Shadow tracker que loguea decisiones del agente sin actuar.

    Diseño:
        - Lazy-init del agente (no carga PPO hasta que es necesario).
        - Persistencia opcional vía ``analytics.db_connection``. Si la BD no
          está disponible (e.g. tests) acumula en memoria.
    """

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        symbol: str = "XAUUSD",
    ) -> None:
        c = dict(cfg or {})
        self._cfg = c
        self._enabled = bool(c.get("enabled", False))
        self._shadow_mode = bool(c.get("shadow_mode", True))
        self._model_path = c.get("model_path")
        self._symbol = symbol.upper()
        self._agent: DRLExitAgent | None = None
        self._loaded = False
        self._in_memory: list[ShadowDecision] = []
        self._db_engine: Any = None
        self._db_session_factory: Any = None
        self._db_attempted = False

    @property
    def is_active(self) -> bool:
        return self._enabled and self._shadow_mode and _drl_available()

    def _ensure_agent(self) -> bool:
        if self._loaded:
            return self._agent is not None and self._agent.is_loaded
        if not self.is_active:
            return False
        self._agent = DRLExitAgent(model_path=self._model_path)
        ok = self._agent.load()
        self._loaded = True
        if not ok:
            logger.warning("DRLExitShadow: modelo no disponible (%s).", self._model_path)
        return ok

    def _ensure_db(self) -> bool:
        if self._db_attempted:
            return self._db_engine is not None
        self._db_attempted = True
        try:
            from analytics.db_connection import get_engine, get_session_factory  # type: ignore[attr-defined]
            from analytics.drl_shadow_models import ensure_table

            self._db_engine = get_engine()
            ensure_table(self._db_engine)
            self._db_session_factory = get_session_factory()
            return True
        except (ImportError, AttributeError, RuntimeError, ConnectionError) as exc:
            logger.warning("DRLExitShadow: BD no disponible, persistencia en memoria. (%s)", exc)
            self._db_engine = None
            return False

    def observe(
        self,
        *,
        trade_ticket: str,
        state_vec: Sequence[float] | np.ndarray,
        actual_action: str,
        notes: str | None = None,
    ) -> int | None:
        """Registra una decisión sombra y devuelve la acción que TOMARÍA el DRL.

        Devuelve ``None`` si shadow no está activo o el agente no cargó.
        """
        if not self.is_active or not self._ensure_agent() or self._agent is None:
            return None

        arr = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != STATE_DIM:
            logger.warning(
                "DRLExitShadow.observe: state_vec inválido shape=%s, esperado (%d,)",
                arr.shape, STATE_DIM,
            )
            return None

        action = self._agent.predict_action(arr, deterministic=True)
        dec = ShadowDecision(
            ts=datetime.now(timezone.utc),
            symbol=self._symbol,
            trade_ticket=str(trade_ticket),
            state_vec=arr.tolist(),
            drl_action=int(action),
            actual_action=str(actual_action),
            notes=notes,
        )
        self._persist(dec)
        return int(action)

    def _persist(self, dec: ShadowDecision) -> None:
        if not self._ensure_db():
            self._in_memory.append(dec)
            return
        try:
            from analytics.drl_shadow_models import DRLShadowDecision

            with self._db_session_factory() as session:
                session.add(DRLShadowDecision(
                    ts=dec.ts,
                    symbol=dec.symbol,
                    trade_ticket=dec.trade_ticket,
                    state_vec=dec.state_vec,
                    drl_action=dec.drl_action,
                    actual_action=dec.actual_action,
                    notes=dec.notes,
                ))
                session.commit()
        except Exception as exc:  # noqa: BLE001 — capturamos todo para nunca romper el loop
            logger.warning("DRLExitShadow: persistencia fallida (%s). Memoria.", exc)
            self._in_memory.append(dec)

    def get_in_memory_decisions(self) -> list[ShadowDecision]:
        """Devuelve copia de las decisiones acumuladas si la BD no estaba disponible."""
        return list(self._in_memory)

    @staticmethod
    def action_name(action: int) -> str:
        try:
            return ExitAction(int(action)).name
        except (ValueError, KeyError):
            return f"UNKNOWN_{action}"

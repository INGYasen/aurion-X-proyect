"""Cola y ciclo de vida de órdenes (market, limit, stop, modificaciones)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from analytics.logger import get_logger

if TYPE_CHECKING:
    from .mt5_connector import MT5Connector

logger = get_logger(__name__)

_LIVE_ENVIRONMENTS = frozenset({"live", "production", "prod"})


class OrderManager:
    """Traduce intenciones de trading en llamadas concretas al conector."""

    def __init__(
        self,
        connector: MT5Connector | None = None,
        *,
        environment: str = "demo",
        allow_simulated: bool | None = None,
    ) -> None:
        """Asocia un conector de ejecución (MT5 u otro).

        Args:
            connector: Instancia tipo ``MT5Connector`` u homóloga.
            environment: ``live`` / ``demo`` / ``backtest`` (en live no hay tickets simulados).
            allow_simulated: Si ``None``, se desactiva en entornos live.
        """
        self._connector = connector
        self._environment = str(environment).strip().lower()
        if allow_simulated is None:
            self._allow_simulated = self._environment not in _LIVE_ENVIRONMENTS
        else:
            self._allow_simulated = bool(allow_simulated)
        self._open_positions: dict[str, dict[str, Any]] = {}
        logger.debug(
            "OrderManager creado env=%s allow_simulated=%s",
            self._environment,
            self._allow_simulated,
        )

    def set_connector(self, connector: MT5Connector | None) -> None:
        """Actualiza el conector sin recrear el gestor (preserva estado interno)."""
        self._connector = connector

    def set_environment(self, environment: str, *, allow_simulated: bool | None = None) -> None:
        """Ajusta política de simulación según entorno de ejecución."""
        self._environment = str(environment).strip().lower()
        if allow_simulated is None:
            self._allow_simulated = self._environment not in _LIVE_ENVIRONMENTS
        else:
            self._allow_simulated = bool(allow_simulated)

    @staticmethod
    def is_broker_ticket(ticket: str | None) -> bool:
        """True si el ticket parece ID numérico del broker (no UUID simulado)."""
        if not ticket:
            return False
        s = str(ticket).strip()
        if len(s) >= 32 and "-" in s:
            return False
        try:
            return int(float(s)) > 0
        except (ValueError, TypeError):
            return False

    def place_market_order(
        self,
        symbol: str,
        volume: float,
        side: str,
        *,
        sl: float | None = None,
        tp: float | None = None,
    ) -> str | None:
        """Envía una orden a mercado.

        Args:
            symbol: Activo negociado.
            volume: Tamaño en lotes o unidades del broker.
            side: ``"buy"`` o ``"sell"``.

        Returns:
            Identificador de orden del broker o ``None`` si falló.
        """
        side_norm = side.lower().strip()
        if side_norm not in {"buy", "sell"} or volume <= 0:
            logger.warning("place_market_order inválido side=%s volume=%s", side, volume)
            return None
        if self._connector is not None:
            ticket_mt5 = self._connector.place_market_order(
                symbol=symbol,
                volume=volume,
                side=side_norm,
                sl=sl,
                tp=tp,
            )
            if ticket_mt5 is not None:
                if not self.is_broker_ticket(ticket_mt5):
                    logger.error(
                        "place_market_order: ticket no numérico symbol=%s ticket=%r",
                        symbol,
                        ticket_mt5,
                    )
                    return None
                return ticket_mt5
            if not self._allow_simulated:
                logger.error(
                    "LIVE: orden no ejecutada en broker (sin fallback simulado) symbol=%s side=%s vol=%s",
                    symbol,
                    side_norm,
                    volume,
                )
                return None
        elif not self._allow_simulated:
            logger.error(
                "LIVE: sin conector MT5, no se puede enviar orden symbol=%s",
                symbol,
            )
            return None
        ticket = str(uuid.uuid4())
        self._open_positions[ticket] = {
            "symbol": symbol,
            "volume": float(volume),
            "side": side_norm,
            "sl": sl,
            "tp": tp,
        }
        logger.info(
            "Orden simulada creada ticket=%s symbol=%s side=%s vol=%s sl=%s tp=%s",
            ticket,
            symbol,
            side_norm,
            volume,
            sl,
            tp,
        )
        return ticket

    def close_position(self, ticket: str) -> bool:
        """Cierra una posición abierta identificada por ``ticket``."""
        if self._connector is not None and self._connector.close_position(ticket):
            return True
        if ticket not in self._open_positions:
            return False
        self._open_positions.pop(ticket, None)
        logger.info("Posición cerrada ticket=%s", ticket)
        return True

    def close_position_partial(self, ticket: str, volume: float) -> bool:
        """Cierra ``volume`` de la posición; en simulador reduce el lote interno."""
        if volume <= 0:
            return False
        if self._connector is not None and hasattr(self._connector, "close_position_partial"):
            fn = getattr(self._connector, "close_position_partial")
            if callable(fn) and bool(fn(ticket, volume)):
                return True
        if ticket not in self._open_positions:
            return False
        pos = self._open_positions[ticket]
        cur = float(pos.get("volume", 0.0))
        if cur <= 0:
            return False
        if volume >= cur - 1e-9:
            return self.close_position(ticket)
        new_v = max(0.0, cur - volume)
        if new_v < 1e-6:
            return self.close_position(ticket)
        pos["volume"] = new_v
        logger.info("Posición parcial ticket=%s cerrado_vol=%.4f restante=%.4f", ticket, volume, new_v)
        return True

    def modify_order(self, ticket: str, **kwargs: Any) -> bool:
        """Modifica SL/TP u otros parámetros soportados por el broker."""
        if self._connector is not None:
            if self._connector.modify_position(
                ticket,
                sl=kwargs.get("sl"),
                tp=kwargs.get("tp"),
            ):
                return True
        if ticket not in self._open_positions:
            return False
        self._open_positions[ticket].update(kwargs)
        logger.debug("Orden modificada ticket=%s fields=%s", ticket, list(kwargs.keys()))
        return True

    def is_position_open(self, ticket: str) -> bool:
        """Indica si la posición sigue abierta en broker/simulador."""
        if self._connector is not None:
            return bool(self._connector.is_position_open(ticket))
        return ticket in self._open_positions

    def get_closed_summary(self, ticket: str) -> dict[str, Any] | None:
        """Resumen de cierre (exit_price/profit/close_time) si el broker lo dispone."""
        if self._connector is None:
            return None
        return self._connector.get_closed_summary(ticket)

    def close_all(self) -> None:
        """Cierra todas las posiciones abiertas (usado por emergencia)."""
        if self._connector is not None:
            self._connector.close_all()
        tickets = list(self._open_positions.keys())
        for ticket in tickets:
            self.close_position(ticket)

"""Recepción y normalización de ticks (precios/volumen/tiempo)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Protocol

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Tick:
    """Representación mínima de un tick de mercado."""

    symbol: str
    bid: Decimal
    ask: Decimal
    time_utc: datetime
    volume: Decimal | None = None


class TickConsumer(Protocol):
    """Protocolo para componentes que consumen ticks secuencialmente."""

    def on_tick(self, tick: Tick) -> None: ...


class TickConnector(Protocol):
    """Fuente de ticks por símbolo (p. ej. MT5 u otro broker)."""

    def get_tick(self, symbol: str) -> Tick | None:
        """Devuelve el último tick disponible o ``None`` si no hay dato."""
        ...


class TickOrchestrator(Protocol):
    """Orquestador que reacciona a ticks en el bucle de mercado."""

    def on_tick(self, symbol: str, tick: Tick) -> None:
        """Procesa un tick ya normalizado para el símbolo dado."""
        ...


class TickHandler:
    """Enruta ticks del conector hacia el orquestador y suscriptores locales."""

    def __init__(
        self,
        connector: TickConnector,
        orchestrator: TickOrchestrator,
        *,
        poll_interval_s: float = 0.05,
    ) -> None:
        """Inicializa manejador con conector y orquestador.

        Args:
            connector: Implementa ``get_tick(symbol)``.
            orchestrator: Recibe ``on_tick(symbol, tick)``.
            poll_interval_s: Pausa entre consultas por símbolo (segundos).
        """
        self._connector = connector
        self._orchestrator = orchestrator
        self._poll_interval_s = poll_interval_s
        self._consumers: list[TickConsumer] = []
        logger.debug("TickHandler creado (poll_interval_s=%s)", poll_interval_s)

    def subscribe(self, consumer: TickConsumer) -> None:
        """Registra un consumidor de ticks.

        Args:
            consumer: Objeto que implementa ``on_tick(tick)`` (un solo argumento).
        """
        self._consumers.append(consumer)
        logger.debug("TickHandler.subscribe: %s consumidores", len(self._consumers))

    def ingest(self, raw: dict[str, Any]) -> Tick:
        """Convierte un payload crudo del broker en ``Tick``.

        Args:
            raw: Claves esperadas: ``symbol``, ``bid``, ``ask``; tiempo en
                ``time_utc`` o ``time`` (datetime o ISO); ``volume`` opcional.

        Returns:
            Instancia ``Tick`` normalizada.

        Raises:
            ValueError: si faltan campos o tipos inválidos.
        """
        try:
            symbol = str(raw["symbol"]).strip()
            bid = Decimal(str(raw["bid"]))
            ask = Decimal(str(raw["ask"]))
        except (KeyError, InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"Payload de tick inválido: {raw!r}") from exc

        t_raw = raw.get("time_utc", raw.get("time"))
        if isinstance(t_raw, datetime):
            dt = t_raw if t_raw.tzinfo else t_raw.replace(tzinfo=timezone.utc)
            time_utc = dt.astimezone(timezone.utc)
        elif isinstance(t_raw, str):
            time_utc = datetime.fromisoformat(t_raw.replace("Z", "+00:00")).astimezone(
                timezone.utc,
            )
        else:
            time_utc = datetime.now(timezone.utc)

        vol: Decimal | None = None
        if raw.get("volume") is not None:
            try:
                vol = Decimal(str(raw["volume"]))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ValueError(f"Volumen inválido: {raw.get('volume')!r}") from exc

        return Tick(symbol=symbol, bid=bid, ask=ask, time_utc=time_utc, volume=vol)

    def dispatch(self, tick: Tick) -> None:
        """Notifica a todos los suscriptores registrados (firma ``on_tick(tick)``)."""
        for consumer in self._consumers:
            consumer.on_tick(tick)

    def run(self, symbols: Iterable[str]) -> None:
        """Bucle infinito: sondea cada símbolo y delega en el orquestador.

        Por cada símbolo: ``get_tick`` → si hay tick, ``orchestrator.on_tick``;
        luego ``sleep(poll_interval_s)``.

        Args:
            symbols: Símbolos a consultar en cada vuelta del bucle externo.
        """
        symbol_list = list(symbols)
        if not symbol_list:
            logger.warning("TickHandler.run: lista de símbolos vacía; bucle sin trabajo")
        logger.info("TickHandler.run: inicio (símbolos=%s)", symbol_list)
        while True:
            for s in symbol_list:
                tick = self._connector.get_tick(s)
                if tick:
                    self._orchestrator.on_tick(s, tick)
                time.sleep(self._poll_interval_s)

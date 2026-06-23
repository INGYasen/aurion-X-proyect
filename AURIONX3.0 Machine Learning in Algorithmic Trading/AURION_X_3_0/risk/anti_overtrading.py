"""Anti-hiperactividad algorítmica."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class AntiOvertradeVerdict:
    block: bool
    reason: str
    cooldown_remaining_s: float


class AntiOvertradingSystem:
    """Detecta sobre-operación por SÍMBOLO.

    Antes era global: ``_entry_times`` y ``_signal_times`` eran únicos para
    todos los símbolos, lo que en producción con 3 instrumentos (XAUUSD,
    NDX100, BTCUSD) saturaba el umbral ``max_signals_per_10min`` casi
    instantáneamente y bloqueaba TODOS los símbolos vía ``apex_anti_overtrade:
    hyperactivity`` aunque cada símbolo individualmente no estuviera
    sobre-operado. Desde el Fix #10 los deques son POR SÍMBOLO.
    """

    def __init__(
        self,
        *,
        max_entries_per_hour: int = 4,
        max_signals_per_10min: int = 8,
        min_spacing_s: float = 120.0,
    ) -> None:
        self._max_hour = max_entries_per_hour
        self._max_burst = max_signals_per_10min
        self._min_spacing = min_spacing_s
        self._entry_times_by_symbol: dict[str, deque[float]] = {}
        self._signal_times_by_symbol: dict[str, deque[float]] = {}
        # Compatibilidad backwards: vista agregada cross-símbolo (solo lectura)
        # para diagnóstico. La lógica de bloqueo usa los deques por símbolo.
        self._entry_times: deque[float] = deque(maxlen=400)
        self._signal_times: deque[float] = deque(maxlen=200)

    def _entry_deque(self, symbol: str) -> deque[float]:
        key = (symbol or "").upper()
        dq = self._entry_times_by_symbol.get(key)
        if dq is None:
            dq = deque(maxlen=200)
            self._entry_times_by_symbol[key] = dq
        return dq

    def _signal_deque(self, symbol: str) -> deque[float]:
        key = (symbol or "").upper()
        dq = self._signal_times_by_symbol.get(key)
        if dq is None:
            dq = deque(maxlen=100)
            self._signal_times_by_symbol[key] = dq
        return dq

    def record_signal(self, symbol: str = "") -> None:
        """Registra una señal para el símbolo dado (o agregado si vacío).

        ``symbol`` opcional por compatibilidad backward — si no se pasa se
        registra solo en la vista agregada, lo que en la práctica no afecta a
        ningún chequeo por símbolo. Los call sites nuevos deben pasar el
        símbolo correspondiente.
        """
        now = monotonic()
        if symbol:
            self._signal_deque(symbol).append(now)
        self._signal_times.append(now)

    def record_entry(self, symbol: str = "") -> None:
        """Registra una entrada ejecutada para el símbolo (o agregada)."""
        now = monotonic()
        if symbol:
            self._entry_deque(symbol).append(now)
            self._signal_deque(symbol).append(now)
        self._entry_times.append(now)
        self._signal_times.append(now)

    def check(self, symbol: str = "") -> AntiOvertradeVerdict:
        """Chequea sobre-operación POR SÍMBOLO.

        Si ``symbol`` no se proporciona se usa el agregado (comportamiento
        previo) por compatibilidad backward, pero los call sites deben pasar
        el símbolo para que el guard sea independiente por instrumento.
        """
        now = monotonic()
        hour_ago = now - 3600.0
        burst_ago = now - 600.0

        if symbol:
            entries_dq = self._entry_deque(symbol)
            signals_dq = self._signal_deque(symbol)
        else:
            entries_dq = self._entry_times
            signals_dq = self._signal_times

        entries_h = sum(1 for t in entries_dq if t >= hour_ago)
        signals_b = sum(1 for t in signals_dq if t >= burst_ago)

        if entries_dq:
            last = entries_dq[-1]
            gap = now - last
            if gap < self._min_spacing:
                return AntiOvertradeVerdict(True, "entry_spacing", self._min_spacing - gap)

        if entries_h >= self._max_hour:
            return AntiOvertradeVerdict(True, "max_entries_hour", 0.0)
        if signals_b >= self._max_burst:
            return AntiOvertradeVerdict(True, "signal_burst", 0.0)
        return AntiOvertradeVerdict(False, "ok", 0.0)

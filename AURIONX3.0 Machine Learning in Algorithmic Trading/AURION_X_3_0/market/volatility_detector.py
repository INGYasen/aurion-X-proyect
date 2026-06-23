"""Detección de régimen de volatilidad para modular la agresividad del sistema."""

from __future__ import annotations

import statistics
from collections import deque
from decimal import Decimal
from enum import Enum
from typing import Sequence

from analytics.logger import get_logger
from market.tick_handler import Tick

logger = get_logger(__name__)


class VolatilityRegime(str, Enum):
    """Clasificación gruesa del régimen de volatilidad."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class VolatilityDetector:
    """Régimen a partir del coeficiente de variación de mids en una ventana de ticks."""

    def __init__(
        self,
        window: int = 100,
        *,
        low_cv: float = 0.00008,
        high_cv: float = 0.00035,
        extreme_cv: float = 0.00070,
    ) -> None:
        """Fija ventana y umbrales de CV (desvío / |media| de mids).

        Args:
            window: Ticks máximos retenidos.
            low_cv: Por debajo → ``LOW``.
            high_cv: Por encima → ``HIGH``.
        """
        self._window = window
        self._low_cv = low_cv
        self._high_cv = high_cv
        self._extreme_cv = max(extreme_cv, high_cv + 1e-9)
        self._mids: deque[float] = deque(maxlen=window)
        self._regime = VolatilityRegime.NORMAL
        logger.debug(
            "VolatilityDetector window=%s low_cv=%s high_cv=%s extreme_cv=%s",
            window,
            low_cv,
            high_cv,
            self._extreme_cv,
        )

    def update(self, tick: Tick) -> None:
        """Incorpora un tick y actualiza el régimen con el CV de la ventana."""
        mid = float((tick.bid + tick.ask) / Decimal("2"))
        self._mids.append(mid)
        if len(self._mids) < 10:
            self._regime = VolatilityRegime.NORMAL
            return
        vals = list(self._mids)
        mu = statistics.mean(vals)
        if abs(mu) < 1e-12:
            self._regime = VolatilityRegime.NORMAL
            return
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        cv = sd / abs(mu)
        if cv < self._low_cv:
            self._regime = VolatilityRegime.LOW
        elif cv > self._extreme_cv:
            self._regime = VolatilityRegime.EXTREME
        elif cv > self._high_cv:
            self._regime = VolatilityRegime.HIGH
        else:
            self._regime = VolatilityRegime.NORMAL

    def regime(self) -> VolatilityRegime:
        """Devuelve el régimen de volatilidad actual."""
        return self._regime

    @staticmethod
    def realized_volatility(returns: Sequence[Decimal]) -> Decimal:
        """Desviación típica poblacional de una secuencia de retornos."""
        rs = [float(x) for x in returns]
        if len(rs) < 2:
            return Decimal("0")
        m = statistics.mean(rs)
        var = sum((r - m) ** 2 for r in rs) / len(rs)
        return Decimal(str(var**0.5))

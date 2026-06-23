"""Detección de estructura de precios: swings, tendencia, rompimientos y pullbacks básicos."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Sequence

from analytics.logger import get_logger

logger = get_logger(__name__)

_LEVEL_MERGE_EPS = Decimal("1e-8")


class Trend(str, Enum):
    """Clasificación gruesa de la estructura de mercado."""

    UP = "UP"
    DOWN = "DOWN"
    RANGE = "RANGE"


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """Punto extremo local (fractal) en la serie de precios."""

    index: int
    price: Decimal
    is_high: bool


@dataclass(frozen=True, slots=True)
class StructureResult:
    """Salida del detector: tendencia y niveles aproximados (soportes/resistencias)."""

    trend: Trend
    key_levels: tuple[Decimal, ...]
    # Métricas de caja lateral (útil cuando ``trend`` es RANGE o mercado encajonado).
    range_high: Decimal | None = None
    range_low: Decimal | None = None
    # 0.0 cerca de range_low, 1.0 cerca de range_high; None si no hay caja válida.
    position_in_range: float | None = None
    # (high-low)/|close| del box inferido por swings recientes.
    range_width_pct: float | None = None


class StructureDetector:
    """Identifica HH/HL, LH/LL, rompimientos simples y pullbacks vía swings fractales."""

    def __init__(
        self,
        left_bars: int = 2,
        right_bars: int = 2,
        *,
        momentum_fallback_bars: int = 0,
        momentum_min_return_pct: float = 1.0,
        conflict_resolve_bars: int = 16,
        conflict_min_net_return_pct: float = 0.15,
    ) -> None:
        """Configura radios para confirmación de swings.

        Args:
            left_bars: Barras a la izquierda del candidato.
            right_bars: Barras a la derecha del candidato.
            momentum_fallback_bars: Si >0 y la tendencia sigue en RANGE, clasificar por
                retorno (close[-1]-close[-k])/close[-k] en porcentaje (valor absoluto).
            momentum_min_return_pct: Umbral mínimo de ese retorno para forzar UP/DOWN.
            conflict_resolve_bars: Ventana al desempatar HH/HL vs LH/LL simultáneos.
            conflict_min_net_return_pct: |retorno neto| mínimo (%) para desempate.
        """
        self._left = left_bars
        self._right = right_bars
        self._momentum_fb = max(0, int(momentum_fallback_bars))
        self._momentum_min_ret = Decimal(str(momentum_min_return_pct)) / Decimal("100")
        self._conflict_bars = max(5, int(conflict_resolve_bars))
        self._conflict_eps = Decimal(str(conflict_min_net_return_pct)) / Decimal("100")
        logger.debug(
            "StructureDetector left=%s right=%s mom_fb=%s",
            left_bars,
            right_bars,
            self._momentum_fb,
        )

    def find_swings(self, highs: Sequence[Decimal], lows: Sequence[Decimal]) -> list[SwingPoint]:
        """Localiza pivotes altos y bajos (fractales simétricos)."""
        h = [Decimal(x) for x in highs]
        l = [Decimal(x) for x in lows]
        n = len(h)
        if n != len(l) or n == 0:
            return []
        L, R = self._left, self._right
        if n < L + R + 1:
            return []

        out: list[SwingPoint] = []
        for i in range(L, n - R):
            w_hi = h[i - L : i + R + 1]
            w_lo = l[i - L : i + R + 1]
            if h[i] == max(w_hi) and h[i] > min(w_hi):
                out.append(SwingPoint(index=i, price=h[i], is_high=True))
            if l[i] == min(w_lo) and l[i] < max(w_lo):
                out.append(SwingPoint(index=i, price=l[i], is_high=False))
        out.sort(key=lambda s: s.index)
        return out

    @staticmethod
    def _last_swings(swings: list[SwingPoint], *, n: int, is_high: bool) -> list[SwingPoint]:
        pts = [s for s in swings if s.is_high == is_high]
        return pts[-n:] if len(pts) >= n else pts

    @staticmethod
    def _monotone_prices(pts: Sequence[SwingPoint], *, increasing: bool) -> bool:
        if len(pts) < 2:
            return False
        for a, b in zip(pts, pts[1:]):
            if increasing and not (b.price > a.price):
                return False
            if not increasing and not (b.price < a.price):
                return False
        return True

    @staticmethod
    def _merge_key_levels(levels: list[Decimal]) -> tuple[Decimal, ...]:
        if not levels:
            return ()
        sorted_lv = sorted(levels)
        merged: list[Decimal] = [sorted_lv[0]]
        for p in sorted_lv[1:]:
            if abs(p - merged[-1]) > _LEVEL_MERGE_EPS:
                merged.append(p)
        return tuple(merged)

    def detect(self, highs: Sequence[Decimal], lows: Sequence[Decimal], closes: Sequence[Decimal]) -> StructureResult:
        """Infere tendencia (HH/HL vs LH/LL), rompimientos simples y niveles clave.

        Heurística:
            * **UP**: últimos swings altos crecientes (HH) y swings bajos crecientes (HL).
            * **DOWN**: swings altos decrecientes (LH) y swings bajos decrecientes (LL).
            * **RANGE**: mezcla o datos insuficientes; puede pasar a UP/DOWN por **rompimiento**
              del último cierre respecto al último swing alto/bajo.
            * **Pullbacks básicos**: los últimos swings bajos (en tendencia alcista) y altos
              (en bajista) se incluyen en ``key_levels`` como zonas de reacción aproximadas.

        Args:
            highs, lows, closes: Misma longitud; ``closes[-1]`` es el precio actual.

        Returns:
            ``StructureResult`` con ``trend`` y ``key_levels`` ordenados de menor a mayor.
        """
        h = [Decimal(x) for x in highs]
        l = [Decimal(x) for x in lows]
        c = [Decimal(x) for x in closes]
        n = len(h)
        if n == 0 or n != len(l) or n != len(c):
            logger.warning("detect: longitudes inconsistentes o vacías")
            return StructureResult(trend=Trend.RANGE, key_levels=())

        swings = self.find_swings(h, l)
        last_close = c[-1]

        swing_highs = self._last_swings(swings, n=4, is_high=True)
        swing_lows = self._last_swings(swings, n=4, is_high=False)

        trend = Trend.RANGE
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh_hl = self._monotone_prices(swing_highs[-2:], increasing=True) and self._monotone_prices(
                swing_lows[-2:],
                increasing=True,
            )
            lh_ll = self._monotone_prices(swing_highs[-2:], increasing=False) and self._monotone_prices(
                swing_lows[-2:],
                increasing=False,
            )
            if hh_hl and not lh_ll:
                trend = Trend.UP
            elif lh_ll and not hh_hl:
                trend = Trend.DOWN
            elif hh_hl and lh_ll:
                win = min(self._conflict_bars, n - 1)
                if win >= 5:
                    base = c[-1 - win]
                    if base != 0:
                        net = (c[-1] - base) / abs(base)
                        if net > self._conflict_eps:
                            trend = Trend.UP
                        elif net < -self._conflict_eps:
                            trend = Trend.DOWN
                        else:
                            trend = Trend.RANGE
                    else:
                        trend = Trend.RANGE
                else:
                    trend = Trend.RANGE

        # Rompimiento simple respecto al último swing significativo
        if swing_highs and last_close > swing_highs[-1].price:
            if trend == Trend.RANGE:
                trend = Trend.UP
        if swing_lows and last_close < swing_lows[-1].price:
            if trend == Trend.RANGE:
                trend = Trend.DOWN

        # Rompimiento contrario a la estructura previa (cierre vs último swing)
        if trend == Trend.UP and swing_lows and last_close < swing_lows[-1].price:
            trend = Trend.DOWN
        elif trend == Trend.DOWN and swing_highs and last_close > swing_highs[-1].price:
            trend = Trend.UP

        if trend == Trend.RANGE and self._momentum_fb > 0 and n > self._momentum_fb:
            k = min(self._momentum_fb, n - 1)
            base = c[-1 - k]
            if base != 0:
                ret = (c[-1] - base) / abs(base)
                if ret >= self._momentum_min_ret:
                    trend = Trend.UP
                elif ret <= -self._momentum_min_ret:
                    trend = Trend.DOWN

        # Niveles aproximados: últimos swings + último cierre (pullback / reacción)
        level_candidates: list[Decimal] = []
        for s in swing_highs[-3:]:
            level_candidates.append(s.price)
        for s in swing_lows[-3:]:
            level_candidates.append(s.price)
        level_candidates.append(last_close)
        key_levels = self._merge_key_levels(level_candidates)

        range_high: Decimal | None = None
        range_low: Decimal | None = None
        position_in_range: float | None = None
        range_width_pct: float | None = None
        if swing_highs and swing_lows and last_close != 0:
            rh = max(s.price for s in swing_highs[-3:])
            rl = min(s.price for s in swing_lows[-3:])
            if rh > rl:
                range_high, range_low = rh, rl
                denom = rh - rl
                position_in_range = float(max(0.0, min(1.0, float((last_close - rl) / denom))))
                range_width_pct = float((rh - rl) / abs(last_close))

        logger.debug(
            "detect: trend=%s swings=%s levels=%s pos_in_range=%s",
            trend.value,
            len(swings),
            len(key_levels),
            position_in_range,
        )
        return StructureResult(
            trend=trend,
            key_levels=key_levels,
            range_high=range_high,
            range_low=range_low,
            position_in_range=position_in_range,
            range_width_pct=range_width_pct,
        )

"""Detección de tramos históricos con movimiento porcentual fuerte (impulsos)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Sequence

from analytics.logger import get_logger
from strategy.structure_detector import StructureDetector, SwingPoint

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ImpulseLeg:
    """Tramo entre dos pivotes con retorno porcentual significativo."""

    direction: Literal["UP", "DOWN"]
    start_time: datetime
    end_time: datetime
    return_pct: float
    start_price: float
    end_price: float


def _swings(highs: Sequence[float], lows: Sequence[float]) -> list[SwingPoint]:
    det = StructureDetector(left_bars=2, right_bars=2)
    hd = [Decimal(str(x)) for x in highs]
    ld = [Decimal(str(x)) for x in lows]
    return det.find_swings(hd, ld)


def scan_impulse_legs(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    bar_starts: Sequence[datetime],
    *,
    min_leg_return_pct: float = 2.5,
    max_legs: int = 8,
) -> list[ImpulseLeg]:
    """Localiza impulsos recientes usando pivotes fractales entre extremos de vela.

    Args:
        highs, lows, closes: OHLC alineados (misma longitud).
        bar_starts: Inicio UTC de cada vela (misma longitud).
        min_leg_return_pct: Umbral mínimo de |Δprecio| / precio_inicio * 100.
        max_legs: Máximo de piernas por dirección a devolver (las más grandes primero).

    Returns:
        Lista ordenada por ``return_pct`` descendente (mayor movimiento primero).
    """
    n = len(closes)
    if n < 8 or len(highs) != n or len(lows) != n or len(bar_starts) != n:
        return []

    hf = [float(x) for x in highs]
    lf = [float(x) for x in lows]
    swings = _swings(hf, lf)
    if len(swings) < 2:
        return []

    legs: list[ImpulseLeg] = []
    for i, s0 in enumerate(swings[:-1]):
        s1 = swings[i + 1]
        if s0.index >= n or s1.index >= n:
            continue
        t0 = bar_starts[s0.index]
        t1 = bar_starts[s1.index]
        if s0.is_high and not s1.is_high:
            # Pico → valle (bajada)
            p0 = float(s0.price)
            p1 = float(s1.price)
            if p0 <= 0:
                continue
            rp = (p0 - p1) / p0 * 100.0
            if rp >= min_leg_return_pct:
                legs.append(
                    ImpulseLeg("DOWN", t0, t1, float(rp), p0, p1),
                )
        elif not s0.is_high and s1.is_high:
            # Valle → pico (subida)
            p0 = float(s0.price)
            p1 = float(s1.price)
            if p0 <= 0:
                continue
            rp = (p1 - p0) / p0 * 100.0
            if rp >= min_leg_return_pct:
                legs.append(
                    ImpulseLeg("UP", t0, t1, float(rp), p0, p1),
                )

    legs.sort(key=lambda x: x.return_pct, reverse=True)
    if max_legs > 0:
        legs = legs[:max_legs]
    logger.debug("scan_impulse_legs: %s piernas (min=%.2f%%)", len(legs), min_leg_return_pct)
    return legs


def format_impulse_summary(legs: Sequence[ImpulseLeg], *, label: str = "") -> str:
    """Texto compacto para logs (fechas ISO en UTC)."""
    if not legs:
        return f"{label} sin impulsos >= umbral" if label else "sin impulsos >= umbral"
    parts: list[str] = []
    for leg in legs[:5]:
        parts.append(
            f"{leg.direction} +{leg.return_pct:.1f}% "
            f"({leg.start_time.isoformat()} → {leg.end_time.isoformat()})",
        )
    prefix = f"{label}: " if label else ""
    return prefix + "; ".join(parts)

"""AURION X 4.1 SMC — Detector de swing points (HH/HL/LH/LL + EqH/EqL).

Algoritmo: pivote por confirmación (``left_bars``/``right_bars``). Para cada
barra ``i`` se exige que sea high (o low) estricto frente a ``left``/``right``
vecinos. Sobre la secuencia ordenada de pivotes etiquetamos:

* ``HH``: high > high anterior
* ``LH``: high < high anterior
* ``HL``: low > low anterior
* ``LL``: low < low anterior

Equal Highs / Equal Lows (``EQH`` / ``EQL``) se marcan cuando dos pivotes del
mismo tipo difieren menos de ``equal_tolerance_atr * ATR`` — es la materia
prima del Liquidity Engine y del Stop Hunt detector.

El detector es puro: recibe arrays, devuelve lista. No mantiene estado.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from .types import SwingKind, SwingPoint


def _atr_local(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> float:
    n = len(closes)
    if n < 2 or len(highs) != n or len(lows) != n:
        return 0.0
    trs: list[float] = []
    for i in range(1, n):
        h, l = float(highs[i]), float(lows[i])
        cp = float(closes[i - 1])
        trs.append(max(h - l, abs(h - cp), abs(l - cp)))
    if not trs:
        return 0.0
    if len(trs) < period:
        return sum(trs) / len(trs)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def detect_swings(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    times: Sequence[datetime] | None = None,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
    equal_tolerance_atr: float = 0.10,
    atr_period: int = 14,
) -> list[SwingPoint]:
    """Detecta swings con clasificación HH/HL/LH/LL/EqH/EqL.

    Args:
        highs/lows/closes: series alineadas (mismo largo).
        times: tiempos UTC alineados. Si es ``None`` rellenamos con ``utcnow``
            (sólo afecta a logs/visual, no a la lógica).
        left_bars/right_bars: nº de barras a izquierda y derecha exigidas para
            confirmar un pivote (estricto >/<).
        equal_tolerance_atr: rango (en ATR) para considerar dos pivotes "iguales"
            (genera EQH/EQL).
        atr_period: ventana del ATR local.

    Returns:
        Lista de ``SwingPoint`` ordenada por índice ascendente. Si la serie es
        demasiado corta devuelve lista vacía.
    """
    n = len(closes)
    if n == 0:
        return []
    if len(highs) != n or len(lows) != n:
        return []
    if n < (left_bars + right_bars + 1):
        return []
    if times is None:
        ts_fallback = datetime.now(timezone.utc)
        times = [ts_fallback] * n
    elif len(times) != n:
        # Padding/truncate seguro: caemos a now() para los huecos.
        times = list(times)
        if len(times) < n:
            pad = datetime.now(timezone.utc)
            times = times + [pad] * (n - len(times))
        else:
            times = times[-n:]

    atr = _atr_local(highs, lows, closes, period=atr_period)
    tol = max(atr * float(equal_tolerance_atr), 1e-9)

    raw_pivots: list[tuple[int, bool, float]] = []  # (idx, is_high, price)
    for i in range(left_bars, n - right_bars):
        h_i = float(highs[i])
        l_i = float(lows[i])
        is_high = True
        for j in range(1, left_bars + 1):
            if not (h_i > float(highs[i - j])):
                is_high = False
                break
        if is_high:
            for j in range(1, right_bars + 1):
                if not (h_i > float(highs[i + j])):
                    is_high = False
                    break
        if is_high:
            raw_pivots.append((i, True, h_i))
            continue
        is_low = True
        for j in range(1, left_bars + 1):
            if not (l_i < float(lows[i - j])):
                is_low = False
                break
        if is_low:
            for j in range(1, right_bars + 1):
                if not (l_i < float(lows[i + j])):
                    is_low = False
                    break
        if is_low:
            raw_pivots.append((i, False, l_i))

    if not raw_pivots:
        return []

    raw_pivots.sort(key=lambda x: x[0])

    swings: list[SwingPoint] = []
    last_high_price: float | None = None
    last_low_price: float | None = None
    for idx, is_high, price in raw_pivots:
        if is_high:
            if last_high_price is None:
                kind = SwingKind.HH  # primer high marcado como HH por convención
            elif price > last_high_price + tol:
                kind = SwingKind.HH
            elif price < last_high_price - tol:
                kind = SwingKind.LH
            else:
                kind = SwingKind.EQH
            last_high_price = price
        else:
            if last_low_price is None:
                kind = SwingKind.LL
            elif price < last_low_price - tol:
                kind = SwingKind.LL
            elif price > last_low_price + tol:
                kind = SwingKind.HL
            else:
                kind = SwingKind.EQL
            last_low_price = price
        strength = 0.0
        if atr > 0 and len(swings) > 0:
            # Strength = magnitud del swing dividido por ATR (cap a 3 sigmas).
            ref_idx = swings[-1].index
            ref_price = swings[-1].price
            magnitude = abs(price - ref_price)
            strength = max(0.0, min(1.0, magnitude / (3.0 * atr)))
        elif atr > 0:
            strength = 0.5
        swings.append(
            SwingPoint(
                index=idx,
                price=price,
                time=times[idx],
                kind=kind,
                is_high=is_high,
                strength=strength,
            ),
        )
    return swings


__all__ = ["detect_swings"]

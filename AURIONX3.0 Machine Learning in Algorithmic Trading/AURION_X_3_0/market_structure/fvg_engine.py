"""AURION X 4.1 SMC — Fair Value Gap (FVG) Engine.

Detecta imbalances de 3 velas (i-1, i, i+1):

* **Bullish FVG**: ``low[i+1] > high[i-1]``. La zona [high[i-1], low[i+1]] es
  un hueco alcista no rellenado por la vela central.
* **Bearish FVG**: ``high[i+1] < low[i-1]``. Zona [high[i+1], low[i-1]].

Estados (lifecycle):

* ``OPEN``: ningún precio posterior entró en la zona.
* ``PARTIAL``: precio entró parcialmente (``fill_percent`` ∈ (0, 1)).
* ``FILLED``: precio cubrió la zona completa.
* ``INVERTED`` (IFVG): el FVG fue rebasado por completo y ahora el precio
  opera del otro lado; pasa a actuar como zona contraria.

Strength = ``gap_size_atr * freshness * (1 - fill_pct)``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from .swing_detector import _atr_local
from .types import FairValueGap, FVGState, StructureDirection


def detect_fvgs(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    times: Sequence[datetime] | None = None,
    *,
    timeframe: str = "M15",
    min_size_atr: float = 0.20,
    max_gaps: int = 16,
    keep_filled: bool = False,
    pip_size: float = 0.10,
) -> list[FairValueGap]:
    """Devuelve los FVGs relevantes (no totalmente rellenados ni invalidados, salvo IFVG).

    Args:
        highs/lows/closes: serie OHLC alineada.
        times: tiempos UTC alineados (opcional).
        timeframe: etiqueta del TF.
        min_size_atr: tamaño mínimo del gap como fracción del ATR.
        max_gaps: cap de FVGs devueltos (más recientes y de mayor strength).
        keep_filled: si ``True``, incluye también gaps con ``FILLED`` (útil
            para análisis histórico; default ``False``).
        pip_size: tamaño del pip para reportar ``gap_size_pips``.

    Returns:
        Lista de ``FairValueGap``.
    """
    n = len(closes)
    if n < 3:
        return []
    if len(highs) != n or len(lows) != n:
        return []
    atr = _atr_local(highs, lows, closes)
    if atr <= 0:
        return []
    if times is None:
        ts_fallback = datetime.now(timezone.utc)
        times_seq: list[datetime] = [ts_fallback] * n
    elif len(times) != n:
        ts_fallback = datetime.now(timezone.utc)
        times_seq = list(times) + [ts_fallback] * max(0, n - len(times))
        times_seq = times_seq[-n:]
    else:
        times_seq = list(times)
    min_gap = min_size_atr * atr

    out: list[FairValueGap] = []
    for i in range(1, n - 1):
        h_prev = float(highs[i - 1])
        l_prev = float(lows[i - 1])
        h_next = float(highs[i + 1])
        l_next = float(lows[i + 1])
        # Bullish FVG: l_next > h_prev
        if l_next > h_prev + min_gap:
            top = l_next
            bottom = h_prev
            direction = StructureDirection.BULLISH
        # Bearish FVG: h_next < l_prev
        elif h_next < l_prev - min_gap:
            top = l_prev
            bottom = h_next
            direction = StructureDirection.BEARISH
        else:
            continue

        gap_size = top - bottom
        gap_size_pips = gap_size / max(pip_size, 1e-9)

        # Lifecycle a partir de i+2.
        state = FVGState.OPEN
        fill_pct = 0.0
        inverted_at: datetime | None = None
        for j in range(i + 2, n):
            cj = float(closes[j])
            hj = float(highs[j])
            lj = float(lows[j])
            # Intersección con la zona
            overlap_top = min(hj, top)
            overlap_bot = max(lj, bottom)
            if overlap_top > overlap_bot:
                penetration = overlap_top - overlap_bot
                pct = penetration / max(gap_size, 1e-9)
                fill_pct = max(fill_pct, min(1.0, pct))
                if fill_pct >= 0.999:
                    state = FVGState.FILLED
                else:
                    state = FVGState.PARTIAL
            # Inversion: el precio cierra del otro lado del gap completo.
            if direction == StructureDirection.BULLISH and cj < bottom:
                state = FVGState.INVERTED
                inverted_at = times_seq[j]
                break
            if direction == StructureDirection.BEARISH and cj > top:
                state = FVGState.INVERTED
                inverted_at = times_seq[j]
                break
            if state == FVGState.FILLED and not keep_filled:
                break

        gap_age = n - 1 - i
        gap_atr = gap_size / max(atr, 1e-9)
        gap_quality = max(0.0, min(1.0, 0.7 * min(gap_atr / 1.5, 1.0) + 0.3 * (1.0 - fill_pct)))
        freshness = 1.0 if state == FVGState.OPEN else (0.6 if state == FVGState.PARTIAL else 0.0)
        fvg_strength = max(0.0, min(1.0, min(gap_atr / 1.5, 1.0) * (1.0 - fill_pct) * (0.4 + 0.6 * freshness)))

        if state == FVGState.FILLED and not keep_filled and state != FVGState.INVERTED:
            continue  # ya no aporta a confluencia (a menos que pidas histórico).

        out.append(
            FairValueGap(
                timeframe=timeframe,
                direction=direction,
                top=top,
                bottom=bottom,
                open_time=times_seq[i],
                open_index=i,
                state=state,
                gap_size=gap_size,
                gap_size_pips=gap_size_pips,
                gap_quality=gap_quality,
                gap_age_bars=gap_age,
                gap_fill_percent=fill_pct,
                fvg_strength=fvg_strength,
                inverted_at=inverted_at,
                extra={
                    "gap_atr": gap_atr,
                    "freshness_bucket": freshness,
                },
            ),
        )

    # Sort por proximidad al precio actual y luego por strength.
    last_close = float(closes[-1])
    def _sort_key(fvg: FairValueGap) -> tuple[float, float]:
        dist = abs(last_close - fvg.mid) / max(atr, 1e-9)
        return (dist, -fvg.fvg_strength)
    out.sort(key=_sort_key)
    return out[:max_gaps]


__all__ = ["detect_fvgs"]

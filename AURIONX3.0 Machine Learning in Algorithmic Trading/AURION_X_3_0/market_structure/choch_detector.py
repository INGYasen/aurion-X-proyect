"""AURION X 4.1 SMC — Change of Character (CHOCH).

Un CHOCH confirma **cambio de sesgo**:

* Sesgo alcista (HH/HL) → vela cierra por debajo del último HL relevante ⇒
  CHOCH bajista.
* Sesgo bajista (LL/LH) → vela cierra por encima del último LH relevante ⇒
  CHOCH alcista.

Diferenciamos:

* ``MAJOR``: el swing roto es el más reciente HL/LH del impulso macro
  (típicamente con displacement >= 1 ATR).
* ``MINOR``: rupturas internas (displacement < 1 ATR) que indican debilidad
  pero no necesariamente reversión.

El detector filtra micro-rupturas (vela con cuerpo < ``min_body_atr * ATR``)
para eliminar el ruido típico del oro en sesión asiática.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from .swing_detector import _atr_local, detect_swings
from .types import CHOCHEvent, CHOCHKind, StructureDirection, SwingKind, SwingPoint


def _last_macro_low(swings: list[SwingPoint]) -> SwingPoint | None:
    """En sesgo alcista, el HL más reciente (su ruptura ⇒ CHOCH bajista)."""
    for s in reversed(swings):
        if (not s.is_high) and s.kind == SwingKind.HL:
            return s
    return None


def _last_macro_high(swings: list[SwingPoint]) -> SwingPoint | None:
    """En sesgo bajista, el LH más reciente (su ruptura ⇒ CHOCH alcista)."""
    for s in reversed(swings):
        if s.is_high and s.kind == SwingKind.LH:
            return s
    return None


def _structure_bias(swings: list[SwingPoint]) -> StructureDirection:
    if not swings:
        return StructureDirection.NEUTRAL
    last_hh = next((s for s in reversed(swings) if s.kind == SwingKind.HH), None)
    last_hl = next((s for s in reversed(swings) if s.kind == SwingKind.HL), None)
    last_ll = next((s for s in reversed(swings) if s.kind == SwingKind.LL), None)
    last_lh = next((s for s in reversed(swings) if s.kind == SwingKind.LH), None)
    if last_hh and last_hl:
        if (last_ll is None) or (last_hh.index > last_ll.index):
            return StructureDirection.BULLISH
    if last_ll and last_lh:
        if (last_hh is None) or (last_ll.index > last_hh.index):
            return StructureDirection.BEARISH
    return StructureDirection.NEUTRAL


def detect_choch(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    times: Sequence[datetime] | None = None,
    *,
    timeframe: str = "M15",
    left_bars: int = 2,
    right_bars: int = 2,
    major_displacement_atr: float = 1.0,
    min_displacement_atr: float = 0.20,
    min_body_atr: float = 0.18,
    quality_followthrough_bars: int = 3,
) -> CHOCHEvent | None:
    """Detecta el último CHOCH confirmado.

    Args:
        highs/lows/closes: serie OHLC alineada.
        times: tiempos UTC alineados (opcional).
        timeframe: etiqueta libre (no afecta la lógica).
        left_bars/right_bars: confirmación de swings.
        major_displacement_atr: ≥ este valor ⇒ CHOCH ``MAJOR``.
        min_displacement_atr: por debajo descartamos como ruido.
        min_body_atr: cuerpo mínimo (en ATR) de la vela que rompe; filtra
            "wick fakes" de sesión asiática.
        quality_followthrough_bars: nº de velas para medir follow-through.

    Returns:
        ``CHOCHEvent`` reciente o ``None``.
    """
    n = len(closes)
    if n < (left_bars + right_bars + 4):
        return None
    swings = detect_swings(
        highs,
        lows,
        closes,
        times=times,
        left_bars=left_bars,
        right_bars=right_bars,
    )
    if not swings:
        return None
    atr = _atr_local(highs, lows, closes)
    if atr <= 0:
        return None

    bias = _structure_bias(swings)
    # Buscamos el CHOCH **más reciente confirmado** en cualquier dirección.
    # Semántica: un CHOCH bajista es la ruptura por debajo de un HL anclando
    # un giro bajista (ocurre durante o tras una fase alcista). Un CHOCH
    # alcista es ruptura por encima de un LH durante una fase bajista. La
    # función devuelve el evento más reciente sin depender del `bias` actual
    # (que es retrospectivo), porque el CHOCH es justamente lo que CAUSÓ ese
    # bias.
    hl_swings = [s for s in swings if (not s.is_high) and s.kind == SwingKind.HL]
    lh_swings = [s for s in swings if s.is_high and s.kind == SwingKind.LH]
    if not hl_swings and not lh_swings:
        return None

    window = max(40, 6 * (right_bars + left_bars) + 20)

    def _scan(anchors: list[SwingPoint], new_dir: StructureDirection) -> tuple[int | None, float, SwingPoint | None]:
        best_idx: int | None = None
        best_price = 0.0
        best_anchor: SwingPoint | None = None
        for anchor in reversed(anchors):
            start = anchor.index + right_bars + 1
            end = min(n, anchor.index + window + 1)
            for i in range(start, end):
                c = float(closes[i])
                prev_c = float(closes[i - 1]) if i > 0 else c
                body = abs(c - prev_c)
                if body < min_body_atr * atr:
                    continue
                if new_dir == StructureDirection.BEARISH and c < anchor.price:
                    if best_idx is None or i > best_idx:
                        best_idx = i
                        best_price = c
                        best_anchor = anchor
                    break
                if new_dir == StructureDirection.BULLISH and c > anchor.price:
                    if best_idx is None or i > best_idx:
                        best_idx = i
                        best_price = c
                        best_anchor = anchor
                    break
            if best_anchor is not None and best_anchor is anchor:
                # primer anchor que produjo break (más reciente del reverse)
                break
        return best_idx, best_price, best_anchor

    bear_idx, bear_price, bear_anchor = _scan(hl_swings, StructureDirection.BEARISH)
    bull_idx, bull_price, bull_anchor = _scan(lh_swings, StructureDirection.BULLISH)

    if bear_idx is None and bull_idx is None:
        return None
    # Nos quedamos con el CHOCH más reciente entre los dos sentidos.
    if bear_idx is not None and (bull_idx is None or bear_idx > bull_idx):
        new_direction = StructureDirection.BEARISH
        break_idx = bear_idx
        break_price = bear_price
        anchor_use = bear_anchor
    else:
        new_direction = StructureDirection.BULLISH
        break_idx = bull_idx  # type: ignore[assignment]
        break_price = bull_price
        anchor_use = bull_anchor
    assert anchor_use is not None and break_idx is not None
    swing_idx = anchor_use.index
    swing_price = anchor_use.price
    swing_time = anchor_use.time

    displacement = abs(break_price - swing_price)
    disp_atr = displacement / atr
    if disp_atr < min_displacement_atr:
        return None  # ruido
    kind = CHOCHKind.MAJOR if disp_atr >= major_displacement_atr else CHOCHKind.MINOR

    # Follow-through
    ft_n = 0
    ft_max = min(quality_followthrough_bars, n - break_idx - 1)
    for j in range(1, ft_max + 1):
        cj = float(closes[break_idx + j])
        if new_direction == StructureDirection.BULLISH and cj > swing_price:
            ft_n += 1
        elif new_direction == StructureDirection.BEARISH and cj < swing_price:
            ft_n += 1
    ft_score = ft_n / float(ft_max) if ft_max > 0 else 0.0

    # Cuerpo relativo: mayor cuerpo ⇒ más convicción.
    bar_open = float(closes[break_idx - 1]) if break_idx - 1 >= 0 else float(closes[break_idx])
    bar_body = abs(break_price - bar_open)
    body_score = max(0.0, min(1.0, bar_body / (1.2 * atr)))

    choch_strength = max(0.0, min(1.0, disp_atr / (major_displacement_atr * 1.25)))
    choch_quality = max(0.0, min(1.0, 0.55 * ft_score + 0.30 * choch_strength + 0.15 * body_score))
    choch_reliability = max(0.0, min(1.0, 0.55 * choch_strength + 0.45 * choch_quality))
    bonus_major = 0.10 if kind == CHOCHKind.MAJOR else 0.0
    choch_score = max(0.0, min(1.0, 0.6 * choch_reliability + 0.3 * choch_quality + bonus_major))

    choch_age_bars = n - 1 - break_idx
    break_time = times[break_idx] if (times is not None and len(times) == n) else swing_time

    return CHOCHEvent(
        timeframe=timeframe,
        direction=new_direction,
        kind=kind,
        broken_swing_price=swing_price,
        broken_swing_time=swing_time,
        break_time=break_time,
        break_price=break_price,
        choch_strength=choch_strength,
        choch_quality=choch_quality,
        choch_reliability=choch_reliability,
        choch_age_bars=choch_age_bars,
        choch_score=choch_score,
        extra={
            "displacement_atr": disp_atr,
            "followthrough_score": ft_score,
            "body_score": body_score,
            "structure_bias_before": bias.value,
        },
    )


__all__ = ["detect_choch"]

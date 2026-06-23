"""AURION X 4.1 SMC — Break of Structure (BOS).

Un BOS confirma **continuación** de la tendencia: un swing HH (en alcista) o
un swing LL (en bajista) previo es rebasado por una vela que cierra más allá
del nivel del swing roto.

Reglas:

* Para BOS alcista necesitamos un swing alcista previo (``HH`` o ``HL`` no es
  válido como nivel a romper — usamos siempre el último high relevante en una
  estructura alcista).
* Para BOS bajista: nivel = último low relevante en estructura bajista.
* La ruptura se mide por *cierre* de vela (no por mecha) para reducir false
  positives típicos en XAUUSD.
* ``bos_strength`` se calcula como ``displacement / (atr * k)``, ``k=2.5``,
  clipeado a [0,1].
* ``bos_quality`` mezcla strength, follow-through (n velas siguientes en
  dirección), y ausencia de wick contrario fuerte.
* ``bos_reliability = 0.6 * strength + 0.4 * quality``.

Devuelve sólo el **último** BOS reciente por timeframe (los anteriores se
persisten vía repositorio pero no se reusan para confluencia inmediata).
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from .swing_detector import _atr_local, detect_swings
from .types import BOSEvent, BOSStrength, StructureDirection, SwingKind, SwingPoint


def _last_valid_high_swing(swings: list[SwingPoint]) -> SwingPoint | None:
    for s in reversed(swings):
        if s.is_high and s.kind in (SwingKind.HH, SwingKind.LH):
            return s
    return None


def _last_valid_low_swing(swings: list[SwingPoint]) -> SwingPoint | None:
    for s in reversed(swings):
        if (not s.is_high) and s.kind in (SwingKind.LL, SwingKind.HL):
            return s
    return None


def _structure_bias(swings: list[SwingPoint]) -> StructureDirection:
    """Aproximación rápida del sesgo según últimas etiquetas."""
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


def detect_bos(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    times: Sequence[datetime] | None = None,
    *,
    timeframe: str = "M15",
    left_bars: int = 2,
    right_bars: int = 2,
    strong_displacement_atr: float = 1.0,
    weak_displacement_atr: float = 0.25,
    quality_followthrough_bars: int = 3,
    pip_size: float = 0.10,
) -> BOSEvent | None:
    """Detecta el último BOS confirmado en la serie.

    Args:
        highs/lows/closes: serie OHLC alineada.
        times: tiempos UTC alineados (opcional).
        timeframe: etiqueta libre (``"M15"``, ``"H1"``…), no afecta a la lógica.
        left_bars/right_bars: confirmación de swings.
        strong_displacement_atr: displacement (en ATR) por encima del cual se
            clasifica el BOS como STRONG.
        weak_displacement_atr: displacement por debajo del cual se descarta el
            BOS (devuelve ``None``).
        quality_followthrough_bars: nº de velas posteriores que se inspeccionan
            para medir follow-through (refuerza ``bos_quality``).
        pip_size: tamaño del pip para reportar la distancia en pips (XAUUSD ≈ 0.10).

    Returns:
        ``BOSEvent`` si hay BOS reciente confirmado, ``None`` si no.
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
    bias = _structure_bias(swings)
    atr = _atr_local(highs, lows, closes)
    if atr <= 0:
        return None

    # Estrategia: iteramos los swings desde el más reciente hacia atrás y
    # buscamos el BOS **más reciente confirmado** dentro de ``window_bars``
    # velas tras el swing. Esto evita devolver `None` cuando el "último swing"
    # aún no ha sido roto (ese swing es el techo actual, no el roto).
    window_bars = max(40, 6 * (right_bars + left_bars) + 20)

    candidate_swing: SwingPoint | None = None
    direction: StructureDirection | None = None
    break_idx: int | None = None
    break_price = 0.0

    # Generamos lista de candidatos (high_swings, low_swings) según bias.
    high_swings = [s for s in swings if s.is_high and s.kind in (SwingKind.HH, SwingKind.LH)]
    low_swings = [s for s in swings if (not s.is_high) and s.kind in (SwingKind.LL, SwingKind.HL)]

    def _find_break(direction_local: StructureDirection, sw_list: list[SwingPoint]) -> tuple[SwingPoint | None, int | None, float]:
        # Recorremos del más reciente al más antiguo.
        for sw in reversed(sw_list):
            start = sw.index + right_bars + 1
            end = min(n, sw.index + window_bars + 1)
            for k in range(start, end):
                c_k = float(closes[k])
                if direction_local == StructureDirection.BULLISH and c_k > sw.price:
                    return (sw, k, c_k)
                if direction_local == StructureDirection.BEARISH and c_k < sw.price:
                    return (sw, k, c_k)
        return (None, None, 0.0)

    # Prioridad por bias detectado, pero probamos ambos sentidos para no
    # perder un BOS opuesto reciente (caso típico durante CHOCH).
    tried: list[StructureDirection] = []
    if bias == StructureDirection.BULLISH:
        tried = [StructureDirection.BULLISH, StructureDirection.BEARISH]
    elif bias == StructureDirection.BEARISH:
        tried = [StructureDirection.BEARISH, StructureDirection.BULLISH]
    else:
        tried = [StructureDirection.BULLISH, StructureDirection.BEARISH]

    best: tuple[SwingPoint | None, int | None, float, StructureDirection] = (None, None, 0.0, StructureDirection.NEUTRAL)
    for dirc in tried:
        sw_list = high_swings if dirc == StructureDirection.BULLISH else low_swings
        sw, k, price = _find_break(dirc, sw_list)
        if sw is not None and k is not None:
            # Nos quedamos con el break más reciente entre los dos sentidos.
            if best[1] is None or k > best[1]:
                best = (sw, k, price, dirc)

    candidate_swing, break_idx, break_price, direction = best
    if candidate_swing is None or break_idx is None or direction == StructureDirection.NEUTRAL:
        return None
    swing_idx = candidate_swing.index
    swing_price = candidate_swing.price
    swing_time = candidate_swing.time

    displacement = abs(break_price - swing_price)
    disp_atr = displacement / atr
    if disp_atr < weak_displacement_atr:
        return None  # ruido, no BOS válido.
    if disp_atr >= strong_displacement_atr:
        strength_cls = BOSStrength.STRONG
    elif disp_atr >= 0.5:
        strength_cls = BOSStrength.STANDARD
    else:
        strength_cls = BOSStrength.WEAK

    # Follow-through quality: ¿cuántas de las siguientes velas continúan?
    ft_n = 0
    ft_max = min(quality_followthrough_bars, n - break_idx - 1)
    for j in range(1, ft_max + 1):
        cj = float(closes[break_idx + j])
        if direction == StructureDirection.BULLISH and cj > break_price * 0.9995:
            ft_n += 1
        elif direction == StructureDirection.BEARISH and cj < break_price * 1.0005:
            ft_n += 1
    ft_score = ft_n / float(ft_max) if ft_max > 0 else 0.0

    # Wick contrario: si la vela de break tiene wick contrario > 50% del cuerpo
    # bajamos quality (institucional débil).
    bar_high = float(highs[break_idx])
    bar_low = float(lows[break_idx])
    bar_open = float(closes[break_idx - 1]) if break_idx - 1 >= 0 else float(closes[break_idx])
    body = max(abs(break_price - bar_open), 1e-9)
    if direction == StructureDirection.BULLISH:
        opposite_wick = max(0.0, bar_open - bar_low) if break_price > bar_open else max(0.0, break_price - bar_low)
    else:
        opposite_wick = max(0.0, bar_high - bar_open) if break_price < bar_open else max(0.0, bar_high - break_price)
    wick_penalty = min(1.0, opposite_wick / max(body, 1e-9))

    bos_strength = max(0.0, min(1.0, disp_atr / (strong_displacement_atr * 1.25)))
    bos_quality = max(0.0, min(1.0, 0.55 * ft_score + 0.30 * bos_strength + 0.15 * (1.0 - wick_penalty)))
    bos_reliability = max(0.0, min(1.0, 0.6 * bos_strength + 0.4 * bos_quality))

    bos_age_bars = n - 1 - break_idx
    bos_distance_pips = displacement / max(pip_size, 1e-9)

    break_time = times[break_idx] if (times is not None and len(times) == n) else swing_time

    return BOSEvent(
        timeframe=timeframe,
        direction=direction,
        strength=strength_cls,
        broken_swing_price=swing_price,
        broken_swing_time=swing_time,
        break_time=break_time,
        break_price=break_price,
        bos_strength=bos_strength,
        bos_quality=bos_quality,
        bos_age_bars=bos_age_bars,
        bos_distance_pips=bos_distance_pips,
        bos_reliability=bos_reliability,
        extra={
            "displacement_atr": disp_atr,
            "followthrough_score": ft_score,
            "wick_penalty": wick_penalty,
            "structure_bias_before": bias.value,
        },
    )


__all__ = ["detect_bos"]

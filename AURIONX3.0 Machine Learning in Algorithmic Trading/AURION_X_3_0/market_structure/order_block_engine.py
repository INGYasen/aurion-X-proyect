"""AURION X 4.1 SMC — Order Block Engine.

Definición operativa que usamos (alineada con ICT/SMC institucional clásica):

* **Bullish OB**: última *vela bajista* (close < open) inmediatamente anterior
  a un displacement alcista (vela siguiente cierra muy por encima del high
  bajista). La zona OB cubre ``[low_bajista, high_bajista]``.
* **Bearish OB**: simétrico: última *vela alcista* antes de un displacement
  bajista. Zona: ``[low_alcista, high_alcista]``.

Estados (lifecycle):

* ``FRESH``: la zona no ha sido tocada por price action posterior.
* ``MITIGATED``: el precio entró en la zona pero salió en la dirección esperada
  (reacción válida).
* ``INVALIDATED``: el cierre rompe la zona en dirección contraria (la OB ya
  no es válida; se mantiene en historial pero no entra a confluencia).

Scores (todos en ``[0,1]``):

* ``ob_volume_score``: volumen relativo de la vela OB vs media de las N
  anteriores. Sin volumen → 0.5 (neutro).
* ``ob_displacement_score``: tamaño del displacement (vela siguiente) en ATR.
* ``ob_reaction_score``: si ya hay reacción posterior (vela toca zona y vuelve
  a salir en dir. esperada), reaction_score → magnitud de la reacción.
* ``ob_freshness``: decae 1.0 → 0.5 al primer toque parcial, 0.5 → 0.0 si
  mitigada completamente.
* ``ob_quality`` = combinación final (definida abajo).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from .swing_detector import _atr_local
from .types import OBState, OrderBlock, StructureDirection


def _bar_open_proxy(closes: Sequence[float], i: int) -> float:
    """Aproxima open de la barra i como close de la i-1 (los detectores trabajan
    con close-only para coherencia con el MultiTimeframeEngine; los detectores
    de OB usan high/low/close — el open es derivado para clasificación
    alcista/bajista)."""
    if i <= 0:
        return float(closes[i])
    return float(closes[i - 1])


def detect_order_blocks(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float] | None = None,
    times: Sequence[datetime] | None = None,
    *,
    timeframe: str = "M15",
    displacement_atr: float = 1.0,
    max_blocks: int = 12,
    fresh_only_in_active: bool = False,
    pip_size: float = 0.10,
    invalidation_close_buffer_atr: float = 0.05,
) -> list[OrderBlock]:
    """Devuelve las OB activas relevantes (FRESH/MITIGATED, ordenadas por proximidad/quality).

    Args:
        highs/lows/closes: serie OHLC alineada.
        volumes: serie de tick-volume alineada (opcional).
        times: tiempos UTC alineados (opcional).
        timeframe: etiqueta del TF.
        displacement_atr: la vela siguiente al candidato debe desplazarse al
            menos esta fracción de ATR para que la OB sea válida.
        max_blocks: cap de OBs activas devueltas (las más recientes y de mayor
            quality).
        fresh_only_in_active: si ``True``, descarta MITIGATED del resultado
            (mantenemos por defecto ``False`` para confluencia institucional).
        pip_size: tamaño del pip (XAUUSD ≈ 0.10).
        invalidation_close_buffer_atr: tolerancia extra (en ATR) al evaluar
            invalidación por cierre del lado contrario.

    Returns:
        Lista de ``OrderBlock`` (puede estar vacía si no hay OB válidas).
    """
    n = len(closes)
    if n < 6:
        return []
    if len(highs) != n or len(lows) != n:
        return []
    use_vol = volumes is not None and len(volumes) == n
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

    # 1) Detectar candidatos: para cada i en [1, n-2], evalúa si (i) es la OB
    #    y (i+1) la vela de displacement.
    raw: list[OrderBlock] = []
    invalidation_buffer = float(invalidation_close_buffer_atr) * atr
    for i in range(1, n - 1):
        h_i = float(highs[i])
        l_i = float(lows[i])
        c_i = float(closes[i])
        o_i = _bar_open_proxy(closes, i)
        c_next = float(closes[i + 1])
        h_next = float(highs[i + 1])
        l_next = float(lows[i + 1])
        # Displacement: vela i+1 cierra más allá del lado opuesto de la vela i.
        disp_bull = max(0.0, c_next - h_i)
        disp_bear = max(0.0, l_i - c_next)

        if disp_bull > displacement_atr * atr and o_i > c_i:
            # Bullish OB: vela bajista (o_i > c_i) antes de displacement alcista.
            direction = StructureDirection.BULLISH
            top = h_i
            bottom = l_i
        elif disp_bear > displacement_atr * atr and o_i < c_i:
            # Bearish OB: vela alcista antes de displacement bajista.
            direction = StructureDirection.BEARISH
            top = h_i
            bottom = l_i
        else:
            continue

        # Volume score
        vol_score = 0.5
        if use_vol:
            window = max(5, min(20, i))
            v_now = float(volumes[i])  # type: ignore[index]
            v_ref = [float(volumes[j]) for j in range(max(0, i - window), i)]  # type: ignore[index]
            v_mean = sum(v_ref) / max(len(v_ref), 1) if v_ref else v_now
            if v_mean > 0:
                ratio = v_now / v_mean
                # 1.0 => 0.5; 2.0 => 0.83; 3.0+ => ~1.0
                vol_score = max(0.0, min(1.0, ratio / 3.0))

        # Displacement score
        displacement = disp_bull if direction == StructureDirection.BULLISH else disp_bear
        disp_score = max(0.0, min(1.0, displacement / (atr * 2.0)))

        # Lifecycle: recorrer velas posteriores y marcar mitigación / invalidación.
        state = OBState.FRESH
        freshness = 1.0
        reaction_score = 0.0
        mit_time: datetime | None = None
        for j in range(i + 2, n):
            hj = float(highs[j])
            lj = float(lows[j])
            cj = float(closes[j])
            # Invalidación por cierre contrario
            if direction == StructureDirection.BULLISH and cj < bottom - invalidation_buffer:
                state = OBState.INVALIDATED
                mit_time = times_seq[j]
                freshness = 0.0
                break
            if direction == StructureDirection.BEARISH and cj > top + invalidation_buffer:
                state = OBState.INVALIDATED
                mit_time = times_seq[j]
                freshness = 0.0
                break
            # Toque parcial
            touched = (lj <= top and hj >= bottom)
            if touched and state == OBState.FRESH:
                state = OBState.MITIGATED
                freshness = 0.5
                mit_time = times_seq[j]
                # Medir reacción: en las velas siguientes (hasta 5) qué tan rápido salió.
                ahead = min(5, n - 1 - j)
                if direction == StructureDirection.BULLISH:
                    max_above = max((float(highs[j + k]) for k in range(1, ahead + 1)), default=hj)
                    reaction = max(0.0, max_above - top) / max(atr, 1e-9)
                else:
                    min_below = min((float(lows[j + k]) for k in range(1, ahead + 1)), default=lj)
                    reaction = max(0.0, bottom - min_below) / max(atr, 1e-9)
                reaction_score = max(reaction_score, max(0.0, min(1.0, reaction / 2.0)))
            elif touched and state == OBState.MITIGATED:
                # Cada nuevo toque acerca la OB a quemarse.
                freshness = max(0.0, freshness - 0.15)

        if fresh_only_in_active and state != OBState.FRESH:
            continue

        ob_age_bars = n - 1 - i
        # Quality compone los 4 scores con pesos cuasi-institucionales.
        quality = max(
            0.0,
            min(
                1.0,
                0.35 * disp_score
                + 0.25 * vol_score
                + 0.20 * reaction_score
                + 0.20 * freshness,
            ),
        )
        strength = max(0.0, min(1.0, 0.55 * disp_score + 0.45 * vol_score))
        raw.append(
            OrderBlock(
                timeframe=timeframe,
                direction=direction,
                top=top,
                bottom=bottom,
                open_time=times_seq[i],
                open_index=i,
                state=state,
                ob_strength=strength,
                ob_volume_score=vol_score,
                ob_displacement_score=disp_score,
                ob_reaction_score=reaction_score,
                ob_freshness=freshness,
                ob_age_bars=ob_age_bars,
                ob_quality=quality,
                mitigation_time=mit_time,
                extra={
                    "displacement_atr": displacement / atr,
                    "height_pips": (top - bottom) / max(pip_size, 1e-9),
                },
            ),
        )

    # Filtrar invalidadas y ordenar.
    active = [ob for ob in raw if ob.state != OBState.INVALIDATED]
    # Orden: primero por proximidad al precio actual (close[-1]) si está cerca,
    # luego por quality.
    last_close = float(closes[-1])
    def _sort_key(ob: OrderBlock) -> tuple[float, float]:
        # Distancia normalizada por ATR (más cercano = mejor).
        dist = abs(last_close - ob.mid) / max(atr, 1e-9)
        return (dist, -ob.ob_quality)
    active.sort(key=_sort_key)
    return active[:max_blocks]


__all__ = ["detect_order_blocks"]

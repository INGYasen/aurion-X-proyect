"""AURION X 4.1 SMC — Liquidity Engine.

Detecta:

* **SSL** (Sell-Side Liquidity): bajo los swing lows / equal lows recientes
  (donde están parados los stops de los compradores).
* **BSL** (Buy-Side Liquidity): sobre los swing highs / equal highs.
* **Equal Highs / Equal Lows**: clusters de pivotes con precios casi idénticos
  (tolerancia ATR-based).
* **Liquidity Pools**: clusters densos (≥ 3 toques) tratados como un único nivel
  con peso reforzado.
* **Stop Hunts (sweeps)**: vela que rompe el nivel y vuelve dentro del rango
  en la misma vela / siguiente con cuerpo significativo.

Salida: dos listas — ``pools: list[LiquidityPool]`` y ``sweeps: list[LiquiditySweep]``.

``liquidity_score`` (por pool) = combinación de density + proximidad + ATR-norm.
``stop_hunt_probability`` aumenta si el pool tiene equal_highs/lows confirmados
y el contexto reciente muestra wicks contra el nivel.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from .swing_detector import _atr_local, detect_swings
from .types import (
    LiquidityKind,
    LiquidityPool,
    LiquiditySweep,
    SwingKind,
    SwingPoint,
    SweepKind,
)


def _cluster_swings(
    swings: list[SwingPoint],
    *,
    is_high: bool,
    tolerance: float,
) -> list[list[SwingPoint]]:
    """Agrupa swings por precio cercano (tolerancia absoluta)."""
    pts = [s for s in swings if s.is_high == is_high]
    if not pts:
        return []
    pts_sorted = sorted(pts, key=lambda s: s.price)
    clusters: list[list[SwingPoint]] = []
    current: list[SwingPoint] = []
    last_price = -1e18
    for s in pts_sorted:
        if not current or abs(s.price - last_price) <= tolerance:
            current.append(s)
        else:
            clusters.append(current)
            current = [s]
        last_price = s.price
    if current:
        clusters.append(current)
    return clusters


def detect_liquidity(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    times: Sequence[datetime] | None = None,
    *,
    timeframe: str = "M15",
    left_bars: int = 2,
    right_bars: int = 2,
    equal_tolerance_atr: float = 0.15,
    pool_min_touches_dense: int = 3,
    sweep_window_bars: int = 3,
    sweep_min_excursion_atr: float = 0.35,
    sweep_min_reentry_pct: float = 0.55,
    max_pools: int = 16,
) -> tuple[list[LiquidityPool], list[LiquiditySweep]]:
    """Detecta pools de liquidez y sweeps recientes.

    Args:
        highs/lows/closes: serie OHLC alineada.
        times: tiempos UTC alineados.
        timeframe: etiqueta del TF.
        left_bars/right_bars: confirmación de swings.
        equal_tolerance_atr: tolerancia (en ATR) para considerar dos pivotes
            como "iguales".
        pool_min_touches_dense: nº de pivotes en un cluster para clasificarlo
            como ``POOL`` (denso).
        sweep_window_bars: nº de velas tras la barrida para evaluar si volvió
            dentro del rango (reentry).
        sweep_min_excursion_atr: amplitud mínima de la barrida (en ATR) para
            contarla.
        sweep_min_reentry_pct: % del excursion que el precio recupera para
            considerar un sweep válido.
        max_pools: cap de pools devueltos.

    Returns:
        Tupla ``(pools, sweeps)``.
    """
    n = len(closes)
    if n < (left_bars + right_bars + 4):
        return ([], [])
    atr = _atr_local(highs, lows, closes)
    if atr <= 0:
        return ([], [])
    if times is None:
        ts_fallback = datetime.now(timezone.utc)
        times_seq: list[datetime] = [ts_fallback] * n
    elif len(times) != n:
        ts_fallback = datetime.now(timezone.utc)
        times_seq = list(times) + [ts_fallback] * max(0, n - len(times))
        times_seq = times_seq[-n:]
    else:
        times_seq = list(times)

    swings = detect_swings(
        highs, lows, closes, times=times_seq,
        left_bars=left_bars, right_bars=right_bars,
        equal_tolerance_atr=equal_tolerance_atr,
    )
    if not swings:
        return ([], [])

    tol = max(atr * equal_tolerance_atr, 1e-9)
    high_clusters = _cluster_swings(swings, is_high=True, tolerance=tol)
    low_clusters = _cluster_swings(swings, is_high=False, tolerance=tol)

    last_close = float(closes[-1])
    pools: list[LiquidityPool] = []

    for cluster in high_clusters:
        if not cluster:
            continue
        prices = [s.price for s in cluster]
        bound_low = min(prices)
        bound_high = max(prices)
        price = max(prices)  # liquidez se acumula sobre el high
        touches = len(cluster)
        if touches >= pool_min_touches_dense:
            kind = LiquidityKind.POOL
        elif touches >= 2:
            kind = LiquidityKind.EQUAL_HIGHS
        else:
            kind = LiquidityKind.BSL
        density = max(0.0, min(1.0, (touches - 1) / 4.0))
        proximity = max(0.0, 1.0 - abs(last_close - price) / max(atr * 5.0, 1e-9))
        liquidity_score = max(
            0.0,
            min(1.0, 0.55 * density + 0.30 * proximity + 0.15),
        )
        last_pivot_idx = max(s.index for s in cluster)
        last_pivot_time = times_seq[last_pivot_idx] if last_pivot_idx < n else times_seq[-1]
        stop_hunt_prob = max(0.0, min(1.0, 0.35 + 0.35 * density + 0.30 * proximity))
        pools.append(
            LiquidityPool(
                timeframe=timeframe,
                kind=kind,
                price=price,
                bound_low=bound_low,
                bound_high=bound_high,
                open_time=last_pivot_time,
                open_index=last_pivot_idx,
                touches=touches,
                density=density,
                liquidity_score=liquidity_score,
                stop_hunt_probability=stop_hunt_prob,
                extra={
                    "side": "BUY_SIDE",
                    "atr_at_detect": atr,
                },
            ),
        )

    for cluster in low_clusters:
        if not cluster:
            continue
        prices = [s.price for s in cluster]
        bound_low = min(prices)
        bound_high = max(prices)
        price = min(prices)  # liquidez sobre el low
        touches = len(cluster)
        if touches >= pool_min_touches_dense:
            kind = LiquidityKind.POOL
        elif touches >= 2:
            kind = LiquidityKind.EQUAL_LOWS
        else:
            kind = LiquidityKind.SSL
        density = max(0.0, min(1.0, (touches - 1) / 4.0))
        proximity = max(0.0, 1.0 - abs(last_close - price) / max(atr * 5.0, 1e-9))
        liquidity_score = max(
            0.0,
            min(1.0, 0.55 * density + 0.30 * proximity + 0.15),
        )
        last_pivot_idx = max(s.index for s in cluster)
        last_pivot_time = times_seq[last_pivot_idx] if last_pivot_idx < n else times_seq[-1]
        stop_hunt_prob = max(0.0, min(1.0, 0.35 + 0.35 * density + 0.30 * proximity))
        pools.append(
            LiquidityPool(
                timeframe=timeframe,
                kind=kind,
                price=price,
                bound_low=bound_low,
                bound_high=bound_high,
                open_time=last_pivot_time,
                open_index=last_pivot_idx,
                touches=touches,
                density=density,
                liquidity_score=liquidity_score,
                stop_hunt_probability=stop_hunt_prob,
                extra={
                    "side": "SELL_SIDE",
                    "atr_at_detect": atr,
                },
            ),
        )

    # Marcar swept: si en las últimas velas se barrió el pool.
    sweeps: list[LiquiditySweep] = []
    for pool in pools:
        # Para cada pool: buscar en ventana posterior al último pivote del cluster
        # una vela que rompa el nivel y vuelva dentro del rango.
        start = pool.open_index + 1
        for i in range(start, n):
            hi = float(highs[i])
            lo = float(lows[i])
            ci = float(closes[i])
            sweep_kind: SweepKind | None = None
            excursion = 0.0
            if pool.kind in (LiquidityKind.BSL, LiquidityKind.EQUAL_HIGHS, LiquidityKind.POOL) and hi > pool.price:
                # Barrida arriba.
                excursion = hi - pool.price
                if excursion >= sweep_min_excursion_atr * atr:
                    # Reentry en la misma vela o las siguientes
                    reentry = 0.0
                    for k in range(0, min(sweep_window_bars + 1, n - i)):
                        ck = float(closes[i + k])
                        if ck < pool.price:
                            reentry = max(reentry, (hi - ck) / max(excursion, 1e-9))
                    if reentry >= sweep_min_reentry_pct:
                        sweep_kind = SweepKind.STOP_HUNT_UP
                    else:
                        sweep_kind = SweepKind.CLEAN_SWEEP
            if pool.kind in (LiquidityKind.SSL, LiquidityKind.EQUAL_LOWS, LiquidityKind.POOL) and lo < pool.price:
                excursion = pool.price - lo
                if excursion >= sweep_min_excursion_atr * atr:
                    reentry = 0.0
                    for k in range(0, min(sweep_window_bars + 1, n - i)):
                        ck = float(closes[i + k])
                        if ck > pool.price:
                            reentry = max(reentry, (ck - lo) / max(excursion, 1e-9))
                    if reentry >= sweep_min_reentry_pct:
                        sweep_kind = SweepKind.STOP_HUNT_DOWN
                    else:
                        sweep_kind = SweepKind.CLEAN_SWEEP

            if sweep_kind is not None:
                rej_strength = max(0.0, min(1.0, excursion / max(atr, 1e-9) / 1.5))
                sweep_strength = max(0.0, min(1.0, excursion / max(atr, 1e-9) / 2.0))
                sweep = LiquiditySweep(
                    timeframe=timeframe,
                    kind=sweep_kind,
                    pool_price=pool.price,
                    sweep_high=hi,
                    sweep_low=lo,
                    sweep_time=times_seq[i],
                    rejection_strength=rej_strength,
                    sweep_strength=sweep_strength,
                    extra={"excursion_atr": excursion / atr, "pool_kind": pool.kind.value},
                )
                sweeps.append(sweep)
                pool.swept = True
                pool.swept_at = sweep.sweep_time
                pool.sweep_kind = sweep_kind
                break  # un sweep por pool

    pools.sort(key=lambda p: (-p.liquidity_score, -p.density))
    return (pools[:max_pools], sweeps[-max_pools:])


__all__ = ["detect_liquidity"]

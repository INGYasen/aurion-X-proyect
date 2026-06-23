"""AURION X 4.1 SMC — Multi-timeframe orchestration.

Recorre los timeframes activos (default: M15/M30/H1/H4/D1/W1, M5 y M1 para
timing, MN1 para contexto) y ejecuta los detectores SMC con parámetros
adaptados por TF (la idea: en TFs altos exigimos displacement mayor; en TFs
operativos relajamos para captar setups intradía).

XAUUSD-only: el módulo está pensado para oro y los defaults reflejan su
volatilidad. Para otros símbolos los parámetros se pueden override vía YAML
pero por diseño este motor sólo se conecta cuando el símbolo activo es
XAUUSD (gate en el Orchestrator).

Jerarquía institucional:

* **Macro** (MN1, W1, D1): define sesgo macro y zonas premium/discount.
* **Institucional** (H4, H1): contiene la OB / FVG / liquidez "premium" para
  entradas con mejor RR.
* **Operativa** (M30, M15): TF principal del oro (default M15).
* **Timing** (M5, M1): ejecución (no se ejecuta detección pesada aquí; sólo
  el Orchestrator consume el sesgo).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from .bos_detector import detect_bos
from .choch_detector import detect_choch
from .fvg_engine import detect_fvgs
from .liquidity_engine import detect_liquidity
from .order_block_engine import detect_order_blocks
from .types import StructureDirection, TFContext


@dataclass(slots=True)
class TFParams:
    """Parámetros por timeframe del motor SMC."""

    # BOS
    bos_strong_disp_atr: float = 1.0
    bos_weak_disp_atr: float = 0.25
    # CHOCH
    choch_major_disp_atr: float = 1.0
    choch_min_disp_atr: float = 0.20
    choch_min_body_atr: float = 0.18
    # OB
    ob_displacement_atr: float = 1.0
    ob_max_blocks: int = 12
    # FVG
    fvg_min_size_atr: float = 0.20
    fvg_max_gaps: int = 16
    # Liquidity
    liq_equal_tol_atr: float = 0.15
    liq_dense_touches: int = 3
    liq_sweep_excursion_atr: float = 0.35
    liq_sweep_reentry_pct: float = 0.55
    # Swing detector
    swing_left: int = 2
    swing_right: int = 2
    # Pip size — XAUUSD default
    pip_size: float = 0.10


DEFAULT_TF_PARAMS_XAUUSD: dict[str, TFParams] = {
    # Macro: exigimos displacement amplio para evitar ruido
    "MN1": TFParams(bos_strong_disp_atr=1.2, choch_major_disp_atr=1.2, ob_displacement_atr=1.2,
                    fvg_min_size_atr=0.25, swing_left=3, swing_right=3),
    "W1": TFParams(bos_strong_disp_atr=1.1, choch_major_disp_atr=1.1, ob_displacement_atr=1.1,
                   fvg_min_size_atr=0.22, swing_left=3, swing_right=3),
    "D1": TFParams(bos_strong_disp_atr=1.0, choch_major_disp_atr=1.0, ob_displacement_atr=1.0,
                   fvg_min_size_atr=0.20, swing_left=2, swing_right=2),
    # Institucional
    "H4": TFParams(bos_strong_disp_atr=0.9, choch_major_disp_atr=0.9, ob_displacement_atr=0.9,
                   fvg_min_size_atr=0.20, swing_left=2, swing_right=2),
    "H1": TFParams(bos_strong_disp_atr=0.85, choch_major_disp_atr=0.85, ob_displacement_atr=0.85,
                   fvg_min_size_atr=0.18, swing_left=2, swing_right=2),
    # Operativa (XAUUSD: M15 es el TF principal del prompt)
    "M30": TFParams(bos_strong_disp_atr=0.80, choch_major_disp_atr=0.80, ob_displacement_atr=0.80,
                    fvg_min_size_atr=0.16, swing_left=2, swing_right=2),
    "M15": TFParams(bos_strong_disp_atr=0.75, choch_major_disp_atr=0.75, ob_displacement_atr=0.75,
                    fvg_min_size_atr=0.15, swing_left=2, swing_right=2),
    # Timing (relajamos un poco para captar setups rápidos)
    "M5": TFParams(bos_strong_disp_atr=0.65, choch_major_disp_atr=0.65, ob_displacement_atr=0.65,
                   fvg_min_size_atr=0.12, swing_left=2, swing_right=2),
    "M1": TFParams(bos_strong_disp_atr=0.55, choch_major_disp_atr=0.55, ob_displacement_atr=0.55,
                   fvg_min_size_atr=0.10, swing_left=2, swing_right=2),
}


# Bars callback: lambda(symbol, tf) -> (highs, lows, closes, volumes, times)
BarsCallback = Callable[[str, str], tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float], Sequence[datetime] | None]]


def _bias_from_tf(ctx: TFContext) -> StructureDirection:
    """Resuelve el sesgo de un TF priorizando CHOCH reciente sobre BOS."""
    if ctx.last_choch is not None:
        return ctx.last_choch.direction
    if ctx.last_bos is not None:
        return ctx.last_bos.direction
    return StructureDirection.NEUTRAL


def evaluate_timeframe(
    symbol: str,
    timeframe: str,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float] | None,
    times: Sequence[datetime] | None,
    params: TFParams,
    *,
    keep_filled_fvg: bool = False,
) -> TFContext:
    """Ejecuta todos los detectores SMC en un único timeframe."""
    bos = detect_bos(
        highs, lows, closes,
        times=times, timeframe=timeframe,
        left_bars=params.swing_left, right_bars=params.swing_right,
        strong_displacement_atr=params.bos_strong_disp_atr,
        weak_displacement_atr=params.bos_weak_disp_atr,
        pip_size=params.pip_size,
    )
    choch = detect_choch(
        highs, lows, closes,
        times=times, timeframe=timeframe,
        left_bars=params.swing_left, right_bars=params.swing_right,
        major_displacement_atr=params.choch_major_disp_atr,
        min_displacement_atr=params.choch_min_disp_atr,
        min_body_atr=params.choch_min_body_atr,
    )
    obs = detect_order_blocks(
        highs, lows, closes, volumes,
        times=times, timeframe=timeframe,
        displacement_atr=params.ob_displacement_atr,
        max_blocks=params.ob_max_blocks,
        pip_size=params.pip_size,
    )
    fvgs = detect_fvgs(
        highs, lows, closes,
        times=times, timeframe=timeframe,
        min_size_atr=params.fvg_min_size_atr,
        max_gaps=params.fvg_max_gaps,
        keep_filled=keep_filled_fvg,
        pip_size=params.pip_size,
    )
    pools, sweeps = detect_liquidity(
        highs, lows, closes,
        times=times, timeframe=timeframe,
        left_bars=params.swing_left, right_bars=params.swing_right,
        equal_tolerance_atr=params.liq_equal_tol_atr,
        pool_min_touches_dense=params.liq_dense_touches,
        sweep_min_excursion_atr=params.liq_sweep_excursion_atr,
        sweep_min_reentry_pct=params.liq_sweep_reentry_pct,
    )

    ctx = TFContext(
        timeframe=timeframe,
        bias=StructureDirection.NEUTRAL,
        last_bos=bos,
        last_choch=choch,
        active_obs=obs,
        active_fvgs=fvgs,
        liquidity_pools=pools,
        recent_sweeps=sweeps,
    )
    ctx.bias = _bias_from_tf(ctx)
    return ctx


def evaluate_symbol_mtf(
    symbol: str,
    timeframes: Sequence[str],
    bars_cb: BarsCallback,
    *,
    params_by_tf: dict[str, TFParams] | None = None,
    min_bars: int = 20,
) -> dict[str, TFContext]:
    """Evalúa SMC para un símbolo en todos los timeframes solicitados.

    Args:
        symbol: símbolo (típicamente "XAUUSD"; el motor SMC sólo se activa
            para XAUUSD por configuración del Orchestrator).
        timeframes: lista de TFs a recorrer (e.g. ``["M15","M30","H1","H4","D1","W1"]``).
        bars_cb: callback que devuelve ``(highs, lows, closes, volumes, times)``
            para ``(symbol, tf)``. Si la longitud devuelta es < ``min_bars``
            se omite ese TF.
        params_by_tf: override de parámetros por TF (cae a defaults XAUUSD).
        min_bars: nº mínimo de velas necesarias para evaluar un TF.

    Returns:
        Dict ``{tf: TFContext}`` con resultados por timeframe.
    """
    out: dict[str, TFContext] = {}
    params_map = params_by_tf or {}
    for tf in timeframes:
        try:
            highs, lows, closes, volumes, times = bars_cb(symbol, tf)
        except Exception:
            continue
        if not closes or len(closes) < min_bars:
            continue
        params = params_map.get(tf) or DEFAULT_TF_PARAMS_XAUUSD.get(tf) or TFParams()
        try:
            ctx = evaluate_timeframe(
                symbol, tf, highs, lows, closes, volumes, times, params,
            )
        except Exception:
            # Sin propagación: SMC nunca debe tirar el Orchestrator.
            continue
        out[tf] = ctx
    return out


__all__ = [
    "TFParams",
    "DEFAULT_TF_PARAMS_XAUUSD",
    "BarsCallback",
    "evaluate_timeframe",
    "evaluate_symbol_mtf",
]

"""AURION X 4.1 SMC — Confluence Engine.

Toma los ``TFContext`` por timeframe y produce:

1. Un score [0,1] por componente (BOS, CHOCH, OB, FVG, Liquidity) — ponderado
   por la **escala Fibonacci** definida en el prompt:

       M15 = 1, M30 = 2, H1 = 3, H4 = 5, D1 = 8, W1 = 13

   (Los TFs no listados explícitamente — MN1, M5, M1 — reciben peso 0 por
   defecto pero el bias macro/timing sí se propaga al snapshot final.)

2. Un ``confluence_score`` ∈ [0,1] que combina los componentes con los pesos
   institucionales del prompt:

       BOS 10% · CHOCH 15% · OB 20% · FVG 15% · LIQUIDITY 10%   (suma = 70%)

   El 30 % restante lo aporta el ML al integrarse con el ``composite_score``
   del QuantEngine (ver ``analytics_smc_integration``).

3. Un ``direction_bias`` macro derivado de los TFs altos (D1/H4/H1), que el
   Orchestrator puede usar para alinear señales scalper con el lado correcto.

Los pesos por componente y los pesos Fibonacci son configurables vía YAML
(``smc.weights.*`` y ``smc.fib_weights.*``).
"""

from __future__ import annotations

from typing import Mapping

from .types import (
    ConfluenceResult,
    FairValueGap,
    LiquidityPool,
    OBState,
    OrderBlock,
    StructureDirection,
    TFContext,
)


DEFAULT_FIB_WEIGHTS: dict[str, float] = {
    "M1": 0.0,
    "M5": 0.0,
    "M15": 1.0,
    "M30": 2.0,
    "H1": 3.0,
    "H4": 5.0,
    "D1": 8.0,
    "W1": 13.0,
    "MN1": 0.0,
}


DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "bos": 0.10,
    "choch": 0.15,
    "ob": 0.20,
    "fvg": 0.15,
    "liquidity": 0.10,
}


def _component_value_bos(ctx: TFContext) -> float:
    if ctx.last_bos is None:
        return 0.0
    return float(ctx.last_bos.bos_reliability)


def _component_value_choch(ctx: TFContext) -> float:
    if ctx.last_choch is None:
        return 0.0
    return float(ctx.last_choch.choch_score)


def _component_value_ob(ctx: TFContext, direction_bias: StructureDirection | None) -> float:
    if not ctx.active_obs:
        return 0.0
    relevant: list[OrderBlock] = []
    for ob in ctx.active_obs:
        if ob.state == OBState.INVALIDATED:
            continue
        if direction_bias is not None and direction_bias != StructureDirection.NEUTRAL:
            # Si tenemos sesgo, OBs alineadas pesan, las contrarias también
            # pero como reducción (i.e., un OB bearish con sesgo bullish es
            # zona de oferta — útil como contexto, peso 0.4).
            if ob.direction == direction_bias:
                relevant.append(ob)
            else:
                # cuenta a la mitad
                relevant.append(_demoted_ob(ob, factor=0.4))
        else:
            relevant.append(ob)
    if not relevant:
        return 0.0
    # Score = mejor OB (max quality * freshness) — para reconocer si hay una OB
    # clave activa, no diluir con OBs viejas.
    best = max(relevant, key=lambda x: x.ob_quality * (0.3 + 0.7 * x.ob_freshness))
    return float(best.ob_quality * (0.3 + 0.7 * best.ob_freshness))


def _demoted_ob(ob: OrderBlock, *, factor: float) -> OrderBlock:
    # No mutamos la instancia original; devolvemos una "copia" con quality reducida.
    # (Como dataclass slots=True no soporta copy.replace si no está frozen, usamos asignación manual.)
    clone = OrderBlock(
        timeframe=ob.timeframe,
        direction=ob.direction,
        top=ob.top,
        bottom=ob.bottom,
        open_time=ob.open_time,
        open_index=ob.open_index,
        state=ob.state,
        ob_strength=ob.ob_strength * factor,
        ob_volume_score=ob.ob_volume_score,
        ob_displacement_score=ob.ob_displacement_score,
        ob_reaction_score=ob.ob_reaction_score,
        ob_freshness=ob.ob_freshness * factor,
        ob_age_bars=ob.ob_age_bars,
        ob_quality=ob.ob_quality * factor,
        mitigation_time=ob.mitigation_time,
        extra={**ob.extra, "demoted_factor": factor},
    )
    return clone


def _component_value_fvg(ctx: TFContext, direction_bias: StructureDirection | None) -> float:
    if not ctx.active_fvgs:
        return 0.0
    relevant_strengths: list[float] = []
    for fvg in ctx.active_fvgs:
        if direction_bias is None or direction_bias == StructureDirection.NEUTRAL:
            relevant_strengths.append(fvg.fvg_strength)
        elif fvg.direction == direction_bias:
            relevant_strengths.append(fvg.fvg_strength)
        else:
            relevant_strengths.append(fvg.fvg_strength * 0.4)
    if not relevant_strengths:
        return 0.0
    # Usamos el max — un FVG fuerte activo justifica el componente.
    return float(max(relevant_strengths))


def _component_value_liquidity(ctx: TFContext) -> float:
    if not ctx.liquidity_pools:
        return 0.0
    # Pools activos no barridos pesan más; pools barridos siguen contando pero
    # con factor 0.6 (la barrida ya consumió liquidez pero el contexto sigue
    # siendo útil para el Orchestrator si la barrida es muy reciente).
    scored = []
    for p in ctx.liquidity_pools:
        s = p.liquidity_score
        if p.swept:
            s *= 0.6
        scored.append(s)
    return float(max(scored))


def _macro_bias(per_tf: Mapping[str, TFContext]) -> StructureDirection:
    """Sesgo macro a partir de D1/W1/MN1; mayoría simple."""
    counts = {StructureDirection.BULLISH: 0, StructureDirection.BEARISH: 0, StructureDirection.NEUTRAL: 0}
    for tf in ("MN1", "W1", "D1"):
        ctx = per_tf.get(tf)
        if ctx is None:
            continue
        counts[ctx.bias] += 1
    if counts[StructureDirection.BULLISH] > counts[StructureDirection.BEARISH]:
        return StructureDirection.BULLISH
    if counts[StructureDirection.BEARISH] > counts[StructureDirection.BULLISH]:
        return StructureDirection.BEARISH
    return StructureDirection.NEUTRAL


def _inst_bias(per_tf: Mapping[str, TFContext]) -> StructureDirection:
    """Sesgo institucional H4+H1."""
    counts = {StructureDirection.BULLISH: 0, StructureDirection.BEARISH: 0, StructureDirection.NEUTRAL: 0}
    for tf in ("H4", "H1"):
        ctx = per_tf.get(tf)
        if ctx is None:
            continue
        counts[ctx.bias] += 1
    if counts[StructureDirection.BULLISH] > counts[StructureDirection.BEARISH]:
        return StructureDirection.BULLISH
    if counts[StructureDirection.BEARISH] > counts[StructureDirection.BULLISH]:
        return StructureDirection.BEARISH
    return StructureDirection.NEUTRAL


def _op_bias(per_tf: Mapping[str, TFContext]) -> StructureDirection:
    """Sesgo operativo M30+M15."""
    counts = {StructureDirection.BULLISH: 0, StructureDirection.BEARISH: 0, StructureDirection.NEUTRAL: 0}
    for tf in ("M30", "M15"):
        ctx = per_tf.get(tf)
        if ctx is None:
            continue
        counts[ctx.bias] += 1
    if counts[StructureDirection.BULLISH] > counts[StructureDirection.BEARISH]:
        return StructureDirection.BULLISH
    if counts[StructureDirection.BEARISH] > counts[StructureDirection.BULLISH]:
        return StructureDirection.BEARISH
    return StructureDirection.NEUTRAL


def compute_confluence(
    symbol: str,
    per_tf: Mapping[str, TFContext],
    *,
    fib_weights: Mapping[str, float] | None = None,
    component_weights: Mapping[str, float] | None = None,
    direction_hint: str | None = None,
) -> ConfluenceResult:
    """Calcula el score SMC global y por componente.

    Args:
        symbol: ticker.
        per_tf: contextos SMC por timeframe.
        fib_weights: pesos por TF (default Fibonacci del prompt).
        component_weights: pesos por componente.
        direction_hint: ``"BUY"``/``"SELL"`` para orientar OB/FVG; ``None``
            calcula score directional-neutro (toma el mejor de cada lado).

    Returns:
        ``ConfluenceResult`` con breakdown completo.
    """
    fw = dict(DEFAULT_FIB_WEIGHTS)
    if fib_weights:
        fw.update({k: float(v) for k, v in fib_weights.items()})
    cw = dict(DEFAULT_COMPONENT_WEIGHTS)
    if component_weights:
        cw.update({k: float(v) for k, v in component_weights.items()})

    direction_bias: StructureDirection | None = None
    if direction_hint:
        d = direction_hint.upper()
        if d in ("BUY", "BULLISH", "LONG", "UP"):
            direction_bias = StructureDirection.BULLISH
        elif d in ("SELL", "BEARISH", "SHORT", "DOWN"):
            direction_bias = StructureDirection.BEARISH

    # Fibonacci-weighted average por componente.
    bos_sum = choch_sum = ob_sum = fvg_sum = liq_sum = 0.0
    w_sum = 0.0
    per_tf_score: dict[str, float] = {}
    for tf, ctx in per_tf.items():
        w = float(fw.get(tf, 0.0))
        if w <= 0:
            continue
        bos_v = _component_value_bos(ctx)
        choch_v = _component_value_choch(ctx)
        ob_v = _component_value_ob(ctx, direction_bias)
        fvg_v = _component_value_fvg(ctx, direction_bias)
        liq_v = _component_value_liquidity(ctx)
        bos_sum += w * bos_v
        choch_sum += w * choch_v
        ob_sum += w * ob_v
        fvg_sum += w * fvg_v
        liq_sum += w * liq_v
        w_sum += w
        tf_local = (
            cw["bos"] * bos_v
            + cw["choch"] * choch_v
            + cw["ob"] * ob_v
            + cw["fvg"] * fvg_v
            + cw["liquidity"] * liq_v
        )
        smc_total = cw["bos"] + cw["choch"] + cw["ob"] + cw["fvg"] + cw["liquidity"]
        per_tf_score[tf] = tf_local / max(smc_total, 1e-9)

    if w_sum <= 0:
        return ConfluenceResult(
            symbol=symbol,
            direction_bias=direction_bias or StructureDirection.NEUTRAL,
            confluence_score=0.0,
            weights=cw,
            per_tf=per_tf_score,
            fib_weighted_sum=0.0,
            fib_weights_total=0.0,
            notes=["no_active_timeframes"],
        )

    bos_c = bos_sum / w_sum
    choch_c = choch_sum / w_sum
    ob_c = ob_sum / w_sum
    fvg_c = fvg_sum / w_sum
    liq_c = liq_sum / w_sum

    smc_weights_total = cw["bos"] + cw["choch"] + cw["ob"] + cw["fvg"] + cw["liquidity"]
    confluence = (
        cw["bos"] * bos_c
        + cw["choch"] * choch_c
        + cw["ob"] * ob_c
        + cw["fvg"] * fvg_c
        + cw["liquidity"] * liq_c
    ) / max(smc_weights_total, 1e-9)
    confluence = max(0.0, min(1.0, confluence))

    return ConfluenceResult(
        symbol=symbol,
        direction_bias=direction_bias or StructureDirection.NEUTRAL,
        confluence_score=confluence,
        bos_component=bos_c,
        choch_component=choch_c,
        ob_component=ob_c,
        fvg_component=fvg_c,
        liquidity_component=liq_c,
        weights=cw,
        per_tf=per_tf_score,
        fib_weighted_sum=w_sum * confluence,
        fib_weights_total=w_sum,
        notes=[],
    )


def macro_institutional_op_bias(
    per_tf: Mapping[str, TFContext],
) -> tuple[StructureDirection, StructureDirection, StructureDirection]:
    """Devuelve (macro, institucional, operativa)."""
    return _macro_bias(per_tf), _inst_bias(per_tf), _op_bias(per_tf)


__all__ = [
    "DEFAULT_FIB_WEIGHTS",
    "DEFAULT_COMPONENT_WEIGHTS",
    "compute_confluence",
    "macro_institutional_op_bias",
]

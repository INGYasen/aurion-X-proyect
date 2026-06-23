"""Smart Stops (AURION X 4.6) — SL Conservador / Balanceado / Agresivo.

Calcula 3 niveles de Stop Loss para cada operación, basados en:

    - **Conservador**: último swing roto del BOS/CHOCH + ATR×1.5 de buffer.
      Stop más amplio, baja probabilidad de stop-out, peor RR.
    - **Balanceado**: límite del Order Block alineado + ATR×0.5.
      Punto técnico estándar SMC.
    - **Agresivo**: medio del Order Block o FVG inmediato + ATR×0.15.
      Stop muy ajustado, alto RR pero stop-out frecuente.

Además calcula:

    - Probabilidad de stop-hunt en cada SL (basado en liquidity pools cercanos)
    - RR esperado contra cada TP (TP1, TP2, TP3, TP4)
    - Recomendación de modo según el ``setup_grade`` y ``smc_score``

NO usa ML; es 100% determinista. La selección final de SL queda al
orchestrator (típicamente "Balanceado" excepto en grade A+ con baja
volatilidad → "Agresivo").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ─── tipos publicos ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class StopLevel:
    """Un nivel de SL con metadatos y prob de stop-hunt."""
    mode: str                  # "conservative" | "balanced" | "aggressive"
    price: float
    distance_pips: float       # XAUUSD: 1 pip = 0.10$
    distance_atr: float
    stop_hunt_probability: float  # 0..1
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "price": float(self.price),
            "distance_pips": float(self.distance_pips),
            "distance_atr": float(self.distance_atr),
            "stop_hunt_probability": float(self.stop_hunt_probability),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SLPlan:
    """Plan completo: 3 SL + RR vs TPs + modo recomendado."""
    conservative: StopLevel
    balanced: StopLevel
    aggressive: StopLevel
    recommended_mode: str
    rr_table: dict[str, dict[str, float]]   # {"conservative": {"tp1": 1.8, "tp2": 3.2, ...}, ...}
    notes: list[str] = field(default_factory=list)

    def selected(self) -> StopLevel:
        if self.recommended_mode == "conservative":
            return self.conservative
        if self.recommended_mode == "aggressive":
            return self.aggressive
        return self.balanced

    def to_dict(self) -> dict[str, Any]:
        return {
            "conservative": self.conservative.to_dict(),
            "balanced": self.balanced.to_dict(),
            "aggressive": self.aggressive.to_dict(),
            "recommended_mode": self.recommended_mode,
            "rr_table": {k: dict(v) for k, v in self.rr_table.items()},
            "notes": list(self.notes),
        }


# ─── helpers ────────────────────────────────────────────────────────────


def _pips(price_distance: float) -> float:
    return abs(price_distance) * 10.0


def _find_swing_reference(snapshot: Any, direction: str, primary_tf: str) -> float | None:
    """Devuelve el broken_swing_price del último BOS/CHOCH alineado con dir."""
    if snapshot is None:
        return None
    d = (direction or "").upper()
    target_dir = "BULLISH" if d in ("BUY", "LONG") else "BEARISH"
    per_tf = getattr(snapshot, "per_tf", None) or {}
    # Priorizamos HTF (H4 > H1 > primary_tf)
    for tf in ("H4", "H1", primary_tf, "M30", "M15"):
        ctx = per_tf.get(tf)
        if ctx is None:
            continue
        for ev_attr in ("last_bos", "last_choch"):
            ev = getattr(ctx, ev_attr, None)
            if ev is None:
                continue
            dir_obj = getattr(ev, "direction", None)
            ev_dir = (dir_obj.value if hasattr(dir_obj, "value")
                      else str(dir_obj or "")).upper()
            if ev_dir != target_dir:
                continue
            sp = float(getattr(ev, "broken_swing_price", 0.0) or 0.0)
            if sp > 0:
                return sp
    return None


def _find_aligned_ob(snapshot: Any, direction: str, entry: float) -> tuple[float | None, float | None]:
    """Devuelve (top, bottom) del OB alineado más cercano al entry (debajo BUY/arriba SELL)."""
    if snapshot is None:
        return None, None
    d = (direction or "").upper()
    is_buy = d in ("BUY", "LONG")
    target_ob_dir = "BULLISH" if is_buy else "BEARISH"
    per_tf = getattr(snapshot, "per_tf", None) or {}
    best = None  # (distance, top, bot)
    for tf, ctx in per_tf.items():
        for ob in (getattr(ctx, "active_obs", None) or []):
            dir_obj = getattr(ob, "direction", None)
            ob_dir = (dir_obj.value if hasattr(dir_obj, "value")
                      else str(dir_obj or "")).upper()
            if ob_dir != target_ob_dir:
                continue
            top = float(getattr(ob, "top", 0.0) or 0.0)
            bot = float(getattr(ob, "bottom", 0.0) or 0.0)
            mid = (top + bot) / 2.0
            if mid <= 0:
                continue
            # BUY: OB debe estar debajo del entry (demanda)
            # SELL: OB debe estar arriba del entry (oferta)
            if is_buy and mid >= entry:
                continue
            if not is_buy and mid <= entry:
                continue
            dist = abs(mid - entry)
            if best is None or dist < best[0]:
                best = (dist, top, bot)
    if best is None:
        return None, None
    return best[1], best[2]


def _stop_hunt_probability(
    snapshot: Any, price: float, direction: str,
) -> float:
    """Calcula prob de stop-hunt: hay pools de liquidez cerca del SL?"""
    if snapshot is None or price <= 0:
        return 0.30
    d = (direction or "").upper()
    is_buy = d in ("BUY", "LONG")
    # Para BUY, peligro = SSL cerca del SL
    # Para SELL, peligro = BSL cerca del SL
    per_tf = getattr(snapshot, "per_tf", None) or {}
    danger_score = 0.0
    for ctx in per_tf.values():
        for pool in (getattr(ctx, "liquidity_pools", None) or []):
            if getattr(pool, "swept", False):
                continue
            p_price = float(getattr(pool, "price", 0.0) or 0.0)
            if p_price <= 0:
                continue
            dist = abs(p_price - price)
            if dist > 5.0:  # > 50 pips XAUUSD → ignorar
                continue
            kind_obj = getattr(pool, "kind", None)
            kind_s = (kind_obj.value if hasattr(kind_obj, "value")
                      else str(kind_obj or "")).upper()
            is_ssl_like = "SSL" in kind_s or "LOW" in kind_s
            is_bsl_like = "BSL" in kind_s or "HIGH" in kind_s
            relevant = (is_buy and is_ssl_like) or (not is_buy and is_bsl_like)
            if not relevant:
                continue
            # Peso inverso a la distancia (más cerca = más peligro)
            danger_score += (1.0 - dist / 5.0) * 0.3
            touches = int(getattr(pool, "touches", 1) or 1)
            danger_score += 0.05 * min(3, touches - 1)
    return max(0.05, min(0.95, 0.30 + danger_score))


# ─── api publica ─────────────────────────────────────────────────────────


def compute_stop_plan(
    *,
    direction: str,
    entry: float,
    snapshot: Any | None,
    atr: float,
    primary_tf: str = "M15",
    setup_grade: str = "B",
    smc_score: float = 0.0,
    targets_prices: list[float] | None = None,
) -> SLPlan:
    """Calcula 3 SL + RR vs TPs.

    Args:
        direction: "BUY"|"SELL"
        entry: precio de entrada
        snapshot: SMCSnapshot
        atr: ATR(14) primary TF
        primary_tf: TF principal (M15 default)
        setup_grade: grade del SMCDecisionEngine ("A+","A","B","C","X")
        smc_score: 0-100 (para decidir recomendado)
        targets_prices: lista de precios TP1..TPn para calcular RR

    Returns:
        SLPlan con conservative/balanced/aggressive + RR table + recommended.
    """
    if entry <= 0 or not direction:
        # fallback degenerado
        d = (direction or "BUY").upper()
        sign = -1.0 if d in ("BUY", "LONG") else 1.0
        fallback_price = entry + sign * max(0.5, atr or 0.5) * 1.5
        sl = StopLevel(
            mode="balanced", price=fallback_price,
            distance_pips=_pips(fallback_price - entry),
            distance_atr=1.5, stop_hunt_probability=0.5,
            reason="fallback (no entry/direction)",
        )
        return SLPlan(
            conservative=sl, balanced=sl, aggressive=sl,
            recommended_mode="balanced",
            rr_table={"conservative": {}, "balanced": {}, "aggressive": {}},
            notes=["fallback degenerado"],
        )

    d = direction.upper()
    is_buy = d in ("BUY", "LONG")
    sign = -1.0 if is_buy else 1.0  # SL siempre al lado opuesto del trade
    atr_safe = max(0.05, float(atr or 0.05))

    # 1) Conservador: swing + ATR*1.5
    swing_ref = _find_swing_reference(snapshot, direction, primary_tf)
    if swing_ref is None:
        # No hay swing → degradamos a entry +- 3 ATR
        cons_price = entry + sign * 3.0 * atr_safe
        cons_reason = f"fallback no_swing entry{'-' if is_buy else '+'}3ATR"
    else:
        cons_price = swing_ref + sign * 1.5 * atr_safe
        cons_reason = f"swing@{swing_ref:.2f}{'-' if is_buy else '+'}1.5ATR"

    # 2) Balanceado: OB opuesto + ATR*0.5  (o swing + ATR*0.5 si no hay OB)
    ob_top, ob_bot = _find_aligned_ob(snapshot, direction, entry)
    if ob_bot is not None and is_buy:
        bal_price = ob_bot + sign * 0.5 * atr_safe
        bal_reason = f"ob_bot@{ob_bot:.2f}-0.5ATR"
    elif ob_top is not None and not is_buy:
        bal_price = ob_top + sign * 0.5 * atr_safe
        bal_reason = f"ob_top@{ob_top:.2f}+0.5ATR"
    elif swing_ref is not None:
        bal_price = swing_ref + sign * 0.5 * atr_safe
        bal_reason = f"swing@{swing_ref:.2f}{'-' if is_buy else '+'}0.5ATR"
    else:
        bal_price = entry + sign * 1.5 * atr_safe
        bal_reason = "fallback entry+/-1.5ATR"

    # 3) Agresivo: mid OB + ATR*0.15  (muy ajustado)
    if ob_top is not None and ob_bot is not None:
        ob_mid = (ob_top + ob_bot) / 2.0
        agr_price = ob_mid + sign * 0.15 * atr_safe
        agr_reason = f"ob_mid@{ob_mid:.2f}{'-' if is_buy else '+'}0.15ATR"
    elif swing_ref is not None:
        # 50% del camino entre entry y swing
        agr_price = (entry + swing_ref) / 2.0 + sign * 0.15 * atr_safe
        agr_reason = f"halfway_swing{'-' if is_buy else '+'}0.15ATR"
    else:
        agr_price = entry + sign * 0.8 * atr_safe
        agr_reason = "fallback entry+/-0.8ATR"

    # Validar que los 3 SL están al lado correcto del entry
    def _validate(price: float, default_dist: float) -> float:
        if (is_buy and price >= entry) or (not is_buy and price <= entry):
            return entry + sign * default_dist
        return price

    cons_price = _validate(cons_price, 3.0 * atr_safe)
    bal_price = _validate(bal_price, 1.5 * atr_safe)
    agr_price = _validate(agr_price, 0.8 * atr_safe)

    # Asegurar orden Agr < Bal < Cons (en distancia)
    cons_d = abs(cons_price - entry)
    bal_d  = abs(bal_price - entry)
    agr_d  = abs(agr_price - entry)
    # Si están en orden inverso por algún motivo, los swappeamos con un ATR mínimo
    if bal_d <= agr_d:
        bal_price = entry + sign * max(bal_d, agr_d + 0.4 * atr_safe)
        bal_d = abs(bal_price - entry)
    if cons_d <= bal_d:
        cons_price = entry + sign * (bal_d + 1.0 * atr_safe)
        cons_d = abs(cons_price - entry)

    # Prob de stop-hunt
    sh_cons = _stop_hunt_probability(snapshot, cons_price, direction)
    sh_bal  = _stop_hunt_probability(snapshot, bal_price, direction)
    sh_agr  = _stop_hunt_probability(snapshot, agr_price, direction)
    # Ajuste: agresivo es por definición más vulnerable
    sh_agr = min(0.95, sh_agr * 1.15)
    sh_cons = max(0.05, sh_cons * 0.85)

    conservative = StopLevel(
        mode="conservative", price=float(cons_price),
        distance_pips=_pips(cons_d), distance_atr=cons_d / atr_safe,
        stop_hunt_probability=sh_cons, reason=cons_reason,
    )
    balanced = StopLevel(
        mode="balanced", price=float(bal_price),
        distance_pips=_pips(bal_d), distance_atr=bal_d / atr_safe,
        stop_hunt_probability=sh_bal, reason=bal_reason,
    )
    aggressive = StopLevel(
        mode="aggressive", price=float(agr_price),
        distance_pips=_pips(agr_d), distance_atr=agr_d / atr_safe,
        stop_hunt_probability=sh_agr, reason=agr_reason,
    )

    # RR table vs TPs
    rr_table: dict[str, dict[str, float]] = {
        "conservative": {}, "balanced": {}, "aggressive": {},
    }
    if targets_prices:
        for i, tp in enumerate(targets_prices, start=1):
            tp_dist = abs(tp - entry)
            for sl_obj, mode in (
                (conservative, "conservative"),
                (balanced, "balanced"),
                (aggressive, "aggressive"),
            ):
                sl_dist = abs(sl_obj.price - entry)
                rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
                rr_table[mode][f"tp{i}"] = round(rr, 2)

    # Recomendar modo según grade + score
    # A+ con baja stop-hunt prob → aggressive permitido
    # A con grade alto → balanced
    # B/C → conservative
    grade = (setup_grade or "B").upper()
    if grade == "A+" and balanced.stop_hunt_probability < 0.40 and smc_score >= 85:
        recommended = "aggressive"
    elif grade in ("A+", "A"):
        recommended = "balanced"
    else:
        recommended = "conservative"

    notes = [
        f"swing_ref={swing_ref}", f"ob_top={ob_top}", f"ob_bot={ob_bot}",
        f"atr={atr_safe:.2f}", f"grade={grade}", f"score={smc_score:.1f}",
        f"recommended={recommended}",
    ]
    return SLPlan(
        conservative=conservative, balanced=balanced, aggressive=aggressive,
        recommended_mode=recommended, rr_table=rr_table, notes=notes,
    )

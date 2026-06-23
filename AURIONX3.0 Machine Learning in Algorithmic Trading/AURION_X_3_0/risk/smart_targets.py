"""Smart Targets (AURION X 4.6) — TP1/TP2/TP3/TP4 desde estructura SMC.

Genera niveles de Take Profit basados en estructura institucional real:

    1. Pools de liquidez NO barridos (BSL para BUY, SSL para SELL)
    2. FVGs HTF opuestos (resistencia/soporte)
    3. Order Blocks opuestos HTF
    4. Highs/Lows del swing del BOS/CHOCH previo
    5. Highs/Lows de sesiones (Asia/London/NY) si están disponibles

A cada TP se le asigna una **probabilidad estimada** de alcance basada en:

    - Distancia normalizada por ATR (más lejos = menor probabilidad)
    - Tipo de nivel (liquidez no barrida > FVG HTF > OB HTF > swing)
    - Fuerza del nivel (touches, liquidity_score, FVG strength)
    - Alineación con sesión actual

NO usa ML ni necesita histórico — heurística determinista calibrada con
estadísticas SMC institucionales (XAUUSD). Cuando hay histórico de
``trade_repository`` se puede recalibrar el ``_PROB_BASE`` por percentil.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# ─── tipos publicos ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Target:
    """Un nivel TP con metadatos y probabilidad estimada."""
    rank: int                # 1..4
    price: float
    distance_pips: float     # XAUUSD: 1 pip = 0.10$
    distance_atr: float      # multiplo de ATR(14)
    source: str              # "liquidity_bsl" | "fvg_h4" | "ob_h1" | "swing_h1" | "session_asia_high"
    timeframe: str           # TF de origen ("H4", "H1", ...)
    probability: float       # 0..1 estimada
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "price": float(self.price),
            "distance_pips": float(self.distance_pips),
            "distance_atr": float(self.distance_atr),
            "source": self.source,
            "timeframe": self.timeframe,
            "probability": float(self.probability),
            "note": self.note,
        }


@dataclass(frozen=True)
class TargetSet:
    """Conjunto ordenado TP1..TP4."""
    targets: list[Target] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def tp(self, n: int) -> Target | None:
        if n < 1 or n > len(self.targets):
            return None
        return self.targets[n - 1]

    def prices(self) -> list[float]:
        return [t.price for t in self.targets]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tps": [t.to_dict() for t in self.targets],
            "notes": list(self.notes),
        }


# ─── tabla de probabilidad base por tipo de nivel ───────────────────────


# Calibrado a partir de SMC institucional XAUUSD: liquidez no barrida
# es el "destino magnético" más probable; FVG HTF actua como freno.
_PROB_BASE: dict[str, float] = {
    "liquidity_bsl_htf": 0.78,   # BSL en H4/D1 no barrido
    "liquidity_ssl_htf": 0.78,
    "liquidity_bsl_mtf": 0.72,   # H1/M30
    "liquidity_ssl_mtf": 0.72,
    "liquidity_bsl":     0.65,   # M15 / generic
    "liquidity_ssl":     0.65,
    "equal_highs":       0.70,
    "equal_lows":        0.70,
    "fvg_h4":            0.62,
    "fvg_h1":            0.55,
    "fvg_m15":           0.48,
    "ob_h4":             0.58,
    "ob_h1":             0.50,
    "ob_m15":            0.42,
    "swing_h4":          0.55,
    "swing_h1":          0.48,
    "swing_m15":         0.40,
    "session_asia_high": 0.60,
    "session_asia_low":  0.60,
    "session_london_high": 0.66,
    "session_london_low":  0.66,
    "session_ny_high":   0.68,
    "session_ny_low":    0.68,
    "session_daily_high": 0.62,
    "session_daily_low":  0.62,
    "session_weekly_high": 0.55,
    "session_weekly_low":  0.55,
}


_TF_TIER: dict[str, str] = {
    "M15": "mtf", "M30": "mtf",
    "H1":  "mtf",
    "H4":  "htf", "D1": "htf", "W1": "htf",
}


# ─── helpers ────────────────────────────────────────────────────────────


def _pips(price_distance: float) -> float:
    """XAUUSD: 1 pip = 0.10$, así 1$ = 10 pips."""
    return abs(price_distance) * 10.0


def _is_above(price: float, ref: float, eps: float = 1e-6) -> bool:
    return price > ref + eps


def _is_below(price: float, ref: float, eps: float = 1e-6) -> bool:
    return price < ref - eps


def _lookup_pool_prob(kind_str: str, tf: str) -> str:
    """Resuelve clave en _PROB_BASE según tipo y TF."""
    k = (kind_str or "").upper()
    is_bsl = "BSL" in k or "HIGH" in k
    is_ssl = "SSL" in k or "LOW" in k
    if "EQUAL_HIGHS" in k:
        return "equal_highs"
    if "EQUAL_LOWS" in k:
        return "equal_lows"
    tier = _TF_TIER.get(tf, "mtf")
    if is_bsl:
        return f"liquidity_bsl_{tier}" if tier != "mtf" else (
            "liquidity_bsl_mtf" if tf in ("M30", "H1") else "liquidity_bsl"
        )
    if is_ssl:
        return f"liquidity_ssl_{tier}" if tier != "mtf" else (
            "liquidity_ssl_mtf" if tf in ("M30", "H1") else "liquidity_ssl"
        )
    return "liquidity_bsl"  # fallback


def _adjust_probability(base: float, distance_atr: float) -> float:
    """Ajusta probabilidad por distancia en ATR.

    distance 0-1 ATR     → factor 1.00 (base)
    distance 1-2 ATR     → factor 0.92
    distance 2-3 ATR     → factor 0.80
    distance 3-5 ATR     → factor 0.65
    distance >5 ATR      → factor 0.45
    """
    da = abs(distance_atr)
    if da <= 1.0:
        f = 1.0
    elif da <= 2.0:
        f = 0.92
    elif da <= 3.0:
        f = 0.80
    elif da <= 5.0:
        f = 0.65
    else:
        f = 0.45
    return max(0.05, min(0.98, base * f))


# ─── extraccion de candidatos ────────────────────────────────────────────


def _collect_candidates_from_snapshot(
    *,
    snapshot: Any,
    direction: str,
    entry: float,
) -> list[Target]:
    """Recorre el SMCSnapshot y extrae niveles candidatos (sin ATR ni prob aun)."""
    out: list[Target] = []
    if snapshot is None:
        return out
    d = (direction or "").upper()
    is_buy = d in ("BUY", "LONG")
    is_sell = d in ("SELL", "SHORT")
    per_tf = getattr(snapshot, "per_tf", None) or {}
    for tf, ctx in per_tf.items():
        if tf is None:
            continue

        # 1) Pools de liquidez no barridos (más probable como destino)
        for pool in (getattr(ctx, "liquidity_pools", None) or []):
            if getattr(pool, "swept", False):
                continue
            price = float(getattr(pool, "price", 0.0) or 0.0)
            if price <= 0:
                continue
            if is_buy and not _is_above(price, entry):
                continue
            if is_sell and not _is_below(price, entry):
                continue
            kind_obj = getattr(pool, "kind", None)
            kind_str = (kind_obj.value if hasattr(kind_obj, "value")
                        else str(kind_obj or "")).upper()
            key = _lookup_pool_prob(kind_str, tf)
            note_parts = [kind_str.lower()]
            touches = int(getattr(pool, "touches", 1) or 1)
            if touches > 1:
                note_parts.append(f"t={touches}")
            liq_sc = float(getattr(pool, "liquidity_score", 0.0) or 0.0)
            base_prob = _PROB_BASE.get(key, 0.55)
            # bonus por touches y liquidity_score
            base_prob = min(0.95, base_prob + 0.03 * min(3, max(0, touches - 1))
                            + 0.05 * max(0.0, min(1.0, liq_sc)))
            out.append(Target(
                rank=0, price=price,
                distance_pips=_pips(price - entry),
                distance_atr=0.0,  # se asigna después
                source=key, timeframe=tf,
                probability=base_prob,
                note=" ".join(note_parts),
            ))

        # 2) FVGs opuestos
        for fvg in (getattr(ctx, "active_fvgs", None) or []):
            state = getattr(fvg, "state", None)
            state_s = (state.value if hasattr(state, "value")
                       else str(state or "")).upper()
            if state_s == "FILLED":
                continue
            top = float(getattr(fvg, "top", 0.0) or 0.0)
            bot = float(getattr(fvg, "bottom", 0.0) or 0.0)
            mid = (top + bot) / 2.0
            if mid <= 0:
                continue
            # BUY → solo FVGs por ENCIMA de entry; SELL → por debajo
            if is_buy and not _is_above(mid, entry):
                continue
            if is_sell and not _is_below(mid, entry):
                continue
            tf_key = (tf.upper() if tf else "M15")
            base_key = "fvg_h4" if tf_key in ("H4", "D1") else (
                "fvg_h1" if tf_key in ("H1", "M30") else "fvg_m15"
            )
            strength = float(getattr(fvg, "strength", 0.0)
                             or getattr(fvg, "fvg_strength", 0.0) or 0.0)
            base_prob = _PROB_BASE.get(base_key, 0.55)
            base_prob = min(0.93, base_prob + 0.05 * max(0.0, min(1.0, strength)))
            out.append(Target(
                rank=0, price=mid,
                distance_pips=_pips(mid - entry),
                distance_atr=0.0,
                source=base_key, timeframe=tf_key,
                probability=base_prob,
                note=f"fvg{'+' if is_buy else '-'}",
            ))

        # 3) Order Blocks opuestos (HTF ideales como TP)
        for ob in (getattr(ctx, "active_obs", None) or []):
            top = float(getattr(ob, "top", 0.0) or 0.0)
            bot = float(getattr(ob, "bottom", 0.0) or 0.0)
            mid = (top + bot) / 2.0
            if mid <= 0:
                continue
            dir_obj = getattr(ob, "direction", None)
            ob_dir = (dir_obj.value if hasattr(dir_obj, "value")
                      else str(dir_obj or "")).upper()
            # Para BUY queremos OBs bearish ARRIBA (oferta=resistencia=TP)
            # Para SELL queremos OBs bullish ABAJO (demanda=soporte=TP)
            if is_buy and ob_dir == "BEARISH" and _is_above(mid, entry):
                target_price = bot  # entrar al OB
            elif is_sell and ob_dir == "BULLISH" and _is_below(mid, entry):
                target_price = top
            else:
                continue
            tf_key = (tf.upper() if tf else "M15")
            base_key = "ob_h4" if tf_key in ("H4", "D1") else (
                "ob_h1" if tf_key in ("H1", "M30") else "ob_m15"
            )
            strength = float(getattr(ob, "strength", 0.0)
                             or getattr(ob, "ob_strength", 0.0) or 0.0)
            base_prob = _PROB_BASE.get(base_key, 0.5)
            base_prob = min(0.92, base_prob + 0.06 * max(0.0, min(1.0, strength)))
            out.append(Target(
                rank=0, price=target_price,
                distance_pips=_pips(target_price - entry),
                distance_atr=0.0,
                source=base_key, timeframe=tf_key,
                probability=base_prob,
                note=f"opposite_ob",
            ))

        # 4) Último swing roto (broken_swing_price del último BOS/CHOCH)
        for ev_attr, base_key_prefix in (("last_bos", "swing"), ("last_choch", "swing")):
            ev = getattr(ctx, ev_attr, None)
            if ev is None:
                continue
            sp = float(getattr(ev, "broken_swing_price", 0.0) or 0.0)
            if sp <= 0:
                continue
            if is_buy and not _is_above(sp, entry):
                continue
            if is_sell and not _is_below(sp, entry):
                continue
            tf_key = (tf.upper() if tf else "M15")
            base_key = f"{base_key_prefix}_h4" if tf_key in ("H4", "D1") else (
                f"{base_key_prefix}_h1" if tf_key in ("H1", "M30") else f"{base_key_prefix}_m15"
            )
            base_prob = _PROB_BASE.get(base_key, 0.48)
            out.append(Target(
                rank=0, price=sp,
                distance_pips=_pips(sp - entry),
                distance_atr=0.0,
                source=base_key, timeframe=tf_key,
                probability=base_prob,
                note=f"swing_{ev_attr}",
            ))
    return out


def _collect_candidates_from_sessions(
    *,
    sessions: dict[str, Any] | None,
    direction: str,
    entry: float,
) -> list[Target]:
    """Inyecta highs/lows de sesiones Asia/London/NY (si están disponibles)."""
    if not sessions:
        return []
    out: list[Target] = []
    d = (direction or "").upper()
    is_buy = d in ("BUY", "LONG")
    is_sell = d in ("SELL", "SHORT")
    for sess_name, info in (sessions or {}).items():
        if not isinstance(info, dict):
            continue
        hi = info.get("high")
        lo = info.get("low")
        swept_hi = bool(info.get("swept_high", False))
        swept_lo = bool(info.get("swept_low", False))
        if hi is not None and not swept_hi:
            price = float(hi)
            if is_buy and _is_above(price, entry):
                key = f"session_{sess_name.lower()}_high"
                base = _PROB_BASE.get(key, 0.60)
                out.append(Target(
                    rank=0, price=price,
                    distance_pips=_pips(price - entry),
                    distance_atr=0.0,
                    source=key, timeframe=sess_name.upper(),
                    probability=base,
                    note=f"session_{sess_name}_high",
                ))
        if lo is not None and not swept_lo:
            price = float(lo)
            if is_sell and _is_below(price, entry):
                key = f"session_{sess_name.lower()}_low"
                base = _PROB_BASE.get(key, 0.60)
                out.append(Target(
                    rank=0, price=price,
                    distance_pips=_pips(price - entry),
                    distance_atr=0.0,
                    source=key, timeframe=sess_name.upper(),
                    probability=base,
                    note=f"session_{sess_name}_low",
                ))
    return out


# ─── api publica ─────────────────────────────────────────────────────────


def compute_targets(
    *,
    direction: str,
    entry: float,
    snapshot: Any | None,
    atr: float,
    sessions: dict[str, Any] | None = None,
    max_targets: int = 4,
    min_distance_pips: float = 5.0,
    min_atr_separation: float = 0.35,
) -> TargetSet:
    """Calcula TP1..TPmax_targets desde estructura SMC.

    Args:
        direction: "BUY" o "SELL".
        entry: precio de entrada.
        snapshot: ``SMCSnapshot`` (puede ser None).
        atr: ATR(14) del primary TF en unidades de precio.
        sessions: dict {"asia": {high, low, swept_high, swept_low}, ...}
        max_targets: cuántos TPs devolver (default 4).
        min_distance_pips: descartar candidatos demasiado cercanos al entry.
        min_atr_separation: separación mínima entre TPs sucesivos (en ATR).

    Returns:
        ``TargetSet`` con TP1..TP4 (cuando hay material suficiente).
    """
    if entry <= 0 or not direction:
        return TargetSet(targets=[], notes=["entry o direction inválidos"])
    atr_safe = max(0.01, float(atr or 0.01))
    cands = _collect_candidates_from_snapshot(
        snapshot=snapshot, direction=direction, entry=entry,
    )
    cands += _collect_candidates_from_sessions(
        sessions=sessions, direction=direction, entry=entry,
    )
    # Filtrar por distancia mínima
    cands = [c for c in cands if c.distance_pips >= min_distance_pips]
    if not cands:
        return TargetSet(targets=[], notes=["sin candidatos válidos"])

    # Calcular distance_atr + ajustar probabilidad
    enriched: list[Target] = []
    for c in cands:
        dist_atr = c.distance_pips / 10.0 / atr_safe
        prob = _adjust_probability(c.probability, dist_atr)
        enriched.append(Target(
            rank=0, price=c.price, distance_pips=c.distance_pips,
            distance_atr=float(dist_atr), source=c.source,
            timeframe=c.timeframe, probability=prob, note=c.note,
        ))

    # Ordenar por distancia ascendente (TP1 más cercano)
    enriched.sort(key=lambda t: t.distance_pips)

    # De-duplicar niveles muy cercanos (< min_atr_separation ATR)
    deduped: list[Target] = []
    last_dist = -1e9
    min_sep_pips = max(2.0, min_atr_separation * atr_safe * 10.0)
    for t in enriched:
        if t.distance_pips - last_dist < min_sep_pips:
            continue
        deduped.append(t)
        last_dist = t.distance_pips
        if len(deduped) >= max_targets:
            break

    # Reasignar ranks
    final = []
    for i, t in enumerate(deduped, start=1):
        final.append(Target(
            rank=i, price=t.price, distance_pips=t.distance_pips,
            distance_atr=t.distance_atr, source=t.source,
            timeframe=t.timeframe, probability=t.probability, note=t.note,
        ))

    notes = [f"n_cands={len(cands)} dedup={len(deduped)} atr={atr_safe:.2f}"]
    return TargetSet(targets=final, notes=notes)

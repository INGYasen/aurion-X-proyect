"""SMC Institutional Heuristics (AURION X 4.5.2) — reglas duras XAUUSD.

Conjunto de reglas deterministas inspiradas en comportamiento institucional
documentado (killzones de Smith, ICT, FundingPips guidelines). Se aplican
DESPUES del score base y antes del grade en ``SMCDecisionEngine``, para
ajustar el ``smc_score`` y dejar trazabilidad de cada regla disparada.

Cada regla:
    * Devuelve ``(delta_score, note, applied)`` donde ``delta_score`` puede
      ser positivo (bonus) o negativo (penalty) y ``applied`` es True si la
      regla tuvo efecto.
    * Es PURA: solo recibe datos del momento, no muta nada.
    * Es soft: si los datos no estan disponibles, simplemente no aplica.

Reglas implementadas (XAUUSD, todos los times en UTC):

  R1  asian_trap         — entradas durante 0-6 UTC en zona equilibrium → penaliza
  R2  london_killzone    — 7-10 UTC con BOS H1 fresco a favor → bonus
  R3  ny_open_killzone   — 12:30-15:00 UTC alineado con MTF → bonus
  R4  ny_lunch_dead      — 15:30-17:00 UTC → penaliza (baja liquidez)
  R5  ny_close_revert    — 19-21 UTC + sweep reciente + CHOCH alineado → bonus
  R6  friday_late_close  — Viernes 16+ UTC → penaliza fuerte (gap risk)
  R7  round_number_trap  — entrada a < 30 pips de un nivel 00 sin sweep previo → penaliza
  R8  round_number_sweep — sweep RECIENTE de un nivel 00 + retroceso → bonus
  R9  equal_highs_lows   — 3+ touches mismo nivel sin romper, direccion los ataca → penaliza
  R10 post_event_spike   — primeros 30min post-evento macro sin sweep → penaliza
  R11 weekend_open_gap   — Domingo 21+ UTC y Lunes <2 UTC → penaliza (apertura semanal)
  R12 stale_setup        — OB/FVG age > 200 barras + multiple toques → penaliza
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class HeuristicResult:
    """Resultado del paso heuristico: delta_total + detalle por regla."""
    delta_total: float
    applied_rules: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_total": float(self.delta_total),
            "applied_rules": list(self.applied_rules),
        }


# ─── helpers ────────────────────────────────────────────────────────────


def _hour_utc(ts: datetime | None) -> int | None:
    if ts is None:
        return None
    try:
        return int(ts.hour)
    except (AttributeError, TypeError):
        return None


def _is_friday(ts: datetime | None) -> bool:
    if ts is None:
        return False
    try:
        return ts.weekday() == 4
    except (AttributeError, TypeError):
        return False


def _is_sunday_or_monday_open(ts: datetime | None) -> bool:
    if ts is None:
        return False
    try:
        wd = ts.weekday()
        hr = ts.hour
        if wd == 6 and hr >= 21:   # domingo tarde
            return True
        if wd == 0 and hr < 2:     # lunes muy temprano
            return True
        return False
    except (AttributeError, TypeError):
        return False


def _distance_to_round(close: float, step: float = 100.0) -> float:
    """Distancia al multiplo de ``step`` mas cercano (XAUUSD: 100 = 4000/4100/...)."""
    try:
        c = float(close)
        if c <= 0 or step <= 0:
            return 1e9
        below = (c // step) * step
        above = below + step
        return float(min(abs(c - below), abs(c - above)))
    except (TypeError, ValueError):
        return 1e9


def _has_recent_sweep(snapshot: Any) -> tuple[bool, bool]:
    """Devuelve (ssl_recent, bsl_recent) si hay sweep barrido en ultimos pools."""
    ssl_seen = False
    bsl_seen = False
    per_tf = getattr(snapshot, "per_tf", None) or {}
    for ctx in per_tf.values():
        for pool in (getattr(ctx, "liquidity_pools", None) or []):
            if not getattr(pool, "swept", False):
                continue
            kind_obj = getattr(pool, "kind", None)
            kind = (kind_obj.value if hasattr(kind_obj, "value")
                    else str(kind_obj or "")).upper()
            if "SSL" in kind or "LOW" in kind:
                ssl_seen = True
            elif "BSL" in kind or "HIGH" in kind:
                bsl_seen = True
    return ssl_seen, bsl_seen


# ─── reglas individuales ────────────────────────────────────────────────


def _r1_asian_trap(
    *, hour: int | None, pd_pct: float, **_: Any,
) -> tuple[float, str, bool]:
    """Sesion asiatica (0-6 UTC) en EQUILIBRIUM → trampa de manipulacion."""
    if hour is None or not (0 <= hour <= 6):
        return 0.0, "", False
    if 0.4 <= pd_pct <= 0.6:
        return -10.0, "asian_equilibrium_trap_zone", True
    return 0.0, "", False


def _r2_london_killzone(
    *, hour: int | None, direction: str, mtf_aligned: int, snapshot: Any, **_: Any,
) -> tuple[float, str, bool]:
    """7-10 UTC + BOS H1 a favor + MTF >=2 → bonus killzone."""
    if hour is None or not (7 <= hour <= 10):
        return 0.0, "", False
    if mtf_aligned < 2:
        return 0.0, "", False
    per_tf = getattr(snapshot, "per_tf", None) or {}
    h1 = per_tf.get("H1")
    if h1 is None:
        return 0.0, "", False
    bos = getattr(h1, "last_bos", None)
    if bos is None:
        return 0.0, "", False
    bos_dir_obj = getattr(bos, "direction", None)
    bos_dir = (bos_dir_obj.value if hasattr(bos_dir_obj, "value")
               else str(bos_dir_obj or "")).upper()
    d = direction.upper()
    aligned = (d in ("BUY", "LONG") and bos_dir == "BULLISH") or (
        d in ("SELL", "SHORT") and bos_dir == "BEARISH"
    )
    if aligned:
        return +8.0, "london_killzone_aligned_h1_bos", True
    return 0.0, "", False


def _r3_ny_open_killzone(
    *, hour: int | None, mtf_aligned: int, **_: Any,
) -> tuple[float, str, bool]:
    """12-15 UTC + MTF full → bonus killzone NY."""
    if hour is None or not (12 <= hour <= 15):
        return 0.0, "", False
    if mtf_aligned >= 3:
        return +6.0, "ny_open_killzone_strong_mtf", True
    return 0.0, "", False


def _r4_ny_lunch_dead(
    *, hour: int | None, **_: Any,
) -> tuple[float, str, bool]:
    """16-17 UTC (lunch) → penaliza por baja liquidez."""
    if hour is None or not (16 <= hour <= 17):
        return 0.0, "", False
    return -8.0, "ny_lunch_low_liquidity", True


def _r5_ny_close_revert(
    *, hour: int | None, sweep_detected: bool, has_choch: bool, **_: Any,
) -> tuple[float, str, bool]:
    """19-21 UTC + sweep + CHOCH alineado → bonus reversion clasica."""
    if hour is None or not (19 <= hour <= 21):
        return 0.0, "", False
    if sweep_detected and has_choch:
        return +7.0, "ny_close_sweep_choch_revert", True
    if not sweep_detected and not has_choch:
        return -4.0, "ny_close_no_sweep_no_choch", True
    return 0.0, "", False


def _r6_friday_late_close(
    *, ts: datetime | None, hour: int | None, **_: Any,
) -> tuple[float, str, bool]:
    """Viernes 16+ UTC → penaliza fuerte (gap weekend risk)."""
    if not _is_friday(ts) or hour is None or hour < 16:
        return 0.0, "", False
    return -15.0, "friday_late_close_weekend_gap_risk", True


def _r7_round_number_trap(
    *, last_close: float, sweep_detected: bool, **_: Any,
) -> tuple[float, str, bool]:
    """Entrada < 30 pips de un nivel 00 sin sweep previo → penaliza."""
    # XAUUSD: 1 pip = 0.10$ → 30 pips = 3.0$
    dist = _distance_to_round(last_close, step=100.0)
    if dist >= 3.0:
        return 0.0, "", False
    if not sweep_detected:
        return -8.0, f"round_number_magnet_close({dist:.2f}$)_no_sweep", True
    return 0.0, "", False


def _r8_round_number_sweep(
    *, last_close: float, sweep_detected: bool, **_: Any,
) -> tuple[float, str, bool]:
    """Sweep reciente + precio retroce > 3$ de un nivel 00 → bonus."""
    dist = _distance_to_round(last_close, step=100.0)
    if dist >= 3.0 and dist <= 8.0 and sweep_detected:
        return +5.0, f"post_round_sweep_retrace({dist:.2f}$)", True
    return 0.0, "", False


def _r9_equal_highs_lows(
    *, direction: str, snapshot: Any, **_: Any,
) -> tuple[float, str, bool]:
    """Pool con touches>=3 sin sweep en direccion del trade → penaliza."""
    per_tf = getattr(snapshot, "per_tf", None) or {}
    d = direction.upper()
    seen_trap = False
    for ctx in per_tf.values():
        for pool in (getattr(ctx, "liquidity_pools", None) or []):
            if getattr(pool, "swept", False):
                continue
            touches = int(getattr(pool, "touches", 0) or 0)
            if touches < 3:
                continue
            kind_obj = getattr(pool, "kind", None)
            kind = (kind_obj.value if hasattr(kind_obj, "value")
                    else str(kind_obj or "")).upper()
            # BUY que va contra BSL (resistencia con 3+ touches) → trap
            # SELL que va contra SSL (soporte con 3+ touches) → trap
            if d in ("BUY", "LONG") and ("BSL" in kind or "HIGH" in kind):
                seen_trap = True
                break
            if d in ("SELL", "SHORT") and ("SSL" in kind or "LOW" in kind):
                seen_trap = True
                break
        if seen_trap:
            break
    if seen_trap:
        return -12.0, "equal_highs_lows_trap_unbroken_3plus_touches", True
    return 0.0, "", False


def _r10_post_event_spike(
    *, in_post_event_window: bool, sweep_detected: bool, **_: Any,
) -> tuple[float, str, bool]:
    """Si flag externo dice post-evento, requerir sweep para entrar."""
    if not in_post_event_window:
        return 0.0, "", False
    if not sweep_detected:
        return -10.0, "post_event_spike_no_sweep", True
    return +4.0, "post_event_clean_sweep", True


def _r11_weekend_open(
    *, ts: datetime | None, **_: Any,
) -> tuple[float, str, bool]:
    """Apertura semanal: domingo 21+ UTC o lunes <2 UTC → penaliza."""
    if _is_sunday_or_monday_open(ts):
        return -10.0, "weekend_open_low_liquidity", True
    return 0.0, "", False


def _r12_stale_setup(
    *, snapshot: Any, direction: str, **_: Any,
) -> tuple[float, str, bool]:
    """OB alineado con age > 200 barras → setup viejo, baja probabilidad."""
    per_tf = getattr(snapshot, "per_tf", None) or {}
    d = direction.upper()
    for ctx in per_tf.values():
        for ob in (getattr(ctx, "active_obs", None) or []):
            dir_obj = getattr(ob, "direction", None)
            ob_dir = (dir_obj.value if hasattr(dir_obj, "value")
                      else str(dir_obj or "")).upper()
            aligned = (d in ("BUY", "LONG") and ob_dir == "BULLISH") or (
                d in ("SELL", "SHORT") and ob_dir == "BEARISH"
            )
            if not aligned:
                continue
            age = int(getattr(ob, "ob_age_bars", 0) or 0)
            if age > 200:
                return -5.0, f"stale_ob_age={age}", True
    return 0.0, "", False


# ─── orquestador de reglas ─────────────────────────────────────────────


_ALL_RULES = [
    ("R1_asian_trap",         _r1_asian_trap),
    ("R2_london_killzone",    _r2_london_killzone),
    ("R3_ny_open_killzone",   _r3_ny_open_killzone),
    ("R4_ny_lunch_dead",      _r4_ny_lunch_dead),
    ("R5_ny_close_revert",    _r5_ny_close_revert),
    ("R6_friday_late_close",  _r6_friday_late_close),
    ("R7_round_number_trap",  _r7_round_number_trap),
    ("R8_round_number_sweep", _r8_round_number_sweep),
    ("R9_equal_highs_lows",   _r9_equal_highs_lows),
    ("R10_post_event_spike",  _r10_post_event_spike),
    ("R11_weekend_open",      _r11_weekend_open),
    ("R12_stale_setup",       _r12_stale_setup),
]


def evaluate_heuristics(
    *,
    ts: datetime | None,
    direction: str,
    last_close: float,
    pd_pct: float,
    mtf_aligned: int,
    sweep_detected: bool,
    has_choch: bool,
    snapshot: Any | None,
    in_post_event_window: bool = False,
    enabled_rules: set[str] | None = None,
    weight: float = 1.0,
) -> HeuristicResult:
    """Aplica todas las reglas activas. Devuelve delta total + detalle.

    ``enabled_rules``: set de IDs (R1_asian_trap, ...) si se quiere
    restringir. ``None`` = todas activas.
    ``weight``: multiplicador global (1.0 default). 0.5 modera, 2.0 amplifica.
    """
    if snapshot is None:
        return HeuristicResult(delta_total=0.0, applied_rules=[])
    hour = _hour_utc(ts)
    context = {
        "ts": ts,
        "hour": hour,
        "direction": direction,
        "last_close": float(last_close or 0.0),
        "pd_pct": float(pd_pct or 0.5),
        "mtf_aligned": int(mtf_aligned or 0),
        "sweep_detected": bool(sweep_detected),
        "has_choch": bool(has_choch),
        "snapshot": snapshot,
        "in_post_event_window": bool(in_post_event_window),
    }
    total = 0.0
    applied: list[dict[str, Any]] = []
    for rule_id, fn in _ALL_RULES:
        if enabled_rules is not None and rule_id not in enabled_rules:
            continue
        try:
            delta, note, did = fn(**context)
        except Exception as exc:  # defensivo: una regla nunca rompe el motor
            applied.append({"id": rule_id, "error": str(exc), "delta": 0.0})
            continue
        if did and delta != 0.0:
            d = float(delta) * float(weight)
            total += d
            applied.append({"id": rule_id, "delta": d, "note": note})
    return HeuristicResult(delta_total=float(total), applied_rules=applied)


def heuristics_from_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Lee config de heuristicas. Devuelve dict con enabled+rules+weight."""
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", True))
    weight = float(cfg.get("weight", 1.0))
    rules = cfg.get("rules")
    enabled_rules: set[str] | None = None
    if rules:
        enabled_rules = set(str(r) for r in rules)
    return {"enabled": enabled, "weight": weight, "enabled_rules": enabled_rules}

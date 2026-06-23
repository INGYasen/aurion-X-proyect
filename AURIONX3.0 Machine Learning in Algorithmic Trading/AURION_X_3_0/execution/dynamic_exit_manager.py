"""Salida dinámica profesional: momentum, compresión, absorción, BE y trailing adaptativo.

La lógica es pura (sin I/O): el orquestador construye ``market_state`` y aplica las acciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


def merge_dynamic_exit_cfg(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fusiona configuración global ``execution.dynamic_exit`` con la del perfil del activo."""
    out = dict(base or {})
    if override and isinstance(override, dict):
        out.update(override)
    return out


def _cfg(m: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(m.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _cfg_bool(m: Mapping[str, Any], key: str, default: bool) -> bool:
    v = m.get(key, default)
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "")
    return bool(v)


def _true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))


def _atr_price(highs: list[float], lows: list[float], closes: list[float], n: int = 14) -> float:
    if len(closes) < 2 or len(highs) != len(closes):
        return max(abs(closes[-1]) * 0.0002, 1e-9) if closes else 1e-9
    m = min(n, len(closes) - 1, len(highs), len(lows))
    if m < 1:
        return max(highs[-1] - lows[-1], 1e-9)
    s = 0.0
    for j in range(len(closes) - m, len(closes)):
        pc = closes[j - 1]
        s += _true_range(highs[j], lows[j], pc)
    return max(s / m, 1e-9)


def evaluate_trade_strength(trade: Mapping[str, Any], market_state: Mapping[str, Any]) -> float:
    """Fuerza agregada del movimiento a favor del trade [0, 1]."""
    cont = float(market_state.get("continuation_score", 0.0))
    disp = float(market_state.get("displacement_score", 0.0))
    mom = float(market_state.get("momentum_score", cont))
    comp = float(market_state.get("compression_score", 0.0))
    # Menos compresión extrema = más capacidad de desplazamiento
    comp_pen = max(0.0, 1.0 - comp * 1.1)
    return max(0.0, min(1.0, 0.38 * cont + 0.32 * disp + 0.20 * mom + 0.10 * comp_pen))


def detect_momentum_loss(trade: Mapping[str, Any], market_state: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    """True si el impulso a favor se debilita respecto al umbral."""
    thr = _cfg(cfg, "momentum_loss_threshold", 0.38)
    cont = float(market_state.get("continuation_score", 0.0))
    vel_drop = bool(market_state.get("velocity_loss", False))
    if cont < thr:
        return True
    if vel_drop and cont < thr + 0.12:
        return True
    return False


def detect_compression(market_state: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    """True si hay compresión prolongada (rango interno vs ATR)."""
    cth = _cfg(cfg, "compression_threshold", 0.72)
    cs = float(market_state.get("compression_score", 0.0))
    return cs >= cth


def detect_absorption_against_trade(market_state: Mapping[str, Any], trade_direction: str) -> float:
    """Intensidad [0,1] de absorción contraria a la posición (última vela M1)."""
    if trade_direction.upper() == "BUY":
        return float(market_state.get("absorption_against_long", 0.0))
    return float(market_state.get("absorption_against_short", 0.0))


def should_secure_profit(
    trade: Mapping[str, Any],
    market_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    """Modo asegurar beneficio: ya hay R razonable y el movimiento pierde fuelle."""
    r = float(trade.get("current_r", 0.0))
    spr = _cfg(cfg, "secure_profit_r", 0.45)
    if r < spr:
        return False
    if _runner_mode(trade, market_state, cfg):
        return False
    mom_loss = detect_momentum_loss(trade, market_state, cfg)
    lateral = bool(market_state.get("lateral_soft", False)) or detect_compression(market_state, cfg)
    decel = bool(market_state.get("velocity_loss", False))
    return mom_loss or lateral or decel


def should_close_trade(trade: Mapping[str, Any], market_state: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    """Cierre total: reversión peligrosa, giveback excesivo o agotamiento claro."""
    r = float(trade.get("current_r", 0.0))
    max_r = float(trade.get("max_r", 0.0))
    if r <= 0.02:
        return False
    give = max(0.0, max_r - r)
    gb = _cfg(cfg, "max_giveback_allowed", 0.35)
    spr = _cfg(cfg, "secure_profit_r", 0.45)
    if max_r >= spr and give >= gb:
        if _runner_mode(trade, market_state, cfg) and r > 0.35 and give < gb + 0.12:
            return False
        logger.debug("dynamic_exit: giveback max_r=%.2f r=%.2f give=%.2f", max_r, r, give)
        return True
    ab_th = _cfg(cfg, "absorption_strength_threshold", 0.55)
    ab = detect_absorption_against_trade(market_state, str(trade.get("direction", "BUY")))
    if ab >= ab_th and r >= 0.18 and detect_momentum_loss(trade, market_state, cfg):
        return True
    if ab >= min(0.95, ab_th + 0.22) and r >= 0.12:
        return True
    if detect_compression(market_state, cfg) and detect_momentum_loss(trade, market_state, cfg) and r >= 0.22:
        if not _runner_mode(trade, market_state, cfg):
            return True
    return False


def should_move_to_break_even(
    trade: Mapping[str, Any],
    market_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    """BE inteligente: mínimo R, estructura válida, momentum no colapsado, poca absorción contraria."""
    if bool(trade.get("be_done", False)):
        return False
    r = float(trade.get("current_r", 0.0))
    be_r = _cfg(cfg, "breakeven_r", 0.5)
    if r < be_r:
        return False
    if not bool(market_state.get("structure_favorable", True)):
        return False
    cont = float(market_state.get("continuation_score", 0.0))
    if cont < 0.18:
        return False
    ab = detect_absorption_against_trade(market_state, str(trade.get("direction", "BUY")))
    ab_th = _cfg(cfg, "absorption_strength_threshold", 0.55)
    if ab >= ab_th * 0.85:
        return False
    if detect_momentum_loss(trade, market_state, cfg) and cont < 0.28:
        return False
    return True


def should_activate_trailing(
    trade: Mapping[str, Any],
    market_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    """Activa o mantiene trailing adaptativo una vez alcanzado R mínimo."""
    r = float(trade.get("current_r", 0.0))
    arm = _cfg(cfg, "trailing_arm_r", 0.55)
    if r < arm:
        return False
    if should_close_trade(trade, market_state, cfg):
        return False
    strength = evaluate_trade_strength(trade, market_state)
    if strength >= 0.72 and _runner_mode(trade, market_state, cfg):
        return True
    if r >= max(arm, 0.62):
        return True
    return False


def _runner_mode(trade: Mapping[str, Any], market_state: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    cont = float(market_state.get("continuation_score", 0.0))
    disp = float(market_state.get("displacement_score", 0.0))
    cont_score = float(market_state.get("momentum_score", cont))
    if cont < 0.58 or disp < 0.42 or cont_score < 0.52:
        return False
    if detect_compression(market_state, cfg):
        return False
    ab = detect_absorption_against_trade(market_state, str(trade.get("direction", "BUY")))
    if ab >= _cfg(cfg, "absorption_strength_threshold", 0.55) * 0.75:
        return False
    return True


@dataclass(slots=True)
class DynamicExitDecision:
    """Plan de salida para un tick de gestión."""

    close_full: bool = False
    close_partial_fraction: float = 0.0
    move_be: bool = False
    trail_sl_price: float | None = None
    reason_full: str = ""
    reason_partial: str = ""
    reason_be: str = ""
    reason_trail: str = ""
    log_keep: str = ""

    def any_action(self) -> bool:
        return self.close_full or self.close_partial_fraction > 0.0 or self.move_be or self.trail_sl_price is not None


def _progress_to_tp(trade: Mapping[str, Any]) -> float | None:
    """Devuelve el progreso entry→TP en [0,1+] usando R objetivo (tp_pips/sl_pips).

    progress = current_r / tp_r. None si tp_r no es válido.
    """
    try:
        tp_pips = float(trade.get("tp_pips", 0) or 0)
        sl_pips = float(trade.get("sl_pips", 0) or 0)
        r = float(trade.get("current_r", 0) or 0)
    except (TypeError, ValueError):
        return None
    if tp_pips <= 0 or sl_pips <= 0:
        return None
    tp_r = tp_pips / sl_pips
    if tp_r <= 0:
        return None
    return r / tp_r


def _profit_lock_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = cfg.get("profit_lock")
    if isinstance(raw, Mapping) and bool(raw.get("enabled", False)):
        return raw
    return None


def _has_reversal_signal(
    trade: Mapping[str, Any],
    market_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
    plock: Mapping[str, Any],
) -> bool:
    """Detecta señal de reversión para activar el lock 'hard'."""
    if bool(plock.get("reversal_momentum_loss", True)) and detect_momentum_loss(trade, market_state, cfg):
        return True
    ab_min = _cfg(plock, "reversal_absorption_min", 0.0)
    if ab_min > 0:
        ab = detect_absorption_against_trade(market_state, str(trade.get("direction", "BUY")))
        if ab >= ab_min:
            return True
    if bool(plock.get("reversal_velocity_loss", True)) and bool(market_state.get("velocity_loss", False)):
        return True
    if bool(plock.get("reversal_lateral_soft", False)) and bool(market_state.get("lateral_soft", False)):
        return True
    return False


def compute_profit_lock_sl_from_tp(
    trade: Mapping[str, Any],
    market_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    mid: float,
) -> tuple[float | None, str]:
    """Calcula el SL de profit lock según % alcanzado del recorrido entry→TP.

    Reglas (configurables en cfg['profit_lock']):
      • progress ≥ soft_trigger_pct (def. 0.20)
            → SL = entry ± soft_lock_pct * tp_distance (def. 3% lock).
      • progress ≥ hard_trigger_pct (def. 0.35) Y señal de reversión
            → SL = entry ± hard_lock_pct * tp_distance (def. 20% lock).
      • Devuelve (None, "") si ningún lock aplica.
    """
    plock = _profit_lock_cfg(cfg)
    if plock is None:
        return None, ""
    progress = _progress_to_tp(trade)
    if progress is None or progress <= 0:
        return None, ""

    direction = str(trade.get("direction", "BUY")).upper()
    try:
        entry = float(trade.get("entry_price", mid))
        risk = float(trade.get("risk_price", 1e-9))
        sl_pips = float(trade.get("sl_pips", 0) or 0)
        tp_pips = float(trade.get("tp_pips", 0) or 0)
    except (TypeError, ValueError):
        return None, ""
    if sl_pips <= 0 or tp_pips <= 0 or risk <= 0:
        return None, ""
    tp_distance = (tp_pips / sl_pips) * risk

    soft_trig = _cfg(plock, "soft_trigger_pct", 0.20)
    soft_lock = _cfg(plock, "soft_lock_pct", 0.03)
    hard_trig = _cfg(plock, "hard_trigger_pct", 0.35)
    hard_lock = _cfg(plock, "hard_lock_pct", 0.20)

    stage = ""
    lock_pct = 0.0
    if progress >= hard_trig and _has_reversal_signal(trade, market_state, cfg, plock):
        lock_pct = hard_lock
        stage = "hard"
    elif progress >= soft_trig:
        lock_pct = soft_lock
        stage = "soft"
    else:
        return None, ""

    if direction == "BUY":
        lock_sl = entry + lock_pct * tp_distance
    else:
        lock_sl = entry - lock_pct * tp_distance
    return float(lock_sl), stage


def compute_trail_sl_price(
    trade: Mapping[str, Any],
    market_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    mid: float,
) -> float | None:
    """SL absoluto trailing adaptativo (ATR + swings + fuerza)."""
    direction = str(trade.get("direction", "BUY")).upper()
    entry = float(trade.get("entry_price", mid))
    risk = float(trade.get("risk_price", 1e-9))
    highs = list(market_state.get("highs_m1", []) or [])
    lows = list(market_state.get("lows_m1", []) or [])
    closes = list(market_state.get("closes_m1", []) or [])
    if len(closes) < 4:
        return None
    atr = float(market_state.get("atr", _atr_price(highs, lows, closes, 14)))
    mult = _cfg(cfg, "trailing_atr_multiplier", 1.15)
    strength = evaluate_trade_strength(trade, market_state)
    widen = 0.75 + 0.55 * strength
    dist = atr * mult * (2.05 - widen)
    tail = min(5, len(closes))
    if direction == "BUY":
        swing = min(lows[-tail:])
        raw = max(mid - dist, swing - atr * 0.12)
        be_floor = entry + 0.03 * risk
        return max(raw, be_floor)
    swing = max(highs[-tail:])
    raw = min(mid + dist, swing + atr * 0.12)
    be_hi = entry - 0.03 * risk
    return min(raw, be_hi)


def decide_dynamic_exit(
    trade: Mapping[str, Any],
    market_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    mid: float,
) -> DynamicExitDecision:
    """Orquesta prioridades: cierre total > parcial > BE > trailing."""
    out = DynamicExitDecision()
    if not _cfg_bool(cfg, "enabled", False):
        return out

    # PROFIT LOCK precedence: si ya hay hard lock disponible (>=35% TP + reversión)
    # ignoramos giveback / momentum_exhaustion → "deja correr, si saca SL ni modo".
    # Sí cerramos por absorción EXTREMA (peligro real de reversión fuerte).
    plock = _profit_lock_cfg(cfg)
    progress = _progress_to_tp(trade) if plock is not None else None
    hard_trig = _cfg(plock, "hard_trigger_pct", 0.35) if plock is not None else 1.0
    has_hard_lock = bool(
        plock is not None
        and progress is not None
        and progress >= hard_trig
        and _has_reversal_signal(trade, market_state, cfg, plock)
    )

    if should_close_trade(trade, market_state, cfg):
        ab = detect_absorption_against_trade(market_state, str(trade.get("direction", "BUY")))
        ab_th = _cfg(cfg, "absorption_strength_threshold", 0.55)
        if has_hard_lock and ab < min(0.95, ab_th + 0.30):
            # El SL del hard lock ya protege ≥20% del TP; dejamos correr.
            pass
        else:
            if ab >= ab_th:
                out.reason_full = "absorption_against_trade"
            elif float(trade.get("max_r", 0.0)) >= _cfg(cfg, "secure_profit_r", 0.45) and (
                float(trade.get("max_r", 0.0)) - float(trade.get("current_r", 0.0))
            ) >= _cfg(cfg, "max_giveback_allowed", 0.35):
                out.reason_full = "max_giveback"
            else:
                out.reason_full = "momentum_exhaustion"
            out.close_full = True
            return out

    pct = _cfg(cfg, "partial_close_percent", 40.0)
    pct = max(5.0, min(85.0, pct))
    if (
        not bool(trade.get("dynamic_exit_partial_done", False))
        and should_secure_profit(trade, market_state, cfg)
        and float(trade.get("current_r", 0.0)) >= _cfg(cfg, "secure_profit_r", 0.45)
    ):
        out.close_partial_fraction = pct / 100.0
        out.reason_partial = "momentum_loss" if detect_momentum_loss(trade, market_state, cfg) else "secure_profit"
        # NO retornamos: permitimos que BE / profit_lock / trailing también
        # actualicen el SL del volumen restante en el mismo tick.

    if should_move_to_break_even(trade, market_state, cfg):
        out.move_be = True
        out.reason_be = "profit_secure"

    # Profit lock (% TP) + trailing ATR (R-based). Elegimos el SL más favorable
    # entre ambos. El orchestrator garantiza que el SL solo mejora (nunca retrocede).
    plock_sl, plock_stage = compute_profit_lock_sl_from_tp(trade, market_state, cfg, mid=mid)
    atr_sl: float | None = None
    if should_activate_trailing(trade, market_state, cfg):
        atr_sl = compute_trail_sl_price(trade, market_state, cfg, mid=mid)

    direction = str(trade.get("direction", "BUY")).upper()
    chosen: float | None = None
    reason: str = ""
    if plock_sl is not None and atr_sl is not None:
        chosen = max(plock_sl, atr_sl) if direction == "BUY" else min(plock_sl, atr_sl)
        # Si el ATR ganó, etiquetamos como atr; si ganó el lock, como profit_lock.
        if (direction == "BUY" and chosen == plock_sl) or (
            direction == "SELL" and chosen == plock_sl
        ):
            reason = f"profit_lock_{plock_stage}"
        else:
            reason = "continuation_valid" if _runner_mode(trade, market_state, cfg) else "adaptive_atr"
    elif plock_sl is not None:
        chosen = plock_sl
        reason = f"profit_lock_{plock_stage}"
    elif atr_sl is not None:
        chosen = atr_sl
        reason = "continuation_valid" if _runner_mode(trade, market_state, cfg) else "adaptive_atr"

    if chosen is not None:
        out.trail_sl_price = chosen
        out.reason_trail = reason

    if not out.any_action() and float(trade.get("current_r", 0.0)) > 0.08:
        if _runner_mode(trade, market_state, cfg):
            out.log_keep = "KEEP_RUNNING: strong_momentum"
        else:
            out.log_keep = "KEEP_RUNNING: monitoring"
        logger.debug("[%s] ticket=%s r=%.2f", out.log_keep, trade.get("ticket"), float(trade.get("current_r", 0.0)))

    return out


def build_exit_market_state(
    *,
    highs_m1: list[float],
    lows_m1: list[float],
    closes_m1: list[float],
    vols: list[float],
    continuation_score: float,
    structure_trend: str,
    trade_direction: str,
    regime: str,
    spread_points: float,
    mid: float,
) -> dict[str, Any]:
    """Construye el dict de mercado para las funciones de salida (sin acceso a MTF engine)."""
    highs = list(highs_m1)
    lows = list(lows_m1)
    closes = list(closes_m1)
    n = len(closes)
    if n < 4:
        return {
            "highs_m1": highs,
            "lows_m1": lows,
            "closes_m1": closes,
            "continuation_score": continuation_score,
            "momentum_score": continuation_score,
            "displacement_score": 0.0,
            "compression_score": 0.0,
            "velocity_loss": False,
            "lateral_soft": False,
            "absorption_against_long": 0.0,
            "absorption_against_short": 0.0,
            "structure_favorable": True,
            "atr": _atr_price(highs, lows, closes, 14),
            "regime": regime,
            "spread_points": spread_points,
            "mid": mid,
        }

    # Desplazamiento reciente vs precio
    leg = abs(closes[-1] - closes[-min(6, n)])
    disp = min(1.0, leg / max(abs(closes[-min(6, n)]), 1e-12) * 120.0)

    # Velocidad: comparar últimos 3 cierres vs los 3 previos
    v1 = abs(closes[-1] - closes[-3]) if n >= 3 else 0.0
    v0 = abs(closes[-4] - closes[-6]) if n >= 6 else v1
    velocity_loss = v1 < v0 * 0.55 and v0 > 1e-12

    # Compresión: rango corto reciente / ATR
    atr = _atr_price(highs, lows, closes, 14)
    w = min(8, n)
    br = max(highs[-w:]) - min(lows[-w:])
    compression_score = min(1.0, max(0.0, 1.0 - min(1.0, br / max(atr * 2.2, 1e-12))))

    # Lateral suave: pocos cuerpos direccionales
    flips = 0
    for j in range(max(1, n - 6), n):
        if closes[j] - closes[j - 1] > 0:
            flips += 1
    lateral_soft = 1 <= flips <= 4 and disp < 0.35

    hi, lo, cl = highs[-1], lows[-1], closes[-1]
    br1 = max(hi - lo, 1e-12)
    o_proxy = (highs[-2] + lows[-2] + closes[-2]) / 3.0 if n >= 2 else cl
    upper_w = (hi - max(cl, o_proxy)) / br1
    lower_w = (min(cl, o_proxy) - lo) / br1
    close_pos = (cl - lo) / br1
    absorption_against_long = max(0.0, min(1.0, 0.55 * upper_w + 0.45 * (1.0 if close_pos < 0.42 else 0.0)))
    absorption_against_short = max(0.0, min(1.0, 0.55 * lower_w + 0.45 * (1.0 if close_pos > 0.58 else 0.0)))

    st = structure_trend.upper()
    d = trade_direction.upper()
    if d == "BUY":
        structure_favorable = st in ("UP", "RANGE", "NEUTRAL") or st == ""
    else:
        structure_favorable = st in ("DOWN", "RANGE", "NEUTRAL") or st == ""

    mom = max(0.0, min(1.0, 0.55 * continuation_score + 0.45 * disp))

    return {
        "highs_m1": highs,
        "lows_m1": lows,
        "closes_m1": closes,
        "continuation_score": float(continuation_score),
        "momentum_score": float(mom),
        "displacement_score": float(disp),
        "compression_score": float(compression_score),
        "velocity_loss": bool(velocity_loss),
        "lateral_soft": bool(lateral_soft),
        "absorption_against_long": float(absorption_against_long),
        "absorption_against_short": float(absorption_against_short),
        "structure_favorable": bool(structure_favorable),
        "atr": float(atr),
        "regime": regime,
        "spread_points": float(spread_points),
        "mid": float(mid),
    }

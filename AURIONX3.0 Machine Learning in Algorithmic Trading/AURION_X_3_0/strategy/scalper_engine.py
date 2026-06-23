"""Motor de lógica orientado a scalping (horizontes muy cortos, M1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from analytics.logger import get_logger
from market.range_quality import RangeMarketMode, assess_range_at_index

if TYPE_CHECKING:
    from market.tick_handler import Tick

logger = get_logger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "ema_fast": 8,
    "ema_slow": 21,
    "atr_period": 14,
    "sl_pips": 12.0,
    "tp_pips": 18.0,
    "min_bars": 45,
    # Separación mínima |EMA_fast - EMA_slow| / precio (0 = desactivado)
    "min_ema_separation_ratio": 0.00006,
    # Momentum mínimo (c - close[i-3])/close[i-3]; 0 recupera lógica laxa previa
    "min_momentum_pct": 0.00018,
    # Habilita entradas por ruptura de rango con momentum/volumen implícito.
    "range_breakout_enabled": True,
    "range_breakout_lookback": 8,
    "range_breakout_momentum_factor": 0.75,
    "range_breakout_min_momentum_score": 0.55,
    # Filtro de participación real por volumen en vela M1.
    "volume_filter_enabled": True,
    "volume_lookback": 20,
    "min_volume_ratio": 1.10,
    "min_volume_score": 0.50,
    "breakout_min_volume_ratio": 1.20,
    # Mínimo _momentum_score() para aceptar MICRO_PULLBACK / breakout (0 = sin filtro).
    "min_setup_momentum_score": 0.35,
    # Reversión a la media en caja M1 (laterales): compra en piso / venta en techo.
    "range_fade_enabled": True,
    "range_fade_lookback": 32,
    "range_fade_min_width_pct": 0.000055,
    "range_fade_max_width_pct": 0.0055,
    "range_fade_edge_pct": 0.15,
    "range_fade_buy_max_position": 0.24,
    "range_fade_sell_min_position": 0.76,
    "range_fade_sweep_tolerance": 0.12,
    "range_fade_close_body_min": 0.32,
    "range_fade_min_volume_ratio": 0.86,
    "range_fade_sl_pips": 13.0,
    "range_fade_tp_pips": 15.0,
    "range_fade_confidence_boost": 0.06,
    # SL por micro-estructura M1 (distancia en precio; el orquestador convierte a pips del símbolo).
    "use_micro_structure_sl": True,
    "micro_swing_lookback": 8,
    "micro_swing_buffer_frac": 0.000015,
    "micro_sl_atr_floor_mult": 0.55,
}


class ScalperSignal(TypedDict):
    """Salida de ``evaluate_m1`` cuando hay operativa."""

    direction: Literal["BUY", "SELL"]
    sl_pips: float
    tp_pips: float
    confidence_base: float
    setup_type: str
    momentum_score: float
    volume_score: float
    volume_ratio: float
    pullback: NotRequired[bool]
    liquidity_sweep: NotRequired[bool]
    range_fade: NotRequired[bool]
    range_quality_score: NotRequired[float]
    range_chop_score: NotRequired[float]
    range_breakout_probability: NotRequired[float]
    range_market_mode: NotRequired[str]
    sl_price_dist: NotRequired[float]
    tp_price_dist: NotRequired[float]


def _ema_series(closes: list[float], span: int) -> list[float]:
    """EMA exponencial (semilla = primer cierre, ``k = 2/(span+1)``)."""
    if not closes:
        return []
    k = 2.0 / (span + 1)
    out: list[float] = []
    ema = closes[0]
    out.append(ema)
    for c in closes[1:]:
        ema = c * k + ema * (1.0 - k)
        out.append(ema)
    return out


def _true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))


def _atr_last(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
    """Último ATR Wilder a partir de la serie de TR."""
    n = len(closes)
    if n < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, n):
        trs.append(_true_range(highs[i], lows[i], closes[i - 1]))
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0.0
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _confidence_from_context(
    close: float,
    ema_f: float,
    ema_s: float,
    momentum_3: float,
    atr: float,
) -> float:
    """Heurística ``confidence_base`` en ``[0, 1]`` (EMA + impulso + ATR relativo)."""
    denom = abs(close) + 1e-12
    sep = abs(ema_f - ema_s) / denom
    mom = abs(momentum_3)
    atr_rel = atr / denom if atr > 0 else 0.0
    raw = 0.2 + min(sep * 80.0, 0.35) + min(mom * 120.0, 0.25) + min(atr_rel * 60.0, 0.2)
    return float(max(0.0, min(1.0, raw)))


def _momentum_score(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    i: int,
) -> float:
    """Score [0,1] de impulso real para filtrar rupturas débiles/fakeouts."""
    if i < 6:
        return 0.0
    last_range = max(highs[i] - lows[i], 1e-12)
    avg_range = sum(max(highs[j] - lows[j], 1e-12) for j in range(i - 5, i)) / 5.0
    range_expansion = min(1.0, last_range / max(avg_range, 1e-12))
    velocity = abs(closes[i] - closes[i - 2]) / max(abs(closes[i - 2]), 1e-12)
    velocity_score = min(1.0, velocity * 250.0)
    follow_through = 1.0 if (closes[i] > closes[i - 1] > closes[i - 2]) or (closes[i] < closes[i - 1] < closes[i - 2]) else 0.0
    return max(0.0, min(1.0, 0.45 * range_expansion + 0.35 * velocity_score + 0.20 * follow_through))


def _volume_score(volumes: list[float], i: int, lookback: int) -> tuple[float, float]:
    """Score [0,1] y ratio de volumen actual vs baseline reciente."""
    if i < 2 or not volumes:
        return 0.0, 0.0
    left = max(0, i - max(5, lookback))
    hist = [max(0.0, float(v)) for v in volumes[left:i]]
    if not hist:
        return 0.0, 0.0
    baseline = sum(hist) / max(len(hist), 1)
    current = max(0.0, float(volumes[i]))
    ratio = current / max(baseline, 1e-9)
    score = max(0.0, min(1.0, ratio / 2.0))
    return score, ratio


def _eval_range_fade(
    hf: list[float],
    lf: list[float],
    cf: list[float],
    vf: list[float],
    i: int,
    params: dict[str, Any],
) -> ScalperSignal | None:
    """Reversión en lateral M1: compras cerca del soporte inferido, ventas en resistencia."""
    if not bool(params.get("range_fade_enabled", False)):
        return None
    min_b = int(params["min_bars"])
    if i < max(min_b, 10):
        return None
    L = max(10, int(params.get("range_fade_lookback", 32)))
    left = max(0, i - L)
    window_h = hf[left : i + 1]
    window_l = lf[left : i + 1]
    if len(window_h) < 8:
        return None
    rh = max(window_h)
    rl = min(window_l)
    c = cf[i]
    hi, lo = hf[i], lf[i]
    if c == 0 or rh <= rl:
        return None
    width_pct = (rh - rl) / abs(c)
    min_w = float(params.get("range_fade_min_width_pct", 0.000055))
    max_w = float(params.get("range_fade_max_width_pct", 0.0055))
    if width_pct < min_w or width_pct > max_w:
        return None
    pos = (c - rl) / (rh - rl)
    pos = max(0.0, min(1.0, pos))
    rng = rh - rl

    rq_cfg = params.get("range_quality") if isinstance(params.get("range_quality"), dict) else {}
    rq_report = assess_range_at_index(hf, lf, cf, vf, i, L, rh, rl, cfg=rq_cfg)
    c_lo = float(rq_cfg.get("center_deadband_low", 0.38))
    c_hi = float(rq_cfg.get("center_deadband_high", 0.62))
    if c_lo < pos < c_hi:
        logger.debug("range_fade: rechazo centro rango pos=%.3f (%.2f-%.2f)", pos, c_lo, c_hi)
        return None
    if rq_report.mode == RangeMarketMode.CHOP:
        logger.debug("range_fade: mercado CHOP (chop=%.1f range=%.1f)", rq_report.chop_score, rq_report.range_score)
        return None
    if rq_report.mode == RangeMarketMode.BREAKOUT_WATCH and bool(rq_cfg.get("skip_fade_on_breakout_watch", True)):
        logger.debug(
            "range_fade: BREAKOUT_WATCH bo=%.1f (fade desactivado en este modo)",
            rq_report.breakout_probability,
        )
        return None
    min_rs = float(rq_cfg.get("min_range_score", 42.0))
    max_ch = float(rq_cfg.get("max_chop_score", 58.0))
    if rq_report.range_score < min_rs:
        logger.debug("range_fade: range_score %.1f < %.1f", rq_report.range_score, min_rs)
        return None
    if rq_report.chop_score > max_ch + 6.0:
        logger.debug("range_fade: chop_score %.1f excesivo", rq_report.chop_score)
        return None

    min_ab = float(rq_cfg.get("min_absorption_edge", 0.30))
    sweep_tol = float(params.get("range_fade_sweep_tolerance", 0.12))
    body_min = float(params.get("range_fade_close_body_min", 0.32))
    buy_max_pos = float(params.get("range_fade_buy_max_position", 0.24))
    sell_min_pos = float(params.get("range_fade_sell_min_position", 0.76))

    vol_filter_enabled = bool(params.get("volume_filter_enabled", True))
    vol_score, vol_ratio = _volume_score(vf, i, int(params.get("volume_lookback", 20)))
    min_vol = float(params.get("range_fade_min_volume_ratio", 0.86))
    win_v = vf[max(0, i - 25) : i + 1]
    has_tick_vol = bool(win_v) and max(abs(float(x)) for x in win_v) > 1e-12
    if vol_filter_enabled and has_tick_vol and vol_ratio < min_vol:
        return None

    bar_range = max(hi - lo, 1e-12)
    close_from_low = (c - lo) / bar_range
    close_from_high = (hi - c) / bar_range

    sweep_low = lo <= rl + sweep_tol * rng
    sweep_high = hi >= rh - sweep_tol * rng
    bull_reject = sweep_low and close_from_low >= body_min and c >= cf[i - 1]
    bear_reject = sweep_high and close_from_high >= body_min and c <= cf[i - 1]

    buy_ok = pos <= buy_max_pos and bull_reject and rq_report.absorption_buy >= min_ab
    sell_ok = pos >= sell_min_pos and bear_reject and rq_report.absorption_sell >= min_ab

    if buy_ok and sell_ok:
        buy_ok = pos <= (1.0 - pos)
        sell_ok = not buy_ok
    if buy_ok and not sell_ok:
        direction: Literal["BUY", "SELL"] = "BUY"
    elif sell_ok and not buy_ok:
        direction = "SELL"
    else:
        return None

    ef_n = int(params["ema_fast"])
    es_n = int(params["ema_slow"])
    ap = int(params["atr_period"])
    ema_fast = _ema_series(cf, ef_n)
    ema_slow = _ema_series(cf, es_n)
    atr_i = _atr_last(hf, lf, cf, ap)
    ef_i, es_i = ema_fast[i], ema_slow[i]
    mom3 = (c - cf[i - 3]) / cf[i - 3] if i >= 3 and cf[i - 3] != 0 else 0.0
    conf = _confidence_from_context(c, ef_i, es_i, mom3, atr_i)
    edge_qual = 1.0 - (pos / buy_max_pos) if direction == "BUY" else (pos - sell_min_pos) / max(1e-9, 1.0 - sell_min_pos)
    edge_qual = max(0.0, min(1.0, edge_qual))
    boost = float(params.get("range_fade_confidence_boost", 0.06))
    conf = float(max(0.0, min(1.0, conf + boost * (0.5 + 0.5 * edge_qual))))
    rq_boost = min(0.12, max(0.0, (rq_report.range_score - 50.0) / 400.0))
    conf = float(max(0.0, min(1.0, conf + rq_boost)))
    mom_score = _momentum_score(hf, lf, cf, i)

    sl_p = float(params.get("range_fade_sl_pips", 13.0))
    tp_p = float(params.get("range_fade_tp_pips", 15.0))
    mid = (rh + rl) / 2.0
    if direction == "BUY":
        dist_mid = max(mid - c, 0.0)
    else:
        dist_mid = max(c - mid, 0.0)
    if dist_mid > 0 and rng > 0:
        tp_scale = min(1.35, max(1.0, (dist_mid / rng) * 2.2))
        tp_p = float(max(tp_p, min(float(params.get("range_fade_tp_pips", 15.0)) * tp_scale, 48.0)))

    out: ScalperSignal = {
        "direction": direction,
        "sl_pips": sl_p,
        "tp_pips": tp_p,
        "confidence_base": conf,
        "setup_type": "RANGE_FADE_REVERSAL",
        "momentum_score": max(mom_score, 0.42),
        "volume_score": vol_score,
        "volume_ratio": vol_ratio,
        "pullback": True,
        "liquidity_sweep": True,
        "range_fade": True,
        "range_quality_score": rq_report.range_score,
        "range_chop_score": rq_report.chop_score,
        "range_breakout_probability": rq_report.breakout_probability,
        "range_market_mode": str(rq_report.mode.value),
    }
    logger.debug("range_fade: %s pos=%.3f width_pct=%.5f", direction, pos, width_pct)
    return out


class ScalperEngine:
    """Señal rápida M1: micro pullback + continuación y confirmación EMA / impulso."""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        """Hiperparámetros opcionales (mezclados con defaults).

        Claves típicas: ``ema_fast``, ``ema_slow``, ``atr_period``, ``sl_pips``,
        ``tp_pips``, ``min_bars``.
        """
        self._params = {**_DEFAULT_PARAMS, **(parameters or {})}
        logger.debug("ScalperEngine params=%s", self._params)

    @property
    def min_bars(self) -> int:
        """Velas M1 mínimas para evaluar (sin contar la vela en formación si se excluye)."""
        return int(self._params["min_bars"])

    def evaluate_m1(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float] | None = None,
    ) -> ScalperSignal | None:
        """Genera señal en M1 si hay pullback micro + continuación y confirmación.

        **BUY**: EMA rápida por encima de la lenta; la vela previa retrocede (mínimo o cierre
        bajo la zona de la EMA rápida); la última vela continúa al alza y cierra por encima
        de ambas EMAs; impulso reciente coherente.

        **SELL**: condición espejo.

        Args:
            highs, lows, closes: Series M1 alineadas (misma longitud).

        Returns:
            ``ScalperSignal`` con ``direction``, ``sl_pips``, ``tp_pips``,
            ``confidence_base``; o ``None`` si no hay setup o datos insuficientes.
        """
        min_b = int(self._params["min_bars"])
        if len(closes) != len(highs) or len(closes) != len(lows) or len(closes) < min_b:
            logger.debug("evaluate_m1: datos insuficientes o longitudes distintas")
            return None

        hf = [float(x) for x in highs]
        lf = [float(x) for x in lows]
        cf = [float(x) for x in closes]
        vf = [float(x) for x in volumes] if volumes is not None and len(volumes) == len(closes) else [0.0] * len(closes)

        i = len(cf) - 1
        fade_sig = _eval_range_fade(hf, lf, cf, vf, i, self._params)
        if fade_sig is not None:
            return fade_sig

        ef_n = int(self._params["ema_fast"])
        es_n = int(self._params["ema_slow"])
        ap = int(self._params["atr_period"])

        ema_fast = _ema_series(cf, ef_n)
        ema_slow = _ema_series(cf, es_n)
        atr_i = _atr_last(hf, lf, cf, ap)

        i1 = i - 1
        i2 = i - 2
        if i2 < 0:
            return None

        ef_i, es_i = ema_fast[i], ema_slow[i]
        ef_i1, es_i1 = ema_fast[i1], ema_slow[i1]
        c, c1 = cf[i], cf[i1]
        l1 = lf[i1]

        min_sep = float(self._params.get("min_ema_separation_ratio", 0.0))
        if min_sep > 0.0:
            sep_ratio = abs(ef_i - es_i) / (abs(c) + 1e-12)
            if sep_ratio < min_sep:
                logger.debug("evaluate_m1: separación EMA insuficiente")
                return None

        mom3 = (c - cf[i - 3]) / cf[i - 3] if i >= 3 and cf[i - 3] != 0 else 0.0
        mom_thr = float(self._params.get("min_momentum_pct", 0.0))
        vol_score, vol_ratio = _volume_score(vf, i, int(self._params.get("volume_lookback", 20)))
        vol_filter_enabled = bool(self._params.get("volume_filter_enabled", True))
        min_vol_ratio = float(self._params.get("min_volume_ratio", 1.10))
        min_vol_score = float(self._params.get("min_volume_score", 0.50))

        bull_regime = ef_i > es_i and ef_i1 >= es_i1 * 0.9999
        bear_regime = ef_i < es_i and ef_i1 <= es_i1 * 1.0001

        pullback_buy = l1 <= ef_i1 * 1.0002 or c1 < ef_i1
        continuation_buy = c > c1 and c >= ef_i and c > es_i
        if mom_thr > 0.0:
            impulse_buy = mom3 > mom_thr and c > c1
        else:
            impulse_buy = mom3 > 0 or c > c1

        pullback_sell = lf[i1] >= ef_i1 * 0.9998 or c1 > ef_i1
        continuation_sell = c < c1 and c <= ef_i and c < es_i
        if mom_thr > 0.0:
            impulse_sell = mom3 < -mom_thr and c < c1
        else:
            impulse_sell = mom3 < 0 or c < c1

        buy_ok = bull_regime and pullback_buy and continuation_buy and impulse_buy
        sell_ok = bear_regime and pullback_sell and continuation_sell and impulse_sell
        setup_type = "MICRO_PULLBACK_MOMENTUM"
        if vol_filter_enabled and (vol_ratio < min_vol_ratio or vol_score < min_vol_score):
            buy_ok = False
            sell_ok = False

        # Modo adicional: ruptura de rango cuando el mercado acelera con vela real.
        range_breakout_enabled = bool(self._params.get("range_breakout_enabled", True))
        if range_breakout_enabled and i >= 8:
            rb_lookback = max(5, int(self._params.get("range_breakout_lookback", 8)))
            rb_mom_factor = float(self._params.get("range_breakout_momentum_factor", 0.75))
            rb_min_mom_score = float(self._params.get("range_breakout_min_momentum_score", 0.55))
            rb_min_vol_ratio = float(self._params.get("breakout_min_volume_ratio", 1.20))
            left = max(0, i - rb_lookback)
            range_high = max(cf[left:i]) if i > left else c
            range_low = min(cf[left:i]) if i > left else c
            mom_thr_eff = mom_thr * rb_mom_factor if mom_thr > 0.0 else 0.0
            mom_score_now = _momentum_score(hf, lf, cf, i)
            buy_range_breakout = (
                c > range_high
                and c > ef_i
                and c >= c1
                and mom3 >= mom_thr_eff
                and mom_score_now >= rb_min_mom_score
                and (not vol_filter_enabled or vol_ratio >= rb_min_vol_ratio)
            )
            sell_range_breakout = (
                c < range_low
                and c < ef_i
                and c <= c1
                and mom3 <= -mom_thr_eff
                and mom_score_now >= rb_min_mom_score
                and (not vol_filter_enabled or vol_ratio >= rb_min_vol_ratio)
            )
            buy_ok = buy_ok or buy_range_breakout
            sell_ok = sell_ok or sell_range_breakout

        if buy_ok and not sell_ok:
            direction: Literal["BUY", "SELL"] = "BUY"
        elif sell_ok and not buy_ok:
            direction = "SELL"
        elif buy_ok and sell_ok:
            direction = "BUY" if (c - es_i) >= (es_i - c) else "SELL"
        else:
            return None

        conf = _confidence_from_context(c, ef_i, es_i, mom3, atr_i)
        mom_score = _momentum_score(hf, lf, cf, i)
        mom_floor = float(self._params.get("min_setup_momentum_score", 0.35))
        if mom_floor > 0.0 and mom_score < mom_floor:
            return None
        sl_p = float(self._params["sl_pips"])
        tp_p = float(self._params["tp_pips"])
        sl_price_dist: float | None = None
        if bool(self._params.get("use_micro_structure_sl", False)):
            try:
                lb = max(3, int(self._params.get("micro_swing_lookback", 8)))
            except (TypeError, ValueError):
                lb = 8
            try:
                bufff = float(self._params.get("micro_swing_buffer_frac", 0.000015) or 0.000015)
            except (TypeError, ValueError):
                bufff = 0.000015
            buf_abs = max(abs(c) * bufff, 1e-9)
            left = max(0, i - lb + 1)
            try:
                atr_f = float(self._params.get("micro_sl_atr_floor_mult", 0.55) or 0.55)
            except (TypeError, ValueError):
                atr_f = 0.55
            atr_floor = max(float(atr_i) * max(0.15, atr_f), 1e-9)
            if direction == "BUY":
                swing_lo = min(lf[left : i + 1])
                struct_sl = max(c - swing_lo, 1e-9) + buf_abs
            else:
                swing_hi = max(hf[left : i + 1])
                struct_sl = max(swing_hi - c, 1e-9) + buf_abs
            sl_price_dist = max(struct_sl, atr_floor)
            sl_p = 0.0

        breakout_window_high = max(cf[max(0, i - 6):i]) if i > 0 else c
        breakout_window_low = min(cf[max(0, i - 6):i]) if i > 0 else c
        is_breakout = c > breakout_window_high or c < breakout_window_low
        if range_breakout_enabled and i >= 8:
            rb_left = max(0, i - max(5, int(self._params.get("range_breakout_lookback", 8))))
            rb_high = max(cf[rb_left:i]) if i > rb_left else c
            rb_low = min(cf[rb_left:i]) if i > rb_left else c
            if c > rb_high or c < rb_low:
                setup_type = "RANGE_BREAKOUT_MOMENTUM"
            elif is_breakout:
                setup_type = "BREAKOUT_CONTINUATION"
        elif is_breakout:
            setup_type = "BREAKOUT_CONTINUATION"

        out: ScalperSignal = {
            "direction": direction,
            "sl_pips": sl_p,
            "tp_pips": tp_p,
            "confidence_base": conf,
            "setup_type": setup_type,
            "momentum_score": mom_score,
            "volume_score": vol_score,
            "volume_ratio": vol_ratio,
        }
        if sl_price_dist is not None:
            out["sl_price_dist"] = float(sl_price_dist)
        logger.debug("evaluate_m1: %s conf=%.3f", direction, conf)
        return out

    def on_tick(self, tick: Tick) -> None:
        """Reservado para estado intrabar basado en ticks (sin lógica aún)."""
        raise NotImplementedError

    def signal(self) -> str | None:
        """Reservado para última señal discreta si se usa ``on_tick``."""
        raise NotImplementedError

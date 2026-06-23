"""Motor intradía: estructura + ruptura + continuación en M1/M5."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from analytics.logger import get_logger

from .structure_detector import StructureDetector, StructureResult, Trend

if TYPE_CHECKING:
    from market.tick_handler import Tick

logger = get_logger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "ema_fast_span": 13,
    "ema_slow_span": 34,
    "min_m1_bars": 55,
    "max_m1_bars": 700,
    "break_buffer": 0.00011,
    "sl_buffer": 0.00010,
    "tp_rr": 1.85,
    # Tope de riesgo estructural en múltiplos de ATR implícito (vía rango M1 reciente).
    "max_sl_range_frac": 0.55,
    # Calidad > cantidad: pullbacks desactivados por defecto (menos ruido).
    "enable_pullback_entries": False,
    "min_structure_key_levels": 3,
    # Ruptura debe superar el buffer × este factor (evita “tocar y volver”).
    "break_extension_mult": 1.35,
    # |EMA_fast − EMA_slow| / precio mínimo (0 = no exigir).
    "min_ema_separation_ratio": 0.000048,
    # Exige sesgo M1 (bias) alineado con la operación.
    "require_m1_bias_align": True,
    # Cierre de la última vela fuerte en la dirección (cuerpo / rango barra).
    "min_breakout_body_ratio": 0.58,
    "pullback_min_body_ratio": 0.66,
    # Movimiento mínimo vela vs vela anterior (fracción del precio).
    "min_bar_momentum_pct": 0.00004,
    # TP hacia zona opuesta estructural (soporte en venta / resistencia en compra).
    "use_structure_tp_zone": True,
    "tp_zone_buffer_frac": 0.000018,
}


class IntradaySignal(TypedDict):
    """Salida de señal intradía."""

    direction: Literal["BUY", "SELL"]
    sl_pips: float
    tp_pips: float
    # Distancia en precio (orquestador convierte con pip_size del símbolo).
    sl_price_dist: NotRequired[float]
    tp_price_dist: NotRequired[float]
    confidence_base: float
    setup_type: NotRequired[str]
    momentum_score: NotRequired[float]
    volume_score: NotRequired[float]
    volume_ratio: NotRequired[float]
    pullback: NotRequired[bool]
    range_fade: NotRequired[bool]
    liquidity_sweep: NotRequired[bool]


def _ema_last(closes: list[float], span: int) -> float:
    if not closes or span < 1:
        return closes[-1] if closes else 0.0
    k = 2.0 / (span + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1.0 - k)
    return ema


class IntradayEngine:
    """Señal intradía basada en estructura, ruptura y continuación."""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        """Parámetros: EMAs, buffers de ruptura/SL y RR objetivo para TP."""
        self._params = {**_DEFAULT_PARAMS, **(parameters or {})}
        self._m1_closes: list[float] = []
        self._m1_highs: list[float] = []
        self._m1_lows: list[float] = []
        self._last_mid: float | None = None
        self._structure = StructureDetector(left_bars=2, right_bars=2)
        logger.debug("IntradayEngine creado params=%s", self._params)

    def push_m1_bar(self, high: float, low: float, close: float) -> None:
        """Registra una vela M1 completa."""
        self._m1_highs.append(float(high))
        self._m1_lows.append(float(low))
        self._m1_closes.append(float(close))
        max_b = int(self._params["max_m1_bars"])
        if len(self._m1_closes) > max_b:
            self._m1_highs = self._m1_highs[-max_b:]
            self._m1_lows = self._m1_lows[-max_b:]
            self._m1_closes = self._m1_closes[-max_b:]

    def set_m1_ohlc(self, highs: list[float], lows: list[float], closes: list[float]) -> None:
        """Reemplaza buffers M1 con OHLC agregado (p. ej. desde MultiTimeframeEngine)."""
        max_b = int(self._params["max_m1_bars"])
        self._m1_highs = [float(x) for x in highs[-max_b:]]
        self._m1_lows = [float(x) for x in lows[-max_b:]]
        self._m1_closes = [float(x) for x in closes[-max_b:]]

    def push_m1_close(self, close: float) -> None:
        """Compatibilidad: solo cierre; high/low se aproximan al cierre."""
        c = float(close)
        self.push_m1_bar(c, c, c)

    def on_tick(self, tick: Tick) -> None:
        """Mantiene último mid para contexto."""
        self._last_mid = float((tick.bid + tick.ask) / Decimal(2))

    def bias(self) -> int:
        """Sesgo EMA rápido/lento: +1, -1, 0."""
        c = self._m1_closes
        if len(c) < int(self._params["min_m1_bars"]):
            return 0
        fast = _ema_last(c, int(self._params["ema_fast_span"]))
        slow = _ema_last(c, int(self._params["ema_slow_span"]))
        if fast > slow * 1.0002:
            return 1
        if fast < slow / 1.0002:
            return -1
        return 0

    def _intraday_passes_quality(
        self,
        setup_tag: str,
        direction: Literal["BUY", "SELL"],
        c_now: float,
        c_prev: float,
        fast: float,
        slow: float,
    ) -> bool:
        """Filtros estrictos: separación EMA, sesgo M1, cuerpo de vela y momentum mínimo."""
        try:
            min_sep = float(self._params.get("min_ema_separation_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_sep = 0.0
        if min_sep > 0.0:
            sep_r = abs(fast - slow) / (abs(c_now) + 1e-12)
            if sep_r < min_sep:
                return False
        if direction == "BUY" and not (fast > slow):
            return False
        if direction == "SELL" and not (fast < slow):
            return False
        if bool(self._params.get("require_m1_bias_align", True)):
            b = self.bias()
            if direction == "BUY" and b != 1:
                return False
            if direction == "SELL" and b != -1:
                return False
        hi_l = self._m1_highs[-1]
        lo_l = self._m1_lows[-1]
        rng = hi_l - lo_l
        if rng <= 0.0:
            return False
        if direction == "BUY":
            body_ratio = (c_now - lo_l) / rng
        else:
            body_ratio = (hi_l - c_now) / rng
        try:
            if setup_tag == "INTRADAY_PULLBACK":
                min_body = float(self._params.get("pullback_min_body_ratio", 0.66))
            else:
                min_body = float(self._params.get("min_breakout_body_ratio", 0.58))
        except (TypeError, ValueError):
            min_body = 0.58
        if body_ratio < min_body:
            return False
        try:
            min_mom = float(self._params.get("min_bar_momentum_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_mom = 0.0
        if min_mom > 0.0:
            mom_pct = abs(c_now - c_prev) / (abs(c_prev) + 1e-12)
            if mom_pct < min_mom:
                return False
        return True

    @staticmethod
    def _tp_dist_to_structure_zone(
        direction: Literal["BUY", "SELL"],
        entry: float,
        key_levels: list[float],
        highs: list[float],
        lows: list[float],
        risk: float,
        rr_min: float,
        rr_max: float,
        zone_buf_frac: float,
    ) -> float:
        """Distancia en precio hasta TP anclado en zona fuerte (soporte demanda / resistencia)."""
        win = min(24, len(lows), len(highs))
        if win < 2:
            return max(risk * rr_min, 1e-6)
        lo_w = lows[-win:]
        hi_w = highs[-win:]
        buf = max(abs(entry) * zone_buf_frac, 1e-9)
        if direction == "SELL":
            below = [float(x) for x in key_levels if float(x) < entry - 1e-9]
            zone = max(below) if below else float(min(lo_w))
            raw = max(0.0, entry - zone) + buf
        else:
            above = [float(x) for x in key_levels if float(x) > entry + 1e-9]
            zone = min(above) if above else float(max(hi_w))
            raw = max(0.0, zone - entry) + buf
        lo_tp = max(risk * rr_min, 1e-6)
        hi_tp = max(risk * rr_max, lo_tp + 1e-9)
        if raw <= 0.0:
            return lo_tp
        return max(lo_tp, min(raw, hi_tp))

    def evaluate_intraday(self, structure_hint: StructureResult | None = None) -> IntradaySignal | None:
        """Genera señal por estructura + ruptura o continuación / pullback en M1.

        Si ``structure_hint`` (p. ej. M15 del orquestador) está en UP/DOWN, se usa esa
        tendencia y niveles; evita el caso en que M1 seguía en RANGE y nunca había intradía
        pese a M15 direccional.

        Lógica:
        - Ruptura clásica: cierre supera nivel clave + continuación con EMAs.
        - Pullback (opcional): tendencia UP/DOWN, retoque a EMA rápida y rechazo a favor.
        """
        n = len(self._m1_closes)
        if n < int(self._params["min_m1_bars"]) or n != len(self._m1_highs) or n != len(self._m1_lows):
            return None

        highs_d = [Decimal(str(x)) for x in self._m1_highs]
        lows_d = [Decimal(str(x)) for x in self._m1_lows]
        closes_d = [Decimal(str(x)) for x in self._m1_closes]
        if structure_hint is not None and structure_hint.trend != Trend.RANGE:
            structure = structure_hint
        else:
            structure = self._structure.detect(highs_d, lows_d, closes_d)
            if structure.trend == Trend.RANGE:
                return None

        min_kl = max(1, int(self._params.get("min_structure_key_levels", 3)))
        if len(structure.key_levels) < min_kl:
            return None

        c = self._m1_closes
        i = n - 1
        c_now = c[i]
        c_prev = c[i - 1]
        fast = _ema_last(c, int(self._params["ema_fast_span"]))
        slow = _ema_last(c, int(self._params["ema_slow_span"]))
        buf = float(self._params["break_buffer"])
        sl_buf = float(self._params["sl_buffer"])
        rr = float(self._params["tp_rr"])
        pull_en = bool(self._params.get("enable_pullback_entries", False))
        try:
            br_mult = max(1.0, float(self._params.get("break_extension_mult", 1.35)))
        except (TypeError, ValueError):
            br_mult = 1.35
        eff_buf = buf * br_mult

        # Niveles referidos a la vela anterior: ruptura = cierre actual vs techo/suelo que
        # acotaban el cierre previo (compra: evita nearest_up >= c_now; venta: análogo).
        kp = [float(x) for x in structure.key_levels]
        upper_levels = [x for x in kp if x >= c_prev]
        lower_levels = [x for x in kp if x <= c_prev]
        win = min(24, n)
        hh_recent = max(self._m1_highs[-win:])
        ll_recent = min(self._m1_lows[-win:])
        if upper_levels:
            nearest_up = min(upper_levels)
        elif kp:
            nearest_up = max(kp)
        else:
            nearest_up = max(c_prev, hh_recent)
        if lower_levels:
            nearest_down = max(lower_levels)
        elif kp:
            nearest_down = min(kp)
        else:
            nearest_down = min(c_prev, ll_recent)

        buy_break = c_now > (nearest_up + eff_buf)
        sell_break = c_now < (nearest_down - eff_buf)
        buy_cont = c_now > c_prev and fast > slow and c_now > fast
        sell_cont = c_now < c_prev and fast < slow and c_now < fast
        pullback_buy = (
            pull_en
            and structure.trend == Trend.UP
            and fast > slow
            and c_prev <= fast * 1.0001
            and c_now > fast
            and c_now > c_prev
        )
        pullback_sell = (
            pull_en
            and structure.trend == Trend.DOWN
            and fast < slow
            and c_prev >= fast * 0.9999
            and c_now < fast
            and c_now < c_prev
        )

        direction: Literal["BUY", "SELL"] | None = None
        setup_tag = "INTRADAY_BREAKOUT"
        if structure.trend == Trend.UP and buy_break and buy_cont:
            direction = "BUY"
        elif structure.trend == Trend.DOWN and sell_break and sell_cont:
            direction = "SELL"
        elif structure.trend == Trend.UP and pullback_buy:
            direction = "BUY"
            setup_tag = "INTRADAY_PULLBACK"
        elif structure.trend == Trend.DOWN and pullback_sell:
            direction = "SELL"
            setup_tag = "INTRADAY_PULLBACK"
        else:
            return None

        if not self._intraday_passes_quality(setup_tag, direction, c_now, c_prev, fast, slow):
            return None

        # SL estructural y TP por RR (tope de riesgo vs rango M1 reciente).
        win = min(24, n)
        range_m1 = max(self._m1_highs[-win:]) - min(self._m1_lows[-win:])
        try:
            max_risk_frac = max(0.15, min(1.0, float(self._params.get("max_sl_range_frac", 0.55))))
        except (TypeError, ValueError):
            max_risk_frac = 0.55
        if direction == "BUY":
            sl_price = min(self._m1_lows[-8:]) - sl_buf
            risk = max(c_now - sl_price, 1e-6)
        else:
            sl_price = max(self._m1_highs[-8:]) + sl_buf
            risk = max(sl_price - c_now, 1e-6)
        if range_m1 > 0.0:
            risk = min(risk, range_m1 * max_risk_frac)
        rr = min(float(self._params["tp_rr"]), 2.0)
        sl = risk
        if bool(self._params.get("use_structure_tp_zone", True)):
            try:
                zbf = float(self._params.get("tp_zone_buffer_frac", 0.000018) or 0.000018)
            except (TypeError, ValueError):
                zbf = 0.000018
            tp = self._tp_dist_to_structure_zone(
                direction,
                c_now,
                kp,
                self._m1_highs,
                self._m1_lows,
                risk,
                max(1.05, float(self._params.get("tp_zone_min_rr", 1.15))),
                rr,
                zbf,
            )
        else:
            tp = risk * rr

        # No usar factor 10000 (FX): confunde metales; el orquestador toma sl_price_dist/tp_price_dist.
        sl_pips = 0.0
        tp_pips = 0.0

        sep = abs(fast - slow) / (abs(c_now) + 1e-12)
        momentum = abs(c_now - c_prev) / (abs(c_prev) + 1e-12)
        conf = max(0.0, min(1.0, 0.28 + min(sep * 120.0, 0.38) + min(momentum * 280.0, 0.32)))
        if setup_tag == "INTRADAY_PULLBACK":
            conf = min(0.92, conf + 0.02)

        out: IntradaySignal = {
            "direction": direction,
            "sl_pips": float(sl_pips),
            "tp_pips": float(tp_pips),
            "sl_price_dist": float(risk),
            "tp_price_dist": float(tp),
            "confidence_base": float(conf),
            "setup_type": setup_tag,
            "momentum_score": float(min(1.0, momentum * 80.0)),
            "volume_score": 0.52,
            "volume_ratio": 1.05,
        }
        logger.debug(
            "evaluate_intraday: %s trend=%s setup=%s sl_dist=%.4f tp_dist=%.4f conf=%.3f",
            direction,
            structure.trend.value,
            setup_tag,
            out["sl_price_dist"],
            out["tp_price_dist"],
            out["confidence_base"],
        )
        return out

    def evaluate_range_structure_reversion(
        self,
        structure: StructureResult,
    ) -> IntradaySignal | None:
        """Operativa en lateral usando la caja del TF de estructura (p. ej. M15) + confirmación M1."""
        if structure.trend != Trend.RANGE or structure.position_in_range is None:
            return None
        n = len(self._m1_closes)
        min_m1 = max(30, int(self._params["min_m1_bars"]) // 2)
        if n < min_m1 or n != len(self._m1_highs) or n != len(self._m1_lows):
            return None

        pos = float(structure.position_in_range)
        c = self._m1_closes[-1]
        c1 = self._m1_closes[-2]
        hi = self._m1_highs[-1]
        lo = self._m1_lows[-1]
        rh_f = float(structure.range_high) if structure.range_high is not None else max(self._m1_highs[-24:])
        rl_f = float(structure.range_low) if structure.range_low is not None else min(self._m1_lows[-24:])
        if rh_f <= rl_f or c <= 0:
            return None
        rng = rh_f - rl_f
        edge = 0.13 * rng
        bar_r = max(hi - lo, 1e-12)
        close_from_low = (c - lo) / bar_r
        close_from_high = (hi - c) / bar_r

        buy_ok = (
            pos <= 0.27
            and lo <= rl_f + edge * 1.15
            and close_from_low >= 0.32
            and c >= c1
        )
        sell_ok = (
            pos >= 0.73
            and hi >= rh_f - edge * 1.15
            and close_from_high >= 0.32
            and c <= c1
        )
        direction: Literal["BUY", "SELL"] | None = None
        if buy_ok and not sell_ok:
            direction = "BUY"
        elif sell_ok and not buy_ok:
            direction = "SELL"
        elif buy_ok and sell_ok:
            direction = "BUY" if pos <= (1.0 - pos) else "SELL"
        if direction is None:
            return None

        sl_buf = float(self._params["sl_buffer"])
        rr = min(float(self._params["tp_rr"]), 1.35)
        try:
            max_risk_frac = max(0.15, min(1.0, float(self._params.get("max_sl_range_frac", 0.45))))
        except (TypeError, ValueError):
            max_risk_frac = 0.45
        if direction == "BUY":
            sl_price = rl_f - sl_buf * max(rng, abs(c) * 1e-4)
            risk = max(c - sl_price, 1e-6)
        else:
            sl_price = rh_f + sl_buf * max(rng, abs(c) * 1e-4)
            risk = max(sl_price - c, 1e-6)
        if rng > 0.0:
            risk = min(risk, rng * max_risk_frac)
        tp = risk * rr
        sl_pips = 0.0
        tp_pips = 0.0

        extremity = (0.27 - pos) / 0.27 if direction == "BUY" else (pos - 0.73) / 0.27
        extremity = max(0.0, min(1.0, extremity))
        conf = float(max(0.38, min(0.92, 0.40 + 0.38 * extremity)))

        out: IntradaySignal = {
            "direction": direction,
            "sl_pips": float(sl_pips),
            "tp_pips": float(tp_pips),
            "sl_price_dist": float(risk),
            "tp_price_dist": float(tp),
            "confidence_base": conf,
            "setup_type": "RANGE_STRUCTURE_M15_FADE",
            "momentum_score": 0.48,
            "volume_score": 0.5,
            "volume_ratio": 1.0,
            "pullback": True,
            "range_fade": True,
            "liquidity_sweep": True,
        }
        logger.debug(
            "evaluate_range_structure_reversion: %s pos=%.3f sl_dist=%.4f tp_dist=%.4f",
            direction,
            pos,
            risk,
            tp,
        )
        return out

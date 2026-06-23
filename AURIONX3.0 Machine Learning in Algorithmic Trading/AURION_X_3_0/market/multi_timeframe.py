"""Agregación OHLC multi-timeframe y estructura por temporalidad."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from analytics.logger import get_logger
from strategy.structure_detector import StructureDetector, StructureResult, Trend

if TYPE_CHECKING:
    from market.tick_handler import Tick

logger = get_logger(__name__)

SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")


@dataclass(slots=True)
class _Bar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def update(self, price: float, volume: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume


class MultiTimeframeEngine:
    """Construye velas OHLC por timeframe y deriva estructura por temporalidad."""

    def __init__(
        self,
        timeframes: list[str] | None = None,
        *,
        max_bars: int = 600,
        trends_cache_ttl_s: float = 0.02,
    ) -> None:
        self._timeframes = tuple(timeframes or SUPPORTED_TIMEFRAMES)
        for tf in self._timeframes:
            if tf not in SUPPORTED_TIMEFRAMES:
                raise ValueError(f"Timeframe no soportado: {tf}")
        self._max_bars = max_bars
        self._trends_cache_ttl_s = max(0.0, trends_cache_ttl_s)
        self._trends_cache: dict[str, tuple[float, dict[str, Trend]]] = {}
        self._open_bars: dict[str, dict[str, _Bar]] = defaultdict(dict)
        self._history: dict[str, dict[str, list[_Bar]]] = defaultdict(lambda: defaultdict(list))
        self._structure_kw: dict[str, Any] = {}
        self._detectors: dict[str, StructureDetector] = {
            tf: StructureDetector(left_bars=2, right_bars=2) for tf in self._timeframes
        }

    @staticmethod
    def _bucket_start(ts: datetime, tf: str) -> datetime:
        dt = ts.astimezone(timezone.utc).replace(microsecond=0)
        if tf.startswith("M") and tf != "MN1":
            minutes = int(tf[1:])
            minute = (dt.minute // minutes) * minutes
            return dt.replace(minute=minute, second=0)
        if tf.startswith("H"):
            hours = int(tf[1:])
            hour = (dt.hour // hours) * hours
            return dt.replace(hour=hour, minute=0, second=0)
        if tf == "D1":
            return dt.replace(hour=0, minute=0, second=0)
        if tf == "W1":
            start_day = (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0)
            return start_day
        if tf == "MN1":
            return dt.replace(day=1, hour=0, minute=0, second=0)
        raise ValueError(f"Timeframe no soportado: {tf}")

    def hydrate_from_mt5(
        self,
        symbol: str,
        timeframes: tuple[str, ...] | list[str] | None = None,
        *,
        bars_per_tf: int = 300,
    ) -> dict[str, int]:
        """Pre-carga historia OHLC desde MetaTrader5 al arranque del bot.

        Esto resuelve el warm-up cold-start del scalper M1 (necesita 25 velas)
        y de los detectores estructurales (necesitan >= 50 velas en HTF). Sin
        hidratación inicial, tras cada reinicio el bot tardaba ~25 min en
        estar listo para operar.

        Soft-fail: si MT5 no responde, simplemente registra el fallo y deja
        que el engine se hidrate por ticks vivos como hacía antes.

        Args:
            symbol: ticker (e.g. "XAUUSD").
            timeframes: lista de TFs a hidratar; por defecto usa todos los del engine.
            bars_per_tf: cuántas velas históricas pedir por TF (300 cubre warm-up
                de todos los detectores estructurales y SMC).

        Returns:
            dict {tf: nbars_loaded} por TF efectivamente cargado.
        """
        loaded: dict[str, int] = {}
        try:
            import MetaTrader5 as mt5  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("[HYDRATE] MetaTrader5 no disponible: %s", exc)
            return loaded
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        tfs = tuple(timeframes) if timeframes else self._timeframes
        for tf in tfs:
            tf_const = tf_map.get(tf)
            if tf_const is None:
                continue
            try:
                rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars_per_tf)
            except Exception as exc:
                logger.debug("[HYDRATE] %s %s: error mt5: %s", symbol, tf, exc)
                continue
            if rates is None or len(rates) == 0:
                continue
            bars: list[_Bar] = []
            for r in rates[:-1]:  # excluimos vela en formación
                try:
                    ts = datetime.fromtimestamp(float(r["time"]), tz=timezone.utc)
                    bars.append(
                        _Bar(
                            start=self._bucket_start(ts, tf),
                            open=float(r["open"]),
                            high=float(r["high"]),
                            low=float(r["low"]),
                            close=float(r["close"]),
                            volume=float(r["tick_volume"]),
                        )
                    )
                except Exception:
                    continue
            if not bars:
                continue
            if len(bars) > self._max_bars:
                bars = bars[-self._max_bars :]
            self._history[symbol][tf] = bars
            loaded[tf] = len(bars)
        if loaded:
            tot = sum(loaded.values())
            detail = ", ".join(f"{tf}={n}" for tf, n in loaded.items())
            logger.info("[HYDRATE] %s OHLC pre-cargado (%d velas: %s)", symbol, tot, detail)
        return loaded

    def ingest_tick(self, symbol: str, tick: Tick) -> None:
        """Incorpora un tick y actualiza/agrega velas en todos los timeframes."""
        price = float((tick.bid + tick.ask) / 2)
        vol = float(tick.volume) if tick.volume is not None else 0.0
        for tf in self._timeframes:
            start = self._bucket_start(tick.time_utc, tf)
            current = self._open_bars[symbol].get(tf)
            if current is None:
                self._open_bars[symbol][tf] = _Bar(start, price, price, price, price, vol)
                continue
            if current.start == start:
                current.update(price, vol)
                continue
            # Cierra vela previa y abre nueva.
            self._history[symbol][tf].append(current)
            if len(self._history[symbol][tf]) > self._max_bars:
                self._history[symbol][tf] = self._history[symbol][tf][-self._max_bars :]
            self._open_bars[symbol][tf] = _Bar(start, price, price, price, price, vol)

    def _bars_series(self, symbol: str, tf: str) -> list[_Bar]:
        bars = list(self._history[symbol][tf])
        current = self._open_bars[symbol].get(tf)
        if current is not None:
            bars.append(current)
        return bars

    def _series(self, symbol: str, tf: str) -> tuple[list[float], list[float], list[float]]:
        bars = self._bars_series(symbol, tf)
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        return highs, lows, closes

    def get_ohlc_series(self, symbol: str, tf: str) -> tuple[list[float], list[float], list[float]]:
        """Serie OHLC alineada (incluye vela en formación si existe)."""
        return self._series(symbol, tf)

    def get_ohlcv_series(self, symbol: str, tf: str) -> tuple[list[float], list[float], list[float], list[float]]:
        """Serie OHLCV alineada (incluye vela en formación si existe)."""
        bars = self._bars_series(symbol, tf)
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        return highs, lows, closes, volumes

    def get_ohlc_with_times(
        self,
        symbol: str,
        tf: str,
    ) -> tuple[list[datetime], list[float], list[float], list[float]]:
        """Inicio UTC de cada vela y OHLC (misma longitud)."""
        bars = self._bars_series(symbol, tf)
        times = [b.start for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        return times, highs, lows, closes

    def set_structure_detector_params(self, params: dict[str, Any] | None) -> None:
        """Recreate structure detectors (p. ej. momentum fallback) desde YAML."""
        allowed = (
            "momentum_fallback_bars",
            "momentum_min_return_pct",
            "conflict_resolve_bars",
            "conflict_min_net_return_pct",
        )
        if not params:
            self._structure_kw = {}
        else:
            self._structure_kw = {k: params[k] for k in allowed if k in params and params[k] is not None}
        self._detectors = {
            tf: StructureDetector(left_bars=2, right_bars=2, **self._structure_kw) for tf in self._timeframes
        }
        self._trends_cache.clear()

    def get_structure(self, symbol: str, tf: str) -> StructureResult:
        """Estructura para un símbolo/timeframe concreto."""
        highs, lows, closes = self._series(symbol, tf)
        if len(closes) < 8:
            return StructureResult(trend=Trend.RANGE, key_levels=())
        highs_d = [Decimal(str(v)) for v in highs]
        lows_d = [Decimal(str(v)) for v in lows]
        closes_d = [Decimal(str(v)) for v in closes]
        return self._detectors[tf].detect(highs_d, lows_d, closes_d)

    def _compute_trends(self, symbol: str) -> dict[str, Trend]:
        out: dict[str, Trend] = {}
        for tf in self._timeframes:
            out[tf] = self.get_structure(symbol, tf).trend
        return out

    def get_trends(self, symbol: str, *, min_interval_s: float | None = None) -> dict[str, Trend]:
        """Mapa trend por timeframe; reutiliza resultado si ``min_interval_s`` > 0 (menos CPU)."""
        interval = self._trends_cache_ttl_s if min_interval_s is None else max(0.0, min_interval_s)
        if interval <= 0:
            return self._compute_trends(symbol)
        now = time.monotonic()
        cached = self._trends_cache.get(symbol)
        if cached is not None and (now - cached[0]) < interval:
            return cached[1]
        trends = self._compute_trends(symbol)
        self._trends_cache[symbol] = (now, trends)
        return trends

    def set_trends_cache_ttl(self, seconds: float) -> None:
        """Ajusta TTL de caché de tendencias en runtime (p. ej. desde YAML)."""
        self._trends_cache_ttl_s = max(0.0, seconds)
        self._trends_cache.clear()

    def set_max_bars(self, n: int) -> None:
        """Limita historia OHLC por timeframe (recorta colas si ya había datos)."""
        self._max_bars = max(50, n)
        for sym_hist in self._history.values():
            for tf, bars in sym_hist.items():
                if len(bars) > self._max_bars:
                    sym_hist[tf] = bars[-self._max_bars :]

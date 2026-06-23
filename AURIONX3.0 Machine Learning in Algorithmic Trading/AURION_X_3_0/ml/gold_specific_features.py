"""AURION X 4.0 — Features exógenas oro-específicas (schema v3).

9 columnas que se concatenan al schema v2 (25 → 34):

    dxy_zscore                       Fuerza del USD (broker DXY o EURUSD inverso).
    us10y_real_yield                 Rendimiento real 10Y (FRED API si disponible).
    vix_zscore                       Aversión al riesgo (broker VIX o ATR S&P).
    gold_silver_ratio                XAUUSD / XAGUSD.
    gold_oil_ratio                   XAUUSD / WTIUSD.
    cot_speculator_net_long          CFTC COT report (viernes 21:00 UTC).
    central_bank_event_proximity     Minutos a/desde FOMC/ECB/BoE/BoJ.
    asia_session_intensity           Volumen vs MA20d en sesión asiática.
    london_fix_proximity             Minutos a/desde 10:30 y 15:00 GMT.

Diseño:
    - Caché por feature con TTL configurable.
    - Fallback determinístico si la fuente falla (último valor cacheado o 0.0).
    - Nunca lanza excepciones al pipeline runtime — devuelve NaN/0.0 y log warn.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from typing import Any, Callable, Mapping

import numpy as np

from analytics.logger import get_logger

logger = get_logger(__name__)

GOLD_FEATURE_ORDER: tuple[str, ...] = (
    "dxy_zscore",
    "us10y_real_yield",
    "vix_zscore",
    "gold_silver_ratio",
    "gold_oil_ratio",
    "cot_speculator_net_long",
    "central_bank_event_proximity",
    "asia_session_intensity",
    "london_fix_proximity",
)
GOLD_FEATURE_LEN: int = len(GOLD_FEATURE_ORDER)


@dataclass
class _CacheEntry:
    value: float = float("nan")
    ts_mono: float = 0.0
    ttl_s: float = 60.0


@dataclass
class GoldFeatureBundle:
    """Salida del provider."""

    values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(GOLD_FEATURE_ORDER, self.values, strict=False))

    def as_vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)


class GoldFeatureProvider:
    """Provider con caché TTL por feature.

    El provider NO se conecta a internet por defecto. Cualquier fuente externa
    se inyecta vía ``data_sources`` (dict callables). Si una source devuelve
    ``None`` o lanza excepción, se mantiene el último valor cacheado o se usa
    el proxy.
    """

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        data_sources: Mapping[str, Callable[[], float | None]] | None = None,
    ) -> None:
        c = dict(cfg or {})
        ttl = dict(c.get("cache_ttl_s") or {})
        self._cache: dict[str, _CacheEntry] = {
            name: _CacheEntry(ttl_s=float(ttl.get(name, 300.0)))
            for name in GOLD_FEATURE_ORDER
        }
        self._sources: dict[str, Callable[[], float | None]] = dict(data_sources or {})
        self._fred_api_key = os.getenv(str(c.get("fred_api_key_env", "FRED_API_KEY")), "")
        london_times = c.get("london_fix_times_utc") or ["10:30", "15:00"]
        self._london_times: list[dtime] = []
        for raw in london_times:
            try:
                h, m = str(raw).split(":")
                self._london_times.append(dtime(int(h) % 24, int(m) % 60))
            except (ValueError, IndexError):
                continue
        self._cb_window_min = int(c.get("central_bank_window_min", 60))
        self._cot_dow = int(c.get("cot_refresh_dow", 4))
        self._cot_hour = int(c.get("cot_refresh_hour_utc", 21))

    def _get(self, name: str, compute: Callable[[], float | None]) -> float:
        entry = self._cache[name]
        now = time.monotonic()
        if math.isfinite(entry.value) and (now - entry.ts_mono) < entry.ttl_s:
            return entry.value
        try:
            v = compute()
        except Exception as exc:  # noqa: BLE001 — el provider nunca debe romper el loop runtime
            logger.debug("gold_feature %s compute err: %s", name, exc)
            v = None
        if v is None or not math.isfinite(float(v)):
            return entry.value if math.isfinite(entry.value) else float("nan")
        entry.value = float(v)
        entry.ts_mono = now
        return entry.value

    # ──────────────────────────────────────────────────────────────────────
    # Features individuales
    # ──────────────────────────────────────────────────────────────────────

    def _dxy_zscore(self, market_state: Mapping[str, Any]) -> float | None:
        """DXY z-score. Si no hay symbol DXY usa EURUSD inverso como proxy."""
        f = self._sources.get("dxy_zscore")
        if f is not None:
            return f()
        # Proxy interno: derivar de market_state si el orchestrator pasa series
        eurusd_series = market_state.get("eurusd_closes") if isinstance(market_state, Mapping) else None
        if isinstance(eurusd_series, (list, tuple)) and len(eurusd_series) >= 60:
            arr = np.asarray(eurusd_series[-60:], dtype=np.float64)
            inv = 1.0 / arr  # proxy invertido del DXY
            mu = inv.mean()
            sd = inv.std(ddof=0) or 1.0
            return float((inv[-1] - mu) / sd)
        return None

    def _us10y_real_yield(self) -> float | None:
        f = self._sources.get("us10y_real_yield")
        if f is not None:
            return f()
        # Sin FRED disponible: no-op (NaN → imputación corriente)
        return None

    def _vix_zscore(self, market_state: Mapping[str, Any]) -> float | None:
        f = self._sources.get("vix_zscore")
        if f is not None:
            return f()
        # Proxy: ATR M5 normalizado de S&P si market_state lo expone
        sp = market_state.get("sp500_atr_m5") if isinstance(market_state, Mapping) else None
        if isinstance(sp, (list, tuple)) and len(sp) >= 60:
            arr = np.asarray(sp[-60:], dtype=np.float64)
            mu = arr.mean()
            sd = arr.std(ddof=0) or 1.0
            return float((arr[-1] - mu) / sd)
        return None

    def _gold_silver_ratio(self, market_state: Mapping[str, Any]) -> float | None:
        xau = market_state.get("last_price")
        xag = market_state.get("xagusd_last")
        if isinstance(xau, (int, float)) and isinstance(xag, (int, float)) and float(xag) > 0:
            return float(xau) / float(xag)
        f = self._sources.get("gold_silver_ratio")
        if f is not None:
            return f()
        return None

    def _gold_oil_ratio(self, market_state: Mapping[str, Any]) -> float | None:
        xau = market_state.get("last_price")
        oil = market_state.get("wtiusd_last")
        if isinstance(xau, (int, float)) and isinstance(oil, (int, float)) and float(oil) > 0:
            return float(xau) / float(oil)
        f = self._sources.get("gold_oil_ratio")
        if f is not None:
            return f()
        return None

    def _cot_speculator_net_long(self) -> float | None:
        f = self._sources.get("cot_speculator_net_long")
        if f is not None:
            return f()
        return None

    @staticmethod
    def _minutes_to_next(target_times: list[dtime], now: datetime) -> int:
        """Minutos al evento más cercano (positivos hacia el futuro)."""
        if not target_times:
            return 9999
        candidates: list[int] = []
        for t in target_times:
            dt_today = datetime.combine(now.date(), t, tzinfo=timezone.utc)
            diff = int((dt_today - now).total_seconds() / 60.0)
            if diff < -30:
                # Si pasó hace más de 30 min, considera el del día siguiente.
                diff += 24 * 60
            candidates.append(diff)
        return min(candidates, key=abs)

    def _london_fix_proximity(self, now: datetime) -> float:
        return float(self._minutes_to_next(self._london_times, now))

    def _central_bank_event_proximity(self, context: Mapping[str, Any]) -> float | None:
        f = self._sources.get("central_bank_event_proximity")
        if f is not None:
            return f()
        # Espera que el orchestrator inyecte ``cb_event_minutes_to`` (int).
        v = context.get("cb_event_minutes_to") if isinstance(context, Mapping) else None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _asia_session_intensity(self, market_state: Mapping[str, Any]) -> float | None:
        f = self._sources.get("asia_session_intensity")
        if f is not None:
            return f()
        ratio = market_state.get("asia_vol_vs_20dma") if isinstance(market_state, Mapping) else None
        if isinstance(ratio, (int, float)):
            return float(ratio)
        return None

    # ──────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────

    def compute(
        self,
        *,
        market_state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        now_utc: datetime | None = None,
    ) -> GoldFeatureBundle:
        ms = dict(market_state or {})
        ctx = dict(context or {})
        now = now_utc or datetime.now(timezone.utc)

        values = [
            self._get("dxy_zscore", lambda: self._dxy_zscore(ms)),
            self._get("us10y_real_yield", self._us10y_real_yield),
            self._get("vix_zscore", lambda: self._vix_zscore(ms)),
            self._get("gold_silver_ratio", lambda: self._gold_silver_ratio(ms)),
            self._get("gold_oil_ratio", lambda: self._gold_oil_ratio(ms)),
            self._get("cot_speculator_net_long", self._cot_speculator_net_long),
            self._get("central_bank_event_proximity", lambda: self._central_bank_event_proximity(ctx)),
            self._get("asia_session_intensity", lambda: self._asia_session_intensity(ms)),
            self._get("london_fix_proximity", lambda: self._london_fix_proximity(now)),
        ]
        # Imputación NaN → 0.0 para no propagar NaN al modelo.
        return GoldFeatureBundle(values=[float(v) if math.isfinite(v) else 0.0 for v in values])

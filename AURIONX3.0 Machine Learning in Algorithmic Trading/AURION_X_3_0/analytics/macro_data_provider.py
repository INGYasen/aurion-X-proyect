"""Provider de datos macro para features ML extendidas.

AURION X 4.3 (7-jun-2026): el modelo XAUUSD v2 (25 features) NO incluye
contexto macro. Para oro, las variables clave que predicen mejor son:

- **DXY** (US Dollar Index): correlacion inversa ~-0.7 con oro. Cuando DXY sube,
  oro tipicamente baja. Direccion + momentum de DXY es feature critica.
- **US10Y** (10-Year Treasury Yield): yield real (US10Y - inflacion esperada)
  es el costo de oportunidad de holdear oro. Subida de yields = oro presionado.
- **VIX** (CBOE Volatility Index): proxy de risk-off. VIX alto = vuelo a oro safe-haven.

Esta clase obtiene cotizaciones via MT5 (si broker ofrece simbolos) y, como
fallback, via yfinance (descargas batch infrecuentes con cache local).

Diseno:
    - Cache de 5min por simbolo (no spamear API).
    - Fallback graceful: si no hay datos, retorna 0.0 para que no rompa pipeline.
    - Symbol normalization: distintos brokers usan tickers diferentes (DXY vs USDX).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from analytics.logger import get_logger

logger = get_logger(__name__)


# Mapeo de simbolos esperados a posibles aliases en MT5.
_SYMBOL_ALIASES = {
    "DXY": ["DXY", "USDX", "DXY.spot", "DOLLARIDX"],
    "US10Y": ["US10Y", "USTNote10Y", "US10YT", "BUNDS"],
    "VIX": ["VIX", "VIX.spot", "VIXMINI"],
}


@dataclass
class _CachedQuote:
    timestamp_utc: datetime
    price: float
    prev_close: float
    history_1h: list[float] = field(default_factory=list)  # ultimos 60 closes 1m

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.timestamp_utc).total_seconds()


class MacroDataProvider:
    """Provee features macro (DXY, US10Y, VIX) con cache y fallback."""

    def __init__(
        self,
        mt5_connector: Any = None,
        cache_ttl_seconds: int = 300,
        fallback_yfinance: bool = True,
    ) -> None:
        """Args:
            mt5_connector: Conector MT5 con .last_tick(symbol). Si None, solo yfinance.
            cache_ttl_seconds: TTL del cache (default 5 min).
            fallback_yfinance: Si True, intenta yfinance cuando MT5 no tiene el simbolo.
        """
        self._mt5 = mt5_connector
        self._cache_ttl = max(60, int(cache_ttl_seconds))
        self._fallback_yf = bool(fallback_yfinance)
        self._cache: dict[str, _CachedQuote] = {}
        self._yf_available: bool | None = None
        self._resolved_aliases: dict[str, str] = {}

    def _try_yfinance(self) -> bool:
        if self._yf_available is not None:
            return self._yf_available
        try:
            import yfinance  # noqa: F401
            self._yf_available = True
        except ImportError:
            self._yf_available = False
            logger.info("MacroDataProvider: yfinance no instalado; solo MT5")
        return self._yf_available

    def _resolve_mt5_symbol(self, canonical: str) -> str | None:
        if canonical in self._resolved_aliases:
            return self._resolved_aliases[canonical] or None
        if self._mt5 is None:
            return None
        aliases = _SYMBOL_ALIASES.get(canonical, [canonical])
        for alias in aliases:
            try:
                tick = self._mt5.last_tick(alias)
                if tick and tick.get("bid", 0) > 0:
                    self._resolved_aliases[canonical] = alias
                    logger.info("MacroDataProvider: %s -> alias MT5 '%s'", canonical, alias)
                    return alias
            except Exception:
                continue
        self._resolved_aliases[canonical] = ""
        return None

    def _fetch_yfinance(self, canonical: str) -> _CachedQuote | None:
        if not self._try_yfinance():
            return None
        try:
            import yfinance as yf
            yf_ticker_map = {"DXY": "DX-Y.NYB", "US10Y": "^TNX", "VIX": "^VIX"}
            yf_sym = yf_ticker_map.get(canonical)
            if not yf_sym:
                return None
            tk = yf.Ticker(yf_sym)
            hist = tk.history(period="2d", interval="1h", auto_adjust=False)
            if hist is None or hist.empty:
                return None
            closes = [float(x) for x in hist["Close"].tolist() if not _is_nan(x)]
            if not closes:
                return None
            price = closes[-1]
            prev_close = closes[-2] if len(closes) > 1 else price
            return _CachedQuote(
                timestamp_utc=datetime.now(timezone.utc),
                price=price,
                prev_close=prev_close,
                history_1h=closes[-60:],
            )
        except Exception as exc:
            logger.debug("MacroDataProvider: yfinance fallo %s: %s", canonical, exc)
            return None

    def get_quote(self, canonical: str) -> _CachedQuote | None:
        """Cotizacion con cache. Devuelve None si no hay datos disponibles."""
        cached = self._cache.get(canonical)
        if cached is not None and cached.age_seconds < self._cache_ttl:
            return cached
        # Intentar MT5 primero.
        mt5_sym = self._resolve_mt5_symbol(canonical)
        if mt5_sym is not None:
            try:
                tick = self._mt5.last_tick(mt5_sym)
                if tick and tick.get("bid", 0) > 0:
                    price = float(tick["bid"])
                    quote = _CachedQuote(
                        timestamp_utc=datetime.now(timezone.utc),
                        price=price,
                        prev_close=cached.price if cached else price,
                    )
                    self._cache[canonical] = quote
                    return quote
            except Exception:
                pass
        # Fallback yfinance.
        if self._fallback_yf:
            quote = self._fetch_yfinance(canonical)
            if quote is not None:
                self._cache[canonical] = quote
                return quote
        return cached  # devuelve cache stale si existe

    def features(self) -> dict[str, float]:
        """Devuelve dict con features macro normalizadas listas para vector ML.

        Features:
            - dxy_return_24h: change DXY 24h en %.
            - dxy_zscore_5d: zscore de DXY ultimas 5 sesiones (approx via 1h candles).
            - us10y_change_24h: cambio absoluto US10Y en %.
            - vix_level: nivel VIX dividido entre 50 (normalizado).
            - vix_change_24h: cambio VIX 24h en %.
        """
        out: dict[str, float] = {}
        dxy = self.get_quote("DXY")
        us10y = self.get_quote("US10Y")
        vix = self.get_quote("VIX")
        if dxy and dxy.prev_close > 0:
            out["dxy_return_24h"] = (dxy.price - dxy.prev_close) / dxy.prev_close
        else:
            out["dxy_return_24h"] = 0.0
        if dxy and dxy.history_1h and len(dxy.history_1h) >= 10:
            hist = dxy.history_1h
            mean = sum(hist) / len(hist)
            var = sum((x - mean) ** 2 for x in hist) / len(hist)
            std = max(1e-9, var ** 0.5)
            out["dxy_zscore_5d"] = (dxy.price - mean) / std
        else:
            out["dxy_zscore_5d"] = 0.0
        if us10y and us10y.prev_close > 0:
            out["us10y_change_24h"] = (us10y.price - us10y.prev_close) / us10y.prev_close
        else:
            out["us10y_change_24h"] = 0.0
        if vix:
            out["vix_level"] = vix.price / 50.0  # normalizado
            if vix.prev_close > 0:
                out["vix_change_24h"] = (vix.price - vix.prev_close) / vix.prev_close
            else:
                out["vix_change_24h"] = 0.0
        else:
            out["vix_level"] = 0.4  # default neutral ~20/50
            out["vix_change_24h"] = 0.0
        return out


def _is_nan(value: float) -> bool:
    return value != value

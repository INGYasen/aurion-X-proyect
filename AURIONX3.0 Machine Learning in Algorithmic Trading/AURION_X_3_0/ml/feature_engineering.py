"""Construcción y transformación de features para modelos supervisados."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, TypedDict

from analytics.logger import get_logger

logger = get_logger(__name__)

_MIN_CLOSES_FOR_FEATURES = 2
_INSUFFICIENT_LOG_EVERY_S = 30.0
_last_insufficient_m1_log_mono: float = 0.0

# Vectores guardados como lista (sin nombres) antes de quitar ``spread`` tenían esta longitud.
FEATURE_LIST_LEGACY_LEN_WITH_SPREAD = 11

FEATURE_ORDER: tuple[str, ...] = (
    "return_1",
    "return_5",
    "atr_m1",
    "dist_ema20",
    "recent_range",
    "tick_volume_proxy",
    "hour_sin",
    "hour_cos",
    "direction",
    "drawdown",
)

# Longitud del vector que entra al modelo (debe coincidir con n_features_in_ tras entrenar).
FEATURE_VECTOR_LEN: int = len(FEATURE_ORDER)

# Incrementar si cambia el orden o el significado de las columnas (reentrenar obligatorio).
FEATURE_SCHEMA_VERSION: int = 1


class MarketState(TypedDict, total=False):
    """Estado de mercado y cuenta para construir el vector de features.

    Recomendado: al menos ~25 velas M1 alineadas en ``closes_m1``/``highs_m1``/``lows_m1``.
    """

    closes_m1: list[float]
    highs_m1: list[float]
    lows_m1: list[float]
    tick_volume_proxy: float
    direction: int
    drawdown: float
    timestamp_utc: datetime | str


def _parse_timestamp(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str):
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _ema_last(closes: list[float], span: int) -> float:
    """EMA al último cierre (semilla SMA en las primeras ``span`` velas)."""
    if not closes:
        return 0.0
    if len(closes) < span:
        return sum(closes) / len(closes)
    ema = sum(closes[:span]) / span
    k = 2.0 / (span + 1)
    for c in closes[span:]:
        ema = c * k + ema * (1.0 - k)
    return ema


def _true_range(h: float, l: float, c_prev: float) -> float:
    return max(h - l, abs(h - c_prev), abs(l - c_prev))


def _atr_m1_wilder(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """ATR (Wilder) sobre la serie M1; ``0.0`` si no hay datos suficientes."""
    n = len(closes)
    if n < 2 or len(highs) != n or len(lows) != n:
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


def _recent_range(highs: list[float], lows: list[float], close_last: float, window: int) -> float:
    if not highs or not lows or close_last == 0:
        return 0.0
    hh = max(highs[-window:])
    ll = min(lows[-window:])
    return (hh - ll) / close_last


def _hour_sin_cos(ts: datetime) -> tuple[float, float]:
    h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    ang = 2.0 * math.pi * (h / 24.0)
    return math.sin(ang), math.cos(ang)


def _direction_feature(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return 0.0
    if v > 0:
        return 1.0
    if v < 0:
        return -1.0
    return 0.0


def build_features(market_state: Mapping[str, Any]) -> tuple[list[float], dict[str, Any]]:
    """Construye vector numérico y dict serializable a partir del estado de mercado.

    Features (en orden):
        * ``return_1``, ``return_5``: retornos log simples del cierre M1.
        * ``atr_m1``: ATR(14) Wilder sobre high/low/close M1.
        * ``dist_ema20``: (close - EMA20) / EMA20.
        * ``recent_range``: (max high - min low) últimas 10 velas / close.
        * ``tick_volume_proxy``: tal cual en ``market_state``.
        * ``hour_sin``, ``hour_cos``: codificación cíclica de la hora UTC.
        * ``direction``: 1 / -1 / 0 según posición o sesgo.
        * ``drawdown``: drawdown actual (típicamente <= 0).

    Args:
        market_state: Debe incluir ``closes_m1`` (lista de floats). Opcionales:
            ``highs_m1``, ``lows_m1`` (si faltan, se usan los cierres como proxy),
            ``tick_volume_proxy``, ``direction``, ``drawdown``,
            ``timestamp_utc`` (``datetime`` o ISO).

    Returns:
        Tupla ``(vector, features_dict)`` con mismas claves en ``FEATURE_ORDER``
        y valores float serializables en el dict.
    """
    closes_all = market_state.get("closes_m1")
    if (
        not isinstance(closes_all, Sequence)
        or isinstance(closes_all, (str, bytes))
        or len(closes_all) < _MIN_CLOSES_FOR_FEATURES
    ):
        global _last_insufficient_m1_log_mono
        now_m = time.monotonic()
        if now_m - _last_insufficient_m1_log_mono >= _INSUFFICIENT_LOG_EVERY_S:
            logger.debug(
                "build_features: closes_m1 ausente o < %s muestras; vector cero hasta acumular M1",
                _MIN_CLOSES_FOR_FEATURES,
            )
            _last_insufficient_m1_log_mono = now_m
        zeros = [0.0] * len(FEATURE_ORDER)
        return zeros, {k: 0.0 for k in FEATURE_ORDER}

    closes = [float(x) for x in closes_all]
    n = len(closes)
    highs_raw = market_state.get("highs_m1")
    lows_raw = market_state.get("lows_m1")
    if (
        isinstance(highs_raw, Sequence)
        and not isinstance(highs_raw, (str, bytes))
        and len(highs_raw) == n
    ):
        highs = [float(x) for x in highs_raw]
    else:
        highs = closes.copy()
    if isinstance(lows_raw, Sequence) and not isinstance(lows_raw, (str, bytes)) and len(lows_raw) == n:
        lows = [float(x) for x in lows_raw]
    else:
        lows = closes.copy()

    c_last = closes[-1]
    c_prev = closes[-2]

    ret_1 = (c_last - c_prev) / c_prev if c_prev != 0 else 0.0
    if n >= 6 and closes[-6] != 0:
        ret_5 = (c_last - closes[-6]) / closes[-6]
    else:
        ret_5 = 0.0

    atr = _atr_m1_wilder(highs, lows, closes, period=14)
    ema20 = _ema_last(closes, span=20)
    dist_ema = (c_last - ema20) / ema20 if ema20 != 0 else 0.0

    window_r = min(10, n)
    recent_range = _recent_range(highs, lows, c_last, window_r)

    vol_proxy = float(market_state.get("tick_volume_proxy", 0.0) or 0.0)

    ts = _parse_timestamp(market_state.get("timestamp_utc"))
    hour_sin, hour_cos = _hour_sin_cos(ts)

    direction = _direction_feature(market_state.get("direction"))
    drawdown = float(market_state.get("drawdown", 0.0) or 0.0)

    vector = [
        float(ret_1),
        float(ret_5),
        float(atr),
        float(dist_ema),
        float(recent_range),
        vol_proxy,
        hour_sin,
        hour_cos,
        direction,
        drawdown,
    ]

    features_dict: dict[str, Any] = dict(zip(FEATURE_ORDER, vector, strict=True))
    return vector, features_dict


def build_feature_matrix(
    ohlcv: Sequence[Sequence[float]],
    *,
    config: dict[str, Any] | None = None,
) -> list[list[float]]:
    """Deriva una matriz de características a partir de OHLCV.

    Args:
        ohlcv: Filas ``[open, high, low, close, volume]`` por periodo.
        config: Parámetros de ventanas e indicadores.

    Returns:
        Lista de vectores de features por fila temporal (stub).
    """
    raise NotImplementedError


class FeaturePipeline:
    """Encadena pasos de normalización y selección de variables."""

    def __init__(self, steps: list[str] | None = None) -> None:
        """Declara nombres de etapas en orden de aplicación.

        Args:
            steps: Identificadores de transformadores (stub).
        """
        self._steps = steps or []
        logger.debug("FeaturePipeline steps=%s", self._steps)

    def fit(self, X: Sequence[Sequence[float]]) -> FeaturePipeline:
        """Ajusta estadísticos internos sobre datos de entrenamiento."""
        raise NotImplementedError

    def transform(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        """Aplica transformaciones aprendidas a ``X``."""
        raise NotImplementedError

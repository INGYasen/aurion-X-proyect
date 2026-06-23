"""AURION X 4.0 — Cargador de histórico M1 desde MT5 con cache parquet en D:.

El terminal MetaTrader 5 limita ``copy_rates_range`` a ``maxbars`` (≈100 000)
por llamada. Para descargar varios meses paginamos hacia atrás en chunks de
``chunk_days`` días, cacheando el resultado a parquet en
``D:/AURION_DATA/cache/m1_<symbol>_<months>m.parquet``.

API pública:

    >>> from analytics.m1_loader import load_m1_bars
    >>> bars = load_m1_bars("XAUUSD", months=6)            # devuelve np.ndarray
    >>> bars.shape
    (180000, 5)   # (T, [close, high, low, volume, spread])

Las columnas devueltas (mismas que ``train_patchtst_xauusd.py`` espera):

    0  close
    1  high
    2  low
    3  tick_volume
    4  spread (en points)

El cache se reusa si tiene <= ``cache_ttl_hours`` (24h por defecto).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = Path("D:/AURION_DATA/cache")
_CACHE_TTL_HOURS = 24
_CHUNK_DAYS = 30  # ~43 200 barras teóricas; bajo el límite maxbars=100 000


def _cache_path(symbol: str, months: int, cache_dir: Path | None = None) -> Path:
    d = cache_dir or _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"m1_{symbol.upper()}_{months}m.parquet"


def _is_cache_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.is_file():
        return False
    age_s = (datetime.now().timestamp() - path.stat().st_mtime)
    return age_s < ttl_hours * 3600


def _to_array(rates: Any) -> np.ndarray:
    """Convierte numpy structured array de MT5 a (T, 5) float64.

    rates dtype: time(i8), open(f8), high(f8), low(f8), close(f8),
                 tick_volume(i8), spread(i4), real_volume(i8).
    """
    arr = np.empty((len(rates), 5), dtype=np.float64)
    arr[:, 0] = rates["close"]
    arr[:, 1] = rates["high"]
    arr[:, 2] = rates["low"]
    arr[:, 3] = rates["tick_volume"].astype(np.float64)
    arr[:, 4] = rates["spread"].astype(np.float64)
    return arr


def _fetch_chunk(
    mt5: Any,
    symbol: str,
    tf_m1: int,
    utc_from: datetime,
    utc_to: datetime,
    retries: int = 3,
) -> np.ndarray | None:
    """Llama copy_rates_range con reintentos. Devuelve (T,5) o None."""
    for attempt in range(retries):
        rates = mt5.copy_rates_range(symbol, tf_m1, utc_from, utc_to)
        if rates is not None and len(rates) > 0:
            return _to_array(rates)
        if rates is None:
            err = mt5.last_error()
            logger.warning("copy_rates_range %s [%s..%s] None err=%s", symbol, utc_from, utc_to, err)
        else:
            logger.debug("copy_rates_range %s [%s..%s] vacío", symbol, utc_from, utc_to)
            return np.empty((0, 5), dtype=np.float64)
    return None


def load_m1_bars(
    symbol: str,
    months: int = 6,
    *,
    cache_dir: Path | None = None,
    cache_ttl_hours: int = _CACHE_TTL_HOURS,
    chunk_days: int = _CHUNK_DAYS,
    force_refresh: bool = False,
) -> np.ndarray:
    """Devuelve un array ``(T, 5)`` con M1 [close, high, low, tick_vol, spread].

    Args:
        symbol: Símbolo del broker (case-insensitive).
        months: Meses históricos a cargar (1..24).
        cache_dir: Carpeta de cache parquet (default ``D:/AURION_DATA/cache``).
        cache_ttl_hours: Si el parquet en cache tiene menos de N horas, se reusa.
        chunk_days: Días por llamada a MT5 (bajo ``maxbars``).
        force_refresh: Si True, ignora el cache.

    Raises:
        SystemExit: si MT5 no inicializa o no hay datos para el símbolo.
    """
    symbol = symbol.upper()
    months = max(1, int(months))
    cache_file = _cache_path(symbol, months, cache_dir)

    if not force_refresh and _is_cache_fresh(cache_file, cache_ttl_hours):
        logger.info("Reutilizando cache %s", cache_file)
        try:
            import pyarrow.parquet as pq  # noqa: WPS433
            table = pq.read_table(cache_file)
            return table.to_pandas().to_numpy(dtype=np.float64)
        except (ImportError, OSError) as exc:
            logger.warning("Cache corrupto/inaccesible (%s); reintentando descarga.", exc)

    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("MetaTrader5 no instalado en el entorno.") from exc

    if not mt5.initialize():
        err = mt5.last_error()
        raise SystemExit(f"mt5.initialize() falló: {err}")

    try:
        tf_m1 = mt5.TIMEFRAME_M1
        utc_end = datetime.now(timezone.utc)
        utc_start = utc_end - timedelta(days=months * 31)

        chunks: list[np.ndarray] = []
        cursor = utc_end
        while cursor > utc_start:
            chunk_from = max(utc_start, cursor - timedelta(days=chunk_days))
            arr = _fetch_chunk(mt5, symbol, tf_m1, chunk_from, cursor)
            if arr is None:
                logger.warning("Chunk %s..%s sin respuesta tras reintentos.", chunk_from, cursor)
            elif arr.size > 0:
                chunks.append(arr)
                logger.info("Chunk %s..%s -> %d barras", chunk_from.date(), cursor.date(), arr.shape[0])
            cursor = chunk_from

        if not chunks:
            raise SystemExit(
                f"Sin barras M1 para {symbol}. Verifica que el broker tenga histórico "
                "y que el símbolo esté en Market Watch."
            )

        full = np.concatenate(chunks[::-1], axis=0)
        # Dedup por close-time aproximado (chunks pueden solaparse en bordes).
        # Como no llevamos timestamps en el array final, dedup simple por close+vol.
        _, uniq_idx = np.unique(full[:, [0, 1, 2, 3]], axis=0, return_index=True)
        full = full[np.sort(uniq_idx)]

        try:
            import pandas as pd  # noqa: WPS433
            import pyarrow as pa  # noqa: WPS433
            import pyarrow.parquet as pq  # noqa: WPS433
            df = pd.DataFrame(full, columns=["close", "high", "low", "tick_volume", "spread"])
            pq.write_table(pa.Table.from_pandas(df), cache_file)
            logger.info("Cache escrito %s (%d barras)", cache_file, full.shape[0])
        except (ImportError, OSError) as exc:
            logger.warning("No se pudo cachear a parquet: %s", exc)

        return full
    finally:
        mt5.shutdown()

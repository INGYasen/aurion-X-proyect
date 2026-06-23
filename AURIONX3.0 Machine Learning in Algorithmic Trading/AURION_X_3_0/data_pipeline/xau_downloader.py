"""AURION X 4.1 — Descargador histórico XAUUSD (3 años · 9 timeframes).

Diseño:

* **MT5 primero** (fuente real del broker, idéntica a la operativa). Si MT5 no
  está disponible cae a un fallback: re-agregar desde el cache M1 existente
  (``analytics.m1_loader``) hacia TFs superiores. Esto garantiza al menos M1
  funcional incluso en entornos sin MT5 (CI, Docker sin MT5).
* **Pagina por chunks** respetando el límite ``maxbars≈100 000`` por llamada
  de MT5: chunks de N días por TF (N escalado al tamaño de la vela).
* **Persistencia parquet** en ``D:/AURION_DATA/historical/XAUUSD/<TF>.parquet``
  con dtype canónico: ``timestamp, open, high, low, close, tick_volume, spread,
  real_volume``. Si pyarrow no está disponible cae a CSV (warning visible).
* **Resumen post-descarga**: devuelve dict con nº de barras, rango temporal y
  tamaño de fichero por TF. El CLI ``scripts/download_xauusd_3y.py`` lo
  imprime al final.

Idempotente: si el parquet existe y está dentro del TTL, se reusa salvo
``force_refresh=True``. Permite descarga incremental sólo de los TFs que
faltan o están desactualizados.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Spec por TF
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class TimeframeSpec:
    """Especificación de un timeframe para descarga incremental."""

    name: str          # "M1", "M15", "H1"…
    minutes: int       # duración de la vela en minutos (W1=10080, MN1=43200)
    chunk_days: int    # nº de días por llamada a MT5 (bajo maxbars)


SUPPORTED_TIMEFRAMES: tuple[TimeframeSpec, ...] = (
    TimeframeSpec("M1", 1, 30),       # ~43k barras/chunk
    TimeframeSpec("M5", 5, 120),      # ~34k barras/chunk
    TimeframeSpec("M15", 15, 365),    # ~35k barras/chunk
    TimeframeSpec("M30", 30, 730),    # ~35k barras/chunk
    TimeframeSpec("H1", 60, 1095),    # ~26k barras/chunk
    TimeframeSpec("H4", 240, 1095),   # ~6.5k barras/chunk
    TimeframeSpec("D1", 1440, 1095),  # ~1k barras/chunk
    TimeframeSpec("W1", 10080, 1095), # ~160 barras/chunk
    TimeframeSpec("MN1", 43200, 1095),# ~36 barras/chunk
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


_DEFAULT_OUT_DIR = Path("D:/AURION_DATA/historical/XAUUSD")


@dataclass(slots=True)
class XAUDownloaderConfig:
    """Configuración del downloader."""

    symbol: str = "XAUUSD"
    years: float = 3.0
    out_dir: Path = field(default_factory=lambda: _DEFAULT_OUT_DIR)
    timeframes: tuple[TimeframeSpec, ...] = SUPPORTED_TIMEFRAMES
    force_refresh: bool = False
    cache_ttl_hours: int = 24
    retries_per_chunk: int = 3
    # Fallback: si MT5 no está, agregamos desde el cache M1 hacia TFs mayores.
    enable_aggregation_fallback: bool = True


# --------------------------------------------------------------------------- #
# Downloader
# --------------------------------------------------------------------------- #


_MT5_TF_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388,
    "D1": 16408, "W1": 32769, "MN1": 49153,
}  # estos números son las constantes mt5.TIMEFRAME_X (verificadas en mt5 5.0.45+)


class XAUDownloader:
    """Descargador histórico XAUUSD con cache parquet por TF."""

    def __init__(self, config: XAUDownloaderConfig | None = None) -> None:
        self._cfg = config or XAUDownloaderConfig()
        self._cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #

    def out_path(self, tf: str) -> Path:
        return self._cfg.out_dir / f"{tf.upper()}.parquet"

    def download_all(self) -> dict[str, dict[str, Any]]:
        """Descarga todos los TFs configurados.

        Returns:
            Diccionario por TF::

                {
                  "M15": {"bars": 70123, "from": "...", "to": "...", "path": "..."},
                  ...
                }
        """
        summary: dict[str, dict[str, Any]] = {}
        mt5_mod = self._init_mt5()
        try:
            for spec in self._cfg.timeframes:
                try:
                    info = self.download_timeframe(spec, mt5=mt5_mod)
                    summary[spec.name] = info
                except Exception as exc:  # nunca dejar caer el resto
                    logger.warning("[XAU-DL] %s falló: %s", spec.name, exc)
                    summary[spec.name] = {"error": str(exc), "bars": 0}
        finally:
            if mt5_mod is not None:
                try:
                    mt5_mod.shutdown()
                except Exception:
                    pass
        return summary

    def download_timeframe(self, spec: TimeframeSpec, *, mt5: Any | None = None) -> dict[str, Any]:
        """Descarga un único TF y persiste a parquet."""
        out = self.out_path(spec.name)
        if not self._cfg.force_refresh and self._is_fresh(out):
            logger.info("[XAU-DL] %s en cache (%.1fh), reuso %s", spec.name, self._age_hours(out), out)
            df = self._read_parquet_or_csv(out)
            if df is not None and len(df) > 0:
                return {
                    "bars": int(len(df)),
                    "from": str(df["timestamp"].iloc[0]) if "timestamp" in df.columns else None,
                    "to": str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else None,
                    "path": str(out),
                    "cached": True,
                }

        if mt5 is not None:
            arr, times = self._download_via_mt5(spec, mt5)
        else:
            arr, times = self._download_via_aggregation(spec)

        if arr is None or arr.size == 0:
            raise RuntimeError(f"sin datos para {spec.name}")

        df = self._array_to_dataframe(arr, times)
        self._write_parquet_or_csv(out, df)
        return {
            "bars": int(len(df)),
            "from": str(df["timestamp"].iloc[0]) if len(df) else None,
            "to": str(df["timestamp"].iloc[-1]) if len(df) else None,
            "path": str(out),
            "cached": False,
        }

    # ------------------------------------------------------------------ #
    # MT5 path
    # ------------------------------------------------------------------ #

    def _init_mt5(self) -> Any | None:
        try:
            import MetaTrader5 as mt5  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("[XAU-DL] MetaTrader5 no instalado; usaremos fallback (sólo M1 desde cache).")
            return None
        try:
            if not mt5.initialize():
                err = mt5.last_error()
                logger.warning("[XAU-DL] mt5.initialize() falló: %s", err)
                return None
        except Exception as exc:
            logger.warning("[XAU-DL] mt5.initialize() excepción: %s", exc)
            return None
        return mt5

    def _download_via_mt5(self, spec: TimeframeSpec, mt5: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        tf_const = _MT5_TF_MAP.get(spec.name)
        if tf_const is None:
            return (None, None)
        utc_end = datetime.now(timezone.utc)
        utc_start = utc_end - timedelta(days=int(self._cfg.years * 365.25))
        chunks: list[np.ndarray] = []
        cursor = utc_end
        while cursor > utc_start:
            chunk_from = max(utc_start, cursor - timedelta(days=spec.chunk_days))
            rates = None
            for _attempt in range(self._cfg.retries_per_chunk):
                try:
                    rates = mt5.copy_rates_range(self._cfg.symbol, tf_const, chunk_from, cursor)
                except Exception as exc:
                    logger.warning("[XAU-DL] %s exc en chunk %s..%s: %s", spec.name, chunk_from, cursor, exc)
                    rates = None
                if rates is not None and len(rates) > 0:
                    break
            if rates is not None and len(rates) > 0:
                chunks.append(self._rates_to_array(rates))
                logger.info("[XAU-DL] %s chunk %s..%s -> %d barras",
                            spec.name, chunk_from.date(), cursor.date(), len(rates))
            else:
                logger.info("[XAU-DL] %s chunk %s..%s sin datos", spec.name, chunk_from.date(), cursor.date())
            cursor = chunk_from
        if not chunks:
            return (None, None)
        full = np.concatenate(chunks[::-1], axis=0)
        # dedup por timestamp
        _, uniq = np.unique(full[:, 0], return_index=True)
        full = full[np.sort(uniq)]
        times = full[:, 0]
        return (full[:, 1:], times)

    @staticmethod
    def _rates_to_array(rates: Any) -> np.ndarray:
        """Convierte numpy structured array MT5 a (T, 8) float64:
        [time, open, high, low, close, tick_volume, spread, real_volume].
        """
        arr = np.empty((len(rates), 8), dtype=np.float64)
        arr[:, 0] = rates["time"].astype(np.float64)  # epoch s
        arr[:, 1] = rates["open"].astype(np.float64)
        arr[:, 2] = rates["high"].astype(np.float64)
        arr[:, 3] = rates["low"].astype(np.float64)
        arr[:, 4] = rates["close"].astype(np.float64)
        arr[:, 5] = rates["tick_volume"].astype(np.float64)
        # Algunos brokers no devuelven spread o real_volume; valores opcionales.
        try:
            arr[:, 6] = rates["spread"].astype(np.float64)
        except Exception:
            arr[:, 6] = 0.0
        try:
            arr[:, 7] = rates["real_volume"].astype(np.float64)
        except Exception:
            arr[:, 7] = 0.0
        return arr

    # ------------------------------------------------------------------ #
    # Fallback: agregación desde M1 cache
    # ------------------------------------------------------------------ #

    def _download_via_aggregation(self, spec: TimeframeSpec) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self._cfg.enable_aggregation_fallback:
            return (None, None)
        if spec.name == "M1":
            # No hay nada que agregar; sólo cargar el cache M1 si existe.
            return self._fallback_load_m1_cache_only()
        # Para TFs > M1, usamos el m1_loader como semilla y reagregamos.
        try:
            from analytics.m1_loader import load_m1_bars
        except Exception as exc:
            logger.warning("[XAU-DL] fallback agregación no disponible: %s", exc)
            return (None, None)
        months = max(1, int(self._cfg.years * 12))
        try:
            bars_m1 = load_m1_bars(self._cfg.symbol, months=months)
        except SystemExit as exc:
            logger.warning("[XAU-DL] fallback agregación: m1_loader no pudo cargar (%s)", exc)
            return (None, None)
        # bars_m1 viene como [close, high, low, tick_vol, spread]; sin timestamp
        # reconstruimos timestamps aproximados (último = ahora, paso = 1 min).
        n = bars_m1.shape[0]
        now_ts = datetime.now(timezone.utc).timestamp()
        times = np.array([now_ts - (n - 1 - i) * 60.0 for i in range(n)], dtype=np.float64)
        # Generar OHLC sintético: open = close anterior, high/low ya están.
        opens = np.empty(n, dtype=np.float64)
        opens[0] = bars_m1[0, 0]
        opens[1:] = bars_m1[:-1, 0]
        # Apilamos: [t, open, high, low, close, vol, spread, real_volume=0]
        full = np.column_stack([
            times,
            opens,
            bars_m1[:, 1],
            bars_m1[:, 2],
            bars_m1[:, 0],
            bars_m1[:, 3],
            bars_m1[:, 4],
            np.zeros(n, dtype=np.float64),
        ])
        # Re-agregamos a spec.minutes
        agg = self._aggregate(full, spec.minutes)
        if agg.size == 0:
            return (None, None)
        return (agg[:, 1:], agg[:, 0])

    def _fallback_load_m1_cache_only(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            from analytics.m1_loader import load_m1_bars
        except Exception as exc:
            logger.warning("[XAU-DL] fallback M1: m1_loader no disponible (%s)", exc)
            return (None, None)
        months = max(1, int(self._cfg.years * 12))
        try:
            bars = load_m1_bars(self._cfg.symbol, months=months)
        except SystemExit as exc:
            logger.warning("[XAU-DL] fallback M1: %s", exc)
            return (None, None)
        n = bars.shape[0]
        now_ts = datetime.now(timezone.utc).timestamp()
        times = np.array([now_ts - (n - 1 - i) * 60.0 for i in range(n)], dtype=np.float64)
        opens = np.empty(n, dtype=np.float64)
        opens[0] = bars[0, 0]
        opens[1:] = bars[:-1, 0]
        out = np.column_stack([
            opens,
            bars[:, 1],
            bars[:, 2],
            bars[:, 0],
            bars[:, 3],
            bars[:, 4],
            np.zeros(n, dtype=np.float64),
        ])
        return (out, times)

    @staticmethod
    def _aggregate(full: np.ndarray, minutes: int) -> np.ndarray:
        """Re-agrega un OHLC[V,spread] (con timestamps epoch s en col 0) a
        velas de ``minutes``. Devuelve array (T_out, 8)."""
        if full.size == 0 or minutes <= 1:
            return full
        bucket = minutes * 60
        buckets = (full[:, 0].astype(np.int64) // bucket) * bucket
        out: list[list[float]] = []
        i = 0
        n = full.shape[0]
        while i < n:
            j = i
            while j < n and buckets[j] == buckets[i]:
                j += 1
            seg = full[i:j]
            out.append([
                float(buckets[i]),
                float(seg[0, 1]),              # open
                float(seg[:, 2].max()),        # high
                float(seg[:, 3].min()),        # low
                float(seg[-1, 4]),             # close
                float(seg[:, 5].sum()),        # tick_volume
                float(seg[:, 6].mean()) if seg.shape[0] > 0 else 0.0,  # spread mean
                float(seg[:, 7].sum()),        # real_volume
            ])
            i = j
        return np.asarray(out, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Persistencia parquet/csv + lectura
    # ------------------------------------------------------------------ #

    def _array_to_dataframe(self, arr: np.ndarray, times: np.ndarray | None) -> Any:
        try:
            import pandas as pd  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError("pandas no instalado") from exc
        cols = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        # arr shape (T, 7)
        df = pd.DataFrame(arr, columns=cols)
        if times is not None and len(times) == len(df):
            df.insert(0, "timestamp", [
                datetime.fromtimestamp(float(t), tz=timezone.utc) for t in times
            ])
        return df

    def _write_parquet_or_csv(self, path: Path, df: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, engine="pyarrow", index=False)
            logger.info("[XAU-DL] escrito %s (%d barras)", path, len(df))
            return
        except Exception as exc:
            logger.warning("[XAU-DL] parquet falló (%s); fallback CSV", exc)
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.info("[XAU-DL] escrito %s (%d barras)", csv_path, len(df))

    def _read_parquet_or_csv(self, path: Path) -> Any | None:
        try:
            import pandas as pd  # noqa: WPS433
        except ImportError:
            return None
        if path.is_file():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file():
            try:
                return pd.read_csv(csv_path)
            except Exception:
                pass
        return None

    def _is_fresh(self, path: Path) -> bool:
        if not path.is_file():
            return False
        age = (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
        return age < self._cfg.cache_ttl_hours

    def _age_hours(self, path: Path) -> float:
        return (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0


# --------------------------------------------------------------------------- #
# Helper API
# --------------------------------------------------------------------------- #


def download_xauusd_history(
    *,
    years: float = 3.0,
    out_dir: Path | str = _DEFAULT_OUT_DIR,
    timeframes: Iterable[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Helper de alto nivel: descarga XAUUSD ``years`` años en ``timeframes``.

    Args:
        years: ventana histórica.
        out_dir: carpeta destino (parquet por TF).
        timeframes: lista de nombres de TF (default = todos los soportados).
        force_refresh: ignora cache.

    Returns:
        Dict resumen por TF (ver ``XAUDownloader.download_all``).
    """
    if timeframes is None:
        tf_specs: tuple[TimeframeSpec, ...] = SUPPORTED_TIMEFRAMES
    else:
        names = {n.upper() for n in timeframes}
        tf_specs = tuple(s for s in SUPPORTED_TIMEFRAMES if s.name in names)
    cfg = XAUDownloaderConfig(
        years=years,
        out_dir=Path(out_dir),
        timeframes=tf_specs,
        force_refresh=force_refresh,
    )
    return XAUDownloader(cfg).download_all()


__all__ = [
    "XAUDownloader",
    "XAUDownloaderConfig",
    "TimeframeSpec",
    "SUPPORTED_TIMEFRAMES",
    "download_xauusd_history",
]

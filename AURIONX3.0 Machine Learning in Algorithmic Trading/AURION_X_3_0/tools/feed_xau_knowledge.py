"""AURION X 4.2 — Alimentador retroactivo del XAU Knowledge Engine.

Recorre los 3 años de M15/H1/H4/D1 históricos y popula las tablas:

* xau_market_memory   — una fila por vela cerrada con regime/trend/volatility/session
* xau_regime_memory   — episodios de régimen (TRENDING_BULL/TRENDING_BEAR/CHOP) con
                        duración promedio y estadísticas
* xau_session_memory  — métricas por sesión (Asia/London/NY/Overlap) por día
* xau_smc_events      — eventos SMC denormalizados con contexto (vinculados a smc_*)
* xau_patterns        — combinaciones de patrones (futuro: BOS+OB, CHOCH+FVG...)

Sin look-ahead: cada vela se procesa con info hasta ese momento.

CLI:
    python tools/feed_xau_knowledge.py [--tfs M15,H1,H4,D1] [--years 3] [--purge]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from analytics.db_connection import init_db, new_session

HIST_DIR = Path("D:/AURION_DATA/historical/XAUUSD")


# -------------------- Régimen, Trend, Volatilidad -------------------- #


def classify_regime(close: np.ndarray, ema_fast: np.ndarray, ema_slow: np.ndarray,
                    atr: np.ndarray, atr_ma: np.ndarray, n: int) -> str:
    """Clasifica régimen en la barra `n` (sin look-ahead)."""
    if n < 60:
        return "WARMUP"
    sep = (ema_fast[n] - ema_slow[n]) / ema_slow[n] if ema_slow[n] != 0 else 0
    vol_ratio = atr[n] / atr_ma[n] if atr_ma[n] != 0 else 1.0
    if abs(sep) < 0.0015:        # menos del 0.15% de separación entre EMAs
        return "CHOP_HIGH_VOL" if vol_ratio > 1.4 else "CHOP_LOW_VOL"
    if sep > 0:
        return "TRENDING_BULL_STRONG" if vol_ratio > 1.2 else "TRENDING_BULL"
    return "TRENDING_BEAR_STRONG" if vol_ratio > 1.2 else "TRENDING_BEAR"


def classify_trend(close: np.ndarray, ema_fast: np.ndarray, n: int) -> str:
    if n < 20:
        return "NEUTRAL"
    if close[n] > ema_fast[n] * 1.002:
        return "UP"
    if close[n] < ema_fast[n] * 0.998:
        return "DOWN"
    return "NEUTRAL"


def session_of_hour(h: int) -> str:
    """Sesiones UTC: ASIA 00-07, LONDON 07-12, OVERLAP 12-16, NY 16-21, OFF 21-24."""
    if 0 <= h < 7:
        return "ASIA"
    if 7 <= h < 12:
        return "LONDON"
    if 12 <= h < 16:
        return "OVERLAP_LDN_NY"
    if 16 <= h < 21:
        return "NY"
    return "OFF"


# -------------------- Carga histórico -------------------- #


def load_tf(tf: str) -> pd.DataFrame:
    p = HIST_DIR / f"{tf}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    tcol = next((c for c in ("timestamp", "time", "datetime") if c in df.columns), None)
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    return df.rename(columns={tcol: "time"}).sort_values("time").reset_index(drop=True)


def ema(arr: np.ndarray, span: int) -> np.ndarray:
    k = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def atr_series(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    out = np.zeros_like(tr)
    out[0] = tr[0]
    a = 1.0 / period
    for i in range(1, len(tr)):
        out[i] = (1 - a) * out[i - 1] + a * tr[i]
    return out


def sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(arr)
    cum = np.cumsum(np.insert(arr, 0, 0))
    for i in range(len(arr)):
        n = min(period, i + 1)
        out[i] = (cum[i + 1] - cum[i + 1 - n]) / n
    return out


# -------------------- Persistence batch -------------------- #


def batch_insert_market_memory(s, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    s.execute(text("""
        INSERT INTO xau_market_memory
            (id, timestamp, symbol, timeframe, open, high, low, close,
             tick_volume, spread_points, regime, trend, range_pct,
             volatility_atr, volatility_zscore, session, hour_utc, extra)
        VALUES
            (:id, :timestamp, :symbol, :timeframe, :open, :high, :low, :close,
             :tick_volume, :spread_points, :regime, :trend, :range_pct,
             :volatility_atr, :volatility_zscore, :session, :hour_utc, :extra)
    """), rows)
    return len(rows)


def batch_insert_session_memory(s, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    s.execute(text("""
        INSERT INTO xau_session_memory
            (id, symbol, session, day, bars_n, avg_atr, avg_range_pct,
             avg_spread_pts, trades_n, trades_wins, sum_r_milli, extra)
        VALUES
            (:id, :symbol, :session, :day, :bars_n, :avg_atr, :avg_range_pct,
             :avg_spread_pts, :trades_n, :trades_wins, :sum_r_milli, :extra)
    """), rows)
    return len(rows)


def batch_insert_regime_memory(s, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    s.execute(text("""
        INSERT INTO xau_regime_memory
            (id, symbol, regime, started_at, ended_at, duration_minutes,
             avg_atr, avg_range_pct, trades_n, trades_wins, sum_r_milli, extra)
        VALUES
            (:id, :symbol, :regime, :started_at, :ended_at, :duration_minutes,
             :avg_atr, :avg_range_pct, :trades_n, :trades_wins, :sum_r_milli, :extra)
    """), rows)
    return len(rows)


# -------------------- Procesamiento por TF -------------------- #


TF_DURATION_MINUTES = {"M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def process_tf(s, tf: str, since: datetime, max_rows: int = 10_000) -> dict[str, int]:
    """Procesa un TF y persiste xau_market_memory + xau_regime_memory + 
    xau_session_memory para él."""
    print(f"\n  -- TF {tf} --", flush=True)
    df = load_tf(tf)
    df = df[df["time"] >= since].reset_index(drop=True)
    if len(df) == 0:
        return {"tf": tf, "memory": 0, "regimes": 0, "sessions": 0}

    # Subsample para no inflar Postgres: solo cada N velas
    # M15: cada 4 (1h) → 8760 rows/año
    # M30: cada 2 (1h) → 8760 rows/año
    # H1, H4, D1: todas
    if tf == "M15":
        step = 4
    elif tf == "M30":
        step = 2
    else:
        step = 1
    df = df.iloc[::step].reset_index(drop=True)
    print(f"  velas (post-subsample step={step}): {len(df):,}", flush=True)

    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    vol_arr = df["tick_volume"].to_numpy() if "tick_volume" in df.columns else np.ones(len(df))
    # Lista plana de datetimes UTC-aware para acceso por índice
    times = list(df["time"].dt.to_pydatetime())

    # Indicadores
    ef = ema(c, 20)
    es = ema(c, 50)
    a = atr_series(h, l, c, 14)
    a_ma = sma(a, 50)
    rng_pct = (h - l) / np.maximum(c, 1e-9)
    # Volatility z-score (rolling 100)
    z = np.zeros_like(a)
    for i in range(len(a)):
        lo = max(0, i - 100 + 1)
        win = a[lo:i + 1]
        if len(win) > 5:
            mu, sd = win.mean(), win.std()
            z[i] = (a[i] - mu) / (sd if sd > 1e-9 else 1.0)

    n = len(df)
    memory_rows: list[dict[str, Any]] = []
    session_aggr: dict[tuple[str, str], dict[str, Any]] = {}  # (day, session) → stats
    regime_episodes: list[dict[str, Any]] = []
    cur_regime = None
    cur_start: datetime | None = None
    cur_atr_sum = 0.0
    cur_rng_sum = 0.0
    cur_count = 0

    for i in range(n):
        ti = times[i]
        hour = ti.hour
        sess = session_of_hour(hour)
        regime = classify_regime(c, ef, es, a, a_ma, i)
        trend = classify_trend(c, ef, i)

        memory_rows.append({
            "id": uuid.uuid4(),
            "timestamp": ti,
            "symbol": "XAUUSD",
            "timeframe": tf,
            "open": float(o[i]),
            "high": float(h[i]),
            "low": float(l[i]),
            "close": float(c[i]),
            "tick_volume": float(vol_arr[i]),
            "spread_points": None,
            "regime": regime[:24],
            "trend": trend[:16],
            "range_pct": float(rng_pct[i]),
            "volatility_atr": float(a[i]),
            "volatility_zscore": float(z[i]),
            "session": sess[:16],
            "hour_utc": int(hour),
            "extra": None,
        })

        # Sesiones (agregado diario)
        day_key = ti.date().isoformat()
        sk = (day_key, sess)
        agg = session_aggr.get(sk)
        if agg is None:
            session_aggr[sk] = agg = {
                "id": uuid.uuid4(),
                "symbol": "XAUUSD",
                "session": sess[:16],
                "day": datetime.combine(ti.date(), datetime.min.time()).replace(tzinfo=timezone.utc),
                "bars_n": 0, "atr_sum": 0.0, "rng_sum": 0.0,
                "trades_n": 0, "trades_wins": 0, "sum_r_milli": 0,
                "extra": None,
            }
        agg["bars_n"] += 1
        agg["atr_sum"] += float(a[i])
        agg["rng_sum"] += float(rng_pct[i])

        # Régimen: episodios
        if cur_regime is None:
            cur_regime = regime
            cur_start = ti
            cur_atr_sum = float(a[i])
            cur_rng_sum = float(rng_pct[i])
            cur_count = 1
        elif regime != cur_regime:
            tf_min = TF_DURATION_MINUTES.get(tf, 60)
            duration_m = int(((ti - cur_start).total_seconds() / 60.0)) if cur_start else cur_count * tf_min
            regime_episodes.append({
                "id": uuid.uuid4(),
                "symbol": "XAUUSD",
                "regime": cur_regime[:24],
                "started_at": cur_start,
                "ended_at": ti,
                "duration_minutes": duration_m,
                "avg_atr": cur_atr_sum / max(cur_count, 1),
                "avg_range_pct": cur_rng_sum / max(cur_count, 1),
                "trades_n": 0, "trades_wins": 0, "sum_r_milli": 0,
                "extra": json.dumps({"timeframe": tf, "bars": cur_count}),
            })
            cur_regime = regime
            cur_start = ti
            cur_atr_sum = float(a[i])
            cur_rng_sum = float(rng_pct[i])
            cur_count = 1
        else:
            cur_atr_sum += float(a[i])
            cur_rng_sum += float(rng_pct[i])
            cur_count += 1

        # Commit batches
        if len(memory_rows) >= 2000:
            batch_insert_market_memory(s, memory_rows)
            s.commit()
            memory_rows = []

    if cur_regime is not None and cur_start is not None and cur_count > 0:
        regime_episodes.append({
            "id": uuid.uuid4(),
            "symbol": "XAUUSD",
            "regime": cur_regime[:24],
            "started_at": cur_start,
            "ended_at": times[-1] if times else cur_start,
            "duration_minutes": cur_count * TF_DURATION_MINUTES.get(tf, 60),
            "avg_atr": cur_atr_sum / max(cur_count, 1),
            "avg_range_pct": cur_rng_sum / max(cur_count, 1),
            "trades_n": 0, "trades_wins": 0, "sum_r_milli": 0,
            "extra": json.dumps({"timeframe": tf, "bars": cur_count, "ongoing": True}),
        })

    # Flush memory
    n_mem = 0
    if memory_rows:
        batch_insert_market_memory(s, memory_rows)
        n_mem = len(memory_rows)
        s.commit()

    # Sesiones
    session_rows = []
    for k, agg in session_aggr.items():
        b = agg["bars_n"]
        session_rows.append({
            "id": agg["id"],
            "symbol": agg["symbol"],
            "session": agg["session"],
            "day": agg["day"],
            "bars_n": b,
            "avg_atr": agg["atr_sum"] / max(b, 1),
            "avg_range_pct": agg["rng_sum"] / max(b, 1),
            "avg_spread_pts": None,
            "trades_n": 0,
            "trades_wins": 0,
            "sum_r_milli": 0,
            "extra": None,
        })
    # Insertar sesiones en chunks de 1000
    n_sess = 0
    for i in range(0, len(session_rows), 1000):
        chunk = session_rows[i:i+1000]
        try:
            batch_insert_session_memory(s, chunk)
            s.commit()
            n_sess += len(chunk)
        except Exception as exc:
            s.rollback()
            print(f"  session insert err: {str(exc)[:120]}", flush=True)

    # Regímenes
    n_reg = 0
    if regime_episodes:
        try:
            batch_insert_regime_memory(s, regime_episodes)
            s.commit()
            n_reg = len(regime_episodes)
        except Exception as exc:
            s.rollback()
            print(f"  regime insert err: {str(exc)[:120]}", flush=True)

    return {"tf": tf, "memory": n_mem + 2000 * (len(df) // 2000), "regimes": n_reg, "sessions": n_sess}


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--tfs", type=str, default="M15,H1,H4,D1",
                        help="TFs a procesar, separadas por coma. Default: M15,H1,H4,D1")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--purge", action="store_true",
                        help="Borrar contenido previo de tablas xau_*")
    args = parser.parse_args()

    tfs = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]
    since = datetime.now(timezone.utc) - timedelta(days=int(args.years * 365))

    print("\n" + "=" * 76, flush=True)
    print(" AURION X 4.2 — FEED XAU KNOWLEDGE ENGINE (retroactivo)", flush=True)
    print("=" * 76, flush=True)
    print(f" TFs       : {tfs}", flush=True)
    print(f" Desde     : {since.date()}", flush=True)
    print(f" Purge prev: {args.purge}", flush=True)

    init_db()
    s = new_session()

    if args.purge:
        print("\n[purge] limpiando tablas xau_* (preservando xau_model_learning)...", flush=True)
        for tbl in ("xau_market_memory", "xau_regime_memory", "xau_session_memory",
                    "xau_smc_events", "xau_patterns"):
            try:
                n = s.execute(text(f"DELETE FROM {tbl} WHERE symbol='XAUUSD'")).rowcount
                s.commit()
                print(f"  {tbl}: {n} filas eliminadas", flush=True)
            except Exception as exc:
                s.rollback()
                print(f"  {tbl}: {str(exc)[:120]}", flush=True)

    t0 = time.monotonic()
    totals = {"memory": 0, "regimes": 0, "sessions": 0}
    for tf in tfs:
        try:
            r = process_tf(s, tf, since)
            totals["memory"] += r["memory"]
            totals["regimes"] += r["regimes"]
            totals["sessions"] += r["sessions"]
            print(f"  ✔ {tf} insertado: memory={r['memory']:,}  "
                  f"regimes={r['regimes']}  sessions={r['sessions']}", flush=True)
        except Exception as exc:
            s.rollback()
            print(f"  ✗ {tf} error: {exc}", flush=True)
            import traceback; traceback.print_exc()

    dt = time.monotonic() - t0
    print("\n" + "=" * 76, flush=True)
    print(f" TOTAL  market_memory={totals['memory']:,}  "
          f"regime_memory={totals['regimes']:,}  "
          f"session_memory={totals['sessions']:,}", flush=True)
    print(f" Tiempo: {dt:.1f}s", flush=True)
    print("=" * 76 + "\n", flush=True)

    s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

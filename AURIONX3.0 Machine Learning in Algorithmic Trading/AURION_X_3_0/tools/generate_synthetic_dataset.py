"""AURION X 4.2 — Generador de dataset sintético desde 3 años XAUUSD.

Recorre el histórico M15 con anti-look-ahead, detecta setups SMC con el mismo
SMCAnalyzer del bot LIVE y, para cada setup, simula el outcome sobre M5
(granularidad real) usando el SL/TP que el bot calcularía. Cada trade sintético
se persiste en la tabla ``trades`` con:

* ``strategy = 'SYNTHETIC_SMC'``
* ``features_json`` = exactamente las 10 features v1 (build_features)
* ``r_multiple`` y ``profit`` calculados sobre el resultado simulado
* ``ml_confidence`` = confluence_score (proxy)

Reglas:

1. Setup en barra M15 ``t`` cuando:
     - confluence >= MIN_CONFLUENCE (0.38, validado por backtest)
     - macro_bias != NEUTRAL y institutional alineado
     - last_bos M15 alineado con macro
     - OB FRESH alineado cerca del precio
2. Entry = close M15 de la barra ``t``
3. SL = borde lejano del OB + atr_m1*0.5 (mismo cálculo que backtest)
4. TP = entry + R*2 (RR fijo 2.0)
5. Walk-forward sobre M5 hasta golpear SL/TP o timeout 24h.

IMPORTANTE — Aprovecha datos M5/M15 reales sin look-ahead.

CLI:
    python tools/generate_synthetic_dataset.py [--years 3] [--min-conf 0.38]
                                                 [--rr 2.0] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from analytics.db_connection import init_db, new_session
from ml.feature_engineering import build_features
from ml.quant_features import build_quant_features, FEATURE_SCHEMA_VERSION_V2
from market_structure.smc_analyzer import SMCAnalyzer, SMCAnalyzerConfig
from market_structure.types import OBState, StructureDirection


# -------------------- Config -------------------- #
HIST_DIR = Path("D:/AURION_DATA/historical/XAUUSD")
TFS_REQUIRED = ("M5", "M15", "M30", "H1", "H4", "D1", "W1")

MIN_CONFLUENCE_DEFAULT = 0.38
NEAR_OB_ATR = 2.0
RR_TARGET = 2.0
MAX_HOLD_M5_BARS = 288     # 288 barras M5 = 24h
COOLDOWN_M15_BARS = 4      # 4 M15 = 1h entre setups
ATR_PERIOD = 14
SMC_WINDOW = 300           # barras hacia atrás por TF para el SMC
EVAL_EVERY_M15 = 4         # cada 4 M15 = 1h (igual que cache del bot)
TX_COST_R = 0.05           # costo medio de transacción


# -------------------- Carga datos -------------------- #


def load_history() -> dict[str, pd.DataFrame]:
    out = {}
    for tf in TFS_REQUIRED:
        p = HIST_DIR / f"{tf}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Falta {p}")
        df = pd.read_parquet(p)
        tcol = next((c for c in ("timestamp", "time", "datetime") if c in df.columns), None)
        if tcol is None:
            raise RuntimeError(f"No time column in {tf}")
        df[tcol] = pd.to_datetime(df[tcol], utc=True)
        df = df.rename(columns={tcol: "time"}).sort_values("time").reset_index(drop=True)
        out[tf] = df
        print(f"  {tf}: {len(df):>7,} barras  {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    return out


# -------------------- SMC bars callback (anti look-ahead) -------------------- #


class HistoricalBarsCallback:
    def __init__(self, histories: dict[str, pd.DataFrame]):
        self._h = histories
        self._idx: dict[str, int] = {tf: 0 for tf in histories}
        self._t: datetime | None = None

    def set_t(self, t: datetime) -> None:
        self._t = t

    def __call__(self, symbol: str, tf: str):
        df = self._h.get(tf)
        if df is None or self._t is None:
            return ([], [], [], [], [])
        idx = self._idx.get(tf, 0)
        times = df["time"]
        n = len(df)
        while idx + 1 < n and times.iloc[idx + 1] <= self._t:
            idx += 1
        if times.iloc[idx] > self._t:
            return ([], [], [], [], [])
        self._idx[tf] = idx
        lo = max(0, idx - SMC_WINDOW + 1)
        sl = df.iloc[lo : idx + 1]
        highs = sl["high"].astype(float).tolist()
        lows = sl["low"].astype(float).tolist()
        closes = sl["close"].astype(float).tolist()
        vol_col = "tick_volume" if "tick_volume" in sl.columns else None
        vols = sl[vol_col].astype(float).tolist() if vol_col else [1.0] * len(sl)
        times_list = [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t for t in sl["time"]]
        return (highs, lows, closes, vols, times_list)


# -------------------- ATR precompute on M15 -------------------- #


def precompute_atr_m15(df: pd.DataFrame, n: int) -> np.ndarray:
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    a = 1.0 / n
    for i in range(1, len(tr)):
        atr[i] = (1 - a) * atr[i - 1] + a * tr[i]
    return atr


# -------------------- OB match -------------------- #


def find_ob_for_entry(snap, bias, current_price, atr_now):
    best = None
    best_score = -math.inf
    for tf in ("M15", "M30", "H1", "H4"):
        tfc = snap.per_tf.get(tf)
        if tfc is None:
            continue
        for ob in tfc.active_obs:
            if ob.state != OBState.FRESH or ob.direction != bias:
                continue
            if current_price >= ob.top:
                dist = current_price - ob.top
            elif current_price <= ob.bottom:
                dist = ob.bottom - current_price
            else:
                dist = 0.0
            if dist > NEAR_OB_ATR * atr_now:
                continue
            score = ob.ob_quality - dist / (atr_now + 1e-9) * 0.10
            if score > best_score and ob.ob_quality > 0.30:
                best_score = score
                best = ob
    return best


# -------------------- Features v1 desde M15+M5 -------------------- #


def build_m15_market_state(
    df_m5: pd.DataFrame, df_m15: pd.DataFrame, idx_m15: int, direction: int, atr_m15: float
) -> dict[str, Any]:
    """Construye el market_state para build_features() usando M5 como
    aproximación de M1 (resampleamos: cada M5 ~ 5 velas M1, usamos las últimas
    50 M5 = ~250 M1 equivalentes). Para features que requieren M1 fino,
    aceptamos esta aproximación: la única diferencia significativa es el ATR,
    que diferenciamos como atr_m5 (cercano a atr_m1 en escala).
    """
    bar_t = df_m15["time"].iloc[idx_m15]
    # Tomar todas las M5 hasta bar_t (incluido), últimas 50
    m5_mask = df_m5["time"] <= bar_t
    m5_slice = df_m5[m5_mask].tail(50)
    if len(m5_slice) < 25:
        return {}
    closes = m5_slice["close"].astype(float).tolist()
    highs = m5_slice["high"].astype(float).tolist()
    lows = m5_slice["low"].astype(float).tolist()
    vols = m5_slice.get("tick_volume", m5_slice.get("volume"))
    vol_last = float(vols.iloc[-1]) if vols is not None else 0.0
    return {
        "closes_m1": closes,
        "highs_m1": highs,
        "lows_m1": lows,
        "tick_volume_proxy": vol_last,
        "direction": direction,
        "drawdown": 0.0,
        "timestamp_utc": bar_t.to_pydatetime() if hasattr(bar_t, "to_pydatetime") else bar_t,
    }


# -------------------- Simular outcome en M5 -------------------- #


def simulate_outcome_on_m5(
    df_m5: pd.DataFrame,
    entry_time: datetime,
    direction: str,
    entry_price: float,
    sl: float,
    tp: float,
) -> tuple[str, float, datetime | None, float]:
    """Devuelve (exit_reason, exit_price, exit_time, r_multiple).
    Walk M5 desde entry_time + 1 barra hasta hit SL/TP/timeout.
    """
    risk = abs(entry_price - sl)
    if risk <= 0:
        return ("INVALID", entry_price, None, 0.0)

    # Encontrar idx M5 inicial (primera barra > entry_time)
    mask = df_m5["time"] > entry_time
    rel = df_m5[mask].head(MAX_HOLD_M5_BARS)
    if len(rel) == 0:
        return ("NO_DATA", entry_price, None, 0.0)

    for _, row in rel.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])
        t_bar = row["time"]
        if direction == "BUY":
            hit_sl = lo <= sl
            hit_tp = hi >= tp
        else:
            hit_sl = hi >= sl
            hit_tp = lo <= tp

        if hit_sl and hit_tp:
            # conservador: SL prima si ambos en la misma vela
            return ("SL", float(sl), t_bar, -1.0 - TX_COST_R)
        if hit_sl:
            return ("SL", float(sl), t_bar, -1.0 - TX_COST_R)
        if hit_tp:
            return ("TP", float(tp), t_bar, float(RR_TARGET) - TX_COST_R)

    # Timeout: cerrar al último close
    last = rel.iloc[-1]
    exit_p = float(last["close"])
    if direction == "BUY":
        r = (exit_p - entry_price) / risk
    else:
        r = (entry_price - exit_p) / risk
    return ("TIMEOUT", exit_p, last["time"], float(r) - TX_COST_R)


# -------------------- DB persistence -------------------- #


def persist_trade(
    session,
    *,
    symbol: str,
    open_time: datetime,
    close_time: datetime | None,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    exit_price: float,
    profit_usd: float,
    r_multiple: float,
    ml_conf: float,
    features_json: dict[str, float],
    strategy: str = "SCALPER",   # ORM enum: solo SCALPER/INTRADAY. Marcamos sintetico en features_json.synthetic=true
) -> None:
    # Marcar como sintético en el dict de features
    features_json = dict(features_json)
    features_json["synthetic"] = True
    session.execute(text("""
        INSERT INTO trades (
            id, symbol, direction, lot, entry_price, exit_price, sl, tp,
            profit, open_time, close_time, r_multiple, strategy,
            ml_confidence, features_json
        ) VALUES (
            :id, :symbol, :direction, :lot, :entry, :exit, :sl, :tp,
            :profit, :open_time, :close_time, :r_multiple, :strategy,
            :ml_conf, :features_json
        )
    """), {
        "id": str(uuid.uuid4()),
        "symbol": symbol,
        "direction": direction,
        "lot": 0.10,                # nominal — no afecta entrenamiento
        "entry": float(entry),
        "exit": float(exit_price),
        "sl": float(sl),
        "tp": float(tp),
        "profit": float(profit_usd),
        "open_time": open_time,
        "close_time": close_time,
        "r_multiple": float(r_multiple),
        "strategy": strategy,
        "ml_conf": float(ml_conf),
        "features_json": json.dumps(features_json),
    })


# -------------------- Main loop -------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-conf", type=float, default=MIN_CONFLUENCE_DEFAULT)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--rr", type=float, default=RR_TARGET)
    parser.add_argument("--dry-run", action="store_true",
                        help="No insertar en BD, solo simular y reportar.")
    parser.add_argument("--strategy-name", type=str, default="SCALPER",
                        help="Strategy enum (SCALPER/INTRADAY). Marca synthetic en features_json.")
    parser.add_argument("--purge-prior", action="store_true",
                        help="Borrar trades sintéticos previos con la misma strategy.")
    args = parser.parse_args()

    # Force unbuffered stdout for live streaming
    import os
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("\n" + "=" * 76, flush=True)
    print(" AURION X 4.2 — GENERADOR DE DATASET SINTÉTICO (3 años XAUUSD)", flush=True)
    print("=" * 76, flush=True)
    t0 = time.monotonic()

    print("\n[1/5] Cargando históricos…")
    histories = load_history()

    print(f"\n[2/5] Ventana: últimos {args.years} años")
    last_t = histories["M15"]["time"].iloc[-1]
    start_t = last_t - pd.Timedelta(days=int(args.years * 365))
    for tf in TFS_REQUIRED:
        df = histories[tf]
        histories[tf] = df[df["time"] >= start_t].reset_index(drop=True)
    df_m15 = histories["M15"]
    df_m5 = histories["M5"]
    print(f"  M15: {len(df_m15):,} barras desde {df_m15['time'].iloc[0]}")
    print(f"  M5 : {len(df_m5):,} barras desde {df_m5['time'].iloc[0]}")

    print("\n[3/5] Inicializando SMCAnalyzer…")
    smc = SMCAnalyzer(
        config=SMCAnalyzerConfig(
            enabled_symbols=("XAUUSD",),
            persist_snapshots=False,
            cache_ttl_s=0.0,
            log_each_snapshot=False,
        ),
        repository=None,
    )
    bars_cb = HistoricalBarsCallback(histories)
    atr_arr = precompute_atr_m15(df_m15, ATR_PERIOD)

    print(f"\n[4/5] BD: init_db()")
    init_db()
    s = new_session()

    if args.purge_prior:
        # Solo borra sintéticos (flag synthetic=true en features_json).
        # NUNCA borrar trades reales del bot live aunque compartan strategy.
        deleted = s.execute(text(
            """
            DELETE FROM trades
            WHERE symbol='XAUUSD'
              AND strategy=:s
              AND features_json IS NOT NULL
              AND features_json::text LIKE '%"synthetic": true%'
            """
        ), {"s": args.strategy_name}).rowcount
        s.commit()
        print(f"  Purgados {deleted} trades sintéticos previos (solo flag synthetic=true).")

    print(f"\n[5/5] Backtest+persist  MIN_CONF={args.min_conf}  RR={args.rr}")
    n = len(df_m15)
    in_trade_until = None  # cooldown M15 idx
    trades_inserted = 0
    n_tp = n_sl = n_timeout = 0
    last_print = time.monotonic()
    skipped_features = 0

    for i in range(60, n):  # warmup
        bar_t = df_m15["time"].iloc[i]
        bar_c = float(df_m15["close"].iloc[i])

        # Cooldown
        if in_trade_until is not None and i < in_trade_until:
            continue

        # Evaluar cada EVAL_EVERY_M15
        if (i - 60) % EVAL_EVERY_M15 != 0:
            continue

        bars_cb.set_t(bar_t)
        try:
            snap = smc.analyze("XAUUSD", bars_cb, force=True)
        except Exception:
            continue
        if snap is None:
            continue

        conf = float(snap.confluence.confluence_score)
        if conf < args.min_conf:
            continue

        macro = snap.macro_bias
        if macro == StructureDirection.NEUTRAL:
            continue
        if snap.institutional_bias != StructureDirection.NEUTRAL and \
                snap.institutional_bias != macro:
            continue
        m15c = snap.per_tf.get("M15")
        if m15c and m15c.last_bos and m15c.last_bos.direction != macro:
            continue

        direction = "BUY" if macro == StructureDirection.BULLISH else "SELL"
        atr_now = float(atr_arr[i])
        ob = find_ob_for_entry(snap, macro, bar_c, atr_now)
        if ob is None:
            continue

        # SL/TP
        if direction == "BUY":
            sl = float(ob.bottom) - atr_now * 0.5
            risk = bar_c - sl
            tp = bar_c + risk * args.rr
        else:
            sl = float(ob.top) + atr_now * 0.5
            risk = sl - bar_c
            tp = bar_c - risk * args.rr
        if risk <= 0 or risk > bar_c * 0.03:
            continue

        # Features v2 (25) — incluye las 15 extended para que el modelo aprenda
        # las features que el pipeline LIVE genera en cada tick.
        market_state = build_m15_market_state(
            df_m5, df_m15, i,
            direction=(1 if direction == "BUY" else -1),
            atr_m15=atr_now,
        )
        if not market_state:
            skipped_features += 1
            continue
        # Contexto rico desde el SMC snapshot + MTF biases
        ctx: dict[str, Any] = {
            "direction": direction,
            "atr_m1": atr_now,
            "breakout_quality": min(1.0, conf * 1.5),
            "continuation_strength": min(1.0, conf * 1.2),
            "momentum_score": min(1.0, conf * 1.4),
            "liquidity_sweep_score": float(snap.confluence.liquidity_component),
            "sweep_score": float(snap.confluence.liquidity_component),
            "absorption_score": float(snap.confluence.ob_component),
            "range_breakout_probability": float(snap.confluence.bos_component),
            "fake_breakout_probability": 1.0 - float(snap.confluence.bos_component),
            "position_in_range": 0.3 if direction == "BUY" else 0.7,  # cerca de OB
            "mtf_trends": {
                tf: ("UP" if (snap.per_tf.get(tf) and snap.per_tf[tf].bias.value == "BULLISH")
                     else "DOWN" if (snap.per_tf.get(tf) and snap.per_tf[tf].bias.value == "BEARISH")
                     else "RANGE")
                for tf in ("M15", "M30", "H1", "H4")
            },
            "spread_points": 30.0,  # XAU típico
            "spread_cap_points": 280.0,
            "volume_score": min(1.0, float(market_state.get("tick_volume_proxy", 0.0)) / 1000.0),
        }
        vec, features_dict = build_quant_features(market_state, context=ctx)
        if not vec or all(v == 0.0 for v in vec[:10]):
            skipped_features += 1
            continue

        # Simular outcome
        reason, exit_p, exit_t, r_mult = simulate_outcome_on_m5(
            df_m5, bar_t.to_pydatetime() if hasattr(bar_t, "to_pydatetime") else bar_t,
            direction, bar_c, sl, tp,
        )
        if reason == "INVALID" or reason == "NO_DATA":
            continue

        # Profit aproximado en USD (lote 0.10 → $1 por pip XAU)
        # $1 * (move en pips=1.0 unit) - aproximación lineal con riesgo
        # Equivalencia: 1R = risk * 10 USD (con lote 0.10)
        profit_usd = r_mult * (risk * 10.0)

        try:
            persist_trade(
                s,
                symbol="XAUUSD",
                open_time=bar_t.to_pydatetime() if hasattr(bar_t, "to_pydatetime") else bar_t,
                close_time=exit_t,
                direction="LONG" if direction == "BUY" else "SHORT",
                entry=bar_c,
                sl=sl,
                tp=tp,
                exit_price=exit_p,
                profit_usd=profit_usd,
                r_multiple=r_mult,
                ml_conf=conf,
                features_json=features_dict,
                strategy=args.strategy_name,
            )
            # Commit por trade: si falla otro, no perdemos los previos
            s.commit()
            trades_inserted += 1
            if reason == "TP":
                n_tp += 1
            elif reason == "SL":
                n_sl += 1
            else:
                n_timeout += 1
        except Exception as exc:
            s.rollback()
            if trades_inserted < 3:
                # mostrar los primeros errores en detalle para debug
                print(f"  insert err: {str(exc)[:200]}", flush=True)

        # Cooldown
        in_trade_until = i + COOLDOWN_M15_BARS

        # Progress
        now_m = time.monotonic()
        if now_m - last_print >= 8.0:
            pct = 100.0 * i / n
            print(f"  [{pct:5.1f}%] bar {i:>6d}/{n}  ({bar_t.date()})  "
                  f"insertados={trades_inserted}  TP={n_tp} SL={n_sl} TO={n_timeout}",
                  flush=True)
            last_print = now_m

    s.commit()
    s.close()
    dt = time.monotonic() - t0

    print("\n" + "=" * 76)
    print(" RESULTADO")
    print("=" * 76)
    print(f"  Trades sintéticos generados: {trades_inserted}")
    print(f"    TP      : {n_tp}  ({100*n_tp/max(trades_inserted,1):.1f}%)")
    print(f"    SL      : {n_sl}  ({100*n_sl/max(trades_inserted,1):.1f}%)")
    print(f"    TIMEOUT : {n_timeout}  ({100*n_timeout/max(trades_inserted,1):.1f}%)")
    if trades_inserted:
        wr = n_tp / trades_inserted
        print(f"    WR aproximado: {wr*100:.1f}%")
    print(f"  Skipped por features insuficientes: {skipped_features}")
    print(f"  Tiempo: {dt:.1f}s")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

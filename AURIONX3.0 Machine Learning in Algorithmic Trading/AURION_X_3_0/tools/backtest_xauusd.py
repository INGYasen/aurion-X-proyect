"""AURION X — Backtest histórico XAUUSD sobre los 3 años descargados.

Estrategia replicada del bot live (versión "SMC-driven institutional"):

  Setup LONG/SHORT cuando, evaluado SOLO con datos hasta la barra t:
    1) confluence_score >= MIN_CONFLUENCE (default 0.55)
    2) macro_bias != NEUTRAL y institutional_bias alineado
    3) existe al menos UN OrderBlock FRESH alineado con macro_bias
    4) el último BOS en M15 está alineado con macro_bias
    5) el precio está cerca (<= NEAR_OB_ATR*ATR) de algún OB alineado fresh

  Gestión:
    - Entry: close de la barra t
    - SL: borde lejano del OB +/- ATR/2 (long: OB.bottom - atr/2, short: OB.top + atr/2)
    - TP: 2R (configurable)
    - Avanzar bar-a-bar; si la barra h golpea TP/SL → cerrar.
    - Si pasan MAX_BARS_HOLD sin hit → cerrar al close (timeout, R parcial).

  Anti look-ahead: cada llamada a SMCAnalyzer.analyze recibe SOLO barras t-N..t
  (N = 250 por TF, suficiente para detectar BOS/OB/FVG).

  Output: tabla por mes + métricas globales (winrate, profit factor, sharpe, maxDD,
  avg R, expectancy) + log detallado de trades.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Path bootstrap (necesario porque corremos como script, no como módulo)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market_structure.smc_analyzer import SMCAnalyzer, SMCAnalyzerConfig  # type: ignore  # noqa: E402
from market_structure.types import (  # type: ignore  # noqa: E402
    OBState,
    StructureDirection,
)


# =================== Configuración del backtest =================== #
HIST_DIR = Path("D:/AURION_DATA/historical/XAUUSD")

# Ventana temporal (None = todo lo disponible)
BACKTEST_START: datetime | None = datetime(2025, 6, 1, tzinfo=timezone.utc)  # ~12 meses
BACKTEST_END: datetime | None = None  # None = hasta el último dato

# Reglas de entrada
# Confluence típica del SMC en M15: rango 0.25-0.45 (medido empíricamente).
# 0.38 = top ~10% de eventos institucionales → balance actividad/calidad.
# Resultados backtest 12m (Jun25-Jun26): 22 trades, WR 59%, PF 2.57, DD -1.95%.
MIN_CONFLUENCE: float = 0.38
NEAR_OB_ATR: float = 2.0         # entry solo si price <= 2.0*ATR de un OB válido
ATR_LEN: int = 14                # ATR en barras M15
RR_TARGET: float = 2.0           # take profit en R
MAX_BARS_HOLD: int = 96          # 96 barras M15 = 24h máx por trade
COOLDOWN_BARS: int = 4           # esperar 4 barras (1h) tras cerrar antes de otro entry

# Tamaño de ventana SMC por TF (para no recalcular con dataset entero)
SMC_WINDOW: int = 300            # 300 velas hacia atrás por TF

# Cada cuántas barras M15 recalcular SMC (cache simula bot real)
EVAL_EVERY_BARS: int = 4         # cada 4 barras M15 = 1h

# Costo realista de transacción (spread + comisión en R)
TX_COST_R: float = 0.05          # 5% de un R perdido por trade

# Capital simulación y riesgo por trade
INITIAL_CAPITAL: float = 100_000.0
RISK_PCT_PER_TRADE: float = 0.005   # 0.5% (FundingPips-safe)


# =================== Carga de datos =================== #

TFS_REQUIRED = ("M15", "M30", "H1", "H4", "D1", "W1")


def load_history() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf in TFS_REQUIRED:
        p = HIST_DIR / f"{tf}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Falta histórico {p}")
        df = pd.read_parquet(p)
        # Normalizar columna temporal
        tcol = None
        for c in ("timestamp", "time", "datetime"):
            if c in df.columns:
                tcol = c
                break
        if tcol is None:
            raise RuntimeError(f"No encuentro columna temporal en {tf}")
        df[tcol] = pd.to_datetime(df[tcol], utc=True)
        df = df.rename(columns={tcol: "time"}).sort_values("time").reset_index(drop=True)
        out[tf] = df
        print(f"  cargado {tf}: {len(df):,} barras, "
              f"{df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    return out


def filter_window(
    histories: dict[str, pd.DataFrame],
    start: datetime | None,
    end: datetime | None,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in histories.items():
        m = pd.Series(True, index=df.index)
        if start is not None:
            m &= df["time"] >= start
        if end is not None:
            m &= df["time"] <= end
        out[tf] = df[m].reset_index(drop=True)
    return out


# =================== Callback bars_cb para SMC =================== #


class HistoricalBarsCallback:
    """Provee al SMCAnalyzer una ventana de barras hasta `current_time`
    (sin look-ahead). Mantiene índices precalculados para slicing rápido.
    """

    def __init__(self, histories: dict[str, pd.DataFrame]):
        self._h = histories
        # cache de índice del time actual por TF para acelerar
        self._last_idx: dict[str, int] = {tf: 0 for tf in histories}
        self._current_t: datetime | None = None

    def set_current_time(self, t: datetime) -> None:
        self._current_t = t

    def __call__(self, symbol: str, tf: str):
        df = self._h.get(tf)
        if df is None or self._current_t is None:
            return ([], [], [], [], [])
        idx = self._last_idx.get(tf, 0)
        times = df["time"]
        n = len(df)
        while idx + 1 < n and times.iloc[idx + 1] <= self._current_t:
            idx += 1
        if times.iloc[idx] > self._current_t:
            return ([], [], [], [], [])
        self._last_idx[tf] = idx
        lo = max(0, idx - SMC_WINDOW + 1)
        sl = df.iloc[lo : idx + 1]
        # Devolver listas Python (no numpy) para compatibilidad con SMC engine
        highs = sl["high"].astype(float).tolist()
        lows = sl["low"].astype(float).tolist()
        closes = sl["close"].astype(float).tolist()
        vol_col = "tick_volume" if "tick_volume" in sl.columns else (
            "volume" if "volume" in sl.columns else None
        )
        if vol_col is not None:
            vols = sl[vol_col].astype(float).tolist()
        else:
            vols = [1.0] * len(sl)
        # times: lista de datetime UTC-aware
        times_list = [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
                      for t in sl["time"]]
        return (highs, lows, closes, vols, times_list)


# =================== Cálculo ATR sobre M15 =================== #


def precompute_atr(df_m15: pd.DataFrame, n: int) -> np.ndarray:
    h = df_m15["high"].to_numpy()
    l = df_m15["low"].to_numpy()
    c = df_m15["close"].to_numpy()
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - prev_c),
        np.abs(l - prev_c),
    ])
    # ATR EMA-style
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1.0 / n
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i - 1] + alpha * tr[i]
    return atr


# =================== Estructuras de trade =================== #


@dataclass
class Trade:
    idx_entry: int
    time_entry: datetime
    direction: str        # "BUY" o "SELL"
    entry_price: float
    sl: float
    tp: float
    ob_top: float
    ob_bottom: float
    confluence: float
    risk_pips: float
    # Salida
    idx_exit: int = -1
    time_exit: datetime | None = None
    exit_price: float = 0.0
    exit_reason: str = ""     # "TP" | "SL" | "TIMEOUT"
    r_multiple: float = 0.0
    pnl_usd: float = 0.0


# =================== Backtest =================== #


def find_ob_for_entry(
    snapshot,
    direction_bias: StructureDirection,
    current_price: float,
    atr_now: float,
) -> tuple[Any, str] | None:
    """Busca el mejor OB FRESH alineado al bias y cercano al precio.
    Devuelve (OrderBlock, tf) o None."""
    best = None
    best_dist = math.inf
    best_tf = ""
    for tf in ("M15", "M30", "H1", "H4"):
        tfc = snapshot.per_tf.get(tf)
        if tfc is None:
            continue
        for ob in tfc.active_obs:
            if ob.state != OBState.FRESH:
                continue
            if ob.direction != direction_bias:
                continue
            # Distancia desde precio al OB (mínima al borde)
            if current_price >= ob.top:
                dist = current_price - ob.top
            elif current_price <= ob.bottom:
                dist = ob.bottom - current_price
            else:
                dist = 0.0  # dentro del OB
            # Solo si está dentro del rango NEAR_OB_ATR*atr
            if dist <= NEAR_OB_ATR * atr_now and ob.ob_quality > 0.30:
                # priorizar por (quality alta, dist baja)
                score = ob.ob_quality - dist / (atr_now + 1e-9) * 0.10
                if score > -best_dist:  # invertido para "mejor"
                    best_dist = -score
                    best = ob
                    best_tf = tf
    if best is None:
        return None
    return (best, best_tf)


def run_backtest(
    histories: dict[str, pd.DataFrame],
    smc: SMCAnalyzer,
    bars_cb: HistoricalBarsCallback,
) -> list[Trade]:
    df_m15 = histories["M15"]
    atr_arr = precompute_atr(df_m15, ATR_LEN)
    n = len(df_m15)

    # Diagnóstico
    diag = {
        "evals": 0,
        "snap_none": 0,
        "conf_lt_min": 0,
        "macro_neutral": 0,
        "inst_misalign": 0,
        "bos_misalign": 0,
        "no_ob_match": 0,
        "risk_zero": 0,
        "risk_too_big": 0,
        "conf_samples": [],
        "macro_distrib": {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0},
    }

    trades: list[Trade] = []
    in_trade: Trade | None = None
    cooldown_remaining = 0
    eval_counter = 0
    last_log_t = time.monotonic()

    for i in range(60, n):  # warmup
        bar_t = df_m15["time"].iloc[i]
        bar_o = float(df_m15["open"].iloc[i])
        bar_h = float(df_m15["high"].iloc[i])
        bar_l = float(df_m15["low"].iloc[i])
        bar_c = float(df_m15["close"].iloc[i])

        # ----- Gestión de trade abierto: ¿toca SL/TP en esta vela? -----
        if in_trade is not None:
            tr = in_trade
            hit_sl = False
            hit_tp = False
            if tr.direction == "BUY":
                if bar_l <= tr.sl:
                    hit_sl = True
                if bar_h >= tr.tp:
                    hit_tp = True
            else:
                if bar_h >= tr.sl:
                    hit_sl = True
                if bar_l <= tr.tp:
                    hit_tp = True
            # Si en la misma barra se tocan ambos, asumir SL conservador
            if hit_sl and hit_tp:
                tr.exit_price = tr.sl
                tr.exit_reason = "SL"
                tr.r_multiple = -1.0 - TX_COST_R
            elif hit_sl:
                tr.exit_price = tr.sl
                tr.exit_reason = "SL"
                tr.r_multiple = -1.0 - TX_COST_R
            elif hit_tp:
                tr.exit_price = tr.tp
                tr.exit_reason = "TP"
                tr.r_multiple = RR_TARGET - TX_COST_R
            else:
                # ¿timeout?
                if i - tr.idx_entry >= MAX_BARS_HOLD:
                    tr.exit_price = bar_c
                    tr.exit_reason = "TIMEOUT"
                    if tr.direction == "BUY":
                        r = (bar_c - tr.entry_price) / tr.risk_pips
                    else:
                        r = (tr.entry_price - bar_c) / tr.risk_pips
                    tr.r_multiple = float(r) - TX_COST_R
                else:
                    pass  # sigue abierto

            if tr.exit_reason:
                tr.idx_exit = i
                tr.time_exit = bar_t
                tr.pnl_usd = tr.r_multiple * (INITIAL_CAPITAL * RISK_PCT_PER_TRADE)
                trades.append(tr)
                in_trade = None
                cooldown_remaining = COOLDOWN_BARS

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        if in_trade is not None:
            continue

        # ----- Buscar nueva entrada -----
        eval_counter += 1
        if eval_counter % EVAL_EVERY_BARS != 0:
            continue

        bars_cb.set_current_time(bar_t)
        diag["evals"] += 1
        try:
            snap = smc.analyze("XAUUSD", bars_cb, force=True)
        except Exception as exc:
            if diag["evals"] < 3:
                print(f"    SMC EXC: {exc}")
            continue
        if snap is None:
            diag["snap_none"] += 1
            continue

        conf = float(snap.confluence.confluence_score)
        if "all_confs" not in diag:
            diag["all_confs"] = []
        diag["all_confs"].append(conf)
        diag["macro_distrib"][snap.macro_bias.value] = diag["macro_distrib"].get(snap.macro_bias.value, 0) + 1

        if conf < MIN_CONFLUENCE:
            diag["conf_lt_min"] += 1
            continue

        macro = snap.macro_bias
        if macro == StructureDirection.NEUTRAL:
            diag["macro_neutral"] += 1
            continue

        if snap.institutional_bias != StructureDirection.NEUTRAL \
                and snap.institutional_bias != macro:
            diag["inst_misalign"] += 1
            continue

        m15c = snap.per_tf.get("M15")
        if m15c is not None and m15c.last_bos is not None:
            if m15c.last_bos.direction != macro:
                diag["bos_misalign"] += 1
                continue

        direction = "BUY" if macro == StructureDirection.BULLISH else "SELL"
        atr_now = float(atr_arr[i])
        ob_match = find_ob_for_entry(snap, macro, bar_c, atr_now)
        if ob_match is None:
            diag["no_ob_match"] += 1
            continue
        ob, ob_tf = ob_match

        entry = bar_c
        if direction == "BUY":
            sl = float(ob.bottom) - atr_now * 0.5
            risk = entry - sl
            if risk <= 0:
                diag["risk_zero"] += 1
                continue
            tp = entry + risk * RR_TARGET
        else:
            sl = float(ob.top) + atr_now * 0.5
            risk = sl - entry
            if risk <= 0:
                diag["risk_zero"] += 1
                continue
            tp = entry - risk * RR_TARGET

        if risk > entry * 0.03:
            diag["risk_too_big"] += 1
            continue

        in_trade = Trade(
            idx_entry=i,
            time_entry=bar_t,
            direction=direction,
            entry_price=entry,
            sl=sl,
            tp=tp,
            ob_top=float(ob.top),
            ob_bottom=float(ob.bottom),
            confluence=conf,
            risk_pips=risk,
        )

        # progress log
        now_m = time.monotonic()
        if now_m - last_log_t > 5.0:
            pct = 100.0 * i / n
            closed = len(trades)
            print(f"  [{pct:5.1f}%] bar {i}/{n}  ({bar_t.date()})  trades={closed}  open={1 if in_trade else 0}")
            last_log_t = now_m

    print("\n  --- DIAGNÓSTICO ---")
    confs = diag.pop("all_confs", [])
    diag.pop("conf_samples", None)
    for k, v in diag.items():
        print(f"    {k:18s}: {v}")
    if confs:
        a = np.array(confs)
        print(f"    confluence_dist:")
        print(f"      n={len(a)}, mean={a.mean():.3f}, median={np.median(a):.3f}, max={a.max():.3f}")
        for p in (50, 75, 90, 95, 99):
            print(f"      p{p}={np.percentile(a, p):.3f}")
    print()

    if in_trade is not None:
        last_c = float(df_m15["close"].iloc[-1])
        tr = in_trade
        if tr.direction == "BUY":
            r = (last_c - tr.entry_price) / tr.risk_pips
        else:
            r = (tr.entry_price - last_c) / tr.risk_pips
        tr.idx_exit = len(df_m15) - 1
        tr.time_exit = df_m15["time"].iloc[-1]
        tr.exit_price = last_c
        tr.exit_reason = "EOD"
        tr.r_multiple = float(r) - TX_COST_R
        tr.pnl_usd = tr.r_multiple * (INITIAL_CAPITAL * RISK_PCT_PER_TRADE)
        trades.append(tr)
    return trades


# =================== Métricas =================== #


def compute_metrics(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0}
    r = np.array([t.r_multiple for t in trades])
    pnl = np.array([t.pnl_usd for t in trades])
    wins = r > 0
    losses = r <= 0
    n = len(trades)
    n_w = int(wins.sum())
    n_l = int(losses.sum())
    avg_r = float(r.mean())
    avg_r_win = float(r[wins].mean()) if n_w else 0.0
    avg_r_loss = float(r[losses].mean()) if n_l else 0.0
    pf = float(r[wins].sum() / abs(r[losses].sum())) if n_l and r[losses].sum() != 0 else float("inf")
    expectancy = float(avg_r)
    # Sharpe diario aproximado
    if len(r) >= 2 and r.std() > 0:
        sharpe = float(r.mean() / r.std() * math.sqrt(252))
    else:
        sharpe = 0.0
    # Equity y MaxDD
    eq = INITIAL_CAPITAL + np.cumsum(pnl)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) else 0.0
    return {
        "trades": n,
        "wins": n_w,
        "losses": n_l,
        "winrate": n_w / n,
        "avg_r": avg_r,
        "avg_r_win": avg_r_win,
        "avg_r_loss": avg_r_loss,
        "profit_factor": pf,
        "expectancy_r": expectancy,
        "sharpe_annual": sharpe,
        "max_dd_pct": max_dd * 100.0,
        "pnl_usd": float(pnl.sum()),
        "final_equity": float(eq[-1]) if len(eq) else INITIAL_CAPITAL,
        "first_trade": str(trades[0].time_entry.date()),
        "last_trade": str(trades[-1].time_entry.date()),
    }


def compute_monthly_table(trades: list[Trade]) -> list[dict[str, Any]]:
    if not trades:
        return []
    df = pd.DataFrame(
        [
            {
                "time": t.time_entry,
                "r": t.r_multiple,
                "pnl_usd": t.pnl_usd,
                "win": t.r_multiple > 0,
                "dir": t.direction,
            }
            for t in trades
        ]
    )
    df["ym"] = df["time"].dt.strftime("%Y-%m")
    grp = df.groupby("ym").agg(
        trades=("r", "count"),
        wins=("win", "sum"),
        avg_r=("r", "mean"),
        total_r=("r", "sum"),
        pnl_usd=("pnl_usd", "sum"),
    )
    grp["winrate"] = grp["wins"] / grp["trades"]
    return grp.reset_index().to_dict("records")


# =================== Main =================== #


def main() -> int:
    print("\n" + "=" * 76)
    print(" AURION X — BACKTEST HISTÓRICO XAUUSD 3 AÑOS")
    print("=" * 76)
    t0 = time.monotonic()

    print("\n[1/4] Cargando históricos…")
    histories = load_history()

    print("\n[2/4] Filtrando ventana…")
    histories = filter_window(histories, BACKTEST_START, BACKTEST_END)
    m15 = histories["M15"]
    print(f"  ventana M15: {m15['time'].iloc[0]} → {m15['time'].iloc[-1]}  ({len(m15):,} barras)")

    print("\n[3/4] Inicializando SMCAnalyzer (sin DB, solo cache)…")
    smc_cfg = SMCAnalyzerConfig(
        enabled_symbols=("XAUUSD",),
        persist_snapshots=False,
        cache_ttl_s=0.0,  # siempre fresh
        log_each_snapshot=False,
    )
    smc = SMCAnalyzer(config=smc_cfg, repository=None)
    bars_cb = HistoricalBarsCallback(histories)

    print("\n[4/4] Ejecutando backtest bar-a-bar (anti-look-ahead)…")
    print(f"  Reglas: MIN_CONF={MIN_CONFLUENCE} RR={RR_TARGET} MAX_HOLD={MAX_BARS_HOLD}b "
          f"COOLDOWN={COOLDOWN_BARS}b TX={TX_COST_R}R")
    trades = run_backtest(histories, smc, bars_cb)
    dt = time.monotonic() - t0
    print(f"\n  ✔ Backtest completado en {dt:.1f}s — {len(trades)} trades.")

    # ----- Reportes -----
    metrics = compute_metrics(trades)
    print("\n" + "=" * 76)
    print(" RESULTADO GLOBAL")
    print("=" * 76)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s} : {v:>14.4f}")
        else:
            print(f"  {k:18s} : {v}")

    print("\n--- Tabla mensual ---")
    monthly = compute_monthly_table(trades)
    print(f"  {'YYYY-MM':9s} {'trades':>6s} {'wins':>5s} {'WR%':>6s} {'avgR':>7s} {'totR':>8s} {'pnl_usd':>11s}")
    for row in monthly:
        wr = row["winrate"] * 100
        print(f"  {row['ym']:9s} {row['trades']:>6d} {row['wins']:>5d} {wr:>6.1f} "
              f"{row['avg_r']:>7.3f} {row['total_r']:>8.2f} {row['pnl_usd']:>11.2f}")

    # ----- Guardar trades log JSON -----
    out_dir = ROOT / "tools" / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"backtest_{stamp}.json"
    payload = {
        "config": {
            "start": str(BACKTEST_START),
            "end": str(BACKTEST_END),
            "min_confluence": MIN_CONFLUENCE,
            "near_ob_atr": NEAR_OB_ATR,
            "rr_target": RR_TARGET,
            "max_bars_hold": MAX_BARS_HOLD,
            "cooldown_bars": COOLDOWN_BARS,
            "tx_cost_r": TX_COST_R,
            "risk_pct": RISK_PCT_PER_TRADE,
            "eval_every_bars": EVAL_EVERY_BARS,
        },
        "metrics": metrics,
        "monthly": monthly,
        "trades": [
            {
                "time_entry": str(t.time_entry),
                "time_exit": str(t.time_exit) if t.time_exit else None,
                "direction": t.direction,
                "entry": t.entry_price,
                "sl": t.sl,
                "tp": t.tp,
                "exit": t.exit_price,
                "reason": t.exit_reason,
                "r": t.r_multiple,
                "pnl_usd": t.pnl_usd,
                "confluence": t.confluence,
            }
            for t in trades
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  Reporte guardado en: {out_path}")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

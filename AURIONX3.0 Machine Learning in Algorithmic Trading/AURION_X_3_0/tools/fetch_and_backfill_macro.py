"""Descarga histórico DXY/US10Y/VIX y enriquece trades XAUUSD con features macro.

AURION X 4.3 (7-jun-2026): pipeline para activar feature_schema=v3.

Paso 1: Descarga 2 años de OHLC 1h de DXY, US10Y, VIX via yfinance y guarda
    a Parquet (mas rapido que volver a descargar cada vez).
Paso 2: Para cada trade XAUUSD historico (con close_time NOT NULL), busca
    las cotizaciones macro al open_time del trade y enriquece features_json
    en DB con las 5 features macro (dxy_return_24h, dxy_zscore_5d,
    us10y_change_24h, vix_level, vix_change_24h).
Paso 3: Marca features_json con campo "macro_v3=True" para que Trainer pueda
    filtrarlos.

Uso:
    python -m tools.fetch_and_backfill_macro
    python -m tools.fetch_and_backfill_macro --period 730d --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from sqlalchemy import text  # noqa: E402

from analytics.db_connection import init_db, new_session  # noqa: E402

DATA_DIR = Path("D:/AURION_DATA/macro")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_history(period: str = "730d", interval: str = "1h") -> dict[str, pd.DataFrame]:
    """Descarga histórico de los 3 símbolos macro y guarda en parquet."""
    out: dict[str, pd.DataFrame] = {}
    for yf_sym, label in [("DX-Y.NYB", "DXY"), ("^TNX", "US10Y"), ("^VIX", "VIX")]:
        print(f"  [{label}] descargando {yf_sym} period={period} interval={interval}...")
        df = yf.Ticker(yf_sym).history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            print(f"  [{label}] EMPTY")
            continue
        # Asegurar tz UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        parquet_path = DATA_DIR / f"{label}_1h.parquet"
        df.to_parquet(parquet_path)
        print(f"  [{label}] guardado {len(df)} barras en {parquet_path}")
        out[label] = df
    return out


def _features_at(macro_data: dict[str, pd.DataFrame], at_utc: datetime) -> dict[str, float]:
    """Calcula 5 macro features para un timestamp dado.

    AURION X 4.3: replica la logica de MacroDataProvider.features() pero
    operando sobre el dataframe historico (no live).
    """
    feats: dict[str, float] = {
        "dxy_return_24h": 0.0,
        "dxy_zscore_5d": 0.0,
        "us10y_change_24h": 0.0,
        "vix_level": 0.40,
        "vix_change_24h": 0.0,
    }
    at_utc = at_utc if at_utc.tzinfo else at_utc.replace(tzinfo=timezone.utc)

    def _last_n_closes_before(df: pd.DataFrame, n: int) -> list[float]:
        subset = df[df.index <= at_utc].tail(n)
        return [float(x) for x in subset["close"].tolist() if not pd.isna(x)]

    dxy = macro_data.get("DXY")
    if dxy is not None and not dxy.empty:
        closes = _last_n_closes_before(dxy, 60)
        if closes:
            price = closes[-1]
            # Prev close = ~24h antes (con interval 1h, indice -25)
            prev_idx = -25 if len(closes) > 24 else (0 if len(closes) > 1 else -1)
            prev = closes[prev_idx]
            if prev > 0:
                feats["dxy_return_24h"] = (price - prev) / prev
            if len(closes) >= 10:
                mean = sum(closes) / len(closes)
                var = sum((x - mean) ** 2 for x in closes) / len(closes)
                std = max(1e-9, var ** 0.5)
                feats["dxy_zscore_5d"] = (price - mean) / std
    us10y = macro_data.get("US10Y")
    if us10y is not None and not us10y.empty:
        closes = _last_n_closes_before(us10y, 30)
        if closes and len(closes) >= 2:
            price = closes[-1]
            prev = closes[-min(25, len(closes))]
            if prev > 0:
                feats["us10y_change_24h"] = (price - prev) / prev
    vix = macro_data.get("VIX")
    if vix is not None and not vix.empty:
        closes = _last_n_closes_before(vix, 30)
        if closes:
            price = closes[-1]
            feats["vix_level"] = price / 50.0
            if len(closes) >= 2:
                prev = closes[-min(25, len(closes))]
                if prev > 0:
                    feats["vix_change_24h"] = (price - prev) / prev
    return feats


def backfill_trades(macro_data: dict[str, pd.DataFrame], symbol: str = "XAUUSD",
                    dry_run: bool = False) -> int:
    """Enriquece features_json de trades historicos con features macro v3."""
    init_db()
    s = new_session()
    updated = 0
    skipped_no_features = 0
    skipped_already_v3 = 0
    skipped_no_macro = 0
    try:
        rows = s.execute(text(
            "SELECT id::text, open_time, features_json FROM trades "
            "WHERE symbol = :sym AND close_time IS NOT NULL "
            "ORDER BY open_time ASC"
        ), {"sym": symbol}).fetchall()
        print(f"  Trades {symbol}: {len(rows)}")
        for trade_id, open_time, features_json in rows:
            if not isinstance(features_json, dict):
                skipped_no_features += 1
                continue
            if features_json.get("macro_v3"):
                skipped_already_v3 += 1
                continue
            ot = open_time
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=timezone.utc)
            macro_feats = _features_at(macro_data, ot)
            # Sanity check: si todas las macro son zero default, los datos historicos
            # no cubren ese timestamp - skip para no contaminar.
            if (macro_feats["dxy_return_24h"] == 0.0
                and macro_feats["dxy_zscore_5d"] == 0.0
                and macro_feats["us10y_change_24h"] == 0.0
                and macro_feats["vix_level"] == 0.40
                and macro_feats["vix_change_24h"] == 0.0):
                skipped_no_macro += 1
                continue
            new_features = dict(features_json)
            new_features.update(macro_feats)
            new_features["macro_v3"] = True
            new_features["feature_schema_version"] = 3
            if dry_run:
                updated += 1
                if updated <= 3:
                    print(f"    DRY {trade_id[:8]} @ {ot}: {macro_feats}")
                continue
            s.execute(
                text("UPDATE trades SET features_json = CAST(:fj AS jsonb) "
                     "WHERE id = CAST(:tid AS uuid)"),
                {"fj": json.dumps(new_features), "tid": trade_id},
            )
            updated += 1
            if updated % 50 == 0:
                s.commit()
                print(f"    +{updated} trades enriched...")
        if not dry_run:
            s.commit()
        print(f"  Resultado: enriched={updated} skipped_no_macro={skipped_no_macro} "
              f"already_v3={skipped_already_v3} no_features={skipped_no_features}")
        return updated
    finally:
        s.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="730d", help="Periodo yfinance (730d = 2 anos)")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true",
                          help="Usar parquets existentes en lugar de redescargar")
    args = parser.parse_args()

    print("=" * 70)
    print(" FETCH MACRO + BACKFILL TRADES (DXY/US10Y/VIX)")
    print("=" * 70)

    if args.skip_fetch:
        print("[1/2] Cargando parquets existentes...")
        macro_data: dict[str, pd.DataFrame] = {}
        for label in ("DXY", "US10Y", "VIX"):
            p = DATA_DIR / f"{label}_1h.parquet"
            if p.exists():
                macro_data[label] = pd.read_parquet(p)
                print(f"  [{label}] cargado {len(macro_data[label])} bars desde {p}")
    else:
        print(f"[1/2] Fetch yfinance period={args.period} interval={args.interval}...")
        macro_data = fetch_history(period=args.period, interval=args.interval)

    if not macro_data:
        print("  ERROR: no se obtuvieron datos macro. Abortando.")
        return

    print(f"\n[2/2] Backfill features_json para trades {args.symbol}...")
    if args.dry_run:
        print("  (DRY RUN — sin escribir en DB)")
    n = backfill_trades(macro_data, symbol=args.symbol, dry_run=args.dry_run)
    print(f"\nOK: {n} trades enriched con macro features v3")
    print("=" * 70)


if __name__ == "__main__":
    main()

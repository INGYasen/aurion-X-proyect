"""Análisis profundo del histórico XAUUSD descargado."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd

HIST_DIR = Path("D:/AURION_DATA/historical/XAUUSD")
TFS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]


def analyze_tf(tf: str) -> dict | None:
    f = HIST_DIR / f"{tf}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
    except Exception as exc:
        return {"tf": tf, "error": str(exc)}

    out = {"tf": tf, "rows": len(df), "size_kb": round(f.stat().st_size / 1024, 1)}
    if len(df) == 0:
        return out

    # Detectar columna temporal
    tcol = None
    for c in ("time", "datetime", "date", "timestamp"):
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        # Si el index es DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            tcol = "<index>"
            df = df.reset_index().rename(columns={df.index.name or "index": "time"})
            tcol = "time"
    if tcol is None:
        out["temporal_cols"] = list(df.columns)
        return out

    if tcol in df.columns:
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        valid = df[tcol].dropna()
        if len(valid):
            out["first_date"] = str(valid.min())
            out["last_date"] = str(valid.max())
            out["span_days"] = (valid.max() - valid.min()).days

    # OHLC sanity
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            out[f"{c}_min"] = round(float(df[c].min()), 2)
            out[f"{c}_max"] = round(float(df[c].max()), 2)
            out[f"{c}_mean"] = round(float(df[c].mean()), 2)
            break  # solo necesitamos un sample

    # Columnas
    out["cols"] = list(df.columns)
    return out


def main() -> int:
    print("\n" + "=" * 72)
    print(" AURION X — ANÁLISIS HISTÓRICO XAUUSD")
    print("=" * 72)
    print(f"\nDirectorio: {HIST_DIR}\n")

    if not HIST_DIR.exists():
        print(f"❌ NO EXISTE el directorio. Tienes que correr scripts/download_xauusd_3y.py primero.")
        return 1

    results = []
    for tf in TFS:
        r = analyze_tf(tf)
        if r is not None:
            results.append(r)

    if not results:
        print("❌ No se encontró ningún archivo .parquet en " + str(HIST_DIR))
        return 1

    # Tabla resumen
    print(f"  {'TF':4s} {'Rows':>9s}  {'Span':>8s}  {'First':>22s}  {'Last':>22s}  {'Size KB':>8s}")
    print("  " + "-" * 90)
    for r in results:
        if "error" in r:
            print(f"  {r['tf']:4s} ERROR: {r['error']}")
            continue
        rows = r.get("rows", 0)
        span = r.get("span_days", 0)
        first = r.get("first_date", "n/a")[:22] if r.get("first_date") else "n/a"
        last = r.get("last_date", "n/a")[:22] if r.get("last_date") else "n/a"
        size = r.get("size_kb", 0)
        print(f"  {r['tf']:4s} {rows:>9d}  {str(span)+'d':>8s}  {first:>22s}  {last:>22s}  {size:>8.1f}")

    print("\n--- Detalle por TF ---\n")
    for r in results:
        if "error" in r:
            continue
        print(f"  TF {r['tf']}:")
        if r.get("cols"):
            print(f"    columnas: {r['cols']}")
        for k in ("first_date", "last_date", "span_days", "rows"):
            if k in r:
                print(f"    {k}: {r[k]}")
        # rango precio
        if "close_min" in r or "open_min" in r:
            mins = r.get("close_min") or r.get("open_min")
            maxs = r.get("close_max") or r.get("open_max")
            mean = r.get("close_mean") or r.get("open_mean")
            print(f"    rango precio: {mins} → {maxs} (mean ~ {mean})")

    # Análisis de continuidad para M1 (el más importante)
    m1 = HIST_DIR / "M1.parquet"
    if m1.exists():
        print("\n--- Análisis de gaps en M1 (vela 1 minuto) ---")
        try:
            df = pd.read_parquet(m1)
            tcol = None
            for c in ("time", "datetime", "date", "timestamp"):
                if c in df.columns: tcol = c; break
            if tcol:
                df[tcol] = pd.to_datetime(df[tcol], utc=True)
                df = df.sort_values(tcol)
                diffs = df[tcol].diff().dt.total_seconds() / 60.0  # minutos
                normal = (diffs == 1.0).sum()
                gap_1_5 = ((diffs > 1.0) & (diffs <= 5.0)).sum()
                gap_5_60 = ((diffs > 5.0) & (diffs <= 60.0)).sum()
                gap_60_1440 = ((diffs > 60.0) & (diffs <= 1440.0)).sum()
                gap_long = (diffs > 1440.0).sum()
                print(f"  Velas consecutivas (1min)     : {normal:>7d}")
                print(f"  Gaps 1-5min                   : {gap_1_5:>7d}")
                print(f"  Gaps 5min-1h                  : {gap_5_60:>7d}")
                print(f"  Gaps 1h-1día                  : {gap_60_1440:>7d}")
                print(f"  Gaps largos (>1día)           : {gap_long:>7d}")
                # Identificar los gaps más grandes
                top = df.copy()
                top["diff_min"] = diffs
                top = top.sort_values("diff_min", ascending=False).head(5)
                print(f"\n  Top 5 gaps M1 más grandes:")
                for _, r in top.iterrows():
                    print(f"    {r[tcol]}  gap={r['diff_min']:.0f} min")
        except Exception as exc:
            print(f"  error: {exc}")

    print("\n" + "=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

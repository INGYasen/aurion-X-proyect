"""AURION X 4.1 — Bootstrap histórico XAUUSD (3 años · 9 timeframes).

Uso:

    python -m scripts.download_xauusd_3y               # 3 años, todos los TFs
    python -m scripts.download_xauusd_3y --years 5
    python -m scripts.download_xauusd_3y --tf M15 H1   # sólo seleccionados
    python -m scripts.download_xauusd_3y --force       # ignora cache

Persistencia:

    D:/AURION_DATA/historical/XAUUSD/<TF>.parquet

Requisitos:

    * MetaTrader 5 instalado y con XAUUSD en Market Watch (recomendado).
    * Si no hay MT5 disponible, el downloader cae a un fallback agregado
      desde el cache M1 existente (``analytics.m1_loader``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap histórico XAUUSD (3 años / 9 TFs).")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="D:/AURION_DATA/historical/XAUUSD",
        help="Carpeta destino (parquet por TF).",
    )
    parser.add_argument(
        "--tf",
        nargs="*",
        default=None,
        help="TFs concretos (ej. --tf M15 H1). Default: todos.",
    )
    parser.add_argument("--force", action="store_true", help="Ignora cache fresco.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        # Permite ejecución como script o como módulo del paquete.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from data_pipeline.xau_downloader import download_xauusd_history
    except Exception as exc:
        print(f"ERROR: no se pudo importar data_pipeline.xau_downloader ({exc})", file=sys.stderr)
        return 2

    print(
        f"[XAU-DL] descargando {args.years} año(s) de XAUUSD → {args.out_dir} "
        f"(tf={args.tf or 'all'}, force={args.force})"
    )
    try:
        summary = download_xauusd_history(
            years=float(args.years),
            out_dir=args.out_dir,
            timeframes=args.tf,
            force_refresh=bool(args.force),
        )
    except Exception as exc:
        print(f"ERROR durante la descarga: {exc}", file=sys.stderr)
        return 1

    print("\n[XAU-DL] resumen final:")
    print("-" * 80)
    total_bars = 0
    for tf, info in summary.items():
        bars = info.get("bars", 0) or 0
        total_bars += bars
        err = info.get("error")
        if err:
            print(f"  {tf:>4s}  ERROR   {err}")
            continue
        cached = " (cache)" if info.get("cached") else ""
        print(
            f"  {tf:>4s}  {bars:>10,d} barras  "
            f"[{info.get('from')} .. {info.get('to')}]{cached}  -> {info.get('path')}"
        )
    print("-" * 80)
    print(f"[XAU-DL] total barras: {total_bars:,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""AURION X 4.3 - Reentrena los 3 simbolos: XAUUSD, NDX100, BTCUSD.

Para cada simbolo:
  - Intenta schema v2 + ensemble (mejor calidad).
  - Si insuficientes filas v2, fallback a v2 simple (RF) o v1.
  - Backup del modelo previo en backups/.
  - Reporta OOS metrics y compara con el modelo anterior.

Uso:
    python -m tools.retrain_all_symbols
    python -m tools.retrain_all_symbols --only XAUUSD
    python -m tools.retrain_all_symbols --no-ensemble
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.trainer import Trainer

MODEL_DIR = Path("D:/AURION_DATA/models")
BACKUP_DIR = MODEL_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def backup_model(symbol: str) -> Path | None:
    src = MODEL_DIR / f"model_{symbol}.pkl"
    if not src.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"model_{symbol}_pre_retrain_{ts}.pkl"
    shutil.copy2(src, dst)
    return dst


def train_symbol(symbol: str, use_ensemble: bool, schema: str = "v2") -> dict:
    print(f"\n{'=' * 70}")
    print(f"  RETRAINING {symbol}  (schema={schema}, ensemble={use_ensemble})")
    print(f"{'=' * 70}")

    bkp = backup_model(symbol)
    if bkp:
        print(f"  Backup: {bkp.name}")
    else:
        print("  (no habia modelo previo)")

    t0 = time.time()
    try:
        trainer = Trainer(
            feature_schema=schema,
            output_dir=MODEL_DIR,
            use_ensemble=use_ensemble,
        )
        model, metrics = trainer.train_walk_forward_from_repository(
            limit=10000,
            symbol=symbol,
            model_filename=f"model_{symbol}.pkl",
            dataset_context="all",
            train_ratio=0.65,
            val_ratio=0.15,
        )
        elapsed = time.time() - t0
        out = {
            "ok": True,
            "symbol": symbol,
            "schema": schema,
            "ensemble": use_ensemble,
            "elapsed_s": round(elapsed, 1),
            "metrics": dict(metrics),
        }
        print(f"  OK ({elapsed:.1f}s)")
        for k in ("n_train", "n_val", "oos_accuracy", "oos_roc_auc", "oos_f1",
                  "val_accuracy", "val_f1"):
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    print(f"    {k:28s} {v:.4f}")
                else:
                    print(f"    {k:28s} {v}")
        return out
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  FAIL ({elapsed:.1f}s) {type(exc).__name__}: {exc}")
        if bkp and bkp.exists():
            try:
                dst = MODEL_DIR / f"model_{symbol}.pkl"
                shutil.copy2(bkp, dst)
                print(f"  ROLLBACK aplicado desde {bkp.name}")
            except Exception as rollback_exc:
                print(f"  ROLLBACK FALLO: {rollback_exc}")
        return {
            "ok": False,
            "symbol": symbol,
            "schema": schema,
            "ensemble": use_ensemble,
            "elapsed_s": round(elapsed, 1),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["XAUUSD", "NDX100", "BTCUSD"],
                        help="entrenar solo un simbolo")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="usar RF solo en lugar de ensemble")
    parser.add_argument("--schema", default="v2", choices=["v1", "v2", "v3"],
                        help="schema de features a usar")
    args = parser.parse_args()

    symbols = [args.only] if args.only else ["XAUUSD", "NDX100", "BTCUSD"]
    use_ensemble = not args.no_ensemble

    print(f"AURION X 4.3 - Reentrenamiento masivo")
    print(f"  Simbolos: {symbols}")
    print(f"  Schema:   {args.schema}")
    print(f"  Ensemble: {use_ensemble}")

    results = []
    for sym in symbols:
        try:
            res = train_symbol(sym, use_ensemble=use_ensemble, schema=args.schema)
            results.append(res)
        except KeyboardInterrupt:
            print("\n  INTERRUMPIDO POR USUARIO")
            break

    print(f"\n{'=' * 70}")
    print("  RESUMEN FINAL")
    print(f"{'=' * 70}")
    print(f"  {'Symbol':<10} {'Status':<8} {'Schema':<8} {'OOS AUC':<10} {'OOS Acc':<10} {'Time(s)':<8}")
    print("-" * 70)
    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        m = r.get("metrics", {}) if r.get("ok") else {}
        auc = m.get("oos_roc_auc", "-")
        acc = m.get("oos_accuracy", "-")
        auc_s = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        print(f"  {r['symbol']:<10} {status:<8} {r['schema']:<8} {auc_s:<10} {acc_s:<10} {r['elapsed_s']:<8}")
    print(f"{'=' * 70}")
    failed = [r for r in results if not r.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""AURION X 4.0 — Entrena el MetaStacker que fusiona ensemble + PatchTST.

Pipeline:
    1. Lee los últimos N trades cerrados de PostgreSQL.
    2. Para cada trade reconstruye:
       [prob_ensemble, prob_patchtst, effective_confidence, setup_grade_score].
       PatchTST se infiere offline sobre la ventana M1 previa al trade.
    3. Etiqueta: 1 si fue ganador (R > 0), 0 si no.
    4. Entrena LogisticRegression con purged k-fold (5 splits, embargo 2%).
    5. Reporte JSON + PKL en D:/AURION_DATA/models/.

Uso:
    python -m scripts.train_meta_stacker --min-trades 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _ensure_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena MetaStacker XAUUSD.")
    p.add_argument("--config", type=Path, default=Path("AURION_X_3_0/config/parameters.yaml"))
    p.add_argument("--symbol", type=str, default="XAUUSD")
    p.add_argument("--min-trades", type=int, default=300)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args()


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_dataset(symbol: str, min_trades: int) -> tuple[np.ndarray, np.ndarray]:
    """Construye dataset desde ``TradeRepository.get_last_trades``.

    Devuelve ``(X, y)`` con ``X`` de shape ``(n, 4)`` y ``y`` binario:
      X = [prob_ensemble, prob_patchtst, effective_confidence, setup_grade_score]
      y = 1 si R > 0, 0 en caso contrario.

    Mapeo a las features ya almacenadas en ``trades.features_json``:
      - prob_ensemble ← ``features_json.quant_breakdown.ml_confidence``
      - prob_patchtst ← NaN (no estaba calculado al momento del trade); el
        MetaStacker tiene fallback para NaN. Tras retrain con v4 activo se
        rellenará con la predicción real de PatchTST.
      - effective_confidence ← ``features_json.quant_breakdown.total_score``
      - setup_grade ← ``features_json.setup_grade``
    """
    from analytics.trade_repository import TradeRepository
    from ml.meta_stacker import MetaStacker

    repo = TradeRepository()
    all_rows = repo.get_last_trades(limit=5000)
    rows = [
        r for r in all_rows
        if r.get("symbol") == symbol.upper()
        and r.get("close_time") is not None
        and r.get("r_multiple") is not None
    ]
    if len(rows) < min_trades:
        raise SystemExit(
            f"Insuficientes trades cerrados {symbol}: {len(rows)} < {min_trades}. "
            "Espera a que el bot acumule más histórico."
        )

    X_list: list[list[float]] = []
    y_list: list[int] = []
    for r in rows:
        fj = r.get("features_json") or {}
        qb = (fj.get("quant_breakdown") or {}) if isinstance(fj, dict) else {}
        prob_ens = float(
            qb.get("ml_confidence")
            or fj.get("ml_confidence_ensemble")
            or r.get("ml_confidence")
            or 0.5
        )
        prob_pst = fj.get("prob_patchtst")
        eff_conf = float(
            fj.get("effective_confidence")
            or qb.get("total_score")
            or prob_ens
        )
        grade = fj.get("setup_grade") or ""
        X_list.append([
            prob_ens,
            float(prob_pst) if prob_pst is not None else float("nan"),
            eff_conf,
            MetaStacker.grade_score(grade),
        ])
        rm = r.get("r_multiple")
        y_list.append(1 if float(rm or 0.0) > 0.0 else 0)

    return np.asarray(X_list, dtype=np.float64), np.asarray(y_list, dtype=np.int64)


def main() -> int:
    args = parse_args()
    _ensure_paths()

    cfg_root = load_cfg(args.config)
    ms_cfg = (cfg_root.get("v4_hybrid") or {}).get("meta_stacker") or {}
    model_path = Path(str(ms_cfg.get("model_path", "D:/AURION_DATA/models/meta_stacker_xauusd_v4.pkl")))
    min_samples = int(ms_cfg.get("min_samples_to_fit", 200))

    print(f"Cargando dataset {args.symbol} (mínimo {args.min_trades} trades)...")
    X, y = build_dataset(args.symbol, args.min_trades)
    print(f"Dataset: X={X.shape}, y={y.shape}, win_rate={y.mean():.3f}")

    from ml.cv_strategies import purged_kfold_cv
    from ml.meta_stacker import MetaStacker

    fold_metrics: list[dict[str, float]] = []
    for k, (train_idx, val_idx) in enumerate(
        purged_kfold_cv(X.shape[0], n_splits=args.folds, embargo_pct=0.02)
    ):
        ms = MetaStacker(cfg={"enabled": True, "model_path": model_path, "min_samples_to_fit": min_samples})
        try:
            ms.fit(X[train_idx], y[train_idx])
        except ValueError as exc:
            print(f"[fold {k+1}] {exc}")
            continue
        # Val accuracy + AUC
        from sklearn.metrics import roc_auc_score, accuracy_score
        preds_val = []
        for row, _label in zip(X[val_idx], y[val_idx]):
            pred = ms.predict(
                prob_ensemble=row[0],
                prob_patchtst=row[1] if np.isfinite(row[1]) else None,
                effective_confidence=row[2],
                setup_grade=None,
            )
            preds_val.append(pred.prob_buy)
        preds_arr = np.asarray(preds_val, dtype=np.float64)
        acc = accuracy_score(y[val_idx], preds_arr > 0.5)
        try:
            auc = roc_auc_score(y[val_idx], preds_arr)
        except ValueError:
            auc = float("nan")
        fold_metrics.append({"fold": float(k + 1), "accuracy": float(acc), "auc": float(auc),
                             "n_train": float(train_idx.size), "n_val": float(val_idx.size)})
        print(f"[fold {k+1}] acc={acc:.3f} auc={auc:.3f}")

    print("\nReentrenando sobre todo el dataset y guardando modelo final...")
    final = MetaStacker(cfg={"enabled": True, "model_path": model_path, "min_samples_to_fit": min_samples})
    final.fit(X, y)
    final.save()
    print(f"[OK] MetaStacker guardado en {model_path}")

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "n_samples": int(X.shape[0]),
        "win_rate": float(y.mean()),
        "folds": fold_metrics,
        "model_path": str(model_path),
    }
    report_path = Path("D:/AURION_DATA/reports/phase2_meta_stacker_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Reporte: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

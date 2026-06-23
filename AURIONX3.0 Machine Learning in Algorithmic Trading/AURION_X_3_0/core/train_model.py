"""CLI para reentrenar el modelo ML desde la base de datos de trades."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

for _pat in (
    r".*sklearn\.utils\.parallel\.delayed.*",
    r".*propagate the scikit-learn configuration.*",
    r".*does not have valid feature names.*",
    r".*X has feature names.*",
):
    warnings.filterwarnings("ignore", message=_pat, category=UserWarning)

import yaml

from analytics.db_connection import init_db
from analytics.logger import configure_root_logger, get_logger
from ml.daily_retrainer import DailyRetrainer
from ml.trainer import Trainer

logger = get_logger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reentrena modelo ML de AURION X")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config" / "parameters.yaml",
        help="Ruta a parameters.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Cantidad máxima de trades cerrados para entrenar",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Usa retrain por lotes y conserva el mejor modelo",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Split temporal train/val/test y métricas OOS",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Filtra trades por símbolo (ej. XAUUSD); vacío = todos",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        default="",
        help="Nombre del PKL de salida (ej. model_XAUUSD.pkl); por defecto el de ml.model_path",
    )
    parser.add_argument(
        "--dataset-context",
        type=str,
        default="",
        choices=["", "all", "range", "momentum"],
        help="Filtra dataset de entrenamiento: range (fades/range), momentum (resto), all",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Entrena ensemble cuantitativo (RF+LR+XGB/LGBM si instalados) con features v2",
    )
    return parser.parse_args()


def _resolve_paths(config_path: Path) -> tuple[Path, Path]:
    cfg_file = config_path if config_path.is_absolute() else (_PROJECT_ROOT / config_path).resolve()
    with cfg_file.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ml_cfg = cfg.get("ml", {}) if isinstance(cfg, dict) else {}
    model_path_raw = str(ml_cfg.get("model_path", "D:/AURION_DATA/models/model.pkl")).strip()
    model_path = Path(model_path_raw)
    return cfg_file, model_path


def main() -> None:
    configure_root_logger()
    args = _parse_args()
    cfg_file, model_path = _resolve_paths(args.config)
    with cfg_file.open(encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    ml_cfg = full_cfg.get("ml", {}) if isinstance(full_cfg, dict) else {}
    wf = bool(ml_cfg.get("walk_forward_enabled", False)) or args.walk_forward
    wf_min = max(50, int(ml_cfg.get("walk_forward_min_samples", 60)))
    wf_tr = float(ml_cfg.get("walk_forward_train_ratio", 0.65))
    wf_va = float(ml_cfg.get("walk_forward_val_ratio", 0.15))

    init_db()
    trainer = Trainer.from_ml_config(
        ml_cfg if isinstance(ml_cfg, dict) else {},
        output_dir=model_path.parent,
    )
    sym = str(args.symbol).strip().upper() or None
    dctx = str(args.dataset_context).strip().lower() or None
    if dctx == "all":
        dctx = None
    out_name = str(args.model_filename).strip() or model_path.name
    logger.info(
        "Reentrenando modelo desde config=%s -> model=%s symbol=%s dataset_context=%s",
        cfg_file,
        model_path.parent / out_name,
        sym or "ALL",
        dctx or trainer._dataset_context,
    )
    quant_cfg = full_cfg.get("quant_ml", {}) if isinstance(full_cfg, dict) else {}
    use_ensemble = args.ensemble or bool(quant_cfg.get("use_ensemble", False))
    try:
        if use_ensemble:
            from analytics.trade_repository import TradeRepository

            dr = DailyRetrainer(
                repository=TradeRepository(),
                models_dir=model_path.parent,
                ml_cfg={**(ml_cfg if isinstance(ml_cfg, dict) else {}), "feature_schema": "v2"},
                use_ensemble=True,
                feature_schema="v2",
                asset_profiles=full_cfg.get("asset_profiles") if isinstance(full_cfg, dict) else None,
                config_symbols=full_cfg.get("symbols") if isinstance(full_cfg, dict) else None,
            )
            metrics = dr.run(
                reason="manual",
                symbol=sym,
                limit=max(100, args.limit),
            )
            if metrics is None:
                raise ValueError("Ensemble manual no produjo modelo")
        elif wf:
            _, metrics = trainer.train_walk_forward_from_repository(
                limit=max(100, args.limit),
                symbol=sym,
                model_filename=out_name,
                train_ratio=wf_tr,
                val_ratio=wf_va,
                min_samples=wf_min,
                dataset_context=dctx,
            )
        elif args.batch:
            _, metrics = trainer.retrain_in_batches(
                total_limit=max(1000, args.limit),
                symbol=sym,
                model_filename=out_name,
                dataset_context=dctx,
            )
        else:
            _, metrics = trainer.retrain_manual(
                limit=max(100, args.limit),
                symbol=sym,
                model_filename=out_name,
                dataset_context=dctx,
            )
        logger.info("Entrenamiento finalizado. métricas=%s", json.dumps(metrics, ensure_ascii=True))
    except ValueError as exc:
        logger.warning("No se pudo entrenar todavía: %s", exc)
        logger.warning("Acción recomendada: acumular más trades cerrados y reintentar.")


if __name__ == "__main__":
    main()

"""Reentrena modelos ML por símbolo con esquema v2 (25 features) + ensemble.

Resuelve el warning ``Predictor: len(features)=25 != modelo n_features=10``
forzando el pipeline completo:

    feature_schema = "v2"     -> 25 features (FEATURE_ORDER_V2)
    use_ensemble   = True     -> RF + LR + (XGB/LGBM si están instalados)
    walk_forward   = True     -> métricas OOS reales

Por cada símbolo en ``DEFAULT_SYMBOLS`` o en ``asset_profiles`` del YAML:
    1. Imprime cuántos trades cerrados hay disponibles.
    2. Llama a ``DailyRetrainer.run(symbol=..., reason="manual_v2")``.
    3. Promueve a ``model_<SYMBOL>.pkl`` solo si supera el filtro OOS interno.

Uso (desde la raíz del proyecto ``AURION_X_3_0``):

    python -m scripts.retrain_v2_all_symbols
    python -m scripts.retrain_v2_all_symbols --limit 8000
    python -m scripts.retrain_v2_all_symbols --symbols XAUUSD,NDX100
    python -m scripts.retrain_v2_all_symbols --min-trades 12
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

# Silencia warnings ruidosos de sklearn/joblib en Py3.13+ y ConvergenceWarning con datasets pequeños.
for _pat in (
    r".*sklearn\.utils\.parallel\.delayed.*",
    r".*propagate the scikit-learn configuration.*",
    r".*Number of distinct clusters.*",
):
    warnings.filterwarnings("ignore", message=_pat)

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analytics.db_connection import init_db  # noqa: E402
from analytics.logger import configure_root_logger, get_logger  # noqa: E402
from analytics.trade_repository import TradeRepository  # noqa: E402
from ml.daily_retrainer import DailyRetrainer  # noqa: E402
from ml.symbol_model_registry import DEFAULT_SYMBOLS  # noqa: E402

logger = get_logger(__name__)


def _load_config(cfg_path: Path) -> dict[str, Any]:
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_symbols(cfg: dict[str, Any], requested: str) -> list[str]:
    if requested.strip():
        return [s.strip().upper() for s in requested.split(",") if s.strip()]
    profiles = cfg.get("asset_profiles") or {}
    profile_syms = [str(k).upper() for k in profiles] if isinstance(profiles, dict) else []
    out: list[str] = []
    seen: set[str] = set()
    for sym in [*profile_syms, *DEFAULT_SYMBOLS]:
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _format_metrics(metrics: dict[str, float] | None) -> str:
    if not metrics:
        return "metrics=None"
    keys = (
        "accuracy",
        "oos_accuracy",
        "winrate",
        "sharpe",
        "expectancy",
        "profit_factor",
        "drawdown",
        "promoted",
    )
    parts: list[str] = []
    for k in keys:
        v = metrics.get(k)
        if v is None:
            continue
        try:
            parts.append(f"{k}={float(v):.4f}")
        except (TypeError, ValueError):
            parts.append(f"{k}={v}")
    if not parts:
        return json.dumps(metrics, default=str)[:160]
    return " ".join(parts)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reentrena modelos ML v2 (25 features + ensemble) por símbolo.")
    p.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config" / "parameters.yaml",
        help="Ruta a parameters.yaml",
    )
    p.add_argument("--limit", type=int, default=5000, help="Máx. trades por símbolo (default 5000).")
    p.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Lista separada por coma; vacío = todos los configurados.",
    )
    p.add_argument(
        "--min-trades",
        type=int,
        default=12,
        help="Mínimo de trades cerrados para intentar reentrenar un símbolo.",
    )
    return p.parse_args()


def main() -> int:
    configure_root_logger()
    args = _parse_args()
    cfg_file = args.config if args.config.is_absolute() else (_PROJECT_ROOT / args.config).resolve()
    if not cfg_file.is_file():
        print(f"[ERROR] Config no encontrada: {cfg_file}")
        return 2

    cfg = _load_config(cfg_file)
    ml_cfg = cfg.get("ml") if isinstance(cfg.get("ml"), dict) else {}
    quant_cfg = cfg.get("quant_ml") if isinstance(cfg.get("quant_ml"), dict) else {}

    model_path = Path(str(ml_cfg.get("model_path", "D:/AURION_DATA/models/model.pkl"))).expanduser()
    models_dir = Path(str(quant_cfg.get("models_dir", model_path.parent))).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    forced_ml_cfg = {
        **ml_cfg,
        "feature_schema": "v2",
        "retrain_min_symbol_trades": int(args.min_trades),
    }

    init_db()
    repo = TradeRepository()

    dr = DailyRetrainer(
        repository=repo,
        models_dir=models_dir,
        ml_cfg=forced_ml_cfg,
        use_ensemble=True,
        feature_schema="v2",
        asset_profiles=cfg.get("asset_profiles") if isinstance(cfg, dict) else None,
        config_symbols=cfg.get("symbols") if isinstance(cfg, dict) else None,
    )

    symbols = _resolve_symbols(cfg, args.symbols)
    print("=" * 78)
    print(f"AURION X 3.0 — Reentrenamiento ML v2 (25 features + ensemble)")
    print(f"models_dir   : {models_dir}")
    print(f"feature_schema: v2  |  use_ensemble: True  |  walk_forward: True")
    print(f"symbols      : {', '.join(symbols)}")
    print(f"limit/sym    : {args.limit}   |   min_trades/sym: {args.min_trades}")
    print("=" * 78)

    summary: list[tuple[str, str]] = []
    for sym in symbols:
        try:
            n_closed = repo.count_closed_trades(sym)
        except Exception as exc:  # pragma: no cover - defensivo
            n_closed = 0
            print(f"[{sym}] WARN: no se pudo contar trades ({exc!r})")

        print()
        print(f"[{sym}] trades cerrados disponibles = {n_closed}")
        if n_closed < args.min_trades:
            print(f"[{sym}] OMITIDO: insuficientes (<{args.min_trades}). Acumula más histórico real.")
            summary.append((sym, f"skipped (n={n_closed})"))
            continue

        try:
            metrics = dr.run(reason="manual_v2", symbol=sym, limit=int(args.limit))
        except ValueError as exc:
            print(f"[{sym}] FAIL dataset: {exc}")
            summary.append((sym, f"fail: {exc}"))
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[{sym}] ERROR: {exc!r}")
            summary.append((sym, f"error: {exc!r}"))
            continue

        if not metrics:
            print(f"[{sym}] sin métricas (no se promovió candidato)")
            summary.append((sym, "no_metrics"))
            continue

        line = _format_metrics(metrics)
        promoted = float(metrics.get("promoted", 0.0)) >= 1.0
        tag = "PROMOTED" if promoted else "candidate_kept"
        print(f"[{sym}] {tag}: {line}")
        summary.append((sym, f"{tag}: {line}"))

    print()
    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)
    for sym, info in summary:
        print(f"  {sym:<10s} {info}")
    print("=" * 78)
    print("Modelos desplegados en:")
    for sym, _ in summary:
        print(f"  {models_dir / f'model_{sym}.pkl'}")
    print("Reinicia el bot para que el Predictor cargue los modelos v2 (25 features).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

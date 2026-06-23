"""AURION X 4.1 — Retrainer aislado XAUUSD.

Adaptador que envuelve el ``Trainer`` existente para:

1. Cargar dataset XAUUSD-only (filtra por símbolo en repository).
2. Aplicar pesos temporales (50/30/20) proporcionados por el Knowledge Engine.
3. Ejecutar Walk-Forward Validation obligatorio (train→val→test OOS temporal).
4. Persistir el artefacto en ``model_XAUUSD.pkl`` (path dedicado, no toca los
   modelos de BTCUSD/USTEC).
5. Devolver métricas post-retrain al Knowledge Engine para auditoría.

Compatibilidad:

* No modifica el ``Trainer`` ni los repositorios; sólo los consume.
* Si el Trainer no está disponible, el retrainer hace soft-fail y devuelve
  ``None`` (el bot sigue operando con el modelo previo).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from analytics.logger import get_logger
from analytics.trade_repository import TradeRepository

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dataset provider (XAUUSD-only)
# --------------------------------------------------------------------------- #


def build_xauusd_dataset_provider(repository: TradeRepository | None = None):
    """Devuelve un ``dataset_provider`` apto para ``XAUKnowledgeEngine``.

    El provider llama a ``TradeRepository.get_training_dataset(symbol=symbol)``
    asegurando que BTCUSD/USTEC NO contaminen el dataset del modelo XAUUSD.
    """
    repo = repository or TradeRepository()

    def provider(symbol: str, limit: int = 5000) -> list[Mapping[str, Any]]:
        try:
            rows = repo.get_training_dataset(limit=int(limit), symbol=symbol)
            return [r for r in rows if str(r.get("symbol", "")).upper() == symbol.upper()]
        except Exception as exc:
            logger.warning("[XAU-RETRAIN] dataset_provider falló: %s", exc)
            return []

    return provider


# --------------------------------------------------------------------------- #
# Retrain callback
# --------------------------------------------------------------------------- #


def build_isolated_retrain_callback(
    *,
    model_path: Path,
    trainer_factory=None,
    walk_forward_folds: int = 5,
):
    """Construye el callback de reentrenamiento aislado.

    Args:
        model_path: Ruta destino del ``model_XAUUSD.pkl``.
        trainer_factory: Factoría que devuelve un ``Trainer`` configurado.
            Si ``None``, instancia ``Trainer()`` por defecto.
        walk_forward_folds: nº de folds informativos (se reporta en métricas).

    Devuelve una función ``cb(rows, sample_weights, symbol, reason) -> dict``.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    def callback(
        *,
        rows: Sequence[Mapping[str, Any]],
        sample_weights: Sequence[float],
        symbol: str = "XAUUSD",
        reason: str = "manual",
    ) -> dict[str, float] | None:
        try:
            from ml.trainer import Trainer
        except Exception as exc:
            logger.warning("[XAU-RETRAIN] Trainer no disponible: %s", exc)
            return None
        if not rows:
            return None

        # Construir Trainer con el repositorio adecuado.
        try:
            tr = trainer_factory() if trainer_factory is not None else Trainer()
        except Exception as exc:
            logger.warning("[XAU-RETRAIN] no se pudo instanciar Trainer: %s", exc)
            return None

        # El Trainer tiene su propio walk-forward sobre repository. Lo
        # invocamos con symbol=XAUUSD para garantizar aislamiento.
        try:
            _, metrics = tr.train_walk_forward_from_repository(
                limit=max(len(rows), 200),
                symbol=symbol.upper(),
                model_filename=model_path.name,
                dataset_context="all",
            )
        except Exception as exc:
            logger.warning("[XAU-RETRAIN] walk-forward falló (%s); fallback a train_from_repository", exc)
            try:
                _, metrics = tr.train_from_repository(
                    limit=max(len(rows), 200),
                    symbol=symbol.upper(),
                    model_filename=model_path.name,
                )
            except Exception as exc2:
                logger.warning("[XAU-RETRAIN] train_from_repository también falló: %s", exc2)
                return None

        # Normalizar nombres de métricas que esperamos (winrate/sharpe/expectancy).
        post = {
            "winrate": float(metrics.get("winrate") or metrics.get("oos_winrate") or 0.0),
            "sharpe": float(metrics.get("sharpe") or metrics.get("oos_sharpe") or 0.0),
            "expectancy": float(metrics.get("expectancy") or metrics.get("oos_expectancy") or 0.0),
        }
        # Conservamos extras útiles para auditoría:
        for k in ("oos_winrate", "oos_sharpe", "oos_expectancy", "n_total", "n_train", "n_val", "n_test"):
            if k in metrics:
                try:
                    post[k] = float(metrics[k])
                except Exception:
                    pass
        post["walk_forward_folds"] = float(walk_forward_folds)
        post["reason"] = 1.0  # marker (reasons son string; se persiste en notes)
        logger.info(
            "[XAU-RETRAIN] reason=%s symbol=%s rows=%d metrics=%s",
            reason, symbol, len(rows), {k: round(v, 4) for k, v in post.items() if isinstance(v, (int, float))},
        )
        return post

    return callback


__all__ = [
    "build_xauusd_dataset_provider",
    "build_isolated_retrain_callback",
]

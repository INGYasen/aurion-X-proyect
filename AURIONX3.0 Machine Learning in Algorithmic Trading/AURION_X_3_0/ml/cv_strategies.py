"""AURION X 4.0 — Estrategias de validación cruzada para series temporales.

Implementa Purged K-Fold CV con embargo (López de Prado, 2018, cap. 7) sin
romper el walk-forward existente en ``ml/trainer.py``.

Diseño:
    - Funciones puras como generadores ``(train_idx, test_idx)``.
    - Sin estado, sin I/O, fácilmente testables.
    - Reproducible (deterministic) — no usa aleatoriedad.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from analytics.logger import get_logger

logger = get_logger(__name__)


def purged_kfold_cv(
    n_samples: int,
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.02,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purged K-Fold con embargo temporal.

    Idea: en series financieras la observación j+1 contiene información derivada
    de j (auto-correlación, etiquetas que solapan ventanas, etc.). El embargo
    elimina las muestras adyacentes al fold de test para evitar leakage.

    Args:
        n_samples: longitud total de la serie.
        n_splits: número de folds (5 por defecto).
        embargo_pct: fracción de ``n_samples`` purgada antes/después del fold de
            test. Valores típicos 0.01–0.05.

    Yields:
        Tuplas ``(train_idx, test_idx)`` con índices contiguos, sin overlap.

    Raises:
        ValueError: si ``n_splits < 2`` o ``n_samples < n_splits * 4``.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits debe ser >= 2 (recibido {n_splits}).")
    if n_samples < n_splits * 4:
        raise ValueError(
            f"n_samples={n_samples} insuficiente para n_splits={n_splits} "
            f"(mínimo {n_splits * 4})."
        )

    fold_size = n_samples // n_splits
    embargo = max(1, int(round(n_samples * embargo_pct))) if embargo_pct > 0 else 0

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_splits - 1 else n_samples

        left_train_end = max(0, test_start - embargo)
        right_train_start = min(n_samples, test_end + embargo)

        left_train = np.arange(0, left_train_end, dtype=np.int64)
        right_train = np.arange(right_train_start, n_samples, dtype=np.int64)
        train_idx = np.concatenate([left_train, right_train])
        test_idx = np.arange(test_start, test_end, dtype=np.int64)

        if train_idx.size == 0:
            logger.warning(
                "purged_kfold_cv: fold %d sin train data (test=%d:%d, embargo=%d).",
                i, test_start, test_end, embargo,
            )
            continue

        yield train_idx, test_idx


def walk_forward_with_embargo(
    n_samples: int,
    *,
    train_ratio: float = 0.65,
    val_ratio: float = 0.15,
    embargo_pct: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward expansivo con embargo (compatible con flujo 3.0).

    Devuelve los índices ``(train, val, test)`` en una sola partición temporal:
        | train ......... | embargo | val ... | embargo | test ... |

    Mantiene la semántica del walk-forward 3.0 pero introduce el embargo de
    López de Prado para reducir leakage.

    Returns:
        Tres arrays de índices np.int64.
    """
    train_end = int(round(n_samples * train_ratio))
    val_end = int(round(n_samples * (train_ratio + val_ratio)))
    embargo = max(0, int(round(n_samples * embargo_pct)))

    train_idx = np.arange(0, max(0, train_end - embargo), dtype=np.int64)
    val_start = min(n_samples, train_end + embargo)
    val_idx_end = max(val_start, val_end - embargo)
    val_idx = np.arange(val_start, val_idx_end, dtype=np.int64)
    test_start = min(n_samples, val_end + embargo)
    test_idx = np.arange(test_start, n_samples, dtype=np.int64)

    return train_idx, val_idx, test_idx


def iter_cv_splits(
    n_samples: int,
    *,
    strategy: str = "walk_forward",
    n_splits: int = 5,
    embargo_pct: float = 0.02,
    train_ratio: float = 0.65,
    val_ratio: float = 0.15,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Dispatcher entre estrategias.

    ``walk_forward``: devuelve UN solo split (train+val unidos como train, test).
    ``purged_kfold``: iterador con ``n_splits`` folds.

    Esta función es el punto de entrada usado desde ``trainer.py`` para no
    duplicar la lógica de dispatch.
    """
    strat = (strategy or "walk_forward").strip().lower()
    if strat == "purged_kfold":
        yield from purged_kfold_cv(n_samples, n_splits=n_splits, embargo_pct=embargo_pct)
    elif strat == "walk_forward":
        tr, va, te = walk_forward_with_embargo(
            n_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            embargo_pct=embargo_pct,
        )
        train_full = np.concatenate([tr, va]) if va.size > 0 else tr
        yield train_full, te
    else:
        raise ValueError(f"strategy desconocida: {strategy!r}")

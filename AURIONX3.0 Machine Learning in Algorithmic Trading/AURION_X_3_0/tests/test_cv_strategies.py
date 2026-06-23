"""AURION X 4.0 — Tests para Purged K-Fold CV (cv_strategies.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.cv_strategies import (
    iter_cv_splits,
    purged_kfold_cv,
    walk_forward_with_embargo,
)


def test_purged_kfold_no_overlap_and_covers_test() -> None:
    n = 1000
    n_splits = 5
    seen_test: set[int] = set()
    fold_count = 0
    for train, test in purged_kfold_cv(n, n_splits=n_splits, embargo_pct=0.0):
        # Sin embargo: train ∪ test debe ser disjunto y juntos cubren n.
        assert np.intersect1d(train, test).size == 0
        seen_test.update(test.tolist())
        fold_count += 1
    assert fold_count == n_splits
    assert seen_test == set(range(n)), "Los folds deben cubrir TODA la serie."


def test_purged_kfold_embargo_creates_gap() -> None:
    n = 1000
    embargo_pct = 0.05  # 50 muestras
    for train, test in purged_kfold_cv(n, n_splits=5, embargo_pct=embargo_pct):
        if train.size == 0 or test.size == 0:
            continue
        # Las muestras del embargo (entre train y test) no deben estar en train.
        test_min, test_max = int(test.min()), int(test.max())
        gap_left = set(range(max(0, test_min - 50), test_min))
        gap_right = set(range(test_max + 1, min(n, test_max + 51)))
        train_set = set(train.tolist())
        assert gap_left.isdisjoint(train_set), "Embargo izquierdo no respetado."
        assert gap_right.isdisjoint(train_set), "Embargo derecho no respetado."


def test_purged_kfold_deterministic() -> None:
    """Misma input → mismo output, sin aleatoriedad."""
    n, n_splits, emb = 800, 4, 0.02
    first = list(purged_kfold_cv(n, n_splits=n_splits, embargo_pct=emb))
    second = list(purged_kfold_cv(n, n_splits=n_splits, embargo_pct=emb))
    assert len(first) == len(second) == n_splits
    for (t1, te1), (t2, te2) in zip(first, second):
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(te1, te2)


def test_purged_kfold_invalid_args() -> None:
    with pytest.raises(ValueError):
        list(purged_kfold_cv(100, n_splits=1))
    with pytest.raises(ValueError):
        list(purged_kfold_cv(5, n_splits=5))  # n_samples < n_splits * 4


def test_walk_forward_partitions_sum_le_total() -> None:
    n = 1000
    tr, va, te = walk_forward_with_embargo(
        n, train_ratio=0.65, val_ratio=0.15, embargo_pct=0.02
    )
    # Las particiones nunca se solapan y respetan orden temporal.
    assert tr.size + va.size + te.size <= n
    assert (tr.max() if tr.size else -1) < (va.min() if va.size else n)
    assert (va.max() if va.size else -1) < (te.min() if te.size else n)


def test_iter_cv_splits_dispatcher() -> None:
    n = 1000
    walks = list(iter_cv_splits(n, strategy="walk_forward"))
    assert len(walks) == 1, "walk_forward devuelve UN split."

    folds = list(iter_cv_splits(n, strategy="purged_kfold", n_splits=5))
    assert len(folds) == 5

    with pytest.raises(ValueError):
        list(iter_cv_splits(n, strategy="random_split"))

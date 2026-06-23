"""Diagnóstico rápido de modelos por símbolo: schema, nombres, predict_proba sample."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    paths = [
        ("GLOBAL", Path("D:/AURION_DATA/models/model.pkl")),
        ("XAUUSD", Path("D:/AURION_DATA/models/model_XAUUSD.pkl")),
        ("NDX100", Path("D:/AURION_DATA/models/model_NDX100.pkl")),
        ("BTCUSD", Path("D:/AURION_DATA/models/model_BTCUSD.pkl")),
    ]
    for sym, p in paths:
        if not p.exists():
            print(f"{sym}: ARCHIVO NO EXISTE en {p}")
            continue
        with p.open("rb") as f:
            payload = pickle.load(f)
        meta = payload.get("meta") or {}
        est = payload["estimator"]
        names = meta.get("feature_names") or []
        n_in = getattr(est, "n_features_in_", "n/a")
        schema = meta.get("feature_schema_version", "?")
        classes = list(getattr(est, "classes_", []))

        n = int(n_in) if isinstance(n_in, int) else 10
        rng = np.random.default_rng(42)
        X = rng.normal(size=(3, n))
        try:
            proba = est.predict_proba(X)
            sample = [tuple(round(float(v), 4) for v in row) for row in proba[:3]]
        except Exception as exc:
            sample = f"ERROR: {exc}"

        print(
            f"{sym}: schema_v={schema} n_in={n_in} classes={classes} "
            f"feature_names_head={names[:3]} proba_sample={sample}"
        )


if __name__ == "__main__":
    main()

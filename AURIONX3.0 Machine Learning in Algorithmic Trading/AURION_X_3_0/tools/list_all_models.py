"""Lista todos los .pkl bajo D:/AURION_DATA/models con n_in y schema."""
from __future__ import annotations

import pickle
from pathlib import Path


def main() -> None:
    base = Path("D:/AURION_DATA/models")
    print(f"Buscando en {base}")
    files = sorted(base.rglob("*.pkl"))
    print(f"Total .pkl: {len(files)}")
    for p in files:
        try:
            with p.open("rb") as f:
                payload = pickle.load(f)
            meta = payload.get("meta") or {}
            est = payload.get("estimator")
            n_in = getattr(est, "n_features_in_", None) if est is not None else None
            schema = meta.get("feature_schema_version", "?")
            kind = payload.get("kind", "?")
            print(f"  {p.relative_to(base)}  n_in={n_in}  schema={schema}  kind={kind}")
        except Exception as exc:
            print(f"  {p.relative_to(base)}  ERROR={exc}")


if __name__ == "__main__":
    main()

"""Diagnostico rapido del modelo XAUUSD activo."""
import pickle
import sys
from pathlib import Path

p = Path(r"D:\AURION_DATA\models\model_XAUUSD.pkl")
if not p.exists():
    print("  NO EXISTE el modelo")
    sys.exit(1)
payload = pickle.load(p.open("rb"))
meta = payload.get("meta", {})
print(f"  Tipo:           {payload.get('kind', '?')}")
print(f"  schema_version: {meta.get('feature_schema_version')}")
print(f"  n_features:     {meta.get('n_features_in')}")
print(f"  estimators:     {meta.get('estimators', [])}")
print(f"  calibrated:     {meta.get('calibrated', False)}")

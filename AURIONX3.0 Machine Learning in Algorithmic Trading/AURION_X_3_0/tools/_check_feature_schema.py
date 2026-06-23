import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session
from ml.quant_features import FEATURE_ORDER_V2, QUANT_EXTENDED_ORDER

init_db()
s = new_session()

V1_KEYS = {"return_1", "return_5", "atr_m1", "dist_ema20", "recent_range",
           "tick_volume_proxy", "hour_sin", "hour_cos", "direction", "drawdown"}
V2_EXTRA = set(QUANT_EXTENDED_ORDER)

q = s.execute(text("""
    SELECT features_json
    FROM trades
    WHERE symbol='XAUUSD' AND close_time IS NOT NULL AND features_json IS NOT NULL
""")).fetchall()
print(f"Total trades con features: {len(q)}")

n_v1_only = 0
n_v2_complete = 0
n_v2_partial = 0
n_synth = 0
sample_keys = set()
for row in q:
    fj = row[0]
    if isinstance(fj, str):
        try: fj = json.loads(fj)
        except: continue
    if not isinstance(fj, dict): continue
    keys = set(fj.keys())
    if "synthetic" in keys: n_synth += 1
    v2_extra_present = V2_EXTRA & keys
    if len(v2_extra_present) == len(V2_EXTRA):
        n_v2_complete += 1
    elif len(v2_extra_present) > 0:
        n_v2_partial += 1
    else:
        n_v1_only += 1
    sample_keys.update(keys)

print(f"\n  Trades con SOLO v1 (10 features): {n_v1_only}")
print(f"  Trades con v2 PARCIAL              : {n_v2_partial}")
print(f"  Trades con v2 COMPLETO (25 features): {n_v2_complete}")
print(f"  Marcados sintéticos                : {n_synth}")

print(f"\n  Total keys distintas vistas: {len(sample_keys)}")
print(f"  v2 EXTRA presentes en algún trade: {sorted(V2_EXTRA & sample_keys)}")

s.close()

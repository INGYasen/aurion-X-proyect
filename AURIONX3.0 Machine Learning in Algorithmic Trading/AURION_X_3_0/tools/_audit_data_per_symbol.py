"""Auditoria de datos por simbolo para decidir schema de reentrenamiento."""
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
try:
    print("=" * 70)
    print(f" {'Symbol':<10} {'Total':>8} {'Closed':>8} {'V2 feats':>10} {'V1 only':>10} {'Recomendacion':>20}")
    print("=" * 70)
    for sym in ("XAUUSD", "NDX100", "BTCUSD"):
        total = s.execute(text("SELECT COUNT(*) FROM trades WHERE symbol = :s"),
                          {"s": sym}).scalar()
        closed = s.execute(text(
            "SELECT COUNT(*) FROM trades WHERE symbol = :s AND close_time IS NOT NULL"
        ), {"s": sym}).scalar()
        # V2: tienen features quant extendidas
        # v2 detection: el trainer detecta v2 si tiene QUANT_EXTENDED_ORDER keys
        from ml.quant_features import QUANT_EXTENDED_ORDER
        sample_keys = list(QUANT_EXTENDED_ORDER)[:3]
        like_clause = " OR ".join([f"features_json::text LIKE '%{k}%'" for k in sample_keys])
        v2 = s.execute(text(
            f"SELECT COUNT(*) FROM trades WHERE symbol = :s AND close_time IS NOT NULL "
            f"AND ({like_clause})"
        ), {"s": sym}).scalar()
        v1_only = closed - v2 if closed else 0
        if v2 >= 200:
            rec = "ENSEMBLE v2"
        elif v2 >= 80:
            rec = "RF v2"
        elif closed >= 80:
            rec = "RF v1"
        else:
            rec = "INSUFICIENTE"
        print(f" {sym:<10} {total:>8} {closed:>8} {v2:>10} {v1_only:>10} {rec:>20}")
    print("=" * 70)

    # Sample real de keys
    print("\n=== Sample keys por simbolo ===")
    for sym in ("XAUUSD", "NDX100", "BTCUSD"):
        row = s.execute(text(
            "SELECT features_json FROM trades WHERE symbol = :s "
            "AND close_time IS NOT NULL "
            "ORDER BY close_time DESC LIMIT 1"
        ), {"s": sym}).fetchone()
        if row and isinstance(row[0], dict):
            keys = sorted(row[0].keys())
            print(f"\n  {sym} ({len(keys)} keys):")
            for i in range(0, len(keys), 6):
                print(f"    {keys[i:i+6]}")
        else:
            print(f"\n  {sym}: sin features_json")
finally:
    s.close()

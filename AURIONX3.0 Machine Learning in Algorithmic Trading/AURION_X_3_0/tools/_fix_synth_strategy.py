"""Cambia los trades sintéticos a strategy='SCALPER' para compatibilidad con
el enum de SQLAlchemy. Mantiene marca distintiva en features_json.tag = 'SMC_SYN'.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
try:
    n_smc = s.execute(text(
        "SELECT COUNT(*) FROM trades WHERE strategy='SMC_SYN'"
    )).fetchone()[0]
    print(f"Trades con strategy='SMC_SYN': {n_smc}")
    if n_smc > 0:
        # Agregar marca en features_json y cambiar strategy
        upd = s.execute(text("""
            UPDATE trades
            SET strategy = 'SCALPER',
                features_json = jsonb_set(
                    features_json::jsonb,
                    '{synthetic}',
                    'true'::jsonb,
                    true
                )
            WHERE strategy='SMC_SYN' AND symbol='XAUUSD'
        """)).rowcount
        s.commit()
        print(f"Actualizados: {upd}")
    n_synth_marked = s.execute(text("""
        SELECT COUNT(*) FROM trades
        WHERE strategy='SCALPER' AND symbol='XAUUSD'
          AND (features_json::jsonb)->>'synthetic' = 'true'
    """)).fetchone()[0]
    print(f"Sintéticos marcados (en SCALPER): {n_synth_marked}")
except Exception as e:
    s.rollback()
    import traceback; traceback.print_exc()
finally:
    s.close()

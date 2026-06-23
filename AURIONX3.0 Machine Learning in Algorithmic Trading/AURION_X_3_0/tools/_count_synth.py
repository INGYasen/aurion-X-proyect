import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
n_smc_syn = s.execute(text("SELECT COUNT(*) FROM trades WHERE strategy='SMC_SYN'")).fetchone()[0]
print(f"Trades con strategy='SMC_SYN' (todavía no migrados): {n_smc_syn}")

n_synth_marked = s.execute(text("""
    SELECT COUNT(*) FROM trades
    WHERE symbol='XAUUSD' AND close_time IS NOT NULL
      AND (features_json::jsonb->>'synthetic')='true'
""")).fetchone()[0]
print(f"Trades sintéticos marcados (synthetic=true): {n_synth_marked}")

n_real = s.execute(text("""
    SELECT COUNT(*) FROM trades
    WHERE symbol='XAUUSD' AND close_time IS NOT NULL
      AND (features_json::jsonb->>'synthetic' IS NULL OR features_json::jsonb->>'synthetic' != 'true')
""")).fetchone()[0]
print(f"Trades reales (no sintéticos): {n_real}")

total = s.execute(text("""
    SELECT COUNT(*) FROM trades
    WHERE symbol='XAUUSD' AND close_time IS NOT NULL
""")).fetchone()[0]
print(f"TOTAL trades cerrados: {total}")

# WR por origen
r = s.execute(text("""
    SELECT
      CASE WHEN (features_json::jsonb->>'synthetic')='true' THEN 'SINT' ELSE 'REAL' END AS origen,
      COUNT(*) AS n,
      AVG(r_multiple)::numeric(10,3) AS avg_r,
      SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END)::float / COUNT(*) AS wr
    FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL
    GROUP BY origen
""")).fetchall()
print(f"\n{'origen':6s} {'n':>5s} {'WR%':>7s} {'avgR':>8s}")
for row in r:
    print(f"{row[0]:6s} {row[1]:>5d} {row[3]*100:>7.2f} {float(row[2]):>8.3f}")

# Si hay SMC_SYN sin migrar, migrar
if n_smc_syn > 0:
    print(f"\nMigrando {n_smc_syn} trades SMC_SYN → SCALPER...")
    s.execute(text("""
        UPDATE trades
        SET strategy = 'SCALPER',
            features_json = jsonb_set(features_json::jsonb, '{synthetic}', 'true'::jsonb, true)
        WHERE strategy='SMC_SYN' AND symbol='XAUUSD'
    """))
    s.commit()
    print("OK")

s.close()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
print("=== Todas las tablas ===")
r = s.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")).fetchall()
for row in r:
    cnt = s.execute(text(f"SELECT COUNT(*) FROM {row[0]}")).fetchone()[0]
    print(f"  {row[0]:40s}  rows={cnt}")
for t in ("trades", "smc_events", "smc_snapshots", "decisions"):
    print(f"\n=== {t} ===")
    try:
        r = s.execute(text(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name='{t}' ORDER BY ordinal_position"
        )).fetchall()
        for row in r:
            print(f"  {row[0]:30s} {row[1]}")
    except Exception as e:
        s.rollback()
        print(f"  ERROR: {e}")

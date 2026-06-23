import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()

rows = s.execute(text("""
    SELECT COUNT(*) FROM trades WHERE symbol='XAUUSD'
""")).scalar()
print(f"Total trades XAUUSD: {rows}")

rows = s.execute(text("""
    SELECT COUNT(*) FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL
""")).scalar()
print(f"  cerrados (close_time NOT NULL): {rows}")

rows = s.execute(text("""
    SELECT COUNT(*) FROM trades WHERE symbol='XAUUSD' AND features_json IS NOT NULL
""")).scalar()
print(f"  con features_json: {rows}")

rows = s.execute(text("""
    SELECT COUNT(*) FROM trades
    WHERE symbol='XAUUSD' AND features_json::text LIKE '%"synthetic": true%'
""")).scalar()
print(f"  sintéticos (synthetic=true): {rows}")

rows = s.execute(text("""
    SELECT COUNT(*) FROM trades
    WHERE symbol='XAUUSD' AND features_json IS NOT NULL
      AND features_json::text LIKE '%feature_schema_version%'
""")).scalar()
print(f"  con feature_schema_version: {rows}")

# Sample
sample = s.execute(text("""
    SELECT id, strategy, side, profit, r_multiple, close_time IS NOT NULL as closed,
           features_json->>'feature_schema_version' as schema
    FROM trades WHERE symbol='XAUUSD'
    ORDER BY id DESC LIMIT 5
""")).fetchall()
print("\nÚltimos 5 trades:")
for r in sample:
    print(f"  id={r[0]} strategy={r[1]} side={r[2]} profit={r[3]} R={r[4]} closed={r[5]} schema={r[6]}")

s.close()

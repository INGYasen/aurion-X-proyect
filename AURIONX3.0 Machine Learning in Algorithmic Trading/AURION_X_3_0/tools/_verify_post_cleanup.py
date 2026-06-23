"""Verificacion final post-cleanup."""
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
try:
    print("Conteo final de trades por simbolo:")
    rows = s.execute(text(
        "SELECT symbol, COUNT(*) FROM trades GROUP BY symbol ORDER BY 2 DESC"
    )).fetchall()
    if not rows:
        print("  (DB vacia)")
    for sym, n in rows:
        print(f"  {sym:<12} {n:>5} trades")
finally:
    s.close()

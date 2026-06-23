"""Diagnóstico del trade R=-19.35: ¿qué pasó con su SL?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
print("=== Distribución de r_multiple en últimos 50 trades cerrados ===")
r = s.execute(text("""
    SELECT r_multiple, open_time, direction, entry_price, sl, exit_price, profit, strategy
    FROM trades WHERE symbol='XAUUSD' AND r_multiple IS NOT NULL
    ORDER BY open_time DESC LIMIT 50
""")).fetchall()
extreme = []
for row in r:
    rmul = float(row[0])
    if abs(rmul) > 3.0:
        risk_pips = abs(float(row[3]) - float(row[4]))
        actual_move = abs(float(row[3]) - float(row[5])) if row[5] else 0
        extreme.append({
            "time": row[1], "dir": row[2], "entry": row[3], "sl": row[4],
            "exit": row[5], "R": rmul, "$": row[6], "risk_pips": risk_pips,
            "actual_move": actual_move, "strat": row[7],
        })

print(f"\n  Trades extremos (|R|>3): {len(extreme)}")
for t in extreme:
    print(f"  {t['time']} {t['dir']} entry={t['entry']} sl={t['sl']} "
          f"exit={t['exit']} R={t['R']} $={t['$']} risk_pips={t['risk_pips']:.2f} "
          f"actual_move={t['actual_move']:.2f} strat={t['strat']}")

print("\n=== Distribución de risk_pips (|entry-sl|) en últimos 100 trades ===")
r = s.execute(text("""
    SELECT (CASE WHEN direction='LONG' THEN entry_price - sl ELSE sl - entry_price END) AS risk_pips
    FROM trades WHERE symbol='XAUUSD' AND r_multiple IS NOT NULL
    ORDER BY open_time DESC LIMIT 100
""")).fetchall()
risks = [float(x[0]) for x in r if x[0] is not None]
if risks:
    import statistics
    print(f"  n={len(risks)} min={min(risks):.2f} mean={statistics.mean(risks):.2f} "
          f"median={statistics.median(risks):.2f} max={max(risks):.2f}")
    sorted_r = sorted(risks)
    for p in (50, 75, 90, 95, 99):
        idx = int(p/100 * len(sorted_r))
        print(f"  p{p}: {sorted_r[min(idx, len(sorted_r)-1)]:.2f}")

s.close()

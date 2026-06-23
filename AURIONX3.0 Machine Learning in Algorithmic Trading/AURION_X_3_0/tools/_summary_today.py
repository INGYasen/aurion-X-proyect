"""Resumen rapido de trades de hoy."""
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
try:
    print("=" * 75)
    print(" TRADES DE HOY (XAUUSD)")
    print("=" * 75)
    rows = s.execute(text(
        "SELECT open_time, close_time, direction, lot, profit, r_multiple, "
        "       close_reason, ml_confidence, features_json->>'setup_type' "
        "FROM trades WHERE symbol='XAUUSD' "
        "AND open_time > NOW() - INTERVAL '6 hours' "
        "ORDER BY open_time DESC"
    )).fetchall()
    print(f"\n  N trades en ultimas 6h: {len(rows)}")
    if rows:
        wins = sum(1 for r in rows if r[4] and float(r[4]) > 0)
        losses = sum(1 for r in rows if r[4] and float(r[4]) < 0)
        bes = sum(1 for r in rows if r[4] and abs(float(r[4])) < 0.01)
        total_pnl = sum(float(r[4] or 0) for r in rows)
        avg_r = sum(float(r[5] or 0) for r in rows) / len(rows) if rows else 0
        wr = (wins / len(rows)) * 100 if rows else 0
        print(f"  WINS:           {wins}")
        print(f"  LOSSES:         {losses}")
        print(f"  BE:             {bes}")
        print(f"  WIN RATE:       {wr:.1f}%")
        print(f"  PnL TOTAL:      ${total_pnl:+.2f}")
        print(f"  AvgR:           {avg_r:+.2f}")
        print()
        print(f"  {'open':<19} {'dir':<5} {'lot':>5} {'pnl':>8} {'R':>7} {'razon':<14} {'mlcf':>6} {'setup':<22}")
        for r in rows:
            ot, ct, dr, lot, pnl, rm, reason, mlc, setup = r
            pnl_s = f"${float(pnl or 0):+.2f}"
            rm_s = f"{float(rm or 0):+.2f}"
            mlc_s = f"{float(mlc or 0):.2f}" if mlc else "  -"
            reason_s = (reason or "-")[:14]
            setup_s = (setup or "-")[:22]
            print(f"  {str(ot)[:19]:<19} {dr:<5} {float(lot or 0):>5.3f} {pnl_s:>8} {rm_s:>7} {reason_s:<14} {mlc_s:>6} {setup_s:<22}")
finally:
    s.close()

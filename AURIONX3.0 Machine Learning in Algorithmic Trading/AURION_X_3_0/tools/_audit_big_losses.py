"""Investigar SL hits con R=-14.40 idéntico repetido del 2026-06-09."""
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
try:
    rows = s.execute(text(
        "SELECT id, open_time, close_time, direction, lot, entry_price, exit_price, "
        "       sl, tp, profit, r_multiple, ml_confidence, features_json->>'setup_type' AS setup, "
        "       close_reason "
        "FROM trades WHERE symbol='XAUUSD' "
        "AND r_multiple < -10 "
        "AND close_time > NOW() - INTERVAL '7 days' "
        "ORDER BY close_time DESC"
    )).fetchall()
    print("=" * 90)
    print(f" {len(rows)} TRADES con R<-10 en ultimos 7 dias")
    print("=" * 90)
    print(f"  {'open':<19} {'dir':<5} {'lot':>5} {'entry':>9} {'exit':>9} {'sl':>9} {'sl_dist':>8} {'exit_dist':>9} {'ratio':>6} {'R':>7}")
    for r in rows:
        (tid, ot, ct, dr, lot, ep, xp, sl, tp, pnl, rm, mlc, setup, reason) = r
        ep_f = float(ep or 0); xp_f = float(xp or 0); sl_f = float(sl or 0)
        if dr == "LONG":
            sl_dist = ep_f - sl_f
            exit_dist = ep_f - xp_f  # positivo si exit < entry (perdida)
        else:
            sl_dist = sl_f - ep_f
            exit_dist = xp_f - ep_f  # positivo si exit > entry (perdida)
        ratio = exit_dist / sl_dist if sl_dist > 0 else 0
        print(f"  {str(ot)[:19]:<19} {dr:<5} {float(lot or 0):>5.2f} "
              f"{ep_f:>9.3f} {xp_f:>9.3f} {sl_f:>9.3f} {sl_dist:>8.3f} {exit_dist:>9.3f} "
              f"{ratio:>6.2f} {float(rm or 0):>+7.2f}")
        print(f"      setup={setup} reason={reason} ml_conf={float(mlc or 0):.2f}")

    print(f"\n  >> Si ratio > 1.5 = SL ejecutado MUY lejos del SL nominal (slippage o bug)")
    print(f"  >> Si ratio == 1.0 = SL exacto (calculo OK)")
    print(f"  >> R=-14.40 exacto repetido sugiere SL fijo mal calibrado para esos setups")
finally:
    s.close()

"""Auditoria de perdidas: que paso en los ultimos 3 dias y por que."""
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
try:
    print("=" * 75)
    print(" AUDITORIA DE PERDIDAS RECIENTES — XAUUSD")
    print("=" * 75)

    # Resumen ultimos 3 dias
    print("\n[1] RESUMEN GLOBAL (ultimos 3 dias)")
    row = s.execute(text(
        "SELECT COUNT(*), "
        "       SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) AS wins, "
        "       SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) AS losses, "
        "       AVG(r_multiple), "
        "       SUM(profit), "
        "       MIN(profit), "
        "       MAX(profit) "
        "FROM trades WHERE symbol='XAUUSD' "
        "AND close_time > NOW() - INTERVAL '3 days'"
    )).fetchone()
    if row and row[0]:
        n, wins, losses, avgr, pnl, min_p, max_p = row
        wr = (wins or 0) / n if n else 0
        print(f"  N trades:        {n}")
        print(f"  Wins:            {wins or 0}  ({wr*100:.1f}%)")
        print(f"  Losses:          {losses or 0}")
        print(f"  AvgR:            {float(avgr or 0):.3f}")
        print(f"  PnL total:       ${float(pnl or 0):.2f}")
        print(f"  Peor trade:      ${float(min_p or 0):.2f}")
        print(f"  Mejor trade:     ${float(max_p or 0):.2f}")
    else:
        print("  No hay trades en los ultimos 3 dias.")
        # Buscar fechas reales
        last = s.execute(text(
            "SELECT MAX(close_time) FROM trades WHERE symbol='XAUUSD'"
        )).scalar()
        print(f"  Ultimo trade XAUUSD: {last}")
        # Pero quizas habia trades mas viejos relevantes:
        last7 = s.execute(text(
            "SELECT COUNT(*), SUM(CASE WHEN profit>0 THEN 1 ELSE 0 END), AVG(r_multiple), SUM(profit) "
            "FROM trades WHERE symbol='XAUUSD' "
            "AND close_time > NOW() - INTERVAL '7 days'"
        )).fetchone()
        if last7 and last7[0]:
            wr = last7[1] / last7[0] if last7[0] else 0
            print(f"\n  Ultimos 7 dias: n={last7[0]} wr={wr*100:.1f}% avgR={float(last7[2] or 0):.3f} pnl=${float(last7[3] or 0):.2f}")

    # Trades cerrados (cualquier rango, los mas recientes)
    print("\n[2] ULTIMOS 25 TRADES (cronologico inverso)")
    rows = s.execute(text(
        "SELECT close_time, direction, profit, r_multiple, close_reason, ml_confidence, "
        "       features_json->>'setup_type' AS setup, "
        "       features_json->>'volatility_regime' AS volreg "
        "FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL "
        "ORDER BY close_time DESC LIMIT 25"
    )).fetchall()
    print(f"  {'fecha':<20} {'dir':<5} {'pnl':>8} {'R':>6} {'razon':<14} {'mlcf':>6} {'setup':<22} {'vol':<8}")
    for r in rows:
        ct, dr, pnl, rm, reason, mlc, setup, vol = r
        pnl_s = f"${float(pnl or 0):+.2f}"
        rm_s = f"{float(rm or 0):+.2f}"
        mlc_s = f"{float(mlc or 0):.2f}" if mlc else "  -"
        reason_s = (reason or "-")[:14]
        setup_s = (setup or "-")[:22]
        vol_s = (vol or "-")[:8]
        print(f"  {str(ct)[:19]:<20} {dr:<5} {pnl_s:>8} {rm_s:>6} {reason_s:<14} {mlc_s:>6} {setup_s:<22} {vol_s:<8}")

    # Breakdown por razon de cierre
    print("\n[3] BREAKDOWN POR RAZON DE CIERRE (ultimos 30 dias)")
    rows = s.execute(text(
        "SELECT close_reason, COUNT(*), "
        "       AVG(r_multiple)::numeric(10,3), "
        "       SUM(profit)::numeric(10,2), "
        "       SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) AS wins "
        "FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL "
        "AND close_time > NOW() - INTERVAL '30 days' "
        "GROUP BY close_reason ORDER BY COUNT(*) DESC"
    )).fetchall()
    if rows:
        print(f"  {'razon':<18} {'N':>4} {'wr':>6} {'avgR':>7} {'pnl':>10}")
        for r in rows:
            reason, n, avgr, pnl, wins = r
            wr = (wins or 0) / n if n else 0
            print(f"  {(reason or '-')[:18]:<18} {n:>4} {wr*100:>5.1f}% {float(avgr or 0):>+7.3f} ${float(pnl or 0):>+9.2f}")

    # Breakdown por setup
    print("\n[4] BREAKDOWN POR SETUP (ultimos 30 dias)")
    rows = s.execute(text(
        "SELECT features_json->>'setup_type' AS setup, COUNT(*), "
        "       AVG(r_multiple)::numeric(10,3), "
        "       SUM(profit)::numeric(10,2), "
        "       SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) AS wins "
        "FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL "
        "AND close_time > NOW() - INTERVAL '30 days' "
        "GROUP BY features_json->>'setup_type' ORDER BY SUM(profit) ASC LIMIT 15"
    )).fetchall()
    if rows:
        print(f"  {'setup':<26} {'N':>4} {'wr':>6} {'avgR':>7} {'pnl':>10}")
        for r in rows:
            setup, n, avgr, pnl, wins = r
            wr = (wins or 0) / n if n else 0
            mark = "*" if float(pnl or 0) < -20 else " "
            print(f" {mark}{(setup or '-')[:26]:<26} {n:>4} {wr*100:>5.1f}% {float(avgr or 0):>+7.3f} ${float(pnl or 0):>+9.2f}")

    # Breakdown por sesion
    print("\n[5] BREAKDOWN POR HORA UTC (ultimos 30 dias)")
    rows = s.execute(text(
        "SELECT EXTRACT(HOUR FROM open_time) AS h, COUNT(*), "
        "       AVG(r_multiple)::numeric(10,3), "
        "       SUM(profit)::numeric(10,2) "
        "FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL "
        "AND close_time > NOW() - INTERVAL '30 days' "
        "GROUP BY EXTRACT(HOUR FROM open_time) ORDER BY SUM(profit) ASC LIMIT 8"
    )).fetchall()
    if rows:
        print(f"  {'hora':<6} {'N':>4} {'avgR':>7} {'pnl':>10}")
        for r in rows:
            h, n, avgr, pnl = r
            mark = "*" if float(pnl or 0) < -20 else " "
            print(f" {mark}{int(h):>3}h    {n:>4} {float(avgr or 0):>+7.3f} ${float(pnl or 0):>+9.2f}")

finally:
    s.close()
print("\n" + "=" * 75)

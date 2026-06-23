"""Auditoría del estado del bot — versión corregida con schema real."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from analytics.db_connection import init_db, new_session


def main() -> int:
    init_db()
    s = new_session()
    try:
        print("=" * 76)
        print(" AUDIT — Estado del bot AURION X 4.2 (post-calibración SMC)")
        print("=" * 76)

        print("\n[1] === Trades XAUUSD últimas 24h ===")
        r = s.execute(text("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN r_multiple <= 0 THEN 1 ELSE 0 END) AS losses,
                   AVG(r_multiple) AS avg_r,
                   SUM(profit) AS pnl_usd,
                   MIN(open_time) AS first_open,
                   MAX(close_time) AS last_close
            FROM trades
            WHERE symbol='XAUUSD' AND close_time IS NOT NULL
              AND close_time > now() - interval '24 hour'
        """)).fetchone()
        print(f"  total={r[0]} wins={r[1]} losses={r[2]} avg_r={r[3]} pnl_usd=${r[4]}")
        print(f"  first={r[5]} last={r[6]}")
        if r[0] and r[0] > 0:
            wr = (r[1] or 0) / r[0]
            print(f"  → winrate 24h: {wr*100:.1f}%")

        print("\n[2] === Trades XAUUSD última hora (post-restart) ===")
        r = s.execute(text("""
            SELECT COUNT(*), AVG(r_multiple), SUM(profit)
            FROM trades
            WHERE symbol='XAUUSD'
              AND open_time > now() - interval '1 hour'
        """)).fetchone()
        print(f"  abiertos+cerrados última hora: {r[0]}  avg_r={r[1]}  profit=${r[2]}")

        print("\n[3] === Trades XAUUSD por estrategia (TODA la historia) ===")
        r = s.execute(text("""
            SELECT strategy, COUNT(*), AVG(r_multiple), SUM(profit),
                   SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END)::float / COUNT(*) AS wr
            FROM trades
            WHERE symbol='XAUUSD' AND close_time IS NOT NULL
            GROUP BY strategy ORDER BY 2 DESC
        """)).fetchall()
        print(f"  {'strategy':20s} {'count':>7s} {'avg_R':>8s} {'pnl_$':>10s} {'WR%':>6s}")
        for row in r:
            wr = (row[4] or 0) * 100
            print(f"  {(row[0] or 'none'):20s} {row[1]:>7d} {row[2] or 0:>8.3f} {row[3] or 0:>10.2f} {wr:>6.1f}")

        print("\n[4] === Dataset entrenable (features_json) ===")
        r = s.execute(text("""
            SELECT COUNT(*) FROM trades
            WHERE symbol='XAUUSD' AND features_json IS NOT NULL
              AND close_time IS NOT NULL AND r_multiple IS NOT NULL
        """)).fetchone()
        print(f"  trades reales con features + label (entrenables): {r[0]}")

        print("\n[5] === ml_features (timeline de aprendizaje passive) ===")
        r = s.execute(text("""
            SELECT COUNT(*),
                   MIN(timestamp), MAX(timestamp),
                   COUNT(DISTINCT symbol)
            FROM ml_features WHERE symbol='XAUUSD'
        """)).fetchone()
        print(f"  XAUUSD features: {r[0]} desde {r[1]} hasta {r[2]}")

        print("\n[6] === SMC histórico acumulado ===")
        for tbl in ("smc_bos", "smc_choch", "smc_order_blocks", "smc_fvg", "smc_liquidity", "smc_confluence", "smc_trade_link"):
            try:
                cnt = s.execute(text(f"SELECT COUNT(*) FROM {tbl} WHERE symbol='XAUUSD'")).fetchone()[0]
                print(f"  {tbl:25s} {cnt:>10d}")
            except Exception:
                s.rollback()
                cnt = s.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()[0]
                print(f"  {tbl:25s} {cnt:>10d} (sin filtro symbol)")

        print("\n[7] === XAU Knowledge Engine — estado ===")
        for tbl in ("xau_market_memory", "xau_patterns", "xau_regime_memory",
                    "xau_session_memory", "xau_smc_events", "xau_model_learning"):
            try:
                cnt = s.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()[0]
                print(f"  {tbl:25s} {cnt:>10d}")
            except Exception as exc:
                s.rollback()
                print(f"  {tbl:25s}  error: {exc}")

        print("\n[8] === Últimos 5 trades XAUUSD ===")
        r = s.execute(text("""
            SELECT open_time, direction, entry_price, sl, tp,
                   exit_price, r_multiple, profit, strategy
            FROM trades
            WHERE symbol='XAUUSD'
            ORDER BY open_time DESC LIMIT 5
        """)).fetchall()
        for row in r:
            print(f"  {row[0]} {row[1]:5s} entry={row[2]} sl={row[3]} tp={row[4]} "
                  f"exit={row[5]} R={row[6]} $={row[7]} strat={row[8]}")

        print("\n" + "=" * 76)
    except Exception as e:
        s.rollback()
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        return 1
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

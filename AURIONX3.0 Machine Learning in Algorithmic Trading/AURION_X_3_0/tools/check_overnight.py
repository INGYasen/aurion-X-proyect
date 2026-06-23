"""Diagnóstico rápido: qué hizo el bot durante las últimas N horas."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from analytics.db_connection import init_db, new_session


def safe(s, sql, params=None):
    """Ejecuta query con rollback automático si falla."""
    try:
        return s.execute(text(sql), params or {})
    except Exception as exc:
        s.rollback()
        raise


def main() -> int:
    init_db()
    s = new_session()
    try:
        utc_now = datetime.now(timezone.utc)
        since = utc_now - timedelta(hours=24)
        print(f"\n=== Diagnóstico AURION X — últimas 24h (desde {since.isoformat()}) ===\n")

        for tbl, label in [
            ("trades", "trades (todos)"),
        ]:
            try:
                row = s.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
                print(f"  {label:35s} histórico total = {row}")
            except Exception:
                s.rollback()
                print(f"  {label:35s} (tabla no existe)")

        print("\n--- Esquema de tabla trades ---")
        try:
            cols = s.execute(text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name='trades' ORDER BY ordinal_position"
            )).fetchall()
            for c in cols:
                print(f"  {c[0]:25s} {c[1]}")
        except Exception as exc:
            s.rollback()
            print(f"  error: {exc}")

        print("\n--- Trades XAUUSD últimas 24h ---")
        try:
            rows = s.execute(text(
                "SELECT * FROM trades WHERE symbol='XAUUSD' "
                "AND (close_time > :since OR open_time > :since) "
                "ORDER BY COALESCE(close_time, open_time) DESC LIMIT 40"
            ), {"since": since}).fetchall()
            if not rows:
                print("  (NINGUNO — no se operó XAUUSD en las últimas 24h)")
            else:
                print(f"  TOTAL: {len(rows)} trades de XAUUSD en 24h")
                for r in rows[:20]:
                    print(f"    {dict(r._mapping)}")
        except Exception as exc:
            s.rollback()
            print(f"  error: {exc}")

        print("\n--- Conteo por símbolo (últimas 24h) ---")
        try:
            rows = s.execute(text(
                "SELECT symbol, count(*) FROM trades "
                "WHERE COALESCE(close_time, open_time) > :since "
                "GROUP BY symbol ORDER BY count DESC"
            ), {"since": since}).fetchall()
            for r in rows:
                print(f"  {r[0]:10s} count={r[1]}")
        except Exception as exc:
            s.rollback()
            print(f"  error: {exc}")

        print("\n--- Snapshots SMC guardados ---")
        try:
            rows = s.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE 'smc%'"
            )).fetchall()
            for r in rows:
                tbl = r[0]
                try:
                    cnt = s.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
                    print(f"  {tbl:30s} total={cnt}")
                except Exception:
                    s.rollback()
        except Exception as exc:
            s.rollback()
            print(f"  error: {exc}")

    finally:
        s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

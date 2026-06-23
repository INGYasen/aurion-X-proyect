"""Resumen rapido EOD (hoy UTC) tras detener el bot."""
from __future__ import annotations
import sys
from datetime import datetime, timezone

from sqlalchemy import and_, desc

from analytics.db_connection import init_db, new_session
from analytics.models import Trade


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    init_db()
    s = new_session()

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    trades_hoy = (
        s.query(Trade)
        .filter(and_(Trade.symbol == "XAUUSD", Trade.open_time >= today_start))
        .order_by(desc(Trade.open_time))
        .all()
    )

    print("=== TRADES DE HOY (UTC) - XAUUSD ===")
    print(f"  Total abiertos hoy: {len(trades_hoy)}")
    cerrados = [t for t in trades_hoy if t.exit_price is not None]
    abiertos = [t for t in trades_hoy if t.exit_price is None]
    print(f"  Cerrados: {len(cerrados)}")
    print(f"  Aun abiertos en DB: {len(abiertos)}")

    if cerrados:
        wins = [t for t in cerrados if t.r_multiple and float(t.r_multiple) > 0]
        sumR = sum(float(t.r_multiple or 0) for t in cerrados)
        sumPnl = sum(float(t.profit or 0) for t in cerrados)
        wr = len(wins) / len(cerrados) * 100
        print(f"  WR cerrados: {len(wins)}/{len(cerrados)} = {wr:.1f}%")
        print(f"  Sum R     : {sumR:+.2f}")
        print(f"  PnL total : ${sumPnl:+.2f}")

    if abiertos:
        print()
        print("=== POSICIONES AUN ABIERTAS EN DB ===")
        print("  (revisa MT5: el bot esta detenido, no se gestionaran mas hoy)")
        for t in abiertos:
            print(
                f"  {t.open_time.strftime('%H:%M')}  "
                f"{t.direction.value:5s}  "
                f"lot={float(t.lot):.2f}  "
                f"entry={float(t.entry_price):.2f}  "
                f"sl={float(t.sl or 0):.2f}  "
                f"tp={float(t.tp or 0):.2f}"
            )

    s.close()


if __name__ == "__main__":
    main()

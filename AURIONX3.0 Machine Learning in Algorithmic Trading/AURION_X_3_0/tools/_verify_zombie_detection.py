"""Verificar que el nuevo zombie detector captura los trades buggeados."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.db_connection import init_db, new_session
from analytics.trade_repository import TradeRepository
from analytics.models import Trade
from sqlalchemy import select

init_db()
s = new_session()
try:
    rows = s.execute(
        select(Trade)
        .where(Trade.symbol == "XAUUSD")
        .where(Trade.close_time.is_not(None))
        .order_by(Trade.close_time.desc())
        .limit(2000)
    ).scalars().all()
    print(f"  Total trades XAUUSD cerrados: {len(rows)}")
    zombies = 0
    big_loss_zombies = 0
    recent_zombies = []
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    for r in rows:
        is_z = TradeRepository._is_zombie_trade(r, 60)
        if is_z:
            zombies += 1
            if r.r_multiple and float(r.r_multiple) < -8:
                big_loss_zombies += 1
                if r.close_time and r.close_time > cutoff:
                    recent_zombies.append({
                        "open": r.open_time, "dir": r.direction.value if r.direction else "?",
                        "R": float(r.r_multiple), "pnl": float(r.profit or 0),
                        "setup": (r.features_json or {}).get("setup_type") if isinstance(r.features_json, dict) else "?",
                    })
    print(f"  Total zombies detectados: {zombies}")
    print(f"  Zombies con R<-8 (big losses): {big_loss_zombies}")
    if recent_zombies:
        print(f"\n  Recent big-loss zombies (ultimos 14d): {len(recent_zombies)}")
        for z in recent_zombies:
            print(f"    {z['open']} {z['dir']:<5} R={z['R']:+.2f} pnl=${z['pnl']:+.2f} {z['setup']}")
finally:
    s.close()

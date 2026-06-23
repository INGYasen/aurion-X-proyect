import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session
from datetime import datetime, timedelta, timezone

init_db()
s = new_session()
since = datetime.now(timezone.utc) - timedelta(hours=6)

cols = s.execute(text("""
    SELECT column_name FROM information_schema.columns WHERE table_name='trades'
""")).fetchall()
colnames = [c[0] for c in cols]
side_col = None
for cand in ("side", "direction", "action", "order_type", "trade_type", "type"):
    if cand in colnames:
        side_col = cand
        break
side_select = side_col if side_col else "'?'::text as side"
exit_col = "exit_reason" if "exit_reason" in colnames else (
    "close_reason" if "close_reason" in colnames else "'-' as exit_reason"
)
exit_select = exit_col if exit_col in colnames else "'-'::text as exit_reason"
rows = s.execute(text(f"""
    SELECT id, strategy, {side_select} as side, open_time, close_time, profit, r_multiple, {exit_select}
    FROM trades
    WHERE symbol='XAUUSD' AND open_time >= :since
    ORDER BY open_time ASC
"""), {"since": since}).fetchall()

print(f"Trades XAUUSD desde hace 6h ({since.strftime('%H:%M UTC')}): {len(rows)}")
print()
tot_pnl = 0.0
tot_r = 0.0
wins = 0
losses = 0
openn = 0
buys = 0
sells = 0
for r in rows:
    closed = r[4] is not None
    pnl = float(r[5] or 0)
    rm = float(r[6] or 0)
    tot_pnl += pnl
    tot_r += rm
    side = str(r[2] or "?")
    if side == "BUY": buys += 1
    elif side == "SELL": sells += 1
    if closed:
        if pnl > 0: wins += 1
        else: losses += 1
    else:
        openn += 1
    open_t = r[3].strftime("%H:%M") if r[3] else "?"
    close_t = r[4].strftime("%H:%M") if r[4] else "OPEN"
    pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
    print(f"  id={str(r[0])[:8]} {str(r[1]):8s} {side:4s} open={open_t} close={close_t} "
          f"R={rm:+.2f} pnl=${pnl_str:>8s}  exit={r[7] or '-'}")

print()
print(f"Resumen 5-6h: {len(rows)} trades")
print(f"  Direccion : BUY={buys} | SELL={sells}")
print(f"  Cerrados  : {wins} WIN | {losses} LOSS  ({(wins/max(1,wins+losses))*100:.1f}% WR)")
print(f"  Abiertos  : {openn}")
print(f"  Net PnL   : ${tot_pnl:+.2f}")
print(f"  Net R     : {tot_r:+.2f}")
s.close()

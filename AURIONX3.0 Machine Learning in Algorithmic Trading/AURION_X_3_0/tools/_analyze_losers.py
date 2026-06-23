"""Forensic post-mortem: ¿por qué cayeron los trades antes del kill switch?"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session
from datetime import datetime, timedelta, timezone

init_db()
s = new_session()

# Detectar columnas existentes
cols_q = s.execute(text("""
    SELECT column_name FROM information_schema.columns WHERE table_name='trades'
""")).fetchall()
colnames = {c[0] for c in cols_q}
dir_col = next((c for c in ("direction", "side", "action", "trade_type") if c in colnames), None)
exit_col = next((c for c in ("exit_reason", "close_reason") if c in colnames), None)
setup_col = next((c for c in ("setup_type", "setup") if c in colnames), None)
entry_col = next((c for c in ("open_price", "entry_price", "entry") if c in colnames), None)
close_col = next((c for c in ("close_price", "exit_price") if c in colnames), None)
sl_col = next((c for c in ("stop_loss", "sl") if c in colnames), None)
tp_col = next((c for c in ("take_profit", "tp") if c in colnames), None)
lot_col = next((c for c in ("lot", "volume", "lots", "size") if c in colnames), None)

print(f"Columnas detectadas: dir={dir_col} exit={exit_col} setup={setup_col} "
      f"entry={entry_col} close={close_col} sl={sl_col} tp={tp_col} lot={lot_col}")

# Trades cerrados de hoy
since = datetime.now(timezone.utc) - timedelta(hours=14)
def fld(name, alias):
    return f"{name} as {alias}" if name else f"NULL::text as {alias}"

rows = s.execute(text(f"""
    SELECT id, strategy, {fld(dir_col,'dir')}, open_time, close_time,
           profit, r_multiple, {fld(exit_col,'exit_r')}, {fld(setup_col,'setup')},
           {fld(entry_col,'entry')}, {fld(close_col,'close_p')},
           {fld(sl_col,'sl')}, {fld(tp_col,'tp')}, {fld(lot_col,'lot')},
           features_json
    FROM trades
    WHERE symbol='XAUUSD' AND open_time >= :since AND close_time IS NOT NULL
    ORDER BY open_time ASC
"""), {"since": since}).fetchall()

print(f"\nTrades cerrados (últimas 14h): {len(rows)}")
print("=" * 110)
print(f"{'ID':10} {'STR':9} {'DIR':5} {'OPEN':5} {'CLOSE':5} {'LOT':>5} "
      f"{'ENTRY':>8} {'EXIT':>8} {'PNL':>8} {'R':>7} {'SETUP':22} EXIT")
print("-" * 110)

losers = []
winners = []
for r in rows:
    trade_id = str(r[0])[:8]
    strat = str(r[1] or "?")[:8]
    direction = str(r[2] or "?")[:5]
    open_t = r[3].strftime("%H:%M") if r[3] else "?"
    close_t = r[4].strftime("%H:%M") if r[4] else "?"
    pnl = float(r[5] or 0)
    rm = float(r[6] or 0)
    exit_r = str(r[7] or "-")
    setup = str(r[8] or "-")[:22]
    entry = float(r[9] or 0)
    close_p = float(r[10] or 0)
    sl = float(r[11] or 0)
    tp = float(r[12] or 0)
    lot = float(r[13] or 0)
    fj_raw = r[14]
    fj = fj_raw if isinstance(fj_raw, dict) else (json.loads(fj_raw) if fj_raw else {})
    
    is_loss = pnl < 0 or rm < 0
    rec = {
        "id": trade_id, "strat": strat, "dir": direction,
        "open": open_t, "close": close_t,
        "pnl": pnl, "rm": rm, "exit": exit_r, "setup": setup,
        "entry": entry, "close_p": close_p, "sl": sl, "tp": tp, "lot": lot,
        "fj": fj,
    }
    if is_loss:
        losers.append(rec)
    else:
        winners.append(rec)
    
    flag = "🔴" if is_loss else "🟢"
    print(f"{flag} {trade_id:9} {strat:9} {direction:5} {open_t:5} {close_t:5} {lot:>5.2f} "
          f"{entry:>8.2f} {close_p:>8.2f} {pnl:>+8.2f} {rm:>+7.2f} {setup:22} {exit_r}")

# Estadísticas
print()
print(f"WINNERS: {len(winners)}  PnL=${sum(w['pnl'] for w in winners):+.2f}  R={sum(w['rm'] for w in winners):+.2f}")
print(f"LOSERS : {len(losers)}  PnL=${sum(l['pnl'] for l in losers):+.2f}  R={sum(l['rm'] for l in losers):+.2f}")

# Análisis de losers
if losers:
    print("\n" + "=" * 110)
    print("ANALISIS DE PERDEDORES")
    print("=" * 110)
    
    # Por hora
    from collections import Counter
    hr = Counter(l["open"].split(":")[0] for l in losers)
    print("Distribución por HORA UTC:")
    for h, n in sorted(hr.items()):
        print(f"  {h}:xx → {n} perdedores")
    
    # Por dirección
    dirs = Counter(l["dir"] for l in losers)
    print("Por dirección:")
    for d, n in dirs.items():
        print(f"  {d}: {n}")
    
    # Por estrategia/setup
    strats = Counter(l["strat"] for l in losers)
    print("Por estrategia:")
    for st, n in strats.items():
        print(f"  {st}: {n}")
    
    setups = Counter(l["setup"] for l in losers)
    print("Por setup:")
    for st, n in setups.items():
        print(f"  {st}: {n}")
    
    # Por exit reason
    exits = Counter(l["exit"] for l in losers)
    print("Por motivo de salida:")
    for ex, n in exits.items():
        print(f"  {ex}: {n}")
    
    # SL/Entry distance
    print("\nDistancia entrada → SL (pips de XAU):")
    for l in losers:
        if l["entry"] > 0 and l["sl"] > 0:
            dist_pips = abs(l["entry"] - l["sl"]) * 10  # XAU: 1 pip = 0.1
            print(f"  {l['id']} {l['dir']} entry={l['entry']:.2f} sl={l['sl']:.2f} dist={dist_pips:.1f} pips R={l['rm']:+.2f}")
    
    # MTF context
    print("\nContexto MTF de cada perdedor (features_json):")
    for l in losers:
        fj = l["fj"] or {}
        mtf = fj.get("mtf_trends") or {}
        macro = fj.get("macro_bias") or fj.get("smc_macro_bias") or "?"
        op = fj.get("operational_bias") or fj.get("smc_op_bias") or "?"
        ml_conf = fj.get("ml_confidence") or fj.get("ml_conf") or "?"
        print(f"  {l['id']} {l['dir']}: macro={macro} op={op} ml_conf={ml_conf} mtf={mtf}")

s.close()

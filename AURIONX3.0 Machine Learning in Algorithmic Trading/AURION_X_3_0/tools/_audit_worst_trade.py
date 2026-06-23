"""Investigar trade catastrofico R=-53.25 del 2026-06-04 18:37:46."""
import json
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
try:
    row = s.execute(text(
        "SELECT id, open_time, close_time, direction, lot, entry_price, exit_price, "
        "       sl, tp, profit, r_multiple, ml_confidence, strategy, close_reason, "
        "       features_json "
        "FROM trades WHERE symbol='XAUUSD' "
        "AND close_time BETWEEN '2026-06-04 18:30' AND '2026-06-04 18:45' "
        "ORDER BY r_multiple ASC LIMIT 1"
    )).fetchone()
    if not row:
        print("Trade no encontrado.")
        sys.exit(1)
    (tid, ot, ct, dr, lot, ep, xp, sl, tp, pnl, rm, mlc, strat, cr, fj) = row
    print("=" * 70)
    print(" TRADE CATASTROFICO 2026-06-04 18:37 (R=-53.25, peor del periodo)")
    print("=" * 70)
    print(f"  ID:              {str(tid)[:12]}")
    print(f"  Open  time:      {ot}")
    print(f"  Close time:      {ct}")
    duration = (ct - ot).total_seconds() if ct and ot else 0
    print(f"  Duracion:        {duration/60:.1f} min")
    print(f"  Direccion:       {dr}")
    print(f"  Lote:            {float(lot or 0):.3f}")
    print(f"  Entry:           {float(ep or 0):.5f}")
    print(f"  Exit:            {float(xp or 0):.5f}")
    print(f"  SL:              {float(sl or 0):.5f}")
    print(f"  TP:              {float(tp or 0):.5f}")
    print(f"  Profit:          ${float(pnl or 0):.2f}")
    print(f"  R-multiple:      {float(rm or 0):.3f}")
    print(f"  ML confidence:   {float(mlc or 0):.3f}")
    print(f"  Strategy:        {strat}")
    print(f"  Close reason:    {cr or '(none)'}")
    print()
    if dr == "LONG":
        sl_dist = float(ep) - float(sl)
        actual_dist = float(ep) - float(xp)
    else:
        sl_dist = float(sl) - float(ep)
        actual_dist = float(xp) - float(ep)
    print(f"  Distancia SL inicial:  {sl_dist:.5f} (R=1)")
    print(f"  Distancia ejecucion:   {actual_dist:.5f}")
    print(f"  Ratio:                 {actual_dist/sl_dist:.2f}x del SL inicial")
    print()
    print("  >> ANALISIS: si ratio > 1, el SL fue ampliado/movido o el broker")
    print("     ejecuto MUY lejos por slippage extremo. R=-53 = SL ejecutado")
    print("     53x mas lejos que el riesgo planeado. Bug critico.")
    print()
    if fj and isinstance(fj, dict):
        print("  FEATURES claves del setup:")
        keys_interesting = [
            "setup_type", "volatility_regime", "score", "score_bucket", "rr",
            "setup_grade", "trading_mode", "session_aggression", "mtf_alignment",
            "mtf_M1", "mtf_M5", "mtf_M15", "mtf_H1", "mtf_H4", "mtf_D1",
            "ml_confidence_v4", "prob_buy_v4",
            "atr_m1", "spread_norm", "fake_breakout_prob", "range_position",
            "sweep_score", "absorption_score", "directional_conviction",
            "v4_used_patchtst",
        ]
        for k in keys_interesting:
            if k in fj:
                v = fj[k]
                if isinstance(v, float):
                    print(f"    {k:30s} {v:.4f}")
                else:
                    print(f"    {k:30s} {v}")

    # Contexto: que pasaba 5 min antes/despues
    print("\n  CONTEXTO: otros trades en +-2 horas")
    rows = s.execute(text(
        "SELECT open_time, direction, profit, r_multiple, features_json->>'setup_type', close_reason "
        "FROM trades WHERE symbol='XAUUSD' "
        "AND open_time BETWEEN :start AND :end "
        "ORDER BY open_time ASC"
    ), {
        "start": ot.replace(hour=ot.hour-2 if ot.hour >= 2 else 0),
        "end": ot.replace(hour=ot.hour+2 if ot.hour <= 21 else 23),
    }).fetchall()
    for r in rows:
        ot2, dr2, pnl2, rm2, setup2, reason2 = r
        print(f"    {ot2} {dr2:<5} pnl=${float(pnl2 or 0):+.2f} R={float(rm2 or 0):+.2f} {setup2 or '-'} ({reason2 or '-'})")
finally:
    s.close()

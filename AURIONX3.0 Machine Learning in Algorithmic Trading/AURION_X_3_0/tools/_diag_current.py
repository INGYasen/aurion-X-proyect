"""Diagnostico rapido del estado actual del bot para identificar
debilidades concretas. AURION X 4.3 (10-jun-2026)."""
from __future__ import annotations
import sys
from sqlalchemy import and_, desc

from analytics.db_connection import init_db, new_session
from analytics.models import Trade


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    init_db()
    s = new_session()

    # 1) Recent 30
    print("=== ULTIMOS 30 TRADES XAUUSD (cerrados) ===")
    recent = (
        s.query(Trade)
        .filter(and_(Trade.symbol == "XAUUSD", Trade.exit_price.isnot(None)))
        .order_by(desc(Trade.open_time))
        .limit(30)
        .all()
    )
    wins = [t for t in recent if t.r_multiple and float(t.r_multiple) > 0]
    total_r = sum(float(t.r_multiple or 0.0) for t in recent)
    total_pnl = sum(float(t.profit or 0.0) for t in recent)
    print(
        f"  N={len(recent)}  W={len(wins)}  L={len(recent)-len(wins)}  "
        f"WR={len(wins)/len(recent)*100:5.1f}%  sumR={total_r:+.2f}  "
        f"PnL=${total_pnl:+.2f}"
    )

    # 2) Setup performance (last 50)
    print("\n=== WINRATE por SETUP (ultimos 50 trades) ===")
    recent50 = (
        s.query(Trade)
        .filter(and_(Trade.symbol == "XAUUSD", Trade.exit_price.isnot(None)))
        .order_by(desc(Trade.open_time))
        .limit(50)
        .all()
    )
    by_setup: dict[str, list[Trade]] = {}
    for t in recent50:
        # setup_type vive dentro de features_json (no es columna directa)
        st = "UNKNOWN"
        try:
            fj = t.features_json or {}
            st = fj.get("setup_type") or fj.get("strategy") or t.strategy.value or "UNKNOWN"
        except Exception:
            pass
        by_setup.setdefault(st, []).append(t)
    for setup, ts in sorted(by_setup.items(), key=lambda x: -len(x[1])):
        ww = sum(1 for t in ts if t.r_multiple and float(t.r_multiple) > 0)
        tot_r = sum(float(t.r_multiple or 0.0) for t in ts)
        flag = ""
        if len(ts) >= 5 and ww / len(ts) < 0.35:
            flag = "  <- DEBIL"
        if len(ts) >= 5 and ww / len(ts) > 0.60:
            flag = "  <- FUERTE"
        print(
            f"  {setup:35s}  N={len(ts):3d}  WR={ww/len(ts)*100:5.1f}%  "
            f"sumR={tot_r:+6.2f}{flag}"
        )

    # 3) Hour performance (last 100)
    print("\n=== WINRATE por HORA UTC (ultimos 100, N>=3) ===")
    recent100 = (
        s.query(Trade)
        .filter(and_(Trade.symbol == "XAUUSD", Trade.exit_price.isnot(None)))
        .order_by(desc(Trade.open_time))
        .limit(100)
        .all()
    )
    by_h: dict[int, list[Trade]] = {}
    for t in recent100:
        if not t.open_time:
            continue
        by_h.setdefault(t.open_time.hour, []).append(t)
    for h in sorted(by_h):
        ts = by_h[h]
        if len(ts) < 3:
            continue
        ww = sum(1 for t in ts if t.r_multiple and float(t.r_multiple) > 0)
        tot_r = sum(float(t.r_multiple or 0.0) for t in ts)
        flag = ""
        if len(ts) >= 5 and ww / len(ts) < 0.35:
            flag = "  <- DEBIL"
        if len(ts) >= 5 and ww / len(ts) > 0.60:
            flag = "  <- FUERTE"
        print(
            f"  h={h:02d}  N={len(ts):3d}  WR={ww/len(ts)*100:5.1f}%  "
            f"sumR={tot_r:+6.2f}{flag}"
        )

    # 4) Direction
    print("\n=== WINRATE LONG vs SHORT (ultimos 100) ===")
    by_d: dict[str, list[Trade]] = {}
    for t in recent100:
        by_d.setdefault(t.direction.value if t.direction else "UNK", []).append(t)
    for d, ts in by_d.items():
        ww = sum(1 for t in ts if t.r_multiple and float(t.r_multiple) > 0)
        tot_r = sum(float(t.r_multiple or 0.0) for t in ts)
        print(
            f"  {d:6s}  N={len(ts):3d}  WR={ww/len(ts)*100:5.1f}%  "
            f"sumR={tot_r:+6.2f}"
        )

    # 5) close_reason
    print("\n=== CLOSE_REASON (ultimos 100) ===")
    by_cr: dict[str, list[Trade]] = {}
    for t in recent100:
        by_cr.setdefault(t.close_reason or "NULL", []).append(t)
    for cr, ts in sorted(by_cr.items(), key=lambda x: -len(x[1])):
        ww = sum(1 for t in ts if t.r_multiple and float(t.r_multiple) > 0)
        tot_r = sum(float(t.r_multiple or 0.0) for t in ts)
        print(
            f"  {cr:25s}  N={len(ts):3d}  WR={ww/len(ts)*100:5.1f}%  "
            f"sumR={tot_r:+6.2f}"
        )

    # 6) ML confidence vs outcome (last 100)
    print("\n=== ML_CONFIDENCE vs RESULTADO (ultimos 100) ===")
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
    for lo, hi in buckets:
        ts = [
            t
            for t in recent100
            if t.ml_confidence is not None and lo <= float(t.ml_confidence) < hi
        ]
        if not ts:
            continue
        ww = sum(1 for t in ts if t.r_multiple and float(t.r_multiple) > 0)
        tot_r = sum(float(t.r_multiple or 0.0) for t in ts)
        print(
            f"  conf {lo:.2f}-{hi:.2f}  N={len(ts):3d}  "
            f"WR={ww/len(ts)*100:5.1f}%  sumR={tot_r:+6.2f}"
        )

    s.close()


if __name__ == "__main__":
    main()

"""Backtest de filtros sobre trades historicos.

AURION X 4.3 (7-jun-2026): valida si los filtros ANTI-CHOP / NEWS / BLACKLIST
actuales habrian mejorado el PnL historico. Util para tunear thresholds sin
arriesgar dinero en vivo.

Que hace:
    1. Lee TODOS los trades XAUUSD historicos (cerrados, no zombies).
    2. Para cada uno, simula los filtros con sus parametros actuales:
        - news_calendar.in_blackout(open_time, "XAUUSD")
        - chop_guard.block_by_setup_blacklist(setup_type)
        - filtro ML conf minima
    3. Reporta:
        - trades originales: count, win_rate, avg_R, total_R, profit_total
        - trades sobrevivientes: mismo
        - trades filtrados: cuantos, win_rate (esperado bajo), total_R perdido
    4. Conclusion: si filtered_win_rate < survivor_win_rate => filtros buenos.

Uso:
    python -m tools.backtest_filters
    python -m tools.backtest_filters --no-news       # solo chop+blacklist
    python -m tools.backtest_filters --min-ml 0.58
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from analytics.db_connection import init_db, new_session  # noqa: E402
from analytics.logger import get_logger  # noqa: E402
from risk.chop_guard import ChopGuard, ChopGuardConfig  # noqa: E402
from risk.economic_calendar import EconomicCalendarGuard  # noqa: E402

logger = get_logger(__name__)


def load_trades(symbol: str = "XAUUSD") -> list[dict]:
    """Carga trades cerrados validos (sin zombies)."""
    init_db()
    s = new_session()
    try:
        rows = s.execute(text(
            "SELECT id::text, symbol, direction, strategy, entry_price, exit_price, "
            "       sl, tp, profit, r_multiple, ml_confidence, "
            "       open_time, close_time, features_json "
            "FROM trades "
            "WHERE symbol = :sym AND close_time IS NOT NULL "
            "  AND EXTRACT(EPOCH FROM (close_time-open_time)) >= 60 "
            "  AND NOT (entry_price > 0 AND ("
            "       ABS(sl-entry_price)/entry_price > 0.05 "
            "    OR ABS(tp-entry_price)/entry_price > 0.05 "
            "    OR tp <= 0)) "
            "ORDER BY open_time ASC"
        ), {"sym": symbol}).fetchall()
        out = []
        for r in rows:
            features = r[13] if isinstance(r[13], dict) else {}
            out.append({
                "id": r[0],
                "symbol": r[1],
                "direction": r[2],
                "strategy": r[3],
                "entry_price": float(r[4] or 0),
                "exit_price": float(r[5] or 0),
                "sl": float(r[6] or 0),
                "tp": float(r[7] or 0),
                "profit": float(r[8] or 0),
                "r_multiple": float(r[9] or 0),
                "ml_confidence": float(r[10] or 0),
                "open_time": r[11],
                "close_time": r[12],
                "setup_type": features.get("setup_type", "") if isinstance(features, dict) else "",
                "range_position": features.get("range_position") if isinstance(features, dict) else None,
            })
        return out
    finally:
        s.close()


def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"count": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0, "profit_total": 0.0}
    wins = sum(1 for t in trades if t["profit"] > 0)
    total_r = sum(t["r_multiple"] for t in trades)
    total_p = sum(t["profit"] for t in trades)
    return {
        "count": len(trades),
        "win_rate": wins / len(trades),
        "avg_r": total_r / len(trades),
        "total_r": total_r,
        "profit_total": total_p,
        "wins": wins,
        "losses": len(trades) - wins,
    }


def fmt_stats(label: str, st: dict) -> str:
    return (f"  {label:<22} n={st['count']:>4}  WR={st['win_rate']*100:>5.1f}%  "
            f"avgR={st['avg_r']:>+5.3f}  totalR={st['total_r']:>+7.2f}  "
            f"PnL=${st['profit_total']:>+9.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--no-news", action="store_true", help="No simular news_calendar")
    parser.add_argument("--no-chop", action="store_true", help="No simular chop_guard.blacklist")
    parser.add_argument("--min-ml", type=float, default=None, help="Bloquear trades con ml_conf < min")
    parser.add_argument("--blocked-setups", nargs="+", default=["RANGE_STRUCTURE_M15_FADE"],
                          help="Setups a filtrar")
    args = parser.parse_args()

    print("=" * 80)
    print(f" BACKTEST DE FILTROS — {args.symbol}")
    print("=" * 80)
    trades = load_trades(args.symbol)
    print(f" Trades validos cargados: {len(trades)}")
    if not trades:
        print(" No hay datos historicos. Aborta.")
        return
    print(f" Primer trade: {trades[0]['open_time']}")
    print(f" Ultimo trade: {trades[-1]['open_time']}")

    # Stats originales (sin filtros nuevos).
    print("\n[1] STATS ORIGINALES (asi opero el bot historicamente):")
    print(fmt_stats("Todos", stats(trades)))

    # Setup filters guards. Lee la config YAML para que el backtest use los
    # mismos parametros que el bot live (incluyendo include_recurring + override).
    news = None
    if not args.no_news:
        try:
            import yaml
            cfg_path = Path(__file__).parent.parent / "config" / "parameters.yaml"
            with cfg_path.open("r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
            news_cfg = (full_cfg.get("risk", {}) or {}).get("news_calendar")
            news = EconomicCalendarGuard.from_config(news_cfg)
        except Exception as exc:
            print(f"  WARN: no se pudo leer YAML news_calendar ({exc}); usando defaults.")
            news = EconomicCalendarGuard(enabled=True)
    chop_cfg = ChopGuardConfig(
        enabled=not args.no_chop,
        blocked_setups=tuple(args.blocked_setups),
    )
    chop = ChopGuard(chop_cfg) if not args.no_chop else None

    # Aplicar filtros.
    blocked: dict[str, list[dict]] = {"news": [], "blacklist": [], "ml_conf": []}
    survivors: list[dict] = []
    for t in trades:
        ot = t["open_time"]
        if isinstance(ot, datetime) and ot.tzinfo is None:
            ot = ot.replace(tzinfo=timezone.utc)
        block_reason = None
        if news is not None and news.enabled:
            # Pasar ml_conf para que override_high_ml_conf funcione.
            in_news, _ = news.in_blackout(ot, symbol=t["symbol"], ml_conf=t.get("ml_confidence"))
            if in_news:
                block_reason = "news"
        if block_reason is None and chop is not None and chop.enabled:
            bl_block, _ = chop.block_by_setup_blacklist(t["setup_type"])
            if bl_block:
                block_reason = "blacklist"
        if block_reason is None and args.min_ml is not None:
            if t["ml_confidence"] < args.min_ml:
                block_reason = "ml_conf"
        if block_reason is None:
            survivors.append(t)
        else:
            blocked[block_reason].append(t)

    print("\n[2] STATS CON FILTROS NUEVOS:")
    print(fmt_stats("Sobrevivientes", stats(survivors)))
    for kind, lst in blocked.items():
        if lst:
            label = f"Filtrados ({kind})"
            st = stats(lst)
            print(fmt_stats(label, st))
            # Validacion: trades filtrados deberian tener WR < sobrevivientes.
            surv_wr = stats(survivors)["win_rate"]
            if st["win_rate"] < surv_wr:
                print(f"    GOOD: WR filtrados < WR sobrevivientes ({st['win_rate']*100:.1f}% < {surv_wr*100:.1f}%)")
            else:
                print(f"    BAD : WR filtrados >= WR sobrevivientes ({st['win_rate']*100:.1f}% >= {surv_wr*100:.1f}%)")

    print("\n[3] CONCLUSION:")
    so = stats(trades)
    ss = stats(survivors)
    delta_wr = (ss["win_rate"] - so["win_rate"]) * 100
    delta_r = ss["avg_r"] - so["avg_r"]
    delta_total_r = ss["total_r"] - so["total_r"]
    delta_pnl = ss["profit_total"] - so["profit_total"]
    print(f"  Δ WinRate   : {delta_wr:+.2f} pp")
    print(f"  Δ avgR      : {delta_r:+.3f}")
    print(f"  Δ totalR    : {delta_total_r:+.2f}")
    print(f"  Δ PnL total : ${delta_pnl:+.2f}")
    if delta_pnl > 0:
        print(f"  RESULTADO   : FILTROS APORTAN ${delta_pnl:.2f} en histórico ({(delta_pnl/abs(so['profit_total']))*100 if so['profit_total'] != 0 else 0:.1f}% mejora)")
    elif delta_pnl == 0:
        print("  RESULTADO   : FILTROS NEUTRALES (no afectaron PnL historico)")
    else:
        print(f"  RESULTADO   : FILTROS RESTAN ${abs(delta_pnl):.2f} (filtros demasiado agresivos)")

    print("=" * 80)


if __name__ == "__main__":
    main()

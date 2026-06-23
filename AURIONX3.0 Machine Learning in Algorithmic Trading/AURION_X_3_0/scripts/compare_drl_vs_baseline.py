"""AURION X 4.0 — Comparador shadow DRL vs DynamicExitManager (baseline 3.0).

Tras N trades en sombra, lee la tabla ``drl_shadow_decisions`` y compara:

    - ROI simulado del agente DRL.
    - ROI real del DynamicExitManager 3.0.
    - Drawdown simulado vs real.

Emite verdict ``promote`` o ``hold``. NO escribe ningún flag automáticamente:
el usuario debe pasar de ``shadow_mode: true`` a ``false`` en config tras
revisar el reporte.

Uso:
    python -m scripts.compare_drl_vs_baseline --min-trades 200
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _ensure_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compara DRL shadow vs baseline 3.0.")
    p.add_argument("--symbol", type=str, default="XAUUSD")
    p.add_argument("--min-trades", type=int, default=200)
    p.add_argument("--min-uplift-pct", type=float, default=15.0)
    p.add_argument("--max-dd-uplift-pct", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_paths()

    try:
        from analytics.db_connection import get_session_factory  # type: ignore[attr-defined]
        from analytics.drl_shadow_models import DRLShadowDecision
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"BD no disponible: {exc}")

    Session = get_session_factory()
    with Session() as session:
        rows = (
            session.query(DRLShadowDecision)
            .filter(DRLShadowDecision.symbol == args.symbol)
            .order_by(DRLShadowDecision.ts.asc())
            .all()
        )

    if len(rows) < args.min_trades:
        print(f"[INFO] Aún no hay suficientes decisiones sombra: {len(rows)} < {args.min_trades}")
        return 1

    drl_pnl = sum(float(r.drl_pnl_sim or 0.0) for r in rows)
    actual_pnl = sum(float(r.actual_pnl or 0.0) for r in rows)

    # Drawdown: max-pico - min-valle de PnL acumulado
    def _max_dd(values: list[float]) -> float:
        if not values:
            return 0.0
        acc = 0.0
        peak = 0.0
        max_dd = 0.0
        for v in values:
            acc += v
            peak = max(peak, acc)
            max_dd = max(max_dd, peak - acc)
        return max_dd

    drl_dd = _max_dd([float(r.drl_pnl_sim or 0.0) for r in rows])
    actual_dd = _max_dd([float(r.actual_pnl or 0.0) for r in rows])

    uplift_pct = ((drl_pnl - actual_pnl) / abs(actual_pnl) * 100.0) if abs(actual_pnl) > 1e-9 else 0.0
    dd_uplift_pct = ((drl_dd - actual_dd) / abs(actual_dd) * 100.0) if abs(actual_dd) > 1e-9 else 0.0

    promote = uplift_pct >= args.min_uplift_pct and dd_uplift_pct <= args.max_dd_uplift_pct
    verdict = "PROMOTE" if promote else "HOLD"

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "n_decisions": len(rows),
        "drl_pnl_sim_total": drl_pnl,
        "actual_pnl_total": actual_pnl,
        "drl_drawdown_sim": drl_dd,
        "actual_drawdown": actual_dd,
        "uplift_pct": uplift_pct,
        "dd_uplift_pct": dd_uplift_pct,
        "min_uplift_required": args.min_uplift_pct,
        "max_dd_uplift_allowed": args.max_dd_uplift_pct,
        "verdict": verdict,
    }
    print(json.dumps(report, indent=2))

    out = Path(f"D:/AURION_DATA/reports/phase3_drl_vs_baseline_{args.symbol}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReporte: {out}")
    return 0 if promote else 2


if __name__ == "__main__":
    sys.exit(main())

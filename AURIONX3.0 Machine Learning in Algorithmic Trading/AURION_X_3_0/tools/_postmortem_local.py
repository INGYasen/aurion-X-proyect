"""CLI ejecutor del Postmortem Analyzer (AURION X 4.5.2).

Uso:
    python tools/_postmortem_local.py [--days N] [--symbol XAUUSD] [--open]

Genera:
    reports/postmortem_XAUUSD_<timestamp>.html
    reports/postmortem_XAUUSD_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.postmortem_analyzer import analyze, render_html  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="AURION X Postmortem (local, sin LLM)")
    parser.add_argument("--symbol", default="XAUUSD", help="simbolo (default XAUUSD)")
    parser.add_argument("--days", type=int, default=1, help="dias hacia atras (default 1)")
    parser.add_argument("--top", type=int, default=10, help="top winners/losers (default 10)")
    parser.add_argument("--open", action="store_true", help="abrir HTML en el navegador al terminar")
    parser.add_argument(
        "--out-dir", default=str(ROOT / "reports"),
        help="directorio de salida (default ./reports)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Analizando {args.symbol} ultimos {args.days} dia(s)...")
    report = analyze(symbol=args.symbol, days_back=args.days, top_n=args.top)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = f"postmortem_{args.symbol}_{ts}"
    html_path = out_dir / f"{base}.html"
    json_path = out_dir / f"{base}.json"

    html_path.write_text(render_html(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Resumen en consola
    o = report.overall
    print()
    print("=" * 70)
    print(f"POSTMORTEM {args.symbol}   {args.days} dia(s)")
    print("=" * 70)
    print(f"  Trades cerrados : {o.n}")
    if o.n == 0:
        print("  (sin trades en el periodo)")
    else:
        print(f"  Win Rate        : {o.win_rate:.1%}  ({o.n_wins}W / {o.n_losses}L)")
        print(f"  Avg R           : {o.avg_r:+.3f}")
        print(f"  Total R         : {o.total_r:+.2f}")
        print(f"  Total PnL       : {o.total_pnl:+.2f} $")
        print(f"  Expectancy      : {o.expectancy:+.3f}")
    print()
    print(f"  Recomendaciones : {len(report.recommendations)}")
    for r in report.recommendations[:8]:
        sev = {"info": " i ", "warning": " ! ", "critical": "!!!"}.get(r.severity, "   ")
        print(f"    [{sev}] {r.target}")
        print(f"          → {r.reason[:140]}")
    if len(report.recommendations) > 8:
        print(f"    ... y {len(report.recommendations) - 8} mas en el HTML")
    print()
    print(f"  HTML : {html_path}")
    print(f"  JSON : {json_path}")

    if args.open:
        try:
            os.startfile(html_path)  # type: ignore[attr-defined]
        except Exception:
            print(f"(abre manualmente: {html_path})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

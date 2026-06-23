"""AURION X 4.3 - Limpieza completa NDX100/BTCUSD de PostgreSQL.

Elimina:
  - trades de NDX100/BTCUSD
  - quant_decisions de NDX100/BTCUSD (si existe la tabla)
  - eventos SMC asociados (smc_bos, smc_choch, smc_order_blocks, smc_fvg,
    smc_liquidity, smc_confluence) filtrados por symbol

Si una tabla no tiene columna 'symbol', la salta con warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from analytics.db_connection import init_db, new_session

TARGETS = ("NDX100", "BTCUSD")
TABLES = [
    "trades",
    "quant_decisions",
    "smc_bos",
    "smc_choch",
    "smc_order_blocks",
    "smc_fvg",
    "smc_liquidity",
    "smc_confluence",
]


def main() -> int:
    init_db()
    s = new_session()
    deleted = {}
    skipped = []
    try:
        for tbl in TABLES:
            try:
                # Verificar existencia tabla + columna symbol
                exists = s.execute(text(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = 'symbol'"
                    ")"
                ), {"t": tbl}).scalar()
                if not exists:
                    skipped.append(f"{tbl} (no existe o sin columna symbol)")
                    continue
                # Conteo previo
                n_before = s.execute(text(
                    f"SELECT COUNT(*) FROM {tbl} WHERE symbol IN :syms"
                ).bindparams(text("'(NDX100, BTCUSD)' as syms")), {}).scalar() if False else 0
                # Mejor con bindparams normales
                n_before = s.execute(text(
                    f"SELECT COUNT(*) FROM {tbl} WHERE symbol = ANY(:syms)"
                ), {"syms": list(TARGETS)}).scalar() or 0
                if n_before == 0:
                    deleted[tbl] = 0
                    continue
                s.execute(text(
                    f"DELETE FROM {tbl} WHERE symbol = ANY(:syms)"
                ), {"syms": list(TARGETS)})
                s.commit()
                deleted[tbl] = n_before
            except Exception as exc:
                s.rollback()
                skipped.append(f"{tbl} (error: {type(exc).__name__}: {exc})")
    finally:
        s.close()

    print("=" * 70)
    print(" LIMPIEZA DB: NDX100 + BTCUSD")
    print("=" * 70)
    total = sum(deleted.values())
    for tbl in TABLES:
        if tbl in deleted:
            mark = "OK" if deleted[tbl] > 0 else "--"
            print(f"  [{mark}] {tbl:<25} {deleted[tbl]:>6} filas eliminadas")
    if skipped:
        print("\n  Saltadas:")
        for sk in skipped:
            print(f"    - {sk}")
    print(f"\n  TOTAL ELIMINADO: {total} filas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

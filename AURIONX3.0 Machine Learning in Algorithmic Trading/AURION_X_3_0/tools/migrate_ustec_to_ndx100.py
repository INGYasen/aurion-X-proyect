"""Migra todas las filas con symbol='USTEC' a symbol='NDX100' en PostgreSQL.

Operativamente cuando el broker renombra el símbolo (Exness USTEC -> FundingPips NDX100),
todos los trades, sesiones y métricas históricas deben quedar bajo el nuevo nombre para
que el self-learning, retrain y telemetría sigan acumulando muestras.

Uso:
    python -m tools.migrate_ustec_to_ndx100             # inspecciona y aplica
    python -m tools.migrate_ustec_to_ndx100 --dry-run   # solo inspecciona
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

from analytics.db_connection import get_engine


OLD = "USTEC"
NEW = "NDX100"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="No aplica cambios, solo muestra el plan")
    args = parser.parse_args()

    engine = get_engine()
    with engine.connect() as conn:
        tables_with_symbol = conn.execute(
            text(
                """
                SELECT table_name FROM information_schema.columns
                WHERE column_name = 'symbol' AND table_schema = 'public'
                ORDER BY table_name
                """
            )
        ).scalars().all()

    if not tables_with_symbol:
        print("[migrate] No se encontraron tablas con columna 'symbol'.")
        return 0

    print(f"[migrate] Tablas con columna 'symbol' detectadas: {tables_with_symbol}")
    print(f"[migrate] Plan: {OLD} -> {NEW}\n")

    plan: list[tuple[str, int]] = []
    with engine.connect() as conn:
        for table in tables_with_symbol:
            n = conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE symbol = :s"),
                {"s": OLD},
            ).scalar() or 0
            n_target = conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE symbol = :s"),
                {"s": NEW},
            ).scalar() or 0
            plan.append((table, int(n)))
            print(f"  - {table}: {OLD}={n}  {NEW}={n_target}")

    total = sum(n for _, n in plan)
    print(f"\n[migrate] Total filas a renombrar: {total}")

    if args.dry_run:
        print("[migrate] --dry-run: no se aplican cambios.")
        return 0

    if total == 0:
        print("[migrate] Nada para migrar.")
        return 0

    with engine.begin() as conn:
        for table, n in plan:
            if n == 0:
                continue
            res = conn.execute(
                text(f"UPDATE {table} SET symbol = :new WHERE symbol = :old"),
                {"new": NEW, "old": OLD},
            )
            print(f"  [OK] {table}: {res.rowcount} filas actualizadas")

    print("\n[migrate] Verificación post-migración:")
    with engine.connect() as conn:
        for table, _ in plan:
            n_old = conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE symbol = :s"),
                {"s": OLD},
            ).scalar() or 0
            n_new = conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE symbol = :s"),
                {"s": NEW},
            ).scalar() or 0
            print(f"  {table}: {OLD}={n_old}  {NEW}={n_new}")

    print("\n[migrate] Migración completada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

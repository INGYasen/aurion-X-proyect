"""Cuenta trades cerrados en PostgreSQL (diagnóstico rápido)."""
from __future__ import annotations

from analytics.db_connection import get_engine, init_db
from sqlalchemy import text

init_db()
with get_engine().connect() as conn:
    closed = conn.execute(
        text("SELECT COUNT(*) FROM trades WHERE close_time IS NOT NULL")
    ).scalar()
    with_feat = conn.execute(
        text(
            "SELECT COUNT(*) FROM trades WHERE close_time IS NOT NULL "
            "AND features_json IS NOT NULL"
        )
    ).scalar()
    print(f"closed_trades={closed}")
    print(f"with_features={with_feat}")

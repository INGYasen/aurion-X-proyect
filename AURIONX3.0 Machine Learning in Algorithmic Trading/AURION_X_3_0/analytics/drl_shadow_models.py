"""AURION X 4.0 — Modelos SQLAlchemy para shadow trading del agente DRL.

Tabla nueva ``drl_shadow_decisions``:

    id            (PK serial)
    ts            (UTC timestamp)
    symbol        (XAUUSD)
    trade_ticket  (FK lógica al ticket del trade abierto)
    state_vec     (JSONB con los 18 features)
    drl_action    (int 0..4)
    actual_action (acción que tomó el DynamicExitManager 3.0)
    drl_reward_sim (reward inmediato simulado)
    drl_pnl_sim   (PnL acumulado simulado al final del trade)
    actual_pnl    (PnL real del trade al cerrar)
    notes         (texto libre)

El registro permite comparar offline las decisiones del agente vs. el manager
actual para decidir promoción a producción.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import declarative_base

Base: Any = declarative_base()


class DRLShadowDecision(Base):
    __tablename__ = "drl_shadow_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(16), nullable=False, index=True)
    trade_ticket = Column(String(64), nullable=True, index=True)
    state_vec = Column(JSON, nullable=False)
    drl_action = Column(Integer, nullable=False)
    actual_action = Column(String(32), nullable=True)
    drl_reward_sim = Column(Float, nullable=True)
    drl_pnl_sim = Column(Float, nullable=True)
    actual_pnl = Column(Float, nullable=True)
    notes = Column(String(255), nullable=True)


def ensure_table(engine: Any) -> None:
    """Crea la tabla si no existe. Idempotente."""
    Base.metadata.create_all(engine, tables=[DRLShadowDecision.__table__])

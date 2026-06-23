"""Modelos de dominio y ORM SQLAlchemy para persistencia analytics."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Date, DateTime, Enum as SAEnum, Integer, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarativa compartida por tablas analytics."""


class StrategyTag(str, Enum):
    """Estrategia que originó la operación."""

    SCALPER = "SCALPER"
    INTRADAY = "INTRADAY"


class Direction(str, Enum):
    """Dirección de la posición."""

    LONG = "LONG"
    SHORT = "SHORT"


class Trade(Base):
    """Operación individual persistida para analytics y ML."""

    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    direction: Mapped[Direction] = mapped_column(
        SAEnum(Direction, values_callable=lambda x: [e.value for e in x], native_enum=False),
        nullable=False,
    )
    lot: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    sl: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    tp: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    profit: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    r_multiple: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    strategy: Mapped[StrategyTag] = mapped_column(
        SAEnum(StrategyTag, values_callable=lambda x: [e.value for e in x], native_enum=False),
        nullable=False,
        index=True,
    )
    ml_confidence: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    features_json: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSON, nullable=True)
    # AURION X 4.3 (7-jun-2026): trade journal mejorado - razón de cierre.
    # Valores esperados: 'tp_hit', 'sl_hit', 'breakeven', 'profit_lock_soft',
    # 'profit_lock_hard', 'trailing_stop', 'secure_profit', 'manual_close',
    # 'time_exit', 'reversal_signal', 'partial_close', 'broker_reject', 'unknown'.
    close_reason: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)


class Session(Base):
    """Agregado de rendimiento por sesión o día (métricas resumidas).

    Nota: el nombre de clase ``Session`` es el pedido de dominio; al usar SQLAlchemy
    ``Session`` de sesión de unidad de trabajo, importar como
    ``from sqlalchemy.orm import Session as OrmSession``.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    pnl: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    drawdown: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    winrate: Mapped[Decimal] = mapped_column(Numeric(7, 4), nullable=False)


class SystemState(Base):
    """Instantánea de estado de cuenta / riesgo en un momento dado."""

    __tablename__ = "system_states"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    equity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    drawdown: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


# --- Dominio en memoria (repositorio / capa de aplicación) ---------------------


class TradeSide(str, Enum):
    """Lado de la operación en el mercado."""

    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    """Estado del ciclo de vida de una orden/operación."""

    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TradeRecord:
    """Representación persistente mínima de una operación (DTO)."""

    id: str | None
    symbol: str
    side: TradeSide
    volume: Decimal
    open_price: Decimal
    opened_at: datetime
    closed_at: datetime | None
    status: TradeStatus
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serializa el registro a un diccionario plano (stub)."""
        raise NotImplementedError

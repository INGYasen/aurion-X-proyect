"""ORM tablas SMC (Smart Money Concepts) — AURION X 4.1.

Mismo metadata ``Base`` que el resto del proyecto, por lo que ``init_db()``
crea estas tablas automáticamente sin tocar nada más (ver ``db_connection.py``).

Tablas:

* ``smc_bos``           — eventos Break of Structure
* ``smc_choch``         — eventos Change of Character
* ``smc_order_blocks``  — Order Blocks (con lifecycle)
* ``smc_fvg``           — Fair Value Gaps
* ``smc_liquidity``     — Liquidity pools (SSL/BSL/EqH/EqL/POOL) + flag swept
* ``smc_confluence``    — snapshot de confluencia agregada por símbolo/momento
* ``smc_trade_link``    — vincula un trade cerrado con los elementos SMC usados
                          y persiste el outcome (R, MFE, MAE) para self-learning.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


class SMCBOS(Base):
    __tablename__ = "smc_bos"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    strength: Mapped[str] = mapped_column(String(16), nullable=False)
    broken_swing_price: Mapped[float] = mapped_column(Float, nullable=False)
    broken_swing_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    break_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    break_price: Mapped[float] = mapped_column(Float, nullable=False)
    bos_strength: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    bos_quality: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    bos_reliability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    bos_age_bars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bos_distance_pips: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SMCCHOCH(Base):
    __tablename__ = "smc_choch"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    broken_swing_price: Mapped[float] = mapped_column(Float, nullable=False)
    broken_swing_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    break_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    break_price: Mapped[float] = mapped_column(Float, nullable=False)
    choch_strength: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    choch_quality: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    choch_reliability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    choch_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    choch_age_bars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SMCOrderBlock(Base):
    __tablename__ = "smc_order_blocks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    top: Mapped[float] = mapped_column(Float, nullable=False)
    bottom: Mapped[float] = mapped_column(Float, nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    ob_strength: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ob_volume_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ob_displacement_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ob_reaction_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ob_freshness: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ob_quality: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ob_age_bars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mitigation_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SMCFVG(Base):
    __tablename__ = "smc_fvg"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    top: Mapped[float] = mapped_column(Float, nullable=False)
    bottom: Mapped[float] = mapped_column(Float, nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    gap_size: Mapped[Decimal | None] = mapped_column(Numeric(14, 6), nullable=True)
    gap_size_pips: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    gap_quality: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    gap_age_bars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gap_fill_percent: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    fvg_strength: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    inverted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SMCLiquidity(Base):
    __tablename__ = "smc_liquidity"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(24), nullable=False, index=True)  # SSL/BSL/EQH/EQL/POOL
    price: Mapped[float] = mapped_column(Float, nullable=False)
    bound_low: Mapped[float] = mapped_column(Float, nullable=False)
    bound_high: Mapped[float] = mapped_column(Float, nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    touches: Mapped[int | None] = mapped_column(Integer, nullable=True)
    density: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    liquidity_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    stop_hunt_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    swept: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    swept_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sweep_kind: Mapped[str | None] = mapped_column(String(24), nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SMCConfluence(Base):
    __tablename__ = "smc_confluence"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    direction_bias: Mapped[str] = mapped_column(String(16), nullable=False)
    confluence_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    bos_component: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    choch_component: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ob_component: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    fvg_component: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    liquidity_component: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    macro_bias: Mapped[str | None] = mapped_column(String(16), nullable=True)
    institutional_bias: Mapped[str | None] = mapped_column(String(16), nullable=True)
    operational_bias: Mapped[str | None] = mapped_column(String(16), nullable=True)
    per_tf: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    weights: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SMCTradeLink(Base):
    """Vincula un ``trade_id`` (cerrado) con los elementos SMC usados al entrar.

    El campo ``outcome`` se actualiza en el on-close hook con R/MFE/MAE para
    permitir aprendizaje automático ("¿qué OBs/BOS/CHOCH funcionaron mejor?").
    """

    __tablename__ = "smc_trade_link"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    trade_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    close_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    primary_timeframe: Mapped[str | None] = mapped_column(String(8), nullable=True)
    confluence_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    smc_macro_bias: Mapped[str | None] = mapped_column(String(16), nullable=True)
    smc_inst_bias: Mapped[str | None] = mapped_column(String(16), nullable=True)
    smc_op_bias: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Persistimos las features SMC al momento de la entrada (clave para
    # entrenar modelos por estructura).
    features_smc: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Snapshot de los elementos clave usados (ids o resumen pequeño).
    used_elements: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Outcome del trade.
    r_multiple: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    mfe_r: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    mae_r: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    win: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

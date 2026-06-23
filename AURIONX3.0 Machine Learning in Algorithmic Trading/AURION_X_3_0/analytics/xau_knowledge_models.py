"""ORM tablas XAU Knowledge Engine — AURION X 4.1.

Tablas dedicadas exclusivamente al aprendizaje continuo de XAUUSD. Conviven
con las tablas SMC (``smc_*``) y las de trades (``trades``) sin duplicar
información: aquí se almacenan **estadísticas agregadas** y **memoria
contextual** orientadas al dashboard de Knowledge.

Tablas:

* ``xau_market_memory``   — observación por vela cerrada (regime/trend/range/
                            volatility/session/spread) — sirve de raw memory.
* ``xau_patterns``        — combinaciones de patrones SMC (e.g. "OB+FVG",
                            "CHOCH+BOS") con frecuencia + winrate posterior.
* ``xau_smc_events``      — eventos SMC con contexto (hora, sesión, régimen)
                            para análisis cruzado rápido (vs. tabla smc_*).
* ``xau_regime_memory``   — duración y comportamiento histórico por régimen.
* ``xau_session_memory``  — métricas por sesión (Asia/London/NY/Overlap).
* ``xau_model_learning``  — historial de reentrenamientos (trigger, métricas,
                            walk-forward, pesos temporales aplicados).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    JSON,
    Numeric,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


class XAUMarketMemory(Base):
    """Snapshot por vela cerrada (todas las TFs caben aquí; ``timeframe`` las
    distingue). Insertamos sólo al cerrar vela (no por tick) para no inflar
    PG."""

    __tablename__ = "xau_market_memory"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="XAUUSD")
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    tick_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Contexto derivado
    regime: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)
    trend: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    range_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    volatility_atr: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    volatility_zscore: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    session: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    hour_utc: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class XAUPattern(Base):
    """Combinaciones de patrones SMC con frecuencia y outcome posterior."""

    __tablename__ = "xau_patterns"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="XAUUSD")
    pattern_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # "OB+FVG", "CHOCH+BOS"…
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    occurrences: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sum_r_milli: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # r_multiple * 1000
    sum_mfe_milli: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sum_mae_milli: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class XAUSMCEvent(Base):
    """Evento SMC con contexto (denormalizado, optimizado para queries de
    knowledge — la tabla canónica es ``smc_*``)."""

    __tablename__ = "xau_smc_events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="XAUUSD")
    event_kind: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # BOS/CHOCH/OB/FVG/SSL/BSL/SWEEP
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    strength: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    quality: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    session: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    regime: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)
    hour_utc: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # Outcome ulterior (rellenado por el on-close del knowledge engine cuando
    # un trade vinculado se cierra usando este tipo de elemento).
    outcome_r_milli: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outcome_win: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class XAURegimeMemory(Base):
    """Memoria de regímenes: duración y comportamiento por episodio."""

    __tablename__ = "xau_regime_memory"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="XAUUSD")
    regime: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_atr: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    avg_range_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    trades_n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trades_wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sum_r_milli: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class XAUSessionMemory(Base):
    """Métricas por sesión y por día (granularidad diaria)."""

    __tablename__ = "xau_session_memory"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="XAUUSD")
    session: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # ASIA/LONDON/NY/OVERLAP_LDN_NY
    day: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    bars_n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_atr: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    avg_range_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    avg_spread_pts: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    trades_n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trades_wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sum_r_milli: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class XAUModelLearning(Base):
    """Historial de reentrenamientos del modelo XAUUSD aislado.

    Cada fila documenta:

    * Trigger (``closed_trades``, ``end_of_day``, ``regime_change``,
      ``degradation``, ``manual``).
    * Métricas pre/post (winrate, sharpe, expectancy).
    * Pesos temporales aplicados (50/30/20 por defecto, configurable).
    * Walk-forward folds y métricas.
    * Path del artefacto resultante (``model_XAUUSD.pkl``).
    """

    __tablename__ = "xau_model_learning"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="XAUUSD")
    trigger: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    samples_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    walk_forward_folds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pre_winrate: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    post_winrate: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    pre_sharpe: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    post_sharpe: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    pre_expectancy: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    post_expectancy: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    weights_used: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics_extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(String(512), nullable=True)


__all__ = [
    "XAUMarketMemory",
    "XAUPattern",
    "XAUSMCEvent",
    "XAURegimeMemory",
    "XAUSessionMemory",
    "XAUModelLearning",
]

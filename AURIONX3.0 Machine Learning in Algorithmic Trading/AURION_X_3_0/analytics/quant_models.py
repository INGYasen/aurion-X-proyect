"""ORM tablas cuantitativas dedicadas (separadas de features_json genérico)."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, DateTime, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


class MLPrediction(Base):
    """Predicción y decisión ML por evaluación de entrada."""

    __tablename__ = "ml_predictions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    trade_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False, default="M1")
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    continuation_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    reversal_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    sweep_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    absorption_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    institutional_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    execution_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ml_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    final_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    setup_grade: Mapped[str | None] = mapped_column(String(4), nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class MLFeatureRow(Base):
    """Vector de features cuantitativas en columnas dedicadas."""

    __tablename__ = "ml_features"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    trade_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    displacement_speed: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    imbalance_strength: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    liquidity_density: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    sweep_rejection_speed: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    volatility_compression: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    directional_conviction: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    session_aggression: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    breakout_quality: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    absorption_strength: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    range_quality: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    trend_quality: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    features_extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class MLRegimeHistory(Base):
    """Instantánea de régimen de mercado por evaluación."""

    __tablename__ = "ml_regime_history"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    market_regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    volatility_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    session_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    spread_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    manipulation_detected: Mapped[bool | None] = mapped_column(nullable=True)
    sweep_density: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    trading_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class MLModelPerformance(Base):
    """Métricas agregadas por versión de modelo y régimen."""

    __tablename__ = "ml_model_performance"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    regime_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    winrate: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    avg_rr: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    sharpe: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    drawdown: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    expectancy: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    accuracy: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    false_signal_rate: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    sample_size: Mapped[int | None] = mapped_column(nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

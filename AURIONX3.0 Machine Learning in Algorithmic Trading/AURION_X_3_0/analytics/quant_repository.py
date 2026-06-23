"""Persistencia en tablas cuantitativas dedicadas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from analytics.db_connection import new_session
from analytics.logger import get_logger
from analytics.quant_models import MLFeatureRow, MLModelPerformance, MLPrediction, MLRegimeHistory

logger = get_logger(__name__)


def _d(v: float | None) -> Decimal | None:
    if v is None:
        return None
    return Decimal(str(v))


class QuantAnalyticsRepository:
    """Repositorio para ml_predictions, ml_features, ml_regime_history, ml_model_performance."""

    def insert_prediction(
        self,
        *,
        symbol: str,
        direction: str,
        timestamp: datetime | None = None,
        timeframe: str = "M1",
        trade_id: str | None = None,
        confidence_score: float | None = None,
        continuation_probability: float | None = None,
        reversal_probability: float | None = None,
        sweep_probability: float | None = None,
        absorption_probability: float | None = None,
        institutional_score: float | None = None,
        execution_score: float | None = None,
        ml_score: float | None = None,
        final_decision: str | None = None,
        model_version: str | None = None,
        setup_grade: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        ts = timestamp or datetime.now(timezone.utc)
        pred_id = uuid.uuid4()
        tid = uuid.UUID(trade_id) if trade_id else None
        session = new_session()
        try:
            with session.begin():
                row = MLPrediction(
                    id=pred_id,
                    trade_id=tid,
                    timestamp=ts,
                    symbol=symbol.upper(),
                    timeframe=timeframe,
                    direction=direction.upper(),
                    confidence_score=_d(confidence_score),
                    continuation_probability=_d(continuation_probability),
                    reversal_probability=_d(reversal_probability),
                    sweep_probability=_d(sweep_probability),
                    absorption_probability=_d(absorption_probability),
                    institutional_score=_d(institutional_score),
                    execution_score=_d(execution_score),
                    ml_score=_d(ml_score),
                    final_decision=final_decision,
                    model_version=model_version,
                    setup_grade=setup_grade,
                    extra=extra,
                )
                session.add(row)
            return pred_id
        except SQLAlchemyError:
            logger.exception("insert_prediction falló symbol=%s", symbol)
            raise
        finally:
            session.close()

    def insert_features(
        self,
        *,
        symbol: str,
        features: dict[str, float],
        prediction_id: uuid.UUID | None = None,
        trade_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> uuid.UUID:
        ts = timestamp or datetime.now(timezone.utc)
        fid = uuid.uuid4()
        tid = uuid.UUID(trade_id) if trade_id else None
        session = new_session()
        try:
            with session.begin():
                row = MLFeatureRow(
                    id=fid,
                    prediction_id=prediction_id,
                    trade_id=tid,
                    timestamp=ts,
                    symbol=symbol.upper(),
                    displacement_speed=_d(features.get("displacement_speed")),
                    imbalance_strength=_d(features.get("candle_imbalance", features.get("imbalance_strength"))),
                    liquidity_density=_d(features.get("liquidity_density")),
                    sweep_rejection_speed=_d(features.get("sweep_score", features.get("sweep_rejection_speed"))),
                    volatility_compression=_d(features.get("volatility_compression")),
                    directional_conviction=_d(features.get("directional_conviction")),
                    session_aggression=_d(features.get("session_aggression")),
                    breakout_quality=_d(features.get("breakout_quality")),
                    absorption_strength=_d(features.get("absorption_score", features.get("absorption_strength"))),
                    range_quality=_d(features.get("range_quality_score", features.get("range_quality"))),
                    trend_quality=_d(features.get("trend_quality", features.get("mtf_alignment"))),
                    features_extra={k: v for k, v in features.items() if k not in ("displacement_speed",)},
                )
                session.add(row)
            return fid
        except SQLAlchemyError:
            logger.exception("insert_features falló symbol=%s", symbol)
            raise
        finally:
            session.close()

    def insert_regime(
        self,
        *,
        symbol: str,
        market_regime: str,
        volatility_state: str,
        session_type: str = "",
        spread_state: str = "",
        manipulation_detected: bool = False,
        sweep_density: float = 0.0,
        trading_mode: str = "",
        timestamp: datetime | None = None,
        extra: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        ts = timestamp or datetime.now(timezone.utc)
        rid = uuid.uuid4()
        session = new_session()
        try:
            with session.begin():
                session.add(
                    MLRegimeHistory(
                        id=rid,
                        timestamp=ts,
                        symbol=symbol.upper(),
                        market_regime=market_regime,
                        volatility_state=volatility_state,
                        session_type=session_type or None,
                        spread_state=spread_state or None,
                        manipulation_detected=manipulation_detected,
                        sweep_density=_d(sweep_density),
                        trading_mode=trading_mode or None,
                        extra=extra,
                    ),
                )
            return rid
        except SQLAlchemyError:
            logger.exception("insert_regime falló symbol=%s", symbol)
            raise
        finally:
            session.close()

    def insert_model_performance(
        self,
        *,
        model_version: str,
        metrics: dict[str, float],
        symbol: str | None = None,
        regime_label: str | None = None,
        sample_size: int | None = None,
        notes: str | None = None,
    ) -> uuid.UUID:
        pid = uuid.uuid4()
        session = new_session()
        try:
            with session.begin():
                session.add(
                    MLModelPerformance(
                        id=pid,
                        recorded_at=datetime.now(timezone.utc),
                        model_version=model_version,
                        symbol=symbol.upper() if symbol else None,
                        regime_label=regime_label,
                        winrate=_d(metrics.get("winrate")),
                        avg_rr=_d(metrics.get("avg_rr", metrics.get("avg_r"))),
                        sharpe=_d(metrics.get("sharpe")),
                        drawdown=_d(metrics.get("drawdown")),
                        expectancy=_d(metrics.get("expectancy")),
                        accuracy=_d(metrics.get("accuracy", metrics.get("oos_accuracy"))),
                        false_signal_rate=_d(metrics.get("false_signal_rate")),
                        sample_size=sample_size,
                        notes=notes,
                        metrics_extra=metrics,
                    ),
                )
            return pid
        except SQLAlchemyError:
            logger.exception("insert_model_performance falló version=%s", model_version)
            raise
        finally:
            session.close()

    def fetch_microstructure_snapshots(
        self,
        *,
        symbol: str,
        last_days: int = 30,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Devuelve snapshots de microestructura para entrenar Anomaly AE.

        Mapea las columnas almacenadas en ``ml_features.features_extra`` a los
        nombres esperados por ``ANOMALY_FEATURE_NAMES``:

            bid_ask_spread_norm   <- features_extra.spread_norm
            tick_volume_z         <- z-score de features_extra.tick_volume_proxy
                                     (calculado por símbolo en ventana móvil)
            cancel_fill_ratio     <- 0.0  (no disponible vía broker; proxy)
            imbalance_score       <- features_extra.candle_imbalance
            aggressor_ratio       <- features_extra.rejection_ratio
            sweep_score           <- features_extra.sweep_score
            displacement_speed    <- features_extra.momentum_strength
            liquidity_density     <- features_extra.liquidity_density

        Args:
            symbol: Símbolo (case-insensitive).
            last_days: ventana hacia atrás.
            limit: nº máximo de filas (None = todas).

        Returns:
            Lista de dicts con las 8 claves del vector de anomalía.
        """
        from datetime import datetime, timedelta, timezone as _tz
        session = new_session()
        cutoff = datetime.now(_tz.utc) - timedelta(days=int(last_days))
        try:
            with session.begin():
                stmt = (
                    select(MLFeatureRow)
                    .where(MLFeatureRow.symbol == symbol.upper())
                    .where(MLFeatureRow.timestamp >= cutoff)
                    .order_by(MLFeatureRow.timestamp.asc())
                )
                if limit is not None and limit > 0:
                    stmt = stmt.limit(int(limit))
                rows = list(session.scalars(stmt).all())

            if not rows:
                return []

            # Z-score on the fly de tick_volume_proxy (ventana global).
            tv = [
                float((r.features_extra or {}).get("tick_volume_proxy", 0.0) or 0.0)
                for r in rows
            ]
            import statistics as _stats
            mu_tv = _stats.fmean(tv) if tv else 0.0
            sd_tv = _stats.pstdev(tv) if len(tv) > 1 else 1.0
            sd_tv = sd_tv if sd_tv > 1e-9 else 1.0

            out: list[dict[str, Any]] = []
            for i, r in enumerate(rows):
                ex = r.features_extra or {}
                out.append({
                    "bid_ask_spread_norm": float(ex.get("spread_norm", 0.0) or 0.0),
                    "tick_volume_z": (tv[i] - mu_tv) / sd_tv,
                    "cancel_fill_ratio": 0.0,
                    "imbalance_score": float(ex.get("candle_imbalance", 0.0) or 0.0),
                    "aggressor_ratio": float(ex.get("rejection_ratio", 0.0) or 0.0),
                    "sweep_score": float(ex.get("sweep_score", 0.0) or 0.0),
                    "displacement_speed": float(ex.get("momentum_strength", 0.0) or 0.0),
                    "liquidity_density": float(ex.get("liquidity_density", 0.0) or 0.0),
                })
            return out
        except SQLAlchemyError:
            logger.exception("fetch_microstructure_snapshots falló symbol=%s", symbol)
            return []
        finally:
            session.close()

    def recent_predictions(self, symbol: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        session = new_session()
        try:
            with session.begin():
                stmt = select(MLPrediction).order_by(MLPrediction.timestamp.desc()).limit(limit)
                if symbol:
                    stmt = stmt.where(MLPrediction.symbol == symbol.upper())
                rows = list(session.scalars(stmt).all())
            out: list[dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "id": str(r.id),
                        "symbol": r.symbol,
                        "direction": r.direction,
                        "confidence_score": float(r.confidence_score) if r.confidence_score is not None else None,
                        "setup_grade": r.setup_grade,
                        "final_decision": r.final_decision,
                        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    },
                )
            return out
        finally:
            session.close()

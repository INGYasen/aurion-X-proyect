"""Repositorio XAU Knowledge — DAO para tablas ``xau_*``.

Diseñado para ser **soft-fail**: cualquier excepción se loguea y se traga, el
Knowledge Engine no debe poder romper el flujo del Orchestrator.

Operaciones:

* ``insert_market_memory_batch(rows)`` — guarda observaciones por vela.
* ``upsert_pattern(...)`` — incrementa contadores de un pattern_key.
* ``insert_smc_event(...)`` — denormaliza un evento SMC para queries rápidas.
* ``start_regime_episode(...)`` / ``close_regime_episode(...)`` — track de
  duración + comportamiento por régimen.
* ``upsert_session_day(...)`` — agrega métricas de una sesión/día.
* ``record_model_retrain(...)`` — auditoría de reentrenamientos.
* Consultas de lectura para dashboard: ``top_patterns``, ``top_smc_events``,
  ``session_summary``, ``regime_summary``, ``recent_retrains``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable, Mapping

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from .db_connection import new_session
from .logger import get_logger
from .xau_knowledge_models import (
    XAUMarketMemory,
    XAUModelLearning,
    XAUPattern,
    XAURegimeMemory,
    XAUSMCEvent,
    XAUSessionMemory,
)

logger = get_logger(__name__)


def _dec(v: Any) -> Decimal | None:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class XAUKnowledgeRepository:
    """Repositorio dedicado al XAU Knowledge Engine."""

    # ------------------------------------------------------------------ #
    # Market memory
    # ------------------------------------------------------------------ #

    def insert_market_memory_batch(self, rows: Iterable[Mapping[str, Any]]) -> int:
        rows_list = [dict(r) for r in rows]
        if not rows_list:
            return 0
        inserted = 0
        try:
            with new_session() as session:
                with session.begin():
                    for r in rows_list:
                        try:
                            row = XAUMarketMemory(
                                timestamp=_utc(r["timestamp"]),
                                symbol=str(r.get("symbol", "XAUUSD")).upper(),
                                timeframe=str(r["timeframe"]).upper(),
                                open=float(r["open"]),
                                high=float(r["high"]),
                                low=float(r["low"]),
                                close=float(r["close"]),
                                tick_volume=float(r["tick_volume"]) if r.get("tick_volume") is not None else None,
                                spread_points=float(r["spread_points"]) if r.get("spread_points") is not None else None,
                                regime=r.get("regime"),
                                trend=r.get("trend"),
                                range_pct=_dec(r.get("range_pct")),
                                volatility_atr=_dec(r.get("volatility_atr")),
                                volatility_zscore=_dec(r.get("volatility_zscore")),
                                session=r.get("session"),
                                hour_utc=int(r["hour_utc"]) if r.get("hour_utc") is not None else None,
                                extra=dict(r.get("extra") or {}),
                            )
                            session.add(row)
                            inserted += 1
                        except Exception as exc:
                            logger.debug("market_memory row inválida: %s", exc)
            return inserted
        except SQLAlchemyError as exc:
            logger.warning("insert_market_memory_batch SQL: %s", exc)
            return inserted
        except Exception as exc:
            logger.warning("insert_market_memory_batch fallo: %s", exc)
            return inserted

    # ------------------------------------------------------------------ #
    # Patterns
    # ------------------------------------------------------------------ #

    def upsert_pattern(
        self,
        *,
        pattern_key: str,
        timeframe: str,
        direction: str | None,
        when_utc: datetime,
        increment_occurrence: bool = True,
        outcome_win: bool | None = None,
        r_multiple: float | None = None,
        mfe_r: float | None = None,
        mae_r: float | None = None,
        score: float | None = None,
        symbol: str = "XAUUSD",
    ) -> None:
        try:
            with new_session() as session:
                with session.begin():
                    row = (
                        session.query(XAUPattern)
                        .filter(XAUPattern.symbol == symbol.upper())
                        .filter(XAUPattern.pattern_key == pattern_key)
                        .filter(XAUPattern.timeframe == timeframe.upper())
                        .filter(XAUPattern.direction == (direction.upper() if direction else None))
                        .first()
                    )
                    if row is None:
                        row = XAUPattern(
                            last_seen=_utc(when_utc),
                            symbol=symbol.upper(),
                            pattern_key=pattern_key,
                            timeframe=timeframe.upper(),
                            direction=direction.upper() if direction else None,
                            occurrences=0,
                            wins=0,
                            losses=0,
                            sum_r_milli=0,
                            sum_mfe_milli=0,
                            sum_mae_milli=0,
                        )
                        session.add(row)
                    if increment_occurrence:
                        row.occurrences = (row.occurrences or 0) + 1
                    row.last_seen = _utc(when_utc) or row.last_seen
                    if outcome_win is True:
                        row.wins = (row.wins or 0) + 1
                    elif outcome_win is False:
                        row.losses = (row.losses or 0) + 1
                    if r_multiple is not None:
                        row.sum_r_milli = (row.sum_r_milli or 0) + int(round(float(r_multiple) * 1000.0))
                    if mfe_r is not None:
                        row.sum_mfe_milli = (row.sum_mfe_milli or 0) + int(round(float(mfe_r) * 1000.0))
                    if mae_r is not None:
                        row.sum_mae_milli = (row.sum_mae_milli or 0) + int(round(float(mae_r) * 1000.0))
                    if score is not None:
                        n = max(row.occurrences or 1, 1)
                        prev = float(row.avg_score) if row.avg_score is not None else 0.0
                        new_avg = ((prev * (n - 1)) + float(score)) / n
                        row.avg_score = _dec(new_avg)
        except SQLAlchemyError as exc:
            logger.warning("upsert_pattern SQL: %s", exc)
        except Exception as exc:
            logger.warning("upsert_pattern fallo: %s", exc)

    # ------------------------------------------------------------------ #
    # SMC events (denormalizado para queries rápidas)
    # ------------------------------------------------------------------ #

    def insert_smc_event(
        self,
        *,
        timestamp: datetime,
        event_kind: str,
        timeframe: str,
        direction: str | None = None,
        strength: float | None = None,
        quality: float | None = None,
        session: str | None = None,
        regime: str | None = None,
        hour_utc: int | None = None,
        context: Mapping[str, Any] | None = None,
        symbol: str = "XAUUSD",
    ) -> uuid.UUID | None:
        try:
            with new_session() as s:
                with s.begin():
                    row = XAUSMCEvent(
                        timestamp=_utc(timestamp),
                        symbol=symbol.upper(),
                        event_kind=event_kind.upper(),
                        timeframe=timeframe.upper(),
                        direction=direction.upper() if direction else None,
                        strength=_dec(strength),
                        quality=_dec(quality),
                        session=session,
                        regime=regime,
                        hour_utc=hour_utc,
                        context=dict(context or {}),
                    )
                    s.add(row)
                    s.flush()
                    return row.id
        except SQLAlchemyError as exc:
            logger.warning("insert_smc_event SQL: %s", exc)
            return None
        except Exception as exc:
            logger.warning("insert_smc_event fallo: %s", exc)
            return None

    def attach_smc_event_outcome(
        self,
        *,
        event_ids: Iterable[uuid.UUID | str],
        r_multiple: float | None,
        win: bool | None,
    ) -> None:
        ids: list[uuid.UUID] = []
        for eid in event_ids:
            if isinstance(eid, uuid.UUID):
                ids.append(eid)
            else:
                try:
                    ids.append(uuid.UUID(str(eid)))
                except Exception:
                    continue
        if not ids:
            return
        try:
            with new_session() as s:
                with s.begin():
                    rows = (
                        s.query(XAUSMCEvent)
                        .filter(XAUSMCEvent.id.in_(ids))
                        .all()
                    )
                    rm = int(round(float(r_multiple) * 1000.0)) if r_multiple is not None else None
                    for r in rows:
                        if rm is not None:
                            r.outcome_r_milli = rm
                        if win is not None:
                            r.outcome_win = bool(win)
        except SQLAlchemyError as exc:
            logger.warning("attach_smc_event_outcome SQL: %s", exc)
        except Exception as exc:
            logger.warning("attach_smc_event_outcome fallo: %s", exc)

    # ------------------------------------------------------------------ #
    # Regime episodes
    # ------------------------------------------------------------------ #

    def start_regime_episode(self, *, regime: str, started_at: datetime, symbol: str = "XAUUSD") -> uuid.UUID | None:
        try:
            with new_session() as s:
                with s.begin():
                    row = XAURegimeMemory(
                        symbol=symbol.upper(),
                        regime=regime.upper(),
                        started_at=_utc(started_at),
                    )
                    s.add(row)
                    s.flush()
                    return row.id
        except SQLAlchemyError as exc:
            logger.warning("start_regime_episode SQL: %s", exc)
            return None
        except Exception as exc:
            logger.warning("start_regime_episode fallo: %s", exc)
            return None

    def close_regime_episode(
        self,
        *,
        episode_id: uuid.UUID | str | None,
        ended_at: datetime,
        avg_atr: float | None = None,
        avg_range_pct: float | None = None,
        trades_n: int | None = None,
        trades_wins: int | None = None,
        r_sum: float | None = None,
    ) -> None:
        if episode_id is None:
            return
        try:
            eid = episode_id if isinstance(episode_id, uuid.UUID) else uuid.UUID(str(episode_id))
        except Exception:
            return
        try:
            with new_session() as s:
                with s.begin():
                    row = s.get(XAURegimeMemory, eid)
                    if row is None:
                        return
                    row.ended_at = _utc(ended_at)
                    if row.started_at and row.ended_at:
                        delta = row.ended_at - row.started_at
                        row.duration_minutes = int(delta.total_seconds() // 60)
                    if avg_atr is not None:
                        row.avg_atr = _dec(avg_atr)
                    if avg_range_pct is not None:
                        row.avg_range_pct = _dec(avg_range_pct)
                    if trades_n is not None:
                        row.trades_n = int(trades_n)
                    if trades_wins is not None:
                        row.trades_wins = int(trades_wins)
                    if r_sum is not None:
                        row.sum_r_milli = int(round(float(r_sum) * 1000.0))
        except SQLAlchemyError as exc:
            logger.warning("close_regime_episode SQL: %s", exc)
        except Exception as exc:
            logger.warning("close_regime_episode fallo: %s", exc)

    # ------------------------------------------------------------------ #
    # Session memory (granularidad diaria)
    # ------------------------------------------------------------------ #

    def upsert_session_day(
        self,
        *,
        session: str,
        day: datetime,
        bars_n_delta: int = 0,
        atr_sample: float | None = None,
        range_pct_sample: float | None = None,
        spread_sample: float | None = None,
        trade_outcome: tuple[bool, float] | None = None,  # (win, r_multiple)
        symbol: str = "XAUUSD",
    ) -> None:
        try:
            with new_session() as s:
                with s.begin():
                    day_utc = _utc(day)
                    if day_utc is not None:
                        day_utc = day_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                    row = (
                        s.query(XAUSessionMemory)
                        .filter(XAUSessionMemory.symbol == symbol.upper())
                        .filter(XAUSessionMemory.session == session.upper())
                        .filter(XAUSessionMemory.day == day_utc)
                        .first()
                    )
                    if row is None:
                        row = XAUSessionMemory(
                            symbol=symbol.upper(),
                            session=session.upper(),
                            day=day_utc,
                        )
                        s.add(row)
                    if bars_n_delta:
                        row.bars_n = (row.bars_n or 0) + int(bars_n_delta)
                    # Media incremental simple para muestras (no rigurosa pero suficiente para dashboards).
                    n = max(row.bars_n or 1, 1)
                    if atr_sample is not None:
                        prev = float(row.avg_atr) if row.avg_atr is not None else 0.0
                        row.avg_atr = _dec(((prev * (n - 1)) + float(atr_sample)) / n)
                    if range_pct_sample is not None:
                        prev = float(row.avg_range_pct) if row.avg_range_pct is not None else 0.0
                        row.avg_range_pct = _dec(((prev * (n - 1)) + float(range_pct_sample)) / n)
                    if spread_sample is not None:
                        prev = float(row.avg_spread_pts) if row.avg_spread_pts is not None else 0.0
                        row.avg_spread_pts = _dec(((prev * (n - 1)) + float(spread_sample)) / n)
                    if trade_outcome is not None:
                        win, rm = trade_outcome
                        row.trades_n = (row.trades_n or 0) + 1
                        if win:
                            row.trades_wins = (row.trades_wins or 0) + 1
                        row.sum_r_milli = (row.sum_r_milli or 0) + int(round(float(rm) * 1000.0))
        except SQLAlchemyError as exc:
            logger.warning("upsert_session_day SQL: %s", exc)
        except Exception as exc:
            logger.warning("upsert_session_day fallo: %s", exc)

    # ------------------------------------------------------------------ #
    # Model learning audit
    # ------------------------------------------------------------------ #

    def record_model_retrain(
        self,
        *,
        trigger: str,
        started_at: datetime,
        finished_at: datetime | None,
        samples_used: int | None,
        walk_forward_folds: int | None,
        metrics_pre: Mapping[str, float] | None,
        metrics_post: Mapping[str, float] | None,
        model_path: str | None,
        weights_used: Mapping[str, float] | None,
        notes: str | None = None,
        symbol: str = "XAUUSD",
    ) -> None:
        try:
            pre = metrics_pre or {}
            post = metrics_post or {}
            with new_session() as s:
                with s.begin():
                    row = XAUModelLearning(
                        started_at=_utc(started_at),
                        finished_at=_utc(finished_at) if finished_at else None,
                        symbol=symbol.upper(),
                        trigger=trigger,
                        samples_used=samples_used,
                        walk_forward_folds=walk_forward_folds,
                        pre_winrate=_dec(pre.get("winrate")),
                        post_winrate=_dec(post.get("winrate")),
                        pre_sharpe=_dec(pre.get("sharpe")),
                        post_sharpe=_dec(post.get("sharpe")),
                        pre_expectancy=_dec(pre.get("expectancy")),
                        post_expectancy=_dec(post.get("expectancy")),
                        model_path=model_path,
                        weights_used=dict(weights_used or {}),
                        metrics_extra={"pre": dict(pre), "post": dict(post)},
                        notes=notes,
                    )
                    s.add(row)
        except SQLAlchemyError as exc:
            logger.warning("record_model_retrain SQL: %s", exc)
        except Exception as exc:
            logger.warning("record_model_retrain fallo: %s", exc)

    # ------------------------------------------------------------------ #
    # Knowledge queries (dashboard)
    # ------------------------------------------------------------------ #

    def top_patterns(self, *, symbol: str = "XAUUSD", limit: int = 15, by: str = "winrate") -> list[dict[str, Any]]:
        """``by``: ``winrate`` | ``avg_r`` | ``occurrences``."""
        try:
            with new_session() as s:
                rows = (
                    s.query(XAUPattern)
                    .filter(XAUPattern.symbol == symbol.upper())
                    .filter(XAUPattern.occurrences > 0)
                    .all()
                )
            out: list[dict[str, Any]] = []
            for r in rows:
                total = max((r.wins or 0) + (r.losses or 0), 1)
                winrate = float(r.wins or 0) / float(total)
                avg_r = (r.sum_r_milli or 0) / 1000.0 / max(r.occurrences or 1, 1)
                out.append({
                    "pattern_key": r.pattern_key,
                    "timeframe": r.timeframe,
                    "direction": r.direction,
                    "occurrences": int(r.occurrences or 0),
                    "wins": int(r.wins or 0),
                    "losses": int(r.losses or 0),
                    "winrate": winrate,
                    "avg_r": avg_r,
                    "avg_score": float(r.avg_score) if r.avg_score is not None else None,
                    "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                })
            key = {"winrate": "winrate", "avg_r": "avg_r", "occurrences": "occurrences"}.get(by, "winrate")
            out.sort(key=lambda x: (x[key] or 0.0), reverse=True)
            return out[:limit]
        except Exception as exc:
            logger.warning("top_patterns fallo: %s", exc)
            return []

    def top_smc_events(self, *, symbol: str = "XAUUSD", event_kind: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        try:
            with new_session() as s:
                q = (
                    s.query(
                        XAUSMCEvent.event_kind,
                        XAUSMCEvent.timeframe,
                        XAUSMCEvent.direction,
                        func.count(XAUSMCEvent.id).label("n"),
                        func.avg(XAUSMCEvent.strength).label("avg_strength"),
                        func.avg(XAUSMCEvent.quality).label("avg_quality"),
                        func.sum(XAUSMCEvent.outcome_r_milli).label("sum_r_milli"),
                        func.sum(
                            func.cast(XAUSMCEvent.outcome_win, type_=__import__("sqlalchemy").Integer)
                        ).label("wins"),
                        func.count(XAUSMCEvent.outcome_win).label("with_outcome"),
                    )
                    .filter(XAUSMCEvent.symbol == symbol.upper())
                )
                if event_kind:
                    q = q.filter(XAUSMCEvent.event_kind == event_kind.upper())
                q = q.group_by(XAUSMCEvent.event_kind, XAUSMCEvent.timeframe, XAUSMCEvent.direction)
                rows = q.all()
            out: list[dict[str, Any]] = []
            for r in rows:
                wins = int(r.wins or 0)
                with_outcome = int(r.with_outcome or 0)
                winrate = (wins / with_outcome) if with_outcome > 0 else None
                avg_r = ((r.sum_r_milli or 0) / 1000.0 / with_outcome) if with_outcome > 0 else None
                out.append({
                    "event_kind": r.event_kind,
                    "timeframe": r.timeframe,
                    "direction": r.direction,
                    "n": int(r.n or 0),
                    "avg_strength": float(r.avg_strength) if r.avg_strength is not None else None,
                    "avg_quality": float(r.avg_quality) if r.avg_quality is not None else None,
                    "winrate": winrate,
                    "avg_r": avg_r,
                    "with_outcome": with_outcome,
                })
            out.sort(key=lambda x: (x["n"] or 0), reverse=True)
            return out[:limit]
        except Exception as exc:
            logger.warning("top_smc_events fallo: %s", exc)
            return []

    def session_summary(self, *, symbol: str = "XAUUSD", days_lookback: int = 60) -> list[dict[str, Any]]:
        try:
            since = datetime.now(timezone.utc) - timedelta(days=days_lookback)
            with new_session() as s:
                rows = (
                    s.query(XAUSessionMemory)
                    .filter(XAUSessionMemory.symbol == symbol.upper())
                    .filter(XAUSessionMemory.day >= since)
                    .all()
                )
            # Agrega por session
            agg: dict[str, dict[str, float]] = {}
            for r in rows:
                a = agg.setdefault(r.session, {
                    "bars_n": 0, "trades_n": 0, "wins": 0, "r_sum_milli": 0,
                    "atr_sum": 0.0, "range_sum": 0.0, "spread_sum": 0.0, "days_n": 0,
                })
                a["bars_n"] += int(r.bars_n or 0)
                a["trades_n"] += int(r.trades_n or 0)
                a["wins"] += int(r.trades_wins or 0)
                a["r_sum_milli"] += int(r.sum_r_milli or 0)
                a["atr_sum"] += float(r.avg_atr) if r.avg_atr is not None else 0.0
                a["range_sum"] += float(r.avg_range_pct) if r.avg_range_pct is not None else 0.0
                a["spread_sum"] += float(r.avg_spread_pts) if r.avg_spread_pts is not None else 0.0
                a["days_n"] += 1
            out: list[dict[str, Any]] = []
            for sess, a in agg.items():
                t_n = a["trades_n"] or 1
                d_n = a["days_n"] or 1
                out.append({
                    "session": sess,
                    "days_n": a["days_n"],
                    "bars_n": a["bars_n"],
                    "trades_n": a["trades_n"],
                    "winrate": a["wins"] / t_n,
                    "avg_r": a["r_sum_milli"] / 1000.0 / t_n,
                    "avg_atr": a["atr_sum"] / d_n,
                    "avg_range_pct": a["range_sum"] / d_n,
                    "avg_spread_pts": a["spread_sum"] / d_n,
                })
            out.sort(key=lambda x: x["avg_r"], reverse=True)
            return out
        except Exception as exc:
            logger.warning("session_summary fallo: %s", exc)
            return []

    def regime_summary(self, *, symbol: str = "XAUUSD") -> list[dict[str, Any]]:
        try:
            with new_session() as s:
                rows = (
                    s.query(XAURegimeMemory)
                    .filter(XAURegimeMemory.symbol == symbol.upper())
                    .all()
                )
            agg: dict[str, dict[str, float]] = {}
            for r in rows:
                a = agg.setdefault(r.regime, {
                    "episodes": 0, "minutes": 0,
                    "trades_n": 0, "wins": 0, "r_sum_milli": 0,
                })
                a["episodes"] += 1
                a["minutes"] += int(r.duration_minutes or 0)
                a["trades_n"] += int(r.trades_n or 0)
                a["wins"] += int(r.trades_wins or 0)
                a["r_sum_milli"] += int(r.sum_r_milli or 0)
            out = []
            for reg, a in agg.items():
                t_n = a["trades_n"] or 1
                out.append({
                    "regime": reg,
                    "episodes": a["episodes"],
                    "total_minutes": a["minutes"],
                    "avg_duration_min": a["minutes"] / max(a["episodes"], 1),
                    "trades_n": a["trades_n"],
                    "winrate": a["wins"] / t_n,
                    "avg_r": a["r_sum_milli"] / 1000.0 / t_n,
                })
            out.sort(key=lambda x: x["avg_r"], reverse=True)
            return out
        except Exception as exc:
            logger.warning("regime_summary fallo: %s", exc)
            return []

    def recent_retrains(self, *, limit: int = 20, symbol: str = "XAUUSD") -> list[dict[str, Any]]:
        try:
            with new_session() as s:
                rows = (
                    s.query(XAUModelLearning)
                    .filter(XAUModelLearning.symbol == symbol.upper())
                    .order_by(XAUModelLearning.started_at.desc())
                    .limit(limit)
                    .all()
                )
            out: list[dict[str, Any]] = []
            for r in rows:
                out.append({
                    "id": str(r.id),
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                    "trigger": r.trigger,
                    "samples_used": r.samples_used,
                    "walk_forward_folds": r.walk_forward_folds,
                    "pre_winrate": float(r.pre_winrate) if r.pre_winrate is not None else None,
                    "post_winrate": float(r.post_winrate) if r.post_winrate is not None else None,
                    "pre_sharpe": float(r.pre_sharpe) if r.pre_sharpe is not None else None,
                    "post_sharpe": float(r.post_sharpe) if r.post_sharpe is not None else None,
                    "pre_expectancy": float(r.pre_expectancy) if r.pre_expectancy is not None else None,
                    "post_expectancy": float(r.post_expectancy) if r.post_expectancy is not None else None,
                    "model_path": r.model_path,
                    "weights_used": r.weights_used,
                    "notes": r.notes,
                })
            return out
        except Exception as exc:
            logger.warning("recent_retrains fallo: %s", exc)
            return []

"""Repositorio SMC — persistencia PostgreSQL para todos los elementos SMC.

Diseño:

* Inserts batch por snapshot (un commit por símbolo/momento).
* Idempotente: cada llamada inserta un snapshot nuevo; el dashboard consulta
  por ``symbol`` y ``MAX(timestamp)`` para mostrar el estado vigente.
* Nunca lanza excepciones al caller (el SMC es opcional, no debe romper el
  flujo del Orchestrator). Errores se loguean y silencian.
* ``link_trade_to_smc(...)`` + ``update_trade_outcome(...)`` son los hooks
  on-open / on-close que el Orchestrator usará para self-learning.

Las consultas para el learner (winrate por OB-quality bucket, etc.) están en
``smc_self_learning.py``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from .db_connection import new_session
from .logger import get_logger
from .smc_models import (
    SMCBOS,
    SMCCHOCH,
    SMCConfluence,
    SMCFVG,
    SMCLiquidity,
    SMCOrderBlock,
    SMCTradeLink,
)

logger = get_logger(__name__)


def _dec(v: Any) -> Decimal | None:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None


def _parse_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


class SMCRepository:
    """Repositorio dedicado para tablas ``smc_*``.

    Métodos públicos:

    * ``persist_snapshot(snapshot)`` — guarda BOS/CHOCH/OB/FVG/Liquidity/
      Confluence del último análisis del símbolo.
    * ``link_trade_to_smc(...)`` — guarda elementos usados al abrir un trade.
    * ``update_trade_outcome(...)`` — actualiza R/MFE/MAE al cierre.
    * ``recent_for_dashboard(symbol)`` — consulta para visualización.
    """

    # ------------------------------------------------------------------ #
    # Snapshot persistence
    # ------------------------------------------------------------------ #

    def persist_snapshot(self, snapshot: Any) -> None:
        """Guarda todos los elementos SMC del snapshot en una sola transacción.

        Args:
            snapshot: ``market_structure.types.SMCSnapshot``. Se accede a
                ``per_tf``, ``confluence`` y campos macro/inst/op de forma
                duck-typed para no acoplar el repo al módulo SMC en imports.
        """
        try:
            symbol = str(getattr(snapshot, "symbol", "") or "").upper()
            if not symbol:
                return
            now = getattr(snapshot, "generated_at", None) or datetime.utcnow()
            with new_session() as session:
                with session.begin():
                    for tf, ctx in dict(getattr(snapshot, "per_tf", {}) or {}).items():
                        self._persist_bos(session, symbol, tf, now, ctx)
                        self._persist_choch(session, symbol, tf, now, ctx)
                        self._persist_obs(session, symbol, tf, now, ctx)
                        self._persist_fvgs(session, symbol, tf, now, ctx)
                        self._persist_liquidity(session, symbol, tf, now, ctx)
                    self._persist_confluence(session, snapshot)
        except SQLAlchemyError as exc:
            logger.warning("SMCRepository.persist_snapshot SQL error: %s", exc)
        except Exception as exc:
            logger.warning("SMCRepository.persist_snapshot unexpected: %s", exc)

    def _persist_bos(self, session, symbol: str, tf: str, now: datetime, ctx: Any) -> None:
        bos = getattr(ctx, "last_bos", None)
        if bos is None:
            return
        row = SMCBOS(
            timestamp=now,
            symbol=symbol,
            timeframe=tf,
            direction=str(bos.direction.value if hasattr(bos.direction, "value") else bos.direction),
            strength=str(bos.strength.value if hasattr(bos.strength, "value") else bos.strength),
            broken_swing_price=float(bos.broken_swing_price),
            broken_swing_time=bos.broken_swing_time,
            break_time=bos.break_time,
            break_price=float(bos.break_price),
            bos_strength=_dec(bos.bos_strength),
            bos_quality=_dec(bos.bos_quality),
            bos_reliability=_dec(bos.bos_reliability),
            bos_age_bars=int(bos.bos_age_bars),
            bos_distance_pips=_dec(bos.bos_distance_pips),
            extra=dict(getattr(bos, "extra", {}) or {}),
        )
        session.add(row)

    def _persist_choch(self, session, symbol: str, tf: str, now: datetime, ctx: Any) -> None:
        choch = getattr(ctx, "last_choch", None)
        if choch is None:
            return
        row = SMCCHOCH(
            timestamp=now,
            symbol=symbol,
            timeframe=tf,
            direction=str(choch.direction.value if hasattr(choch.direction, "value") else choch.direction),
            kind=str(choch.kind.value if hasattr(choch.kind, "value") else choch.kind),
            broken_swing_price=float(choch.broken_swing_price),
            broken_swing_time=choch.broken_swing_time,
            break_time=choch.break_time,
            break_price=float(choch.break_price),
            choch_strength=_dec(choch.choch_strength),
            choch_quality=_dec(choch.choch_quality),
            choch_reliability=_dec(choch.choch_reliability),
            choch_score=_dec(choch.choch_score),
            choch_age_bars=int(choch.choch_age_bars),
            extra=dict(getattr(choch, "extra", {}) or {}),
        )
        session.add(row)

    def _persist_obs(self, session, symbol: str, tf: str, now: datetime, ctx: Any) -> None:
        for ob in (getattr(ctx, "active_obs", None) or []):
            row = SMCOrderBlock(
                timestamp=now,
                symbol=symbol,
                timeframe=tf,
                direction=str(ob.direction.value if hasattr(ob.direction, "value") else ob.direction),
                top=float(ob.top),
                bottom=float(ob.bottom),
                open_time=ob.open_time,
                state=str(ob.state.value if hasattr(ob.state, "value") else ob.state),
                ob_strength=_dec(ob.ob_strength),
                ob_volume_score=_dec(ob.ob_volume_score),
                ob_displacement_score=_dec(ob.ob_displacement_score),
                ob_reaction_score=_dec(ob.ob_reaction_score),
                ob_freshness=_dec(ob.ob_freshness),
                ob_quality=_dec(ob.ob_quality),
                ob_age_bars=int(ob.ob_age_bars),
                mitigation_time=ob.mitigation_time,
                extra=dict(getattr(ob, "extra", {}) or {}),
            )
            session.add(row)

    def _persist_fvgs(self, session, symbol: str, tf: str, now: datetime, ctx: Any) -> None:
        for fvg in (getattr(ctx, "active_fvgs", None) or []):
            row = SMCFVG(
                timestamp=now,
                symbol=symbol,
                timeframe=tf,
                direction=str(fvg.direction.value if hasattr(fvg.direction, "value") else fvg.direction),
                top=float(fvg.top),
                bottom=float(fvg.bottom),
                open_time=fvg.open_time,
                state=str(fvg.state.value if hasattr(fvg.state, "value") else fvg.state),
                gap_size=_dec(fvg.gap_size),
                gap_size_pips=_dec(fvg.gap_size_pips),
                gap_quality=_dec(fvg.gap_quality),
                gap_age_bars=int(fvg.gap_age_bars),
                gap_fill_percent=_dec(fvg.gap_fill_percent),
                fvg_strength=_dec(fvg.fvg_strength),
                inverted_at=fvg.inverted_at,
                extra=dict(getattr(fvg, "extra", {}) or {}),
            )
            session.add(row)

    def _persist_liquidity(self, session, symbol: str, tf: str, now: datetime, ctx: Any) -> None:
        for pool in (getattr(ctx, "liquidity_pools", None) or []):
            row = SMCLiquidity(
                timestamp=now,
                symbol=symbol,
                timeframe=tf,
                kind=str(pool.kind.value if hasattr(pool.kind, "value") else pool.kind),
                price=float(pool.price),
                bound_low=float(pool.bound_low),
                bound_high=float(pool.bound_high),
                open_time=pool.open_time,
                touches=int(pool.touches),
                density=_dec(pool.density),
                liquidity_score=_dec(pool.liquidity_score),
                stop_hunt_probability=_dec(pool.stop_hunt_probability),
                swept=bool(pool.swept),
                swept_at=pool.swept_at,
                sweep_kind=str(pool.sweep_kind.value) if pool.sweep_kind is not None else None,
                extra=dict(getattr(pool, "extra", {}) or {}),
            )
            session.add(row)

    def _persist_confluence(self, session, snapshot: Any) -> None:
        c = getattr(snapshot, "confluence", None)
        if c is None:
            return
        row = SMCConfluence(
            timestamp=getattr(snapshot, "generated_at", datetime.utcnow()),
            symbol=str(getattr(snapshot, "symbol", "")).upper(),
            direction_bias=str(c.direction_bias.value if hasattr(c.direction_bias, "value") else c.direction_bias),
            confluence_score=_dec(c.confluence_score),
            bos_component=_dec(c.bos_component),
            choch_component=_dec(c.choch_component),
            ob_component=_dec(c.ob_component),
            fvg_component=_dec(c.fvg_component),
            liquidity_component=_dec(c.liquidity_component),
            macro_bias=str(snapshot.macro_bias.value if hasattr(snapshot.macro_bias, "value") else snapshot.macro_bias),
            institutional_bias=str(snapshot.institutional_bias.value if hasattr(snapshot.institutional_bias, "value") else snapshot.institutional_bias),
            operational_bias=str(snapshot.operational_bias.value if hasattr(snapshot.operational_bias, "value") else snapshot.operational_bias),
            per_tf=dict(c.per_tf),
            weights=dict(c.weights),
            extra={
                "notes": list(c.notes),
                "primary_timeframe": getattr(snapshot, "primary_timeframe", "M15"),
            },
        )
        session.add(row)

    # ------------------------------------------------------------------ #
    # Trade link
    # ------------------------------------------------------------------ #

    def link_trade_to_smc(
        self,
        *,
        trade_id: str | uuid.UUID | None,
        symbol: str,
        direction: str,
        open_time: datetime,
        snapshot: Any,
        features_smc: dict[str, Any] | None = None,
        used_elements: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> uuid.UUID | None:
        try:
            tid: uuid.UUID | None = None
            if isinstance(trade_id, uuid.UUID):
                tid = trade_id
            elif isinstance(trade_id, str) and trade_id:
                try:
                    tid = uuid.UUID(trade_id)
                except Exception:
                    tid = None
            c = getattr(snapshot, "confluence", None)
            link = SMCTradeLink(
                trade_id=tid,
                symbol=symbol.upper(),
                open_time=open_time,
                direction=direction.upper(),
                primary_timeframe=getattr(snapshot, "primary_timeframe", "M15"),
                confluence_score=_dec(c.confluence_score) if c is not None else None,
                smc_macro_bias=str(snapshot.macro_bias.value if hasattr(snapshot.macro_bias, "value") else snapshot.macro_bias),
                smc_inst_bias=str(snapshot.institutional_bias.value if hasattr(snapshot.institutional_bias, "value") else snapshot.institutional_bias),
                smc_op_bias=str(snapshot.operational_bias.value if hasattr(snapshot.operational_bias, "value") else snapshot.operational_bias),
                features_smc=dict(features_smc or {}),
                used_elements=dict(used_elements or {}),
                extra=dict(extra or {}),
            )
            with new_session() as session:
                with session.begin():
                    session.add(link)
                    session.flush()
                    return link.id
        except SQLAlchemyError as exc:
            logger.warning("SMCRepository.link_trade_to_smc SQL error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("SMCRepository.link_trade_to_smc unexpected: %s", exc)
            return None

    def update_trade_outcome(
        self,
        *,
        link_id: uuid.UUID | str | None = None,
        trade_id: uuid.UUID | str | None = None,
        close_time: datetime | None = None,
        r_multiple: float | None = None,
        mfe_r: float | None = None,
        mae_r: float | None = None,
        win: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Cierra el ciclo: actualiza outcome del trade en ``smc_trade_link``.

        Si ``link_id`` se proporciona se usa esa fila directamente. En su
        defecto se busca por ``trade_id``. Si ninguno existe, no hace nada.
        """
        try:
            link_uuid: uuid.UUID | None = None
            if isinstance(link_id, uuid.UUID):
                link_uuid = link_id
            elif isinstance(link_id, str) and link_id:
                try:
                    link_uuid = uuid.UUID(link_id)
                except Exception:
                    link_uuid = None
            trade_uuid: uuid.UUID | None = None
            if isinstance(trade_id, uuid.UUID):
                trade_uuid = trade_id
            elif isinstance(trade_id, str) and trade_id:
                try:
                    trade_uuid = uuid.UUID(trade_id)
                except Exception:
                    trade_uuid = None
            if link_uuid is None and trade_uuid is None:
                return
            with new_session() as session:
                with session.begin():
                    if link_uuid is not None:
                        link = session.get(SMCTradeLink, link_uuid)
                    else:
                        link = (
                            session.query(SMCTradeLink)
                            .filter(SMCTradeLink.trade_id == trade_uuid)
                            .order_by(SMCTradeLink.open_time.desc())
                            .first()
                        )
                    if link is None:
                        return
                    if close_time is not None:
                        link.close_time = close_time
                    if r_multiple is not None:
                        link.r_multiple = _dec(r_multiple)
                    if mfe_r is not None:
                        link.mfe_r = _dec(mfe_r)
                    if mae_r is not None:
                        link.mae_r = _dec(mae_r)
                    if win is not None:
                        link.win = bool(win)
                    if extra:
                        merged = dict(link.extra or {})
                        merged.update(extra)
                        link.extra = merged
        except SQLAlchemyError as exc:
            logger.warning("SMCRepository.update_trade_outcome SQL error: %s", exc)
        except Exception as exc:
            logger.warning("SMCRepository.update_trade_outcome unexpected: %s", exc)

    # ------------------------------------------------------------------ #
    # Read for dashboard / learner
    # ------------------------------------------------------------------ #

    def recent_for_dashboard(self, symbol: str, *, limit_per_tf: int = 30) -> dict[str, Any]:
        """Devuelve el snapshot SMC más reciente del símbolo (para /api/smc/<sym>)."""
        sym = symbol.upper()
        out: dict[str, Any] = {
            "symbol": sym,
            "bos": [],
            "choch": [],
            "order_blocks": [],
            "fvg": [],
            "liquidity": [],
            "confluence": None,
        }
        try:
            with new_session() as session:
                bos_rows = (
                    session.query(SMCBOS)
                    .filter(SMCBOS.symbol == sym)
                    .order_by(SMCBOS.timestamp.desc())
                    .limit(limit_per_tf)
                    .all()
                )
                out["bos"] = [self._row_to_dict(r) for r in bos_rows]
                choch_rows = (
                    session.query(SMCCHOCH)
                    .filter(SMCCHOCH.symbol == sym)
                    .order_by(SMCCHOCH.timestamp.desc())
                    .limit(limit_per_tf)
                    .all()
                )
                out["choch"] = [self._row_to_dict(r) for r in choch_rows]
                ob_rows = (
                    session.query(SMCOrderBlock)
                    .filter(SMCOrderBlock.symbol == sym)
                    .order_by(SMCOrderBlock.timestamp.desc())
                    .limit(limit_per_tf)
                    .all()
                )
                out["order_blocks"] = [self._row_to_dict(r) for r in ob_rows]
                fvg_rows = (
                    session.query(SMCFVG)
                    .filter(SMCFVG.symbol == sym)
                    .order_by(SMCFVG.timestamp.desc())
                    .limit(limit_per_tf)
                    .all()
                )
                out["fvg"] = [self._row_to_dict(r) for r in fvg_rows]
                liq_rows = (
                    session.query(SMCLiquidity)
                    .filter(SMCLiquidity.symbol == sym)
                    .order_by(SMCLiquidity.timestamp.desc())
                    .limit(limit_per_tf)
                    .all()
                )
                out["liquidity"] = [self._row_to_dict(r) for r in liq_rows]
                conf_row = (
                    session.query(SMCConfluence)
                    .filter(SMCConfluence.symbol == sym)
                    .order_by(SMCConfluence.timestamp.desc())
                    .first()
                )
                if conf_row is not None:
                    out["confluence"] = self._row_to_dict(conf_row)
        except SQLAlchemyError as exc:
            logger.warning("SMCRepository.recent_for_dashboard SQL error: %s", exc)
        except Exception as exc:
            logger.warning("SMCRepository.recent_for_dashboard unexpected: %s", exc)
        return out

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for col in row.__table__.columns.keys():  # type: ignore[union-attr]
            v = getattr(row, col, None)
            if isinstance(v, Decimal):
                v = float(v)
            elif isinstance(v, uuid.UUID):
                v = str(v)
            elif isinstance(v, datetime):
                v = v.isoformat()
            out[col] = v
        return out

    # ------------------------------------------------------------------ #
    # Learning aggregates (used by ``smc_self_learning``)
    # ------------------------------------------------------------------ #

    def learning_aggregates(self, symbol: str, *, lookback_trades: int = 400) -> dict[str, Any]:
        """Agrega métricas por TF / componente sobre los últimos N trades vinculados.

        Devuelve algo como::

            {
              "by_timeframe": {"M15": {"n": 12, "winrate": 0.41, "avg_r": 0.27}, ...},
              "by_confluence_bucket": {"low": {...}, "mid": {...}, "high": {...}},
              "total": 73
            }
        """
        out: dict[str, Any] = {"by_timeframe": {}, "by_confluence_bucket": {}, "total": 0}
        sym = symbol.upper()
        try:
            with new_session() as session:
                rows = (
                    session.query(SMCTradeLink)
                    .filter(SMCTradeLink.symbol == sym)
                    .filter(SMCTradeLink.r_multiple.isnot(None))
                    .order_by(SMCTradeLink.open_time.desc())
                    .limit(lookback_trades)
                    .all()
                )
                total = len(rows)
                out["total"] = total
                if total == 0:
                    return out

                by_tf: dict[str, dict[str, float]] = {}
                by_bucket: dict[str, dict[str, float]] = {
                    "low": {"n": 0.0, "wins": 0.0, "rsum": 0.0},
                    "mid": {"n": 0.0, "wins": 0.0, "rsum": 0.0},
                    "high": {"n": 0.0, "wins": 0.0, "rsum": 0.0},
                }
                for r in rows:
                    tf = r.primary_timeframe or "M15"
                    bucket = by_tf.setdefault(tf, {"n": 0.0, "wins": 0.0, "rsum": 0.0})
                    bucket["n"] += 1.0
                    rm = float(r.r_multiple) if r.r_multiple is not None else 0.0
                    bucket["rsum"] += rm
                    if r.win:
                        bucket["wins"] += 1.0
                    # Confluence buckets
                    cs = float(r.confluence_score) if r.confluence_score is not None else 0.0
                    if cs >= 0.65:
                        cb = "high"
                    elif cs >= 0.40:
                        cb = "mid"
                    else:
                        cb = "low"
                    by_bucket[cb]["n"] += 1.0
                    by_bucket[cb]["rsum"] += rm
                    if r.win:
                        by_bucket[cb]["wins"] += 1.0
                out["by_timeframe"] = {
                    tf: {
                        "n": int(v["n"]),
                        "winrate": v["wins"] / max(v["n"], 1.0),
                        "avg_r": v["rsum"] / max(v["n"], 1.0),
                    }
                    for tf, v in by_tf.items()
                }
                out["by_confluence_bucket"] = {
                    cb: {
                        "n": int(v["n"]),
                        "winrate": v["wins"] / max(v["n"], 1.0),
                        "avg_r": v["rsum"] / max(v["n"], 1.0),
                    }
                    for cb, v in by_bucket.items()
                }
        except SQLAlchemyError as exc:
            logger.warning("SMCRepository.learning_aggregates SQL error: %s", exc)
        except Exception as exc:
            logger.warning("SMCRepository.learning_aggregates unexpected: %s", exc)
        return out

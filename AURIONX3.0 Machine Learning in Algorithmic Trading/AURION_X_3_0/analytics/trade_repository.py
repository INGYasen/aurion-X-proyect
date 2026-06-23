"""Repositorio de acceso a datos para operaciones (ORM)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, cast

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from .db_connection import DatabaseConnection, get_engine, new_session
from .logger import get_logger
from .models import Direction, StrategyTag, Trade

logger = get_logger(__name__)

_REQUIRED_OPEN_KEYS = frozenset(
    {"symbol", "direction", "lot", "entry_price", "open_time", "strategy"},
)


class TradeRepository:
    """Persistencia de trades con transacciones explícitas vía ``session.begin()``."""

    def __init__(self, connection: DatabaseConnection | None = None) -> None:
        """Opcionalmente asocia un ``DatabaseConnection`` para fijar DSN antes de operar.

        Args:
            connection: Si se indica, se llama a ``connect()`` para materializar el motor.
        """
        self._connection = connection
        logger.debug(
            "TradeRepository creado (connection=%s)",
            connection is not None,
        )

    def _ensure_engine(self) -> None:
        if self._connection is not None:
            self._connection.connect()
        else:
            get_engine()

    @staticmethod
    def _to_decimal(value: Any, field: str) -> Decimal:
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"Campo numérico inválido '{field}': {value!r}") from exc

    @staticmethod
    def _parse_direction(raw: Any) -> Direction:
        if isinstance(raw, Direction):
            return raw
        if isinstance(raw, str):
            upper = raw.strip().upper()
            alias = {
                "BUY": Direction.LONG,
                "SELL": Direction.SHORT,
                "LONG": Direction.LONG,
                "SHORT": Direction.SHORT,
            }
            if upper in alias:
                return alias[upper]
            try:
                return Direction(upper)
            except ValueError as exc:
                raise ValueError(f"Dirección no válida: {raw!r}") from exc
        raise ValueError(f"Dirección no válida: {raw!r}")

    @staticmethod
    def _parse_strategy(raw: Any) -> StrategyTag:
        if isinstance(raw, StrategyTag):
            return raw
        if isinstance(raw, str):
            upper = raw.strip().upper()
            try:
                return StrategyTag(upper)
            except ValueError as exc:
                raise ValueError(f"Estrategia no válida: {raw!r}") from exc
        raise ValueError(f"Estrategia no válida: {raw!r}")

    @staticmethod
    def _ensure_aware_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _trade_to_public_dict(row: Trade) -> dict[str, Any]:
        return {
            "id": str(row.id),
            "symbol": row.symbol,
            "direction": row.direction.value if isinstance(row.direction, Direction) else row.direction,
            "lot": row.lot,
            "entry_price": row.entry_price,
            "exit_price": row.exit_price,
            "sl": row.sl,
            "tp": row.tp,
            "profit": row.profit,
            "open_time": row.open_time,
            "close_time": row.close_time,
            "r_multiple": row.r_multiple,
            "strategy": row.strategy.value if isinstance(row.strategy, StrategyTag) else row.strategy,
            "ml_confidence": row.ml_confidence,
            "features_json": row.features_json,
        }

    @staticmethod
    def _decimal_to_jsonable(value: Decimal | None) -> float | None:
        if value is None:
            return None
        return float(value)

    def insert_trade_open(self, data: Mapping[str, Any]) -> str:
        """Inserta un trade en apertura (sin cierre ni profit).

        Args:
            data: Debe incluir ``symbol``, ``direction``, ``lot``, ``entry_price``,
                ``open_time``, ``strategy``. Opcionales: ``id``, ``sl``, ``tp``,
                ``ml_confidence``, ``features_json``.

        Returns:
            UUID del trade (string).

        Raises:
            ValueError: datos incompletos o tipos inválidos.
            SQLAlchemyError: error de persistencia (tras log).
        """
        self._ensure_engine()
        missing = _REQUIRED_OPEN_KEYS - frozenset(data.keys())
        if missing:
            raise ValueError(f"Faltan claves obligatorias en data: {sorted(missing)}")

        trade_id: uuid.UUID | None = None
        if "id" in data and data["id"] is not None:
            trade_id = uuid.UUID(str(data["id"]))

        row = Trade(
            id=trade_id if trade_id is not None else uuid.uuid4(),
            symbol=str(data["symbol"]).strip(),
            direction=self._parse_direction(data["direction"]),
            lot=self._to_decimal(data["lot"], "lot"),
            entry_price=self._to_decimal(data["entry_price"], "entry_price"),
            exit_price=None,
            sl=self._to_decimal(data["sl"], "sl") if data.get("sl") is not None else None,
            tp=self._to_decimal(data["tp"], "tp") if data.get("tp") is not None else None,
            profit=None,
            open_time=self._ensure_aware_utc(cast(datetime, data["open_time"])),
            close_time=None,
            r_multiple=None,
            strategy=self._parse_strategy(data["strategy"]),
            ml_confidence=self._to_decimal(data["ml_confidence"], "ml_confidence")
            if data.get("ml_confidence") is not None
            else None,
            features_json=data.get("features_json"),
        )

        session = new_session()
        try:
            with session.begin():
                session.add(row)
                session.flush()
                out_id = str(row.id)
            return out_id
        except SQLAlchemyError:
            logger.exception("insert_trade_open: fallo SQLAlchemy")
            raise
        except Exception:
            logger.exception("insert_trade_open: error inesperado")
            raise
        finally:
            session.close()

    def update_trade_close(
        self,
        trade_id: str | uuid.UUID,
        exit_price: Decimal | float | str,
        profit: Decimal | float | str,
        r_multiple: Decimal | float | str | None,
        close_reason: str | None = None,
    ) -> None:
        """Actualiza precio de salida, profit, múltiplo R y marca ``close_time`` (UTC).

        Args:
            close_reason: AURION X 4.3 - razón de cierre para trade journal.
                Valores: 'tp_hit', 'sl_hit', 'breakeven', 'profit_lock_soft',
                'profit_lock_hard', 'trailing_stop', 'secure_profit', 'manual_close',
                'time_exit', 'reversal_signal', 'partial_close', 'broker_reject', 'unknown'.

        Raises:
            ValueError: si el trade no existe.
            SQLAlchemyError: error de persistencia (tras log).
        """
        self._ensure_engine()
        pk = trade_id if isinstance(trade_id, uuid.UUID) else uuid.UUID(str(trade_id))
        exit_d = self._to_decimal(exit_price, "exit_price")
        profit_d = self._to_decimal(profit, "profit")
        r_d = self._to_decimal(r_multiple, "r_multiple") if r_multiple is not None else None
        closed_at = datetime.now(timezone.utc)
        valid_reasons = {
            "tp_hit", "sl_hit", "breakeven", "profit_lock_soft", "profit_lock_hard",
            "trailing_stop", "secure_profit", "manual_close", "time_exit",
            "reversal_signal", "partial_close", "broker_reject", "unknown",
        }
        reason_norm = None
        if close_reason is not None:
            r_clean = str(close_reason).strip().lower()
            reason_norm = r_clean if r_clean in valid_reasons else "unknown"

        session = new_session()
        try:
            with session.begin():
                row = session.get(Trade, pk)
                if row is None:
                    raise ValueError(f"No existe trade con id={pk}")
                row.exit_price = exit_d
                row.profit = profit_d
                row.r_multiple = r_d
                row.close_time = closed_at
                if reason_norm is not None:
                    row.close_reason = reason_norm
        except ValueError:
            logger.warning("update_trade_close: trade no encontrado id=%s", pk)
            raise
        except SQLAlchemyError:
            logger.exception("update_trade_close: fallo SQLAlchemy id=%s", pk)
            raise
        except Exception:
            logger.exception("update_trade_close: error inesperado id=%s", pk)
            raise
        finally:
            session.close()

    def get_last_trades(self, limit: int) -> list[dict[str, Any]]:
        """Últimos trades por ``open_time`` descendente.

        Raises:
            SQLAlchemyError: error de consulta (tras log).
        """
        self._ensure_engine()
        if limit < 1:
            raise ValueError("limit debe ser >= 1")

        session = new_session()
        try:
            with session.begin():
                stmt = (
                    select(Trade)
                    .order_by(Trade.open_time.desc())
                    .limit(limit)
                )
                rows = list(session.scalars(stmt).all())
            return [self._trade_to_public_dict(r) for r in rows]
        except SQLAlchemyError:
            logger.exception("get_last_trades: fallo SQLAlchemy limit=%s", limit)
            raise
        except Exception:
            logger.exception("get_last_trades: error inesperado limit=%s", limit)
            raise
        finally:
            session.close()

    def count_closed_trades(self, symbol: str | None = None) -> int:
        """Número de trades con ``close_time`` (para disparar reentrenamiento)."""
        self._ensure_engine()
        session = new_session()
        try:
            with session.begin():
                stmt = select(func.count()).select_from(Trade).where(Trade.close_time.is_not(None))
                if symbol is not None and str(symbol).strip():
                    stmt = stmt.where(Trade.symbol == str(symbol).strip().upper())
                return int(session.scalar(stmt) or 0)
        except SQLAlchemyError:
            logger.exception("count_closed_trades: fallo SQLAlchemy")
            raise
        finally:
            session.close()

    def get_training_dataset(
        self,
        limit: int,
        symbol: str | None = None,
        *,
        exclude_zombies: bool = True,
        min_duration_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        """Trades cerrados ordenados por ``close_time`` descendente (filas para ML).

        Cada elemento incluye identificadores, targets numéricos y ``features_json``
        en tipos serializables (``float`` donde aplica).

        Args:
            limit: Maximo de filas.
            symbol: Si se indica, solo trades de ese simbolo (mayusculas/minusculas normalizadas).
            exclude_zombies: Si True (default), excluye trades con SL/TP buggeados o
                cierre instantaneo del broker. Aplica los siguientes filtros:
                - duracion (close_time - open_time) < min_duration_seconds (cierre instantaneo)
                - profit=0 AND r_multiple=0 AND duracion < 7200s (timeouts y rechazos)
                - SL/TP absurdos: max(|sl-entry|, |tp-entry|) / entry > 5% (eran ~50%+ en zombies)
                Estos trades son artefactos de bugs y NO deben entrenar al modelo.
            min_duration_seconds: Umbral de duracion minima para no marcar zombie.

        Raises:
            SQLAlchemyError: error de consulta (tras log).
        """
        self._ensure_engine()
        if limit < 1:
            raise ValueError("limit debe ser >= 1")

        session = new_session()
        try:
            with session.begin():
                stmt = select(Trade).where(Trade.close_time.is_not(None))
                if symbol is not None and str(symbol).strip():
                    stmt = stmt.where(Trade.symbol == str(symbol).strip().upper())
                stmt = stmt.order_by(Trade.close_time.desc()).limit(limit * 2 if exclude_zombies else limit)
                rows = list(session.scalars(stmt).all())
            out: list[dict[str, Any]] = []
            zombies_filtered = 0
            for r in rows:
                if exclude_zombies and self._is_zombie_trade(r, min_duration_seconds):
                    zombies_filtered += 1
                    continue
                out.append(
                    {
                        "id": str(r.id),
                        "symbol": r.symbol,
                        "direction": r.direction.value,
                        "strategy": r.strategy.value,
                        "lot": self._decimal_to_jsonable(r.lot),
                        "entry_price": self._decimal_to_jsonable(r.entry_price),
                        "exit_price": self._decimal_to_jsonable(r.exit_price),
                        "profit": self._decimal_to_jsonable(r.profit),
                        "r_multiple": self._decimal_to_jsonable(r.r_multiple),
                        "ml_confidence": self._decimal_to_jsonable(r.ml_confidence),
                        "open_time": r.open_time.isoformat() if r.open_time else None,
                        "close_time": r.close_time.isoformat() if r.close_time else None,
                        "features_json": r.features_json,
                    },
                )
                if len(out) >= limit:
                    break
            if zombies_filtered:
                logger.info(
                    "get_training_dataset: %s zombies excluidos del entrenamiento (sym=%s)",
                    zombies_filtered,
                    symbol or "ALL",
                )
            return out
        except SQLAlchemyError:
            logger.exception("get_training_dataset: fallo SQLAlchemy limit=%s", limit)
            raise
        except Exception:
            logger.exception("get_training_dataset: error inesperado limit=%s", limit)
            raise
        finally:
            session.close()

    @staticmethod
    def _is_zombie_trade(trade: Trade, min_duration_seconds: int) -> bool:
        """Detecta trades con SL/TP buggeados o cierre instantaneo del broker.

        AURION X 4.3 (7-jun-2026): post-mortem identificó 59 trades XAUUSD con
        ml_conf alto (>=0.70) que tenian profit=0 y R=0 por SL/TP absurdos
        (ej. SL=6764 con entry=4555, TP negativo, etc) generados antes del fix
        de ``max_sl_price_pct``. Estos contaminan el dataset si se usan para
        entrenar (el modelo aprende a confiar mucho en setups que terminan en BE).
        """
        try:
            if trade.open_time is None or trade.close_time is None:
                return False
            duration_s = (trade.close_time - trade.open_time).total_seconds()
            # Caso 1: Cierre instantaneo del broker (< min_duration_s) con profit=0
            if duration_s < min_duration_seconds and trade.profit is not None:
                if abs(float(trade.profit)) < 0.01:
                    return True
            # Caso 2: SL o TP fuera de rango fisico (> 5% del precio = error)
            if trade.entry_price and trade.entry_price > 0:
                entry = float(trade.entry_price)
                if trade.sl is not None:
                    sl_dist_pct = abs(float(trade.sl) - entry) / entry
                    if sl_dist_pct > 0.05:
                        return True
                if trade.tp is not None:
                    tp_dist_pct = abs(float(trade.tp) - entry) / entry
                    if tp_dist_pct > 0.05:
                        return True
                    # TP negativo o cero (bug claro)
                    if float(trade.tp) <= 0:
                        return True

                # AURION X 4.3 (10-jun-2026): detectar exit_price corrupto o
                # R-multiple inconsistente. Bug confirmado en post-mortem:
                #   - 06-04 LONG R=-53 con exit en lado favorable (datos invalidos)
                #   - 06-09 4x SHORTS R=-14.40 EXACTO con exit cerca del entry
                #     (R mal calculado, debio ser ~-1R cada uno)
                #   - 06-10 SHORT R=-10.30 con exit 13.89x del SL nominal (slippage)
                # El modelo no debe aprender de estos artefactos.
                # Reglas (cumplir cualquiera → zombie):
                if (trade.exit_price is not None and trade.sl is not None
                        and trade.r_multiple is not None and trade.direction is not None):
                    exit_p = float(trade.exit_price)
                    sl_p = float(trade.sl)
                    r_val = float(trade.r_multiple)
                    if trade.direction.value == "LONG":
                        sl_dist = entry - sl_p
                        exit_dist = entry - exit_p
                    else:
                        sl_dist = sl_p - entry
                        exit_dist = exit_p - entry
                    if sl_dist > 0:
                        ratio = exit_dist / sl_dist
                        # (A) |R| muy grande + exit_dist muy lejos del SL → slippage masivo
                        if abs(r_val) > 8.0 and abs(ratio) > 3.0:
                            return True
                        # (B) R negativo grande + exit en lado favorable → datos contradictorios
                        if r_val < -3.0 and ratio < -0.5:
                            return True
                        # (C) R<-8 pero exit CERCA del entry (|ratio|<1.5) → R-multiple
                        # mal calculado. Si el exit no se movio mas del SL pero el R
                        # dice -10/-14, es un bug aritmetico (no aprende nada util).
                        if r_val < -8.0 and abs(ratio) < 1.5:
                            return True
                        # AURION X 4.4 (10-jun-2026): (D) sl_hit con slippage moderado
                        # pero claramente fuera de lo esperado. Diagnostico mostro
                        # close_reason=sl_hit con avg R=-13.5 (esperado -1R). El bot
                        # esta operando con SLs mal ejecutados (broker side). El
                        # modelo NO debe aprender de estos casos: el patron no
                        # era malo, fue ruido de ejecucion.
                        reason = (trade.close_reason or "").lower()
                        if "sl" in reason and r_val < -2.5:
                            return True
            return False
        except (ValueError, TypeError, AttributeError):
            return False

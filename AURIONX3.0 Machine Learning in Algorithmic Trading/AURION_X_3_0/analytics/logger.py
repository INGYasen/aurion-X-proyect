"""Configuración centralizada de logging para AURION X 3.0."""

from __future__ import annotations

import logging
import sys
from typing import Any


def configure_root_logger(
    level: int = logging.INFO,
    *,
    format_string: str | None = None,
) -> None:
    """Configura el logger raíz con un manejador de consola.

    Args:
        level: Nivel mínimo de severidad (por ejemplo ``logging.INFO``).
        format_string: Formato opcional; si es ``None`` se usa un formato por defecto.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = format_string or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger hijo con el nombre del módulo o componente.

    Args:
        name: Identificador del logger (típicamente ``__name__``).

    Returns:
        Instancia de ``logging.Logger`` lista para emitir registros.
    """
    return logging.getLogger(name)


def log_context(**kwargs: Any) -> dict[str, Any]:
    """Reserva para enriquecer registros con contexto estructurado (sin lógica aún).

    Args:
        **kwargs: Pares clave-valor de contexto.

    Returns:
        Diccionario de contexto (stub).
    """
    return dict(kwargs)


def _fmt_num(value: Any, *, digits: int = 4) -> str:
    """Formatea números para logs compactos; fallback a ``str`` para otros tipos."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_trade_line(
    *,
    symbol: str,
    direction: str,
    lot: float,
    confidence: float,
    r_multiple: float | None = None,
    pnl: float | None = None,
) -> str:
    """Construye línea de log en formato:
    ``TRADE | symbol= | dir= | lot= | conf= | R= | pnl=``.
    """
    return (
        "TRADE | "
        f"symbol={symbol} | "
        f"dir={direction} | "
        f"lot={_fmt_num(lot, digits=4)} | "
        f"conf={_fmt_num(confidence, digits=3)} | "
        f"R={_fmt_num(r_multiple, digits=3)} | "
        f"pnl={_fmt_num(pnl, digits=2)}"
    )


def format_block_line(reason: str) -> str:
    """Construye línea de bloqueo:
    ``BLOCK | reason=``.
    """
    return f"BLOCK | reason={reason}"


def format_state_line(*, dd: float, equity: float, trades_session: int) -> str:
    """Construye línea de estado:
    ``STATE | dd= | equity= | trades_session=``.
    """
    return (
        "STATE | "
        f"dd={_fmt_num(dd, digits=4)} | "
        f"equity={_fmt_num(equity, digits=2)} | "
        f"trades_session={trades_session}"
    )


def log_trade(
    logger: logging.Logger,
    *,
    symbol: str,
    direction: str,
    lot: float,
    confidence: float,
    r_multiple: float | None = None,
    pnl: float | None = None,
) -> None:
    """Emite un registro TRADE en nivel INFO."""
    logger.info(
        format_trade_line(
            symbol=symbol,
            direction=direction,
            lot=lot,
            confidence=confidence,
            r_multiple=r_multiple,
            pnl=pnl,
        ),
    )


def log_block(logger: logging.Logger, reason: str) -> None:
    """Emite un registro BLOCK en nivel WARNING."""
    logger.warning(format_block_line(reason))


def log_state(logger: logging.Logger, *, dd: float, equity: float, trades_session: int) -> None:
    """Emite un registro STATE en nivel INFO."""
    logger.info(format_state_line(dd=dd, equity=equity, trades_session=trades_session))

"""Conexión a base de datos para analytics y auditoría de operaciones."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .logger import get_logger

logger = get_logger(__name__)


def _redact_url(url: str) -> str:
    """Oculta contraseña en DSN para logs."""
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, hostpart = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0] if ":" in creds else creds
    return f"{scheme}://{user}:***@{hostpart}"


def _database_url_candidates(explicit: str | None = None) -> list[str]:
    """URLs a probar: explícita, env, luego Docker típico (5433 secundario, 5432 primario).

    Usa ``127.0.0.1`` para evitar fallos por resolución ``localhost`` → ``::1``.
    """
    if explicit is not None and str(explicit).strip():
        return [str(explicit).strip()]
    env = os.environ.get("DATABASE_URL", "").strip()
    if env:
        return [env]
    return [
        "postgresql://aurion:aurion123@127.0.0.1:5432/aurionx",
        "postgresql://aurion:aurion123@127.0.0.1:5433/aurionx",
    ]

_DEFAULT_POOL_SIZE = 5
_DEFAULT_MAX_OVERFLOW = 10
_DEFAULT_POOL_RECYCLE_S = 3600
_CONNECT_RETRIES = 3
_CONNECT_RETRY_DELAY_S = 1.0

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_engine_url: str | None = None
_lock = Lock()


def _verify_engine_connectivity(engine: Engine) -> None:
    """Ejecuta una comprobación mínima sobre el pool."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def _create_engine_instance(url: str) -> Engine:
    """Construye el motor SQLAlchemy con pool y saneamiento de conexiones."""
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=_DEFAULT_POOL_SIZE,
        max_overflow=_DEFAULT_MAX_OVERFLOW,
        pool_recycle=_DEFAULT_POOL_RECYCLE_S,
        echo=False,
    )


def _create_engine_with_retry(
    url: str,
    *,
    max_attempts: int,
    log_traceback: bool,
) -> Engine:
    """Crea el motor y valida conectividad con reintentos."""
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            engine = _create_engine_instance(url)
            _verify_engine_connectivity(engine)
            logger.info(
                "Motor PostgreSQL listo en %s (intento %s/%s)",
                _redact_url(url),
                attempt,
                max_attempts,
            )
            return engine
        except Exception as exc:
            last_error = exc
            if log_traceback:
                logger.error(
                    "Fallo al crear o verificar el motor %s (intento %s/%s): %s",
                    _redact_url(url),
                    attempt,
                    max_attempts,
                    exc,
                    exc_info=True,
                )
            else:
                logger.warning(
                    "Sin conexión a %s (intento %s/%s): %s",
                    _redact_url(url),
                    attempt,
                    max_attempts,
                    exc,
                )
            if attempt < max_attempts:
                time.sleep(_CONNECT_RETRY_DELAY_S)
    assert last_error is not None
    raise last_error


def get_engine(url: str | None = None) -> Engine:
    """Devuelve el motor singleton con pool de conexiones.

    Args:
        url: URI SQLAlchemy; si es ``None`` se usa ``DATABASE_URL`` o failover 5433→5432.

    Returns:
        Instancia de ``Engine`` compartida en el proceso.
    """
    global _engine, _engine_url, _session_factory
    candidates = _database_url_candidates(url)
    attempts_per_url = 1 if len(candidates) > 1 else _CONNECT_RETRIES
    with _lock:
        if _engine is not None:
            if url is not None and url != _engine_url:
                logger.warning(
                    "get_engine: el motor ya existe con otra URL; se ignora la nueva URI",
                )
            return _engine
        last_error: BaseException | None = None
        for i, candidate in enumerate(candidates):
            is_last_candidate = i == len(candidates) - 1
            try:
                _engine = _create_engine_with_retry(
                    candidate,
                    max_attempts=attempts_per_url,
                    log_traceback=is_last_candidate,
                )
                _engine_url = candidate
                _session_factory = None
                if len(candidates) > 1:
                    logger.info(
                        "Base de datos: usando %s (failover desde lista de candidatos)",
                        _redact_url(candidate),
                    )
                return _engine
            except Exception as exc:
                last_error = exc
                if not is_last_candidate:
                    logger.info(
                        "Probando otra URL de base de datos (falló %s)",
                        _redact_url(candidate),
                    )
        assert last_error is not None
        raise last_error


def _get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            class_=Session,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


def new_session() -> Session:
    """Devuelve una sesión ORM nueva del pool.

    Usar ``with session.begin():`` para delimitar la transacción; cerrar la sesión
    al terminar (p. ej. en un bloque ``finally``).
    """
    return _get_session_factory()()


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager de sesión ORM con commit/rollback automáticos.

    Yields:
        ``Session`` activa; hace ``commit`` al salir sin error y ``rollback`` si hubo excepción.
    """
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Sesión de base de datos revertida por error")
        raise
    finally:
        session.close()


def init_db() -> None:
    """Crea las tablas definidas en el metadata ORM si no existen."""
    from . import models as _models  # noqa: F401 — registra mapeos ORM
    from . import quant_models as _quant_models  # noqa: F401 — tablas ML dedicadas
    try:
        from . import smc_models as _smc_models  # noqa: F401 — tablas SMC AURION X 4.1
    except Exception as exc:  # pragma: no cover - import opcional defensivo
        logger.warning("init_db: smc_models no disponible (%s)", exc)
        _smc_models = None  # type: ignore[assignment]
    try:
        from . import xau_knowledge_models as _xau_models  # noqa: F401 — tablas XAU Knowledge
    except Exception as exc:  # pragma: no cover - import opcional defensivo
        logger.warning("init_db: xau_knowledge_models no disponible (%s)", exc)
        _xau_models = None  # type: ignore[assignment]

    engine = get_engine()
    _models.Base.metadata.create_all(bind=engine)
    _quant_models.Base.metadata.create_all(bind=engine)
    if _smc_models is not None:
        _smc_models.Base.metadata.create_all(bind=engine)
    if _xau_models is not None:
        _xau_models.Base.metadata.create_all(bind=engine)
    logger.info("init_db: metadata sincronizado (create_all + SMC + XAU Knowledge)")


def dispose_engine() -> None:
    """Cierra el pool y libera el motor (p. ej. tests o apagado)."""
    global _engine, _session_factory, _engine_url
    with _lock:
        if _engine is not None:
            _engine.dispose()
            logger.info("Motor de base de datos dispuesto correctamente")
        _engine = None
        _session_factory = None
        _engine_url = None


class DatabaseConnection:
    """Abstrae el motor compartido para código que inyecta esta dependencia."""

    def __init__(self, dsn: str | None = None) -> None:
        """Inicializa la configuración de conexión (sin abrir pool hasta ``connect``).

        Args:
            dsn: Cadena SQLAlchemy; ``None`` usa la URL por defecto del proyecto.
        """
        self._dsn = dsn
        logger.debug("DatabaseConnection instanciado (dsn definido: %s)", dsn is not None)

    def connect(self) -> None:
        """Materializa el motor y el pool (idempotente)."""
        get_engine(self._dsn)

    def disconnect(self) -> None:
        """Libera el motor global si está activo."""
        dispose_engine()

    def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Ejecuta una sentencia SQL con parámetros nombrados.

        Args:
            query: Sentencia SQL (usar parámetros enlazados, no concatenación).
            params: Valores nombrados para la consulta.

        Returns:
            Lista de filas materializadas (``result.all()``) dentro del alcance de la conexión.
        """
        engine = get_engine(self._dsn)
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.all()

"""Filtrado por sesión de negociación (Asia, Londres, NY, etc.)."""

from __future__ import annotations

from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from analytics.logger import get_logger

logger = get_logger(__name__)


def parse_session_time(raw: Any, *, fallback: time) -> time:
    """Convierte ``"HH:MM"``, ``time`` o ``None`` en ``datetime.time``."""
    if raw is None:
        return fallback
    if isinstance(raw, time):
        return raw
    s = str(raw).strip()
    if not s:
        return fallback
    parts = s.replace(".", ":").split(":", 1)
    try:
        h = int(parts[0]) % 24
        m = int(parts[1]) if len(parts) > 1 else 0
        m = max(0, min(m, 59))
        return time(h, m)
    except (ValueError, IndexError):
        logger.warning("parse_session_time: valor inválido %r, se usa %s", raw, fallback)
        return fallback


class SessionFilter:
    """Decide si en un instante dado se permite operar según calendario de sesión."""

    def __init__(
        self,
        timezone: str = "UTC",
        *,
        start: time | None = None,
        end: time | None = None,
        enabled: bool = False,
    ) -> None:
        """Configura zona y franja diaria (interpretada en esa zona).

        Args:
            timezone: Nombre IANA (p. ej. ``UTC``, ``Europe/Madrid``, ``America/Bogota``).
            start: Hora de inicio inclusive; por defecto 07:00.
            end: Hora de fin inclusive; por defecto 20:00.
            enabled: Si es falso (por defecto), ``allow_new_entries`` siempre es verdadero (24h).
        """
        self._tz = ZoneInfo(timezone)
        self._start = start if start is not None else time(hour=7, minute=0)
        self._end = end if end is not None else time(hour=20, minute=0)
        self._enabled = enabled
        self._timezone_name = timezone
        # Evita ventana degenerada (p. ej. 00:00–00:00) que bloquearía casi todo el día.
        if self._enabled and self._start == self._end:
            logger.warning(
                "SessionFilter: start == end (%s); se interpreta como sesión 24h (filtro desactivado).",
                self._start,
            )
            self._enabled = False
        logger.debug(
            "SessionFilter tz=%s %s–%s enabled=%s",
            timezone,
            self._start,
            self._end,
            self._enabled,
        )

    def _to_local(self, when: datetime) -> datetime:
        """Normaliza ``when`` a la zona de la sesión.

        Si ``when`` es naive, se asume UTC (no hora local del sistema) para evitar
        corrimientos silenciosos de sesión.
        """
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return when.astimezone(self._tz)

    @property
    def hint(self) -> str:
        """Texto corto para logs (ventana + zona)."""
        if not self._enabled:
            return "desactivado"
        return f"{self._start.strftime('%H:%M')}–{self._end.strftime('%H:%M')} {self._timezone_name}"

    def is_within_session(self, when: datetime, *, start: time, end: time) -> bool:
        """Comprueba si ``when`` cae dentro del intervalo diario ``[start, end]``.

        Args:
            when: Instante a evaluar (con o sin tz; se normalizará).
            start: Hora local de apertura de la ventana.
            end: Hora local de cierre de la ventana.

        Returns:
            ``True`` si la sesión está activa para operar.
        """
        local = self._to_local(when)
        t = local.timetz().replace(tzinfo=None)
        if start <= end:
            return start <= t <= end
        return t >= start or t <= end

    def allow_new_entries(self, when: datetime) -> bool:
        """Política agregada: si se permiten nuevas entradas en ``when``."""
        if not self._enabled:
            return True
        return self.is_within_session(when, start=self._start, end=self._end)

    def classify_profile(self, when: datetime) -> str:
        """Clasifica sesión operativa para calibración de agresividad."""
        local = self._to_local(when)
        minute = local.hour * 60 + local.minute
        # Ventanas en UTC por defecto para priorizar Londres/NY y su solape.
        # Si la zona no es UTC, la clasificación sigue siendo consistente dentro de la sesión local.
        london_start, london_end = 7 * 60, 16 * 60
        ny_start, ny_end = 13 * 60, 22 * 60
        asia_start, asia_end = 0, 7 * 60

        in_london = london_start <= minute <= london_end
        in_ny = ny_start <= minute <= ny_end
        in_asia = asia_start <= minute < asia_end

        if in_london and in_ny:
            return "OVERLAP_LONDON_NY"
        if in_london:
            return "LONDON"
        if in_ny:
            return "NEW_YORK"
        if in_asia:
            return "ASIA"
        return "DEAD_TRANSITION"

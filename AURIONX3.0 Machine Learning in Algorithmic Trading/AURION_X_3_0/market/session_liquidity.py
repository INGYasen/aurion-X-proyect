"""Session Liquidity Engine (AURION X 4.6) — Asia/London/NY High-Low pools.

Detecta los high/low de cada sesión de trading y los expone como
"liquidity pools de sesión" que sirven como:

    1. Niveles objetivo (Take Profit) — magnetismo institucional
    2. Niveles a barrer (Liquidity Sweep) — entradas post-sweep
    3. Indicador de manipulación (Asian range trap)

Sesiones (UTC):

    * Asia    : 00:00 - 07:00
    * London  : 07:00 - 15:00  (killzone 7-10)
    * NY      : 12:00 - 20:00  (killzone 13-15)
    * Overlap London-NY: 12:00 - 15:00

Sesión "actual" usada en panel/heurísticas. Estado por sesión:
``{"high", "low", "open_time", "close_time", "swept_high", "swept_low",
"is_active"}``.

NO depende de un broker ni timezone — siempre UTC. El consumer convierte
si necesita.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable


# ─── tipos publicos ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionRange:
    """Rango de una sesión."""
    name: str                    # "asia" | "london" | "ny" | "overlap"
    date_utc: date
    open_utc: datetime
    close_utc: datetime
    is_active: bool
    high: float | None
    low: float | None
    swept_high: bool
    swept_low: bool
    high_time: datetime | None
    low_time: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "date_utc": self.date_utc.isoformat(),
            "open_utc": self.open_utc.isoformat(),
            "close_utc": self.close_utc.isoformat(),
            "is_active": bool(self.is_active),
            "high": float(self.high) if self.high is not None else None,
            "low":  float(self.low)  if self.low  is not None else None,
            "swept_high": bool(self.swept_high),
            "swept_low":  bool(self.swept_low),
            "high_time": self.high_time.isoformat() if self.high_time else None,
            "low_time":  self.low_time.isoformat()  if self.low_time  else None,
        }


# ─── definiciones de sesion (UTC) ───────────────────────────────────────


_SESSIONS: dict[str, tuple[int, int]] = {
    # name: (open_hour_utc, close_hour_utc)
    "asia":    (0,  7),
    "london":  (7,  15),
    "ny":      (12, 20),
    "overlap": (12, 15),
}


# ─── helpers ────────────────────────────────────────────────────────────


def _is_active(now: datetime, oh: int, ch: int) -> bool:
    h = now.hour + now.minute / 60.0
    return oh <= h < ch


def _today_window(now: datetime, oh: int, ch: int) -> tuple[datetime, datetime]:
    today = now.astimezone(timezone.utc).date()
    o = datetime.combine(today, time(oh, 0), tzinfo=timezone.utc)
    c = datetime.combine(today, time(ch, 0), tzinfo=timezone.utc)
    return o, c


# ─── api publica ─────────────────────────────────────────────────────────


def current_session(now: datetime | None = None) -> str:
    """Devuelve nombre de la sesión activa por prioridad: overlap > ny > london > asia > dead."""
    n = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    # Prioridad: overlap > ny > london > asia
    for name in ("overlap", "ny", "london", "asia"):
        oh, ch = _SESSIONS[name]
        if _is_active(n, oh, ch):
            return name
    return "dead"


def compute_session_ranges(
    *,
    candles: Iterable[Any],
    now: datetime | None = None,
    last_close: float | None = None,
    tolerance_pips: float = 1.0,
) -> dict[str, SessionRange]:
    """Calcula high/low de cada sesión del día UTC actual.

    Args:
        candles: iterable de velas con atributos ``time``, ``high``, ``low``,
                 ``close``. Pueden ser M1, M5 o M15 — cuantas más fino, mejor.
        now: timestamp de referencia (default datetime.now UTC).
        last_close: último cierre (para evaluar swept_high/low) si no hay vela post.
        tolerance_pips: tolerancia para considerar high/low "tocados" (XAUUSD 1 pip = 0.10$).

    Returns:
        Dict { "asia": SessionRange, "london": ..., "ny": ..., "overlap": ... }
    """
    n = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    candle_list = list(candles or [])
    tol = float(tolerance_pips) / 10.0  # pips → $ (XAUUSD)

    def _ts_of(c: Any) -> datetime | None:
        # Acepta .time, .time_utc o .start (compatible con _Bar del registry MTF)
        ct = getattr(c, "time", None)
        if ct is None:
            ct = getattr(c, "time_utc", None)
        if ct is None:
            ct = getattr(c, "start", None)
        return ct

    result: dict[str, SessionRange] = {}
    for name, (oh, ch) in _SESSIONS.items():
        open_utc, close_utc = _today_window(n, oh, ch)
        active = _is_active(n, oh, ch)
        hi: float | None = None
        lo: float | None = None
        hi_t: datetime | None = None
        lo_t: datetime | None = None

        for c in candle_list:
            ct = _ts_of(c)
            if ct is None:
                continue
            if isinstance(ct, datetime) and ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            # Solo velas dentro del rango de hoy
            if ct < open_utc:
                continue
            # Si la sesión ya cerró, no aceptes velas posteriores
            if ct >= close_utc:
                continue
            try:
                ch_high = float(getattr(c, "high"))
                ch_low  = float(getattr(c, "low"))
            except (AttributeError, TypeError, ValueError):
                continue
            if hi is None or ch_high > hi:
                hi = ch_high
                hi_t = ct
            if lo is None or ch_low < lo:
                lo = ch_low
                lo_t = ct

        # Evaluar swept usando velas POST cierre de sesión + last_close
        swept_hi = False
        swept_lo = False
        if hi is not None and last_close is not None:
            if float(last_close) > hi + tol:
                swept_hi = True
        if lo is not None and last_close is not None:
            if float(last_close) < lo - tol:
                swept_lo = True
        # También: si en velas posteriores al cierre se barrió
        if not active and (hi is not None or lo is not None):
            for c in candle_list:
                ct = _ts_of(c)
                if ct is None:
                    continue
                if isinstance(ct, datetime) and ct.tzinfo is None:
                    ct = ct.replace(tzinfo=timezone.utc)
                if ct < close_utc:
                    continue
                try:
                    ch_high = float(getattr(c, "high"))
                    ch_low  = float(getattr(c, "low"))
                except (AttributeError, TypeError, ValueError):
                    continue
                if hi is not None and ch_high > hi + tol:
                    swept_hi = True
                if lo is not None and ch_low < lo - tol:
                    swept_lo = True

        result[name] = SessionRange(
            name=name, date_utc=open_utc.date(),
            open_utc=open_utc, close_utc=close_utc,
            is_active=active, high=hi, low=lo,
            swept_high=swept_hi, swept_low=swept_lo,
            high_time=hi_t, low_time=lo_t,
        )

    return result


def session_pools_for_targets(
    ranges: dict[str, SessionRange],
) -> dict[str, dict[str, Any]]:
    """Adapta el output de compute_session_ranges al formato esperado por
    ``risk.smart_targets.compute_targets(sessions=...)``.

    Devuelve:
        { "asia":  {"high":..., "low":..., "swept_high":..., "swept_low":...},
          "london": {...}, ... }
    """
    out: dict[str, dict[str, Any]] = {}
    for name, sr in (ranges or {}).items():
        if sr.high is None and sr.low is None:
            continue
        out[name] = {
            "high": sr.high, "low": sr.low,
            "swept_high": sr.swept_high, "swept_low": sr.swept_low,
        }
    # Daily H/L estimado: el max/min de las 3 sesiones principales
    highs = [sr.high for sr in ranges.values() if sr.high is not None]
    lows  = [sr.low  for sr in ranges.values() if sr.low  is not None]
    if highs and lows:
        out["daily"] = {
            "high": max(highs), "low": min(lows),
            "swept_high": all(sr.swept_high for sr in ranges.values()
                              if sr.high is not None and not sr.is_active),
            "swept_low":  all(sr.swept_low  for sr in ranges.values()
                              if sr.low  is not None and not sr.is_active),
        }
    return out


def session_classification(
    *,
    now: datetime | None = None,
    smc_score: float = 0.0,
) -> dict[str, Any]:
    """Devuelve dict con sesión actual + calidad histórica del bot (placeholder).

    La "calidad histórica" se rellena externamente desde ``postmortem`` cuando
    haya datos; por ahora retorna baseline neutro.
    """
    s = current_session(now)
    quality_baseline = {
        "overlap": 0.75, "ny": 0.65, "london": 0.62,
        "asia": 0.42, "dead": 0.20,
    }
    return {
        "current_session": s,
        "quality_baseline": float(quality_baseline.get(s, 0.5)),
        "is_killzone": s in ("london", "ny", "overlap"),
        "is_low_quality": s in ("asia", "dead"),
    }

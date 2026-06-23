"""Filtro de calendario economico - blackout de trading en eventos high-impact.

AURION X 4.3 (7-jun-2026): Los movimientos erraticos en releases macro
(NFP, FOMC, CPI, GDP, PCE) causan whipsaws que destruyen scalpers / intraday
basados en estructura. Este modulo bloquea la apertura de nuevas posiciones
durante ventanas configurables alrededor del release.

Diseno:
    - Eventos recurrentes hardcoded (NFP = 1er viernes 13:30 UTC, FOMC = ad-hoc)
      con metadata mensual aproximada.
    - Ventana default: 30min antes + 60min despues del release.
    - Symbol-aware: USD events afectan XAUUSD, USTEC, ES, BTCUSD.
    - Soporte para overrides via YAML (lista de eventos custom con timestamp UTC).

Uso:
    cal = EconomicCalendarGuard.from_config(cfg["risk"]["news_calendar"])
    if cal.in_blackout(now, symbol="XAUUSD"):
        # No abrir posiciones
        return
"""
from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Iterable, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


# Simbolos sensibles a eventos USD (afectados por blackout).
_USD_SENSITIVE = frozenset({"XAUUSD", "USTEC", "NDX100", "NAS100", "US100", "SPX500",
                             "US30", "DJI30", "BTCUSD", "ETHUSD", "EURUSD", "GBPUSD",
                             "USDJPY", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD"})


@dataclass(frozen=True)
class EconomicEvent:
    """Evento del calendario economico con ventana de blackout."""

    name: str
    timestamp_utc: datetime
    severity: str = "high"  # low | medium | high
    affected_symbols: frozenset[str] = field(default_factory=lambda: _USD_SENSITIVE)
    blackout_before_min: int = 30
    blackout_after_min: int = 60


@dataclass
class EconomicCalendarGuard:
    """Bloquea trading durante ventanas alrededor de releases macro high-impact.

    AURION X 4.3 (7-jun-2026): backtest sobre 320 trades XAUUSD historicos
    revelo que filtrar TODOS los eventos macro restaba PnL. Razon: trades
    cerca de news pasaban por filtros previos (SMC + ChopGuard) que ya los
    seleccionaban como alta calidad. Por eso ahora el filtro es QUIRURGICO:

        - Solo NFP (1er viernes) por default (no CPI/PPI/Retail).
        - FOMC se carga via custom_events cuando se anuncia fecha.
        - ``override_high_ml_conf``: si ml_conf >= threshold, no bloquea.
          Esto preserva trades de muy alta calidad que pueden capitalizar
          la volatilidad del release.

    El consumidor tipicamente llama:
        cal.in_blackout(now, symbol="XAUUSD", ml_conf=0.83) -> bool
    """

    enabled: bool = True
    default_before_min: int = 30
    default_after_min: int = 60
    # Eventos extra cargados desde YAML (ej. FOMC con fecha precisa).
    custom_events: list[EconomicEvent] = field(default_factory=list)
    # AURION X 4.3: si ml_conf >= este threshold, NO se aplica el blackout.
    # Permite que trades de muy alta calidad capitalicen volatilidad del release.
    override_high_ml_conf: float = 0.85
    # AURION X 4.3: lista de eventos recurrentes a INCLUIR (por defecto solo NFP).
    # Posibles: "NFP", "CPI", "PPI", "PCE_CORE", "RETAIL_SALES".
    include_recurring: tuple[str, ...] = ("NFP",)
    # Cache de eventos recurrentes calculados para el mes actual.
    _recurring_cache_year_month: tuple[int, int] | None = None
    _recurring_cache_events: list[EconomicEvent] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Mapping[str, object] | None) -> "EconomicCalendarGuard":
        """Crea guard desde seccion ``risk.news_calendar`` del YAML."""
        if not cfg:
            return cls(enabled=False)
        enabled = bool(cfg.get("enabled", True))
        before = int(cfg.get("blackout_before_min", 30) or 30)
        after = int(cfg.get("blackout_after_min", 60) or 60)
        override_ml = float(cfg.get("override_high_ml_conf", 0.85) or 0.85)
        include_raw = cfg.get("include_recurring", ("NFP",)) or ("NFP",)
        if isinstance(include_raw, str):
            include = (include_raw.strip().upper(),)
        elif isinstance(include_raw, (list, tuple)):
            include = tuple(str(x).strip().upper() for x in include_raw if str(x).strip())
        else:
            include = ("NFP",)
        custom_raw = cfg.get("custom_events") or []
        custom: list[EconomicEvent] = []
        if isinstance(custom_raw, list):
            for item in custom_raw:
                try:
                    ts_str = str(item.get("timestamp_utc", "")) if isinstance(item, dict) else ""
                    if not ts_str:
                        continue
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    name = str(item.get("name", "CUSTOM")) if isinstance(item, dict) else "CUSTOM"
                    custom.append(EconomicEvent(
                        name=name,
                        timestamp_utc=ts,
                        severity=str(item.get("severity", "high")) if isinstance(item, dict) else "high",
                        blackout_before_min=int(item.get("blackout_before_min", before)) if isinstance(item, dict) else before,
                        blackout_after_min=int(item.get("blackout_after_min", after)) if isinstance(item, dict) else after,
                    ))
                except (ValueError, AttributeError) as exc:
                    logger.warning("EconomicCalendar: evento custom invalido: %s", exc)
        return cls(
            enabled=enabled,
            default_before_min=before,
            default_after_min=after,
            custom_events=custom,
            override_high_ml_conf=override_ml,
            include_recurring=include,
        )

    def _build_recurring_for_month(self, year: int, month: int) -> list[EconomicEvent]:
        """Genera eventos recurrentes USD high-impact para year/month dado."""
        events: list[EconomicEvent] = []
        # 1er viernes 13:30 UTC = NFP (Non-Farm Payrolls)
        cal_month = _cal.Calendar(firstweekday=0)
        fridays = [d for d in cal_month.itermonthdates(year, month)
                   if d.month == month and d.weekday() == 4]
        if fridays:
            nfp_date = fridays[0]
            nfp_dt = datetime.combine(nfp_date, time(13, 30), tzinfo=timezone.utc)
            events.append(EconomicEvent(name="NFP", timestamp_utc=nfp_dt))
            # PCE Core: ultimo viernes
            pce_dt = datetime.combine(fridays[-1], time(13, 30), tzinfo=timezone.utc)
            events.append(EconomicEvent(name="PCE_CORE", timestamp_utc=pce_dt,
                                          blackout_before_min=20, blackout_after_min=45))
        # CPI ~dia 12-15 (Buscar dia 13 o dia habil mas cercano)
        cpi_day = 13
        while cpi_day <= 15:
            try:
                dt = datetime(year, month, cpi_day, 13, 30, tzinfo=timezone.utc)
                if dt.weekday() < 5:  # Mon-Fri
                    events.append(EconomicEvent(name="CPI", timestamp_utc=dt))
                    break
            except ValueError:
                pass
            cpi_day += 1
        # PPI ~dia 14-16
        ppi_day = 14
        while ppi_day <= 16:
            try:
                dt = datetime(year, month, ppi_day, 13, 30, tzinfo=timezone.utc)
                if dt.weekday() < 5:
                    events.append(EconomicEvent(name="PPI", timestamp_utc=dt,
                                                  blackout_before_min=20, blackout_after_min=40))
                    break
            except ValueError:
                pass
            ppi_day += 1
        # Retail Sales ~dia 15-17
        rs_day = 15
        while rs_day <= 17:
            try:
                dt = datetime(year, month, rs_day, 13, 30, tzinfo=timezone.utc)
                if dt.weekday() < 5:
                    events.append(EconomicEvent(name="RETAIL_SALES", timestamp_utc=dt,
                                                  blackout_before_min=20, blackout_after_min=40))
                    break
            except ValueError:
                pass
            rs_day += 1
        return events

    def _get_recurring(self, now_utc: datetime) -> list[EconomicEvent]:
        ym = (now_utc.year, now_utc.month)
        if self._recurring_cache_year_month != ym:
            all_events = self._build_recurring_for_month(*ym)
            include_set = set(self.include_recurring) if self.include_recurring else set()
            filtered = [ev for ev in all_events if ev.name in include_set]
            object.__setattr__(self, "_recurring_cache_events", filtered)
            object.__setattr__(self, "_recurring_cache_year_month", ym)
        return self._recurring_cache_events

    def in_blackout(
        self,
        now_utc: datetime,
        symbol: str = "XAUUSD",
        ml_conf: float | None = None,
    ) -> tuple[bool, str]:
        """Retorna (bloqueado, nombre_evento). Si no hay evento, ("", "").

        Args:
            now_utc: Tiempo actual UTC aware.
            symbol: Simbolo del trade a abrir. Eventos USD solo bloquean simbolos sensibles.
            ml_conf: Confianza ML del trade actual. Si >= override_high_ml_conf,
                NO se aplica el blackout (permite trades de muy alta calidad).
        """
        if not self.enabled:
            return False, ""
        # AURION X 4.3: override por alta confluencia ML.
        if ml_conf is not None and ml_conf >= self.override_high_ml_conf:
            return False, ""
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        sym_upper = (symbol or "").upper().strip()
        events = list(self._get_recurring(now_utc)) + list(self.custom_events)
        for ev in events:
            if sym_upper not in ev.affected_symbols:
                continue
            window_start = ev.timestamp_utc - timedelta(minutes=ev.blackout_before_min)
            window_end = ev.timestamp_utc + timedelta(minutes=ev.blackout_after_min)
            if window_start <= now_utc <= window_end:
                return True, ev.name
        return False, ""

    def next_blackout_window(self, now_utc: datetime, symbol: str = "XAUUSD") -> tuple[datetime, str] | None:
        """Proximo evento que afectara al simbolo (None si no hay en el mes)."""
        if not self.enabled:
            return None
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        sym_upper = (symbol or "").upper().strip()
        events = list(self._get_recurring(now_utc)) + list(self.custom_events)
        future = [(ev.timestamp_utc, ev.name) for ev in events
                  if sym_upper in ev.affected_symbols and ev.timestamp_utc > now_utc]
        if not future:
            return None
        future.sort(key=lambda t: t[0])
        return future[0]

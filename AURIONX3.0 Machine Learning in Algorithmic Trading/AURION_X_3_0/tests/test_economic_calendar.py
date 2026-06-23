"""Tests para risk.economic_calendar."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from risk.economic_calendar import EconomicCalendarGuard, EconomicEvent


def _utc(y: int, mo: int, d: int, h: int = 12, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


class TestEconomicCalendarGuard:
    def test_disabled_never_blocks(self) -> None:
        cal = EconomicCalendarGuard(enabled=False)
        now = _utc(2026, 6, 5, 13, 30)  # primer viernes junio = NFP
        blocked, name = cal.in_blackout(now, "XAUUSD")
        assert blocked is False
        assert name == ""

    def test_nfp_first_friday_blocks_xauusd(self) -> None:
        cal = EconomicCalendarGuard(enabled=True)
        # 5 jun 2026 = primer viernes
        now = _utc(2026, 6, 5, 13, 25)  # 5 min antes de NFP 13:30 UTC
        blocked, name = cal.in_blackout(now, "XAUUSD")
        assert blocked is True
        assert name == "NFP"

    def test_nfp_after_window_does_not_block(self) -> None:
        cal = EconomicCalendarGuard(enabled=True)
        # 5 jun 2026, 1h despues + 1min = fuera ventana 60min default
        now = _utc(2026, 6, 5, 14, 31)
        blocked, _ = cal.in_blackout(now, "XAUUSD")
        assert blocked is False

    def test_override_high_ml_conf_bypasses_blackout(self) -> None:
        cal = EconomicCalendarGuard(enabled=True, override_high_ml_conf=0.85)
        now = _utc(2026, 6, 5, 13, 25)  # en plena ventana NFP
        # ml_conf BAJO -> bloqueado
        b1, _ = cal.in_blackout(now, "XAUUSD", ml_conf=0.70)
        assert b1 is True
        # ml_conf ALTO -> bypass
        b2, _ = cal.in_blackout(now, "XAUUSD", ml_conf=0.90)
        assert b2 is False

    def test_non_usd_symbol_not_affected(self) -> None:
        cal = EconomicCalendarGuard(enabled=True)
        now = _utc(2026, 6, 5, 13, 25)
        # Simbolo no listado en _USD_SENSITIVE no se bloquea
        blocked, _ = cal.in_blackout(now, "EURJPY")
        assert blocked is False

    def test_include_recurring_only_nfp_by_default(self) -> None:
        cal = EconomicCalendarGuard(enabled=True)
        # CPI sale ~13 del mes a 13:30 UTC. Como NO esta en include_recurring,
        # no se construye => no bloquea.
        now_cpi = _utc(2026, 6, 15, 13, 30)
        blocked, name = cal.in_blackout(now_cpi, "XAUUSD")
        # Si name vacio, no hubo evento que matchee
        assert blocked is False or name != "CPI"

    def test_include_recurring_can_be_expanded(self) -> None:
        cal = EconomicCalendarGuard(enabled=True, include_recurring=("NFP", "CPI"))
        # Buscar primer dia habil entre 13-15 jun 2026 = lunes 15
        now = _utc(2026, 6, 15, 13, 30)
        blocked, name = cal.in_blackout(now, "XAUUSD")
        assert blocked is True
        assert name == "CPI"

    def test_custom_event_blocks(self) -> None:
        future_fomc = _utc(2026, 7, 16, 18, 0)
        cal = EconomicCalendarGuard(
            enabled=True,
            custom_events=[EconomicEvent(name="FOMC", timestamp_utc=future_fomc,
                                          blackout_before_min=60, blackout_after_min=120)],
        )
        # 30min antes del FOMC -> bloqueo
        blocked, name = cal.in_blackout(future_fomc - timedelta(minutes=30), "XAUUSD")
        assert blocked is True
        assert name == "FOMC"

    def test_from_config_yaml_shape(self) -> None:
        cfg = {
            "enabled": True,
            "blackout_before_min": 20,
            "blackout_after_min": 45,
            "override_high_ml_conf": 0.80,
            "include_recurring": ["NFP"],
            "custom_events": [
                {"name": "FOMC", "timestamp_utc": "2026-09-17T18:00:00Z"},
            ],
        }
        cal = EconomicCalendarGuard.from_config(cfg)
        assert cal.enabled is True
        assert cal.default_before_min == 20
        assert cal.default_after_min == 45
        assert cal.override_high_ml_conf == 0.80
        assert "NFP" in cal.include_recurring
        assert len(cal.custom_events) == 1
        assert cal.custom_events[0].name == "FOMC"

    def test_next_blackout_window_returns_future_event(self) -> None:
        cal = EconomicCalendarGuard(enabled=True)
        now = _utc(2026, 6, 1, 12, 0)  # antes del NFP primer viernes (5 jun)
        nxt = cal.next_blackout_window(now, "XAUUSD")
        assert nxt is not None
        ts, name = nxt
        assert name == "NFP"
        assert ts > now

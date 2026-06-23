from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from risk.risk_manager import RiskManager


def test_loss_streak_reset_after_cooldown() -> None:
    rm = RiskManager(
        limits={
            "max_drawdown_pause_pct": "99",
            "max_drawdown_hard_stop_pct": "99",
            "max_trades_per_session": 100,
            "cooldown_seconds": 1,
            "loss_streak_pause": 3,
            "loss_streak_reset_after_cooldown": True,
            "risk_per_trade_pct": "5",
        },
    )
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    rm.start_session(Decimal("10000"), now_utc=base)
    for _ in range(3):
        rm.register_trade_result(Decimal("-10"), closed_at=base)
    assert rm.loss_streak >= 3
    assert not rm.can_open_new_trade(Decimal("10000"), Decimal("1"), now_utc=base)
    t1 = base + timedelta(seconds=2)
    assert rm.can_open_new_trade(Decimal("10000"), Decimal("1"), now_utc=t1)
    assert rm.loss_streak == 0


def test_max_trades_zero_means_unlimited() -> None:
    rm = RiskManager(
        limits={
            "max_drawdown_pause_pct": "99",
            "max_drawdown_hard_stop_pct": "99",
            "max_trades_per_session": 0,
            "cooldown_seconds": 1,
            "loss_streak_pause": 0,
            "risk_per_trade_pct": "5",
        },
    )
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    rm.start_session(Decimal("10000"), now_utc=base)
    for _ in range(80):
        rm.register_trade_open(opened_at=base)
    assert rm.can_open_new_trade(Decimal("10000"), Decimal("1"), now_utc=base)


def test_trades_in_last_hour() -> None:
    rm = RiskManager(limits={"max_trades_per_session": 0, "risk_per_trade_pct": "5"})
    base = datetime(2026, 5, 20, 11, 0, tzinfo=timezone.utc)
    rm.start_session(Decimal("10000"), now_utc=base)
    rm.register_trade_open(opened_at=base)
    rm.register_trade_open(opened_at=base + timedelta(minutes=30))
    assert rm.trades_in_last_hour(base + timedelta(minutes=45)) == 2
    assert rm.trades_in_last_hour(base + timedelta(hours=2)) == 0

"""Tests para políticas por día de la semana (domingo/lunes)."""

from market.weekday_filters import (
    WeekdayPolicy,
    evaluate_weekday_policy,
    get_config_for_day,
    load_weekday_policies,
)
from strategy.structure_detector import Trend


def test_sunday_disabled() -> None:
    p = WeekdayPolicy(enabled=False)
    allow, reason = evaluate_weekday_policy("XAUUSD", "RANGE_FADE", p)
    assert not allow
    assert reason == "weekday_disabled"


def test_monday_strict() -> None:
    p = WeekdayPolicy(
        enabled=True,
        lot_factor=0.3,
        ml_threshold_increase=0.15,
        max_trades_per_symbol=1,
        skip_setups=("RANGE_FADE", "RANGE_BREAKOUT_MOMENTUM"),
    )
    assert p.lot_factor == 0.3
    assert p.effective_ml_threshold(0.5) == 0.65
    allow, reason = evaluate_weekday_policy(
        "XAUUSD",
        "RANGE_FADE",
        p,
        trade_count_today=0,
    )
    assert not allow
    assert "weekday_setup_blocked" in reason
    allow2, reason2 = evaluate_weekday_policy(
        "XAUUSD",
        "MICRO_PULLBACK_MOMENTUM",
        p,
        trade_count_today=1,
    )
    assert not allow2
    assert reason2 == "weekday_max_trades"


def test_load_weekday_policies_from_yaml() -> None:
    policies = load_weekday_policies({
        "0": {"enabled": True, "lot_factor": 1.0},
        "6": {"enabled": False, "lot_factor": 0.0},
    })
    assert policies[0].enabled is True
    assert policies[6].enabled is False


def test_get_config_for_day_monday() -> None:
    p = get_config_for_day(0, {
        "0": {
            "enabled": True,
            "lot_factor": 0.25,
            "ml_threshold_increase": 0.12,
            "max_trades_per_symbol": 1,
            "skip_setups": ["RANGE_FADE"],
        },
    })
    assert p.lot_factor == 0.25
    assert "RANGE_FADE" in p.skip_setups


def test_mtf_align_requires_non_range() -> None:
    p = WeekdayPolicy(require_mtf_full_align=True)
    allow, reason = evaluate_weekday_policy(
        "BTCUSD",
        "MICRO_PULLBACK_MOMENTUM",
        p,
        mtf_trends={"M1": Trend.RANGE, "M5": Trend.UP, "M15": Trend.UP},
    )
    assert not allow
    assert reason == "weekday_mtf_align"

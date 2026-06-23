from __future__ import annotations

from core.orchestrator import Orchestrator


def test_legacy_sl_contamination_fixed_for_xau() -> None:
    o = Orchestrator()
    o._min_sl_atr_m1_mult = 0.0
    o._max_sl_atr_m1_mult = 0.0
    o._min_tp_rr_after_sl_floor = 1.22
    o._max_tp_rr = 2.25
    o._max_tp_rr_intraday = 2.45
    o._asset_profiles = {"XAUUSD": {"max_sl_pips": 80}}
    o._config = {"execution": {"legacy_sl_contamination_threshold_pct": 0.0022}}
    sl, tp = o._normalize_entry_sl_tp(
        "XAUUSD",
        "INTRADAY",
        {},
        sl_pips=22000.0,
        tp_pips=40000.0,
        atr_now=0.0,
        entry_price=4555.0,
    )
    assert sl < 300.0
    assert tp < 700.0


def test_order_sl_tp_sane_rejects_bad_sell() -> None:
    o = Orchestrator()
    ok, msg = o._order_sl_tp_sane("XAUUSD", "SELL", 4555.0, 6764.0, 1574.0)
    assert ok is False
    assert "entrada" in msg.lower() or "lejos" in msg.lower()


def test_normalize_uses_price_dist_for_xau() -> None:
    o = Orchestrator()
    o._min_sl_atr_m1_mult = 0.0
    o._max_sl_atr_m1_mult = 0.0
    o._asset_profiles = {
        "XAUUSD": {"min_sl_pips": 18, "max_sl_pips": 50, "max_tp_rr": 2.2},
    }
    signal = {"sl_price_dist": 2.5, "tp_price_dist": 4.5}
    sl, tp = o._normalize_entry_sl_tp(
        "XAUUSD",
        "INTRADAY",
        signal,
        sl_pips=25000.0,
        tp_pips=55000.0,
        atr_now=0.0,
    )
    assert sl == 25.0
    assert tp == 45.0


def test_normalize_caps_tp_rr_scalper() -> None:
    o = Orchestrator()
    o._min_sl_atr_m1_mult = 0.0
    o._max_sl_atr_m1_mult = 0.0
    o._max_tp_rr = 2.25
    o._min_tp_rr_after_sl_floor = 1.22
    o._asset_profiles = {"XAUUSD": {"max_tp_rr": 2.2}}
    sl, tp = o._normalize_entry_sl_tp(
        "XAUUSD",
        "SCALPER",
        {},
        sl_pips=20.0,
        tp_pips=80.0,
        atr_now=0.0,
    )
    assert sl == 20.0
    assert tp == 44.0


def test_normalize_caps_sl_by_price_pct() -> None:
    o = Orchestrator()
    o._min_sl_atr_m1_mult = 0.0
    o._max_sl_atr_m1_mult = 0.0
    o._asset_profiles = {"XAUUSD": {"max_sl_pips": 80, "max_sl_price_pct": 0.001}}
    sl, tp = o._normalize_entry_sl_tp(
        "XAUUSD",
        "SCALPER",
        {},
        sl_pips=60.0,
        tp_pips=90.0,
        atr_now=0.0,
        entry_price=2500.0,
    )
    assert sl == 25.0
    assert tp == 58.75


def test_sl_exceeds_hard_cap_blocks() -> None:
    o = Orchestrator()
    o._asset_profiles = {"XAUUSD": {"max_sl_price_pct": 0.001}}
    blocked, msg = o._sl_exceeds_hard_cap("XAUUSD", 30.0, 2500.0)
    assert blocked is True
    assert "SL" in msg


def test_normalize_atr_floor() -> None:
    o = Orchestrator()
    o._min_sl_atr_m1_mult = 1.0
    o._max_sl_atr_m1_mult = 0.0
    o._min_tp_rr_after_sl_floor = 1.22
    o._asset_profiles = {}
    sl, tp = o._normalize_entry_sl_tp(
        "BTCUSD",
        "SCALPER",
        {},
        sl_pips=10.0,
        tp_pips=18.0,
        atr_now=55.0,
    )
    assert sl == 55.0
    assert tp == 67.1

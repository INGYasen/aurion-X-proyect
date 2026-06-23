"""Tests del gestor de salida dinámica."""

from __future__ import annotations

from execution.dynamic_exit_manager import (
    compute_profit_lock_sl_from_tp,
    decide_dynamic_exit,
    detect_momentum_loss,
    merge_dynamic_exit_cfg,
    should_close_trade,
)


def _ms_base() -> dict:
    return {
        "highs_m1": [100.0, 100.1, 100.2, 100.15, 100.1, 100.0, 99.95, 99.9, 99.85, 99.8, 99.75, 99.7],
        "lows_m1": [99.5, 99.6, 99.7, 99.65, 99.6, 99.55, 99.5, 99.45, 99.4, 99.35, 99.3, 99.25],
        "closes_m1": [99.8, 99.85, 99.9, 99.88, 99.82, 99.78, 99.7, 99.65, 99.6, 99.55, 99.5, 99.45],
        "continuation_score": 0.22,
        "momentum_score": 0.25,
        "displacement_score": 0.2,
        "compression_score": 0.78,
        "velocity_loss": True,
        "lateral_soft": True,
        "absorption_against_long": 0.62,
        "absorption_against_short": 0.1,
        "structure_favorable": True,
        "atr": 0.35,
        "regime": "NORMAL",
        "spread_points": 12.0,
        "mid": 99.5,
    }


def test_merge_dynamic_exit_cfg_profile_overrides() -> None:
    base = {"enabled": False, "max_giveback_allowed": 0.35, "breakeven_r": 0.5}
    merged = merge_dynamic_exit_cfg(base, {"enabled": True, "max_giveback_allowed": 0.22})
    assert merged["enabled"] is True
    assert merged["max_giveback_allowed"] == 0.22
    assert merged["breakeven_r"] == 0.5


def test_decide_disabled_returns_empty() -> None:
    trade = {"ticket": "1", "direction": "BUY", "current_r": 0.8, "max_r": 1.0, "be_done": False, "dynamic_exit_partial_done": False}
    dec = decide_dynamic_exit(trade, _ms_base(), {"enabled": False}, mid=99.5)
    assert not dec.any_action()


def test_giveback_triggers_full_close() -> None:
    cfg = {
        "enabled": True,
        "secure_profit_r": 0.45,
        "max_giveback_allowed": 0.3,
        "momentum_loss_threshold": 0.4,
        "absorption_strength_threshold": 0.55,
        "compression_threshold": 0.72,
    }
    trade = {
        "ticket": "t1",
        "direction": "BUY",
        "entry_price": 100.0,
        "risk_price": 1.0,
        "current_r": 0.25,
        "max_r": 0.8,
        "be_done": False,
        "dynamic_exit_partial_done": False,
    }
    ms = _ms_base()
    ms["absorption_against_long"] = 0.1
    assert should_close_trade(trade, ms, cfg) is True
    dec = decide_dynamic_exit(trade, ms, cfg, mid=99.5)
    assert dec.close_full is True
    assert dec.reason_full == "max_giveback"


def test_detect_momentum_loss_low_continuation() -> None:
    trade = {"direction": "BUY", "current_r": 0.5}
    ms = _ms_base()
    assert detect_momentum_loss(trade, ms, {"momentum_loss_threshold": 0.38}) is True


def test_partial_when_secure_profit() -> None:
    cfg = {
        "enabled": True,
        "secure_profit_r": 0.4,
        "partial_close_percent": 35.0,
        "momentum_loss_threshold": 0.5,
        "max_giveback_allowed": 0.9,
        "absorption_strength_threshold": 0.95,
        "compression_threshold": 0.99,
    }
    trade = {
        "ticket": "p1",
        "direction": "SELL",
        "entry_price": 102.0,
        "risk_price": 1.0,
        "current_r": 0.55,
        "max_r": 0.55,
        "be_done": False,
        "dynamic_exit_partial_done": False,
    }
    ms = _ms_base()
    ms["continuation_score"] = 0.2
    ms["velocity_loss"] = True
    ms["compression_score"] = 0.8
    dec = decide_dynamic_exit(trade, ms, cfg, mid=100.5)
    assert dec.close_partial_fraction > 0
    assert "momentum_loss" in dec.reason_partial or dec.reason_partial == "secure_profit"


# ============================================================================
# AURION X 4.2.3 — Profit Lock progresivo por % del recorrido entry→TP
# ============================================================================

def _plock_cfg() -> dict:
    """Config minimal con profit_lock activo."""
    return {
        "enabled": True,
        "secure_profit_r": 999.0,        # desactiva partial/secure
        "breakeven_r": 999.0,             # desactiva BE clásico
        "trailing_arm_r": 999.0,          # desactiva trailing ATR
        "absorption_strength_threshold": 0.99,
        "compression_threshold": 0.99,
        "max_giveback_allowed": 99.0,
        "profit_lock": {
            "enabled": True,
            "soft_trigger_pct": 0.20,
            "soft_lock_pct": 0.03,
            "hard_trigger_pct": 0.35,
            "hard_lock_pct": 0.20,
            "reversal_momentum_loss": True,
            "reversal_absorption_min": 0.50,
            "reversal_velocity_loss": True,
        },
    }


def test_profit_lock_soft_buy_at_20pct_tp() -> None:
    """BUY: a 20% del TP → SL = entry + 3% del recorrido entry→TP."""
    # tp_r=2.0 (tp_pips=20, sl_pips=10). progress=0.20 → current_r=0.40.
    trade = {
        "direction": "BUY",
        "entry_price": 100.0,
        "risk_price": 1.0,          # 1 unidad de precio por R
        "sl_pips": 10.0,
        "tp_pips": 20.0,
        "current_r": 0.40,          # 20% del TP
        "max_r": 0.40,
    }
    sl, stage = compute_profit_lock_sl_from_tp(trade, _ms_base(), _plock_cfg(), mid=100.4)
    assert stage == "soft"
    # tp_distance = (20/10)*1.0 = 2.0; lock_sl = 100 + 0.03*2.0 = 100.06
    assert abs(sl - 100.06) < 1e-9


def test_profit_lock_hard_buy_at_35pct_with_reversal() -> None:
    """BUY: a 35% del TP + reversión → SL = entry + 20% del recorrido."""
    trade = {
        "direction": "BUY",
        "entry_price": 100.0,
        "risk_price": 1.0,
        "sl_pips": 10.0,
        "tp_pips": 20.0,
        "current_r": 0.70,          # 35% del TP (tp_r=2.0)
        "max_r": 0.80,
    }
    ms = _ms_base()
    ms["continuation_score"] = 0.10  # momentum perdido → reversal
    sl, stage = compute_profit_lock_sl_from_tp(trade, ms, _plock_cfg(), mid=100.7)
    assert stage == "hard"
    # lock_sl = 100 + 0.20*2.0 = 100.40
    assert abs(sl - 100.40) < 1e-9


def test_profit_lock_hard_sell_symmetric() -> None:
    """SELL: 35% del TP + reversión → SL = entry - 20% del recorrido."""
    trade = {
        "direction": "SELL",
        "entry_price": 100.0,
        "risk_price": 1.0,
        "sl_pips": 10.0,
        "tp_pips": 20.0,
        "current_r": 0.75,
        "max_r": 0.80,
    }
    ms = _ms_base()
    ms["absorption_against_short"] = 0.80  # reversal vs short
    sl, stage = compute_profit_lock_sl_from_tp(trade, ms, _plock_cfg(), mid=99.3)
    assert stage == "hard"
    # SELL: lock_sl = 100 - 0.20*2.0 = 99.60
    assert abs(sl - 99.60) < 1e-9


def test_profit_lock_below_soft_trigger_returns_none() -> None:
    """Si progress < soft_trigger_pct → no aplica lock."""
    trade = {
        "direction": "BUY",
        "entry_price": 100.0,
        "risk_price": 1.0,
        "sl_pips": 10.0,
        "tp_pips": 20.0,
        "current_r": 0.30,          # 15% del TP, debajo de 0.20
        "max_r": 0.30,
    }
    sl, stage = compute_profit_lock_sl_from_tp(trade, _ms_base(), _plock_cfg(), mid=100.3)
    assert sl is None
    assert stage == ""


def test_profit_lock_35pct_without_reversal_stays_soft() -> None:
    """Si llegamos a 35% TP pero NO hay reversión → se queda en lock soft."""
    trade = {
        "direction": "BUY",
        "entry_price": 100.0,
        "risk_price": 1.0,
        "sl_pips": 10.0,
        "tp_pips": 20.0,
        "current_r": 0.70,
        "max_r": 0.80,
    }
    ms = _ms_base()
    # Forzamos NO reversión: momentum fuerte, sin absorción, sin velocity_loss
    ms["continuation_score"] = 0.80
    ms["velocity_loss"] = False
    ms["lateral_soft"] = False
    ms["absorption_against_long"] = 0.10
    sl, stage = compute_profit_lock_sl_from_tp(trade, ms, _plock_cfg(), mid=100.7)
    assert stage == "soft"
    assert abs(sl - 100.06) < 1e-9


def test_profit_lock_hard_overrides_giveback_close() -> None:
    """Con hard lock activo, giveback NO debe cerrar el trade (deja correr)."""
    cfg = _plock_cfg()
    cfg["secure_profit_r"] = 0.45
    cfg["max_giveback_allowed"] = 0.30
    cfg["momentum_loss_threshold"] = 0.5
    trade = {
        "ticket": "lk1",
        "direction": "BUY",
        "entry_price": 100.0,
        "risk_price": 1.0,
        "sl_pips": 10.0,
        "tp_pips": 20.0,
        "current_r": 0.75,           # >= 35% del TP
        "max_r": 1.20,                # giveback = 0.45 > 0.30 (normal cerraría)
        "be_done": False,
        "dynamic_exit_partial_done": False,
    }
    ms = _ms_base()
    ms["continuation_score"] = 0.10   # momentum perdido → reversal
    ms["absorption_against_long"] = 0.40  # absorción moderada, NO extrema
    dec = decide_dynamic_exit(trade, ms, cfg, mid=100.7)
    assert dec.close_full is False, "hard lock activo debe bypasear giveback"
    assert dec.trail_sl_price is not None
    assert "profit_lock" in dec.reason_trail or "atr" in dec.reason_trail

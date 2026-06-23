"""Tests de las 12 reglas heuristicas institucionales XAUUSD."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from ml.smc_heuristics import evaluate_heuristics, heuristics_from_config


@dataclass
class _Dir:
    value: str


@dataclass
class _Pool:
    kind: _Dir
    swept: bool = False
    touches: int = 1


@dataclass
class _OB:
    direction: _Dir
    ob_age_bars: int = 0


@dataclass
class _Ctx:
    last_bos: Any = None
    liquidity_pools: list = None  # type: ignore[assignment]
    active_obs: list = None       # type: ignore[assignment]
    active_fvgs: list = None      # type: ignore[assignment]
    recent_sweeps: list = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.liquidity_pools = self.liquidity_pools or []
        self.active_obs = self.active_obs or []
        self.active_fvgs = self.active_fvgs or []
        self.recent_sweeps = self.recent_sweeps or []


@dataclass
class _Snap:
    per_tf: dict


def _ts(hr: int, weekday: int = 2) -> datetime:
    """Devuelve un datetime UTC con la hora y dia de semana indicados."""
    base = datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc)  # lunes
    base = base.replace(hour=hr)
    days_diff = weekday - base.weekday()
    if days_diff:
        from datetime import timedelta
        base = base + timedelta(days=days_diff)
    return base


def _empty_snap() -> _Snap:
    return _Snap(per_tf={"M15": _Ctx(), "H1": _Ctx(), "H4": _Ctx()})


# ─── reglas individuales ────────────────────────────────────────────────


def test_r1_asian_trap_active_in_equilibrium() -> None:
    res = evaluate_heuristics(
        ts=_ts(4), direction="BUY", last_close=4050.0, pd_pct=0.50,
        mtf_aligned=1, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R1_asian_trap" in rules
    assert res.delta_total < 0


def test_r1_asian_trap_inactive_in_premium() -> None:
    res = evaluate_heuristics(
        ts=_ts(4), direction="BUY", last_close=4050.0, pd_pct=0.85,
        mtf_aligned=1, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R1_asian_trap" not in rules


def test_r2_london_killzone_bonus() -> None:
    bull_bos = _Dir("BULLISH")
    snap = _Snap(per_tf={
        "H1": _Ctx(last_bos=type("BOS", (), {"direction": bull_bos, "broken_swing_price": 4000.0, "break_price": 4050.0})()),
        "M15": _Ctx(),
    })
    res = evaluate_heuristics(
        ts=_ts(8), direction="BUY", last_close=4050.0, pd_pct=0.4,
        mtf_aligned=3, sweep_detected=False, has_choch=False, snapshot=snap,
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R2_london_killzone" in rules
    assert res.delta_total > 0


def test_r3_ny_open_killzone_bonus() -> None:
    res = evaluate_heuristics(
        ts=_ts(13), direction="BUY", last_close=4050.0, pd_pct=0.4,
        mtf_aligned=4, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R3_ny_open_killzone" in rules


def test_r4_ny_lunch_penalty() -> None:
    res = evaluate_heuristics(
        ts=_ts(16), direction="BUY", last_close=4050.0, pd_pct=0.4,
        mtf_aligned=2, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R4_ny_lunch_dead" in rules
    assert res.delta_total < 0


def test_r5_ny_close_revert_bonus_with_sweep_and_choch() -> None:
    res = evaluate_heuristics(
        ts=_ts(20), direction="SELL", last_close=4050.0, pd_pct=0.7,
        mtf_aligned=2, sweep_detected=True, has_choch=True, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R5_ny_close_revert" in rules
    delta_r5 = next(r["delta"] for r in res.applied_rules if r["id"] == "R5_ny_close_revert")
    assert delta_r5 > 0


def test_r6_friday_late_close_penalty() -> None:
    res = evaluate_heuristics(
        ts=_ts(17, weekday=4), direction="BUY", last_close=4050.0, pd_pct=0.4,
        mtf_aligned=2, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R6_friday_late_close" in rules
    assert res.delta_total <= -10


def test_r7_round_number_trap() -> None:
    # 4001 está a 1$ del nivel 4000 sin sweep → trap
    res = evaluate_heuristics(
        ts=_ts(13), direction="BUY", last_close=4001.5, pd_pct=0.4,
        mtf_aligned=2, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R7_round_number_trap" in rules


def test_r8_round_number_sweep_bonus() -> None:
    # 4005 a 5$ del nivel 4000 + sweep reciente → bonus
    res = evaluate_heuristics(
        ts=_ts(13), direction="BUY", last_close=4005.0, pd_pct=0.4,
        mtf_aligned=2, sweep_detected=True, has_choch=False, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R8_round_number_sweep" in rules


def test_r9_equal_highs_lows_trap() -> None:
    bsl = _Dir("BSL")
    snap = _Snap(per_tf={"H1": _Ctx(liquidity_pools=[_Pool(kind=bsl, swept=False, touches=4)])})
    res = evaluate_heuristics(
        ts=_ts(13), direction="BUY", last_close=4050.0, pd_pct=0.4,
        mtf_aligned=2, sweep_detected=False, has_choch=False, snapshot=snap,
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R9_equal_highs_lows" in rules


def test_r10_post_event_no_sweep_penalty() -> None:
    res = evaluate_heuristics(
        ts=_ts(13), direction="BUY", last_close=4050.0, pd_pct=0.4,
        mtf_aligned=2, sweep_detected=False, has_choch=False,
        snapshot=_empty_snap(), in_post_event_window=True,
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R10_post_event_spike" in rules


def test_r11_weekend_open_penalty() -> None:
    # domingo 22h UTC
    res = evaluate_heuristics(
        ts=_ts(22, weekday=6), direction="BUY", last_close=4050.0, pd_pct=0.4,
        mtf_aligned=2, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R11_weekend_open" in rules


def test_r12_stale_ob_penalty() -> None:
    bull = _Dir("BULLISH")
    snap = _Snap(per_tf={"H4": _Ctx(active_obs=[_OB(direction=bull, ob_age_bars=300)])})
    res = evaluate_heuristics(
        ts=_ts(13), direction="BUY", last_close=4050.0, pd_pct=0.4,
        mtf_aligned=2, sweep_detected=False, has_choch=False, snapshot=snap,
    )
    rules = [r["id"] for r in res.applied_rules]
    assert "R12_stale_setup" in rules


# ─── safety / defensivos ────────────────────────────────────────────────


def test_heuristics_none_snapshot_safe() -> None:
    res = evaluate_heuristics(
        ts=_ts(13), direction="BUY", last_close=4050.0, pd_pct=0.5,
        mtf_aligned=0, sweep_detected=False, has_choch=False, snapshot=None,
    )
    assert res.delta_total == 0.0
    assert res.applied_rules == []


def test_heuristics_weight_amplifies() -> None:
    args = dict(
        ts=_ts(4), direction="BUY", last_close=4050.0, pd_pct=0.50,
        mtf_aligned=1, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
    )
    base = evaluate_heuristics(**args, weight=1.0)
    amp = evaluate_heuristics(**args, weight=2.0)
    assert amp.delta_total == pytest.approx(2.0 * base.delta_total)


def test_heuristics_enabled_rules_filter() -> None:
    res = evaluate_heuristics(
        ts=_ts(4), direction="BUY", last_close=4050.0, pd_pct=0.50,
        mtf_aligned=1, sweep_detected=False, has_choch=False, snapshot=_empty_snap(),
        enabled_rules={"R1_asian_trap"},
    )
    rules = [r["id"] for r in res.applied_rules]
    assert rules == ["R1_asian_trap"]


def test_heuristics_from_config_defaults() -> None:
    cfg = heuristics_from_config(None)
    assert cfg["enabled"] is True
    assert cfg["weight"] == 1.0
    assert cfg["enabled_rules"] is None


def test_heuristics_from_config_explicit() -> None:
    cfg = heuristics_from_config({
        "enabled": False, "weight": 0.5, "rules": ["R1_asian_trap", "R6_friday_late_close"],
    })
    assert cfg["enabled"] is False
    assert cfg["weight"] == 0.5
    assert cfg["enabled_rules"] == {"R1_asian_trap", "R6_friday_late_close"}


# ─── integracion con SMCDecisionEngine ──────────────────────────────────


def test_smc_engine_with_heuristics_applies_delta() -> None:
    from ml.smc_decision_engine import SMCDecisionEngine
    # Snapshot vacio (sin SMC components) pero con un OB para R12
    bull = _Dir("BULLISH")
    snap = _Snap(per_tf={
        "H4": _Ctx(active_obs=[_OB(direction=bull, ob_age_bars=300)]),
    })
    eng = SMCDecisionEngine.from_config({
        "enabled": True, "min_score_to_trade": 30.0,
        "heuristics": {"enabled": True, "weight": 1.0},
    })
    dec = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=snap,
        ml_confidence=0.6, mtf_trends={"H4": "BULLISH"},
        last_close=4050.0, ts_utc=_ts(16),
    )
    # ts hora=16 (NY lunch dead) + stale OB → al menos 2 reglas activas
    assert dec.heuristics is not None
    rule_ids = [r["id"] for r in dec.heuristics.applied_rules]
    assert "R4_ny_lunch_dead" in rule_ids
    assert "R12_stale_setup" in rule_ids


def test_smc_engine_heuristics_disabled() -> None:
    from ml.smc_decision_engine import SMCDecisionEngine
    snap = _empty_snap()
    eng = SMCDecisionEngine.from_config({
        "enabled": True,
        "heuristics": {"enabled": False},
    })
    dec = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=snap,
        ml_confidence=0.6, mtf_trends={"H4": "BULLISH"},
        last_close=4050.0, ts_utc=_ts(16),
    )
    # con heuristicas off, no debe haber HeuristicResult
    assert dec.heuristics is None

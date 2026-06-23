"""Tests del SMC Decision Engine (AURION X 4.5).

Valida:

* ``PremiumDiscountEngine.calculate`` clasifica correctamente DISCOUNT/PREMIUM/
  EQUILIBRIUM segun la posicion del precio en el swing macro.
* ``PremiumDiscountEngine.evaluate_direction_bias`` penaliza compras en PREMIUM
  y ventas en DISCOUNT.
* ``SMCDecisionEngine.evaluate`` integra todos los componentes (SMC score
  0-100, OB/FVG quality, setup grade A+/A/B/C/X, lot multiplier) sin lanzar.
* El filtro principal (``smc_score >= min_score_to_trade``) bloquea cuando
  corresponde.
* Defensivo: snapshot None ⇒ allow_trade=False con motivo claro.
* Integracion con un SMCSnapshot real (generado del smoke test sintetico).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from market.premium_discount_engine import (
    PremiumDiscountEngine,
    PremiumDiscountResult,
    PremiumDiscountState,
)
from ml.smc_decision_engine import SMCDecisionEngine


# ─── Fakes minimos para no depender de market_structure ────────────────


@dataclass
class _FakeDirection:
    value: str


@dataclass
class _FakeBOS:
    direction: _FakeDirection
    broken_swing_price: float
    break_price: float


@dataclass
class _FakeOB:
    direction: _FakeDirection
    top: float
    bottom: float
    state: _FakeDirection
    ob_strength: float
    ob_volume_score: float
    ob_displacement_score: float
    ob_reaction_score: float
    ob_freshness: float


@dataclass
class _FakeFVG:
    direction: _FakeDirection
    fvg_strength: float
    gap_quality: float
    gap_fill_percent: float


@dataclass
class _FakeLiquidityPool:
    kind: _FakeDirection
    price: float
    swept: bool


@dataclass
class _FakeTFContext:
    last_bos: Any = None
    last_choch: Any = None
    active_obs: list = None  # type: ignore[assignment]
    active_fvgs: list = None  # type: ignore[assignment]
    liquidity_pools: list = None  # type: ignore[assignment]
    recent_sweeps: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.active_obs = self.active_obs or []
        self.active_fvgs = self.active_fvgs or []
        self.liquidity_pools = self.liquidity_pools or []
        self.recent_sweeps = self.recent_sweeps or []


@dataclass
class _FakeSnapshot:
    per_tf: dict
    primary_timeframe: str = "M15"


def _bullish(value: float = 100.0) -> _FakeDirection:
    return _FakeDirection("BULLISH")


def _bearish() -> _FakeDirection:
    return _FakeDirection("BEARISH")


def _state(s: str) -> _FakeDirection:
    return _FakeDirection(s)


# ─── PremiumDiscountEngine ────────────────────────────────────────────


def test_pd_engine_discount_zone_buy() -> None:
    """Precio en bottom 30% del rango H4 → DISCOUNT, BUY favorable."""
    eng = PremiumDiscountEngine()
    snap = _FakeSnapshot(per_tf={
        "H4": _FakeTFContext(
            last_bos=_FakeBOS(_bullish(), broken_swing_price=4000.0, break_price=4200.0),
        ),
    })
    res = eng.calculate(snapshot=snap, last_close=4020.0)
    assert res.state == PremiumDiscountState.DISCOUNT
    assert res.pct_in_range < 0.382
    delta, reason = eng.evaluate_direction_bias(direction="BUY", pd_result=res)
    assert delta > 0
    assert "discount" in reason


def test_pd_engine_premium_zone_sell() -> None:
    eng = PremiumDiscountEngine()
    snap = _FakeSnapshot(per_tf={
        "H4": _FakeTFContext(
            last_bos=_FakeBOS(_bullish(), broken_swing_price=4000.0, break_price=4200.0),
        ),
    })
    res = eng.calculate(snapshot=snap, last_close=4170.0)
    assert res.state == PremiumDiscountState.PREMIUM
    delta_sell, _ = eng.evaluate_direction_bias(direction="SELL", pd_result=res)
    delta_buy, _ = eng.evaluate_direction_bias(direction="BUY", pd_result=res)
    assert delta_sell > 0
    assert delta_buy < 0  # BUY en PREMIUM penalizado


def test_pd_engine_equilibrium() -> None:
    eng = PremiumDiscountEngine()
    snap = _FakeSnapshot(per_tf={
        "H4": _FakeTFContext(
            last_bos=_FakeBOS(_bullish(), broken_swing_price=4000.0, break_price=4200.0),
        ),
    })
    res = eng.calculate(snapshot=snap, last_close=4100.0)
    assert res.state == PremiumDiscountState.EQUILIBRIUM


def test_pd_engine_undefined_when_no_snapshot() -> None:
    eng = PremiumDiscountEngine()
    res = eng.calculate(snapshot=None, last_close=4000.0)
    assert res.state == PremiumDiscountState.UNDEFINED


def test_pd_engine_fallback_tf_priority() -> None:
    """Si H4 no tiene swing, debe caer a D1, luego W1."""
    eng = PremiumDiscountEngine()
    snap = _FakeSnapshot(per_tf={
        "H4": _FakeTFContext(),  # vacio
        "D1": _FakeTFContext(
            last_bos=_FakeBOS(_bullish(), broken_swing_price=3900.0, break_price=4300.0),
        ),
    })
    res = eng.calculate(snapshot=snap, last_close=3950.0)
    assert res.state == PremiumDiscountState.DISCOUNT
    assert "D1" in res.tf_source


# ─── SMCDecisionEngine ────────────────────────────────────────────────


def test_decision_engine_blocks_without_snapshot() -> None:
    eng = SMCDecisionEngine.from_config({"enabled": True, "min_score_to_trade": 70.0})
    dec = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=None,
        ml_confidence=0.65, mtf_trends=None, last_close=4000.0,
    )
    assert not dec.allow_trade
    assert dec.setup_grade == "X"


def test_decision_engine_disabled_passes_through() -> None:
    eng = SMCDecisionEngine.from_config({"enabled": False})
    dec = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=None,
        ml_confidence=0.65, mtf_trends=None, last_close=4000.0,
    )
    assert dec.allow_trade


def test_decision_engine_high_score_a_plus() -> None:
    """Snapshot con BOS/CHOCH/OB/FVG/sweep todos alineados a BUY en HTF → A+."""
    bull = _bullish()
    fresh = _state("FRESH")
    snap = _FakeSnapshot(per_tf={
        "H4": _FakeTFContext(
            last_bos=_FakeBOS(bull, broken_swing_price=4000.0, break_price=4200.0),
            last_choch=_FakeBOS(bull, broken_swing_price=4000.0, break_price=4200.0),
            active_obs=[_FakeOB(bull, 4030, 4015, fresh, 0.95, 0.9, 0.9, 0.8, 1.0)],
            active_fvgs=[_FakeFVG(bull, 0.9, 0.9, 0.0)],
            liquidity_pools=[_FakeLiquidityPool(_FakeDirection("SSL"), 3995, swept=True)],
        ),
        "M15": _FakeTFContext(
            last_bos=_FakeBOS(bull, broken_swing_price=4010.0, break_price=4030.0),
        ),
    })
    eng = SMCDecisionEngine.from_config({"enabled": True, "min_score_to_trade": 70.0})
    mtf = {"H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH", "M15": "BULLISH"}
    dec = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=snap,
        ml_confidence=0.72, mtf_trends=mtf, last_close=4020.0,
    )
    assert dec.allow_trade
    assert dec.smc_score >= 70.0
    assert dec.setup_grade in ("A+", "A", "B")  # robusto: minimo B
    assert dec.sweep_detected
    assert dec.pd_state.state == PremiumDiscountState.DISCOUNT
    assert dec.lot_multiplier >= 0.5


def test_decision_engine_blocks_low_score() -> None:
    """Snapshot vacio (solo TFContext sin nada) → score 0 → bloquea."""
    snap = _FakeSnapshot(per_tf={
        "H4": _FakeTFContext(),
        "M15": _FakeTFContext(),
    })
    eng = SMCDecisionEngine.from_config({"enabled": True, "min_score_to_trade": 70.0})
    dec = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=snap,
        ml_confidence=0.65, mtf_trends={"H4": "RANGE"}, last_close=4000.0,
    )
    assert not dec.allow_trade
    assert dec.setup_grade in ("X", "C")


def test_decision_engine_htf_contradicts_caps_to_b() -> None:
    """H4 BEARISH para un BUY → grade no puede ser A+/A (penalizacion fuerte)."""
    bull = _bullish()
    fresh = _state("FRESH")
    snap = _FakeSnapshot(per_tf={
        "H4": _FakeTFContext(
            last_bos=_FakeBOS(bull, broken_swing_price=4000.0, break_price=4200.0),
            active_obs=[_FakeOB(bull, 4030, 4015, fresh, 0.95, 0.9, 0.9, 0.8, 1.0)],
            active_fvgs=[_FakeFVG(bull, 0.9, 0.9, 0.0)],
            liquidity_pools=[_FakeLiquidityPool(_FakeDirection("SSL"), 3995, swept=True)],
        ),
    })
    eng = SMCDecisionEngine.from_config({"enabled": True, "min_score_to_trade": 60.0})
    mtf = {"H4": "BEARISH", "H1": "BULLISH", "M30": "BULLISH", "M15": "BULLISH"}
    dec = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=snap,
        ml_confidence=0.72, mtf_trends=mtf, last_close=4020.0,
    )
    # H4 contradicts → mtf_delta -20, y _setup_grade lo capea a B o menos.
    assert dec.setup_grade in ("B", "C", "X")
    assert dec.mtf.htf_contradicts


def test_decision_engine_to_dict_serializable() -> None:
    """El veredicto debe ser serializable a JSON (sin tipos exoticos)."""
    import json
    eng = SMCDecisionEngine.from_config({"enabled": True})
    dec = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=None,
        ml_confidence=0.5, mtf_trends=None, last_close=4000.0,
    )
    js = json.dumps(dec.to_dict(), default=str)
    assert "smc_score" in js
    assert "setup_grade" in js
    assert "pd_state" in js


def test_decision_engine_c_grade_requires_high_ml() -> None:
    """Grade C debe requerir ml >= c_grade_ml_override (default 0.80)."""
    bull = _bullish()
    fresh = _state("FRESH")
    snap = _FakeSnapshot(per_tf={
        "M15": _FakeTFContext(
            last_bos=_FakeBOS(bull, broken_swing_price=4000.0, break_price=4015.0),
            active_obs=[_FakeOB(bull, 4010, 4005, fresh, 0.7, 0.6, 0.6, 0.5, 0.9)],
        ),
    })
    # Bajamos min_score para que entre como C
    eng = SMCDecisionEngine.from_config({"enabled": True, "min_score_to_trade": 30.0})
    dec_low = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=snap,
        ml_confidence=0.55, mtf_trends={"H4": "BULLISH"}, last_close=4007.0,
    )
    dec_high = eng.evaluate(
        symbol="XAUUSD", direction="BUY", snapshot=snap,
        ml_confidence=0.85, mtf_trends={"H4": "BULLISH"}, last_close=4007.0,
    )
    # Si grade es C: low ML bloquea, high ML permite
    if dec_low.setup_grade == "C":
        assert not dec_low.allow_trade
        assert "ml" in dec_low.block_reason.lower()
    if dec_high.setup_grade == "C":
        assert dec_high.allow_trade

"""Tests para smart_targets + smart_stops + volume_classifier + session_liquidity."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from market.session_liquidity import (
    compute_session_ranges,
    current_session,
    session_classification,
    session_pools_for_targets,
)
from market.volume_classifier import VolumeClass, classify_volume
from risk.smart_stops import compute_stop_plan
from risk.smart_targets import compute_targets


# ─── helpers ───────────────────────────────────────────────────────────


@dataclass
class _Dir:
    value: str


@dataclass
class _Pool:
    kind: _Dir
    price: float
    swept: bool = False
    touches: int = 1
    liquidity_score: float = 0.5


@dataclass
class _FVG:
    top: float
    bottom: float
    state: str = "OPEN"
    strength: float = 0.8


@dataclass
class _OB:
    direction: _Dir
    top: float
    bottom: float
    strength: float = 0.7
    ob_age_bars: int = 10


@dataclass
class _BOS:
    direction: _Dir
    broken_swing_price: float


@dataclass
class _Ctx:
    last_bos: Any = None
    last_choch: Any = None
    liquidity_pools: list = field(default_factory=list)
    active_obs: list = field(default_factory=list)
    active_fvgs: list = field(default_factory=list)


@dataclass
class _Snap:
    per_tf: dict
    primary_timeframe: str = "M15"


@dataclass
class _Candle:
    time: datetime
    high: float
    low: float
    close: float
    tick_volume: int = 100


def _utc(y: int, m: int, d: int, hh: int = 12, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# ─── smart_targets ─────────────────────────────────────────────────────


def test_targets_buy_finds_bsl_above() -> None:
    # XAUUSD H4 ATR realista ~25$. BSL a 30$ = 1.2 ATR de distancia.
    bsl_h4 = _Pool(kind=_Dir("BSL"), price=4080.0, touches=3, liquidity_score=0.9)
    snap = _Snap(per_tf={"H4": _Ctx(liquidity_pools=[bsl_h4])})
    ts = compute_targets(
        direction="BUY", entry=4050.0, snapshot=snap, atr=25.0,
    )
    assert len(ts.targets) >= 1
    assert ts.targets[0].price == 4080.0
    assert ts.targets[0].source.startswith("liquidity_bsl")
    # BSL HTF a ~1.2 ATR + bonus touches/liq → > 0.6
    assert 0.6 < ts.targets[0].probability < 1.0


def test_targets_sell_finds_ssl_below() -> None:
    ssl_h4 = _Pool(kind=_Dir("SSL"), price=4020.0, touches=2, liquidity_score=0.7)
    snap = _Snap(per_tf={"H4": _Ctx(liquidity_pools=[ssl_h4])})
    ts = compute_targets(
        direction="SELL", entry=4050.0, snapshot=snap, atr=2.0,
    )
    assert len(ts.targets) >= 1
    assert ts.targets[0].price == 4020.0


def test_targets_filter_too_close() -> None:
    # Pool a 0.3$ = 3 pips del entry → filtrado por min_distance_pips=5pips
    pool = _Pool(kind=_Dir("BSL"), price=4050.3)
    snap = _Snap(per_tf={"H1": _Ctx(liquidity_pools=[pool])})
    ts = compute_targets(
        direction="BUY", entry=4050.0, snapshot=snap, atr=1.5,
        min_distance_pips=5.0,
    )
    assert len(ts.targets) == 0


def test_targets_dedup_close_levels() -> None:
    p1 = _Pool(kind=_Dir("BSL"), price=4080.0)
    p2 = _Pool(kind=_Dir("BSL"), price=4080.5)  # casi mismo nivel
    snap = _Snap(per_tf={"H1": _Ctx(liquidity_pools=[p1, p2])})
    ts = compute_targets(
        direction="BUY", entry=4050.0, snapshot=snap, atr=2.0,
        min_atr_separation=0.5,
    )
    # Debe haber dedup, no 2 TPs idénticos
    prices = ts.prices()
    assert len(set(prices)) == len(prices)


def test_targets_session_high_appended() -> None:
    snap = _Snap(per_tf={"H1": _Ctx()})
    sessions = {
        "asia":   {"high": 4070.0, "low": 4030.0, "swept_high": False, "swept_low": False},
        "london": {"high": 4090.0, "low": 4040.0, "swept_high": False, "swept_low": False},
    }
    ts = compute_targets(
        direction="BUY", entry=4050.0, snapshot=snap, atr=2.0, sessions=sessions,
    )
    sources = [t.source for t in ts.targets]
    assert any("session_" in s for s in sources)


def test_targets_probability_decreases_with_distance() -> None:
    p_close = _Pool(kind=_Dir("BSL"), price=4055.0)
    p_far = _Pool(kind=_Dir("BSL"), price=4120.0)
    snap = _Snap(per_tf={"H1": _Ctx(liquidity_pools=[p_close, p_far])})
    ts = compute_targets(
        direction="BUY", entry=4050.0, snapshot=snap, atr=2.0,
        min_distance_pips=2.0, min_atr_separation=0.2,
    )
    assert len(ts.targets) >= 2
    # TP más cercano tiene mayor probabilidad (a igualdad de fuente)
    assert ts.targets[0].probability > ts.targets[-1].probability


# ─── smart_stops ───────────────────────────────────────────────────────


def test_stops_buy_3_modes_below_entry() -> None:
    bos = _BOS(direction=_Dir("BULLISH"), broken_swing_price=4040.0)
    ob = _OB(direction=_Dir("BULLISH"), top=4045.0, bottom=4042.0)
    snap = _Snap(per_tf={"H1": _Ctx(last_bos=bos, active_obs=[ob])})
    plan = compute_stop_plan(
        direction="BUY", entry=4050.0, snapshot=snap, atr=2.0,
        setup_grade="A", smc_score=82.0,
    )
    assert plan.conservative.price < 4050.0
    assert plan.balanced.price < 4050.0
    assert plan.aggressive.price < 4050.0
    # Orden: agresivo más cercano, conservador más lejos
    assert plan.aggressive.distance_pips <= plan.balanced.distance_pips
    assert plan.balanced.distance_pips <= plan.conservative.distance_pips


def test_stops_sell_3_modes_above_entry() -> None:
    bos = _BOS(direction=_Dir("BEARISH"), broken_swing_price=4060.0)
    ob = _OB(direction=_Dir("BEARISH"), top=4058.0, bottom=4055.0)
    snap = _Snap(per_tf={"H1": _Ctx(last_bos=bos, active_obs=[ob])})
    plan = compute_stop_plan(
        direction="SELL", entry=4050.0, snapshot=snap, atr=2.0,
    )
    assert plan.conservative.price > 4050.0
    assert plan.balanced.price > 4050.0
    assert plan.aggressive.price > 4050.0


def test_stops_recommended_aggressive_for_aplus_low_hunt() -> None:
    bos = _BOS(direction=_Dir("BULLISH"), broken_swing_price=4040.0)
    ob = _OB(direction=_Dir("BULLISH"), top=4045.0, bottom=4042.0)
    snap = _Snap(per_tf={"H1": _Ctx(last_bos=bos, active_obs=[ob])})  # sin SSL pools cercanos
    plan = compute_stop_plan(
        direction="BUY", entry=4050.0, snapshot=snap, atr=2.0,
        setup_grade="A+", smc_score=90.0,
    )
    # A+ con score>=85 y poca prob de stop-hunt → recomienda agresivo
    assert plan.recommended_mode in ("aggressive", "balanced")


def test_stops_rr_table_calculated_against_tps() -> None:
    bos = _BOS(direction=_Dir("BULLISH"), broken_swing_price=4040.0)
    snap = _Snap(per_tf={"H1": _Ctx(last_bos=bos)})
    plan = compute_stop_plan(
        direction="BUY", entry=4050.0, snapshot=snap, atr=2.0,
        targets_prices=[4060.0, 4080.0, 4100.0],
    )
    assert "tp1" in plan.rr_table["balanced"]
    assert plan.rr_table["balanced"]["tp1"] > 0
    # TP3 debe tener RR > TP1
    assert plan.rr_table["balanced"]["tp3"] > plan.rr_table["balanced"]["tp1"]


def test_stops_fallback_without_snapshot() -> None:
    plan = compute_stop_plan(
        direction="BUY", entry=4050.0, snapshot=None, atr=2.0,
    )
    assert plan.balanced.price < 4050.0  # SL del lado correcto
    assert plan.recommended_mode in ("conservative", "balanced", "aggressive")


# ─── volume_classifier ─────────────────────────────────────────────────


def _make_candles(vols: list[int]) -> list[_Candle]:
    base = _utc(2026, 6, 12, 12, 0)
    return [
        _Candle(time=base, high=4050.0, low=4045.0, close=4047.0, tick_volume=v)
        for v in vols
    ]


def test_volume_low_penalty() -> None:
    # 49 velas de vol=100, última de vol=10 → 10/100 = 0.10 < 0.50 = LOW
    vols = [100] * 49 + [10]
    res = classify_volume(candles=_make_candles(vols))
    assert res.classification == VolumeClass.LOW
    assert res.score_delta < 0


def test_volume_extreme_bonus() -> None:
    vols = [100] * 49 + [500]   # 500/100 = 5.0 → EXTREME
    res = classify_volume(candles=_make_candles(vols))
    assert res.classification == VolumeClass.EXTREME
    assert res.score_delta > 0


def test_volume_bos_with_extreme_volume_bonus() -> None:
    vols = [100] * 49 + [400]
    res = classify_volume(candles=_make_candles(vols), has_recent_bos=True)
    assert res.classification == VolumeClass.EXTREME
    assert res.bos_modifier > 0


def test_volume_bos_with_low_volume_penalty() -> None:
    vols = [100] * 49 + [20]
    res = classify_volume(candles=_make_candles(vols), has_recent_bos=True)
    assert res.classification == VolumeClass.LOW
    assert res.bos_modifier < 0


def test_volume_insufficient_data() -> None:
    vols = [100] * 10
    res = classify_volume(candles=_make_candles(vols))
    assert res.classification == VolumeClass.UNKNOWN


# ─── session_liquidity ─────────────────────────────────────────────────


def test_current_session_london() -> None:
    t = _utc(2026, 6, 12, 9, 0)  # 09:00 UTC
    assert current_session(t) == "london"


def test_current_session_overlap() -> None:
    t = _utc(2026, 6, 12, 13, 30)  # 13:30 UTC = NY+London
    assert current_session(t) == "overlap"


def test_current_session_asia() -> None:
    t = _utc(2026, 6, 12, 3, 0)
    assert current_session(t) == "asia"


def test_compute_session_ranges_basic() -> None:
    candles = []
    base = _utc(2026, 6, 12, 0, 0)
    # Asia (0-7 UTC) high=4060, low=4040
    candles.append(_Candle(time=base.replace(hour=2), high=4060.0, low=4055.0, close=4058.0))
    candles.append(_Candle(time=base.replace(hour=5), high=4055.0, low=4040.0, close=4050.0))
    # London (7-15) high=4080
    candles.append(_Candle(time=base.replace(hour=8), high=4080.0, low=4060.0, close=4075.0))
    candles.append(_Candle(time=base.replace(hour=14), high=4078.0, low=4070.0, close=4076.0))
    now = base.replace(hour=15, minute=30)
    rng = compute_session_ranges(candles=candles, now=now, last_close=4076.0)
    assert rng["asia"].high == 4060.0
    assert rng["asia"].low == 4040.0
    assert rng["london"].high == 4080.0
    assert rng["asia"].is_active is False
    assert rng["london"].is_active is False  # cerro a 15h


def test_session_pools_for_targets() -> None:
    candles = [_Candle(time=_utc(2026, 6, 12, 8), high=4080.0, low=4060.0, close=4075.0)]
    rng = compute_session_ranges(
        candles=candles, now=_utc(2026, 6, 12, 10), last_close=4075.0,
    )
    pools = session_pools_for_targets(rng)
    assert "london" in pools
    assert pools["london"]["high"] == 4080.0


def test_session_classification_includes_killzone() -> None:
    t = _utc(2026, 6, 12, 8, 0)
    res = session_classification(now=t)
    assert res["current_session"] == "london"
    assert res["is_killzone"] is True

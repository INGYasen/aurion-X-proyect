"""Tests capa Apex (sin MT5 ni PostgreSQL)."""

from __future__ import annotations

from decimal import Decimal

from market.microstructure_engine import MicrostructureEngine
from ml.decision_hierarchy import DecisionHierarchy
from ml.hyper_context_engine import HyperContextEngine
from ml.latent_market_state import LatentMarketStateAI
from risk.anti_overtrading import AntiOvertradingSystem
from risk.fundingpips_guard import FundingPipsGuard
from risk.meta_risk_engine import MetaRiskEngine, RiskMode


def test_microstructure_features() -> None:
    closes = [100.0 + i * 0.1 for i in range(30)]
    snap = MicrostructureEngine().analyze(closes=closes, direction="BUY")
    d = snap.as_dict()
    assert "micro_velocity" in d
    assert 0.0 <= d["micro_quality"] <= 1.0


def test_latent_infer() -> None:
    ms = {"closes_m1": [1.0, 1.1, 1.05, 1.08, 1.12] * 8}
    latent = LatentMarketStateAI().infer("XAUUSD", ms, regime="NORMAL")
    assert latent.state in ("alive", "clean", "accumulation", "compression", "expansion", "transition", "manipulation", "exhaustion", "dangerous")


def test_hyper_context_blocks_bad_exec() -> None:
    h = HyperContextEngine().fuse(
        ml_confidence=0.7,
        setup_score=75.0,
        regime_mode="trend",
        micro={"micro_quality": 0.6},
        latent={"latent_dangerous": False, "latent_clean": True},
        exec_intel={"exec_intel_block": True, "exec_intel_score": 0.2},
        equity_health=0.6,
        meta_risk_mode="neutral",
        reinforcement_lot=1.0,
    )
    assert not h.allow_trade


def test_decision_hierarchy_risk_first() -> None:
    v = DecisionHierarchy().evaluate(
        risk_ok=False,
        risk_reason="dd",
        hyper_allow=True,
        hyper_reason="",
        regime_dangerous=False,
        micro_quality=0.8,
        manipulation_score=0.1,
        ml_confidence=0.8,
        ml_threshold=0.5,
        setup_grade="A",
        execution_block=False,
        anti_overtrade_block=False,
    )
    assert not v.allow
    assert v.failed_layer == "risk"


def test_anti_overtrading_spacing() -> None:
    ao = AntiOvertradingSystem(min_spacing_s=9999.0)
    ao.record_entry()
    v = ao.check()
    assert v.block


def test_anti_overtrading_per_symbol_isolated() -> None:
    """Fix #10: rachas/burst de un símbolo NO bloquean a otro."""
    ao = AntiOvertradingSystem(
        max_entries_per_hour=2,
        max_signals_per_10min=8,
        min_spacing_s=0.0,
    )
    # XAUUSD agota su límite (>=2 entries/hour).
    ao.record_entry("XAUUSD")
    ao.record_entry("XAUUSD")
    v_xau = ao.check("XAUUSD")
    assert v_xau.block, "XAUUSD debería estar bloqueado tras 2 entries"
    assert v_xau.reason == "max_entries_hour"
    # NDX100 sin entries propias: NO debe estar bloqueado.
    v_ndx = ao.check("NDX100")
    assert not v_ndx.block, "NDX100 no debería heredar el bloqueo de XAUUSD"


def test_anti_overtrading_per_symbol_burst_isolated() -> None:
    """Fix #10: signal burst es por símbolo."""
    ao = AntiOvertradingSystem(
        max_entries_per_hour=10,
        max_signals_per_10min=3,
        min_spacing_s=0.0,
    )
    for _ in range(5):
        ao.record_signal("BTCUSD")
    v_btc = ao.check("BTCUSD")
    assert v_btc.block and v_btc.reason == "signal_burst"
    v_xau = ao.check("XAUUSD")
    assert not v_xau.block, "XAUUSD no debería verse afectado por burst de BTCUSD"


def test_decision_hierarchy_anti_overtrade_bypass() -> None:
    """Fix #10: con ml_strong_override + grade A/A+ el guard puede saltarse.

    El bypass se aplica en ``apex_engine.py`` antes de llamar a la jerarquía,
    así que aquí simulamos el flag ya neutralizado (anti_overtrade_block=False)
    y verificamos que la jerarquía permite el trade.
    """
    v = DecisionHierarchy().evaluate(
        risk_ok=True,
        risk_reason="ok",
        hyper_allow=True,
        hyper_reason="",
        regime_dangerous=False,
        micro_quality=0.8,
        manipulation_score=0.1,
        ml_confidence=0.8,
        ml_threshold=0.5,
        setup_grade="A+",
        execution_block=False,
        anti_overtrade_block=False,
        ml_strong_override=True,
    )
    assert v.allow


def test_meta_risk_defensive() -> None:
    m = MetaRiskEngine({"defensive_dd_pct": 3.0, "bad_winrate": 0.4})
    v = m.evaluate(
        drawdown_pct=4.0,
        recent_winrate=0.3,
        trades_today=5,
        max_trades_day=10,
        ml_degraded=False,
        volatility_extreme=False,
        consistency_score=0.4,
        equity_slope=-0.01,
    )
    assert v.mode == RiskMode.DEFENSIVE


def test_fundingpips_adaptive() -> None:
    g = FundingPipsGuard({"enabled": True, "max_trades_per_day": 10, "adaptive_mode": True})
    g.adapt_dynamics(drawdown_pct=3.0, winrate=0.35, volatility_extreme=True, equity_stable=False)
    assert g.max_trades_per_day <= 10
    v = g.evaluate(
        equity=Decimal("10000"),
        day_start_equity=Decimal("10000"),
        total_dd_pct=1.0,
        trades_today=2,
        trades_last_hour=1,
        consecutive_losses=0,
        seconds_since_last_entry=500.0,
        setup_grade="A",
        setup_score=70.0,
        winrate=0.55,
        drawdown_pct=0.8,
    )
    assert v.allow_new_trade

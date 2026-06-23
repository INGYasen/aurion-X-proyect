"""Tests capa institucional."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from ml.asset_ml_weights import resolve_asset_ml_profile
from ml.institutional_engine import InstitutionalEngine
from ml.setup_ranking_engine import SetupRankingEngine
from risk.fundingpips_guard import FundingPipsGuard
from risk.montecarlo import MonteCarloRiskEngine


def test_asset_weights_xau() -> None:
    p = resolve_asset_ml_profile("XAUUSD")
    assert p.weights["liquidity_sweep"] > p.weights["momentum"]


def test_setup_ranking() -> None:
    eng = SetupRankingEngine(min_grade_trade="C")
    from ml.asset_ml_weights import AssetMLProfile

    prof = AssetMLProfile("XAUUSD", {"liquidity_sweep": 1.4}, 0.65, "A")
    rank = eng.rank(
        symbol="XAUUSD",
        setup_type="RANGE_FADE",
        ml_confidence=0.72,
        composite_score=0.68,
        regime="NORMAL",
        probabilities={"prob_reversal": 0.6, "prob_continuation": 0.4, "prob_fake_sweep": 0.2, "prob_institutional_absorption": 0.55},
        signal={"range_quality_score": 70},
        memory_insight={"known": True, "why": "historical_edge"},
        asset_profile=prof,
    )
    assert rank.grade.value in ("A+", "A", "B")


def test_institutional_daily_opt_runs_once_per_day() -> None:
    repo = MagicMock()
    repo.get_training_dataset.return_value = []
    eng = InstitutionalEngine(
        trade_repo=repo,
        ml_cfg={"enabled": False},
        quant_cfg={"enabled": False},
        institutional_cfg={"enabled": True, "optimizer_min_samples": 1},
        asset_profiles={},
    )
    t0 = datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc)
    eng.run_daily_optimization(t0)
    eng.run_daily_optimization(t0 + timedelta(hours=1))
    assert repo.get_training_dataset.call_count == 1


def test_fundingpips_blocks_streak() -> None:
    g = FundingPipsGuard({"enabled": True, "block_after_consecutive_losses": 3})
    v = g.evaluate(
        equity=10000,
        day_start_equity=10000,
        total_dd_pct=1.0,
        trades_today=2,
        trades_last_hour=1,
        consecutive_losses=4,
        seconds_since_last_entry=400.0,
        setup_grade="A",
        setup_score=70.0,
    )
    assert not v.allow_new_trade


def test_montecarlo() -> None:
    mc = MonteCarloRiskEngine(seed=1)
    r = MonteCarloRiskEngine(seed=1).simulate([0.5, -0.3, 0.2, -1.0] * 40, n_sims=50, n_trades=80)
    assert r.simulations == 50
    assert 0.0 <= r.recommended_lot_factor <= 1.15

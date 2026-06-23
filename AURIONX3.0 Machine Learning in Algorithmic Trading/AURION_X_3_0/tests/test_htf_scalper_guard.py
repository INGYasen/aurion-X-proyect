from __future__ import annotations

from core.orchestrator import Orchestrator
from strategy.structure_detector import Trend


def test_counter_htf_blocks_sell_vs_h1_up() -> None:
    o = Orchestrator()
    o._block_scalper_counter_htf = True
    trends = {
        "M1": Trend.DOWN,
        "M5": Trend.DOWN,
        "M15": Trend.RANGE,
        "H1": Trend.UP,
        "H4": Trend.RANGE,
    }
    sig = {"direction": "SELL", "setup_type": "MICRO_PULLBACK_MOMENTUM", "range_fade": False}
    blocked, _reason = o._counter_htf_scalper_block(sig, trends)
    assert blocked is True


def test_counter_htf_skips_range_fade() -> None:
    o = Orchestrator()
    o._block_scalper_counter_htf = True
    trends = {"H1": Trend.UP, "H4": Trend.UP}
    sig = {"direction": "SELL", "setup_type": "RANGE_FADE_REVERSAL", "range_fade": True}
    blocked, _reason = o._counter_htf_scalper_block(sig, trends)
    assert blocked is False

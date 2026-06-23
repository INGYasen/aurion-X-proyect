"""Cambio automático de modo operativo según régimen de mercado."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from market.volatility_detector import VolatilityRegime
from strategy.structure_detector import Trend


class TradingMode(str, Enum):
    SCALPING = "scalping"
    TREND_FOLLOWING = "trend_following"
    RANGE_TRADING = "range_trading"
    DEFENSIVE = "defensive"
    HIGH_VOLATILITY = "high_volatility"


@dataclass(frozen=True)
class RegimeSwitchDecision:
    mode: TradingMode
    ml_threshold_add: float
    score_bias: int
    max_trades_factor: float
    reason: str


class MarketRegimeSwitcher:
    """Selecciona estrategia dominante según volatilidad + estructura + drawdown."""

    def __init__(self, *, defensive_dd_pct: float = 4.0) -> None:
        self._defensive_dd = defensive_dd_pct

    def decide(
        self,
        *,
        volatility: VolatilityRegime,
        structure_trend: Trend,
        mtf_trends: Mapping[str, Trend],
        drawdown_pct: float,
        recent_winrate: float,
        spread_spike: bool,
    ) -> RegimeSwitchDecision:
        if drawdown_pct >= self._defensive_dd or recent_winrate < 0.32:
            return RegimeSwitchDecision(
                mode=TradingMode.DEFENSIVE,
                ml_threshold_add=0.06,
                score_bias=-8,
                max_trades_factor=0.5,
                reason="drawdown_or_winrate",
            )
        if spread_spike:
            return RegimeSwitchDecision(
                mode=TradingMode.DEFENSIVE,
                ml_threshold_add=0.04,
                score_bias=-5,
                max_trades_factor=0.6,
                reason="spread_spike",
            )

        if volatility == VolatilityRegime.EXTREME:
            return RegimeSwitchDecision(
                mode=TradingMode.HIGH_VOLATILITY,
                ml_threshold_add=0.05,
                score_bias=-3,
                max_trades_factor=0.75,
                reason="extreme_vol",
            )

        if structure_trend == Trend.RANGE:
            m5 = mtf_trends.get("M5", Trend.RANGE)
            m15 = mtf_trends.get("M15", Trend.RANGE)
            if m5 == Trend.RANGE and m15 == Trend.RANGE:
                return RegimeSwitchDecision(
                    mode=TradingMode.RANGE_TRADING,
                    ml_threshold_add=-0.01,
                    score_bias=2,
                    max_trades_factor=1.0,
                    reason="double_range_mtf",
                )
            return RegimeSwitchDecision(
                mode=TradingMode.RANGE_TRADING,
                ml_threshold_add=0.0,
                score_bias=0,
                max_trades_factor=0.9,
                reason="structure_range",
            )

        m1 = mtf_trends.get("M1", Trend.RANGE)
        h1 = mtf_trends.get("H1", Trend.RANGE)
        aligned = (
            structure_trend in (Trend.UP, Trend.DOWN)
            and m1 == structure_trend
            and h1 in (structure_trend, Trend.RANGE)
        )
        if aligned:
            return RegimeSwitchDecision(
                mode=TradingMode.TREND_FOLLOWING,
                ml_threshold_add=-0.02,
                score_bias=4,
                max_trades_factor=1.1,
                reason="trend_stack",
            )

        if volatility == VolatilityRegime.LOW:
            return RegimeSwitchDecision(
                mode=TradingMode.SCALPING,
                ml_threshold_add=0.02,
                score_bias=0,
                max_trades_factor=0.85,
                reason="low_vol_scalp",
            )

        return RegimeSwitchDecision(
            mode=TradingMode.SCALPING,
            ml_threshold_add=0.0,
            score_bias=0,
            max_trades_factor=1.0,
            reason="default_scalp",
        )

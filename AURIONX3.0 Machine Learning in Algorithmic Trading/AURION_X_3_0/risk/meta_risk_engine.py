"""Meta-riesgo: modo agresivo / neutral / defensivo dinámico."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RiskMode(str, Enum):
    AGGRESSIVE = "aggressive"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"


@dataclass(frozen=True)
class MetaRiskVerdict:
    mode: RiskMode
    lot_factor: float
    threshold_add: float
    max_trades_factor: float
    frequency_factor: float
    sensitivity: float
    reason: str


class MetaRiskEngine:
    """Ajusta exposición según salud reciente del sistema."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        c = dict(cfg or {})
        self._defensive_dd = float(c.get("defensive_dd_pct", 3.5))
        self._aggressive_wr = float(c.get("aggressive_winrate", 0.56))
        self._bad_wr = float(c.get("bad_winrate", 0.38))

    def evaluate(
        self,
        *,
        drawdown_pct: float,
        recent_winrate: float,
        trades_today: int,
        max_trades_day: int,
        ml_degraded: bool,
        volatility_extreme: bool,
        consistency_score: float,
        equity_slope: float,
    ) -> MetaRiskVerdict:
        if (
            drawdown_pct >= self._defensive_dd
            or recent_winrate < self._bad_wr
            or ml_degraded
            or (max_trades_day > 0 and trades_today >= max_trades_day * 0.9)
        ):
            return MetaRiskVerdict(
                mode=RiskMode.DEFENSIVE,
                lot_factor=0.62,
                threshold_add=0.05,
                max_trades_factor=0.55,
                frequency_factor=0.5,
                sensitivity=1.25,
                reason="defensive_trigger",
            )
        if volatility_extreme:
            return MetaRiskVerdict(
                mode=RiskMode.DEFENSIVE,
                lot_factor=0.75,
                threshold_add=0.03,
                max_trades_factor=0.7,
                frequency_factor=0.65,
                sensitivity=1.1,
                reason="extreme_vol",
            )
        if (
            recent_winrate >= self._aggressive_wr
            and drawdown_pct < 1.5
            and consistency_score >= 0.6
            and equity_slope > 0
        ):
            return MetaRiskVerdict(
                mode=RiskMode.AGGRESSIVE,
                lot_factor=1.08,
                threshold_add=-0.02,
                max_trades_factor=1.05,
                frequency_factor=1.0,
                sensitivity=0.92,
                reason="healthy_edge",
            )
        return MetaRiskVerdict(
            mode=RiskMode.NEUTRAL,
            lot_factor=0.92,
            threshold_add=0.01,
            max_trades_factor=0.85,
            frequency_factor=0.8,
            sensitivity=1.0,
            reason="neutral",
        )

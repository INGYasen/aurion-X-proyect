"""Inteligencia sobre la curva de equity propia."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EquityHealth:
    score: float
    slope: float
    deterioration: bool
    overtrading: bool
    stable_growth: bool
    drawdown_velocity: float
    lot_factor: float
    reason: str


class EquityCurveAI:
    """Analiza equity reciente para proteger capital."""

    def analyze(
        self,
        equity_points: Sequence[float],
        *,
        trades_per_day: int = 0,
        max_trades_day: int = 12,
    ) -> EquityHealth:
        pts = [float(x) for x in equity_points if x is not None]
        if len(pts) < 5:
            return EquityHealth(0.5, 0.0, False, False, False, 0.0, 1.0, "insufficient_data")

        start = pts[0]
        end = pts[-1]
        slope = (end - start) / max(abs(start), 1e-9)
        peak = max(pts)
        dd_vel = (peak - end) / max(peak, 1e-9) if peak > 0 else 0.0

        recent = pts[-10:]
        if len(recent) >= 3:
            down_moves = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
            deterior = down_moves >= max(2, len(recent) * 2 // 3) and end < peak * 0.985
        else:
            deterior = end < start * 0.99

        overtrade = max_trades_day > 0 and trades_per_day >= int(max_trades_day * 0.85)
        stable = slope > 0.002 and dd_vel < 0.02 and not deterior

        score = 0.5 + slope * 5.0 - dd_vel * 2.0
        if stable:
            score += 0.15
        if deterior:
            score -= 0.25
        if overtrade:
            score -= 0.15
        score = max(0.0, min(1.0, score))

        lot_f = 1.0
        if deterior or dd_vel > 0.04:
            lot_f = 0.65
        elif overtrade:
            lot_f = 0.78
        elif stable and score > 0.65:
            lot_f = 1.05

        reason = "stable" if stable else ("deterioration" if deterior else ("overtrading" if overtrade else "neutral"))
        return EquityHealth(score, slope, deterior, overtrade, stable, dd_vel, lot_f, reason)

"""Calidad de ejecución: spread, slippage, latencia aprendibles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class ExecutionSample:
    spread_points: float
    slippage_points: float
    fill_ok: bool
    latency_ms: float


@dataclass
class ExecutionQualityReport:
    score: float
    spread_penalty: float
    slippage_penalty: float
    fill_rate: float
    recommendation: str


class ExecutionQualityEngine:
    """Acumula muestras de ejecución y penaliza condiciones malas."""

    def __init__(self, *, window: int = 80) -> None:
        self._samples: Deque[ExecutionSample] = deque(maxlen=window)

    def record(
        self,
        *,
        spread_points: float,
        slippage_points: float = 0.0,
        fill_ok: bool = True,
        latency_ms: float = 0.0,
    ) -> None:
        self._samples.append(
            ExecutionSample(
                spread_points=spread_points,
                slippage_points=slippage_points,
                fill_ok=fill_ok,
                latency_ms=latency_ms,
            ),
        )

    def report(self, *, current_spread: float, spread_cap: float) -> ExecutionQualityReport:
        if not self._samples:
            norm = min(1.0, current_spread / max(spread_cap, 1.0))
            return ExecutionQualityReport(
                score=max(0.0, 1.0 - norm * 0.5),
                spread_penalty=norm * 0.3,
                slippage_penalty=0.0,
                fill_rate=1.0,
                recommendation="no_history",
            )

        spreads = [s.spread_points for s in self._samples]
        slips = [s.slippage_points for s in self._samples]
        fills = [s.fill_ok for s in self._samples]
        med_spread = sorted(spreads)[len(spreads) // 2]
        med_slip = sorted(slips)[len(slips) // 2] if slips else 0.0
        fill_rate = sum(1 for f in fills if f) / len(fills)

        spread_pen = min(1.0, current_spread / max(spread_cap, 1.0))
        if current_spread > med_spread * 1.35:
            spread_pen = min(1.0, spread_pen + 0.15)
        slip_pen = min(1.0, med_slip / max(spread_cap * 0.5, 1.0))

        score = max(0.0, min(1.0, 0.55 * (1.0 - spread_pen) + 0.25 * (1.0 - slip_pen) + 0.20 * fill_rate))
        rec = "good"
        if score < 0.45:
            rec = "avoid_entries"
        elif score < 0.62:
            rec = "caution"

        return ExecutionQualityReport(
            score=score,
            spread_penalty=spread_pen,
            slippage_penalty=slip_pen,
            fill_rate=fill_rate,
            recommendation=rec,
        )

"""Simulación Monte Carlo para riesgo de racha y drawdown."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    median_final_equity: float
    worst_dd_pct: float
    p95_dd_pct: float
    ruin_probability: float
    recommended_lot_factor: float
    recommended_aggression: float


class MonteCarloRiskEngine:
    """Simula secuencias de R-multiples para ajustar exposición."""

    def __init__(self, *, seed: int | None = 42) -> None:
        self._rng = random.Random(seed)

    def simulate(
        self,
        r_samples: Sequence[float],
        *,
        start_equity: float = 10_000.0,
        risk_pct_per_trade: float = 1.0,
        n_sims: int = 500,
        n_trades: int = 200,
        ruin_dd_pct: float = 12.0,
    ) -> MonteCarloResult:
        if not r_samples or n_sims < 10:
            return MonteCarloResult(
                simulations=0,
                median_final_equity=start_equity,
                worst_dd_pct=0.0,
                p95_dd_pct=0.0,
                ruin_probability=0.0,
                recommended_lot_factor=1.0,
                recommended_aggression=1.0,
            )

        pool = list(r_samples)
        finals: list[float] = []
        max_dds: list[float] = []
        ruins = 0

        for _ in range(n_sims):
            eq = start_equity
            peak = eq
            max_dd = 0.0
            for _t in range(n_trades):
                r = pool[self._rng.randrange(len(pool))]
                eq += eq * (risk_pct_per_trade / 100.0) * r
                peak = max(peak, eq)
                dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
            finals.append(eq)
            max_dds.append(max_dd)
            if max_dd >= ruin_dd_pct:
                ruins += 1

        finals.sort()
        max_dds.sort()
        median_eq = finals[len(finals) // 2]
        p95_dd = max_dds[int(len(max_dds) * 0.95)]
        worst_dd = max_dds[-1]
        ruin_p = ruins / n_sims

        lot_factor = 1.0
        if p95_dd > 8.0:
            lot_factor = 0.75
        if p95_dd > 11.0:
            lot_factor = 0.55
        if ruin_p > 0.08:
            lot_factor = min(lot_factor, 0.50)

        aggression = 1.0
        if median_eq > start_equity * 1.05 and p95_dd < 6.0:
            aggression = 1.08
        if p95_dd > 9.0:
            aggression = 0.85

        return MonteCarloResult(
            simulations=n_sims,
            median_final_equity=median_eq,
            worst_dd_pct=worst_dd,
            p95_dd_pct=p95_dd,
            ruin_probability=ruin_p,
            recommended_lot_factor=lot_factor,
            recommended_aggression=aggression,
        )

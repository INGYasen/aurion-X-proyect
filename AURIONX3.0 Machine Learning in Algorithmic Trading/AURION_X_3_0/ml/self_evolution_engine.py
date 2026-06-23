"""Auto-evolución: ajusta pesos, umbrales y filtros desde rendimiento real."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics.logger import get_logger
from analytics.performance_profiler import PerformanceProfiler
from ml.self_optimizer import SelfOptimizer

logger = get_logger(__name__)


@dataclass
class EvolutionState:
    ml_weights: dict[str, float] = field(default_factory=dict)
    threshold_delta: float = 0.0
    sensitivity: float = 1.0
    aggression: float = 1.0
    min_setup_grade: str = "C"
    confidence_boost: float = 0.0
    notes: str = ""


class SelfEvolutionEngine:
    """Combina profiler + optimizer para evolución diaria."""

    def __init__(self, *, min_samples: int = 15) -> None:
        self._optimizer = SelfOptimizer(min_samples=min_samples)
        self._profiler = PerformanceProfiler()
        self._state = EvolutionState()

    @property
    def state(self) -> EvolutionState:
        return self._state

    def evolve_from_rows(self, rows: list[dict[str, Any]]) -> EvolutionState:
        opt = self._optimizer.analyze_day(rows)
        prof = self._profiler.profile(rows)

        weights = dict(opt.recalibrated_weights)
        for dim, wr in prof.best_dimensions.items():
            if "RANGE" in dim.upper():
                weights["range"] = min(1.4, weights.get("range", 1.0) * 1.05)
            if "MOMENTUM" in dim.upper() or "BREAKOUT" in dim.upper():
                weights["momentum"] = min(1.4, weights.get("momentum", 1.0) * 1.05)

        thr_delta = 0.0
        sens = 1.0
        agg = 1.0
        if prof.overall_winrate < 0.42:
            thr_delta = 0.04
            sens = 1.15
            agg = 0.75
        elif prof.overall_winrate >= 0.54 and prof.overall_expectancy > 0.05:
            thr_delta = -0.02
            agg = 1.05

        min_grade = "C"
        if prof.overall_winrate < 0.38:
            min_grade = "B"

        self._state = EvolutionState(
            ml_weights=weights,
            threshold_delta=thr_delta,
            sensitivity=sens,
            aggression=agg,
            min_setup_grade=min_grade,
            confidence_boost=0.02 if prof.overall_winrate >= 0.52 else -0.03,
            notes=f"best_hour={prof.best_hour} best_setup={prof.best_setup}",
        )
        logger.info("[EVOLVE] %s weights=%s", self._state.notes, weights)
        return self._state

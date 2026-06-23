"""Auto-optimización diaria: setups, horarios, sweeps, sesiones."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationReport:
    best_setups: list[tuple[str, float]]
    worst_setups: list[tuple[str, float]]
    best_hours: list[tuple[int, float]]
    best_sessions: list[tuple[str, float]]
    sweep_effectiveness: float
    recalibrated_weights: dict[str, float]
    notes: str = ""


class SelfOptimizer:
    """Analiza histórico cerrado y propone recalibración."""

    def __init__(self, *, min_samples: int = 12) -> None:
        self._min_samples = min_samples
        self._last_weights: dict[str, float] = {}

    def analyze_day(self, rows: list[dict[str, Any]]) -> OptimizationReport:
        setup_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "w": 0.0, "sum_r": 0.0})
        hour_stats: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "w": 0.0})
        session_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "w": 0.0})
        sweep_wins = 0.0
        sweep_n = 0.0

        for row in rows:
            try:
                profit = float(row.get("profit") or 0.0)
                r = float(row.get("r_multiple") or 0.0)
            except (TypeError, ValueError):
                continue
            fx = row.get("features_json")
            if not isinstance(fx, dict):
                continue
            setup = str(fx.get("setup_type", "UNKNOWN")).upper()
            st = setup_stats[setup]
            st["n"] += 1.0
            if profit > 0:
                st["w"] += 1.0
            st["sum_r"] += r

            hr = int(fx.get("hour_slot", -1))
            if 0 <= hr <= 23:
                hs = hour_stats[hr]
                hs["n"] += 1.0
                if profit > 0:
                    hs["w"] += 1.0

            sess = str(fx.get("session_type", fx.get("ml_dataset_context", "unknown"))).lower()
            ss = session_stats[sess]
            ss["n"] += 1.0
            if profit > 0:
                ss["w"] += 1.0

            if bool(fx.get("liquidity_sweep", False)):
                sweep_n += 1.0
                if profit > 0:
                    sweep_wins += 1.0

        def _wr(st: dict[str, float]) -> float:
            return st["w"] / max(st["n"], 1.0)

        def _exp(st: dict[str, float]) -> float:
            return st["sum_r"] / max(st["n"], 1.0)

        setup_ranked = [
            (k, _wr(v) * 0.5 + max(-1.0, min(1.0, _exp(v))) * 0.5)
            for k, v in setup_stats.items()
            if v["n"] >= self._min_samples / 2
        ]
        setup_ranked.sort(key=lambda x: x[1], reverse=True)
        best_setups = setup_ranked[:5]
        worst_setups = sorted(setup_ranked, key=lambda x: x[1])[:5]

        hour_ranked = [(h, _wr(v)) for h, v in hour_stats.items() if v["n"] >= 4]
        hour_ranked.sort(key=lambda x: x[1], reverse=True)
        sess_ranked = [(s, _wr(v)) for s, v in session_stats.items() if v["n"] >= 4]
        sess_ranked.sort(key=lambda x: x[1], reverse=True)

        sweep_eff = sweep_wins / max(sweep_n, 1.0)

        weights = {
            "momentum": 1.0,
            "range": 1.0,
            "liquidity_sweep": 1.0,
            "absorption": 1.0,
            "session": 1.0,
        }
        if sweep_eff >= 0.55:
            weights["liquidity_sweep"] = 1.15
        elif sweep_eff < 0.38:
            weights["liquidity_sweep"] = 0.88

        for setup, score in best_setups:
            if "RANGE" in setup:
                weights["range"] = min(1.35, weights["range"] + 0.05)
            if "MOMENTUM" in setup or "BREAKOUT" in setup:
                weights["momentum"] = min(1.35, weights["momentum"] + 0.05)
        for setup, score in worst_setups:
            if score < 0.35 and "RANGE" in setup:
                weights["range"] = max(0.75, weights["range"] - 0.06)

        self._last_weights = weights
        logger.info(
            "SelfOptimizer: best_setup=%s sweep_eff=%.2f weights=%s",
            best_setups[0] if best_setups else None,
            sweep_eff,
            weights,
        )
        return OptimizationReport(
            best_setups=best_setups,
            worst_setups=worst_setups,
            best_hours=hour_ranked[:6],
            best_sessions=sess_ranked[:4],
            sweep_effectiveness=sweep_eff,
            recalibrated_weights=weights,
        )

    @property
    def last_weights(self) -> dict[str, float]:
        return dict(self._last_weights)

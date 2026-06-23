"""Motor de probabilidades contextuales (continuación, reversión, sweeps, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProbabilitySnapshot:
    continuation: float
    reversal: float
    fake_sweep: float
    clean_breakout: float
    stop_hunt: float
    institutional_absorption: float

    def as_dict(self) -> dict[str, float]:
        return {
            "prob_continuation": self.continuation,
            "prob_reversal": self.reversal,
            "prob_fake_sweep": self.fake_sweep,
            "prob_clean_breakout": self.clean_breakout,
            "prob_stop_hunt": self.stop_hunt,
            "prob_institutional_absorption": self.institutional_absorption,
        }


class ProbabilityEngine:
    """Heurísticas + señales de estrategia → probabilidades [0,1]."""

    def compute(
        self,
        *,
        direction: str,
        signal: Mapping[str, Any],
        regime: str,
        mtf_trends: Mapping[str, str],
        structure_trend: str,
        continuation_strength: float,
        spread_points: float,
        spread_cap: float,
    ) -> ProbabilitySnapshot:
        d = direction.upper()
        setup = str(signal.get("setup_type", "")).upper()
        cont = max(0.0, min(1.0, float(continuation_strength)))
        mom = float(signal.get("momentum_score", signal.get("min_setup_momentum_score", 0.0)) or 0.0)
        if mom > 1.0:
            mom = min(1.0, mom / 100.0)

        aligned = 0.0
        conflict = 0.0
        for tf in ("M1", "M5", "M15"):
            t = str(mtf_trends.get(tf, "RANGE")).upper()
            if (d == "BUY" and t == "UP") or (d == "SELL" and t == "DOWN"):
                aligned += 1.0 / 3.0
            elif (d == "BUY" and t == "DOWN") or (d == "SELL" and t == "UP"):
                conflict += 1.0 / 3.0

        continuation = min(0.95, 0.35 * cont + 0.30 * aligned + 0.20 * mom + 0.15 * (1.0 - conflict))
        reversal = min(0.95, max(0.05, 1.0 - continuation + 0.15 * conflict))
        if "FADE" in setup or "RANGE" in setup:
            reversal = min(0.92, reversal + 0.12)
            continuation = max(0.08, continuation - 0.10)

        sweep = 0.35
        if bool(signal.get("liquidity_sweep", False)):
            sweep = 0.72
        sweep_score = float(signal.get("liquidity_sweep_score", 0.0) or 0.0)
        if sweep_score > 0:
            sweep = max(sweep, min(1.0, sweep_score))

        fake_sweep = 0.25
        if sweep > 0.6 and conflict > 0.35:
            fake_sweep = min(0.88, 0.40 + conflict * 0.5)
        bo_p = float(signal.get("range_breakout_probability", 0.0) or 0.0)
        if bo_p > 55:
            fake_sweep = max(fake_sweep, min(0.9, bo_p / 100.0))

        clean_breakout = 0.30
        if "BREAKOUT" in setup or "MOMENTUM" in setup:
            clean_breakout = min(0.90, 0.45 + mom * 0.35 + aligned * 0.25)
        if regime.upper() in ("EXTREME", "HIGH"):
            clean_breakout *= 0.85

        stop_hunt = min(0.85, 0.20 + sweep * 0.35 + (spread_points / max(spread_cap, 1.0)) * 0.25)
        absorption = 0.25
        if bool(signal.get("pullback", False)):
            absorption = 0.62
        abs_s = float(signal.get("absorption_score", 0.0) or 0.0)
        if abs_s > 0:
            absorption = max(absorption, min(1.0, abs_s))
        if structure_trend.upper() == "RANGE":
            absorption = min(0.92, absorption + 0.08)

        return ProbabilitySnapshot(
            continuation=float(continuation),
            reversal=float(reversal),
            fake_sweep=float(fake_sweep),
            clean_breakout=float(clean_breakout),
            stop_hunt=float(stop_hunt),
            institutional_absorption=float(absorption),
        )

"""Score compuesto de confianza para decisiones dominadas por ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .probability_engine import ProbabilitySnapshot


@dataclass(frozen=True)
class ConfidenceBreakdown:
    confidence: float
    probability: float
    quality: float
    institutional: float
    execution: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "ml_confidence": self.confidence,
            "probability_score": self.probability,
            "quality_score": self.quality,
            "institutional_score": self.institutional,
            "execution_score": self.execution,
            "total_score": self.total,
        }


class MLConfidenceEngine:
    """Combina ML, probabilidades contextuales y calidad de ejecución."""

    def __init__(
        self,
        *,
        ml_weight: float = 0.45,
        probability_weight: float = 0.25,
        quality_weight: float = 0.15,
        institutional_weight: float = 0.10,
        execution_weight: float = 0.05,
    ) -> None:
        wsum = ml_weight + probability_weight + quality_weight + institutional_weight + execution_weight
        self._ml_w = ml_weight / wsum
        self._prob_w = probability_weight / wsum
        self._qual_w = quality_weight / wsum
        self._inst_w = institutional_weight / wsum
        self._exec_w = execution_weight / wsum

    def compute(
        self,
        *,
        direction: str,
        prob_buy: float,
        prob_sell: float,
        ml_confidence: float,
        rule_score: int,
        probabilities: ProbabilitySnapshot,
        signal: Mapping[str, Any],
        spread_points: float,
        spread_cap: float,
        adaptive_weights: Mapping[str, float] | None = None,
    ) -> ConfidenceBreakdown:
        d = direction.upper()
        p_dir = prob_buy if d == "BUY" else prob_sell
        conf = max(0.0, min(1.0, float(ml_confidence)))
        if conf <= 0:
            conf = max(prob_buy, prob_sell)

        probs = probabilities.as_dict()
        aw = dict(adaptive_weights or {})
        w_cont = float(aw.get("momentum", 1.0))
        w_sweep = float(aw.get("liquidity_sweep", 1.0))
        w_abs = float(aw.get("absorption", 1.0))
        w_range = float(aw.get("range", 1.0))

        if "RANGE" in str(signal.get("setup_type", "")).upper():
            prob_score = (
                probabilities.reversal * w_range * 0.5
                + (1.0 - probabilities.fake_sweep) * 0.3
                + probabilities.institutional_absorption * w_abs * 0.2
            )
        else:
            prob_score = (
                probabilities.continuation * w_cont * 0.45
                + probabilities.clean_breakout * 0.30
                + (1.0 - probabilities.stop_hunt) * w_sweep * 0.25
            )
        prob_score = max(0.0, min(1.0, prob_score))

        quality = max(0.0, min(1.0, float(rule_score) / 100.0))
        rq = float(signal.get("range_quality_score", 0.0) or 0.0)
        if rq > 0:
            quality = max(quality, min(1.0, rq / 100.0))

        institutional = max(
            0.0,
            min(
                1.0,
                probabilities.institutional_absorption * w_abs * 0.6
                + probabilities.clean_breakout * 0.4,
            ),
        )

        spread_pen = min(1.0, spread_points / max(spread_cap, 1.0))
        vol_ratio = float(signal.get("volume_ratio", 1.0) or 1.0)
        execution = max(0.0, min(1.0, 0.7 * (1.0 - spread_pen * 0.6) + 0.3 * min(1.2, vol_ratio) / 1.2))

        aligned_ml = p_dir
        total = (
            self._ml_w * aligned_ml
            + self._prob_w * prob_score
            + self._qual_w * quality
            + self._inst_w * institutional
            + self._exec_w * execution
        )
        total = max(0.0, min(0.99, total * 0.65 + conf * 0.35))

        return ConfidenceBreakdown(
            confidence=conf,
            probability=prob_score,
            quality=quality,
            institutional=institutional,
            execution=execution,
            total=total,
        )

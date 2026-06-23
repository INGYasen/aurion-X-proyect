"""Meta-ranking de setups: A+, A, B, C, basura."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .asset_ml_weights import AssetMLProfile


class SetupGrade(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    TRASH = "X"


@dataclass(frozen=True)
class SetupRank:
    grade: SetupGrade
    score: float
    reasons: tuple[str, ...]
    lot_multiplier: float
    allow_aggressive: bool


class SetupRankingEngine:
    """Rankea calidad del setup antes de ejecutar."""

    def __init__(self, *, min_grade_trade: str = "C") -> None:
        self._min_grade = min_grade_trade.upper().replace("BASURA", "X")

    @staticmethod
    def _grade_order(g: SetupGrade) -> int:
        return {"A+": 5, "A": 4, "B": 3, "C": 2, "X": 0}.get(g.value, 0)

    def rank(
        self,
        *,
        symbol: str,
        setup_type: str,
        ml_confidence: float,
        composite_score: float,
        regime: str,
        probabilities: Mapping[str, float],
        signal: Mapping[str, Any],
        memory_insight: Mapping[str, Any] | None,
        asset_profile: AssetMLProfile,
    ) -> SetupRank:
        reasons: list[str] = []
        score = 0.0
        w = asset_profile.weights

        score += min(35.0, ml_confidence * 40.0 * w.get("trend", 1.0))
        score += min(20.0, composite_score * 25.0)
        cont = float(probabilities.get("prob_continuation", 0.0))
        rev = float(probabilities.get("prob_reversal", 0.0))
        sweep = float(probabilities.get("prob_fake_sweep", 0.0))
        absorp = float(probabilities.get("prob_institutional_absorption", 0.0))

        st = setup_type.upper()
        if "RANGE" in st:
            score += rev * 18.0 * w.get("range", 1.0)
            rq = float(signal.get("range_quality_score", 0.0) or 0.0)
            if rq >= 65:
                score += 12.0
                reasons.append("range_quality_high")
            elif rq > 0 and rq < 45:
                score -= 15.0
                reasons.append("range_quality_weak")
        else:
            score += cont * 15.0 * w.get("momentum", 1.0)
            score += float(signal.get("momentum_score", 0.0) or 0.0) * 10.0 * w.get("displacement", 1.0)

        if bool(signal.get("liquidity_sweep", False)):
            score += 8.0 * w.get("liquidity_sweep", 1.0)
            reasons.append("sweep")
        if absorp >= 0.55:
            score += 7.0 * w.get("absorption", 1.0)
        if sweep >= 0.65:
            score -= 12.0 * w.get("fake_breakout", 1.0)
            reasons.append("fake_breakout_risk")

        if regime.upper() in ("EXTREME", "HIGH"):
            score += 3.0 * w.get("volatility", 1.0)
        if regime.upper() == "LOW" and "BREAKOUT" in st:
            score -= 8.0

        mem = memory_insight or {}
        if mem.get("why") == "historical_edge":
            score += 10.0
            reasons.append("memory_edge")
        elif mem.get("why") == "historical_drag":
            score -= 14.0
            reasons.append("memory_drag")

        if score >= 78:
            grade = SetupGrade.A_PLUS
            lot_mult = 1.25
        elif score >= 62:
            grade = SetupGrade.A
            lot_mult = 1.10
        elif score >= 48:
            grade = SetupGrade.B
            lot_mult = 0.90
        elif score >= 32:
            grade = SetupGrade.C
            lot_mult = 0.70
        else:
            grade = SetupGrade.TRASH
            lot_mult = 0.0

        allow_agg = self._grade_order(grade) >= self._grade_order(SetupGrade.A)
        if grade == SetupGrade.B and asset_profile.min_setup_grade_for_aggressive == "B":
            allow_agg = True

        return SetupRank(
            grade=grade,
            score=score,
            reasons=tuple(reasons),
            lot_multiplier=lot_mult,
            allow_aggressive=allow_agg,
        )

    def allows_trade(self, rank: SetupRank) -> bool:
        min_ord = self._grade_order(SetupGrade(self._min_grade if self._min_grade in ("A+", "A", "B", "C", "X") else "C"))
        return self._grade_order(rank.grade) >= min_ord and rank.grade != SetupGrade.TRASH

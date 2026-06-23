"""Motor profesional de trading en rango (mean reversion en extremos)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from strategy.structure_detector import StructureResult, Trend


class RangeQuality(str, Enum):
    CLEAN = "CLEAN"
    MANIPULATIVE = "MANIPULATIVE"
    COMPRESSION = "COMPRESSION"
    CHOP = "CHOP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RangeAnalysis:
    quality: RangeQuality
    tradeable: bool
    edge_side: str
    score: float
    reason: str
    sweep_lateral: bool
    institutional_compression: bool


class RangeTradingEngine:
    """Evalúa si el rango es operable y desde qué extremo."""

    def __init__(
        self,
        *,
        min_width_pct: float = 0.00004,
        max_width_pct: float = 0.008,
        min_key_levels: int = 2,
    ) -> None:
        self._min_w = min_width_pct
        self._max_w = max_width_pct
        self._min_kl = min_key_levels

    def analyze(
        self,
        structure: StructureResult,
        signal: Mapping[str, Any],
        *,
        regime: str,
        spread_points: float,
        spread_cap: float,
    ) -> RangeAnalysis:
        if structure.trend != Trend.RANGE or structure.range_width_pct is None:
            return RangeAnalysis(
                quality=RangeQuality.UNKNOWN,
                tradeable=False,
                edge_side="",
                score=0.0,
                reason="no_range_structure",
                sweep_lateral=False,
                institutional_compression=False,
            )

        rw = float(structure.range_width_pct)
        pin = float(structure.position_in_range or 0.5)
        rq = float(signal.get("range_quality_score", 0.0) or 0.0)
        bo_p = float(signal.get("range_breakout_probability", 0.0) or 0.0)
        sweep = bool(signal.get("liquidity_sweep", False))
        spread_norm = spread_points / max(spread_cap, 1.0)

        quality = RangeQuality.CLEAN
        if rw < self._min_w or rw > self._max_w:
            quality = RangeQuality.CHOP
        elif bo_p >= 70 and regime.upper() != "LOW":
            quality = RangeQuality.MANIPULATIVE
        elif rw < self._min_w * 3 and regime.upper() in ("LOW", "NORMAL"):
            quality = RangeQuality.COMPRESSION

        n_levels = len(structure.key_levels)
        if n_levels < self._min_kl:
            return RangeAnalysis(
                quality=quality,
                tradeable=False,
                edge_side="",
                score=0.0,
                reason="insufficient_levels",
                sweep_lateral=sweep,
                institutional_compression=quality == RangeQuality.COMPRESSION,
            )

        if bo_p >= 68 and "BREAKOUT" in str(signal.get("setup_type", "")).upper():
            return RangeAnalysis(
                quality=RangeQuality.MANIPULATIVE,
                tradeable=False,
                edge_side="",
                score=max(0.0, rq - bo_p * 0.3),
                reason="false_breakout_inside_range",
                sweep_lateral=sweep,
                institutional_compression=False,
            )

        edge = ""
        if pin <= 0.32:
            edge = "BUY"
        elif pin >= 0.68:
            edge = "SELL"

        score = rq * 0.6 + (1.0 - abs(pin - 0.5) * 2) * 25.0
        if quality == RangeQuality.CLEAN:
            score += 12.0
        if quality == RangeQuality.COMPRESSION:
            score += 6.0
        if spread_norm > 0.75:
            score -= 15.0
        if sweep and quality != RangeQuality.MANIPULATIVE:
            score += 8.0

        tradeable = (
            quality in (RangeQuality.CLEAN, RangeQuality.COMPRESSION)
            and edge != ""
            and score >= 48.0
            and spread_norm < 0.85
        )

        return RangeAnalysis(
            quality=quality,
            tradeable=tradeable,
            edge_side=edge,
            score=min(100.0, score),
            reason=f"range_{quality.value.lower()}_pin={pin:.2f}",
            sweep_lateral=sweep,
            institutional_compression=quality == RangeQuality.COMPRESSION,
        )

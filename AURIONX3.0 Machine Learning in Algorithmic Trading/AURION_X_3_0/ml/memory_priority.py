"""Memoria con prioridad: más peso a trades recientes y régimen actual."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def recency_weight(close_time: datetime | str | None, *, half_life_days: float = 7.0) -> float:
    """Peso exponencial por antigüedad (1.0 = hoy)."""
    if close_time is None:
        return 0.5
    if isinstance(close_time, str):
        try:
            ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        except ValueError:
            return 0.5
    else:
        ct = close_time
    if ct.tzinfo is None:
        ct = ct.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - ct).total_seconds() / 86400.0)
    lam = math.log(2) / max(half_life_days, 0.5)
    return float(math.exp(-lam * age_days))


def regime_match_weight(row_regime: str, current_regime: str) -> float:
    if not row_regime or not current_regime:
        return 0.7
    if row_regime.upper() == current_regime.upper():
        return 1.25
    if row_regime.upper()[:3] == current_regime.upper()[:3]:
        return 1.05
    return 0.65


def weighted_sample_weight(
    row: dict[str, Any],
    *,
    current_regime: str,
    weight_by_r: bool = True,
    half_life_days: float = 7.0,
) -> float:
    """Peso de entrenamiento con memoria prioritaria."""
    w = recency_weight(row.get("close_time"), half_life_days=half_life_days)
    fx = row.get("features_json")
    row_reg = ""
    if isinstance(fx, dict):
        row_reg = str(fx.get("volatility_regime", ""))
    w *= regime_match_weight(row_reg, current_regime)
    if weight_by_r:
        raw = row.get("r_multiple")
        if raw is not None:
            try:
                w *= min(3.0, 1.0 + abs(float(raw)))
            except (TypeError, ValueError):
                pass
    return max(0.05, min(4.0, w))

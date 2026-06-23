"""Motor de consistencia: curva de equity estable para cuentas funded."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Mapping


@dataclass(frozen=True)
class ConsistencyVerdict:
    allow: bool
    lot_factor: float
    reason: str


class ConsistencyEngine:
    """Evita días explosivos y estabiliza ganancias."""

    def __init__(self, cfg: Mapping[str, object] | None = None) -> None:
        c = dict(cfg or {})
        self.enabled = bool(c.get("enabled", True))
        self.max_daily_profit_pct = float(c.get("max_daily_profit_pct", 2.5))
        self.target_daily_profit_pct = float(c.get("target_daily_profit_pct", 0.8))
        self.reduce_after_green_day = bool(c.get("reduce_after_green_day", True))

    def evaluate(
        self,
        *,
        equity: Decimal,
        day_pnl: Decimal,
        day_start_equity: Decimal,
        trading_day: date,
    ) -> ConsistencyVerdict:
        if not self.enabled or day_start_equity <= 0:
            return ConsistencyVerdict(True, 1.0, "disabled")

        daily_pct = float(day_pnl / day_start_equity * Decimal("100"))
        if daily_pct >= self.max_daily_profit_pct:
            return ConsistencyVerdict(False, 0.0, "daily_target_cap_reached")

        lot_factor = 1.0
        if self.reduce_after_green_day and daily_pct >= self.target_daily_profit_pct:
            lot_factor = 0.72
            return ConsistencyVerdict(True, lot_factor, "protect_green_day")

        if daily_pct < -1.5:
            lot_factor = 0.80
            return ConsistencyVerdict(True, lot_factor, "soft_recovery")

        return ConsistencyVerdict(True, lot_factor, "ok")

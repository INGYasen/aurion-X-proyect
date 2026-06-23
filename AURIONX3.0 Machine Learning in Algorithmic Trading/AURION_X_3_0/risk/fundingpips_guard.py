"""Modo seguro FundingPips: límites DD, anti-revenge, anti-sobreoperación."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Mapping


@dataclass(frozen=True)
class FundingPipsVerdict:
    allow_new_trade: bool
    reason: str
    lot_cap_factor: float


class FundingPipsGuard:
    """Guardrails para cuentas funded (FundingPips y similares)."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        c = dict(cfg or {})
        self.enabled = bool(c.get("enabled", False))
        self.max_daily_dd_pct = float(c.get("max_daily_dd_pct", 4.0))
        self.max_total_dd_pct = float(c.get("max_total_dd_pct", 8.0))
        self.max_trades_per_day = int(c.get("max_trades_per_day", 12))
        self.max_trades_per_hour = int(c.get("max_trades_per_hour", 4))
        self.revenge_cooldown_s = int(c.get("revenge_cooldown_s", 600))
        self.min_minutes_between_entries = float(c.get("min_minutes_between_entries", 3.0))
        self.block_after_consecutive_losses = int(c.get("block_after_consecutive_losses", 4))
        self.impulsive_score_max = int(c.get("impulsive_score_max", 45))
        self._adaptive = bool(c.get("adaptive_mode", True))
        self.adaptive_mode = self._adaptive
        self._base_max_trades_day = self.max_trades_per_day
        self._base_min_spacing = self.min_minutes_between_entries

    def adapt_dynamics(
        self,
        *,
        drawdown_pct: float,
        winrate: float,
        volatility_extreme: bool,
        equity_stable: bool,
    ) -> None:
        """Modo funded dinámico: endurece o relaja según salud."""
        if not self._adaptive:
            return
        if drawdown_pct >= self.max_daily_dd_pct * 0.6 or winrate < 0.40 or volatility_extreme:
            self.max_trades_per_day = max(3, int(self._base_max_trades_day * 0.55))
            self.min_minutes_between_entries = self._base_min_spacing * 1.6
            self.impulsive_score_max = max(50, self.impulsive_score_max - 5)
        elif equity_stable and winrate >= 0.52 and drawdown_pct < 1.2:
            self.max_trades_per_day = self._base_max_trades_day
            self.min_minutes_between_entries = max(2.0, self._base_min_spacing * 0.9)
        else:
            self.max_trades_per_day = max(4, int(self._base_max_trades_day * 0.75))
            self.min_minutes_between_entries = self._base_min_spacing * 1.2

    def evaluate(
        self,
        *,
        equity: Decimal,
        day_start_equity: Decimal,
        total_dd_pct: float,
        trades_today: int,
        trades_last_hour: int,
        consecutive_losses: int,
        seconds_since_last_entry: float,
        setup_grade: str,
        setup_score: float,
        winrate: float = 0.5,
        drawdown_pct: float = 0.0,
        now_utc: datetime | None = None,
    ) -> FundingPipsVerdict:
        if not self.enabled:
            return FundingPipsVerdict(True, "disabled", 1.0)

        if day_start_equity > 0:
            daily_dd = float((day_start_equity - equity) / day_start_equity) * 100.0
            if daily_dd >= self.max_daily_dd_pct:
                return FundingPipsVerdict(False, "daily_dd_limit", 0.0)

        if total_dd_pct >= self.max_total_dd_pct:
            return FundingPipsVerdict(False, "total_dd_limit", 0.0)

        if trades_today >= self.max_trades_per_day:
            return FundingPipsVerdict(False, "max_trades_day", 0.0)

        if trades_last_hour >= self.max_trades_per_hour:
            return FundingPipsVerdict(False, "max_trades_hour", 0.0)

        if consecutive_losses >= self.block_after_consecutive_losses:
            return FundingPipsVerdict(False, "loss_streak_revenge_guard", 0.0)

        min_gap = self.min_minutes_between_entries * 60.0
        if seconds_since_last_entry < min_gap and seconds_since_last_entry >= 0:
            return FundingPipsVerdict(False, "entry_spacing", 0.0)

        lot_cap = 1.0
        if setup_grade in ("C", "X") or setup_score < self.impulsive_score_max:
            return FundingPipsVerdict(False, "low_setup_grade", 0.0)
        if setup_grade == "B":
            lot_cap = 0.85
        elif setup_grade == "A+":
            lot_cap = min(1.12, lot_cap * 1.08)
        if consecutive_losses >= 2:
            lot_cap = min(lot_cap, 0.70)
        if winrate >= 0.55 and drawdown_pct < 1.0:
            lot_cap = min(1.05, lot_cap * 1.05)

        return FundingPipsVerdict(True, "ok", lot_cap)

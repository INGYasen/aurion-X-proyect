"""IA de salida: detecta pérdida de momentum y cierre antes del retroceso."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExitAISignal:
    should_exit: bool
    urgency: float
    reason: str
    partial_fraction: float


class TradeExitAI:
    """Evalúa salida anticipada sobre trade abierto + estado de mercado."""

    def __init__(
        self,
        *,
        momentum_loss_thr: float = 0.36,
        min_r_to_consider: float = 0.18,
        max_r_before_trail: float = 1.2,
        absorption_exit_thr: float = 0.62,
    ) -> None:
        self._mom_thr = momentum_loss_thr
        self._min_r = min_r_to_consider
        self._max_r = max_r_before_trail
        self._abs_thr = absorption_exit_thr

    def evaluate(
        self,
        *,
        trade: Mapping[str, Any],
        market_state: Mapping[str, Any],
        signal_context: Mapping[str, Any] | None = None,
    ) -> ExitAISignal:
        direction = str(trade.get("direction", "BUY")).upper()
        current_r = float(trade.get("current_r", trade.get("max_r", 0.0)) or 0.0)
        max_r = float(trade.get("max_r", current_r) or 0.0)
        cont = float(market_state.get("continuation_score", 0.0))
        disp = float(market_state.get("displacement_score", 0.0))
        comp = float(market_state.get("compression_score", 0.0))
        vel_loss = bool(market_state.get("velocity_loss", False))
        ctx = signal_context or {}

        # Micro-reversión: últimas velas contra la posición
        closes = list(market_state.get("closes_m1") or [])
        micro_rev = False
        if len(closes) >= 4:
            d1 = closes[-1] - closes[-2]
            d2 = closes[-2] - closes[-3]
            if direction == "BUY" and d1 < 0 and d2 < 0:
                micro_rev = True
            if direction == "SELL" and d1 > 0 and d2 > 0:
                micro_rev = True

        absorption_against = float(ctx.get("counter_absorption", 0.0))
        if bool(ctx.get("pullback", False)) and direction == "BUY":
            absorption_against = max(absorption_against, 0.5)
        if bool(ctx.get("pullback", False)) and direction == "SELL":
            absorption_against = max(absorption_against, 0.5)

        giveback = 0.0
        if max_r > 0.15:
            giveback = max(0.0, (max_r - current_r) / max_r)

        reasons: list[str] = []
        urgency = 0.0

        if current_r < self._min_r and cont < self._mom_thr - 0.08:
            urgency = 0.55
            reasons.append("weak_early_momentum")

        if cont < self._mom_thr or vel_loss:
            urgency = max(urgency, 0.65)
            reasons.append("momentum_loss")

        if micro_rev and current_r >= self._min_r:
            urgency = max(urgency, 0.72)
            reasons.append("micro_reversal")

        if absorption_against >= self._abs_thr and current_r >= 0.25:
            urgency = max(urgency, 0.78)
            reasons.append("counter_absorption")

        if giveback >= 0.45 and max_r >= 0.35:
            urgency = max(urgency, 0.85)
            reasons.append("profit_giveback")

        if comp > 0.75 and current_r >= 0.4:
            urgency = max(urgency, 0.60)
            reasons.append("volatility_compression")

        # Divergencia simple: precio sube pero momentum cae (o viceversa)
        if len(closes) >= 6 and current_r > 0.3:
            price_up = closes[-1] > closes[-4]
            if direction == "BUY" and price_up and cont < 0.35:
                urgency = max(urgency, 0.70)
                reasons.append("divergence")
            if direction == "SELL" and not price_up and cont < 0.35:
                urgency = max(urgency, 0.70)
                reasons.append("divergence")

        micro = market_state.get("micro") or {}
        if isinstance(micro, dict):
            if float(micro.get("directional_exhaustion", 0)) > 0.65 and current_r >= 0.2:
                urgency = max(urgency, 0.74)
                reasons.append("micro_exhaustion")
            if float(micro.get("impulse_decay", 0)) > 0.7 and current_r >= 0.25:
                urgency = max(urgency, 0.70)
                reasons.append("displacement_loss")
            if float(micro.get("absorption_pressure", 0)) > 0.7:
                absorption_against = max(absorption_against, 0.65)

        should = urgency >= 0.68 and current_r >= self._min_r
        if current_r >= self._max_r and urgency < 0.80:
            should = False
            reasons.append("let_winner_run")

        partial = self._smart_partial_close(current_r, max_r, urgency, reasons)
        if should and partial <= 0:
            partial = self._momentum_take_profit(current_r, cont, disp)

        return ExitAISignal(
            should_exit=should,
            urgency=urgency,
            reason="|".join(reasons) if reasons else "hold",
            partial_fraction=partial,
        )

    def _smart_partial_close(
        self,
        current_r: float,
        max_r: float,
        urgency: float,
        reasons: list[str],
    ) -> float:
        if "profit_giveback" in reasons or "anti_giveback" in reasons:
            return 0.55 if current_r < 0.9 else 0.70
        if 0.35 <= current_r < 0.75 and urgency >= 0.72:
            return 0.45
        if current_r >= 0.85 and urgency >= 0.75:
            return 0.60
        return 0.0

    def _momentum_take_profit(self, current_r: float, cont: float, disp: float) -> float:
        if current_r >= 0.45 and cont > 0.55 and disp > 0.4:
            return 0.35
        return 0.0

    def institutional_exit_trigger(
        self,
        trade: Mapping[str, Any],
        market_state: Mapping[str, Any],
        signal_context: Mapping[str, Any] | None = None,
    ) -> ExitAISignal:
        base = self.evaluate(trade=trade, market_state=market_state, signal_context=signal_context)
        current_r = float(trade.get("current_r", 0.0) or 0.0)
        max_r = float(trade.get("max_r", current_r) or 0.0)
        protected = self.anti_giveback_protection(current_r, max_r, base.urgency)
        if protected and not base.should_exit:
            return ExitAISignal(True, max(base.urgency, 0.72), "anti_giveback", 0.50)
        return base

    def anti_giveback_protection(self, current_r: float, max_r: float, urgency: float) -> bool:
        if max_r < 0.3:
            return False
        giveback = (max_r - current_r) / max_r
        return giveback >= 0.42 and current_r >= 0.18 and urgency >= 0.55

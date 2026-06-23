"""Inteligencia de ejecución: aprende calidad real del broker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionIntelSnapshot:
    score: float
    block_entry: bool
    spread_penalty: float
    slippage_penalty: float
    latency_penalty: float
    session_quality: float
    volatility_quality: float
    reason: str

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "exec_intel_score": self.score,
            "exec_intel_block": self.block_entry,
            "exec_spread_penalty": self.spread_penalty,
            "exec_slippage_penalty": self.slippage_penalty,
            "exec_latency_penalty": self.latency_penalty,
            "exec_session_quality": self.session_quality,
            "exec_volatility_quality": self.volatility_quality,
            "exec_intel_reason": self.reason,
        }


@dataclass
class _ExecSample:
    spread: float
    slippage: float
    latency_ms: float
    fill_ok: bool
    requote: bool
    session: str
    volatility: str


class ExecutionIntelligenceEngine:
    """Capa avanzada sobre execution_quality — bloquea malas condiciones."""

    def __init__(
        self,
        *,
        window: int = 120,
        block_threshold: float = 0.38,
        latency_block_ms: float = 650.0,
        spread_decay_factor: float = 1.35,
    ) -> None:
        self._samples: Deque[_ExecSample] = deque(maxlen=window)
        self._block_threshold = block_threshold
        self._latency_block_ms = latency_block_ms
        self._spread_decay_factor = spread_decay_factor
        self._session_stats: dict[str, list[float]] = {}
        self._spread_history: Deque[float] = deque(maxlen=window)

    def record_fill(
        self,
        *,
        spread_points: float,
        slippage_points: float = 0.0,
        latency_ms: float = 0.0,
        fill_ok: bool = True,
        requote: bool = False,
        session: str = "",
        volatility: str = "NORMAL",
    ) -> None:
        self._samples.append(
            _ExecSample(
                spread=spread_points,
                slippage=slippage_points,
                latency_ms=latency_ms,
                fill_ok=fill_ok,
                requote=requote,
                session=session,
                volatility=volatility,
            ),
        )
        self._spread_history.append(spread_points)
        key = session or "unknown"
        sc = 1.0 if fill_ok and not requote else 0.0
        self._session_stats.setdefault(key, []).append(sc)
        if len(self._session_stats[key]) > 50:
            self._session_stats[key] = self._session_stats[key][-50:]

    def evaluate(
        self,
        *,
        spread_points: float,
        spread_cap: float,
        slippage_points: float = 0.0,
        latency_ms: float = 0.0,
        session: str = "",
        volatility: str = "NORMAL",
    ) -> ExecutionIntelSnapshot:
        spread_pen = min(1.0, spread_points / max(spread_cap, 1.0))
        slip_pen = min(1.0, slippage_points / max(spread_cap * 0.5, 1.0))
        lat_pen = min(1.0, latency_ms / 800.0) if latency_ms > 0 else 0.0

        sess_q = 0.7
        if session and session in self._session_stats:
            vals = self._session_stats[session]
            sess_q = sum(vals) / len(vals) if vals else 0.7

        vol_q = {"LOW": 0.85, "NORMAL": 0.75, "HIGH": 0.55, "EXTREME": 0.35}.get(volatility.upper(), 0.6)

        spread_decay_pen = 0.0
        if len(self._spread_history) >= 5:
            recent = list(self._spread_history)[-8:]
            avg_recent = sum(recent) / len(recent)
            if avg_recent > 0 and spread_points > avg_recent * self._spread_decay_factor:
                spread_decay_pen = min(0.35, (spread_points / avg_recent - 1.0) * 0.25)
                spread_pen = min(1.0, spread_pen + spread_decay_pen)

        slip_anomaly = 0.0
        if self._samples:
            recent_sp = sorted(s.spread for s in self._samples)
            med = recent_sp[len(recent_sp) // 2]
            if spread_points > med * 1.4:
                spread_pen = min(1.0, spread_pen + 0.15)
            slips = [s.slippage for s in self._samples if s.slippage > 0]
            if slips:
                med_slip = sorted(slips)[len(slips) // 2]
                if slippage_points > med_slip * 2.2 and slippage_points > spread_cap * 0.25:
                    slip_anomaly = 0.2
                    slip_pen = min(1.0, slip_pen + slip_anomaly)
            bad_fills = sum(1 for s in self._samples if not s.fill_ok or s.requote)
            if bad_fills / len(self._samples) > 0.25:
                spread_pen = min(1.0, spread_pen + 0.1)

        if latency_ms >= self._latency_block_ms:
            lat_pen = min(1.0, lat_pen + 0.25)

        score = max(
            0.0,
            min(1.0, 0.40 * (1.0 - spread_pen) + 0.25 * (1.0 - slip_pen) + 0.15 * (1.0 - lat_pen) + 0.12 * sess_q + 0.08 * vol_q),
        )
        block = (
            score < self._block_threshold
            or spread_pen > 0.88
            or latency_ms >= self._latency_block_ms * 1.2
            or slip_anomaly >= 0.18
        )
        reason = "block_poor_execution"
        if block and latency_ms >= self._latency_block_ms:
            reason = "latency_protection"
        elif block and spread_decay_pen > 0.15:
            reason = "spread_decay"
        elif block and slip_anomaly > 0:
            reason = "slippage_anomaly"
        elif not block:
            reason = "caution" if score < 0.55 else "ok"
        return ExecutionIntelSnapshot(
            score=score,
            block_entry=block,
            spread_penalty=spread_pen,
            slippage_penalty=slip_pen,
            latency_penalty=lat_pen,
            session_quality=sess_q,
            volatility_quality=vol_q,
            reason=reason,
        )

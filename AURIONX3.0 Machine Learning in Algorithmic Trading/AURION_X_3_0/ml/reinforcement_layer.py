"""Capa de aprendizaje por recompensa (policy weights, sin dependencias RL pesadas)."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RewardBreakdown:
    total: float
    profit_quality: float
    rr_component: float
    drawdown_penalty: float
    duration_bonus: float
    entry_quality: float
    stability: float
    consistency: float


@dataclass
class PolicyState:
    threshold_delta: float = 0.0
    lot_factor: float = 1.0
    aggression: float = 1.0
    entry_strictness: float = 1.0


class ReinforcementLayer:
    """Memoria de recompensas + adaptación suave de política."""

    def __init__(self, memory_path: Path | None = None, *, learning_rate: float = 0.08) -> None:
        self._lr = max(0.01, min(0.25, learning_rate))
        self._memory: deque[dict[str, float]] = deque(maxlen=500)
        self._policy = PolicyState()
        self._path = memory_path

    def _load(self) -> None:
        if self._path is None or not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            p = raw.get("policy", {})
            self._policy = PolicyState(
                threshold_delta=float(p.get("threshold_delta", 0.0)),
                lot_factor=float(p.get("lot_factor", 1.0)),
                aggression=float(p.get("aggression", 1.0)),
                entry_strictness=float(p.get("entry_strictness", 1.0)),
            )
        except Exception:
            pass

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"policy": self._policy.__dict__, "n_memory": len(self._memory)}),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("RL save: %s", exc)

    def compute_reward(
        self,
        *,
        profit: float,
        r_multiple: float,
        duration_minutes: float,
        entry_confidence: float,
        drawdown_pct: float,
        consistency_score: float,
    ) -> RewardBreakdown:
        pq = max(-1.0, min(1.0, profit / 100.0)) if profit != 0 else 0.0
        rr_c = max(-1.5, min(1.5, r_multiple))
        dd_pen = -min(1.0, drawdown_pct / 10.0)
        dur = 0.1 if 5.0 <= duration_minutes <= 120.0 else -0.05
        entry_q = (entry_confidence - 0.5) * 0.5
        stab = 0.15 if abs(r_multiple) < 2.0 else -0.1
        cons = (consistency_score - 0.5) * 0.3
        total = pq * 0.3 + rr_c * 0.35 + dd_pen * 0.15 + dur * 0.05 + entry_q * 0.08 + stab * 0.04 + cons * 0.03
        return RewardBreakdown(total, pq, rr_c, dd_pen, dur, entry_q, stab, cons)

    def observe_outcome(self, reward: RewardBreakdown, *, context_key: str = "") -> PolicyState:
        self._memory.append({"total": reward.total, "key": hash(context_key) % 1000})
        avg = sum(m["total"] for m in self._memory) / max(len(self._memory), 1)
        if reward.total > 0.15:
            self._policy.lot_factor = min(1.25, self._policy.lot_factor + self._lr * 0.5)
            self._policy.threshold_delta = max(-0.04, self._policy.threshold_delta - self._lr * 0.02)
            self._policy.aggression = min(1.15, self._policy.aggression + self._lr * 0.3)
        elif reward.total < -0.2 or avg < -0.05:
            self._policy.lot_factor = max(0.55, self._policy.lot_factor - self._lr * 0.6)
            self._policy.threshold_delta = min(0.08, self._policy.threshold_delta + self._lr * 0.04)
            self._policy.aggression = max(0.70, self._policy.aggression - self._lr * 0.4)
            self._policy.entry_strictness = min(1.35, self._policy.entry_strictness + self._lr * 0.2)
        self._save()
        return self._policy

    def policy(self) -> PolicyState:
        if self._path:
            self._load()
        return self._policy

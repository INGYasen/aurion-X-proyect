"""Memoria histórica ponderada por recencia, régimen y sesión."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryMatch:
    key: str
    samples: int
    winrate: float
    avg_r: float
    similarity: float
    weight: float


class TradeMemorySystem:
    """Índice por bucket con decay temporal y olvido de setups degradados."""

    def __init__(
        self,
        storage_path: Path,
        *,
        max_entries: int = 8000,
        recency_half_life_days: float = 14.0,
        degrade_winrate: float = 0.35,
        degrade_min_samples: int = 10,
    ) -> None:
        self._path = storage_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = max_entries
        self._half_life = max(1.0, recency_half_life_days)
        self._degrade_wr = degrade_winrate
        self._degrade_min_n = degrade_min_samples
        self._buckets: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "n": 0.0,
                "wins": 0.0,
                "sum_r": 0.0,
                "sum_score": 0.0,
                "sum_w": 0.0,
                "last_ts": 0.0,
            },
        )
        self._load()

    @staticmethod
    def context_key(ctx: dict[str, Any]) -> str:
        setup = str(ctx.get("setup_type", "UNKNOWN")).upper()
        regime = str(ctx.get("volatility_regime", ctx.get("market_regime", "UNKNOWN"))).upper()
        session = str(ctx.get("session_type", ctx.get("hour_slot", "NA")))
        sweep = "SW" if bool(ctx.get("liquidity_sweep", False)) else "NS"
        sym = str(ctx.get("symbol", "")).upper()
        prefix = f"{sym}|" if sym else ""
        return f"{prefix}{setup}|{regime}|{session}|{sweep}"

    def _recency_weight(self, last_ts: float) -> float:
        if last_ts <= 0:
            return 1.0
        age_days = max(0.0, (time.time() - last_ts) / 86400.0)
        return 0.5 ** (age_days / self._half_life)

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict):
                        self._buckets[k] = {str(x): float(v[x]) for x in v}
        except Exception as exc:
            logger.warning("TradeMemory: no se pudo cargar %s (%s)", self._path, exc)

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps(dict(self._buckets)), encoding="utf-8")
        except Exception as exc:
            logger.warning("TradeMemory: fallo guardado (%s)", exc)

    def forget_degraded_buckets(self) -> int:
        removed = 0
        for key in list(self._buckets.keys()):
            st = self._buckets[key]
            n = st.get("sum_w", st.get("n", 0.0))
            if n < self._degrade_min_n:
                continue
            wr = st.get("wins", 0.0) / max(n, 1.0)
            avg_r = st.get("sum_r", 0.0) / max(n, 1.0)
            if wr < self._degrade_wr and avg_r < -0.08:
                del self._buckets[key]
                removed += 1
        if removed:
            self._save()
        return removed

    def record_outcome(
        self,
        ctx: dict[str, Any],
        *,
        profit: float,
        r_multiple: float,
        score: float,
    ) -> None:
        key = self.context_key(ctx)
        st = self._buckets[key]
        w = self._recency_weight(st.get("last_ts", 0.0))
        st["n"] += 1.0
        st["sum_w"] += w
        if profit > 0:
            st["wins"] += w
        st["sum_r"] += float(r_multiple) * w
        st["sum_score"] += float(score) * w
        st["last_ts"] = time.time()

        if len(self._buckets) > self._max_entries:
            sorted_keys = sorted(self._buckets, key=lambda k: self._buckets[k].get("last_ts", 0.0))
            for k in sorted_keys[: max(1, len(sorted_keys) - self._max_entries)]:
                del self._buckets[k]
        self.forget_degraded_buckets()
        self._save()

    def match_context(self, ctx: dict[str, Any], *, min_samples: int = 8) -> MemoryMatch | None:
        key = self.context_key(ctx)
        st = self._buckets.get(key)
        if not st:
            return None
        n_eff = st.get("sum_w", st.get("n", 0.0))
        if n_eff < min_samples:
            return None
        wr = float(st.get("wins", 0.0) / max(n_eff, 1.0))
        avg_r = float(st.get("sum_r", 0.0) / max(n_eff, 1.0))
        rw = self._recency_weight(st.get("last_ts", 0.0))
        return MemoryMatch(
            key=key,
            samples=int(st.get("n", 0)),
            winrate=wr,
            avg_r=avg_r,
            similarity=1.0,
            weight=rw,
        )

    def meta_insight(self, ctx: dict[str, Any], *, min_samples: int = 8) -> dict[str, Any]:
        m = self.match_context(ctx, min_samples=min_samples)
        if m is None:
            return {"known": False}
        why = "neutral"
        if m.winrate >= 0.55 and m.avg_r >= 0.15:
            why = "historical_edge"
        elif m.winrate < 0.42 or m.avg_r < -0.12:
            why = "historical_drag"
        return {
            "known": True,
            "key": m.key,
            "samples": m.samples,
            "winrate": m.winrate,
            "avg_r": m.avg_r,
            "why": why,
            "recency_weight": m.weight,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

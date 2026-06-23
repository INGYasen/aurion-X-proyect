"""Estados latentes del mercado (HMM/clustering + transiciones)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from analytics.logger import get_logger

logger = get_logger(__name__)

LATENT_STATES = (
    "alive",
    "clean",
    "accumulation",
    "compression",
    "expansion",
    "transition",
    "manipulation",
    "exhaustion",
    "dangerous",
)


@dataclass(frozen=True)
class LatentStateSnapshot:
    state: str
    confidence: float
    transition_risk: float
    cluster_id: int
    is_dangerous: bool
    is_clean: bool
    transition_probs: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "latent_state": self.state,
            "latent_confidence": self.confidence,
            "latent_transition_risk": self.transition_risk,
            "latent_cluster": self.cluster_id,
            "latent_dangerous": self.is_dangerous,
            "latent_clean": self.is_clean,
            "latent_transition_probs": self.transition_probs,
        }


class LatentMarketStateAI:
    """Detecta régimen oculto vía clustering + matriz de transición."""

    def __init__(self, *, n_states: int = 6) -> None:
        self._n_states = max(3, min(8, n_states))
        self._transition_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._last_cluster: dict[str, int] = {}
        self._centroids: np.ndarray | None = None

    @staticmethod
    def _feature_vector(
        market_state: Mapping[str, Any],
        micro: Mapping[str, float] | None,
        regime: str,
    ) -> np.ndarray:
        m = micro or {}
        ret1 = 0.0
        closes = market_state.get("closes_m1") or []
        if isinstance(closes, list) and len(closes) >= 2:
            ret1 = (closes[-1] - closes[-2]) / max(abs(closes[-2]), 1e-12)
        vol_map = {"LOW": 0.2, "NORMAL": 0.5, "HIGH": 0.8, "EXTREME": 1.0}
        return np.array(
            [
                ret1 * 100.0,
                float(m.get("micro_velocity", 0.0)),
                float(m.get("tick_acceleration", 0.0)),
                float(m.get("impulse_decay", 0.0)),
                float(m.get("manipulation_score", 0.0)),
                float(m.get("absorption_pressure", 0.0)),
                vol_map.get(str(regime).upper(), 0.5),
                float(market_state.get("tick_volume_proxy", 0.0) or 0.0) / 1000.0,
            ],
            dtype=np.float64,
        )

    def _cluster_label(self, x: np.ndarray) -> int:
        if self._centroids is None or len(self._centroids) == 0:
            # Heurística sin entrenamiento previo
            manip = x[4]
            vol = x[6]
            if manip > 0.55:
                return 5
            if vol > 0.75:
                return 4
            if abs(x[0]) < 0.02 and x[2] < 0.3:
                return 2
            if x[3] > 0.6:
                return 6
            if abs(x[0]) > 0.08:
                return 0
            return 1
        dists = [float(np.linalg.norm(x - c)) for c in self._centroids]
        return int(np.argmin(dists))

    def _state_name(self, cluster: int) -> str:
        mapping = {
            0: "alive",
            1: "clean",
            2: "accumulation",
            3: "compression",
            4: "expansion",
            5: "manipulation",
            6: "exhaustion",
            7: "dangerous",
        }
        return mapping.get(cluster % 8, "transition")

    def _transition_probs(self, prev: int, curr: int) -> dict[str, float]:
        self._transition_counts[prev][curr] += 1
        row = self._transition_counts[prev]
        total = sum(row.values()) or 1
        probs: dict[str, float] = {}
        for cid, cnt in row.items():
            probs[self._state_name(cid)] = cnt / total
        return probs

    def fit_recent(self, samples: list[np.ndarray]) -> None:
        if len(samples) < 30:
            return
        try:
            from sklearn.cluster import KMeans

            X = np.vstack(samples)
            km = KMeans(n_clusters=self._n_states, random_state=42, n_init=10)
            km.fit(X)
            self._centroids = km.cluster_centers_
            logger.debug("LatentMarketStateAI: KMeans entrenado n=%s k=%s", len(samples), self._n_states)
        except Exception as exc:
            logger.debug("LatentMarketStateAI fit omitido: %s", exc)

    def infer(
        self,
        symbol: str,
        market_state: Mapping[str, Any],
        *,
        micro: Mapping[str, float] | None = None,
        regime: str = "NORMAL",
    ) -> LatentStateSnapshot:
        x = self._feature_vector(market_state, micro, regime)
        cid = self._cluster_label(x)
        s = symbol.upper()
        prev = self._last_cluster.get(s)
        trans_probs: dict[str, float] = {}
        trans_risk = 0.2
        if prev is not None:
            trans_probs = self._transition_probs(prev, cid)
            if prev != cid:
                trans_risk = min(0.95, 0.35 + max(trans_probs.values(), default=0.0))
        self._last_cluster[s] = cid

        state = self._state_name(cid)
        micro_map = micro if isinstance(micro, Mapping) else {}
        dangerous = state in ("manipulation", "dangerous", "exhaustion") or float(micro_map.get("manipulation_score", 0)) > 0.6
        clean = state in ("clean", "alive") and float(micro_map.get("micro_quality", 0.5)) > 0.55

        conf = max(0.35, min(0.95, 1.0 - trans_risk * 0.4 + (0.15 if clean else 0.0)))

        return LatentStateSnapshot(
            state=state,
            confidence=conf,
            transition_risk=trans_risk,
            cluster_id=cid,
            is_dangerous=dangerous,
            is_clean=clean,
            transition_probs=trans_probs,
        )

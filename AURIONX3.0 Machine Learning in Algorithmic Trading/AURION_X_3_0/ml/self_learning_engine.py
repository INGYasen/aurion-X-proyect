"""Auto-ajuste de pesos, umbrales y detección de degradación por símbolo."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from analytics.logger import get_logger
from ml.asset_ml_weights import ASSET_PRESETS

logger = get_logger(__name__)


DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 1.0,
    "liquidity_sweep": 1.0,
    "absorption": 1.0,
    "momentum": 1.0,
    "range": 1.0,
    "session": 1.0,
    "volatility": 1.0,
    "fake_breakout": 1.0,
    "displacement": 1.0,
    "manipulation": 1.0,
}


@dataclass
class LearningState:
    adaptive_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    threshold_delta: float = 0.0
    score_bias: int = 0
    regime_label: str = "UNKNOWN"
    degradation_detected: bool = False
    last_eval_trades: int = 0
    lot_factor: float = 1.0
    aggression: float = 1.0
    block_weak_setups: bool = False
    consistency_score: float = 0.5
    funded_priority: float = 1.0


class SelfLearningEngine:
    """Aprende de trades cerrados por símbolo (sin mezclar activos)."""

    def __init__(
        self,
        *,
        min_samples: int = 20,
        lookback: int = 400,
        degradation_winrate: float = 0.38,
        degradation_avg_r: float = -0.18,
        max_drawdown_r: float = -1.2,
    ) -> None:
        self._min_samples = max(8, min_samples)
        self._lookback = max(50, lookback)
        self._degradation_winrate = degradation_winrate
        self._degradation_avg_r = degradation_avg_r
        self._max_drawdown_r = max_drawdown_r
        self.state = LearningState()
        self._state_by_symbol: dict[str, LearningState] = {}
        self._regime_counts: dict[str, int] = defaultdict(int)

    def state_for(self, symbol: str) -> LearningState:
        s = symbol.upper()
        if s not in self._state_by_symbol:
            base = dict(DEFAULT_WEIGHTS)
            base.update(ASSET_PRESETS.get(s, {}))
            self._state_by_symbol[s] = LearningState(adaptive_weights=base)
        return self._state_by_symbol[s]

    @staticmethod
    def _filter_symbol_rows(rows: list[dict[str, Any]], symbol: str | None) -> list[dict[str, Any]]:
        if not symbol:
            return rows
        sym = symbol.upper()
        return [r for r in rows if str(r.get("symbol", "")).upper() == sym]

    @staticmethod
    def _bucket_from_row(row: dict[str, Any], symbol: str) -> str:
        fx = row.get("features_json")
        if not isinstance(fx, dict):
            fx = {}
        setup = str(fx.get("setup_type", row.get("setup_type", "UNKNOWN"))).upper()
        if "RANGE" in setup:
            return "range"
        if "SWEEP" in setup or bool(fx.get("liquidity_sweep", False)):
            return "liquidity_sweep"
        if "FADE" in setup or bool(fx.get("pullback", False)):
            return "absorption"
        if "BREAKOUT" in setup or "MOMENTUM" in setup:
            return "momentum"
        if "BTC" in symbol and ("SPIKE" in setup or fx.get("volatility_regime") == "EXTREME"):
            return "fake_breakout"
        if "USTEC" in symbol or "NAS" in symbol or "NDX" in symbol or "US100" in symbol:
            return "displacement"
        return "trend"

    def ingest_rows(self, rows: list[dict[str, Any]], *, symbol: str | None = None) -> LearningState:
        filtered = self._filter_symbol_rows(rows, symbol)
        sym_key = (symbol or "GLOBAL").upper()
        st = self.state_for(sym_key) if symbol else self.state

        if len(filtered) < self._min_samples:
            return st

        wins = 0
        rs: list[float] = []
        bucket_perf: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "w": 0.0, "sum_r": 0.0})
        session_perf: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "w": 0.0, "sum_r": 0.0})

        for row in filtered[: self._lookback]:
            try:
                profit = float(row.get("profit") or 0.0)
            except (TypeError, ValueError):
                continue
            if profit > 0:
                wins += 1
            raw_r = row.get("r_multiple")
            r_val = None
            if raw_r is not None:
                try:
                    r_val = float(raw_r)
                    rs.append(r_val)
                except (TypeError, ValueError):
                    pass
            fx = row.get("features_json")
            if not isinstance(fx, dict):
                fx = {}
            regime = str(fx.get("volatility_regime", "UNKNOWN")).upper()
            self._regime_counts[regime] += 1
            bucket = self._bucket_from_row(row, sym_key)
            bp = bucket_perf[bucket]
            bp["n"] += 1.0
            if profit > 0:
                bp["w"] += 1.0
            if r_val is not None:
                bp["sum_r"] += r_val
            sess = str(fx.get("session_type", fx.get("hour_slot", "NA")))
            sp = session_perf[sess]
            sp["n"] += 1.0
            if profit > 0:
                sp["w"] += 1.0
            if r_val is not None:
                sp["sum_r"] += r_val

        n = max(len(filtered[: self._lookback]), 1)
        winrate = wins / n
        avg_r = sum(rs) / len(rs) if rs else 0.0
        max_dd = min(rs) if rs else 0.0
        equity_stability = 1.0 - min(1.0, abs(max_dd) / max(abs(self._max_drawdown_r), 0.01))

        st.degradation_detected = (
            winrate < self._degradation_winrate
            and avg_r < self._degradation_avg_r
        ) or max_dd < self._max_drawdown_r
        st.last_eval_trades = n
        st.consistency_score = max(0.0, min(1.0, winrate * 0.5 + equity_stability * 0.5))
        st.funded_priority = st.consistency_score

        if self._regime_counts:
            st.regime_label = max(self._regime_counts, key=self._regime_counts.get)

        preset = dict(DEFAULT_WEIGHTS)
        preset.update(ASSET_PRESETS.get(sym_key, {}))
        weights = dict(preset)
        for key, bp in bucket_perf.items():
            if bp["n"] < 6:
                continue
            wr = bp["w"] / max(bp["n"], 1.0)
            exp_r = bp["sum_r"] / max(bp["n"], 1.0)
            if wr >= 0.54 and exp_r >= 0.08:
                weights[key] = min(1.45, weights.get(key, 1.0) * 1.12)
            elif wr < 0.40 or exp_r < -0.10:
                weights[key] = max(0.55, weights.get(key, 1.0) * 0.88)

        st.adaptive_weights = weights

        if st.degradation_detected:
            st.threshold_delta = 0.05
            st.score_bias = -8
            st.lot_factor = 0.72
            st.aggression = 0.78
            st.block_weak_setups = True
        elif winrate >= 0.52 and avg_r >= 0.05 and st.consistency_score >= 0.52:
            st.threshold_delta = -0.02
            st.score_bias = 2
            st.lot_factor = min(1.08, 1.0 + avg_r * 0.1)
            st.aggression = 1.02
            st.block_weak_setups = False
        else:
            st.threshold_delta = 0.01
            st.score_bias = 0
            st.lot_factor = 0.92
            st.aggression = 0.95
            st.block_weak_setups = False

        if symbol:
            self._state_by_symbol[sym_key] = st
        else:
            self.state = st

        logger.debug(
            "SelfLearning[%s]: wr=%.3f avg_r=%.3f cons=%.3f degrade=%s lot=%.2f",
            sym_key,
            winrate,
            avg_r,
            st.consistency_score,
            st.degradation_detected,
            st.lot_factor,
        )
        return st

    def detect_regime_change(self, current_regime: str, *, symbol: str | None = None) -> bool:
        st = self.state_for(symbol) if symbol else self.state
        if not st.regime_label or st.regime_label == "UNKNOWN":
            return False
        return current_regime.upper() != st.regime_label.upper()

    def should_trigger_retrain(self, closed_since_baseline: int, *, every_n: int, symbol: str | None = None) -> bool:
        st = self.state_for(symbol) if symbol else self.state
        if st.degradation_detected:
            return True
        return closed_since_baseline >= every_n

    @staticmethod
    def is_end_of_trading_day(now_utc: datetime, *, tz_hour_cutoff: int = 22) -> bool:
        return now_utc.hour >= tz_hour_cutoff

    def daily_key(self, now_utc: datetime) -> date:
        return now_utc.date()

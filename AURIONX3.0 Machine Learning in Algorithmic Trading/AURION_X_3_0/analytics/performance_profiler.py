"""Perfil de rendimiento por sesión, hora, setup y régimen."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceProfile:
    overall_winrate: float
    overall_expectancy: float
    best_setup: str
    best_hour: int
    best_session: str
    best_regime: str
    best_dimensions: dict[str, float] = field(default_factory=dict)
    by_setup: dict[str, dict[str, float]] = field(default_factory=dict)
    by_hour: dict[int, dict[str, float]] = field(default_factory=dict)


class PerformanceProfiler:
    def profile(self, rows: list[dict[str, Any]]) -> PerformanceProfile:
        by_setup: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "w": 0, "sum_r": 0.0})
        by_hour: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0, "w": 0, "sum_r": 0.0})
        by_sess: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "w": 0, "sum_r": 0.0})
        by_reg: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "w": 0, "sum_r": 0.0})
        wins = 0
        rs: list[float] = []

        for row in rows:
            try:
                profit = float(row.get("profit") or 0)
                r = float(row.get("r_multiple") or 0)
            except (TypeError, ValueError):
                continue
            if profit > 0:
                wins += 1
            rs.append(r)
            fx = row.get("features_json")
            if not isinstance(fx, dict):
                continue
            st = str(fx.get("setup_type", "UNK"))
            s = by_setup[st]
            s["n"] += 1
            if profit > 0:
                s["w"] += 1
            s["sum_r"] += r
            hr = int(fx.get("hour_slot", -1))
            if 0 <= hr <= 23:
                h = by_hour[hr]
                h["n"] += 1
                if profit > 0:
                    h["w"] += 1
                h["sum_r"] += r
            sess = str(fx.get("session_type", "unknown"))
            ss = by_sess[sess]
            ss["n"] += 1
            if profit > 0:
                ss["w"] += 1
            reg = str(fx.get("volatility_regime", "UNK"))
            rg = by_reg[reg]
            rg["n"] += 1
            if profit > 0:
                rg["w"] += 1

        n = max(len(rows), 1)
        wr = wins / n
        exp = sum(rs) / len(rs) if rs else 0.0

        def _best(d: dict) -> tuple[str, float]:
            best_k, best_v = "", -1.0
            for k, v in d.items():
                if v["n"] < 4:
                    continue
                score = v["w"] / v["n"]
                if score > best_v:
                    best_v = score
                    best_k = k
            return best_k, best_v

        bs, _ = _best(by_setup)
        bh_tuple = max(
            ((h, v["w"] / max(v["n"], 1)) for h, v in by_hour.items() if v["n"] >= 4),
            key=lambda x: x[1],
            default=(0, 0.0),
        )
        bsess, _ = _best(by_sess)
        breg, _ = _best(by_reg)

        return PerformanceProfile(
            overall_winrate=wr,
            overall_expectancy=exp,
            best_setup=bs,
            best_hour=int(bh_tuple[0]),
            best_session=bsess,
            best_regime=breg,
            best_dimensions={k: v["w"] / max(v["n"], 1) for k, v in by_setup.items() if v["n"] >= 4},
            by_setup=dict(by_setup),
            by_hour=dict(by_hour),
        )

"""Validación OOS: no promover modelos que degradan métricas funded.

AURION X 4.0: la clase ``OOSValidator`` admite gates estadísticos opcionales
(DSR + PBO, Bailey & López de Prado) detrás de feature flag. El comportamiento
3.0 se conserva intacto cuando ``v4_hybrid.statistical_gates.enabled`` es
``False``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _safe_mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        if peak > 0:
            max_dd = max(max_dd, (peak - x) / peak)
    return max_dd


def _sharpe(rs: Sequence[float]) -> float:
    if len(rs) < 3:
        return 0.0
    m = _safe_mean(rs)
    var = sum((r - m) ** 2 for r in rs) / max(len(rs) - 1, 1)
    sd = math.sqrt(var)
    if sd < 1e-9:
        return 0.0
    return (m / sd) * math.sqrt(min(len(rs), 252))


def compute_oos_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    r_multiples: Sequence[float] | None = None,
    train_accuracy: float | None = None,
) -> dict[str, float]:
    """Métricas de trading + estabilidad para ml_model_performance."""
    n = max(len(y_true), 1)
    correct = sum(int(p == t) for p, t in zip(y_pred, y_true, strict=False))
    acc = correct / n
    wins = sum(1 for t in y_true if int(t) == 1)
    winrate = wins / n

    rs = [float(r) for r in (r_multiples or []) if r is not None]
    expectancy = _safe_mean(rs) if rs else (winrate * 2.0 - 1.0) * 0.15
    sharpe = _sharpe(rs) if rs else 0.0

    equity = [1.0]
    for r in rs:
        equity.append(equity[-1] + r * 0.01)
    drawdown = _max_drawdown(equity) * 100.0 if rs else 0.0

    gross_win = sum(r for r in rs if r > 0)
    gross_loss = abs(sum(r for r in rs if r < 0))
    profit_factor = gross_win / gross_loss if gross_loss > 1e-9 else (2.0 if gross_win > 0 else 1.0)

    stability = 1.0
    if len(rs) >= 12:
        chunk = max(3, len(rs) // 4)
        wrs = []
        for i in range(0, len(rs) - chunk + 1, chunk):
            seg = rs[i : i + chunk]
            wrs.append(sum(1 for x in seg if x > 0) / len(seg))
        if wrs:
            m_wr = _safe_mean(wrs)
            var_wr = sum((w - m_wr) ** 2 for w in wrs) / len(wrs)
            stability = max(0.0, min(1.0, 1.0 - math.sqrt(var_wr)))

    consistency = max(0.0, min(1.0, (winrate * 0.4 + (1.0 - drawdown / 15.0) * 0.35 + stability * 0.25)))

    overfit = 0.0
    if train_accuracy is not None and train_accuracy - acc > 0.12:
        overfit = min(1.0, (train_accuracy - acc) / 0.25)

    confidence_penalty = min(0.25, overfit * 0.5 + max(0.0, drawdown - 8.0) * 0.02)

    return {
        "oos_accuracy": float(acc),
        "accuracy": float(acc),
        "winrate": float(winrate),
        "expectancy": float(expectancy),
        "sharpe": float(sharpe),
        "drawdown": float(drawdown),
        "profit_factor": float(profit_factor),
        "stability_score": float(stability),
        "consistency_score": float(consistency),
        "overfitting_score": float(overfit),
        "confidence_penalty": float(confidence_penalty),
        "n_test": float(n),
        "false_signal_rate": float(1.0 - acc),
    }


class OOSValidator:
    """Gate de promoción orientado a supervivencia funded, no solo winrate.

    A partir de AURION X 4.0 admite además gates estadísticos formales (DSR +
    PBO). Si ``statistical_gates.enabled`` es ``False`` (default), el flujo
    permanece idéntico al 3.0.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        *,
        statistical_gates_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        c = dict(cfg or {})
        self._min_expectancy = float(c.get("min_expectancy", -0.05))
        self._max_drawdown = float(c.get("max_oos_drawdown_pct", 12.0))
        self._min_sharpe = float(c.get("min_sharpe", -0.5))
        self._max_overfit = float(c.get("max_overfitting", 0.35))
        self._min_improve_rank = float(c.get("min_rank_improve", 0.002))

        # v4: gates estadísticos (off por defecto)
        sg = dict(statistical_gates_cfg or {})
        self._stat_enabled = bool(sg.get("enabled", False))
        self._min_dsr = float(sg.get("min_dsr", 0.60))
        self._max_pbo = float(sg.get("max_pbo", 0.30))
        self._n_trials_default = int(sg.get("n_trials_default", 100))
        self._bars_per_year = int(sg.get("bars_per_year_value", 525_600))

    @staticmethod
    def composite_rank(metrics: dict[str, float]) -> float:
        sharpe = metrics.get("sharpe", 0.0)
        exp = metrics.get("expectancy", 0.0)
        cons = metrics.get("consistency_score", 0.5)
        stab = metrics.get("stability_score", 0.5)
        acc = metrics.get("oos_accuracy", metrics.get("accuracy", 0.0))
        pen = metrics.get("confidence_penalty", 0.0)
        dd_pen = min(0.3, metrics.get("drawdown", 0.0) / 40.0)
        return (
            sharpe * 0.22
            + exp * 0.28
            + cons * 0.20
            + stab * 0.12
            + float(acc) * 0.18
            - pen
            - dd_pen
        )

    def passes_hard_gates(self, metrics: dict[str, float]) -> tuple[bool, str]:
        if metrics.get("expectancy", 0.0) < self._min_expectancy:
            return False, "expectancy_too_low"
        if metrics.get("drawdown", 0.0) > self._max_drawdown:
            return False, "drawdown_too_high"
        if metrics.get("sharpe", 0.0) < self._min_sharpe:
            return False, "sharpe_too_low"
        if metrics.get("overfitting_score", 0.0) > self._max_overfit:
            return False, "overfitting"
        if metrics.get("n_test", 0) < 8:
            return False, "insufficient_oos_samples"
        return True, "ok"

    def should_promote(
        self,
        candidate: dict[str, float],
        best: dict[str, float] | None,
        *,
        oos_returns: Sequence[float] | None = None,
        candidates_returns_matrix: Any = None,
        n_trials: int | None = None,
    ) -> tuple[bool, str]:
        """Decide si promover un modelo.

        Args:
            candidate: métricas del modelo candidato.
            best: métricas del mejor modelo actual.
            oos_returns: retornos OOS del candidato (solo si stat_gates activo).
            candidates_returns_matrix: matriz (T, N) con retornos de los
                últimos N candidatos para calcular PBO. Opcional.
            n_trials: número de configuraciones probadas en la búsqueda. Si
                ``None``, se usa el default de config.
        """
        ok, reason = self.passes_hard_gates(candidate)
        if not ok:
            return False, reason

        # v4: gates estadísticos (DSR + PBO) ANTES de comparar con best
        if self._stat_enabled and oos_returns is not None:
            from .statistical_gates import passes_statistical_gates  # lazy

            trials = int(n_trials) if n_trials is not None else self._n_trials_default
            stat_ok, stat_reason, _stat_metrics = passes_statistical_gates(
                returns=oos_returns,
                n_trials=trials,
                returns_matrix=(
                    np.asarray(candidates_returns_matrix, dtype=float)
                    if candidates_returns_matrix is not None else None
                ),
                min_dsr=self._min_dsr,
                max_pbo=self._max_pbo,
                bars_per_year=self._bars_per_year,
            )
            if not stat_ok:
                return False, f"statistical_gate:{stat_reason}"

        if not best:
            return True, "first_model"
        rank_new = self.composite_rank(candidate)
        rank_old = self.composite_rank(best)
        if rank_new < rank_old + self._min_improve_rank:
            return False, "no_improvement_vs_best"
        return True, "promoted"

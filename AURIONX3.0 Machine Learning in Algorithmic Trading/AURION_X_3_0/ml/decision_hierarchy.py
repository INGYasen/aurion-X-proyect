"""Jerarquía de decisión: riesgo → contexto → régimen → micro → ML → setup → ejecución."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class HierarchyVerdict:
    allow: bool
    failed_layer: str
    reason: str
    confidence_adjust: float


class DecisionHierarchy:
    """Gates en orden; si falla uno superior, no trade."""

    def evaluate(
        self,
        *,
        risk_ok: bool,
        risk_reason: str,
        hyper_allow: bool,
        hyper_reason: str,
        regime_dangerous: bool,
        micro_quality: float,
        manipulation_score: float,
        ml_confidence: float,
        ml_threshold: float,
        setup_grade: str,
        execution_block: bool,
        anti_overtrade_block: bool,
        ml_neutral_override: bool = False,
        ml_strong_override: bool = False,
    ) -> HierarchyVerdict:
        if not risk_ok:
            return HierarchyVerdict(False, "risk", risk_reason, 0.0)
        if anti_overtrade_block:
            return HierarchyVerdict(False, "anti_overtrade", "hyperactivity", 0.0)
        if not hyper_allow:
            return HierarchyVerdict(False, "context", hyper_reason, 0.0)
        # regime_dangerous: si el régimen latente está marcado como peligroso
        # bloqueamos por defecto. Excepción: ml_strong_override + setup A/A+
        # representa una convicción combinada (modelo + ranker) que ya filtró
        # señales mediocres. Hyper_context puede tomar esa decisión con su
        # propio override; aquí espejamos el criterio para no bloquear de
        # nuevo el mismo trade.
        if regime_dangerous and not (
            ml_strong_override and setup_grade in ("A+", "A")
        ):
            return HierarchyVerdict(False, "regime", "latent_dangerous", 0.0)
        if micro_quality < 0.30 or manipulation_score > 0.78:
            return HierarchyVerdict(False, "microstructure", "poor_micro_or_manipulation", -0.05)
        # Cold-start: si el modelo ML no es útil (mismatch / no entrenado), no
        # podemos confiar en ``ml_confidence`` ni en ``setup_grade`` (que depende
        # del ML scoring). Bypass ambos para permitir generar histórico.
        if not ml_neutral_override:
            # Bypass de ``below_threshold`` cuando hay ml_strong_override Y el
            # ranker considera el setup A+/A/B. El effective_confidence puede
            # ser arrastrado por componentes con histórico pobre (institutional,
            # quality) y producir below_threshold a pesar de ml_raw ≥ 0.55 y un
            # setup de buena calidad. Apex micro/context/exec siguen como filtros.
            below = ml_confidence < ml_threshold
            strong_with_good_grade = ml_strong_override and setup_grade in ("A+", "A", "B")
            if below and not strong_with_good_grade:
                return HierarchyVerdict(False, "ml", "below_threshold", 0.0)
            # ML strong override: cuando el modelo está claramente sesgado (>=0.65)
            # confiamos en él aunque setup_grade sea "X" (ranker entrenado con
            # histórico malo). Sólo bypass del setup_rank, NO de below_threshold.
            if setup_grade in ("X",) and not ml_strong_override:
                return HierarchyVerdict(False, "setup_rank", "trash_grade", 0.0)
        if execution_block:
            return HierarchyVerdict(False, "execution", "poor_execution", 0.0)
        adj = 0.02 if setup_grade in ("A+", "A") else 0.0
        return HierarchyVerdict(True, "entry", "ok", adj)

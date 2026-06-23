"""Contexto total del mercado — fusión de todas las capas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HyperContext:
    score: float
    trade_quality: float
    risk_penalty: float
    execution_penalty: float
    micro_penalty: float
    latent_penalty: float
    allow_trade: bool
    block_reason: str
    fused_confidence: float
    cold_start: bool = False
    min_score_used: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hyper_score": self.score,
            "hyper_trade_quality": self.trade_quality,
            "hyper_fused_confidence": self.fused_confidence,
            "hyper_allow": self.allow_trade,
            "hyper_block": self.block_reason,
            "hyper_cold_start": self.cold_start,
            "hyper_min_score_used": self.min_score_used,
        }


_BASE_WEIGHTS: dict[str, float] = {
    "ml": 0.30,
    "setup": 0.25,
    "micro": 0.20,
    "exec": 0.15,
    "equity": 0.10,
}


def _is_ml_neutral(ml_confidence: float, *, override: bool = False) -> bool:
    """ML neutral implica modelo no útil/no cargado: redistribuir su peso.

    Se acepta tolerancia amplia (0.5 +/- 0.02) o flag explícito ``override`` cuando
    el caller ya detectó mismatch / modelo no entrenado.
    """
    if override:
        return True
    return abs(float(ml_confidence) - 0.5) < 0.02


def _is_equity_unknown(equity_health: float) -> bool:
    """Sin histórico de equity para calificar: no penalizar, redistribuir."""
    return float(equity_health) <= 0.0


class HyperContextEngine:
    """Integra sesión, régimen, micro, latente, ejecución, consistencia.

    Cold-start: si el predictor devuelve ML neutral (0.5) o si el equity_health
    no tiene histórico (<=0), los pesos correspondientes se redistribuyen
    proporcionalmente sobre el resto y el umbral mínimo baja a ``cold_start_min``.
    Así el bot puede abrir trades para generar histórico sin quedar bloqueado
    en ``low_hyper_score`` para siempre.
    """

    def __init__(
        self,
        *,
        min_score: float = 0.42,
        cold_start_min: float = 0.30,
    ) -> None:
        self._min_score = float(min_score)
        self._cold_start_min = float(min(cold_start_min, min_score))

    def fuse(
        self,
        *,
        ml_confidence: float,
        setup_score: float,
        regime_mode: str,
        micro: Mapping[str, float],
        latent: Mapping[str, Any],
        exec_intel: Mapping[str, Any],
        equity_health: float,
        meta_risk_mode: str,
        reinforcement_lot: float,
        ml_neutral_override: bool = False,
        ml_strong_override: bool = False,
        ml_raw_confidence: float | None = None,
    ) -> HyperContext:
        micro_q = float(micro.get("micro_quality", 0.5))
        latent_danger = bool(latent.get("latent_dangerous", False))
        latent_clean = bool(latent.get("latent_clean", False))
        exec_block = bool(exec_intel.get("exec_intel_block", False))
        exec_score = float(exec_intel.get("exec_intel_score", 0.6))

        ml_neutral = _is_ml_neutral(ml_confidence, override=ml_neutral_override)

        micro_pen = 0.0 if micro_q >= 0.5 else (0.5 - micro_q)
        # latent_pen relajado: si no hay clasificación explícita (ni clean ni
        # dangerous) no penalizamos tan duro (0.04) para no matar el trade_q en
        # mercados normales/sin señal latente clara.
        latent_pen = 0.25 if latent_danger else (0.0 if latent_clean else 0.04)
        exec_pen = 0.35 if exec_block else max(0.0, 0.55 - exec_score)
        risk_pen = 0.15 if meta_risk_mode == "defensive" else 0.0
        # ML strong: reduce micro_pen y latent_pen a la mitad. micro_quality
        # tiende a bajar en chop (NDX100/XAUUSD micro=0.27-0.40) pero el modelo
        # sigue siendo capaz de detectar oportunidades. No castigar 2x.
        if ml_strong_override and not ml_neutral:
            micro_pen *= 0.5
            latent_pen *= 0.5
        equity_unknown = _is_equity_unknown(equity_health)
        weights = dict(_BASE_WEIGHTS)
        if ml_neutral:
            weights["ml"] = 0.0
        if equity_unknown:
            weights["equity"] = 0.0
        # ML strong: el modelo está claramente sesgado. El setup_score depende del
        # ranker que se entrena con histórico (a menudo malo), así que reducimos
        # su peso a la mitad y reforzamos ml; equity también pesa menos cuando hay
        # buen ML pero poca historia confiable.
        if ml_strong_override and not ml_neutral:
            weights["setup"] *= 0.5
            weights["ml"] *= 1.6
        active_sum = sum(weights.values())
        if active_sum > 0:
            scale = 1.0 / active_sum
            for k in weights:
                weights[k] *= scale

        setup_norm = min(1.0, max(0.0, float(setup_score) / 100.0))
        # Cuando hay ml_strong y el setup ranker dio score~0 (grade X) pero el ML
        # raw confía, usamos el ML como proxy del setup para no anular ese eje.
        if ml_strong_override and setup_norm < 0.1 and ml_raw_confidence is not None:
            setup_norm = max(setup_norm, 0.5 * float(ml_raw_confidence))
        ml_axis = float(ml_raw_confidence) if (ml_strong_override and ml_raw_confidence is not None) else float(ml_confidence)
        trade_q = (
            ml_axis * weights["ml"]
            + setup_norm * weights["setup"]
            + micro_q * weights["micro"]
            + exec_score * weights["exec"]
            + max(0.0, float(equity_health)) * weights["equity"]
        )
        trade_q = max(0.0, min(1.0, trade_q - micro_pen - latent_pen - exec_pen - risk_pen))

        fused = max(0.0, min(0.99, trade_q * reinforcement_lot))
        cold_start = ml_neutral or equity_unknown or ml_strong_override
        min_score = self._cold_start_min if cold_start else self._min_score
        # latent_danger override: si tenemos ML strong override + ML raw
        # confidence muy alta + setup ranker razonable (grade A/A+), aceptamos
        # operar aún con latent_danger. La regla previa bloqueaba trades
        # grade=A con ML 0.80+ (visto en logs nocturnos), perdiendo
        # oportunidades válidas. Si el ML está MUY convencido y el setup es
        # de calidad, vale la pena el riesgo (penalizamos trade_q en estos
        # casos para que el sizing sea más conservador en otros ejes).
        latent_danger_overridden = False
        if (
            latent_danger
            and ml_strong_override
            and ml_raw_confidence is not None
            and float(ml_raw_confidence) >= 0.80
            and float(setup_score) >= 70.0
        ):
            latent_danger_overridden = True

        allow = trade_q >= min_score and not exec_block and (
            not latent_danger or latent_danger_overridden
        )
        block = ""
        if exec_block:
            block = "execution"
        elif latent_danger and not latent_danger_overridden:
            block = "latent_danger"
        elif trade_q < min_score:
            block = "low_hyper_score_cold_start" if cold_start else "low_hyper_score"

        if not allow and block.startswith("low_hyper_score"):
            logger.debug(
                "HyperContext block: trade_q=%.3f min=%.3f cold_start=%s "
                "ml=%.3f setup=%.3f micro=%.3f exec=%.3f equity=%.3f",
                trade_q,
                min_score,
                cold_start,
                float(ml_confidence),
                setup_norm,
                micro_q,
                exec_score,
                float(equity_health),
            )

        return HyperContext(
            score=trade_q,
            trade_quality=trade_q,
            risk_penalty=risk_pen,
            execution_penalty=exec_pen,
            micro_penalty=micro_pen,
            latent_penalty=latent_pen,
            allow_trade=allow,
            block_reason=block,
            fused_confidence=fused,
            cold_start=cold_start,
            min_score_used=min_score,
        )

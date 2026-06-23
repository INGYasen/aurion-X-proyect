"""Logs cuantitativos estructurados para entradas y bloqueos."""

from __future__ import annotations

from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


class QuantDecisionLogger:
    """Registra por qué entró o bloqueó con scores desglosados."""

    def log_entry(
        self,
        *,
        symbol: str,
        direction: str,
        allow: bool,
        block_reason: str,
        scores: Mapping[str, float],
        layers: Mapping[str, Any] | None = None,
    ) -> None:
        msg = (
            "[QUANT_DECISION] %s %s allow=%s block=%s "
            "total=%.3f ml=%.3f inst=%.3f micro=%.3f risk=%.3f exec=%.3f hyper=%.3f"
        )
        sc = dict(scores)
        logger.info(
            msg,
            symbol,
            direction,
            allow,
            block_reason or "-",
            float(sc.get("total", 0)),
            float(sc.get("ml", 0)),
            float(sc.get("institutional", 0)),
            float(sc.get("micro", 0)),
            float(sc.get("risk", 0)),
            float(sc.get("execution", 0)),
            float(sc.get("hyper", 0)),
            extra={"quant_layers": dict(layers or {})},
        )

    def log_block(self, symbol: str, layer: str, reason: str, **scores: float) -> None:
        logger.info(
            "[QUANT_BLOCK] %s layer=%s reason=%s scores=%s",
            symbol,
            layer,
            reason,
            scores,
        )

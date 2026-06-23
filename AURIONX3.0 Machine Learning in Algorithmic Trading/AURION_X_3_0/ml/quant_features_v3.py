"""AURION X 4.0 — Schema v3 (34 features = v2 + 9 oro-específicas).

Compatibilidad:
    - Extiende ``FEATURE_ORDER_V2`` con ``GOLD_FEATURE_ORDER``.
    - Función ``build_quant_features_v3`` envuelve la v2 y concatena gold
      features.
    - Mantiene la v2 intacta: si ``gold_features.enabled`` es false, el
      orchestrator sigue usando v2 sin modificación.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import numpy as np

from .gold_specific_features import (
    GOLD_FEATURE_ORDER,
    GOLD_FEATURE_LEN,
    GoldFeatureProvider,
)
from .quant_features import (
    FEATURE_ORDER_V2,
    FEATURE_SCHEMA_VERSION_V2,
    FEATURE_VECTOR_LEN_V2,
    build_quant_features,
)

FEATURE_ORDER_V3: tuple[str, ...] = FEATURE_ORDER_V2 + GOLD_FEATURE_ORDER
FEATURE_VECTOR_LEN_V3: int = len(FEATURE_ORDER_V3)
FEATURE_SCHEMA_VERSION_V3: int = 3


def build_quant_features_v3(
    market_state: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    gold_provider: GoldFeatureProvider | None = None,
    now_utc: datetime | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Construye vector v3 + dict de features.

    Args:
        market_state: estado de mercado (igual que v2).
        context: contexto opcional (sesión, calendario).
        gold_provider: si ``None`` se crea uno temporal (sin caché persistente).
        now_utc: timestamp UTC; default ``datetime.now``.

    Returns:
        Tupla ``(vector_v3, features_dict)``. El dict mezcla las features v2
        con las 9 features gold y la flag ``schema_version_v3``.
    """
    vec_v2, fdict = build_quant_features(market_state, context=context)
    provider = gold_provider or GoldFeatureProvider()
    bundle = provider.compute(market_state=market_state, context=context, now_utc=now_utc)
    gold_vals = bundle.values
    if len(gold_vals) != GOLD_FEATURE_LEN:
        raise ValueError(
            f"gold bundle len={len(gold_vals)} != esperado {GOLD_FEATURE_LEN}"
        )
    full = list(vec_v2) + list(gold_vals)
    if len(full) != FEATURE_VECTOR_LEN_V3:
        raise ValueError(
            f"vector v3 len={len(full)} != esperado {FEATURE_VECTOR_LEN_V3}"
        )
    fdict = dict(fdict)
    fdict.update(bundle.as_dict())
    fdict["schema_version"] = FEATURE_SCHEMA_VERSION_V3
    return full, fdict


def vector_from_features_json_v3(raw: Any) -> list[float] | None:
    """Convierte un blob persistido (dict o list) a vector v3."""
    if isinstance(raw, dict):
        out: list[float] = []
        for k in FEATURE_ORDER_V3:
            try:
                out.append(float(raw.get(k, 0.0) or 0.0))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
    if isinstance(raw, list):
        try:
            vals = [float(v) for v in raw]
        except (TypeError, ValueError):
            return None
        if len(vals) >= FEATURE_VECTOR_LEN_V3:
            return vals[:FEATURE_VECTOR_LEN_V3]
        return vals + [0.0] * (FEATURE_VECTOR_LEN_V3 - len(vals))
    return None

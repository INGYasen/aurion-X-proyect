"""Inferencia con salida prob_buy / prob_sell / confidence."""

from __future__ import annotations

import warnings
from typing import Any, TypedDict  # Any: TradingModel | EnsembleTradingModel

import numpy as np

# Silencia el spam UserWarning de sklearn/LGBM "X does not have valid feature
# names" en cada predicción (el modelo fue entrenado con DataFrame y se
# predice con ndarray ya alineado por el feature pipeline). Es ruido, no error.
for _pat in (
    r".*does not have valid feature names.*",
    r".*X has feature names.*",
):
    warnings.filterwarnings("ignore", message=_pat, category=UserWarning)

from analytics.logger import get_logger

from .feature_engineering import FEATURE_ORDER, FEATURE_SCHEMA_VERSION
from .model import TradingModel
from .quant_features import (
    FEATURE_ORDER_V2,
    FEATURE_ORDER_V3,
    FEATURE_SCHEMA_VERSION_V2,
    FEATURE_SCHEMA_VERSION_V3,
)

logger = get_logger(__name__)


class PredictionResult(TypedDict):
    """Salida de ``Predictor.predict``."""

    prob_buy: float
    prob_sell: float
    confidence: float


def _indices_buy_sell(classes: list[int]) -> tuple[int, int]:
    """Devuelve índices de columnas predict_proba para vender y comprar."""
    s = set(classes)
    if s == {0, 1}:
        return classes.index(0), classes.index(1)
    if s == {-1, 1}:
        return classes.index(-1), classes.index(1)
    if len(classes) == 2:
        logger.warning(
            "Predictor: clases no estándar %s; se asume [sell, buy] por orden.",
            classes,
        )
        return 0, 1
    raise ValueError(
        f"Se esperan 2 clases (0/1 o -1/1); clases del modelo: {classes}",
    )


class Predictor:
    """Aplica ``TradingModel`` y expone probabilidades interpretables."""

    def __init__(self, model: TradingModel | Any | None = None) -> None:
        """Asocia un modelo cargado.

        Args:
            model: Instancia entrenada o cargada desde PKL.
        """
        self._model = model
        self._mismatch_warned_for: set[tuple[int, int]] = set()
        self._mismatch_silent_count: int = 0
        self._schema_warned_versions: set[tuple[str, int | None]] = set()
        logger.debug("Predictor model asignado: %s", model is not None)

    @property
    def model(self) -> Any | None:
        """Modelo subyacente."""
        return self._model

    def set_model(self, model: TradingModel | Any) -> None:
        """Sustituye el modelo asociado."""
        self._model = model
        self._mismatch_warned_for.clear()
        self._mismatch_silent_count = 0
        self._schema_warned_versions.clear()

    def expected_n_features(self) -> int | None:
        """Dimensión esperada del vector de features, si el modelo ya está calibrado."""
        if self._model is None:
            return None
        return self._model.n_features_expected()

    def predict(self, features: list[float]) -> PredictionResult:
        """Inferencia para un único vector de features.

        Args:
            features: Vector ``(n_features,)`` alineado con el entrenamiento.

        Returns:
            Dict con ``prob_buy``, ``prob_sell`` y ``confidence = max(prob_buy, prob_sell)``.
        """
        if self._model is None:
            raise RuntimeError("Predictor sin modelo; asigna un TradingModel entrenado.")
        if not features:
            raise ValueError("features no puede estar vacío.")

        n_exp = self.expected_n_features()
        if n_exp is not None and len(features) != n_exp:
            key = (len(features), int(n_exp))
            if key not in self._mismatch_warned_for:
                self._mismatch_warned_for.add(key)
                logger.warning(
                    "Predictor: len(features)=%s != modelo n_features=%s; salida neutral. "
                    "Reentrena (scripts/retrain_v2_all_symbols.ps1) o revisa pipeline. "
                    "Este mensaje no se repetirá para esta combinación.",
                    len(features),
                    n_exp,
                )
            else:
                self._mismatch_silent_count += 1
                if self._mismatch_silent_count % 1000 == 0:
                    logger.info(
                        "Predictor: %s evaluaciones omitidas por mismatch len=%s vs n_features=%s",
                        self._mismatch_silent_count,
                        len(features),
                        n_exp,
                    )
            return PredictionResult(prob_buy=0.5, prob_sell=0.5, confidence=0.5)

        names = self._model.metadata().get("feature_names")
        ver = self._model.feature_schema_version()
        # AURION X 4.3: soporte schema v3 (v2 + 5 features macro DXY/US10Y/VIX).
        if ver == FEATURE_SCHEMA_VERSION_V3:
            expected_names = FEATURE_ORDER_V3
            expected_ver = FEATURE_SCHEMA_VERSION_V3
        elif ver == FEATURE_SCHEMA_VERSION_V2:
            expected_names = FEATURE_ORDER_V2
            expected_ver = FEATURE_SCHEMA_VERSION_V2
        else:
            expected_names = FEATURE_ORDER
            expected_ver = FEATURE_SCHEMA_VERSION
        if names is not None and list(names) != list(expected_names):
            key_names: tuple[str, int | None] = ("names_mismatch", ver)
            if key_names not in self._schema_warned_versions:
                self._schema_warned_versions.add(key_names)
                logger.warning(
                    "Predictor: orden/nombres de features del PKL no coinciden con schema v%s; salida neutral",
                    ver,
                )
            return PredictionResult(prob_buy=0.5, prob_sell=0.5, confidence=0.5)
        if ver is not None and ver not in (
            FEATURE_SCHEMA_VERSION,
            FEATURE_SCHEMA_VERSION_V2,
            FEATURE_SCHEMA_VERSION_V3,
        ):
            key_ver: tuple[str, int | None] = ("unsupported_schema", ver)
            if key_ver not in self._schema_warned_versions:
                self._schema_warned_versions.add(key_ver)
                logger.warning(
                    "Predictor: feature_schema_version PKL=%s no soportada; salida neutral",
                    ver,
                )
            return PredictionResult(prob_buy=0.5, prob_sell=0.5, confidence=0.5)
        if ver is not None and ver != expected_ver and len(features) == len(FEATURE_ORDER):
            # Modelo v1 con vector corto en runtime v2: permitir si dimensiones coinciden con PKL
            pass

        classes = list(self._model.estimator.classes_)
        idx_sell, idx_buy = _indices_buy_sell(classes)

        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        proba = self._model.estimator.predict_proba(row)[0]
        prob_sell = float(proba[idx_sell])
        prob_buy = float(proba[idx_buy])
        confidence = max(prob_buy, prob_sell)

        return PredictionResult(
            prob_buy=prob_buy,
            prob_sell=prob_sell,
            confidence=confidence,
        )

    def update_context(self, market_state: dict[str, Any]) -> None:
        """Reservado para buffers incrementales (sin lógica aún)."""
        raise NotImplementedError

    def score(self) -> float:
        """Reservado para score agregado de contexto (sin lógica aún)."""
        raise NotImplementedError

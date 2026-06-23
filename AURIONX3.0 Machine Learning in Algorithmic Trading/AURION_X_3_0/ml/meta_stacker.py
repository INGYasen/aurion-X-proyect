"""AURION X 4.0 — Meta-stacker que fusiona ensemble 3.0 + PatchTST + contexto.

Modelo de segundo nivel (LogisticRegression de sklearn) que recibe:

    [ prob_ensemble, prob_patchtst, effective_confidence, setup_grade_score ]

y devuelve la probabilidad final de ``buy``. Entrenado con purged k-fold CV
sobre los últimos N trades cerrados.

Compatibilidad:
    - Si PatchTST no está activo o no hay stacker entrenado, ``predict()``
      devuelve la probabilidad del ensemble actual sin modificación.
    - Persistencia PKL en ``D:/AURION_DATA/models/meta_stacker_xauusd_v4.pkl``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from analytics.logger import get_logger

logger = get_logger(__name__)

_GRADE_SCORE = {"A+": 1.0, "A": 0.85, "B": 0.65, "C": 0.40, "X": 0.10}

_FEATURE_NAMES = (
    "prob_ensemble",
    "prob_patchtst",
    "effective_confidence",
    "setup_grade_score",
)


@dataclass(frozen=True)
class MetaStackerPrediction:
    prob_buy: float
    prob_sell: float
    confidence: float
    used_patchtst: bool


class MetaStacker:
    """Meta-stacker basado en regresión logística."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        c = dict(cfg or {})
        self._enabled = bool(c.get("enabled", False))
        self._model_path = Path(str(c.get("model_path", "")))
        self._min_samples = int(c.get("min_samples_to_fit", 200))
        self._fallback = bool(c.get("fallback_to_ensemble", True))
        self._clf: LogisticRegression | None = None
        self._feature_means: np.ndarray | None = None  # fallback imputación NaN
        self._loaded = False

        if self._enabled and self._model_path and self._model_path.is_file():
            self._try_load()

    @property
    def is_fitted(self) -> bool:
        return self._clf is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _FEATURE_NAMES

    @staticmethod
    def grade_score(grade: str | None) -> float:
        if not grade:
            return 0.0
        return float(_GRADE_SCORE.get(str(grade).upper(), 0.0))

    def _try_load(self) -> bool:
        try:
            with self._model_path.open("rb") as fh:
                payload = pickle.load(fh)
            self._clf = payload["clf"]
            self._feature_means = np.asarray(payload["feature_means"], dtype=np.float64)
            self._loaded = True
            logger.info("MetaStacker cargado: %s", self._model_path.name)
            return True
        except (FileNotFoundError, OSError, KeyError, pickle.UnpicklingError) as exc:
            logger.warning("MetaStacker no se pudo cargar (%s): %s", self._model_path, exc)
            self._clf = None
            return False

    def save(self) -> None:
        if self._clf is None:
            raise RuntimeError("MetaStacker sin entrenar.")
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with self._model_path.open("wb") as fh:
            pickle.dump(
                {"clf": self._clf, "feature_means": self._feature_means},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _build_row(
        self,
        *,
        prob_ensemble: float,
        prob_patchtst: float | None,
        effective_confidence: float,
        setup_grade: str | None,
    ) -> tuple[np.ndarray, bool]:
        """Devuelve ``(vector, used_patchtst)``."""
        used = prob_patchtst is not None and np.isfinite(prob_patchtst)
        ppt = float(prob_patchtst) if used else (
            float(self._feature_means[1]) if self._feature_means is not None else 0.5
        )
        vec = np.array(
            [
                float(prob_ensemble),
                ppt,
                float(effective_confidence),
                self.grade_score(setup_grade),
            ],
            dtype=np.float64,
        )
        return vec, used

    def fit(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        y: np.ndarray | Sequence[int],
        *,
        sample_weight: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        """Entrena el stacker.

        Args:
            X: matriz ``(n_samples, 4)`` con las columnas de
                :pyattr:`feature_names`.
            y: labels binarios (1 = buy correcto, 0 = no).
            sample_weight: pesos opcionales por muestra.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        if X_arr.ndim != 2 or X_arr.shape[1] != 4:
            raise ValueError(f"X debe tener shape (n, 4); recibido {X_arr.shape}")
        if X_arr.shape[0] < self._min_samples:
            raise ValueError(
                f"Insuficientes muestras: {X_arr.shape[0]} < min={self._min_samples}"
            )

        # Imputación: NaN → media de columna. Si TODA la columna es NaN (caso
        # típico: prob_patchtst durante el bootstrap inicial de v4), usamos
        # un default neutro (0.5 para probabilidades, 0.0 para grade).
        with np.errstate(invalid="ignore"):
            col_means = np.nanmean(X_arr, axis=0)
        default_fill = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float64)
        col_means = np.where(np.isfinite(col_means), col_means, default_fill)
        self._feature_means = col_means
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            X_arr[:, j] = np.where(np.isfinite(col), col, col_means[j])

        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        if sample_weight is not None:
            clf.fit(X_arr, y_arr, sample_weight=np.asarray(sample_weight, dtype=np.float64))
        else:
            clf.fit(X_arr, y_arr)
        self._clf = clf

    def predict(
        self,
        *,
        prob_ensemble: float,
        prob_patchtst: float | None,
        effective_confidence: float,
        setup_grade: str | None,
    ) -> MetaStackerPrediction:
        """Predice probabilidad final. Si no hay stacker, devuelve el ensemble."""
        if self._clf is None:
            if self._fallback:
                pe = max(0.0, min(1.0, float(prob_ensemble)))
                return MetaStackerPrediction(
                    prob_buy=pe,
                    prob_sell=1.0 - pe,
                    confidence=max(pe, 1.0 - pe),
                    used_patchtst=False,
                )
            raise RuntimeError("MetaStacker sin entrenar y fallback desactivado.")

        vec, used = self._build_row(
            prob_ensemble=prob_ensemble,
            prob_patchtst=prob_patchtst,
            effective_confidence=effective_confidence,
            setup_grade=setup_grade,
        )
        proba = self._clf.predict_proba(vec.reshape(1, -1))[0]
        # Mapeo robusto a buy: clase 1 = buy.
        classes = list(self._clf.classes_)
        if 1 in classes:
            p_buy = float(proba[classes.index(1)])
        else:
            p_buy = float(proba[-1])
        p_buy = max(0.0, min(1.0, p_buy))
        return MetaStackerPrediction(
            prob_buy=p_buy,
            prob_sell=1.0 - p_buy,
            confidence=max(p_buy, 1.0 - p_buy),
            used_patchtst=bool(used),
        )

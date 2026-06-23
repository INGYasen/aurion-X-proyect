"""Modelo de trading con RandomForestClassifier y persistencia PKL."""

from __future__ import annotations

import contextlib
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Iterator, Sequence, cast

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from analytics.logger import get_logger

logger = get_logger(__name__)


def _default_rf_n_jobs() -> int:
    """Por defecto ``n_jobs=1`` (sin workers joblib → sin spam de warnings ni stderr en hijos).

    ``AURION_RF_N_JOBS=-1`` usa todos los cores al entrenar (puede mostrar avisos en Py3.13+).
    """
    raw = os.environ.get("AURION_RF_N_JOBS", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("AURION_RF_N_JOBS=%r inválido; se usa el default", raw)
    return 1


def _register_sklearn_parallel_warning_filters() -> None:
    """Filtro global sin ``module=`` (los workers a veces no coinciden con sklearn.utils.parallel)."""
    for pat in (
        r".*sklearn\.utils\.parallel\.delayed.*",
        r".*propagate the scikit-learn configuration.*",
    ):
        warnings.filterwarnings("ignore", message=pat, category=UserWarning)


def _coerce_estimator_n_jobs_one(est: Any) -> None:
    """Tras cargar PKL antiguo con ``n_jobs=-1``, fuerza un solo hilo en inferencia."""
    cur: Any = est
    for _ in range(4):
        if cur is None:
            return
        nj = getattr(cur, "n_jobs", None)
        if nj is not None and nj != 1 and hasattr(cur, "set_params"):
            try:
                cur.set_params(n_jobs=1)
            except (ValueError, TypeError):
                pass
        nxt = getattr(cur, "estimator", None) or getattr(cur, "base_estimator", None)
        if nxt is cur:
            return
        cur = nxt


_register_sklearn_parallel_warning_filters()


@contextlib.contextmanager
def _suppress_sklearn_parallel_config_warning() -> Iterator[None]:
    """Refuerzo local (p. ej. calibración / inferencia con RF antiguo ``n_jobs=-1``)."""
    with warnings.catch_warnings():
        for pat in (
            r"sklearn\.utils\.parallel\.delayed.*parallel\.Parallel",
            r"propagate the scikit-learn configuration",
        ):
            warnings.filterwarnings("ignore", message=pat, category=UserWarning)
        yield


_DEFAULT_RF_KWARGS: dict[str, Any] = {
    "n_estimators": 280,
    "max_depth": None,
    "min_samples_leaf": 3,
    "min_samples_split": 6,
    "random_state": 42,
    "n_jobs": _default_rf_n_jobs(),
    "class_weight": "balanced_subsample",
}


class TradingModel:
    """Contenedor de ``RandomForestClassifier`` con fit, predict_proba y PKL."""

    def __init__(
        self,
        artifact_path: Path | None = None,
        *,
        rf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Crea el estimador (no entrenado hasta ``fit``).

        Args:
            artifact_path: Ruta opcional asociada para ``save``/``load``.
            rf_kwargs: Parámetros extra para ``RandomForestClassifier`` (mezclados con defaults).
        """
        self._path = artifact_path
        kw = {**_DEFAULT_RF_KWARGS, **(rf_kwargs or {})}
        self._clf = RandomForestClassifier(**kw)
        self._meta: dict[str, Any] = {}
        logger.debug("TradingModel RandomForest kwargs keys=%s", sorted(kw.keys()))

    @property
    def estimator(self) -> Any:
        """Estimador sklearn (``RandomForestClassifier`` o ``CalibratedClassifierCV`` tras calibrar)."""
        return self._clf

    def attach_estimator(self, clf: Any) -> None:
        """Sustituye el estimador (p. ej. tras calibración Platt/isotónica en holdout)."""
        self._clf = clf
        self._meta["calibrated"] = True

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int] | Sequence[float],
        *,
        sample_weight: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
        feature_schema_version: int | None = None,
    ) -> TradingModel:
        """Entrena el bosque aleatorio.

        Args:
            X: Matriz ``(n_samples, n_features)``.
            y: Etiquetas de clasificación (p. ej. 0=vender, 1=comprar).
            sample_weight: Pesos por muestra (p. ej. magnitud |R| del trade).
            feature_names: Nombres de columnas alineados con ``X`` (trazabilidad en inferencia).
            feature_schema_version: Versión del esquema de features (debe coincidir con live).

        Returns:
            ``self`` encadenable.
        """
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        if x_arr.ndim != 2:
            raise ValueError("X debe ser 2D (n_samples, n_features)")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("X e y deben tener la mismo número de filas")
        sw_arr: np.ndarray | None = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=np.float64)
            if sw_arr.shape[0] != x_arr.shape[0]:
                raise ValueError("sample_weight debe tener una entrada por fila de X")
        with _suppress_sklearn_parallel_config_warning():
            self._clf.fit(x_arr, y_arr, sample_weight=sw_arr)
        self._meta = {
            "n_features_in": int(self._clf.n_features_in_),
            "classes": [int(c) for c in self._clf.classes_],
            "n_estimators": int(self._clf.n_estimators),
        }
        if feature_names is not None:
            self._meta["feature_names"] = [str(n) for n in feature_names]
        if feature_schema_version is not None:
            self._meta["feature_schema_version"] = int(feature_schema_version)
        logger.info(
            "TradingModel.fit: muestras=%s features=%s clases=%s schema=%s",
            x_arr.shape[0],
            x_arr.shape[1],
            self._meta["classes"],
            self._meta.get("feature_schema_version", "—"),
        )
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        """Probabilidades por clase para cada fila de ``X``.

        Args:
            X: Matriz ``(n_samples, n_features)``.

        Returns:
            Lista de filas ``[p_clase0, p_clase1, ...]`` alineada con ``self._clf.classes_``.
        """
        if not hasattr(self._clf, "classes_"):
            raise RuntimeError("El modelo no está entrenado; llama a fit primero.")
        x_arr = np.asarray(X, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        with _suppress_sklearn_parallel_config_warning():
            proba = self._clf.predict_proba(x_arr)
        return cast(list[list[float]], proba.astype(np.float64).tolist())

    def save(self, path: Path | None = None) -> None:
        """Serializa estimador y metadatos en ``path`` (``.pkl``).

        Args:
            path: Ruta de salida; si es ``None`` se usa ``artifact_path`` del constructor.
        """
        target = path or self._path
        if target is None:
            raise ValueError("Indica path en save() o artifact_path en el constructor.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "kind": "RandomForestClassifier",
            "estimator": self._clf,
            "meta": dict(self._meta),
        }
        with target.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("TradingModel.save: %s", target)

    def load(self, path: Path | None = None) -> None:
        """Carga estimador y metadatos desde un ``.pkl`` generado por ``save``.

        Solo cargar artefactos de confianza (pickle puede ejecutar código arbitrario).
        """
        src = path or self._path
        if src is None:
            raise ValueError("Indica path en load() o artifact_path en el constructor.")
        with src.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "estimator" not in payload:
            raise ValueError(f"Archivo PKL inválido o formato desconocido: {src}")
        self._clf = payload["estimator"]
        self._meta = dict(payload.get("meta", {}))
        self._path = src
        _coerce_estimator_n_jobs_one(self._clf)
        logger.info("TradingModel.load: %s clases=%s", src, self._meta.get("classes"))

    def metadata(self) -> dict[str, Any]:
        """Metadatos del último ``fit`` o del artefacto cargado."""
        return dict(self._meta)

    def n_features_expected(self) -> int | None:
        """Número de features del modelo entrenado, o ``None`` si aún no aplica."""
        raw = self._meta.get("n_features_in")
        if raw is None:
            if hasattr(self._clf, "n_features_in_"):
                return int(getattr(self._clf, "n_features_in_"))
            return None
        return int(raw)

    def feature_schema_version(self) -> int | None:
        """Versión del esquema guardada al entrenar, o ``None`` en modelos antiguos."""
        raw = self._meta.get("feature_schema_version")
        return int(raw) if raw is not None else None


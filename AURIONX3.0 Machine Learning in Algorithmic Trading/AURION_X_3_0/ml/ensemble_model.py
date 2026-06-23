"""Ensemble supervisado: RF + LR + opcional XGB/LGBM + meta votación."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Sequence, cast

# Silencia UserWarning de sklearn/LGBM en inferencia (feature names mismatch
# es esperado: entrenamos con DataFrame, predecimos con ndarray ya alineado).
for _pat in (
    r".*does not have valid feature names.*",
    r".*X has feature names.*",
):
    warnings.filterwarnings("ignore", message=_pat, category=UserWarning)

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analytics.logger import get_logger
from .model import _DEFAULT_RF_KWARGS, _suppress_sklearn_parallel_config_warning

logger = get_logger(__name__)


def _optional_xgb() -> type[Any] | None:
    try:
        from xgboost import XGBClassifier  # type: ignore[import-untyped]

        return XGBClassifier
    except ImportError:
        return None


def _optional_lgbm() -> type[Any] | None:
    try:
        from lightgbm import LGBMClassifier  # type: ignore[import-untyped]

        return LGBMClassifier
    except ImportError:
        return None


class EnsembleTradingModel:
    """Voting ensemble con persistencia PKL compatible con ``TradingModel``."""

    def __init__(self, artifact_path: Path | None = None) -> None:
        self._path = artifact_path
        self._clf: Any = None
        self._meta: dict[str, Any] = {}

    @property
    def estimator(self) -> Any:
        if self._clf is None:
            raise RuntimeError("Ensemble sin entrenar")
        return self._clf

    def attach_estimator(self, clf: Any) -> None:
        """Sustituye el clasificador (p.ej. tras calibracion externa).

        AURION X 4.3: necesario para compatibilidad con ``Trainer._apply_calibration``.
        El voting soft ya proporciona probabilidades calibradas por construccion
        (promedio ponderado de probs), por lo que se acepta el override pero
        solo se aplica si el caller insiste.
        """
        self._clf = clf
        self._meta["calibrated"] = True

    def _build_estimators(
        self,
        hyperparams: dict[str, Any] | None,
        *,
        y_balance: tuple[int, int] | None = None,
    ) -> list[tuple[str, Any]]:
        """Construye estimadores con scaling/balance correctos.

        AURION X 4.3 (FIX bug ML audit): LR sin StandardScaler colapsa al
        feature de mayor magnitud (ATR domina); XGB sin scale_pos_weight ignora
        clases desbalanceadas. Ambos fixes son indispensables para que LR/XGB
        aporten edge real al voting ensemble (antes eran ruido).
        """
        hp = hyperparams or {}
        rf_kw = {
            **_DEFAULT_RF_KWARGS,
            "max_depth": int(hp.get("rf_max_depth", 12)),
            "min_samples_leaf": int(hp.get("rf_min_samples_leaf", 4)),
            **{k: v for k, v in hp.items() if k.startswith("rf_") or k in _DEFAULT_RF_KWARGS},
        }
        # AURION X 4.3 FIX: LR wrapped in pipeline con StandardScaler + max_iter alto.
        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )),
        ])
        estimators: list[tuple[str, Any]] = [
            ("rf", RandomForestClassifier(**rf_kw)),
            ("lr", lr_pipeline),
        ]
        # AURION X 4.3 FIX: XGB scale_pos_weight = neg/pos para balance de clases.
        scale_pos_weight = 1.0
        if y_balance is not None and y_balance[1] > 0:
            scale_pos_weight = float(y_balance[0]) / float(y_balance[1])
        XGB = _optional_xgb()
        if XGB is not None:
            estimators.append(
                (
                    "xgb",
                    XGB(
                        n_estimators=int(hp.get("xgb_n_estimators", 120)),
                        max_depth=int(hp.get("xgb_max_depth", 5)),
                        learning_rate=float(hp.get("xgb_learning_rate", 0.08)),
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            )
        LGBM = _optional_lgbm()
        if LGBM is not None:
            estimators.append(
                (
                    "lgbm",
                    LGBM(
                        n_estimators=int(hp.get("lgbm_n_estimators", 120)),
                        max_depth=int(hp.get("lgbm_max_depth", 6)),
                        learning_rate=float(hp.get("lgbm_learning_rate", 0.06)),
                        subsample=0.85,
                        colsample_bytree=0.85,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            )
        return estimators

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int] | Sequence[float],
        *,
        sample_weight: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
        feature_schema_version: int | None = None,
        hyperparams: dict[str, Any] | None = None,
    ) -> EnsembleTradingModel:
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        if x_arr.ndim != 2 or y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("X/y inválidos para ensemble")
        if len(np.unique(y_arr)) < 2:
            raise ValueError("Ensemble requiere al menos dos clases en y")

        # AURION X 4.3 FIX: pasar balance de clases para scale_pos_weight de XGB.
        # Asume y binario {0,1} o {-1,1}. Para binario 0/1: pos=1, neg=0.
        unique, counts = np.unique(y_arr, return_counts=True)
        if len(unique) == 2:
            pos_label = max(unique)
            neg_count = int(counts[unique != pos_label][0])
            pos_count = int(counts[unique == pos_label][0])
            y_balance: tuple[int, int] | None = (neg_count, pos_count)
        else:
            y_balance = None
        estimators = self._build_estimators(hyperparams, y_balance=y_balance)
        weights = [2.0] + [1.0] * (len(estimators) - 1)
        vote = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
        # AURION X 4.3 FIX bug ML audit #5: NO propagar sample_weight al ensemble
        # cuando algun estimador es Pipeline (sklearn.Pipeline.fit no acepta
        # sample_weight sin prefijo step). El balance de clases ya se maneja en
        # cada estimador: RF (class_weight=balanced_subsample), LR (class_weight=
        # balanced via pipeline), XGB (scale_pos_weight calculado), LGBM
        # (class_weight=balanced).
        if sample_weight is not None:
            logger.debug(
                "EnsembleTradingModel.fit: sample_weight ignorado (Pipeline LR no acepta sin prefijo); "
                "balance de clases delegado a cada estimador",
            )
        with _suppress_sklearn_parallel_config_warning():
            vote.fit(x_arr, y_arr)
        self._clf = vote
        self._meta = {
            "n_features_in": int(x_arr.shape[1]),
            "classes": [int(c) for c in vote.classes_],
            "estimators": [name for name, _ in estimators],
            "kind": "ensemble_voting",
        }
        if feature_names is not None:
            self._meta["feature_names"] = [str(n) for n in feature_names]
        if feature_schema_version is not None:
            self._meta["feature_schema_version"] = int(feature_schema_version)
        logger.info(
            "EnsembleTradingModel.fit: muestras=%s features=%s estimators=%s",
            x_arr.shape[0],
            x_arr.shape[1],
            self._meta["estimators"],
        )
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        x_arr = np.asarray(X, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        with _suppress_sklearn_parallel_config_warning():
            proba = self._clf.predict_proba(x_arr)
        return cast(list[list[float]], proba.astype(np.float64).tolist())

    def save(self, path: Path | None = None) -> None:
        target = path or self._path
        if target is None:
            raise ValueError("Indica path en save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "kind": "ensemble_voting",
            "estimator": self._clf,
            "meta": dict(self._meta),
        }
        with target.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("EnsembleTradingModel.save: %s", target)

    def load(self, path: Path | None = None) -> None:
        src = path or self._path
        if src is None:
            raise ValueError("Indica path en load()")
        with src.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "estimator" not in payload:
            raise ValueError(f"PKL ensemble inválido: {src}")
        self._clf = payload["estimator"]
        self._meta = dict(payload.get("meta", {}))
        self._path = src
        logger.info("EnsembleTradingModel.load: %s estimators=%s", src, self._meta.get("estimators"))

    def metadata(self) -> dict[str, Any]:
        return dict(self._meta)

    def n_features_expected(self) -> int | None:
        raw = self._meta.get("n_features_in")
        if raw is not None:
            return int(raw)
        if self._clf is not None and hasattr(self._clf, "n_features_in_"):
            return int(self._clf.n_features_in_)
        return None

    def feature_schema_version(self) -> int | None:
        raw = self._meta.get("feature_schema_version")
        return int(raw) if raw is not None else None

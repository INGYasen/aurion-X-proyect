"""Entrenamiento, validación temporal y exportación de modelos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

try:
    from sklearn.frozen import FrozenEstimator as _FrozenEstimator
except ImportError:  # sklearn < 1.6
    _FrozenEstimator = None  # type: ignore[misc, assignment]
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from analytics.trade_repository import TradeRepository
from analytics.logger import get_logger

from .feature_engineering import (
    FEATURE_LIST_LEGACY_LEN_WITH_SPREAD,
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    FEATURE_VECTOR_LEN,
)
from .model import TradingModel, _suppress_sklearn_parallel_config_warning
from .quant_features import (
    FEATURE_ORDER_V2,
    FEATURE_ORDER_V3,
    FEATURE_SCHEMA_VERSION_V2,
    FEATURE_SCHEMA_VERSION_V3,
    FEATURE_VECTOR_LEN_V2,
    FEATURE_VECTOR_LEN_V3,
    MACRO_FEATURES_ORDER,
    vector_from_features_json,
)
from .ensemble_model import EnsembleTradingModel

logger = get_logger(__name__)

_RF_PARAM_KEYS = frozenset(RandomForestClassifier().get_params(deep=False).keys())


class Trainer:
    """Entrena ``TradingModel`` (RandomForest) y evalúa con métricas sklearn."""

    def __init__(
        self,
        output_dir: Path | None = None,
        *,
        repository: TradeRepository | None = None,
        training_weight_by_r: bool = True,
        walk_forward_embargo_ratio: float = 0.02,
        label_mode: str = "win_loss",
        min_r_for_positive: float = 0.25,
        calibration_method: str | None = "sigmoid",
        calibration_holdout_ratio: float = 0.15,
        dataset_context: str | None = None,
        feature_schema: str = "v1",
        cv_strategy: str = "walk_forward",
        cv_n_splits: int = 5,
        use_ensemble: bool = False,
    ) -> None:
        """Fija directorio de artefactos generados.

        Args:
            output_dir: Carpeta donde guardar modelos y métricas.
            label_mode: ``win_loss`` (profit>0) o ``r_quality`` (R >= min_r_for_positive).
            calibration_method: ``sigmoid``, ``isotonic`` o ``None`` para desactivar.
            calibration_holdout_ratio: Fracción temporal final reservada a calibración (train_from_repository).
            cv_strategy: ``walk_forward`` (3.0, default) o ``purged_kfold`` (4.0,
                López de Prado 2018). El uso real se delega en helpers que
                consumen ``ml/cv_strategies.py``.
            cv_n_splits: número de folds cuando ``cv_strategy="purged_kfold"``.
        """
        self._output_dir = output_dir or Path("artifacts")
        self._repository = repository or TradeRepository()
        self._training_weight_by_r = bool(training_weight_by_r)
        self._walk_forward_embargo_ratio = max(0.0, min(0.08, float(walk_forward_embargo_ratio)))
        lm = str(label_mode or "win_loss").strip().lower()
        if lm not in ("win_loss", "r_quality"):
            logger.warning("label_mode %r inválido; se usa win_loss", label_mode)
            lm = "win_loss"
        self._label_mode = lm
        self._min_r_for_positive = float(min_r_for_positive)
        cm = calibration_method
        if cm is None or str(cm).strip().lower() in ("", "none", "false", "0", "off"):
            self._calibration_method: str | None = None
        else:
            c = str(cm).strip().lower()
            self._calibration_method = c if c in ("sigmoid", "isotonic") else None
            if self._calibration_method is None:
                logger.warning("calibration_method %r inválido; sin calibración", calibration_method)
        self._calibration_holdout_ratio = max(0.08, min(0.35, float(calibration_holdout_ratio)))
        dc = (dataset_context or "all").strip().lower()
        if dc not in ("all", "range", "momentum"):
            logger.warning("dataset_context %r inválido; se usa 'all'", dataset_context)
            dc = "all"
        self._dataset_context: str = dc
        fs = str(feature_schema or "v1").strip().lower()
        self._feature_schema = fs if fs in ("v1", "v2", "v3") else "v1"
        if self._feature_schema == "v3":
            logger.info("Trainer: schema v3 (30 features: 25 quant + 5 macro DXY/US10Y/VIX)")
        cv = (cv_strategy or "walk_forward").strip().lower()
        if cv not in ("walk_forward", "purged_kfold"):
            logger.warning("cv_strategy %r inválido; se usa walk_forward.", cv_strategy)
            cv = "walk_forward"
        self._cv_strategy = cv
        self._cv_n_splits = max(2, int(cv_n_splits))
        # AURION X 4.3: ensemble flag (RF+LR+XGB+LGBM voting soft).
        self._use_ensemble = bool(use_ensemble)
        logger.debug(
            "Trainer output_dir=%s dataset_context=%s schema=%s cv_strategy=%s use_ensemble=%s",
            output_dir,
            self._dataset_context,
            self._feature_schema,
            self._cv_strategy,
            self._use_ensemble,
        )

    @property
    def cv_strategy(self) -> str:
        """Estrategia de validación cruzada activa."""
        return self._cv_strategy

    def iter_cv_splits(self, n_samples: int):
        """Itera splits ``(train_idx, test_idx)`` según ``cv_strategy``.

        Punto de entrada centralizado para que el resto del pipeline no necesite
        importar ``cv_strategies`` directamente.
        """
        from .cv_strategies import iter_cv_splits as _iter  # lazy

        yield from _iter(
            n_samples,
            strategy=self._cv_strategy,
            n_splits=self._cv_n_splits,
            embargo_pct=self._walk_forward_embargo_ratio,
        )

    def _y_label_from_row(self, row: dict[str, Any]) -> int | None:
        if self._label_mode == "win_loss":
            try:
                profit = float(row.get("profit") if row.get("profit") is not None else 0.0)
            except (TypeError, ValueError):
                return None
            return 1 if profit > 0.0 else 0
        raw = row.get("r_multiple")
        if raw is None:
            return None
        try:
            r = float(raw)
        except (TypeError, ValueError):
            return None
        return 1 if r >= self._min_r_for_positive else 0

    def _wrap_prefit_estimator(self, base: Any) -> Any:
        """Envuelve un estimador ya entrenado para calibración (sklearn >= 1.6 usa ``FrozenEstimator``)."""
        if _FrozenEstimator is not None:
            return _FrozenEstimator(base)
        return base

    def _calibrated_classifier(self, base: Any, *, method: str) -> CalibratedClassifierCV:
        """``CalibratedClassifierCV`` compatible con sklearn 1.6+ (sin ``cv='prefit'``)."""
        if _FrozenEstimator is not None:
            return CalibratedClassifierCV(self._wrap_prefit_estimator(base), method=method)
        return CalibratedClassifierCV(base, method=method, cv="prefit")

    def _calibrate_if_needed(self, model: TradingModel, X_cal: list[list[float]], y_cal: list[int]) -> None:
        if self._calibration_method is None or len(X_cal) < 15:
            return
        # AURION X 4.4 (10-jun-2026): post-mortem revelo overconfidence severa
        # en el ensemble (conf>0.70 -> WR 0%, conf 0.50-0.55 -> WR 83%).
        # voting soft NO produce probs bien calibradas con dataset desbalanceado
        # (~45% WR). Sigmoid sigue redundante (saturacion), pero ISOTONIC sobre
        # holdout SI corrige la sobre/sub-confianza del ensemble.
        if isinstance(model, EnsembleTradingModel) and self._calibration_method == "sigmoid":
            logger.info(
                "Calibracion sigmoid omitida en ensemble (redundante con voting soft)",
            )
            return
        y_arr = np.asarray(y_cal, dtype=np.int64)
        if len(np.unique(y_arr)) < 2:
            logger.warning(
                "Calibración omitida: holdout con una sola clase (n=%s, clases=%s)",
                len(y_cal),
                np.unique(y_arr).tolist(),
            )
            return
        method = self._calibration_method
        base = model.estimator
        cal = self._calibrated_classifier(base, method=method)
        try:
            with _suppress_sklearn_parallel_config_warning():
                cal.fit(np.asarray(X_cal, dtype=np.float64), y_arr)
        except (ValueError, IndexError) as exc:
            logger.warning("Calibración %s omitida: %s", method, exc)
            return
        model.attach_estimator(cal)
        logger.info("Calibración %s aplicada (%s muestras holdout)", method, len(X_cal))

    @staticmethod
    def _as_labels(y: list[float]) -> np.ndarray:
        return np.asarray([int(v) for v in y], dtype=np.int64)

    @staticmethod
    def _rf_kwargs(hyperparams: dict[str, Any] | None) -> dict[str, Any]:
        if not hyperparams:
            return {}
        return {k: v for k, v in hyperparams.items() if k in _RF_PARAM_KEYS}

    def _features_from_row(self, row: dict[str, Any]) -> list[float] | None:
        raw = row.get("features_json")
        schema = self._feature_schema
        if schema == "v3":
            # AURION X 4.3: v3 = v2 (25) + 5 macro features. Solo aceptamos rows
            # explicitamente enriquecidos (features_json["macro_v3"]=True por
            # tools/fetch_and_backfill_macro.py).
            if not isinstance(raw, dict):
                return None
            if not raw.get("macro_v3"):
                return None
            from .quant_features import QUANT_EXTENDED_ORDER
            quant_keys = sum(1 for k in QUANT_EXTENDED_ORDER if k in raw)
            macro_keys = sum(1 for k in MACRO_FEATURES_ORDER if k in raw)
            if quant_keys < len(QUANT_EXTENDED_ORDER) // 2 or macro_keys < len(MACRO_FEATURES_ORDER):
                return None
            vec = vector_from_features_json(raw, schema="v3")
            if vec is not None and len(vec) == FEATURE_VECTOR_LEN_V3:
                return vec
            return None
        if schema == "v2":
            # AURION X 4.3 FIX bug ML audit #4: NO hacer padding cero de v1->v2.
            # Padear con ceros las 15 features SMC/quant hacia v2 enseñaba al
            # modelo que esas features valen "siempre 0", destruyendo su senal.
            # Solo aceptamos rows con vector v2 completo y real (no padded).
            if not isinstance(raw, dict):
                # Listas crudas: solo válidas si ya tienen 25 elementos (no se
                # puede saber si los 15 finales son reales o zeros artificiales).
                if isinstance(raw, list):
                    try:
                        vals = [float(v) for v in raw]
                    except (TypeError, ValueError):
                        return None
                    if len(vals) == FEATURE_VECTOR_LEN_V2:
                        return vals
                return None
            # Detectar si el dict trae las 15 quant features explicitas.
            from .quant_features import QUANT_EXTENDED_ORDER
            quant_keys_present = sum(1 for k in QUANT_EXTENDED_ORDER if k in raw)
            if quant_keys_present < len(QUANT_EXTENDED_ORDER) // 2:
                # Row v1 puro: descartar para entrenamiento v2 (no contaminar).
                return None
            vec = vector_from_features_json(raw, schema="v2")
            if vec is not None and len(vec) == FEATURE_VECTOR_LEN_V2:
                return vec
            return None
        if isinstance(raw, dict):
            try:
                vec = [float(raw.get(k, 0.0) or 0.0) for k in FEATURE_ORDER]
            except (TypeError, ValueError):
                return None
            return vec if len(vec) == FEATURE_VECTOR_LEN else None
        if isinstance(raw, list):
            try:
                vals = [float(v) for v in raw]
            except (TypeError, ValueError):
                return None
            if len(vals) == FEATURE_LIST_LEGACY_LEN_WITH_SPREAD:
                vals = vals[:5] + vals[6:]
            if len(vals) >= FEATURE_VECTOR_LEN:
                vec = vals[:FEATURE_VECTOR_LEN]
            else:
                vec = vals + [0.0] * (FEATURE_VECTOR_LEN - len(vals))
            return vec if len(vec) == FEATURE_VECTOR_LEN else None
        return None

    @staticmethod
    def _sample_weight_from_row(row: dict[str, Any], *, enabled: bool) -> float:
        """Mayor peso a trades con |R| alto (aprende más de resultados extremos)."""
        if not enabled:
            return 1.0
        raw = row.get("r_multiple")
        if raw is None:
            return 1.0
        try:
            r = abs(float(raw))
        except (TypeError, ValueError):
            return 1.0
        return float(min(3.0, 1.0 + r))

    @classmethod
    def _close_time_sort_key(cls, row: dict[str, Any]) -> str:
        raw = row.get("close_time")
        if raw is None:
            return ""
        if isinstance(raw, datetime):
            return raw.isoformat()
        return str(raw)

    def _row_matches_dataset_context(self, row: dict[str, Any]) -> bool:
        """Filtra por contexto RANGE vs MOMENTUM usando ``features_json`` persistido."""
        if self._dataset_context == "all":
            return True
        raw = row.get("features_json")
        setup = ""
        ctx = ""
        if isinstance(raw, dict):
            setup = str(raw.get("setup_type", "") or raw.get("strategy_detail", "")).upper()
            ctx = str(raw.get("ml_dataset_context", "") or "").strip().lower()
        is_range_like = ctx == "range" or (
            "RANGE_FADE" in setup
            or "RANGE_" in setup
            or "FADE" in setup
            or "INTRADAY_RANGE" in setup
        )
        if self._dataset_context == "range":
            return is_range_like
        if self._dataset_context == "momentum":
            return not is_range_like
        return True

    def _xy_timed_rows(self, rows: Iterable[dict[str, Any]], *, weight_by_r: bool) -> list[tuple[str, list[float], int, float]]:
        """Filas válidas con tiempo de cierre para orden temporal walk-forward."""
        out: list[tuple[str, list[float], int, float]] = []
        for row in rows:
            if not self._row_matches_dataset_context(row):
                continue
            x = self._features_from_row(row)
            if x is None:
                continue
            y = self._y_label_from_row(row)
            if y is None:
                continue
            ct = self._close_time_sort_key(row)
            if not ct:
                continue
            w = self._sample_weight_from_row(row, enabled=weight_by_r)
            out.append((ct, x, y, w))
        return out

    def _xy_from_dataset(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        weight_by_r: bool,
    ) -> tuple[list[list[float]], list[int], list[float]]:
        X: list[list[float]] = []
        y: list[int] = []
        w: list[float] = []
        skipped = 0
        for row in rows:
            if not self._row_matches_dataset_context(row):
                skipped += 1
                continue
            x = self._features_from_row(row)
            if x is None:
                skipped += 1
                continue
            yv = self._y_label_from_row(row)
            if yv is None:
                skipped += 1
                continue
            X.append(x)
            y.append(int(yv))
            w.append(self._sample_weight_from_row(row, enabled=weight_by_r))
        if skipped:
            logger.warning("Trainer: filas ignoradas por datos inválidos=%s", skipped)
        return X, y, w

    def train(
        self,
        X: list[list[float]],
        y: list[float],
        *,
        sample_weight: list[float] | None = None,
        hyperparams: dict[str, Any] | None = None,
    ) -> TradingModel:
        """Entrena un ``TradingModel`` con matriz tabular ``X`` y etiquetas ``y``.

        Args:
            X: Matriz ``(n_samples, n_features)``.
            y: Etiquetas de clasificación (0/1 o -1/1; se convierten a ``int``).
            sample_weight: Pesos por muestra (opcional).
            hyperparams: Parámetros válidos de ``RandomForestClassifier``.

        Returns:
            Modelo entrenado listo para ``predict_proba`` / ``save``.
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("X e y deben ser no vacíos y con la misma longitud")
        if sample_weight is not None and len(sample_weight) != len(X):
            raise ValueError("sample_weight debe alinear con X")
        if self._feature_schema == "v3":
            fnames = FEATURE_ORDER_V3
            fver = FEATURE_SCHEMA_VERSION_V3
        elif self._feature_schema == "v2":
            fnames = FEATURE_ORDER_V2
            fver = FEATURE_SCHEMA_VERSION_V2
        else:
            fnames = FEATURE_ORDER
            fver = FEATURE_SCHEMA_VERSION
        if self._use_ensemble:
            # AURION X 4.3: ensemble RF + LR + XGB(if) + LGBM(if) con voting soft.
            ensemble = EnsembleTradingModel()
            ensemble.fit(
                X,
                self._as_labels(y),
                sample_weight=sample_weight,
                feature_names=fnames,
                feature_schema_version=fver,
                hyperparams=hyperparams,
            )
            logger.info(
                "Trainer.train (ENSEMBLE): muestras=%s estimators=%s",
                len(X),
                ensemble.metadata().get("estimators"),
            )
            return ensemble  # type: ignore[return-value]
        rf_kw = self._rf_kwargs(hyperparams)
        model = TradingModel(rf_kwargs=rf_kw)
        model.fit(
            X,
            y,
            sample_weight=sample_weight,
            feature_names=fnames,
            feature_schema_version=fver,
        )
        logger.info("Trainer.train: muestras=%s rf_params=%s", len(X), sorted(rf_kw))
        return model

    def evaluate(self, model: TradingModel, X: list[list[float]], y: list[float]) -> dict[str, float]:
        """Métricas de clasificación en el conjunto dado (accuracy, F1, ROC-AUC si binario)."""
        if not X or len(X) != len(y):
            raise ValueError("X e y inválidos para evaluate")
        y_true = self._as_labels(y)
        y_hat = model.estimator.predict(np.asarray(X, dtype=np.float64))
        out: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_hat)),
        }
        classes = np.unique(y_true)
        if len(classes) >= 2:
            average = "binary" if len(classes) == 2 else "weighted"
            out["f1"] = float(f1_score(y_true, y_hat, average=average, zero_division=0))
            if len(classes) == 2:
                try:
                    proba = model.estimator.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]
                    out["roc_auc"] = float(roc_auc_score(y_true, proba))
                except ValueError:
                    out["roc_auc"] = float("nan")
        return out

    def train_from_repository(
        self,
        *,
        limit: int = 5000,
        symbol: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        model_filename: str = "model.pkl",
        dataset_context: str | None = None,
    ) -> tuple[TradingModel, dict[str, float]]:
        """Entrena usando dataset desde ``TradeRepository`` y guarda ``model.pkl``.

        Dataset:
            - ``X`` = ``features_json``
            - ``y`` = según ``label_mode`` (win_loss o umbral R).
        """
        prev_ctx = self._dataset_context
        ctx_changed = False
        if dataset_context is not None:
            dc = str(dataset_context).strip().lower()
            if dc in ("all", "range", "momentum"):
                self._dataset_context = dc
                ctx_changed = True
        try:
            rows = self._repository.get_training_dataset(limit=limit, symbol=symbol)
            rows_sorted = sorted(
                (r for r in rows if r.get("close_time")),
                key=lambda r: self._close_time_sort_key(r),
            )
            X, y, sw = self._xy_from_dataset(rows_sorted, weight_by_r=self._training_weight_by_r)
            if len(X) < 30:
                raise ValueError("Dataset insuficiente para entrenar (se requieren al menos 30 filas válidas)")

            # AURION X 4.3 FIX bug ML audit #2: evaluar OUT-OF-SAMPLE (holdout temporal
            # ultimo 20%), no IN-SAMPLE como antes. Las metricas reportadas se usan
            # por OOSValidator y DailyRetrainer para decidir promociones; IN-SAMPLE
            # las inflaba a ~100% engañando al sistema.
            if len(X) >= 60:
                n_oos = max(15, int(len(X) * 0.20))
                X_train_full = X[:-n_oos]
                y_train_full = y[:-n_oos]
                sw_train_full = sw[:-n_oos]
                X_oos = X[-n_oos:]
                y_oos = y[-n_oos:]
            else:
                X_train_full = X
                y_train_full = y
                sw_train_full = sw
                X_oos = X
                y_oos = y

            if self._calibration_method and len(X_train_full) >= 50:
                # Calibration holdout interno: ultimo X% del set de entrenamiento.
                n_cal = max(15, int(len(X_train_full) * self._calibration_holdout_ratio))
                model = self.train(
                    X_train_full[:-n_cal],
                    [float(v) for v in y_train_full[:-n_cal]],
                    sample_weight=sw_train_full[:-n_cal],
                    hyperparams=hyperparams,
                )
                self._calibrate_if_needed(model, X_train_full[-n_cal:], y_train_full[-n_cal:])
            else:
                model = self.train(
                    X_train_full,
                    [float(v) for v in y_train_full],
                    sample_weight=sw_train_full,
                    hyperparams=hyperparams,
                )
            # Metricas OOS (no IS): solo el holdout final.
            metrics = self.evaluate(model, X_oos, [float(v) for v in y_oos])

            out_path = self._output_dir / model_filename
            model.save(out_path)
            logger.info(
                "train_from_repository: entrenado con %s filas, OOS holdout=%s, guardado en %s",
                len(X_train_full),
                len(X_oos),
                out_path,
            )
            return model, metrics
        finally:
            if ctx_changed:
                self._dataset_context = prev_ctx

    def retrain_manual(
        self,
        *,
        limit: int = 5000,
        symbol: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        model_filename: str = "model.pkl",
        dataset_context: str | None = None,
    ) -> tuple[TradingModel, dict[str, float]]:
        """Dispara reentrenamiento manual (one-shot) desde repositorio."""
        return self.train_from_repository(
            limit=limit,
            symbol=symbol,
            hyperparams=hyperparams,
            model_filename=model_filename,
            dataset_context=dataset_context,
        )

    def retrain_in_batches(
        self,
        *,
        total_limit: int = 10000,
        batch_size: int = 2000,
        symbol: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        model_filename: str = "model.pkl",
        dataset_context: str | None = None,
    ) -> tuple[TradingModel, dict[str, float]]:
        """Reentrena por lotes del histórico (manual por lotes).

        Estrategia:
            - Descarga dataset total (hasta ``total_limit``)
            - Lo procesa en ventanas de ``batch_size``
            - Entrena un modelo nuevo en cada lote
            - Conserva y guarda el mejor por ``accuracy``
        """
        prev_ctx = self._dataset_context
        ctx_changed = False
        if dataset_context is not None:
            dc = str(dataset_context).strip().lower()
            if dc in ("all", "range", "momentum"):
                self._dataset_context = dc
                ctx_changed = True
        try:
            rows = self._repository.get_training_dataset(limit=total_limit, symbol=symbol)
            # AURION X 4.3 FIX bug ML audit #3: el repo retorna ORDER BY close_time DESC,
            # pero los batches y la calibracion final necesitan orden ASC para que:
            #   1) cada batch contenga trades temporalmente contiguos (no random).
            #   2) la calibracion final use los trades MAS RECIENTES (no los mas antiguos).
            # Sin este sort, calibracion Platt aplicada sobre datos antiguos producia
            # leakage temporal severo.
            rows_sorted = sorted(
                (r for r in rows if r.get("close_time")),
                key=lambda r: self._close_time_sort_key(r),
            )
            X_all, y_all, sw_all = self._xy_from_dataset(rows_sorted, weight_by_r=self._training_weight_by_r)
            if len(X_all) < 30:
                raise ValueError("Dataset insuficiente para reentrenamiento por lotes")

            def _rank(met: dict[str, float]) -> float:
                auc = met.get("roc_auc")
                if auc is not None and auc == auc:
                    return float(auc)
                return float(met.get("f1", 0.0)) * 0.6 + float(met.get("accuracy", 0.0)) * 0.4

            best_model: TradingModel | None = None
            best_rank = -1.0
            best_metrics: dict[str, float] = {}

            for i in range(0, len(X_all), batch_size):
                xb = X_all[i : i + batch_size]
                yb = y_all[i : i + batch_size]
                wb = sw_all[i : i + batch_size]
                if len(xb) < 30:
                    continue
                # AURION X 4.3 FIX (paralelo a bug #2): evaluar OOS, no IS.
                # Splits internos del batch para holdout temporal.
                if len(xb) >= 50:
                    n_oos_b = max(10, int(len(xb) * 0.20))
                    xb_tr, yb_tr, wb_tr = xb[:-n_oos_b], yb[:-n_oos_b], wb[:-n_oos_b]
                    xb_oos, yb_oos = xb[-n_oos_b:], yb[-n_oos_b:]
                else:
                    xb_tr, yb_tr, wb_tr = xb, yb, wb
                    xb_oos, yb_oos = xb, yb
                model = self.train(xb_tr, [float(v) for v in yb_tr], sample_weight=wb_tr, hyperparams=hyperparams)
                metrics = self.evaluate(model, xb_oos, [float(v) for v in yb_oos])
                rnk = _rank(metrics)
                if rnk >= best_rank:
                    best_rank = rnk
                    best_model = model
                    best_metrics = dict(metrics)

            if best_model is None:
                raise ValueError("No hubo lotes válidos para entrenar")

            if self._calibration_method and len(X_all) >= 50:
                n_cal = max(15, int(len(X_all) * self._calibration_holdout_ratio))
                # Ahora X_all[-n_cal:] son los MAS RECIENTES (sort ASC arriba).
                self._calibrate_if_needed(best_model, X_all[-n_cal:], y_all[-n_cal:])

            out_path = self._output_dir / model_filename
            best_model.save(out_path)
            logger.info(
                "retrain_in_batches: mejor rank=%.4f (roc_auc/f1 mix) accuracy=%.4f -> %s",
                best_rank,
                best_metrics.get("accuracy", float("nan")),
                out_path,
            )
            return best_model, best_metrics
        finally:
            if ctx_changed:
                self._dataset_context = prev_ctx

    def train_walk_forward_from_repository(
        self,
        *,
        limit: int = 5000,
        symbol: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        model_filename: str = "model.pkl",
        train_ratio: float = 0.65,
        val_ratio: float = 0.15,
        min_samples: int = 80,
        dataset_context: str | None = None,
    ) -> tuple[TradingModel, dict[str, float]]:
        """Entrena con split temporal (train → val para chequeo, modelo final train+val; test OOS).

        Orden: ``close_time`` ascendente. El modelo guardado se ajusta con train+val; las métricas
        ``oos_*`` se calculan solo sobre el tramo test (out-of-sample temporal).
        """
        prev_ctx = self._dataset_context
        ctx_changed = False
        if dataset_context is not None:
            dc = str(dataset_context).strip().lower()
            if dc in ("all", "range", "momentum"):
                self._dataset_context = dc
                ctx_changed = True
        try:
            rows = self._repository.get_training_dataset(limit=limit, symbol=symbol)
            timed = self._xy_timed_rows(rows, weight_by_r=self._training_weight_by_r)
            timed.sort(key=lambda t: t[0])
            if len(timed) < min_samples:
                raise ValueError(
                    f"Walk-forward: dataset insuficiente (n={len(timed)}, mínimo {min_samples})",
                )
            tr = max(0.0, min(0.90, float(train_ratio)))
            vr = max(0.05, min(0.30, float(val_ratio)))
            if tr + vr >= 0.95:
                tr, vr = 0.65, 0.15
            n = len(timed)
            embargo = max(2, int(n * self._walk_forward_embargo_ratio))
            i_tr = max(30, int(n * tr))
            i_va = max(i_tr + embargo + 5, int(n * (tr + vr)))
            if i_va >= n - 5:
                i_va = n - max(5, n // 10)
            if i_va <= i_tr + embargo:
                i_va = min(n - 5, i_tr + embargo + max(5, n // 20))
            X_tr = [t[1] for t in timed[:i_tr]]
            y_tr = [t[2] for t in timed[:i_tr]]
            w_tr = [t[3] for t in timed[:i_tr]]
            i_va_start = i_tr + embargo
            if i_va_start >= i_va:
                raise ValueError(
                    "Walk-forward: sin muestras de validación tras embargo; reduce ratios o embargo",
                )
            X_va = [t[1] for t in timed[i_va_start:i_va]]
            y_va = [t[2] for t in timed[i_va_start:i_va]]
            w_va = [t[3] for t in timed[i_va_start:i_va]]
            X_te = [t[1] for t in timed[i_va:]]
            y_te = [t[2] for t in timed[i_va:]]
            if len(X_te) < 5:
                raise ValueError("Walk-forward: tramo test demasiado pequeño")

            probe = self.train(
                X_tr,
                [float(v) for v in y_tr],
                sample_weight=w_tr,
                hyperparams=hyperparams,
            )
            metrics_val = self.evaluate(probe, X_va, [float(v) for v in y_va])
            X_fit = [t[1] for t in timed[:i_va]]
            y_fit = [t[2] for t in timed[:i_va]]
            w_fit = [t[3] for t in timed[:i_va]]
            final = self.train(
                X_fit,
                [float(v) for v in y_fit],
                sample_weight=w_fit,
                hyperparams=hyperparams,
            )
            self._calibrate_if_needed(final, X_va, y_va)
            metrics_oos = self.evaluate(final, X_te, [float(v) for v in y_te])

            out_metrics: dict[str, float] = {
                "walk_forward": 1.0,
                "n_total": float(n),
                "n_train": float(len(X_tr)),
                "n_val": float(len(X_va)),
                "n_test": float(len(X_te)),
                "val_accuracy": float(metrics_val.get("accuracy", 0.0)),
                "val_f1": float(metrics_val.get("f1", 0.0)),
                "oos_accuracy": float(metrics_oos.get("accuracy", 0.0)),
                "oos_f1": float(metrics_oos.get("f1", 0.0)),
            }
            if "roc_auc" in metrics_oos and metrics_oos["roc_auc"] == metrics_oos["roc_auc"]:
                out_metrics["oos_roc_auc"] = float(metrics_oos["roc_auc"])

            out_path = self._output_dir / model_filename
            final.save(out_path)
            logger.info(
                "train_walk_forward: n=%s oos_accuracy=%.4f val_accuracy=%.4f -> %s",
                n,
                out_metrics["oos_accuracy"],
                out_metrics["val_accuracy"],
                out_path,
            )
            return final, out_metrics
        finally:
            if ctx_changed:
                self._dataset_context = prev_ctx

    @classmethod
    def from_ml_config(
        cls,
        ml_cfg: dict[str, Any],
        *,
        output_dir: Path,
        repository: TradeRepository | None = None,
        v4_training_cfg: dict[str, Any] | None = None,
    ) -> Trainer:
        """Construye un ``Trainer`` desde el bloque ``ml`` del YAML.

        Args:
            v4_training_cfg: subbloque ``v4_hybrid.training`` opcional. Si está
                presente y ``cv_strategy`` viene definido, se aplica. Si no, se
                conserva ``walk_forward`` (comportamiento 3.0).
        """
        v4 = dict(v4_training_cfg or {})
        return cls(
            output_dir=output_dir,
            repository=repository,
            training_weight_by_r=bool(ml_cfg.get("training_sample_weight_by_r", True)),
            walk_forward_embargo_ratio=float(
                v4.get("purged_kfold_embargo_pct",
                       ml_cfg.get("walk_forward_embargo_ratio", 0.02))
            ),
            label_mode=str(ml_cfg.get("label_mode", "win_loss")),
            min_r_for_positive=float(ml_cfg.get("min_r_for_positive", 0.25)),
            calibration_method=ml_cfg.get("calibration_method", "sigmoid"),
            calibration_holdout_ratio=float(ml_cfg.get("calibration_holdout_ratio", 0.15)),
            dataset_context=ml_cfg.get("train_dataset_context"),
            feature_schema=str(ml_cfg.get("feature_schema", "v1")),
            cv_strategy=str(v4.get("cv_strategy", "walk_forward")),
            cv_n_splits=int(v4.get("purged_kfold_splits", 5)),
        )

"""Motor de reentrenamiento diario y por bloques con validación walk-forward por símbolo."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from analytics.logger import get_logger
from analytics.quant_repository import QuantAnalyticsRepository
from analytics.trade_repository import TradeRepository

from .ensemble_model import EnsembleTradingModel
from .model_versioning import ModelVersionManager
from .oos_validator import compute_oos_metrics
from .quant_features import FEATURE_ORDER_V2, FEATURE_SCHEMA_VERSION_V2
from .symbol_model_registry import SymbolModelRegistry
from .trainer import Trainer

logger = get_logger(__name__)


class DailyRetrainer:
    """Recopila trades por símbolo, entrena, versiona y promueve solo si mejora OOS."""

    def __init__(
        self,
        *,
        repository: TradeRepository,
        models_dir: Path,
        ml_cfg: dict[str, Any],
        use_ensemble: bool = True,
        feature_schema: str = "v2",
        asset_profiles: dict[str, Any] | None = None,
        config_symbols: list[str] | None = None,
        quant_repo: QuantAnalyticsRepository | None = None,
    ) -> None:
        self._repo = repository
        self._models_dir = Path(models_dir)
        self._ml_cfg = ml_cfg
        self._use_ensemble = use_ensemble
        self._feature_schema = feature_schema
        self._registry = SymbolModelRegistry(
            self._models_dir,
            asset_profiles=asset_profiles,
            config_symbols=config_symbols,
        )
        self._quant_repo = quant_repo or QuantAnalyticsRepository()
        self._oos_cfg = dict(ml_cfg.get("oos_validation", {}) or {})
        self._last_daily_date: date | None = None
        self._min_symbol_trades = max(8, int(ml_cfg.get("retrain_min_symbol_trades", 12)))

    def _trainer(self) -> Trainer:
        return Trainer.from_ml_config(self._ml_cfg, output_dir=self._models_dir, repository=self._repo)

    def active_symbols(self) -> list[str]:
        return self._registry.active_symbols(self._repo, min_closed_trades=self._min_symbol_trades)

    def _version_mgr(self, symbol: str | None) -> ModelVersionManager:
        root = self._registry.version_manager_root(symbol) if symbol else self._models_dir
        return ModelVersionManager(root, symbol=symbol, oos_cfg=self._oos_cfg)

    def _extract_test_rs(self, rows: list[dict[str, Any]], i_start: int) -> list[float]:
        rs: list[float] = []
        for row in rows[i_start:]:
            raw = row.get("r_multiple")
            if raw is None:
                continue
            try:
                rs.append(float(raw))
            except (TypeError, ValueError):
                pass
        return rs

    def _dominant_regime(self, rows: list[dict[str, Any]]) -> str:
        counts: dict[str, int] = {}
        for row in rows[:200]:
            fx = row.get("features_json")
            if isinstance(fx, dict):
                reg = str(fx.get("volatility_regime", "UNKNOWN")).upper()
                counts[reg] = counts.get(reg, 0) + 1
        return max(counts, key=counts.get) if counts else "UNKNOWN"

    def _train_ensemble_walk_forward(
        self,
        *,
        limit: int,
        symbol: str,
    ) -> tuple[Any, dict[str, float]]:
        trainer = self._trainer()
        trainer._feature_schema = self._feature_schema  # type: ignore[attr-defined]
        rows = self._repo.get_training_dataset(limit=limit, symbol=symbol)
        timed = trainer._xy_timed_rows(rows, weight_by_r=trainer._training_weight_by_r)  # noqa: SLF001
        if len(timed) < max(80, int(self._ml_cfg.get("walk_forward_min_samples", 60))):
            raise ValueError(f"Dataset insuficiente {symbol}: {len(timed)}")

        n = len(timed)
        tr_ratio = float(self._ml_cfg.get("walk_forward_train_ratio", 0.65))
        va_ratio = float(self._ml_cfg.get("walk_forward_val_ratio", 0.15))
        i_tr = max(30, int(n * tr_ratio))
        i_va = min(n - 10, i_tr + max(10, int(n * va_ratio)))

        X_tr = [t[1] for t in timed[:i_tr]]
        y_tr = [t[2] for t in timed[:i_tr]]
        w_tr = [t[3] for t in timed[:i_tr]]
        X_va = [t[1] for t in timed[i_tr:i_va]]
        y_va = [t[2] for t in timed[i_tr:i_va]]
        X_te = [t[1] for t in timed[i_va:]]
        y_te = [t[2] for t in timed[i_va:]]

        train_acc = 0.0
        if self._use_ensemble:
            model = EnsembleTradingModel()
            model.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                feature_names=FEATURE_ORDER_V2,
                feature_schema_version=FEATURE_SCHEMA_VERSION_V2,
            )
            if X_va and y_va:
                from numpy import asarray

                proba_va = model.predict_proba(asarray(X_va, dtype=float))
                preds_va = [1 if p[1] >= p[0] else 0 for p in proba_va]
                train_acc = sum(int(p == y) for p, y in zip(preds_va, y_va, strict=True)) / max(len(y_va), 1)
            from numpy import asarray

            proba = model.predict_proba(asarray(X_te, dtype=float))
            y_arr = [int(v) for v in y_te]
            preds = [1 if p[1] >= p[0] else 0 for p in proba]
        else:
            _, wf_metrics = trainer.train_walk_forward_from_repository(
                limit=limit,
                model_filename="candidate_model.pkl",
                symbol=symbol,
            )
            return None, wf_metrics

        test_rs = self._extract_test_rs(rows, max(0, int(len(rows) * (i_va / max(n, 1)))))
        metrics = compute_oos_metrics(
            y_true=y_arr,
            y_pred=preds,
            r_multiples=test_rs,
            train_accuracy=train_acc,
        )
        metrics["walk_forward"] = 1.0
        return model, metrics

    def run(
        self,
        *,
        reason: str,
        symbol: str | None = None,
        limit: int | None = None,
        reload_cb: Callable[[str | None], None] | None = None,
    ) -> dict[str, float] | None:
        """Entrena un símbolo (obligatorio en modo Apex) o global legacy."""
        lim = limit or int(self._ml_cfg.get("auto_retrain_limit", 5000))
        sym = str(symbol).upper() if symbol else None
        if not sym:
            deploy = self._models_dir / "model.pkl"
            candidate = self._models_dir / "model.pkl"
        else:
            deploy = self._registry.deploy_path(sym)
            candidate = self._registry.candidate_path(sym)

        try:
            if self._use_ensemble and self._feature_schema == "v2" and sym:
                model, metrics = self._train_ensemble_walk_forward(limit=lim, symbol=sym)
                if model is not None:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    model.save(candidate)
            elif sym:
                trainer = self._trainer()
                _, metrics = trainer.train_walk_forward_from_repository(
                    limit=lim,
                    model_filename=candidate.name,
                    symbol=sym,
                )
            else:
                trainer = self._trainer()
                _, metrics = trainer.train_walk_forward_from_repository(
                    limit=lim,
                    model_filename="model.pkl",
                    symbol=None,
                )

            if not candidate.is_file():
                return None

            regime = "UNKNOWN"
            if sym:
                rows = self._repo.get_training_dataset(limit=min(lim, 500), symbol=sym)
                regime = self._dominant_regime(rows)

            vm = self._version_mgr(sym)
            promoted, vid = vm.register_candidate(
                candidate,
                metrics=metrics,
                reason=reason,
                deploy_path=deploy,
                regime_label=regime,
            )
            if sym and promoted:
                best = self._registry.best_path(sym)
                best.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(deploy, best)

            metrics["promoted"] = 1.0 if promoted else 0.0
            metrics["version_id"] = float(hash(vid) % 10_000)
            metrics["symbol"] = float(hash(sym or "GLOBAL") % 10_000)

            try:
                self._quant_repo.insert_model_performance(
                    model_version=vid,
                    symbol=sym,
                    regime_label=regime,
                    metrics={
                        "accuracy": metrics.get("accuracy"),
                        "winrate": metrics.get("winrate"),
                        "sharpe": metrics.get("sharpe"),
                        "expectancy": metrics.get("expectancy"),
                        "drawdown": metrics.get("drawdown"),
                        "profit_factor": metrics.get("profit_factor"),
                        "stability_score": metrics.get("stability_score"),
                        "consistency_score": metrics.get("consistency_score"),
                        "oos_accuracy": metrics.get("oos_accuracy"),
                        "false_signal_rate": metrics.get("false_signal_rate"),
                        "confidence_penalty": metrics.get("confidence_penalty"),
                        "overfitting_score": metrics.get("overfitting_score"),
                    },
                    sample_size=int(metrics.get("n_test", 0)),
                    notes=f"{reason}|promoted={promoted}",
                )
            except Exception as exc:
                logger.debug("insert_model_performance: %s", exc)

            if reload_cb:
                reload_cb(sym)
            return metrics
        except Exception as exc:
            logger.warning("DailyRetrainer.run(%s %s) falló: %s", reason, symbol, exc)
            return None

    def run_all_symbols(
        self,
        *,
        reason: str,
        reload_cb: Callable[[str | None], None] | None = None,
    ) -> dict[str, dict[str, float] | None]:
        results: dict[str, dict[str, float] | None] = {}
        for sym in self.active_symbols():
            results[sym] = self.run(reason=f"{reason}_{sym}", symbol=sym, reload_cb=reload_cb)
        return results

    def run_if_new_day(
        self,
        now_utc: datetime,
        *,
        reload_cb: Callable[[str | None], None] | None = None,
        include_global: bool = False,
    ) -> bool:
        d = now_utc.astimezone(timezone.utc).date()
        if self._last_daily_date == d:
            return False
        if now_utc.hour < int(self._ml_cfg.get("daily_retrain_utc_hour", 22)):
            return False
        results = self.run_all_symbols(reason="daily", reload_cb=reload_cb)
        if include_global:
            results["GLOBAL"] = self.run(reason="daily_global", symbol=None, reload_cb=reload_cb)
        ok = any(v is not None for v in results.values())
        if ok:
            self._last_daily_date = d
            logger.info("DailyRetrainer: fin de día %s results=%s", d, {k: (v or {}).get("promoted") for k, v in results.items()})
        return ok

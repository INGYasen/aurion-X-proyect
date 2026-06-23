"""Coordinador del motor cuantitativo auto-adaptativo (integración con Orchestrator)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

from analytics.logger import get_logger
from analytics.quant_decision_log import QuantDecisionLogger
from analytics.trade_repository import TradeRepository

from .confidence_engine import ConfidenceBreakdown, MLConfidenceEngine
from .daily_retrainer import DailyRetrainer
from .ensemble_model import EnsembleTradingModel
from .model import TradingModel
from .model_versioning import ModelVersionManager
from .predictor import Predictor
from .probability_engine import ProbabilityEngine
from .quant_features import (
    FEATURE_ORDER_V2,
    FEATURE_SCHEMA_VERSION_V2,
    build_quant_features,
)
from .self_learning_engine import SelfLearningEngine
from .symbol_model_registry import SymbolModelRegistry
from .trade_memory import TradeMemorySystem

logger = get_logger(__name__)


@dataclass
class QuantDecision:
    prob_buy: float
    prob_sell: float
    ml_confidence: float
    effective_confidence: float
    composite_score: float
    threshold_delta: float
    score_bias: int
    breakdown: dict[str, float]
    probabilities: dict[str, float]
    memory_insight: dict[str, Any]
    feature_vector: list[float]
    features_dict: dict[str, Any]
    lot_factor: float = 1.0
    block_weak_setups: bool = False


class QuantMLEngine:
    """Motor cuantitativo: features v2, ensemble, self-learning, retrain por símbolo."""

    def __init__(
        self,
        *,
        trade_repo: TradeRepository,
        ml_cfg: dict[str, Any],
        quant_cfg: dict[str, Any],
        asset_profiles: dict[str, Any],
        config_symbols: list[str] | None = None,
    ) -> None:
        self._repo = trade_repo
        self._ml_cfg = ml_cfg
        self._cfg = quant_cfg
        self._asset_profiles = asset_profiles
        self._config_symbols = config_symbols or []
        self.enabled = bool(quant_cfg.get("enabled", True))
        self._use_extended = bool(quant_cfg.get("use_extended_features", True))
        self._use_ensemble = bool(quant_cfg.get("use_ensemble", True))
        self._decision_weight = float(quant_cfg.get("ml_decision_weight", 0.62))
        self._min_structural_score = int(quant_cfg.get("min_structural_score", 28))
        models_dir = Path(str(quant_cfg.get("models_dir", ml_cfg.get("model_path", "D:/AURION_DATA/models/model.pkl"))))
        if models_dir.suffix.lower() == ".pkl":
            models_dir = models_dir.parent
        self._models_dir = models_dir
        self._global_path = Path(str(ml_cfg.get("model_path", models_dir / "model.pkl")))
        self._registry = SymbolModelRegistry(
            self._models_dir,
            asset_profiles=asset_profiles,
            config_symbols=self._config_symbols,
            legacy_global=self._global_path,
        )
        self._retrain_every = max(5, int(quant_cfg.get("retrain_every_closed_trades", 10)))
        self._retrain_min_interval_s = max(120.0, float(quant_cfg.get("retrain_min_interval_s", 300)))
        self._last_retrain_mono = 0.0
        self._closed_baseline = 0
        self._closed_baseline_by_symbol: dict[str, int] = {}
        self._last_learn_mono = 0.0
        self._learn_interval_s = max(60.0, float(quant_cfg.get("self_learn_interval_s", 120)))

        self._prob_engine = ProbabilityEngine()
        self._conf_engine = MLConfidenceEngine(
            ml_weight=float(quant_cfg.get("confidence_ml_weight", 0.45)),
            probability_weight=float(quant_cfg.get("confidence_probability_weight", 0.25)),
            quality_weight=float(quant_cfg.get("confidence_quality_weight", 0.15)),
            institutional_weight=float(quant_cfg.get("confidence_institutional_weight", 0.10)),
            execution_weight=float(quant_cfg.get("confidence_execution_weight", 0.05)),
        )
        self._self_learning = SelfLearningEngine(
            min_samples=int(quant_cfg.get("self_learn_min_samples", 20)),
            lookback=int(quant_cfg.get("self_learn_lookback", 400)),
        )
        mem_base = Path(str(quant_cfg.get("memory_path", "D:/AURION_DATA/memory/trade_memory.json")))
        self._memory_by_symbol: dict[str, TradeMemorySystem] = {}
        for sym in self._registry.configured_symbols():
            self._memory_by_symbol[sym] = TradeMemorySystem(
                self._registry.memory_path(sym, mem_base),
                recency_half_life_days=float(quant_cfg.get("memory_recency_half_life_days", 14)),
            )
        self._memory = self._memory_by_symbol.get("XAUUSD") or TradeMemorySystem(mem_base)

        merged_ml = {**ml_cfg, "feature_schema": "v2" if self._use_extended else "v1"}
        self._daily = DailyRetrainer(
            repository=trade_repo,
            models_dir=self._models_dir,
            ml_cfg=merged_ml,
            use_ensemble=self._use_ensemble,
            feature_schema="v2" if self._use_extended else "v1",
            asset_profiles=asset_profiles,
            config_symbols=self._config_symbols,
        )
        self._decision_log = QuantDecisionLogger()
        self._predictor_global = Predictor()
        self._predictor_by_symbol: dict[str, Predictor] = {}
        self._model_load_path: dict[str, Path] = {}
        self._reload_models()

        # AURION X 4.0 — hooks híbridos (opt-in; instanciar con attach_v4_hybrid)
        self._v4_transformer: Any = None
        self._v4_meta_stacker: Any = None

    def attach_v4_hybrid(
        self,
        *,
        transformer_predictor: Any = None,
        meta_stacker: Any = None,
    ) -> None:
        """Asocia las ramas Transformer + MetaStacker (AURION X 4.0).

        Si ambos son ``None`` (default), ``evaluate_entry`` se comporta como en
        3.0. Si están presentes, las probabilidades del ensemble se fusionan a
        través del MetaStacker.
        """
        self._v4_transformer = transformer_predictor
        self._v4_meta_stacker = meta_stacker
        logger.info(
            "QuantMLEngine: hooks v4 adjuntos (transformer=%s, meta_stacker=%s)",
            transformer_predictor is not None,
            meta_stacker is not None,
        )

    def _memory_for(self, symbol: str) -> TradeMemorySystem:
        s = symbol.upper()
        if s not in self._memory_by_symbol:
            mem_base = Path(str(self._cfg.get("memory_path", "D:/AURION_DATA/memory/trade_memory.json")))
            self._memory_by_symbol[s] = TradeMemorySystem(self._registry.memory_path(s, mem_base))
        return self._memory_by_symbol[s]

    def _load_model_into_predictor(self, path: Path, pred: Predictor) -> bool:
        if not path.is_file():
            return False
        try:
            payload_kind = ModelVersionManager.load_payload(path).get("kind", "")
            if payload_kind == "ensemble_voting":
                model: Any = EnsembleTradingModel(artifact_path=path)
            else:
                model = TradingModel(artifact_path=path)
            model.load()
            pred.set_model(model)  # type: ignore[arg-type]
            return True
        except Exception as exc:
            logger.warning("QuantMLEngine: no se cargó %s (%s)", path, exc)
            return False

    def _expected_feature_count(self) -> int:
        """Cantidad de features que el pipeline emite en runtime (v1 vs v2)."""
        return len(FEATURE_ORDER_V2) if self._use_extended else 10

    def _path_matches_pipeline_schema(self, path: Path) -> bool:
        """Confirma que n_features_in del .pkl casa con el pipeline runtime."""
        try:
            n_in = ModelVersionManager._payload_n_features(path)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("QuantMLEngine: no se pudo inspeccionar %s (%s)", path, exc)
            return False
        if n_in is None:
            return False
        return int(n_in) == int(self._expected_feature_count())

    def _resolve_model_path(self, symbol: str) -> Path | None:
        """Selecciona el .pkl cuyo schema casa con el pipeline en runtime.

        Si el archivo "principal" (best/deploy/global) no coincide en n_features,
        cae al candidate o al rollback antes que devolver un modelo que produce
        salidas neutrales (ml=0.5) por mismatch silencioso.
        """
        s = symbol.upper()
        primary = [
            self._registry.best_path(s),
            self._registry.deploy_path(s),
            self._global_path,
        ]
        fallback = [
            self._registry.candidate_path(s),
            self._registry.rollback_path(s),
        ]
        # 1) primaria con schema compatible
        for p in primary:
            if p.is_file() and self._path_matches_pipeline_schema(p):
                return p
        # 2) fallback compatible (candidate suele estar al día con v2)
        for p in fallback:
            if p.is_file() and self._path_matches_pipeline_schema(p):
                logger.warning(
                    "[QUANT_RELOAD] %s: usando %s por mismatch de schema en best/deploy",
                    s,
                    p,
                )
                return p
        # 3) último recurso: lo primero que exista (mejor un modelo neutral que ninguno)
        for p in primary + fallback:
            if p.is_file():
                return p
        return None

    def hot_reload_symbol(self, symbol: str | None, *, reload_cb: Callable[[], None] | None = None) -> bool:
        """Hot-reload sin reiniciar; fallback automático si falla."""
        if symbol:
            ok = self._reload_symbol(symbol.upper())
            if reload_cb:
                reload_cb()
            return ok
        self._reload_models()
        if reload_cb:
            reload_cb()
        return True

    def _reload_symbol(self, symbol: str) -> bool:
        path = self._resolve_model_path(symbol)
        if path is None:
            return False
        pr = Predictor()
        if self._load_model_into_predictor(path, pr):
            self._predictor_by_symbol[symbol] = pr
            self._model_load_path[symbol] = path
            logger.info("[QUANT_RELOAD] %s ← %s", symbol, path)
            return True
        rollback = self._registry.rollback_path(symbol)
        if rollback.is_file() and self._load_model_into_predictor(rollback, pr):
            self._predictor_by_symbol[symbol] = pr
            self._model_load_path[symbol] = rollback
            logger.warning("[QUANT_RELOAD] %s fallback rollback %s", symbol, rollback)
            return True
        return False

    def _reload_models(self) -> None:
        self._predictor_global = Predictor()
        if self._load_model_into_predictor(self._global_path, self._predictor_global):
            logger.info("QuantMLEngine: modelo global %s", self._global_path)
        for sym in self._registry.configured_symbols():
            self._reload_symbol(sym)

    def predictor_for(self, symbol: str) -> Predictor:
        return self._predictor_by_symbol.get(symbol.upper(), self._predictor_global)

    def has_model(self, symbol: str) -> bool:
        return self.predictor_for(symbol).model is not None

    def build_features(
        self,
        market_state: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        if self._use_extended:
            return build_quant_features(market_state, context=context)
        from .feature_engineering import build_features

        return build_features(market_state)

    def evaluate_entry(
        self,
        *,
        symbol: str,
        direction: str,
        market_state: Mapping[str, Any],
        signal: Mapping[str, Any],
        regime: str,
        mtf_trends: Mapping[str, str],
        structure_trend: str,
        continuation_strength: float,
        spread_points: float,
        spread_cap: float,
        rule_score: int,
        rule_confidence: float,
        context: Mapping[str, Any] | None = None,
    ) -> QuantDecision:
        sym = symbol.upper()
        learn_st = self._self_learning.state_for(sym)
        vec, fdict = self.build_features(market_state, context=context)
        pred = self.predictor_for(sym)
        prob_buy, prob_sell, ml_conf = 0.5, 0.5, 0.0
        if pred.model is not None:
            try:
                out = pred.predict(vec)
                prob_buy = float(out["prob_buy"])
                prob_sell = float(out["prob_sell"])
                ml_conf = float(out["confidence"])
            except Exception as exc:
                logger.debug("Quant predict fallback %s: %s", sym, exc)

        probs = self._prob_engine.compute(
            direction=direction,
            signal=signal,
            regime=regime,
            mtf_trends=mtf_trends,
            structure_trend=structure_trend,
            continuation_strength=continuation_strength,
            spread_points=spread_points,
            spread_cap=spread_cap,
        )
        breakdown: ConfidenceBreakdown = self._conf_engine.compute(
            direction=direction,
            prob_buy=prob_buy,
            prob_sell=prob_sell,
            ml_confidence=ml_conf,
            rule_score=rule_score,
            probabilities=probs,
            signal=signal,
            spread_points=spread_points,
            spread_cap=spread_cap,
            adaptive_weights=learn_st.adaptive_weights,
        )
        w = max(0.0, min(1.0, self._decision_weight))
        effective = (1.0 - w) * max(rule_confidence, 0.0) + w * breakdown.total
        effective = max(0.0, min(0.99, effective))

        # AURION X 4.1 — Blending institucional con la capa SMC (XAUUSD-only).
        # Si el Orchestrator inyectó features SMC en fdict (vía context), aplicamos
        # el split del prompt: BOS 10% + CHOCH 15% + OB 20% + FVG 15% + Liquidity
        # 10% (= 70% capa SMC) y ML conserva el 30% restante. El blending nunca
        # baja `effective` (sólo lo reescala suavemente) y respeta el peso ML
        # dominante porque el ML ya está dentro de `effective` previo.
        smc_keys_present = any(
            k in fdict for k in (
                "bos_strength", "choch_strength", "ob_strength",
                "fvg_strength", "liquidity_score", "confluence_score",
            )
        )
        if smc_keys_present:
            try:
                w_bos = 0.10
                w_choch = 0.15
                w_ob = 0.20
                w_fvg = 0.15
                w_liq = 0.10
                w_ml = 0.30
                # Si features_extra incluye pesos custom (vía context), respetarlos.
                ctx_weights = (context or {}).get("smc_weights") if isinstance(context, Mapping) else None
                if isinstance(ctx_weights, Mapping):
                    w_bos = float(ctx_weights.get("bos", w_bos))
                    w_choch = float(ctx_weights.get("choch", w_choch))
                    w_ob = float(ctx_weights.get("ob", w_ob))
                    w_fvg = float(ctx_weights.get("fvg", w_fvg))
                    w_liq = float(ctx_weights.get("liquidity", w_liq))
                    w_ml = float(ctx_weights.get("ml", w_ml))
                smc_layer = (
                    w_bos * float(fdict.get("bos_strength", 0.0))
                    + w_choch * float(fdict.get("choch_strength", 0.0))
                    + w_ob * float(fdict.get("ob_strength", 0.0))
                    + w_fvg * float(fdict.get("fvg_strength", 0.0))
                    + w_liq * float(fdict.get("liquidity_score", 0.0))
                )
                w_total = w_bos + w_choch + w_ob + w_fvg + w_liq + w_ml
                if w_total > 0:
                    blended = (smc_layer + w_ml * effective) / w_total
                    # Mezcla suave: 70% blended SMC + 30% effective ML para
                    # honrar la dominancia ML que pide el prompt y evitar saltos.
                    effective = max(0.0, min(0.99, 0.7 * blended + 0.3 * effective))
                    fdict["smc_blended_composite"] = float(effective)
            except Exception as exc:
                logger.debug("smc blending falló: %s", exc)

        # AURION X 4.0 — fusión opcional con PatchTST + MetaStacker.
        # Si los hooks no están conectados (default), el flujo es idéntico al 3.0.
        if self._v4_meta_stacker is not None and getattr(self._v4_meta_stacker, "is_fitted", False):
            prob_pst: float | None = None
            if self._v4_transformer is not None and getattr(self._v4_transformer, "is_active", False):
                tx_pred = self._v4_transformer.predict(market_state)
                if tx_pred is not None:
                    prob_pst = float(tx_pred["prob_buy"])
                    fdict["prob_patchtst"] = prob_pst
                    fdict["patchtst_latency_ms"] = float(tx_pred.get("latency_ms", 0.0))
            stacked = self._v4_meta_stacker.predict(
                prob_ensemble=prob_buy,
                prob_patchtst=prob_pst,
                effective_confidence=effective,
                setup_grade=str(signal.get("setup_grade") or ""),
            )
            # Sustituye prob_buy/sell por la versión fusionada (preserva 3.0 si
            # used_patchtst es False y solo ajusta sutil la magnitud).
            prob_buy = float(stacked.prob_buy)
            prob_sell = float(stacked.prob_sell)
            ml_conf = max(ml_conf, float(stacked.confidence))
            fdict["prob_buy_v4"] = prob_buy
            fdict["ml_confidence_v4"] = ml_conf
            fdict["v4_used_patchtst"] = bool(stacked.used_patchtst)

        mem_ctx = dict(fdict)
        mem_ctx.update(
            {
                "symbol": sym,
                "setup_type": signal.get("setup_type"),
                "volatility_regime": regime,
                "liquidity_sweep": signal.get("liquidity_sweep"),
                "session_type": (context or {}).get("session_type"),
            },
        )
        insight = self._memory_for(sym).meta_insight(mem_ctx)

        micro_q = float((market_state.get("micro") or {}).get("micro_quality", 0.5)) if isinstance(market_state.get("micro"), dict) else 0.5

        decision = QuantDecision(
            prob_buy=prob_buy,
            prob_sell=prob_sell,
            ml_confidence=ml_conf,
            effective_confidence=effective,
            composite_score=breakdown.total,
            threshold_delta=learn_st.threshold_delta,
            score_bias=learn_st.score_bias,
            breakdown=breakdown.as_dict(),
            probabilities=probs.as_dict(),
            memory_insight=insight,
            feature_vector=vec,
            features_dict=fdict,
            lot_factor=learn_st.lot_factor,
            block_weak_setups=learn_st.block_weak_setups,
        )

        if learn_st.block_weak_setups and insight.get("why") == "historical_drag":
            self._decision_log.log_block(sym, "memory", "degraded_setup", ml=ml_conf, micro=micro_q)

        will_block_weak = (
            learn_st.block_weak_setups and insight.get("why") == "historical_drag"
        )
        self._decision_log.log_entry(
            symbol=sym,
            direction=direction,
            allow=not will_block_weak,
            block_reason="weak_setup_memory" if will_block_weak else "-",
            scores={
                "total": effective,
                "ml": ml_conf,
                "institutional": breakdown.institutional,
                "micro": micro_q,
                "risk": 1.0 - learn_st.degradation_detected,
                "execution": breakdown.execution,
                "hyper": breakdown.total,
            },
            layers={"memory": insight, "model_path": str(self._model_load_path.get(sym, ""))},
        )
        return decision

    def blend_rule_score(self, rule_score: int, *, composite: float = 0.5, symbol: str | None = None) -> int:
        st = self._self_learning.state_for(symbol) if symbol else self._self_learning.state
        w = max(0.0, min(1.0, self._decision_weight))
        ml_part = int(max(0.0, min(1.0, composite)) * 100)
        blended = int((1.0 - w) * rule_score + w * ml_part) + st.score_bias
        return max(self._min_structural_score, min(100, blended))

    def periodic_learn(self) -> None:
        now_m = monotonic()
        if now_m - self._last_learn_mono < self._learn_interval_s:
            return
        self._last_learn_mono = now_m
        try:
            rows = self._repo.get_training_dataset(limit=int(self._cfg.get("self_learn_lookback", 400)))
            for sym in self._registry.active_symbols(self._repo, min_closed_trades=8):
                sym_rows = [r for r in rows if str(r.get("symbol", "")).upper() == sym]
                self._self_learning.ingest_rows(sym_rows, symbol=sym)
            self._self_learning.ingest_rows(rows, symbol=None)
        except Exception as exc:
            logger.debug("Quant periodic_learn: %s", exc)

    def on_trade_closed(
        self,
        *,
        symbol: str | None = None,
        features_json: dict[str, Any],
        profit: float,
        r_multiple: float,
        score: float,
        closed_counter: int,
        reload_cb: Callable[[], None] | None = None,
    ) -> None:
        sym = (symbol or str(features_json.get("symbol", ""))).upper() or "GLOBAL"
        fx = dict(features_json)
        fx.setdefault("symbol", sym)
        self._memory_for(sym).record_outcome(fx, profit=profit, r_multiple=r_multiple, score=score)

        baseline = self._closed_baseline_by_symbol.get(sym, self._closed_baseline)
        new_since = max(0, closed_counter - baseline)
        now_m = monotonic()
        learn_st = self._self_learning.state_for(sym)
        regime = str(fx.get("volatility_regime", "UNKNOWN"))
        regime_changed = self._self_learning.detect_regime_change(regime, symbol=sym)
        degrade = learn_st.degradation_detected
        interval_ok = (now_m - self._last_retrain_mono) >= self._retrain_min_interval_s
        should = interval_ok and (
            new_since >= self._retrain_every
            or degrade
            or regime_changed
            or self._self_learning.should_trigger_retrain(new_since, every_n=self._retrain_every, symbol=sym)
        )

        def _reload_one(s: str | None) -> None:
            self.hot_reload_symbol(s, reload_cb=reload_cb)

        if should:
            metrics = self._daily.run(
                reason="degradation" if degrade else "block",
                symbol=sym,
                reload_cb=_reload_one,
            )
            if metrics:
                self._last_retrain_mono = now_m
                self._closed_baseline_by_symbol[sym] = closed_counter
                self._closed_baseline = closed_counter
                logger.info("[QUANT_RETRAIN] %s promoted=%s metrics=%s", sym, metrics.get("promoted"), metrics)

    def on_end_of_day(self, now_utc: datetime, reload_cb: Callable[[], None] | None = None) -> None:
        if not self.enabled:
            return

        def _reload_one(s: str | None) -> None:
            self.hot_reload_symbol(s, reload_cb=reload_cb)

        ran = self._daily.run_if_new_day(now_utc, reload_cb=_reload_one, include_global=False)
        if ran:
            self._reload_models()

    def set_closed_baseline(self, n: int, *, symbol: str | None = None) -> None:
        if symbol:
            self._closed_baseline_by_symbol[symbol.upper()] = n
        else:
            self._closed_baseline = n
            for sym in self._registry.configured_symbols():
                self._closed_baseline_by_symbol[sym] = n

    def enrich_features_for_db(self, base: dict[str, Any], quant: QuantDecision) -> dict[str, Any]:
        out = dict(base)
        out.update(quant.features_dict)
        out["quant_breakdown"] = quant.breakdown
        out["quant_probabilities"] = quant.probabilities
        out["quant_memory"] = quant.memory_insight
        out["quant_lot_factor"] = quant.lot_factor
        out["feature_schema_version"] = (
            FEATURE_SCHEMA_VERSION_V2 if self._use_extended else out.get("feature_schema_version", 1)
        )
        return out

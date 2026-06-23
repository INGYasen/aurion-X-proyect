"""Motor Apex: capa institucional + microestructura + contexto total + meta-riesgo."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from analytics.equity_curve_ai import EquityCurveAI
from analytics.logger import get_logger
from analytics.performance_profiler import PerformanceProfiler
from execution.execution_intelligence import ExecutionIntelligenceEngine
from market.microstructure_engine import MicrostructureEngine
from ml.decision_hierarchy import DecisionHierarchy
from ml.hyper_context_engine import HyperContextEngine
from ml.institutional_engine import InstitutionalDecision, InstitutionalEngine
from ml.latent_market_state import LatentMarketStateAI
from ml.quant_engine import QuantDecision
from ml.reinforcement_layer import ReinforcementLayer
from ml.self_evolution_engine import SelfEvolutionEngine
from ml.trade_exit_ai import ExitAISignal
from risk.anti_overtrading import AntiOvertradingSystem
from risk.meta_risk_engine import MetaRiskEngine, RiskMode

logger = get_logger(__name__)


@dataclass
class ApexDecision:
    institutional: InstitutionalDecision
    hyper_score: float = 0.0
    meta_risk_mode: str = "neutral"
    micro_quality: float = 0.5
    latent_state: str = ""
    apex_block: str = ""
    hierarchy_layer: str = ""


class ApexEngine:
    """Envuelve InstitutionalEngine sin reemplazarlo; añade capas Apex."""

    def __init__(
        self,
        *,
        trade_repo: Any,
        ml_cfg: dict[str, Any],
        quant_cfg: dict[str, Any],
        institutional_cfg: dict[str, Any],
        asset_profiles: dict[str, Any],
        apex_cfg: dict[str, Any] | None = None,
    ) -> None:
        self._apex_cfg = dict(apex_cfg or {})
        self.enabled = bool(self._apex_cfg.get("enabled", True))
        self._inst = InstitutionalEngine(
            trade_repo=trade_repo,
            ml_cfg=ml_cfg,
            quant_cfg=quant_cfg,
            institutional_cfg=institutional_cfg,
            asset_profiles=asset_profiles,
        )
        self.enabled = self.enabled and self._inst.enabled

        self._micro = MicrostructureEngine()
        self._latent = LatentMarketStateAI(n_states=int(self._apex_cfg.get("latent_states", 6)))
        rl_path = self._apex_cfg.get("rl_memory_path")
        self._rl = ReinforcementLayer(
            Path(rl_path) if rl_path else None,
            learning_rate=float(self._apex_cfg.get("rl_learning_rate", 0.08)),
        )
        self._rl._load()
        self._meta_risk = MetaRiskEngine(self._apex_cfg.get("meta_risk", {}))
        self._hyper = HyperContextEngine(
            min_score=float(self._apex_cfg.get("hyper_min_score", 0.25)),
            cold_start_min=float(self._apex_cfg.get("hyper_cold_start_min", 0.15)),
        )
        self._hierarchy = DecisionHierarchy()
        self._anti_ot = AntiOvertradingSystem(
            max_entries_per_hour=int(self._apex_cfg.get("max_entries_per_hour", 4)),
            max_signals_per_10min=int(self._apex_cfg.get("max_signals_per_10min", 60)),
            min_spacing_s=float(self._apex_cfg.get("min_entry_spacing_s", 120.0)),
        )
        self._exec_intel = ExecutionIntelligenceEngine(
            window=int(self._apex_cfg.get("exec_intel_window", 120)),
            block_threshold=float(self._apex_cfg.get("exec_block_threshold", 0.38)),
        )
        self._equity_ai = EquityCurveAI()
        self._evolution = SelfEvolutionEngine(min_samples=int(self._apex_cfg.get("evolution_min_samples", 15)))
        self._profiler = PerformanceProfiler()
        self._equity_snapshots: deque[float] = deque(maxlen=80)
        self._last_apex: dict[str, ApexDecision] = {}
        self._last_evolution_day: date | None = None
        self._last_evolution_mono: float | None = None

        # AURION X 4.0 — anomaly detector opcional (instalado vía attach_v4_hybrid)
        self._v4_anomaly: Any = None

    def attach_v4_anomaly_detector(self, detector: Any) -> None:
        """Asocia ``ml.anomaly_autoencoder.AnomalyDetector``.

        Si nunca se llama, el flujo 3.0 se mantiene intacto. Si el detector
        está conectado pero ``is_active`` es ``False``, tampoco hace nada.
        """
        self._v4_anomaly = detector
        logger.info("ApexEngine: anomaly detector v4 adjunto (%s)", detector is not None)

    def notify_entry_opened(
        self,
        when_utc: datetime | None = None,
        symbol: str | None = None,
    ) -> None:
        """Registra apertura para anti-overtrade y contadores institucionales.

        ``symbol`` opcional permite que el anti-overtrade tracking sea
        independiente por instrumento (Fix #10).
        """
        self._anti_ot.record_entry(symbol or "")
        self._inst.notify_entry_opened(when_utc)

    def notify_strategy_signal(self, symbol: str | None = None) -> None:
        """Registra una señal real de estrategia (no por tick) en anti-overtrade.

        Debe invocarse desde el orquestador SOLO cuando la estrategia emite
        un BUY/SELL concreto, no en cada evaluación de tick. Esto evita que
        miles de evaluaciones rutinarias disparen ``signal_burst``.

        ``symbol`` opcional para tracking por instrumento (Fix #10).
        """
        self._anti_ot.record_signal(symbol or "")

    @property
    def quant_engine(self) -> Any:
        return self._inst.quant_engine

    @property
    def institutional(self) -> InstitutionalEngine:
        return self._inst

    def has_model(self, symbol: str) -> bool:
        return self._inst.quant_engine.has_model(symbol)

    def _session_from_context(self, context: Mapping[str, Any] | None) -> str:
        return str((context or {}).get("session_type", "") or "")

    def _enrich_market_state(
        self,
        market_state: Mapping[str, Any],
        *,
        direction: str,
        spread_points: float,
        symbol: str,
        regime: str,
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        ms = dict(market_state)
        closes = list(ms.get("closes_m1") or ms.get("closes") or [])
        highs = list(ms.get("highs_m1") or ms.get("highs") or closes)
        lows = list(ms.get("lows_m1") or ms.get("lows") or closes)
        micro = self._micro.analyze(
            closes=closes,
            highs=highs,
            lows=lows,
            spread_points=spread_points,
            direction=direction,
        )
        micro_d = micro.as_dict()
        latent = self._latent.infer(symbol, ms, micro=micro_d, regime=regime)
        latent_d = latent.as_dict()
        ms["micro"] = micro_d
        ms["latent"] = latent_d
        return ms

    def evaluate_entry(
        self,
        *,
        symbol: str,
        direction: str,
        market_state: Mapping[str, Any],
        signal: Mapping[str, Any],
        regime: Any,
        mtf_trends: Any,
        structure: Any,
        continuation_strength: float,
        spread_points: float,
        spread_cap: float,
        rule_score: int,
        rule_confidence: float,
        drawdown_pct: float,
        recent_winrate: float,
        context: Mapping[str, Any] | None = None,
    ) -> InstitutionalDecision:
        reg_val = regime.value if hasattr(regime, "value") else str(regime)
        vol_extreme = reg_val.upper() in ("EXTREME", "HIGH_SPIKE")

        enriched = self._enrich_market_state(
            market_state,
            direction=direction,
            spread_points=spread_points,
            symbol=symbol,
            regime=reg_val,
            context=context,
        )
        micro_d = enriched.get("micro") or {}
        latent_d = enriched.get("latent") or {}

        base = self._inst.evaluate_entry(
            symbol=symbol,
            direction=direction,
            market_state=enriched,
            signal=signal,
            regime=regime,
            mtf_trends=mtf_trends,
            structure=structure,
            continuation_strength=continuation_strength,
            spread_points=spread_points,
            spread_cap=spread_cap,
            rule_score=rule_score,
            rule_confidence=rule_confidence,
            drawdown_pct=drawdown_pct,
            recent_winrate=recent_winrate,
            context=context,
        )

        if not self.enabled:
            return base

        try:
            eq = float(context.get("equity", 0) if context else 0)
            if eq > 0:
                self._equity_snapshots.append(eq)
        except (TypeError, ValueError):
            pass

        equity_health = self._equity_ai.analyze(
            list(self._equity_snapshots) or [10000.0],
            trades_per_day=getattr(self._inst, "_trades_today", 0),
            max_trades_day=int(
                (self._inst._funding_guard.max_trades_per_day if hasattr(self._inst, "_funding_guard") else 10)
            ),
        )

        self._inst._funding_guard.adapt_dynamics(
            drawdown_pct=drawdown_pct,
            winrate=recent_winrate,
            volatility_extreme=vol_extreme,
            equity_stable=equity_health.stable_growth,
        )

        exec_snap = self._exec_intel.evaluate(
            spread_points=spread_points,
            spread_cap=spread_cap,
            session=self._session_from_context(context),
            volatility=reg_val,
        )
        exec_d = exec_snap.as_dict()

        meta = self._meta_risk.evaluate(
            drawdown_pct=drawdown_pct,
            recent_winrate=recent_winrate,
            trades_today=getattr(self._inst, "_trades_today", 0),
            max_trades_day=self._inst._funding_guard.max_trades_per_day,
            ml_degraded=base.quant.ml_confidence < 0.35,
            volatility_extreme=vol_extreme,
            consistency_score=equity_health.score,
            equity_slope=equity_health.slope,
        )

        policy = self._rl.policy()
        evo = self._evolution.state
        # Cold-start: detectar si el predictor está devolviendo salida neutral
        # (mismatch de features o modelo no cargado). En ese caso ``prob_buy`` y
        # ``prob_sell`` son ambos exactamente 0.5; el ML no es informativo y
        # debemos hacer bypass de las capas que dependen de él.
        ml_neutral_override = bool(
            abs(float(base.quant.prob_buy) - 0.5) < 1e-6
            and abs(float(base.quant.prob_sell) - 0.5) < 1e-6
        )
        # ML strong override: cuando el modelo está claramente sesgado (>=0.65),
        # confiamos más en él que en setup_grade o effective_confidence (que blendea
        # con métricas de mala historia). Permite bypass de grade=X y ajusta pesos
        # del hyper para que setup_score no domine cuando es bajo.
        ml_strong_override = (
            not ml_neutral_override
            and float(base.quant.ml_confidence) >= float(self._apex_cfg.get("ml_strong_threshold", 0.55))
        )
        hyper = self._hyper.fuse(
            ml_confidence=base.quant.effective_confidence,
            setup_score=base.setup_rank.score,
            regime_mode=base.trading_mode,
            micro=micro_d,
            latent=latent_d,
            exec_intel=exec_d,
            equity_health=equity_health.score,
            meta_risk_mode=meta.mode.value,
            reinforcement_lot=policy.lot_factor,
            ml_neutral_override=ml_neutral_override,
            ml_strong_override=ml_strong_override,
            ml_raw_confidence=float(base.quant.ml_confidence),
        )

        ml_thr_base = float(self._apex_cfg.get("ml_threshold_base", 0.36))
        ml_thr = ml_thr_base + base.threshold_delta + meta.threshold_add + policy.threshold_delta + evo.threshold_delta
        # Cuando el ML está claramente sesgado (raw>=0.65) el effective_confidence
        # del blender suele estar machacado por componentes débiles (institutional,
        # quality, etc. entrenados con histórico malo). Usamos un proxy que mezcla
        # effective con ml_raw para no rechazar entradas con convicción clara.
        ml_conf_for_hier = float(base.quant.effective_confidence)
        if ml_strong_override:
            # Boost máximo: cuando ML raw es fuerte (>=0.55), usar 95% directo.
            # El effective_confidence acumula penalizaciones de institucional/quality
            # con histórico pobre, lo que produce apex_ml:below_threshold a pesar
            # de ml_raw=0.6-0.85. Con 0.95*0.55=0.522 nos quedamos sobre el thr
            # base 0.36+deltas.
            ml_conf_for_hier = max(ml_conf_for_hier, 0.95 * float(base.quant.ml_confidence))
        else:
            # ML medio (0.50-0.54): mezcla más generosa para no bloquear cuando
            # el modelo está tibio pero direccionalmente acertado.
            ml_raw_blend = float(base.quant.ml_confidence)
            if ml_raw_blend >= 0.50:
                ml_conf_for_hier = max(ml_conf_for_hier, 0.90 * ml_raw_blend)
        ot = self._anti_ot.check(symbol)
        # Fix #10: permitir bypass del bloqueo anti-overtrade cuando hay
        # convicción combinada (ml_strong_override + setup A/A+). Antes el
        # guard tumbaba CUALQUIER entrada, incluso grade A+ con ml=0.87, lo
        # que en los logs nocturnos bloqueó >40% de las señales válidas. El
        # bypass NO aplica a "entry_spacing" (mínima separación entre trades),
        # solo a "max_entries_hour" y "signal_burst" donde la decisión
        # combinada justifica saltarse el límite de frecuencia.
        anti_overtrade_block = ot.block
        if (
            ot.block
            and ot.reason in ("max_entries_hour", "signal_burst")
            and ml_strong_override
            and base.setup_rank.grade.value in ("A+", "A")
        ):
            anti_overtrade_block = False

        # AURION X 4.0 — gate de anomalía (opcional, off por defecto).
        # Si el detector no está adjunto, ``self._v4_anomaly`` es None → no-op.
        if self._v4_anomaly is not None and getattr(self._v4_anomaly, "is_active", False):
            from time import monotonic as _mono
            # build_anomaly_input necesita spread_points/spread_cap/signal/range_quality
            # en el dict del market_state. Aquí componemos esa vista a partir del
            # market_state enriquecido más los parámetros locales.
            ms_for_anom = dict(enriched)
            ms_for_anom.setdefault("spread_points", float(spread_points))
            ms_for_anom.setdefault("spread_cap", float(spread_cap))
            ms_for_anom.setdefault("signal", dict(signal) if isinstance(signal, Mapping) else {})
            blocked_anom, anom_reason, _anom_score = self._v4_anomaly.should_block(
                now_mono=_mono(),
                micro=micro_d,
                market_state=ms_for_anom,
            )
            if blocked_anom:
                out = InstitutionalDecision(
                    quant=base.quant,
                    setup_rank=base.setup_rank,
                    asset_profile=base.asset_profile,
                    trading_mode=base.trading_mode,
                    lot_multiplier=0.0,
                    threshold_delta=base.threshold_delta,
                    score_bias=base.score_bias,
                    allow_trade=False,
                    block_reason=anom_reason,
                    execution_score=base.execution_score,
                    prediction_id=base.prediction_id,
                )
                self._last_apex[symbol.upper()] = ApexDecision(
                    institutional=out,
                    hyper_score=hyper.score,
                    meta_risk_mode=meta.mode.value,
                    micro_quality=float(micro_d.get("micro_quality", 0.5)),
                    latent_state=str(latent_d.get("latent_state", "")),
                    apex_block=anom_reason,
                    hierarchy_layer="anomaly",
                )
                return out

        hier = self._hierarchy.evaluate(
            risk_ok=drawdown_pct < 8.0,
            risk_reason="drawdown_cap" if drawdown_pct >= 8.0 else "ok",
            hyper_allow=hyper.allow_trade,
            hyper_reason=hyper.block_reason,
            regime_dangerous=bool(latent_d.get("latent_dangerous")),
            micro_quality=float(micro_d.get("micro_quality", 0.5)),
            manipulation_score=float(micro_d.get("manipulation_score", 0)),
            ml_confidence=ml_conf_for_hier,
            ml_threshold=ml_thr,
            setup_grade=base.setup_rank.grade.value,
            execution_block=exec_snap.block_entry,
            anti_overtrade_block=anti_overtrade_block,
            ml_neutral_override=ml_neutral_override,
            ml_strong_override=ml_strong_override,
        )

        allow = base.allow_trade and hier.allow
        block = base.block_reason or hier.reason
        if not hier.allow:
            block = f"apex_{hier.failed_layer}:{hier.reason}"
        elif not hyper.allow_trade:
            allow = False
            block = f"apex_hyper:{hyper.block_reason}"

        lot_mult = (
            base.lot_multiplier
            * meta.lot_factor
            * policy.lot_factor
            * equity_health.lot_factor
            * evo.aggression
        )
        if meta.mode == RiskMode.DEFENSIVE:
            lot_mult *= 0.88
        elif meta.mode == RiskMode.AGGRESSIVE and recent_winrate >= 0.55:
            lot_mult = min(lot_mult * 1.06, base.lot_multiplier * 1.15)

        thr_delta = base.threshold_delta + meta.threshold_add + policy.threshold_delta + evo.threshold_delta
        score_bias = base.score_bias + int(hier.confidence_adjust * 100)

        out = InstitutionalDecision(
            quant=base.quant,
            setup_rank=base.setup_rank,
            asset_profile=base.asset_profile,
            trading_mode=base.trading_mode,
            lot_multiplier=lot_mult,
            threshold_delta=thr_delta,
            score_bias=score_bias,
            allow_trade=allow,
            block_reason=block,
            execution_score=min(base.execution_score, exec_snap.score),
            prediction_id=base.prediction_id,
        )
        self._last_apex[symbol.upper()] = ApexDecision(
            institutional=out,
            hyper_score=hyper.score,
            meta_risk_mode=meta.mode.value,
            micro_quality=float(micro_d.get("micro_quality", 0.5)),
            latent_state=str(latent_d.get("latent_state", "")),
            apex_block=block,
            hierarchy_layer=hier.failed_layer if not hier.allow else "entry",
        )
        return out

    def evaluate_exit(
        self,
        *,
        trade: Mapping[str, Any],
        market_state: Mapping[str, Any],
        signal_context: Mapping[str, Any] | None = None,
    ) -> ExitAISignal:
        return self._inst._exit_ai.institutional_exit_trigger(
            trade=trade,
            market_state=market_state,
            signal_context=signal_context,
        )

    def fundingpips_check(self, **kwargs: Any) -> Any:
        return self._inst.fundingpips_check(**kwargs)

    def consistency_check(self, **kwargs: Any) -> Any:
        return self._inst.consistency_check(**kwargs)

    def run_daily_optimization(self, now_utc: datetime | None = None) -> None:
        # Usar fecha REAL (no la del tick) para evitar dispararse durante warm-up
        # con ticks históricos cross-day, y cooldown wallclock mínimo de 12h.
        from time import monotonic as _mono
        now = datetime.now(timezone.utc)
        self._inst.run_daily_optimization(now)
        if self._last_evolution_day == now.date():
            return
        now_m = _mono()
        last_m = getattr(self, "_last_evolution_mono", None)
        if last_m is not None and (now_m - last_m) < 12 * 3600.0:
            return
        self._last_evolution_day = now.date()
        self._last_evolution_mono = now_m
        try:
            rows = self._inst._trade_repo.get_training_dataset(limit=800)
            self._evolution.evolve_from_rows(rows)
            self._profiler.profile(rows)
            samples = []
            for r in rows[-120:]:
                try:
                    import numpy as np

                    from ml.latent_market_state import LatentMarketStateAI as L

                    ms = {"closes_m1": [float(r.get("entry_price", 0) or 0)] * 20}
                    samples.append(L._feature_vector(ms, None, str(r.get("volatility_regime", "NORMAL"))))
                except Exception:
                    pass
            if len(samples) >= 30:
                import numpy as np

                self._latent.fit_recent(samples)
            logger.info("[APEX] evolución diaria thr=%.3f agg=%.2f", self._evolution.state.threshold_delta, self._evolution.state.aggression)
        except Exception as exc:
            logger.debug("Apex daily evolution: %s", exc)

    def on_trade_closed(self, **kwargs: Any) -> None:
        self._inst.on_trade_closed(**kwargs)
        try:
            profit = float(kwargs.get("profit", 0) or 0)
            r_mult = float(kwargs.get("r_multiple", 0) or 0)
            dur = float(kwargs.get("duration_minutes", 30) or 30)
            conf = float(kwargs.get("entry_confidence", 0.5) or 0.5)
            dd = float(kwargs.get("drawdown_pct", 0) or 0)
            cons = float(kwargs.get("consistency_score", 0.5) or 0.5)
            rew = self._rl.compute_reward(
                profit=profit,
                r_multiple=r_mult,
                duration_minutes=dur,
                entry_confidence=conf,
                drawdown_pct=dd,
                consistency_score=cons,
            )
            self._rl.observe_outcome(rew, context_key=str(kwargs.get("symbol", "")))
        except Exception as exc:
            logger.debug("Apex on_trade_closed RL: %s", exc)

    def periodic_learn(self) -> None:
        self._inst.periodic_learn()

    def blend_rule_score(self, rule_score: int, *, composite: float) -> int:
        return self._inst.blend_rule_score(rule_score, composite=composite)

    def enrich_features_for_db(self, base: dict[str, Any], quant: QuantDecision) -> dict[str, Any]:
        out = self._inst.enrich_features_for_db(base, quant)
        sym = str(base.get("symbol", "")).upper()
        apex = self._last_apex.get(sym)
        if apex:
            out["apex_hyper_score"] = apex.hyper_score
            out["apex_meta_risk"] = apex.meta_risk_mode
            out["apex_micro_quality"] = apex.micro_quality
            out["apex_latent_state"] = apex.latent_state
        return out

    def record_execution_fill(self, **kwargs: Any) -> None:
        self._exec_intel.record_fill(**kwargs)

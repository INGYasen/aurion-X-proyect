"""Capa institucional: coordina quant + ranking + régimen + riesgo funded."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping

from analytics.logger import get_logger
from analytics.quant_repository import QuantAnalyticsRepository
from analytics.trade_repository import TradeRepository
from execution.execution_quality_engine import ExecutionQualityEngine
from market.market_regime_switcher import MarketRegimeSwitcher, TradingMode
from market.range_trading_engine import RangeTradingEngine
from market.volatility_detector import VolatilityRegime
from ml.asset_ml_weights import AssetMLProfile, resolve_asset_ml_profile
from ml.quant_engine import QuantDecision, QuantMLEngine
from ml.self_optimizer import SelfOptimizer
from ml.setup_ranking_engine import SetupGrade, SetupRankingEngine, SetupRank
from ml.trade_exit_ai import ExitAISignal, TradeExitAI
from risk.consistency_engine import ConsistencyEngine
from risk.fundingpips_guard import FundingPipsGuard
from risk.montecarlo import MonteCarloRiskEngine
from strategy.structure_detector import StructureResult, Trend

logger = get_logger(__name__)


@dataclass
class InstitutionalDecision:
    quant: QuantDecision
    setup_rank: SetupRank
    asset_profile: AssetMLProfile
    trading_mode: str
    lot_multiplier: float
    threshold_delta: float
    score_bias: int
    allow_trade: bool
    block_reason: str
    execution_score: float
    prediction_id: str | None = None


class InstitutionalEngine:
    """Motor institucional sobre QuantMLEngine (no lo reemplaza)."""

    def __init__(
        self,
        *,
        trade_repo: TradeRepository,
        ml_cfg: dict[str, Any],
        quant_cfg: dict[str, Any],
        institutional_cfg: dict[str, Any],
        asset_profiles: dict[str, Any],
    ) -> None:
        self._inst_cfg = institutional_cfg
        self.enabled = bool(institutional_cfg.get("enabled", True)) and bool(quant_cfg.get("enabled", True))
        symbols = [
            str(s).upper()
            for s in (institutional_cfg.get("symbols") or asset_profiles.keys() or [])
        ]
        self._quant = QuantMLEngine(
            trade_repo=trade_repo,
            ml_cfg=ml_cfg,
            quant_cfg=quant_cfg,
            asset_profiles=asset_profiles,
            config_symbols=symbols if symbols else None,
        )
        self._quant_repo = QuantAnalyticsRepository()
        self._trade_repo = trade_repo
        self._asset_profiles = asset_profiles
        self._setup_ranker = SetupRankingEngine(
            min_grade_trade=str(institutional_cfg.get("min_setup_grade", "C")),
        )
        self._regime_switcher = MarketRegimeSwitcher(
            defensive_dd_pct=float(institutional_cfg.get("defensive_dd_pct", 4.0)),
        )
        self._range_engine = RangeTradingEngine()
        self._exit_ai = TradeExitAI()
        self._exec_quality = ExecutionQualityEngine(window=int(institutional_cfg.get("exec_quality_window", 80)))
        self._funding_guard = FundingPipsGuard(institutional_cfg.get("fundingpips", {}))
        self._consistency = ConsistencyEngine(institutional_cfg.get("consistency", {}))
        self._self_opt = SelfOptimizer(min_samples=int(institutional_cfg.get("optimizer_min_samples", 12)))
        self._montecarlo = MonteCarloRiskEngine()
        self._mc_lot_factor = 1.0
        self._last_opt_day: datetime | None = None
        self._last_opt_mono: float | None = None
        self._trades_today = 0
        self._trades_hour = 0
        self._last_hour_key = ""
        self._last_trade_day: date | None = None

    def notify_entry_opened(self, when_utc: datetime | None = None) -> None:
        """Actualiza contadores intradía usados por meta-riesgo Apex."""
        now = when_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        day = now.date()
        if self._last_trade_day != day:
            self._trades_today = 0
            self._last_trade_day = day
        self._trades_today += 1
        hour_key = now.strftime("%Y%m%d%H")
        if hour_key != self._last_hour_key:
            self._trades_hour = 0
            self._last_hour_key = hour_key
        self._trades_hour += 1

    @property
    def quant_engine(self) -> QuantMLEngine:
        return self._quant

    def has_model(self, symbol: str) -> bool:
        return self._quant.has_model(symbol)

    def _asset(self, symbol: str) -> AssetMLProfile:
        return resolve_asset_ml_profile(symbol, self._asset_profiles, self._inst_cfg)

    def evaluate_entry(
        self,
        *,
        symbol: str,
        direction: str,
        market_state: Mapping[str, Any],
        signal: Mapping[str, Any],
        regime: VolatilityRegime,
        mtf_trends: Mapping[str, Trend],
        structure: StructureResult,
        continuation_strength: float,
        spread_points: float,
        spread_cap: float,
        rule_score: int,
        rule_confidence: float,
        drawdown_pct: float,
        recent_winrate: float,
        context: Mapping[str, Any] | None = None,
    ) -> InstitutionalDecision:
        asset = self._asset(symbol)
        self._quant._decision_weight = asset.ml_decision_weight  # noqa: SLF001

        regime_switch = self._regime_switcher.decide(
            volatility=regime,
            structure_trend=structure.trend,
            mtf_trends=mtf_trends,
            drawdown_pct=drawdown_pct,
            recent_winrate=recent_winrate,
            spread_spike=spread_points > spread_cap * 0.9,
        )

        setup_type = str(signal.get("setup_type", "")).upper()
        if "RANGE" in setup_type or regime_switch.mode.value == "range_trading":
            range_an = self._range_engine.analyze(
                structure,
                signal,
                regime=regime.value,
                spread_points=spread_points,
                spread_cap=spread_cap,
            )
            if not range_an.tradeable and "RANGE" in setup_type:
                q_stub = self._quant.evaluate_entry(
                    symbol=symbol,
                    direction=direction,
                    market_state=market_state,
                    signal=signal,
                    regime=regime.value,
                    mtf_trends={k: v.value for k, v in mtf_trends.items()},
                    structure_trend=structure.trend.value,
                    continuation_strength=continuation_strength,
                    spread_points=spread_points,
                    spread_cap=spread_cap,
                    rule_score=rule_score,
                    rule_confidence=rule_confidence,
                    context=context,
                )
                # ML strong escape: cuando el modelo está claramente sesgado
                # (>=0.60) dejamos pasar al flujo normal con setup_rank degradado
                # para que el Apex decida con su propia red de seguridad.
                # Cubre los casos:
                #   - no_range_structure: el structure detector no encontró
                #     estructura de rango (mismatch con un setup tipo RANGE).
                #   - range_clean_pin=X / range_compression_pin=X /
                #     range_chop_pin=X: el motor sí tiene rango pero la calidad
                #     o el extremo (edge) no cumplen el umbral. Si ML tiene
                #     convicción, no cortamos aquí.
                #   - insufficient_levels: la estructura tiene pocos niveles
                #     clave pero el contexto puede ser válido para un setup.
                ml_raw = float(getattr(q_stub, "ml_confidence", 0.5) or 0.5)
                _r = range_an.reason or ""
                _bypassable = (
                    _r == "no_range_structure"
                    or _r == "insufficient_levels"
                    or _r.startswith("range_clean_pin")
                    or _r.startswith("range_compression_pin")
                    or _r.startswith("range_chop_pin")
                )
                if ml_raw >= 0.55 and _bypassable:
                    pass  # cae al cálculo normal abajo
                else:
                    return InstitutionalDecision(
                        quant=q_stub,
                        setup_rank=SetupRank(
                            grade=SetupGrade.TRASH,
                            score=range_an.score,
                            reasons=(range_an.reason,),
                            lot_multiplier=0.0,
                            allow_aggressive=False,
                        ),
                        asset_profile=asset,
                        trading_mode=regime_switch.mode.value,
                        lot_multiplier=0.0,
                        threshold_delta=0.05,
                        score_bias=-10,
                        allow_trade=False,
                        block_reason=f"range_not_tradeable:{range_an.reason}",
                        execution_score=0.0,
                    )

        quant_dec = self._quant.evaluate_entry(
            symbol=symbol,
            direction=direction,
            market_state=market_state,
            signal=signal,
            regime=regime.value,
            mtf_trends={k: v.value for k, v in mtf_trends.items()},
            structure_trend=structure.trend.value,
            continuation_strength=continuation_strength,
            spread_points=spread_points,
            spread_cap=spread_cap,
            rule_score=rule_score,
            rule_confidence=rule_confidence,
            context=context,
        )

        opt_weights = self._self_opt.last_weights
        merged_weights = {**asset.weights, **opt_weights}

        rank = self._setup_ranker.rank(
            symbol=symbol,
            setup_type=setup_type,
            ml_confidence=quant_dec.ml_confidence,
            composite_score=quant_dec.composite_score,
            regime=regime.value,
            probabilities=quant_dec.probabilities,
            signal=signal,
            memory_insight=quant_dec.memory_insight,
            asset_profile=AssetMLProfile(
                symbol=asset.symbol,
                weights=merged_weights,
                ml_decision_weight=asset.ml_decision_weight,
                min_setup_grade_for_aggressive=asset.min_setup_grade_for_aggressive,
            ),
        )

        exec_rep = self._exec_quality.report(current_spread=spread_points, spread_cap=spread_cap)
        self._exec_quality.record(spread_points=spread_points)

        lot_mult = rank.lot_multiplier * self._mc_lot_factor * float(getattr(quant_dec, "lot_factor", 1.0))
        if exec_rep.recommendation == "avoid_entries":
            lot_mult *= 0.5
        elif exec_rep.recommendation == "caution":
            lot_mult *= 0.85

        allow = self._setup_ranker.allows_trade(rank)
        block = ""
        if not allow:
            block = f"setup_grade_{rank.grade.value}"
        mem_insight = quant_dec.memory_insight or {}
        mem_samples = int(mem_insight.get("samples", 0) or 0)
        mem_known = bool(mem_insight.get("known", False))
        if (
            getattr(quant_dec, "block_weak_setups", False)
            and mem_insight.get("why") == "historical_drag"
            and mem_known
            and mem_samples >= 12
        ):
            allow = False
            block = block or "quant_weak_setup_memory"

        thr_delta = quant_dec.threshold_delta + regime_switch.ml_threshold_add
        score_bias = quant_dec.score_bias + regime_switch.score_bias

        pred_id: str | None = None
        try:
            pred_id = str(
                self._quant_repo.insert_prediction(
                    symbol=symbol,
                    direction=direction,
                    confidence_score=quant_dec.effective_confidence,
                    continuation_probability=quant_dec.probabilities.get("prob_continuation"),
                    reversal_probability=quant_dec.probabilities.get("prob_reversal"),
                    sweep_probability=quant_dec.probabilities.get("prob_fake_sweep"),
                    absorption_probability=quant_dec.probabilities.get("prob_institutional_absorption"),
                    institutional_score=quant_dec.breakdown.get("institutional_score"),
                    execution_score=exec_rep.score,
                    ml_score=quant_dec.composite_score,
                    final_decision="ALLOW" if allow and not block else "BLOCK",
                    setup_grade=rank.grade.value,
                    extra={"trading_mode": regime_switch.mode.value, "reasons": list(rank.reasons)},
                ),
            )
            self._quant_repo.insert_features(
                symbol=symbol,
                features=quant_dec.features_dict,
                prediction_id=uuid.UUID(pred_id) if pred_id else None,
            )
            self._quant_repo.insert_regime(
                symbol=symbol,
                market_regime=structure.trend.value,
                volatility_state=regime.value,
                session_type=str((context or {}).get("session_type", "")),
                spread_state="high" if spread_points > spread_cap * 0.7 else "normal",
                manipulation_detected=quant_dec.probabilities.get("prob_fake_sweep", 0) > 0.65,
                sweep_density=float(signal.get("liquidity_sweep_score", 0) or 0),
                trading_mode=regime_switch.mode.value,
            )
        except Exception as exc:
            logger.debug("Persistencia quant analytics: %s", exc)

        return InstitutionalDecision(
            quant=quant_dec,
            setup_rank=rank,
            asset_profile=asset,
            trading_mode=regime_switch.mode.value,
            lot_multiplier=lot_mult,
            threshold_delta=thr_delta,
            score_bias=score_bias,
            allow_trade=allow and not block,
            block_reason=block,
            execution_score=exec_rep.score,
            prediction_id=pred_id,
        )

    def evaluate_exit(
        self,
        *,
        trade: Mapping[str, Any],
        market_state: Mapping[str, Any],
        signal_context: Mapping[str, Any] | None = None,
    ) -> ExitAISignal:
        return self._exit_ai.evaluate(
            trade=trade,
            market_state=market_state,
            signal_context=signal_context,
        )

    def fundingpips_check(self, **kwargs: Any) -> Any:
        return self._funding_guard.evaluate(**kwargs)

    def consistency_check(self, **kwargs: Any) -> Any:
        return self._consistency.evaluate(**kwargs)

    def run_daily_optimization(self, now_utc: datetime | None = None) -> None:
        # Usar fecha REAL (no la del tick) para no disparar durante warm-up de
        # ticks históricos cross-day, y añadir cooldown wallclock mínimo de 12h.
        from time import monotonic as _mono
        now = datetime.now(timezone.utc)
        if self._last_opt_day and self._last_opt_day.date() == now.date():
            return
        now_m = _mono()
        last_m = getattr(self, "_last_opt_mono", None)
        if last_m is not None and (now_m - last_m) < 12 * 3600.0:
            return
        self._last_opt_day = now
        self._last_opt_mono = now_m
        try:
            rows = self._trade_repo.get_training_dataset(limit=800)
            report = self._self_opt.analyze_day(rows)
            self._quant_repo.insert_model_performance(
                model_version="institutional_daily",
                metrics={
                    "winrate": report.best_setups[0][1] if report.best_setups else 0.0,
                    "sweep_effectiveness": report.sweep_effectiveness,
                },
                notes=str(report.recalibrated_weights),
                sample_size=len(rows),
            )
            rs = []
            for r in rows[:300]:
                raw = r.get("r_multiple")
                if raw is not None:
                    try:
                        rs.append(float(raw))
                    except (TypeError, ValueError):
                        pass
            if len(rs) >= 30:
                mc = self._montecarlo.simulate(rs)
                self._mc_lot_factor = mc.recommended_lot_factor
            self._quant.on_end_of_day(now, reload_cb=None)
            logger.info("[INST] daily optimization weights=%s mc_lot=%.2f", report.recalibrated_weights, self._mc_lot_factor)
        except Exception as exc:
            logger.warning("run_daily_optimization: %s", exc)

    def on_trade_closed(self, **kwargs: Any) -> None:
        self._quant.on_trade_closed(**kwargs)

    def periodic_learn(self) -> None:
        self._quant.periodic_learn()

    def blend_rule_score(self, rule_score: int, *, composite: float) -> int:
        return self._quant.blend_rule_score(rule_score, composite=composite)

    def enrich_features_for_db(self, base: dict[str, Any], quant: QuantDecision) -> dict[str, Any]:
        return self._quant.enrich_features_for_db(base, quant)

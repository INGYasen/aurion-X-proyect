"""Orquestador del pipeline mercado → estrategia → riesgo → ejecución."""

from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Mapping
from zoneinfo import ZoneInfo

import yaml

from analytics.db_connection import DatabaseConnection
from analytics.logger import get_logger
from analytics.trade_repository import TradeRepository
from execution.dynamic_exit_manager import build_exit_market_state, decide_dynamic_exit, merge_dynamic_exit_cfg
from execution.mt5_connector import MT5Connector
from execution.order_manager import OrderManager
from market.impulse_scan import format_impulse_summary, scan_impulse_legs
from market.multi_timeframe import SUPPORTED_TIMEFRAMES, MultiTimeframeEngine
from market.session_filter import SessionFilter, parse_session_time
from market.volatility_detector import VolatilityDetector, VolatilityRegime
from ml.feature_engineering import build_features
from ml.model import TradingModel
from ml.predictor import Predictor
from ml.apex_engine import ApexEngine
from ml.institutional_engine import InstitutionalDecision, InstitutionalEngine
from ml.quant_engine import QuantDecision, QuantMLEngine
from ml.trainer import Trainer
from risk.chop_guard import ChopGuard, ChopGuardConfig
from risk.emergency_controller import EmergencyController
from risk.position_sizer import PositionSizer
from risk.risk_manager import RiskManager
from strategy.entry_validator import EntryValidator
from strategy.intraday_engine import IntradayEngine
from strategy.scalper_engine import ScalperEngine
from market.weekday_filters import (
    WeekdayPolicy,
    build_weekday_policies_from_config,
    evaluate_weekday_policy,
    load_weekday_policies,
)
from strategy.structure_detector import StructureResult, Trend

if TYPE_CHECKING:
    from market.tick_handler import Tick

logger = get_logger(__name__)

_DEFAULT_CONFIG: dict[str, Any] = {
    "app": {"name": "AURION_X_3_0", "log_level": "INFO"},
    "equity": 10_000.0,
    "ml": {"threshold": 0.65},
    "execution": {"pip_value_per_lot": 1.0, "lot_step": 0.01, "max_lot": 1.0},
    "risk": {"risk_pct": 1.5},
}


@dataclass
class OpenTradeState:
    """Estado de gestión activa para un trade abierto."""

    ticket: str
    symbol: str
    direction: str
    entry_price: float
    risk_price: float
    sl_pips: float
    tp_pips: float
    lot: float
    be_done: bool = False
    trailing_done: bool = False
    reentry_done: bool = False
    tp_extended: bool = False
    tp_tightened: bool = False
    max_r: float = 0.0
    db_id: str | None = None
    setup_type: str = "UNKNOWN"
    sniper_key: str = ""
    # Gestión TP parcial en RANGE_FADE (objetivo: tomar beneficio antes del extremo opuesto).
    range_partial_tp_done: bool = False
    initial_tp_pips_planned: float = 0.0
    # Distancia en precio hasta TP al abrir (gestión intradía % recorrido hacia TP).
    initial_tp_distance_price: float = 0.0
    # Salida dinámica (execution.dynamic_exit)
    current_sl_price: float | None = None
    dynamic_exit_partial_done: bool = False
    last_dynamic_trail_sl: float | None = None


class Orchestrator:
    """Coordina el ciclo de vida del bot y el reparto de eventos entre subsistemas."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Crea el orquestador con ruta opcional a parámetros YAML.

        Args:
            config_path: Ubicación de ``parameters.yaml``; ``None`` usa valores por defecto.
        """
        self._config_path = config_path
        self._config: dict[str, Any] = {}
        self._running = False
        self._equity = Decimal("10000")
        self._session_filter = SessionFilter(enabled=False)
        self._volatility = VolatilityDetector(window=120)
        self._mtf = MultiTimeframeEngine()
        self._scalper = ScalperEngine()
        self._intraday = IntradayEngine()
        self._intraday_min_key_levels = 3
        self._entry_validator = EntryValidator()
        self._structure_timeframe = "M15"
        self._predictor = Predictor()
        self._predictor_range: Predictor | None = None
        self._predictor_by_symbol: dict[str, Predictor] = {}
        self._quant_engine: QuantMLEngine | InstitutionalEngine | ApexEngine | None = None
        self._institutional_engine: InstitutionalEngine | ApexEngine | None = None
        self._quant_decision_by_symbol: dict[str, QuantDecision] = {}
        # AURION X 4.0 HÍBRIDO — referencias a los componentes opcionales.
        # Se inicializan en _wire_v4_hybrid() si v4_hybrid.enabled: true.
        self._v4_gold_provider: Any = None
        self._v4_transformer: Any = None
        self._v4_meta_stacker: Any = None
        self._v4_anomaly_detector: Any = None
        self._v4_drl_shadow: Any = None
        self._inst_decision_by_symbol: dict[str, InstitutionalDecision] = {}
        self._risk = RiskManager()
        self._position_sizer = PositionSizer()
        self._order_manager = OrderManager()
        self._emergency = EmergencyController()
        self._trade_repo = TradeRepository(DatabaseConnection())
        self._bars: dict[str, dict[str, list[float]]] = {}
        self._open_trades: dict[str, OpenTradeState] = {}
        self._last_signal_by_symbol: dict[str, str] = {}
        self._atr_feature_hist: list[float] = []
        self._atr_spike_enabled = False
        self._atr_min_samples = 400
        self._last_emergency_warn_ts = 0.0
        self._last_emergency_reason: str | None = None
        self._last_mtf_log_ts: dict[str, float] = {}
        self._entry_skip_log_interval_s = 45.0
        self._entry_skip_log_mono: dict[tuple[str, str], float] = {}
        # Resumen rodante cross-símbolo: motivo de bloqueo → count. Se vuelca a INFO
        # cada `_block_summary_window_s` segundos (configurable) y se resetea.
        self._block_summary_window_s = 300.0
        self._block_summary_counts: dict[str, int] = {}
        # Counter por (reason_code, setup_type) para detectar setups tóxicos
        # antes de que generen pérdidas. Se vuelca junto a BLOCK_SUMMARY.
        self._block_summary_by_setup: dict[tuple[str, str], int] = {}
        self._block_summary_window_start_mono: float = monotonic()
        # Estado del régimen "chop macro persistente" por símbolo. Se activa
        # cuando H1+H4+D1+W1 están TODOS en RANGE durante un período mínimo
        # (default 30 min). En ese estado solo permitimos setups mean-reverting
        # (RANGE_FADE_REVERSAL, MEAN_REVERSION) salvo ml_strong override.
        self._chop_macro_state: dict[str, dict[str, Any]] = {}
        self._chop_macro_min_duration_s: float = 30.0 * 60.0
        # Tracker de bias ML por símbolo (Fix #8). Mantiene las últimas N
        # predicciones (ml_dir) y emite un WARN si >=80% van en una sola
        # dirección — síntoma de sesgo de modelo (re-entrenamiento desbalanceado
        # o feature drift). El threshold y la ventana son configurables.
        self._ml_dir_history: dict[str, deque[str]] = {}
        self._ml_bias_window: int = 200
        self._ml_bias_warn_ratio: float = 0.80
        self._ml_bias_min_samples: int = 80
        self._ml_bias_last_warn_mono: dict[str, float] = {}
        self._ml_bias_warn_interval_s: float = 1800.0
        self._min_entry_spacing_s = 0.0
        self._last_new_entry_mono_by_symbol: dict[str, float] = {}
        self._last_daily_opt_date: date | None = None
        self._last_daily_opt_mono: float = 0.0
        # Evita ráfagas intrabar: 1 entrada por vela M1 por símbolo.
        # Key = símbolo; value = "candle_id" (UTC, minuto).
        self._last_entry_m1_candle_id_by_symbol: dict[str, str] = {}
        self._mtf_entry_min_support = 2
        self._mtf_allow_all_entry_range = False
        self._mtf_ml_strong_override_th = 0.80
        self._impulse_log_cfg: dict[str, Any] = {}
        self._last_impulse_log_mono: dict[str, float] = {}
        self._ml_threshold_floor = 0.40
        self._spread_high_points = 120.0
        self._spread_extreme_points = 220.0
        self._score_min_normal = 55
        self._score_min_strong = 70
        self._asset_profiles: dict[str, Any] = {}
        self._scalper_by_symbol: dict[str, ScalperEngine] = {}
        self._telemetry_enabled = True
        self._telemetry_interval_s = 120.0
        self._telemetry_last_mono: dict[str, float] = {}
        self._telemetry_stats: dict[str, dict[str, int]] = {}
        self._ny_turbo_enabled = True
        self._ny_turbo_symbols: set[str] = {"XAUUSD"}
        self._ny_turbo_start = time(9, 30)
        self._ny_turbo_end = time(11, 30)
        # Horario NY turbo distinto por símbolo (p. ej. BTC más acotado); vacío = usa start/end globales.
        self._ny_turbo_symbol_windows: dict[str, tuple[time, time]] = {}
        self._adaptive_entry_bias: dict[str, float] = {}
        self._adaptive_perf_bias: dict[str, float] = {}
        self._adaptive_risk_multiplier: dict[str, float] = {}
        self._perf_learning_enabled = True
        self._perf_learning_interval_s = 180.0
        self._perf_learning_min_samples = 12
        self._perf_learning_lookback = 250
        self._last_perf_learn_mono: float = 0.0
        self._tp_hunter_enabled = True
        self._tp_hunter_mult_impulse = 1.22
        self._tp_hunter_mult_impulse_strong = 1.35
        self._tp_hunter_mult_low_vol = 0.90
        self._tp_hunter_min_rr = 1.15
        self._chart_hints_enabled = True
        self._chart_hints_min_conf = 0.45
        self._chart_hints_require_trade_ready = True
        self._chart_hints_path = Path("AURION_X_3_0/mql5/aurion_live_hints.csv")
        self._chart_hints_write_interval_ms = 250.0
        self._chart_hints_dedupe_by_m1_bar = True
        self._chart_hints_conf_round_decimals = 3
        self._chart_hints_min_score = 0
        self._chart_hints_rows_cache: dict[str, list[str]] = {}
        self._chart_hints_last_payload: dict[str, tuple[str, str, str, str, str, str, str]] = {}
        self._chart_hints_last_flush_mono = 0.0
        self._spread_guard_enabled = True
        self._spread_guard_lookback = 30
        self._spread_guard_spike_mult = 1.8
        self._spread_guard_max_points = 260.0
        self._dynamic_exit_enabled = False
        self._dynamic_exit_cfg: dict[str, Any] = {}
        self._ml_enabled = True
        self._ml_veto_enabled = True
        self._decision_interval_ms = 35.0
        self._last_decision_mono: dict[str, float] = {}
        self._sniper_enabled = True
        self._sniper_dead_hours_utc: set[int] = set()
        self._sniper_tz = ZoneInfo("America/New_York")
        self._sniper_min_samples = 10
        self._sniper_min_expectancy = 0.05
        self._sniper_min_winrate = 0.48
        self._sniper_allow_strong_override = True
        self._sniper_require_whitelist = False
        self._sniper_whitelist: dict[str, set[str]] = {}
        self._sniper_bucket_threshold_bias: dict[str, dict[str, float]] = {}
        self._sniper_bucket_score_bias: dict[str, dict[str, int]] = {}
        self._sniper_bucket_risk_mult: dict[str, dict[str, float]] = {}
        self._online_setup_perf: dict[str, dict[str, dict[str, float]]] = {}
        self._online_setup_min_samples = 5
        self._online_setup_bad_expectancy = -0.08
        self._online_setup_good_expectancy = 0.10
        self._online_setup_good_winrate = 0.52
        self._auto_retrain_enabled = False
        self._auto_retrain_min_closed = 60
        self._auto_retrain_every_closed = 25
        self._auto_retrain_limit = 5000
        self._auto_retrain_batch = True
        self._auto_retrain_min_interval_s = 1800.0
        self._auto_retrain_last_mono = 0.0
        self._closed_trades_counter = 0
        self._ml_closed_baseline = 0
        self._bootstrap_ml_on_start = True
        self._bootstrap_ml_min_closed = 30
        self._bootstrap_ml_force_on_start = False
        self._ml_model_path = Path("D:/AURION_DATA/models/model.pkl")
        self._ml_walk_forward_enabled = True
        self._ml_walk_forward_min_samples = 80
        self._ml_walk_forward_train_ratio = 0.65
        self._ml_walk_forward_val_ratio = 0.15
        self._correlation_groups: list[frozenset[str]] = []
        self._max_open_per_correlation_group = 1
        self._max_open_positions_total = 0
        self._max_open_positions_per_symbol = 0
        self._intraday_kill_enabled = True
        self._intraday_kill_window = 12
        self._intraday_kill_min_trades = 8
        self._intraday_kill_bad_winrate = 0.38
        self._intraday_kill_bad_avg_r = -0.12
        self._intraday_kill_require_both = True
        self._intraday_kill_cooldown_s = 1800
        # Fix #11: early circuit breaker per-símbolo (más sensible).
        self._early_kill_enabled = True
        self._early_kill_min_trades = 3
        self._early_kill_bad_winrate = 0.10
        self._early_kill_bad_avg_r = -3.0
        self._early_kill_require_both = True
        self._intraday_recent_r: dict[str, list[float]] = {}
        self._intraday_roll_day: dict[str, date] = {}
        self._weekday_policies: dict[int, WeekdayPolicy] = {}
        self._weekday_trade_counts: dict[tuple[str, date], int] = {}
        self._weekday_trade_day: date | None = None
        self._symbol_daily_pnl: dict[tuple[str, date], Decimal] = {}
        self._symbol_daily_pnl_day: date | None = None
        self._symbol_daily_loss_enabled = False
        self._symbol_daily_loss_limit = Decimal("0")
        self._intraday_kill_until: dict[str, datetime] = {}
        # ChopGuard: protecciones post-mortem 2-jun-2026 (range_position + micro cooldown)
        # Carga sus parámetros desde `execution.chop_guard` en parameters.yaml.
        self._chop_guard: ChopGuard | None = None
        self._ml_disabled_threshold_relax = 0.08
        self._ml_disabled_confidence_boost = 0.06
        self._lenient_extreme_volatility = False
        self._skip_mtf_filter = False
        self._broker_connector: MT5Connector | None = None
        self._use_live_equity = True
        self._entry_score_floor = 0
        self._log_trade_attempts = False
        self._symbols_trading: list[str] = []
        self._order_magic = 300000
        self._reconcile_filter_magic = True
        self._reconcile_interval_s = 0.0
        self._last_reconcile_mono: float = 0.0
        self._reconcile_before_new_entry = True
        self._scalper_require_h4_align = True
        self._block_scalper_counter_htf = False
        # Scalper momentum/breakout en chop (EXTREME + M5 RANGE) suele generar muchas pérdidas pequeñas.
        self._skip_scalper_extreme_m5_range = True
        self._min_sl_atr_m1_mult = 0.0
        self._max_sl_atr_m1_mult = 1.75
        self._max_sl_price_pct = 0.0012
        self._min_tp_rr_after_sl_floor = 1.12
        self._max_tp_rr = 2.35
        self._max_tp_rr_intraday = 2.55
        self._enforce_broker_position_caps = True
        self._environment = "demo"
        self._edge_fusion_enabled = False
        self._edge_fusion_mtf_stack_bonus = 12
        self._edge_fusion_ml_prob_aligned_boost = 0.04
        self._edge_fusion_ml_prob_min = 0.56
        logger.info("Orchestrator inicializado")

    def _m1_candle_id(self, when_utc: datetime) -> str:
        """Identificador estable por minuto (vela M1) en UTC."""
        dt = when_utc.astimezone(timezone.utc) if when_utc.tzinfo else when_utc.replace(tzinfo=timezone.utc)
        dt = dt.replace(second=0, microsecond=0)
        return dt.isoformat()

    def bind_connector(self, connector: MT5Connector) -> None:
        """Vincula conector de ejecución para enviar órdenes reales vía OrderManager."""
        self._broker_connector = connector
        if self._config:
            app_cfg = self._config.get("app", {}) or {}
            self._environment = str(app_cfg.get("environment", self._environment)).strip().lower()
        self._order_manager.set_connector(connector)
        self._order_manager.set_environment(self._environment)

    def _pip_size(self, symbol: str) -> float:
        s = symbol.upper()
        if s == "XAUUSD":
            return 0.1
        if s in {"USTEC", "NDX100", "NAS100", "US100"}:
            return 1.0
        if s in {"BTCUSD", "BITCOIN"}:
            return 1.0
        return 0.0001

    def _normalize_entry_sl_tp(
        self,
        symbol: str,
        strategy_name: str,
        signal: dict[str, Any],
        sl_pips: float,
        tp_pips: float,
        *,
        atr_now: float,
        entry_price: float | None = None,
    ) -> tuple[float, float]:
        """Suelo/topo de SL y TP en pips del símbolo (evita stops demasiado cortos y TP exagerados)."""
        pip_size = max(self._pip_size(symbol), 1e-12)
        prof = self._profile_for_symbol(symbol)
        exec_cfg = self._config.get("execution", {}) or {}

        try:
            leg_thr = float(exec_cfg.get("legacy_sl_contamination_threshold_pct", 0.0022) or 0.0022)
        except (TypeError, ValueError):
            leg_thr = 0.0022
        if entry_price is not None and entry_price > 0.0 and leg_thr > 0.0:
            max_phys = float(entry_price) * leg_thr
            raw_sl = sl_pips * pip_size
            # Solo el patrón FX antiguo (riesgo×10000 como "pips") cuando la distancia es claramente absurda.
            if raw_sl > max_phys and (
                sl_pips >= 800.0 or raw_sl > max_phys * 2.5
            ):
                alt = (sl_pips / 10000.0) / pip_size
                if alt * pip_size <= max_phys * 1.25:
                    sl_pips = max(alt, 1.0)
            raw_tp = tp_pips * pip_size
            if raw_tp > max_phys * 3.5 and (
                tp_pips >= 1200.0 or raw_tp > max_phys * 8.0
            ):
                alt_tp = (tp_pips / 10000.0) / pip_size
                if alt_tp * pip_size <= max_phys * 6.0:
                    tp_pips = max(alt_tp, sl_pips * 1.01)

        sl_dist = signal.get("sl_price_dist")
        tp_dist = signal.get("tp_price_dist")
        if sl_dist is not None:
            try:
                sl_pips = max(float(sl_dist) / pip_size, 1.0)
                if tp_dist is not None:
                    tp_pips = max(float(tp_dist) / pip_size, sl_pips * 1.01)
            except (TypeError, ValueError):
                pass

        mn_atr = float(self._min_sl_atr_m1_mult)
        if mn_atr > 0.0 and atr_now > 0.0:
            # ChopGuard: en sesión asia el ATR M1 absoluto es bajo pero el spread
            # es relativamente alto → SLs muy ajustados son barridos. Aplicamos
            # un multiplicador adicional sólo en esas horas.
            asia_mult = 1.0
            if self._chop_guard is not None and self._chop_guard.enabled:
                try:
                    asia_mult = self._chop_guard.adjusted_sl_multiplier()
                except Exception:
                    asia_mult = 1.0
            sl_pips = max(sl_pips, (atr_now / pip_size) * mn_atr * asia_mult)

        try:
            prof_min_sl = prof.get("min_sl_pips")
            if prof_min_sl is not None:
                sl_pips = max(sl_pips, float(prof_min_sl))
        except (TypeError, ValueError):
            pass

        mx_atr = float(self._max_sl_atr_m1_mult)
        if mx_atr > 0.0 and atr_now > 0.0:
            sl_pips = min(sl_pips, (atr_now / pip_size) * mx_atr)

        try:
            prof_max_sl = prof.get("max_sl_pips")
            if prof_max_sl is not None:
                sl_pips = min(sl_pips, float(prof_max_sl))
        except (TypeError, ValueError):
            pass
        try:
            glob_max_sl = exec_cfg.get("max_sl_pips")
            if glob_max_sl is not None:
                sl_pips = min(sl_pips, float(glob_max_sl))
        except (TypeError, ValueError):
            pass

        if entry_price is not None and entry_price > 0.0:
            max_pct = prof.get("max_sl_price_pct")
            if max_pct is None:
                max_pct = exec_cfg.get("max_sl_price_pct", self._max_sl_price_pct)
            try:
                pct = float(max_pct)
                if pct > 0.0:
                    cap_pips = (float(entry_price) * pct) / pip_size
                    if cap_pips >= 1.0:
                        sl_pips = min(sl_pips, cap_pips)
            except (TypeError, ValueError):
                pass

        rr_fl = max(1.01, float(self._min_tp_rr_after_sl_floor))
        tp_pips = max(tp_pips, sl_pips * rr_fl)

        is_intraday = str(strategy_name).upper().startswith("INTRADAY")
        try:
            prof_max_rr = prof.get("max_tp_rr")
            if prof_max_rr is not None:
                max_rr = float(prof_max_rr)
            elif is_intraday:
                max_rr = float(self._max_tp_rr_intraday)
            else:
                max_rr = float(self._max_tp_rr)
        except (TypeError, ValueError):
            max_rr = float(self._max_tp_rr_intraday if is_intraday else self._max_tp_rr)

        tp_pips = min(tp_pips, sl_pips * max(1.05, max_rr))

        try:
            prof_max_tp = prof.get("max_tp_pips")
            if prof_max_tp is not None:
                tp_pips = min(tp_pips, float(prof_max_tp))
        except (TypeError, ValueError):
            pass

        return max(sl_pips, 1.0), max(tp_pips, sl_pips * rr_fl)

    def _sl_exceeds_hard_cap(
        self,
        symbol: str,
        sl_pips: float,
        entry_price: float,
    ) -> tuple[bool, str]:
        """True si el SL normalizado sigue siendo demasiado amplio (riesgo de margen/liquidación)."""
        if entry_price <= 0.0 or sl_pips <= 0.0:
            return False, ""
        pip_size = max(self._pip_size(symbol), 1e-12)
        prof = self._profile_for_symbol(symbol)
        exec_cfg = self._config.get("execution", {}) or {}
        sl_dist = sl_pips * pip_size
        max_pct = prof.get("max_sl_price_pct")
        if max_pct is None:
            max_pct = exec_cfg.get("max_sl_price_pct", self._max_sl_price_pct)
        try:
            pct = float(max_pct)
        except (TypeError, ValueError):
            pct = float(self._max_sl_price_pct)
        if pct > 0.0 and (sl_dist / entry_price) > pct * 1.02:
            return True, f"SL {sl_dist:.4f} > {pct * 100:.2f}% del precio"
        try:
            prof_max = prof.get("max_sl_pips")
            if prof_max is not None and sl_pips > float(prof_max) * 1.02:
                return True, f"SL {sl_pips:.1f} pips > tope {float(prof_max):.1f}"
        except (TypeError, ValueError):
            pass
        return False, ""

    def _order_sl_tp_sane(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
    ) -> tuple[bool, str]:
        """Valida orientación y magnitudes razonables de SL/TP vs entrada antes de order_send."""
        pip = max(self._pip_size(symbol), 1e-12)
        eps = max(pip * 2.0, abs(entry) * 1e-6)
        d = str(direction).upper()
        if d == "BUY":
            if not (sl < entry - eps and tp > entry + eps):
                return False, "SL/TP mal orientados para BUY"
        elif d == "SELL":
            if not (tp < entry - eps and sl > entry + eps):
                return False, "SL/TP mal orientados para SELL"
        else:
            return False, f"dirección inválida {direction!r}"
        prof = self._profile_for_symbol(symbol)
        exec_cfg = self._config.get("execution", {}) or {}
        raw_pct = prof.get("max_sl_price_pct")
        if raw_pct is None:
            raw_pct = exec_cfg.get("max_sl_price_pct", self._max_sl_price_pct)
        try:
            pct = float(raw_pct) * 1.4
        except (TypeError, ValueError):
            pct = 0.0025
        if abs(entry - sl) / max(abs(entry), 1e-9) > max(pct, 0.003):
            return False, "SL demasiado lejos de entrada tras calcular precios"
        if abs(entry - tp) / max(abs(entry), 1e-9) > max(pct, 0.003) * 4.5:
            return False, "TP demasiado lejos de entrada tras calcular precios"
        return True, ""

    def _min_entry_spacing_for_symbol(self, symbol: str) -> float:
        """Cooldown entre entradas: global desde YAML + opcional por activo en ``asset_profiles.<SYM>``."""
        base = self._min_entry_spacing_s
        prof = self._profile_for_symbol(symbol)
        raw = prof.get("min_seconds_between_entries")
        if raw is None:
            return base
        try:
            sym_s = max(0.0, float(raw))
        except (TypeError, ValueError):
            return base
        return max(base, sym_s)

    def _mid_price(self, tick: Tick) -> float:
        return float((tick.bid + tick.ask) / Decimal("2"))

    def _ensure_symbol_state(self, symbol: str) -> dict[str, list[float]]:
        if symbol not in self._bars:
            self._bars[symbol] = {"highs": [], "lows": [], "closes": [], "spreads": [], "vols": []}
        return self._bars[symbol]

    def _telemetry_symbol(self, symbol: str) -> dict[str, int]:
        s = symbol.upper()
        if s not in self._telemetry_stats:
            self._telemetry_stats[s] = {
                "signals": 0,
                "trades_opened": 0,
                "trades_closed": 0,
                "wins": 0,
                "losses": 0,
                "blocked_total": 0,
                "blocked_session": 0,
                "blocked_mtf": 0,
                "blocked_ml_direction": 0,
                "blocked_validator": 0,
                "blocked_score": 0,
                "blocked_low_confidence": 0,
                "blocked_risk": 0,
                "blocked_no_signal": 0,
                "blocked_other": 0,
                "sum_pnl_cents": 0,
                "sum_r_milli": 0,
            }
        return self._telemetry_stats[s]

    def _telemetry_inc(self, symbol: str, key: str, amount: int = 1) -> None:
        stats = self._telemetry_symbol(symbol)
        stats[key] = stats.get(key, 0) + amount

    def _telemetry_block(
        self,
        symbol: str,
        reason_code: str,
        *,
        setup_type: str | None = None,
    ) -> None:
        self._telemetry_inc(symbol, "blocked_total")
        mapping = {
            "session": "blocked_session",
            "mtf_align": "blocked_mtf",
            "ml_direction": "blocked_ml_direction",
            "validator": "blocked_validator",
            "score_low": "blocked_score",
            "low_confidence": "blocked_low_confidence",
            "risk_pause": "blocked_risk",
            "risk_sizing": "blocked_risk",
            "m1_warmup": "blocked_no_signal",
            "no_strategy_signal": "blocked_no_signal",
            "order_none": "blocked_other",
            "zero_lot": "blocked_risk",
            "sniper_dead_hour": "blocked_other",
            "sniper_whitelist": "blocked_other",
            "spread_guard": "blocked_other",
            "intraday_kill": "blocked_other",
            "correlation_cap": "blocked_other",
            "scalper_extreme_m5_range": "blocked_other",
            "extreme_chop_no_volume": "blocked_extreme_chop",
            "range_breakout_in_chop": "blocked_extreme_chop",
            "chop_macro_persistente": "blocked_extreme_chop",
            "fundingpips": "blocked_revenge",
            "post_loss_cooldown": "blocked_revenge",
            "institutional_block": "blocked_institutional",
            "position_cap": "blocked_position_cap",
        }
        # Cualquier código que empiece por "apex_" (apex_ml:below_threshold,
        # apex_microstructure:..., apex_context:..., apex_hyper:...) va al bucket apex.
        if reason_code.startswith("apex_"):
            self._telemetry_inc(symbol, "blocked_apex")
        else:
            self._telemetry_inc(symbol, mapping.get(reason_code, "blocked_other"))
        # Agregador rodante cross-símbolo cada `_block_summary_window_s` segundos.
        self._block_summary_counts[reason_code] = (
            self._block_summary_counts.get(reason_code, 0) + 1
        )
        # Top setups bloqueados por reason_code (Fix #9). Permite identificar
        # antes setups tóxicos (p.ej. RANGE_BREAKOUT_MOMENTUM siendo bloqueado
        # repetidas veces por chop) sin esperar a que generen pérdidas.
        if setup_type:
            stype = str(setup_type).upper()
            pair_key = (reason_code, stype)
            self._block_summary_by_setup[pair_key] = (
                self._block_summary_by_setup.get(pair_key, 0) + 1
            )
        self._maybe_log_block_summary()

    def _track_ml_dir(self, symbol: str, prob_buy: float, prob_sell: float) -> None:
        """Registra la dirección ML predicha y emite warning si hay sesgo.

        ``ml_dir`` se decide por mayoría (BUY/SELL/NEUTRAL si gap pequeño).
        Si en las últimas ``_ml_bias_window`` predicciones de un símbolo
        >= ``_ml_bias_warn_ratio`` van a un solo lado, se emite WARN cada
        ``_ml_bias_warn_interval_s`` segundos. Sirve para detectar modelos
        sesgados antes de que generen pérdidas sistemáticas.
        """
        gap = abs(float(prob_buy) - float(prob_sell))
        if gap < 0.05:
            ml_dir = "NEUTRAL"
        else:
            ml_dir = "BUY" if prob_buy >= prob_sell else "SELL"
        key = symbol.upper()
        dq = self._ml_dir_history.get(key)
        if dq is None:
            dq = deque(maxlen=self._ml_bias_window)
            self._ml_dir_history[key] = dq
        dq.append(ml_dir)
        if len(dq) < self._ml_bias_min_samples:
            return
        buy_n = sum(1 for d in dq if d == "BUY")
        sell_n = sum(1 for d in dq if d == "SELL")
        total = len(dq)
        buy_ratio = buy_n / total
        sell_ratio = sell_n / total
        biased_side: str | None = None
        biased_ratio: float = 0.0
        if buy_ratio >= self._ml_bias_warn_ratio:
            biased_side = "BUY"
            biased_ratio = buy_ratio
        elif sell_ratio >= self._ml_bias_warn_ratio:
            biased_side = "SELL"
            biased_ratio = sell_ratio
        if biased_side is None:
            return
        now_m = monotonic()
        last = self._ml_bias_last_warn_mono.get(key, 0.0)
        if now_m - last < self._ml_bias_warn_interval_s:
            return
        self._ml_bias_last_warn_mono[key] = now_m
        logger.warning(
            "ML_BIAS %s: %.0f%% predicciones %s en últimas %d (%d BUY / %d SELL) — "
            "posible sesgo del modelo, revisar entrenamiento",
            symbol,
            biased_ratio * 100.0,
            biased_side,
            total,
            buy_n,
            sell_n,
        )

    def _ml_direction_bias_ratio(self, symbol: str) -> tuple[float, float]:
        """Devuelve (buy_ratio, sell_ratio) de las últimas predicciones ML.

        Sirve para detectar modelos con sesgo extremo (>=80% de un único lado)
        y, en ese caso, ajustar el comportamiento de filtros como
        ``ml_strong_mtf_override`` o calibraciones anti-bias en runtime. Si
        aún no hay suficientes muestras (``< _ml_bias_min_samples``) devuelve
        (0.0, 0.0) para evitar decisiones sobre datos insuficientes.
        """
        key = symbol.upper()
        dq = self._ml_dir_history.get(key)
        if dq is None or len(dq) < self._ml_bias_min_samples:
            return (0.0, 0.0)
        total = len(dq)
        if total <= 0:
            return (0.0, 0.0)
        buy_n = sum(1 for d in dq if d == "BUY")
        sell_n = sum(1 for d in dq if d == "SELL")
        return (buy_n / total, sell_n / total)

    def _maybe_log_block_summary(self) -> None:
        """Cada N segundos vuelca un único INFO con el top de motivos de bloqueo.

        Permite ver de un vistazo dónde se concentra el rechazo (cross-símbolo)
        sin sumar logs por evento. Window configurable en
        `execution.block_summary_window_s` (def. 300s). Desde Fix #9 también
        añade el top de setups bloqueados para detectar setups tóxicos antes de
        que generen pérdidas reales.
        """
        now_m = monotonic()
        if now_m - self._block_summary_window_start_mono < self._block_summary_window_s:
            return
        if self._block_summary_counts:
            total = sum(self._block_summary_counts.values())
            top = sorted(
                self._block_summary_counts.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:10]
            top_str = ", ".join(f"{k}={v}" for k, v in top)
            if self._block_summary_by_setup:
                top_setups = sorted(
                    self._block_summary_by_setup.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:5]
                top_setups_str = " | ".join(
                    f"{reason}@{stype}={cnt}" for (reason, stype), cnt in top_setups
                )
                logger.info(
                    "BLOCK_SUMMARY(%.0fs): total=%d top=%s top_setups=%s",
                    self._block_summary_window_s,
                    total,
                    top_str,
                    top_setups_str,
                )
            else:
                logger.info(
                    "BLOCK_SUMMARY(%.0fs): total=%d top=%s",
                    self._block_summary_window_s,
                    total,
                    top_str,
                )
        self._block_summary_counts.clear()
        self._block_summary_by_setup.clear()
        self._block_summary_window_start_mono = now_m

    def _maybe_log_telemetry(self, symbol: str) -> None:
        if not self._telemetry_enabled:
            return
        now_m = monotonic()
        last = self._telemetry_last_mono.get(symbol, 0.0)
        if now_m - last < self._telemetry_interval_s:
            return
        self._telemetry_last_mono[symbol] = now_m
        stats = self._telemetry_symbol(symbol)
        signals = max(0, stats.get("signals", 0))
        opened = max(0, stats.get("trades_opened", 0))
        closed = max(0, stats.get("trades_closed", 0))
        wins = max(0, stats.get("wins", 0))
        losses = max(0, stats.get("losses", 0))
        conversion = (100.0 * opened / signals) if signals > 0 else 0.0
        winrate = (100.0 * wins / closed) if closed > 0 else 0.0
        avg_r = (stats.get("sum_r_milli", 0) / 1000.0 / closed) if closed > 0 else 0.0
        net_pnl = stats.get("sum_pnl_cents", 0) / 100.0
        setup_stats = self._online_setup_perf.get(symbol.upper(), {})
        top_rows: list[tuple[str, int, float, float]] = []
        for setup, st in setup_stats.items():
            n = int(st.get("n", 0.0))
            if n <= 0:
                continue
            wins = float(st.get("wins", 0.0))
            sum_r = float(st.get("sum_r", 0.0))
            wr = (wins / max(float(n), 1.0)) * 100.0
            avg_r = sum_r / max(float(n), 1.0)
            top_rows.append((setup, n, wr, avg_r))
        top_rows.sort(key=lambda x: (x[3], x[2], x[1]), reverse=True)
        top_txt = " | ".join(
            f"{name}:n={n},wr={wr:.1f}%,avgR={avg_r:.2f}"
            for name, n, wr, avg_r in top_rows[:3]
        ) or "sin_datos"
        breakdown = [
            ("session", stats.get("blocked_session", 0)),
            ("mtf", stats.get("blocked_mtf", 0)),
            ("ml_dir", stats.get("blocked_ml_direction", 0)),
            ("validator", stats.get("blocked_validator", 0)),
            ("score", stats.get("blocked_score", 0)),
            ("low_conf", stats.get("blocked_low_confidence", 0)),
            ("risk", stats.get("blocked_risk", 0)),
            ("no_signal", stats.get("blocked_no_signal", 0)),
            ("other", stats.get("blocked_other", 0)),
        ]
        dominant = max(breakdown, key=lambda kv: kv[1])
        dominant_txt = f"{dominant[0]}={dominant[1]}" if dominant[1] > 0 else "n/a"
        logger.info(
            "[TELEMETRY] symbol=%s signals=%s opened=%s closed=%s conv=%.1f%% winrate=%.1f%% avgR=%.3f netPnl=%.2f "
            "blocked=%s dominante=%s session=%s mtf=%s ml_dir=%s validator=%s score=%s low_conf=%s risk=%s "
            "no_signal=%s other=%s top_setups=%s",
            symbol.upper(),
            signals,
            opened,
            closed,
            conversion,
            winrate,
            avg_r,
            net_pnl,
            stats.get("blocked_total", 0),
            dominant_txt,
            stats.get("blocked_session", 0),
            stats.get("blocked_mtf", 0),
            stats.get("blocked_ml_direction", 0),
            stats.get("blocked_validator", 0),
            stats.get("blocked_score", 0),
            stats.get("blocked_low_confidence", 0),
            stats.get("blocked_risk", 0),
            stats.get("blocked_no_signal", 0),
            stats.get("blocked_other", 0),
            top_txt,
        )

    def _append_tick_to_bars(self, symbol: str, tick: Tick) -> None:
        state = self._ensure_symbol_state(symbol)
        mid = self._mid_price(tick)
        spread = float(tick.ask - tick.bid)
        vol = float(tick.volume) if tick.volume is not None else 0.0
        state["highs"].append(mid)
        state["lows"].append(mid)
        state["closes"].append(mid)
        state["spreads"].append(spread)
        state["vols"].append(vol)
        max_len = 900
        for k in state:
            if len(state[k]) > max_len:
                state[k] = state[k][-max_len:]
        self._mtf.ingest_tick(symbol, tick)

    def _mtf_allows(self, direction: str, trends: dict[str, Trend]) -> bool:
        """Filtro multi-timeframe desde mínimo operativo a mensual.

        ``mtf_entry_min_support`` (1–3): mínimo de timeframes M5/M15/H1 alineados con la señal.
        ``mtf_allow_when_all_entry_range``: si los tres son RANGE y el macro no contradice, permite pasar.
        """
        d = direction.upper()
        macro_tfs = ("H4", "D1", "W1", "MN1")
        entry_tfs = ("M5", "M15", "H1")
        need = max(1, min(3, int(self._mtf_entry_min_support)))

        def _all_entry_range() -> bool:
            return all(trends.get(tf, Trend.RANGE) == Trend.RANGE for tf in entry_tfs)

        if d == "BUY":
            if any(trends.get(tf) == Trend.DOWN for tf in macro_tfs):
                return False
            support = sum(1 for tf in entry_tfs if trends.get(tf) == Trend.UP)
            if support >= need:
                return True
            if self._mtf_allow_all_entry_range and _all_entry_range():
                return True
            return False
        if d == "SELL":
            if any(trends.get(tf) == Trend.UP for tf in macro_tfs):
                return False
            support = sum(1 for tf in entry_tfs if trends.get(tf) == Trend.DOWN)
            if support >= need:
                return True
            if self._mtf_allow_all_entry_range and _all_entry_range():
                return True
            return False
        return False

    def _maybe_log_mtf(self, symbol: str, trends: dict[str, Trend]) -> None:
        """Loguea snapshot MTF periódico para observabilidad sin spam."""
        now_ts = datetime.now(timezone.utc).timestamp()
        last_ts = self._last_mtf_log_ts.get(symbol, 0.0)
        if now_ts - last_ts < 60.0:
            return
        ordered_tfs = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
        trends_txt = ", ".join(f"{tf}:{trends.get(tf, Trend.RANGE).value}" for tf in ordered_tfs)
        logger.info("MTF %s | %s", symbol, trends_txt)
        self._last_mtf_log_ts[symbol] = now_ts

    def _maybe_compute_smc(self, symbol: str) -> None:
        """Ejecuta y persiste el snapshot SMC periódicamente.

        Esto desacopla la capa institucional del flujo de señales: incluso
        cuando el scalper / range / impulso no producen candidatos, los Order
        Blocks, BOS, CHOCH, FVG y liquidez se siguen calculando y
        persistiendo, alimentando el dashboard y el EA de visualización en
        MT5. Soft-fail y throttled (cada 30s por símbolo).
        """
        if getattr(self, "_smc_analyzer", None) is None:
            return
        now_m = monotonic()
        if not hasattr(self, "_smc_last_compute_mono"):
            self._smc_last_compute_mono: dict[str, float] = {}
        last = self._smc_last_compute_mono.get(symbol, 0.0)
        if (now_m - last) < 30.0:
            return
        self._smc_last_compute_mono[symbol] = now_m
        try:
            snap = self._smc_analyze(symbol)
        except Exception as exc:
            logger.debug("[SMC] compute periódico falló %s: %s", symbol, exc)
            return
        if snap is None:
            return
        try:
            ob_n = 0
            bos_n = 0
            choch_n = 0
            fvg_n = 0
            liq_n = 0
            per_tf = getattr(snap, "per_tf", {}) or {}
            for tfc in per_tf.values():
                ob_n += len(getattr(tfc, "active_obs", []) or [])
                fvg_n += len(getattr(tfc, "active_fvgs", []) or [])
                liq_n += len(getattr(tfc, "liquidity_pools", []) or [])
                if getattr(tfc, "last_bos", None) is not None:
                    bos_n += 1
                if getattr(tfc, "last_choch", None) is not None:
                    choch_n += 1
            now_ts = datetime.now(timezone.utc).timestamp()
            if not hasattr(self, "_smc_last_log_ts"):
                self._smc_last_log_ts: dict[str, float] = {}
            last_log = self._smc_last_log_ts.get(symbol, 0.0)
            if (now_ts - last_log) >= 60.0:
                self._smc_last_log_ts[symbol] = now_ts
                logger.info(
                    "[SMC] %s snapshot ob=%d bos=%d choch=%d fvg=%d liq=%d "
                    "macro=%s inst=%s op=%s",
                    symbol,
                    ob_n,
                    bos_n,
                    choch_n,
                    fvg_n,
                    liq_n,
                    getattr(snap.macro_bias, "value", "?"),
                    getattr(snap.institutional_bias, "value", "?"),
                    getattr(snap.operational_bias, "value", "?"),
                )
        except Exception:
            pass
        try:
            self._knowledge_propagate_smc(symbol, snap.primary_timeframe, snap)
        except Exception:
            pass
        # AURION X 4.6 — actualiza ambient cache (sessions+volume) aun sin signal
        try:
            self._update_smc_ambient_cache(symbol, snap)
        except Exception as exc:
            logger.debug("[SMC_AMBIENT] update fallo (soft): %s", exc)

    def _update_smc_ambient_cache(self, symbol: str, snapshot: Any) -> None:
        """AURION X 4.6 — escribe contexto ambient (sessions + volume) al cache
        SMC aun cuando no hay signal candidata.

        El cache es el MISMO archivo ``runtime/smc_decisions/<SYM>.json`` que
        usa ``_write_smc_decision_cache`` para preservar compatibilidad. Si ya
        existia una ``decision`` la mantenemos y solo refrescamos sessions/vol.
        """
        import json
        import os
        from datetime import datetime, timezone
        from pathlib import Path
        try:
            from market.volume_classifier import classify_volume
            from market.session_liquidity import (
                compute_session_ranges, session_pools_for_targets,
            )
        except Exception:
            return
        # Velas M15 para ambient (compartido con _process_signal). El registry MTF
        # (MultiTimeframeEngine) expone `_bars_series(symbol, tf)` → list[_Bar]
        # con atributos .start/.open/.high/.low/.close/.volume — formato que ya
        # entienden classify_volume() y compute_session_ranges() (este último
        # acepta .start como alias de .time).
        candles: list = []
        try:
            mtf = getattr(self, "_mtf", None) or getattr(self, "_market_data", None)
            if mtf is not None and hasattr(mtf, "_bars_series"):
                bars = list(mtf._bars_series(symbol, "M15") or [])
                if bars:
                    candles = bars[-120:]  # cubre Asia+Londres+NY (M15)
        except Exception:
            candles = []
        # Volumen
        vol_class: str | None = None
        try:
            v = classify_volume(candles=candles)
            vol_class = v.classification.value
        except Exception:
            pass
        # Last close (mid del tick si tenemos)
        last_close: float | None = None
        try:
            if candles:
                last_close = float(getattr(candles[-1], "close", 0.0) or 0.0)
        except Exception:
            last_close = None
        # Sessions
        session_pools: dict[str, Any] | None = None
        try:
            ranges = compute_session_ranges(
                candles=candles,
                now=datetime.now(timezone.utc),
                last_close=last_close,
            )
            session_pools = session_pools_for_targets(ranges)
        except Exception:
            session_pools = None

        cache_dir = Path(__file__).resolve().parent.parent / "runtime" / "smc_decisions"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{symbol.upper()}.json"
        existing: dict[str, Any] = {}
        try:
            if target.exists():
                with open(target, "r", encoding="utf-8") as fp:
                    existing = json.load(fp) or {}
        except Exception:
            existing = {}
        # Fusionar: mantener decision/smart_targets/smart_stops si existían
        merged: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        merged["symbol"] = symbol.upper()
        merged["generated_at"] = datetime.now(timezone.utc).isoformat()
        merged["ambient_only"] = "decision" not in merged
        if vol_class is not None:
            merged["volume_class"] = vol_class
        if last_close is not None:
            merged["entry_price"] = float(last_close)
        if session_pools is not None:
            merged["session_pools"] = session_pools
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(merged, fp, ensure_ascii=False, default=str)
        os.replace(tmp, target)

    def _maybe_log_entry_skip(
        self,
        symbol: str,
        reason_code: str,
        fmt: str,
        *args: Any,
        setup_type: str | None = None,
    ) -> None:
        """INFO con throttle por (símbolo, código) para no saturar el log.

        ``setup_type`` opcional alimenta el agregador de setups bloqueados
        (Fix #9, BLOCK_SUMMARY enriquecido).
        """
        self._telemetry_block(symbol, reason_code, setup_type=setup_type)
        now_m = monotonic()
        key = (symbol, reason_code)
        last = self._entry_skip_log_mono.get(key, 0.0)
        if now_m - last < self._entry_skip_log_interval_s:
            return
        self._entry_skip_log_mono[key] = now_m
        logger.info(fmt, *args)

    def _weekday_policy_for(self, when_utc: datetime) -> WeekdayPolicy:
        wd = when_utc.weekday()
        return self._weekday_policies.get(wd, WeekdayPolicy())

    def _evaluate_weekday_policy(
        self,
        symbol: str,
        setup_type: str,
        policy: WeekdayPolicy,
        *,
        mtf_trends: dict[str, Trend] | None = None,
        base_ml_threshold: float = 0.0,
        trade_count_today: int = 0,
    ) -> tuple[bool, str]:
        return evaluate_weekday_policy(
            symbol,
            setup_type,
            policy,
            mtf_trends=mtf_trends,
            base_ml_threshold=base_ml_threshold,
            trade_count_today=trade_count_today,
        )

    def _weekday_trade_count_today(self, symbol: str, when_utc: datetime) -> int:
        d = when_utc.date()
        if self._weekday_trade_day != d:
            self._weekday_trade_counts.clear()
            self._weekday_trade_day = d
        return self._weekday_trade_counts.get((symbol.upper(), d), 0)

    def _increment_weekday_trade(self, symbol: str, when_utc: datetime) -> None:
        d = when_utc.date()
        key = (symbol.upper(), d)
        self._weekday_trade_counts[key] = self._weekday_trade_counts.get(key, 0) + 1

    def _symbol_daily_pnl_today(self, symbol: str, when_utc: datetime) -> Decimal:
        d = when_utc.date()
        if self._symbol_daily_pnl_day != d:
            self._symbol_daily_pnl.clear()
            self._symbol_daily_pnl_day = d
        return self._symbol_daily_pnl.get((symbol.upper(), d), Decimal("0"))

    def _accumulate_symbol_daily_pnl(self, symbol: str, pnl: Decimal, when_utc: datetime) -> None:
        d = when_utc.date()
        if self._symbol_daily_pnl_day != d:
            self._symbol_daily_pnl.clear()
            self._symbol_daily_pnl_day = d
        key = (symbol.upper(), d)
        self._symbol_daily_pnl[key] = self._symbol_daily_pnl.get(key, Decimal("0")) + pnl

    def _symbol_daily_loss_blocks(self, symbol: str, when_utc: datetime) -> tuple[bool, str]:
        if not self._symbol_daily_loss_enabled or self._symbol_daily_loss_limit <= 0:
            return False, ""
        pnl = self._symbol_daily_pnl_today(symbol, when_utc)
        if pnl <= -self._symbol_daily_loss_limit:
            return (
                True,
                f"pérdida diaria {float(pnl):.2f} >= límite {float(self._symbol_daily_loss_limit):.2f}",
            )
        return False, ""

    def _mtf_scalper_allows(self, direction: str, trends: dict[str, Trend]) -> bool:
        """Scalper: M1 a favor; M5/M15 coherentes; H4 no puede ir en contra del giro (evita sells en rally 4H)."""
        d = direction.upper()
        m1 = trends.get("M1", Trend.RANGE)
        m5 = trends.get("M5", Trend.RANGE)
        m15 = trends.get("M15", Trend.RANGE)
        h4 = trends.get("H4", Trend.RANGE)
        if d == "BUY":
            if m1 != Trend.UP:
                return False
            if m5 == Trend.DOWN:
                return False
            if m15 == Trend.DOWN:
                return False
            if self._scalper_require_h4_align and h4 == Trend.DOWN:
                return False
            return m5 in (Trend.UP, Trend.RANGE) and m15 in (Trend.UP, Trend.RANGE)
        if d == "SELL":
            if m1 != Trend.DOWN:
                return False
            if m5 == Trend.UP:
                return False
            if m15 == Trend.UP:
                return False
            if self._scalper_require_h4_align and h4 == Trend.UP:
                return False
            return m5 in (Trend.DOWN, Trend.RANGE) and m15 in (Trend.DOWN, Trend.RANGE)
        return False

    def _counter_htf_scalper_block(
        self,
        signal: Mapping[str, Any],
        trends: dict[str, Trend],
    ) -> tuple[bool, str]:
        """Evita scalper en contra de H1/H4 claros.

        v4.1: relajamos el bloqueo cuando H4 está en RANGE y los timeframes
        intermedios (M5+M15) ya están alineados con la dirección de la
        señal. En ese caso el contexto es un rango macro y la pierna de
        scalper en contra de H1 es una operativa válida (mean-reversion,
        retrocesos), siempre que setup/momentum lo soporten más adelante en
        la cadena de filtros.
        """
        if not self._block_scalper_counter_htf:
            return False, ""
        if bool(signal.get("range_fade")):
            return False, ""
        st = str(signal.get("setup_type", "")).upper()
        if "RANGE_FADE" in st:
            return False, ""
        d = str(signal.get("direction", "")).upper()
        h1 = trends.get("H1", Trend.RANGE)
        h4 = trends.get("H4", Trend.RANGE)
        m5 = trends.get("M5", Trend.RANGE)
        m15 = trends.get("M15", Trend.RANGE)

        # Excepción rango macro: H4 RANGE + M5/M15 alineados con la señal.
        # Sólo H1 yendo en contra no debe bloquear porque el contexto sigue
        # siendo lateral. Esto resuelve el caso "venta con H1/H4 alcista"
        # cuando H4 está claramente en RANGE.
        if h4 == Trend.RANGE:
            if d == "SELL" and m5 == Trend.DOWN and m15 == Trend.DOWN:
                return False, ""
            if d == "BUY" and m5 == Trend.UP and m15 == Trend.UP:
                return False, ""

        if d == "SELL" and (h1 == Trend.UP or h4 == Trend.UP):
            return True, "bloqueo HTF: venta con H1/H4 alcista"
        if d == "BUY" and (h1 == Trend.DOWN or h4 == Trend.DOWN):
            return True, "bloqueo HTF: compra con H1/H4 bajista"
        return False, ""

    def _detect_mode(
        self,
        *,
        tick: Tick,
        symbol: str,
        volatility: VolatilityRegime,
        trends: dict[str, Trend],
        direction: str,
    ) -> str:
        spread_now = float(tick.ask - tick.bid)
        spread_points = spread_now / max(self._pip_size(symbol), 1e-9)
        profile = self._profile_for_symbol(symbol)
        spread_high = float(profile.get("spread_high_points", self._spread_high_points))
        spread_extreme = float(profile.get("spread_extreme_points", self._spread_extreme_points))
        session_profile = self._session_filter.classify_profile(tick.time_utc)
        m1 = trends.get("M1", Trend.RANGE)
        m5 = trends.get("M5", Trend.RANGE)
        aligned_m1_m5 = (
            direction.upper() == "BUY" and m1 == Trend.UP and m5 == Trend.UP
        ) or (
            direction.upper() == "SELL" and m1 == Trend.DOWN and m5 == Trend.DOWN
        )
        vol_extreme = volatility == VolatilityRegime.EXTREME
        if self._lenient_extreme_volatility:
            vol_extreme = False
        if vol_extreme or spread_points >= spread_extreme:
            return "CONSERVATIVE"
        # Conservativo si el SÍMBOLO ACTUAL lleva 3+ pérdidas. Antes la racha
        # global aplicaba modo conservativo a todos los activos aunque uno
        # nunca hubiera operado.
        if self._risk.loss_streak_for(symbol) >= 3:
            return "CONSERVATIVE"
        preferred_sessions = profile.get(
            "preferred_sessions",
            ["LONDON", "NEW_YORK", "OVERLAP_LONDON_NY"],
        )
        if (
            session_profile in set(preferred_sessions)
            and aligned_m1_m5
            and spread_points <= spread_high
        ):
            return "AGGRESSIVE_CONTROLLED"
        return "NORMAL"

    def _adaptive_ml_threshold(self, mode: str) -> float:
        if mode == "CONSERVATIVE":
            return max(self._ml_threshold_floor, 0.44)
        if mode == "AGGRESSIVE_CONTROLLED":
            return max(self._ml_threshold_floor, 0.38)
        return max(self._ml_threshold_floor, 0.44)

    def _capital_multiplier(self) -> float:
        """Escala exposición según crecimiento/caída de equity vs base."""
        base_eq = float(self._config.get("equity", 10000.0))
        cur_eq = float(self._equity)
        if base_eq <= 0:
            return 1.0
        ratio = cur_eq / base_eq
        if ratio >= 1.5:
            return 1.30
        if ratio >= 1.2:
            return 1.15
        if ratio <= 0.8:
            return 0.75
        if ratio <= 0.9:
            return 0.85
        return 1.0

    @staticmethod
    def _edge_multiplier(effective_conf: float, signal: Mapping[str, Any], score: int) -> float:
        """Aproxima expectativa para subir/bajar riesgo por calidad estadística."""
        sl = max(float(signal.get("sl_pips", 0.0)), 1e-9)
        tp = max(float(signal.get("tp_pips", 0.0)), 1e-9)
        rr = tp / sl
        edge = (effective_conf * rr) - (1.0 - effective_conf)
        if edge >= 0.45 and score >= 60:
            return 1.20
        if edge >= 0.25 and score >= 50:
            return 1.08
        if edge <= 0.02:
            return 0.85
        return 1.0

    def _learn_from_closed_trades(self) -> None:
        if not self._perf_learning_enabled:
            return
        now_m = monotonic()
        if now_m - self._last_perf_learn_mono < self._perf_learning_interval_s:
            return
        self._last_perf_learn_mono = now_m
        try:
            rows = self._trade_repo.get_training_dataset(limit=self._perf_learning_lookback)
        except Exception as exc:
            logger.debug("perf-learning: no disponible (%s)", exc)
            return
        by_symbol: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            sym = str(r.get("symbol", "")).upper().strip()
            if not sym:
                continue
            by_symbol.setdefault(sym, []).append(r)
        new_whitelist: dict[str, set[str]] = {}
        new_thr_bias: dict[str, dict[str, float]] = {}
        new_score_bias: dict[str, dict[str, int]] = {}
        new_risk_mult: dict[str, dict[str, float]] = {}
        for sym, vals in by_symbol.items():
            if len(vals) < self._perf_learning_min_samples:
                continue
            wins = 0
            rs: list[float] = []
            bucket_stats: dict[str, dict[str, float]] = {}
            for row in vals:
                profit = float(row.get("profit") or 0.0)
                r_mult = row.get("r_multiple")
                if profit > 0:
                    wins += 1
                if r_mult is not None:
                    try:
                        rs.append(float(r_mult))
                    except Exception:
                        pass
                fx = row.get("features_json")
                if not isinstance(fx, dict):
                    continue
                setup_type = str(fx.get("setup_type", "UNKNOWN")).upper()
                regime = str(fx.get("volatility_regime", "UNKNOWN")).upper()
                hour_slot = int(fx.get("hour_slot", -1))
                if hour_slot < 0 or hour_slot > 23:
                    continue
                key = f"{setup_type}|{regime}|{hour_slot:02d}"
                st = bucket_stats.setdefault(key, {"n": 0.0, "wins": 0.0, "sum_r": 0.0})
                st["n"] += 1.0
                if profit > 0:
                    st["wins"] += 1.0
                if r_mult is not None:
                    try:
                        st["sum_r"] += float(r_mult)
                    except Exception:
                        pass
            n = max(1, len(vals))
            winrate = wins / n
            avg_r = (sum(rs) / len(rs)) if rs else 0.0
            expectancy = (winrate * max(avg_r, 0.0)) - ((1.0 - winrate) * 1.0)
            if expectancy >= 0.15 and winrate >= 0.52:
                self._adaptive_perf_bias[sym] = -0.03
                self._adaptive_risk_multiplier[sym] = 1.10
            elif expectancy <= -0.05 or winrate < 0.43:
                self._adaptive_perf_bias[sym] = 0.04
                self._adaptive_risk_multiplier[sym] = 0.82
            else:
                self._adaptive_perf_bias[sym] = 0.0
                self._adaptive_risk_multiplier[sym] = 1.0
            allowed: set[str] = set()
            sym_thr: dict[str, float] = {}
            sym_score: dict[str, int] = {}
            sym_risk: dict[str, float] = {}
            for key, st in bucket_stats.items():
                n_key = int(st["n"])
                if n_key < self._sniper_min_samples:
                    continue
                wr = st["wins"] / max(st["n"], 1.0)
                avg_r_key = st["sum_r"] / max(st["n"], 1.0)
                exp_key = (wr * max(avg_r_key, 0.0)) - ((1.0 - wr) * 1.0)
                if wr >= self._sniper_min_winrate and exp_key >= self._sniper_min_expectancy:
                    allowed.add(key)
                    sym_thr[key] = -0.02
                    sym_score[key] = 3
                    sym_risk[key] = 1.08
                elif wr < 0.42 or exp_key < -0.05:
                    sym_thr[key] = 0.03
                    sym_score[key] = -4
                    sym_risk[key] = 0.85
                else:
                    sym_thr[key] = 0.0
                    sym_score[key] = 0
                    sym_risk[key] = 1.0
            new_whitelist[sym] = allowed
            new_thr_bias[sym] = sym_thr
            new_score_bias[sym] = sym_score
            new_risk_mult[sym] = sym_risk
        self._sniper_whitelist = new_whitelist
        self._sniper_bucket_threshold_bias = new_thr_bias
        self._sniper_bucket_score_bias = new_score_bias
        self._sniper_bucket_risk_mult = new_risk_mult

    def _sniper_key(self, setup_type: str, regime: VolatilityRegime, when_utc: datetime) -> str:
        local = when_utc.astimezone(self._sniper_tz) if when_utc.tzinfo else when_utc.replace(tzinfo=timezone.utc).astimezone(self._sniper_tz)
        return f"{setup_type.upper()}|{regime.value.upper()}|{local.hour:02d}"

    def _sniper_blocks_dead_hour(self, when_utc: datetime) -> bool:
        return when_utc.hour in self._sniper_dead_hours_utc

    def _sniper_bucket_adjustments(self, symbol: str, key: str) -> tuple[float, int, float]:
        s = symbol.upper()
        thr = float(self._sniper_bucket_threshold_bias.get(s, {}).get(key, 0.0))
        score = int(self._sniper_bucket_score_bias.get(s, {}).get(key, 0))
        risk_mult = float(self._sniper_bucket_risk_mult.get(s, {}).get(key, 1.0))
        return thr, score, risk_mult

    def _register_online_setup_outcome(self, symbol: str, setup_type: str, r_multiple: float) -> None:
        s = symbol.upper()
        setup = setup_type.upper().strip() or "UNKNOWN"
        by_symbol = self._online_setup_perf.setdefault(s, {})
        st = by_symbol.setdefault(setup, {"n": 0.0, "wins": 0.0, "sum_r": 0.0})
        st["n"] += 1.0
        if r_multiple > 0:
            st["wins"] += 1.0
        st["sum_r"] += float(r_multiple)

    def _online_setup_adjustments(self, symbol: str, setup_type: str) -> tuple[float, int, float]:
        s = symbol.upper()
        setup = setup_type.upper().strip() or "UNKNOWN"
        st = self._online_setup_perf.get(s, {}).get(setup)
        if not st:
            return 0.0, 0, 1.0
        n = int(st.get("n", 0.0))
        if n < self._online_setup_min_samples:
            return 0.0, 0, 1.0
        wr = float(st.get("wins", 0.0)) / max(float(st.get("n", 1.0)), 1.0)
        avg_r = float(st.get("sum_r", 0.0)) / max(float(st.get("n", 1.0)), 1.0)
        expectancy = (wr * max(avg_r, 0.0)) - ((1.0 - wr) * 1.0)
        if expectancy <= self._online_setup_bad_expectancy:
            return 0.04, -5, 0.80
        if expectancy >= self._online_setup_good_expectancy and wr >= self._online_setup_good_winrate:
            return -0.02, 3, 1.08
        return 0.0, 0, 1.0

    def _spread_guard_blocks(self, symbol: str, spread_points: float) -> tuple[bool, str]:
        if not self._spread_guard_enabled:
            return False, ""
        profile = self._profile_for_symbol(symbol)
        abs_cap = float(profile.get("spread_extreme_points", self._spread_guard_max_points))
        if spread_points >= abs_cap:
            return True, f"spread abs alto ({spread_points:.1f}>={abs_cap:.1f})"
        state = self._ensure_symbol_state(symbol)
        spreads = state.get("spreads", [])
        if len(spreads) < max(8, self._spread_guard_lookback):
            return False, ""
        pip = max(self._pip_size(symbol), 1e-9)
        recent_pts = sorted((s / pip) for s in spreads[-self._spread_guard_lookback :])
        med = recent_pts[len(recent_pts) // 2] if recent_pts else spread_points
        if med <= 0:
            return False, ""
        if spread_points >= med * self._spread_guard_spike_mult:
            return True, f"spread spike ({spread_points:.1f} vs med {med:.1f})"
        return False, ""

    def _correlation_group_for_symbol(self, symbol: str) -> frozenset[str] | None:
        s = symbol.upper()
        for g in self._correlation_groups:
            if s in g:
                return g
        return None

    def _count_open_positions_in_group(self, group: frozenset[str]) -> int:
        return sum(1 for st in self._open_trades.values() if st.symbol.upper() in group)

    def _correlation_blocks_new_entry(self, symbol: str) -> tuple[bool, str]:
        if self._max_open_per_correlation_group <= 0 or not self._correlation_groups:
            return False, ""
        g = self._correlation_group_for_symbol(symbol)
        if g is None:
            return False, ""
        n = self._count_open_positions_in_group(g)
        if n >= self._max_open_per_correlation_group:
            return True, f"correlación: grupo {sorted(g)} ya tiene {n} abierta(s)"
        return False, ""

    def _count_open_positions_for_symbol(self, symbol: str) -> int:
        s = symbol.upper()
        return sum(1 for st in self._open_trades.values() if st.symbol.upper() == s)

    def _broker_open_count_for_symbol(self, symbol: str) -> int:
        """Posiciones abiertas en MT5 para el símbolo (alinea topes con el libro real del bróker)."""
        conn = self._broker_connector
        if not self._enforce_broker_position_caps or conn is None or not bool(getattr(conn, "initialized", False)):
            return 0
        try:
            return int(
                conn.count_open_positions_for_symbol(
                    symbol,
                    magic=int(self._order_magic),
                    filter_by_magic=bool(self._reconcile_filter_magic),
                ),
            )
        except Exception as exc:
            logger.debug("broker_open_count %s: %s", symbol, exc)
            return 0

    def _position_caps_block(
        self,
        symbol: str,
        *,
        pending_same_symbol: int = 0,
        pending_total_extra: int = 0,
    ) -> tuple[bool, str]:
        """Límites de posiciones abiertas (0 en config = sin tope)."""
        sym = symbol.upper()
        internal = self._count_open_positions_for_symbol(sym)
        broker_n = self._broker_open_count_for_symbol(sym)
        n_sym = max(internal, broker_n) + max(0, pending_same_symbol)
        n_tot = len(self._open_trades) + max(0, pending_total_extra)
        if self._max_open_positions_total > 0 and n_tot >= self._max_open_positions_total:
            return True, f"tope global de posiciones abiertas ({n_tot}/{self._max_open_positions_total})"
        if self._max_open_positions_per_symbol > 0 and n_sym >= self._max_open_positions_per_symbol:
            return True, f"tope en {sym} ({n_sym}/{self._max_open_positions_per_symbol} abiertas)"
        return False, ""

    def _open_trade_state_from_broker(self, row: dict[str, Any]) -> OpenTradeState | None:
        """Construye estado de gestión a partir de una fila MT5 reconciliada."""
        sym = str(row.get("symbol", "") or "").strip().upper()
        if not sym:
            return None
        pip = max(self._pip_size(sym), 1e-12)
        entry = float(row.get("price_open") or 0.0)
        sl = float(row.get("sl") or 0.0)
        tp = float(row.get("tp") or 0.0)
        direction = "BUY" if row.get("type_buy") else "SELL"
        if entry <= 0:
            return None
        prof = self._profile_for_symbol(sym)
        min_sl_floor = 8.0
        try:
            if prof.get("min_sl_pips") is not None:
                min_sl_floor = max(min_sl_floor, float(prof["min_sl_pips"]))
        except (TypeError, ValueError):
            pass
        if sl <= 0:
            risk_price = max(abs(entry) * 0.00025, pip * min_sl_floor)
        else:
            risk_price = abs(entry - sl)
        sl_pips = max(risk_price / pip, min_sl_floor)
        if tp > 0:
            tp_pips = max(abs(tp - entry) / pip, sl_pips * 1.05)
        else:
            tp_pips = sl_pips * 1.25
        sl_px = float(sl) if sl and float(sl) > 0 else None
        return OpenTradeState(
            ticket=str(row["ticket"]),
            symbol=sym,
            direction=direction,
            entry_price=entry,
            risk_price=risk_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            lot=float(row.get("volume") or 0.01),
            setup_type="BROKER_SYNC",
            sniper_key="",
            current_sl_price=sl_px,
        )

    def _reconcile_positions_with_broker(self) -> None:
        """Alinea `_open_trades` con posiciones reales en MT5 (magic/símbolos configurados)."""
        conn = self._broker_connector
        if conn is None or not bool(getattr(conn, "initialized", False)):
            return
        syms = {str(s).strip().upper() for s in self._symbols_trading if str(s).strip()}
        if not syms:
            return
        try:
            rows = conn.list_positions_for_reconcile(
                syms,
                magic=int(self._order_magic),
                filter_by_magic=bool(self._reconcile_filter_magic),
            )
        except Exception as exc:
            logger.warning("reconcile MT5: error listando posiciones (%s)", exc)
            return
        broker_tix = {str(r.get("ticket", "")) for r in rows}
        for tix in list(self._open_trades.keys()):
            st = self._open_trades[tix]
            if st.symbol.upper() not in syms:
                continue
            if tix.isdigit() and tix not in broker_tix:
                logger.info("[RECONCILE] ticket=%s ya no está en MT5; limpiando libro interno", tix)
                self._open_trades.pop(tix, None)
        for r in rows:
            tix = str(r.get("ticket", ""))
            if not tix or tix in self._open_trades:
                continue
            st = self._open_trade_state_from_broker(r)
            if st is not None:
                self._open_trades[tix] = st
                logger.info(
                    "[RECONCILE] posición MT5 incorporada ticket=%s %s %s lot=%.4f",
                    tix,
                    st.symbol,
                    st.direction,
                    st.lot,
                )

    def _maybe_reconcile_broker(self) -> None:
        if self._reconcile_interval_s <= 0:
            return
        now_m = monotonic()
        if (now_m - self._last_reconcile_mono) < self._reconcile_interval_s:
            return
        self._last_reconcile_mono = now_m
        self._reconcile_positions_with_broker()

    def _record_intraday_close_r(self, symbol: str, r_multiple: float, when_utc: datetime) -> None:
        if not self._intraday_kill_enabled:
            return
        s = symbol.upper()
        wt = when_utc if when_utc.tzinfo else when_utc.replace(tzinfo=timezone.utc)
        d = wt.date()
        if self._intraday_roll_day.get(s) != d:
            self._intraday_recent_r[s] = []
            self._intraday_roll_day[s] = d
        rs = self._intraday_recent_r.setdefault(s, [])
        rs.append(float(r_multiple))
        cap = max(self._intraday_kill_window, self._intraday_kill_min_trades + 4)
        if len(rs) > cap:
            del rs[0 : len(rs) - cap]

        # ChopGuard: arma cooldown si los últimos N trades son micro (|R|≈0).
        if self._chop_guard is not None and self._chop_guard.enabled:
            try:
                self._chop_guard.maybe_arm_micro_cooldown(s, rs, wt)
            except Exception as exc:
                logger.debug("[CHOP_GUARD] maybe_arm_micro_cooldown falló: %s", exc)

        # Fix #11: EARLY kill switch — pausa el símbolo el resto del día UTC
        # cuando con muy pocos trades el resultado es ya sangrante (wr<=10%
        # AND avgR<=-3.0R). En los logs nocturnos XAUUSD/NDX100/BTCUSD tenían
        # 0% wr con avgR -30R / -19R / -4.6R en 3-4 trades y el bot seguía
        # intentando entrar (con apex_anti_overtrade tapando el síntoma pero
        # no la causa raíz). Aquí lo cortamos al inicio.
        if (
            self._early_kill_enabled
            and len(rs) >= self._early_kill_min_trades
        ):
            wins = sum(1 for x in rs if x > 0)
            wr_d = wins / max(len(rs), 1)
            avg_r_d = sum(rs) / max(len(rs), 1)
            bad_wr_d = wr_d <= self._early_kill_bad_winrate
            bad_r_d = avg_r_d <= self._early_kill_bad_avg_r
            trig_early = (
                (bad_wr_d and bad_r_d)
                if self._early_kill_require_both
                else (bad_wr_d or bad_r_d)
            )
            if trig_early:
                # Pausa hasta el inicio del próximo día UTC.
                next_day = datetime(
                    d.year, d.month, d.day, tzinfo=timezone.utc
                ) + timedelta(days=1)
                cur_until = self._intraday_kill_until.get(s)
                if cur_until is None or cur_until < next_day:
                    self._intraday_kill_until[s] = next_day
                    logger.warning(
                        "[EARLY_KILL] symbol=%s pausa_hasta=%s trades=%d wr=%.0f%% avgR=%.2fR "
                        "(circuit breaker tras pérdida sangrante)",
                        s,
                        next_day.isoformat(),
                        len(rs),
                        wr_d * 100.0,
                        avg_r_d,
                    )

        if len(rs) < self._intraday_kill_min_trades:
            return
        window = rs[-self._intraday_kill_window :] if len(rs) >= self._intraday_kill_window else rs
        wins = sum(1 for x in window if x > 0)
        wr = wins / max(len(window), 1)
        avg_r = sum(window) / max(len(window), 1)
        bad_wr = wr < self._intraday_kill_bad_winrate
        bad_r = avg_r < self._intraday_kill_bad_avg_r
        trigger = (bad_wr and bad_r) if self._intraday_kill_require_both else (bad_wr or bad_r)
        if trigger:
            until = wt + timedelta(seconds=int(self._intraday_kill_cooldown_s))
            self._intraday_kill_until[s] = until
            logger.warning(
                "[INTRADAY_KILL] symbol=%s pausa_nuevas_entradas_hasta=%s ventana=%s winrate=%.2f avgR=%.3f",
                s,
                until.isoformat(),
                len(window),
                wr * 100.0,
                avg_r,
            )

    def _recent_winrate(self, symbol: str) -> float:
        rs = self._intraday_recent_r.get(symbol.upper(), [])
        if len(rs) < 4:
            return 0.5
        return sum(1 for x in rs if x > 0) / len(rs)

    def _count_trades_today(self, when_utc: datetime) -> int:
        d = when_utc.date()
        return sum(v for (_sym, dt), v in self._weekday_trade_counts.items() if dt == d)

    def _warn_ml_schema_mismatch(self) -> None:
        """Avisa si los PKL son v1 (10 features) y el pipeline usa v2 (25)."""
        qe = self._quant_engine
        if qe is None:
            return
        use_v2 = bool(
            (self._config.get("quant_ml") or {}).get("use_extended_features", True)
            or str((self._config.get("ml") or {}).get("feature_schema", "v2")).lower() == "v2"
        )
        if not use_v2:
            return
        symbols = self._config.get("symbols") or ["XAUUSD"]
        for sym in symbols:
            s = str(sym).upper()
            if not qe.has_model(s):
                continue
            pred = qe.predictor_for(s)
            n_exp = pred.expected_n_features()
            if n_exp is not None and n_exp < 20:
                logger.warning(
                    "[ML_SCHEMA] %s: modelo con %s features pero pipeline v2 (~25). "
                    "Inferencia en neutro hasta reentrenar: python -m core.train_model",
                    s,
                    n_exp,
                )

    def _maybe_run_daily_ml_optimization(self, when_utc: datetime) -> None:
        """Optimización institucional/Apex como máximo una vez por día UTC (no en cada tick).

        Doble guard:
            1. Fecha UTC actual (real, no la del tick): evita disparos durante warm-up
               cuando se procesan ticks históricos de múltiples días.
            2. Cooldown wallclock mínimo (12 h) por `monotonic()`: red de seguridad
               anti-loop en caso de cualquier reset del flag de fecha.
        """
        now_real = datetime.now(timezone.utc)
        d = now_real.date()
        if self._last_daily_opt_date == d:
            return
        now_m = monotonic()
        min_gap_s = 12 * 3600.0
        if (now_m - self._last_daily_opt_mono) < min_gap_s:
            return
        self._last_daily_opt_date = d
        self._last_daily_opt_mono = now_m
        if self._institutional_engine is not None and self._institutional_engine.enabled:
            try:
                self._institutional_engine.run_daily_optimization(now_real)
            except Exception as exc:
                logger.warning("run_daily_optimization: %s", exc)

    def _notify_ml_entry_opened(
        self, when_utc: datetime, symbol: str | None = None
    ) -> None:
        """Apertura real: anti-overtrade Apex y contadores institucionales.

        ``symbol`` permite que el guard anti-overtrade tracke por instrumento
        (Fix #10) y que las rachas/burst de un símbolo no bloqueen a otros.
        """
        notify = getattr(self._institutional_engine, "notify_entry_opened", None)
        if callable(notify):
            try:
                try:
                    notify(when_utc, symbol=symbol)
                except TypeError:
                    notify(when_utc)
            except Exception as exc:
                logger.debug("notify_entry_opened: %s", exc)

    def _smc_on_trade_opened(
        self,
        *,
        symbol: str,
        direction: str,
        open_time: datetime,
        trade_id: Any = None,
        signal: Mapping[str, Any] | None = None,
    ) -> None:
        """AURION X 4.1 — Vincula el trade recién abierto con el snapshot SMC
        vigente para que el self-learner pueda evaluar qué estructuras
        funcionaron al cierre. Soft-fail.
        """
        if getattr(self, "_smc_analyzer", None) is None or self._smc_repo is None:
            return
        try:
            snap = self._smc_analyzer.snapshot(symbol)
            if snap is None:
                return
            features = self._smc_analyzer.features(symbol, direction_hint=direction)
            used = {
                "primary_tf": snap.primary_timeframe,
                "macro_bias": snap.macro_bias.value,
                "institutional_bias": snap.institutional_bias.value,
                "operational_bias": snap.operational_bias.value,
                "per_tf_score": dict(snap.confluence.per_tf),
                "bos_by_tf": {
                    tf: (ctx.last_bos.to_dict() if ctx.last_bos else None)
                    for tf, ctx in snap.per_tf.items()
                },
                "choch_by_tf": {
                    tf: (ctx.last_choch.to_dict() if ctx.last_choch else None)
                    for tf, ctx in snap.per_tf.items()
                },
                "best_ob_by_tf": {
                    tf: (ctx.active_obs[0].to_dict() if ctx.active_obs else None)
                    for tf, ctx in snap.per_tf.items()
                },
                "best_fvg_by_tf": {
                    tf: (ctx.active_fvgs[0].to_dict() if ctx.active_fvgs else None)
                    for tf, ctx in snap.per_tf.items()
                },
            }
            link_id = self._smc_repo.link_trade_to_smc(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                open_time=open_time,
                snapshot=snap,
                features_smc=features,
                used_elements=used,
                extra={
                    "setup_type": (signal or {}).get("setup_type"),
                    "signal_confidence": float((signal or {}).get("confidence_base", 0.0) or 0.0),
                },
            )
            if link_id is not None:
                self._smc_link_id_by_symbol[symbol.upper()] = link_id
        except Exception as exc:
            logger.debug("[SMC] link_trade_to_smc falló: %s", exc)

    def _smc_on_trade_closed(
        self,
        *,
        symbol: str,
        close_time: datetime | None = None,
        r_multiple: float | None = None,
        mfe_r: float | None = None,
        mae_r: float | None = None,
        win: bool | None = None,
        trade_id: Any = None,
    ) -> None:
        """AURION X 4.1 — Cierra el loop de self-learning SMC. Soft-fail."""
        if getattr(self, "_smc_repo", None) is None:
            return
        try:
            link_id = self._smc_link_id_by_symbol.pop(symbol.upper(), None)
            self._smc_repo.update_trade_outcome(
                link_id=link_id,
                trade_id=trade_id,
                close_time=close_time,
                r_multiple=r_multiple,
                mfe_r=mfe_r,
                mae_r=mae_r,
                win=win,
            )
        except Exception as exc:
            logger.debug("[SMC] update_trade_outcome falló: %s", exc)

    def _intraday_kill_blocks_new_entry(self, symbol: str, when_utc: datetime) -> tuple[bool, str]:
        if not self._intraday_kill_enabled:
            return False, ""
        s = symbol.upper()
        until = self._intraday_kill_until.get(s)
        if until is None:
            return False, ""
        now = when_utc.astimezone(timezone.utc) if when_utc.tzinfo else when_utc.replace(tzinfo=timezone.utc)
        end = until.astimezone(timezone.utc) if until.tzinfo else until.replace(tzinfo=timezone.utc)
        if now < end:
            return True, f"kill-switch hasta {end.isoformat()}"
        return False, ""

    def _reload_ml_models(self, model_path: Path) -> None:
        loaded = TradingModel(artifact_path=model_path)
        loaded.load()
        self._predictor.set_model(loaded)
        self._load_per_symbol_ml_models()
        if self._institutional_engine is not None:
            self._institutional_engine.quant_engine._reload_models()
        elif self._quant_engine is not None:
            self._quant_engine._reload_models()
        self._ml_enabled = bool(self._predictor.model) or bool(self._predictor_by_symbol)

    def _run_ml_retrain(self, *, reason: str) -> dict[str, float] | None:
        """Entrena desde PostgreSQL y recarga el predictor. Devuelve métricas o None si falla."""
        ml_cfg = self._config.get("ml", {}) if isinstance(self._config, dict) else {}
        model_path_raw = str(ml_cfg.get("model_path", "D:/AURION_DATA/models/model.pkl")).strip()
        model_path = Path(model_path_raw) if model_path_raw else Path("D:/AURION_DATA/models/model.pkl")
        try:
            trainer = Trainer.from_ml_config(
                ml_cfg if isinstance(ml_cfg, dict) else {},
                output_dir=model_path.parent,
                repository=self._trade_repo,
            )
            art_ctx = ml_cfg.get("auto_retrain_dataset_context")
            if self._ml_walk_forward_enabled:
                _, metrics = trainer.train_walk_forward_from_repository(
                    limit=max(200, int(self._auto_retrain_limit)),
                    model_filename=model_path.name,
                    train_ratio=self._ml_walk_forward_train_ratio,
                    val_ratio=self._ml_walk_forward_val_ratio,
                    min_samples=self._ml_walk_forward_min_samples,
                    dataset_context=art_ctx,
                )
            elif self._auto_retrain_batch:
                _, metrics = trainer.retrain_in_batches(
                    total_limit=max(1000, int(self._auto_retrain_limit)),
                    batch_size=max(
                        200,
                        int(min(2000, self._auto_retrain_limit // 4 if self._auto_retrain_limit > 0 else 500)),
                    ),
                    model_filename=model_path.name,
                    dataset_context=art_ctx,
                )
            else:
                _, metrics = trainer.retrain_manual(
                    limit=max(200, int(self._auto_retrain_limit)),
                    model_filename=model_path.name,
                    dataset_context=art_ctx,
                )
            self._reload_ml_models(model_path)
            self._ml_closed_baseline = self._trade_repo.count_closed_trades()
            logger.info("ML retrain (%s) ok baseline_closed=%s model=%s metrics=%s", reason, self._ml_closed_baseline, model_path, metrics)
            return metrics
        except Exception as exc:
            logger.warning("ML retrain (%s) omitido/fallido: %s", reason, exc)
            return None

    def _wire_v4_hybrid(self) -> None:
        """Cablea los componentes AURION X 4.0 HÍBRIDO si el flag está activo.

        Lee ``config['v4_hybrid']`` y, para cada subcomponente con
        ``enabled: true``, instancia el módulo correspondiente y lo asocia a
        los engines de 3.0 vía sus métodos ``attach_v4_*``. Cada paso está
        protegido por try/except: si un componente falla, los demás siguen.

        Componentes:
            - gold_features      → ``self._v4_gold_provider``
            - patchtst           → ``self._v4_transformer`` (TransformerPredictor)
            - meta_stacker       → ``self._v4_meta_stacker``
            - quant_engine.attach_v4_hybrid(transformer, meta_stacker)
            - anomaly_autoencoder→ ``self._v4_anomaly_detector``
            - apex_engine.attach_v4_anomaly_detector(detector)
            - drl_exit (shadow)  → ``self._v4_drl_shadow``

        Idempotente: llamar de nuevo no recrea componentes ya inicializados
        (queda como mejora futura — por ahora reinicializa).
        """
        v4 = self._config.get("v4_hybrid") if isinstance(self._config, dict) else None
        if not isinstance(v4, dict) or not bool(v4.get("enabled", False)):
            logger.debug("v4_hybrid desactivado; skip wiring.")
            return

        symbol_focus = str(v4.get("symbol_focus", "XAUUSD")).upper()
        logger.info("[V4] cableando componentes AURION X 4.0 HIBRIDO (focus=%s)", symbol_focus)

        gold_cfg = v4.get("gold_features") or {}
        if isinstance(gold_cfg, dict) and bool(gold_cfg.get("enabled", False)):
            try:
                from ml.gold_specific_features import GoldFeatureProvider  # noqa: WPS433
                self._v4_gold_provider = GoldFeatureProvider(cfg=gold_cfg)
                logger.info("[V4] gold_features activo (schema=%s)", gold_cfg.get("feature_schema", "v3"))
            except Exception as exc:
                logger.warning("[V4] gold_features no se pudo inicializar: %s", exc)
                self._v4_gold_provider = None

        patchtst_cfg = v4.get("patchtst") or {}
        if isinstance(patchtst_cfg, dict) and bool(patchtst_cfg.get("enabled", False)):
            try:
                from ml.transformer_predictor import TransformerPredictor  # noqa: WPS433
                self._v4_transformer = TransformerPredictor(cfg=patchtst_cfg)
                logger.info("[V4] patchtst predictor instanciado (model=%s)",
                            patchtst_cfg.get("model_path"))
            except Exception as exc:
                logger.warning("[V4] patchtst no se pudo inicializar: %s", exc)
                self._v4_transformer = None

        ms_cfg = v4.get("meta_stacker") or {}
        if isinstance(ms_cfg, dict) and bool(ms_cfg.get("enabled", False)):
            try:
                from ml.meta_stacker import MetaStacker  # noqa: WPS433
                ms = MetaStacker(cfg=ms_cfg)
                self._v4_meta_stacker = ms
                logger.info("[V4] meta_stacker instanciado (fitted=%s)", ms.is_fitted)
            except Exception as exc:
                logger.warning("[V4] meta_stacker no se pudo inicializar: %s", exc)
                self._v4_meta_stacker = None

        if self._quant_engine is not None and hasattr(self._quant_engine, "attach_v4_hybrid"):
            try:
                self._quant_engine.attach_v4_hybrid(
                    transformer_predictor=self._v4_transformer,
                    meta_stacker=self._v4_meta_stacker,
                )
                logger.info("[V4] quant_engine.attach_v4_hybrid OK (tfmr=%s, stacker=%s)",
                            self._v4_transformer is not None,
                            self._v4_meta_stacker is not None)
            except Exception as exc:
                logger.warning("[V4] quant_engine.attach_v4_hybrid falló: %s", exc)

        anom_cfg = v4.get("anomaly_autoencoder") or {}
        if isinstance(anom_cfg, dict) and bool(anom_cfg.get("enabled", False)):
            try:
                from ml.anomaly_autoencoder import AnomalyDetector  # noqa: WPS433
                self._v4_anomaly_detector = AnomalyDetector(cfg=anom_cfg)
                logger.info("[V4] anomaly_autoencoder detector instanciado (model=%s)",
                            anom_cfg.get("model_path"))
            except Exception as exc:
                logger.warning("[V4] anomaly_autoencoder no se pudo inicializar: %s", exc)
                self._v4_anomaly_detector = None

        if self._institutional_engine is not None and hasattr(self._institutional_engine, "attach_v4_anomaly_detector"):
            try:
                self._institutional_engine.attach_v4_anomaly_detector(self._v4_anomaly_detector)
                logger.info("[V4] apex_engine.attach_v4_anomaly_detector OK (detector=%s)",
                            self._v4_anomaly_detector is not None)
            except Exception as exc:
                logger.warning("[V4] apex_engine.attach_v4_anomaly_detector falló: %s", exc)

        drl_cfg = v4.get("drl_exit") or {}
        if isinstance(drl_cfg, dict) and bool(drl_cfg.get("enabled", False)):
            try:
                from ml.drl_exit_shadow import DRLExitShadow  # noqa: WPS433
                self._v4_drl_shadow = DRLExitShadow(cfg=drl_cfg, symbol=symbol_focus)
                logger.info("[V4] drl_exit_shadow instanciado (shadow_mode=%s, model=%s)",
                            drl_cfg.get("shadow_mode", True),
                            drl_cfg.get("model_path"))
            except Exception as exc:
                logger.warning("[V4] drl_exit_shadow no se pudo inicializar: %s", exc)
                self._v4_drl_shadow = None

        logger.info(
            "[V4] cableado completo: gold=%s tfmr=%s stacker=%s anomaly=%s drl_shadow=%s",
            self._v4_gold_provider is not None,
            self._v4_transformer is not None,
            self._v4_meta_stacker is not None,
            self._v4_anomaly_detector is not None,
            self._v4_drl_shadow is not None,
        )

    def _wire_smc_layer(self) -> None:
        """AURION X 4.1 — Cablea la capa SMC institucional (XAUUSD-only).

        Defensivo: si el módulo ``market_structure`` no está disponible o la
        config tiene ``smc.enabled=false`` el resto del Orchestrator opera
        idéntico a antes. La capa nunca bloquea operaciones, sólo modula
        ``effective_confidence`` y ``setup_grade`` vía ``score_modifier``.
        """
        self._smc_analyzer = None
        self._smc_repo = None
        self._smc_decision_engine = None
        self._smc_link_id_by_symbol: dict[str, Any] = {}
        smc_cfg = self._config.get("smc") if isinstance(self._config, dict) else None
        if not isinstance(smc_cfg, dict) or not bool(smc_cfg.get("enabled", False)):
            logger.info("[SMC] desactivado (config smc.enabled=false).")
            return
        try:
            from market_structure import SMCAnalyzer, SMCAnalyzerConfig  # noqa: WPS433
        except Exception as exc:
            logger.warning("[SMC] paquete market_structure no disponible: %s", exc)
            return
        try:
            enabled_syms = tuple(
                str(s).upper() for s in (smc_cfg.get("enabled_symbols") or ["XAUUSD"])
            )
            tfs = tuple(str(t).upper() for t in (smc_cfg.get("timeframes") or ["M15", "M30", "H1", "H4", "D1", "W1"]))
            weights = dict(smc_cfg.get("weights") or {})
            fib_weights = dict(smc_cfg.get("fib_weights") or {})
            cfg = SMCAnalyzerConfig(
                enabled_symbols=enabled_syms,
                timeframes=tfs,
                primary_timeframe=str(smc_cfg.get("primary_timeframe", "M15")).upper(),
                cache_ttl_s=float(smc_cfg.get("cache_ttl_s", 8.0)),
                persist_snapshots=bool(smc_cfg.get("persist_snapshots", True)),
                component_weights={**({} if not weights else weights)},
                fib_weights={**({} if not fib_weights else fib_weights)},
                max_conf_delta=float(smc_cfg.get("max_conf_delta", 0.06)),
                grade_bump_threshold=float(smc_cfg.get("grade_bump_threshold", 0.70)),
                grade_demote_threshold=float(smc_cfg.get("grade_demote_threshold", 0.25)),
            )
            repo = None
            if cfg.persist_snapshots:
                try:
                    from analytics.smc_repository import SMCRepository  # noqa: WPS433
                    repo = SMCRepository()
                except Exception as exc:
                    logger.warning("[SMC] repositorio no disponible (%s); persistencia desactivada", exc)
            self._smc_repo = repo
            self._smc_analyzer = SMCAnalyzer(cfg, repository=repo)
            logger.info(
                "[SMC] capa institucional activa: symbols=%s tfs=%s primary=%s persist=%s",
                enabled_syms, tfs, cfg.primary_timeframe, cfg.persist_snapshots and repo is not None,
            )
            # AURION X 4.5 (11-jun-2026) — SMC Decision Engine (capa nueva).
            # Filtra setups mediocres por SMC score 0-100 + grade A+/A/B/C/X
            # + Premium/Discount + OB quality + FVG quality + MTF alignment.
            # Soft-fail: si la subseccion no existe o falla, el resto sigue.
            self._smc_decision_engine = None
            try:
                from ml.smc_decision_engine import SMCDecisionEngine  # noqa: WPS433
                dec_cfg = (smc_cfg.get("decision_engine") or {})
                if bool(dec_cfg.get("enabled", True)):
                    self._smc_decision_engine = SMCDecisionEngine.from_config(dec_cfg)
                    _h_rules = self._smc_decision_engine._heur_rules
                    _h_rules_s = "all" if _h_rules is None else f"{len(_h_rules)} custom"
                    logger.info(
                        "[SMC_DECISION] activo: min_score=%.0f require_sweep=%s "
                        "require_mtf=%s c_ml_override=%.2f heuristics=%s(w=%.2f,rules=%s)",
                        self._smc_decision_engine._min_score,
                        self._smc_decision_engine._require_sweep,
                        self._smc_decision_engine._require_mtf,
                        self._smc_decision_engine._c_grade_ml_override,
                        self._smc_decision_engine._heur_enabled,
                        self._smc_decision_engine._heur_weight,
                        _h_rules_s,
                    )
                else:
                    logger.info("[SMC_DECISION] desactivado (decision_engine.enabled=false)")
            except Exception as exc:
                logger.warning("[SMC_DECISION] no se pudo cargar: %s", exc)
                self._smc_decision_engine = None
        except Exception as exc:
            logger.warning("[SMC] inicialización falló: %s", exc)
            self._smc_analyzer = None
            self._smc_repo = None
            self._smc_decision_engine = None

    def _smc_bars_cb(self, symbol: str, tf: str):
        """Adaptador del MultiTimeframeEngine para el SMCAnalyzer.

        Devuelve ``(highs, lows, closes, volumes, times)`` desde la historia
        OHLC ya acumulada por el engine. Los times se reconstruyen a partir de
        ``get_ohlc_with_times`` (devuelve inicio de cada vela en UTC).

        Si el MTF no tiene aún suficiente historia (típico al arranque para
        TFs altos como H4/D1/W1), hace fallback a ``mt5_connector.get_rates``
        para hidratar el SMC desde el histórico de MT5. Esto permite que
        OB/BOS/CHOCH/FVG/Liquidity aparezcan en el dashboard y el EA desde el
        primer ciclo, no después de ~25 minutos de warm-up.
        """
        MIN_BARS_FOR_SMC = 60
        bars: tuple[list, list, list, list, list] = ([], [], [], [], [])
        try:
            times, highs, lows, closes = self._mtf.get_ohlc_with_times(symbol, tf)
            _, _, _, volumes = self._mtf.get_ohlcv_series(symbol, tf)
            bars = (list(highs), list(lows), list(closes), list(volumes), list(times))
        except Exception:
            bars = ([], [], [], [], [])
        if len(bars[2]) >= MIN_BARS_FOR_SMC:
            return bars
        # Fallback: pedir velas históricas directo a MetaTrader5.
        # Esto sólo se invoca durante warm-up (primeros minutos) o para TFs
        # altos (H4/D1/W1) donde el MTF aún no ha hidratado historia
        # suficiente. Soft-fail: si MT5 no responde devuelve lo que hay.
        try:
            cached = getattr(self, "_smc_rates_cache", None)
            if cached is None:
                self._smc_rates_cache = {}  # type: ignore[attr-defined]
                cached = self._smc_rates_cache
            ck = (symbol.upper(), tf)
            now_m = monotonic()
            entry = cached.get(ck)
            if entry is not None and (now_m - entry[0]) < 60.0:
                return entry[1]
            import MetaTrader5 as mt5  # type: ignore[import-not-found]
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
                "MN1": mt5.TIMEFRAME_MN1,
            }
            tf_const = tf_map.get(tf)
            if tf_const is None:
                return bars
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 400)
            if rates is None or len(rates) == 0:
                return bars
            ts_list: list = []
            h_list: list[float] = []
            l_list: list[float] = []
            c_list: list[float] = []
            v_list: list[float] = []
            for r in rates:
                try:
                    t = datetime.fromtimestamp(float(r["time"]), tz=timezone.utc)
                    ts_list.append(t)
                    h_list.append(float(r["high"]))
                    l_list.append(float(r["low"]))
                    c_list.append(float(r["close"]))
                    v_list.append(float(r["tick_volume"]))
                except Exception:
                    continue
            if len(c_list) >= MIN_BARS_FOR_SMC:
                result = (h_list, l_list, c_list, v_list, ts_list)
                cached[ck] = (now_m, result)
                return result
        except Exception as exc:
            logger.debug("[SMC] fallback MT5 rates %s %s falló: %s", symbol, tf, exc)
        return bars

    def _smc_analyze(self, symbol: str):
        """Calcula (o reusa cache) snapshot SMC del símbolo. Soft-fail."""
        if self._smc_analyzer is None:
            return None
        try:
            return self._smc_analyzer.analyze(symbol, self._smc_bars_cb)
        except Exception as exc:
            logger.debug("[SMC] analyze %s falló: %s", symbol, exc)
            return None

    def _write_smc_decision_cache(
        self, symbol: str, decision: Any,
        *, smart_targets: Any = None, smart_stops: Any = None,
        session_pools: Any = None, volume_class: str | None = None,
        entry_price: float | None = None,
    ) -> None:
        """AURION X 4.5/4.6 — escribe el ultimo SMCDecision + smart_targets +
        smart_stops + session_pools a ``runtime/smc_decisions/<SYM>.json``.

        Atomico (tmp + os.replace). El dashboard_server lo lee para exponerlo
        al panel MT5 sin tocar la BD ni crear tablas nuevas.
        """
        import json
        import os
        from pathlib import Path
        cache_dir = Path(__file__).resolve().parent.parent / "runtime" / "smc_decisions"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{symbol.upper()}.json"
        tmp = target.with_suffix(".json.tmp")
        payload: dict[str, Any] = {
            "symbol": symbol.upper(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": decision.to_dict() if hasattr(decision, "to_dict") else dict(decision),
        }
        if entry_price is not None:
            payload["entry_price"] = float(entry_price)
        if volume_class is not None:
            payload["volume_class"] = str(volume_class)
        if smart_targets is not None and hasattr(smart_targets, "to_dict"):
            payload["smart_targets"] = smart_targets.to_dict()
        if smart_stops is not None and hasattr(smart_stops, "to_dict"):
            payload["smart_stops"] = smart_stops.to_dict()
        if session_pools is not None:
            payload["session_pools"] = (
                session_pools if isinstance(session_pools, dict) else {}
            )
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, default=str)
        os.replace(tmp, target)

    # =====================================================================
    # AURION X 4.1 — XAU Knowledge Engine wiring (XAUUSD-only, soft, opcional)
    # =====================================================================
    def _wire_xau_knowledge(self) -> None:
        """Inicializa la capa de aprendizaje continuo XAUUSD.

        Defensivo y opcional: si ``xau_knowledge.enabled=false`` o el módulo
        no está disponible el resto del Orchestrator no cambia su comportamiento.
        El engine NO ejecuta órdenes ni gestiona riesgo; sólo observa y dispara
        reentrenamientos controlados del modelo XAUUSD aislado.
        """
        self._xau_knowledge = None  # type: ignore[attr-defined]
        self._xau_knowledge_last_bar_ts: dict[str, "datetime"] = {}
        ke_cfg = self._config.get("xau_knowledge") if isinstance(self._config, dict) else None
        if not isinstance(ke_cfg, dict) or not bool(ke_cfg.get("enabled", False)):
            logger.info("[KNOWLEDGE] desactivado (xau_knowledge.enabled=false).")
            return
        try:
            from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine
            from ml.xau_isolated_retrainer import (
                build_xauusd_dataset_provider,
                build_isolated_retrain_callback,
            )
        except Exception as exc:
            logger.warning("[KNOWLEDGE] módulos no disponibles: %s", exc)
            return
        try:
            from pathlib import Path as _Path
            model_path = _Path(str(ke_cfg.get("model_path", "D:/AURION_DATA/models/model_XAUUSD.pkl")))
            cfg = XAUKnowledgeConfig(
                enabled=True,
                symbol=str(ke_cfg.get("symbol", "XAUUSD")).upper(),
                observe_timeframes=tuple(
                    str(t).upper()
                    for t in (ke_cfg.get("observe_timeframes") or ["M15", "M30", "H1", "H4", "D1"])
                ),
                market_memory_flush_size=int(ke_cfg.get("market_memory_flush_size", 64)),
                retrain_closed_trades_n=int(ke_cfg.get("retrain_closed_trades_n", 10)),
                retrain_end_of_day_utc_hour=int(ke_cfg.get("retrain_end_of_day_utc_hour", 23)),
                retrain_regime_change_min_duration_min=int(
                    ke_cfg.get("retrain_regime_change_min_duration_min", 30)
                ),
                retrain_degradation_winrate_drop=float(
                    ke_cfg.get("retrain_degradation_winrate_drop", 0.10)
                ),
                retrain_degradation_window_trades=int(
                    ke_cfg.get("retrain_degradation_window_trades", 20)
                ),
                retrain_min_interval_minutes=int(ke_cfg.get("retrain_min_interval_minutes", 60)),
                model_path=model_path,
                walk_forward_folds=int(ke_cfg.get("walk_forward_folds", 5)),
            )
            tw_raw = ke_cfg.get("time_weights")
            if isinstance(tw_raw, dict) and tw_raw:
                try:
                    cfg.time_weights = {
                        k: {"max_age_days": float(v["max_age_days"]), "weight": float(v["weight"])}
                        for k, v in tw_raw.items()
                        if isinstance(v, dict) and "max_age_days" in v and "weight" in v
                    }
                except Exception as exc:
                    logger.debug("[KNOWLEDGE] time_weights inválidos, usando defaults: %s", exc)
            dataset_provider = build_xauusd_dataset_provider()
            retrain_cb = build_isolated_retrain_callback(
                model_path=cfg.model_path,
                walk_forward_folds=cfg.walk_forward_folds,
            )
            self._xau_knowledge = XAUKnowledgeEngine(  # type: ignore[attr-defined]
                cfg,
                dataset_provider=dataset_provider,
                retrain_callback=retrain_cb,
            )
            logger.info(
                "[KNOWLEDGE] activa: symbol=%s tfs=%s model=%s triggers=closed>=%d/EOD@%dh/regime>=%dm/drop>=%.2f",
                cfg.symbol, cfg.observe_timeframes, cfg.model_path,
                cfg.retrain_closed_trades_n, cfg.retrain_end_of_day_utc_hour,
                cfg.retrain_regime_change_min_duration_min,
                cfg.retrain_degradation_winrate_drop,
            )
        except Exception as exc:
            logger.warning("[KNOWLEDGE] inicialización falló: %s", exc)
            self._xau_knowledge = None  # type: ignore[attr-defined]

    def _knowledge_observe_bar(self, symbol: str, timeframe: str, *, smc_snapshot=None) -> None:
        """Llama a ``observe_bar_close`` cuando se detecta una vela nueva en
        un TF supervisado. Sólo XAUUSD."""
        ke = getattr(self, "_xau_knowledge", None)
        if ke is None or not ke.enabled:
            return
        if symbol.upper() != "XAUUSD":
            return
        try:
            times, highs, lows, closes = self._mtf.get_ohlc_with_times(symbol, timeframe)
            if not times or not closes:
                return
            last_ts = times[-1]
            key = f"{symbol}:{timeframe}"
            prev_ts = self._xau_knowledge_last_bar_ts.get(key)
            if prev_ts is not None and prev_ts == last_ts:
                return  # vela ya observada
            self._xau_knowledge_last_bar_ts[key] = last_ts

            opens = closes[:-1] + [closes[-1]] if len(closes) >= 1 else []
            o = float(opens[-2]) if len(closes) >= 2 else float(closes[-1])
            h = float(highs[-1]); l_ = float(lows[-1]); c = float(closes[-1])
            atr = None
            try:
                from market_structure.swing_detector import _wilder_atr  # type: ignore
                atr = _wilder_atr(highs, lows, closes, period=14)
            except Exception:
                pass
            range_pct = (h - l_) / max(c, 1e-9)
            regime = None
            try:
                regime = self._regime_engine.classify(symbol, timeframe)  # type: ignore[attr-defined]
            except Exception:
                regime = None
            trend = None
            try:
                trend_obj = self._mtf.get_trend(symbol, timeframe)
                trend = getattr(trend_obj, "name", None) or (str(trend_obj) if trend_obj else None)
            except Exception:
                trend = None
            ke.observe_bar_close(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=last_ts,
                ohlc=(o, h, l_, c),
                regime=str(regime).upper() if regime else None,
                trend=str(trend).upper() if trend else None,
                range_pct=range_pct,
                atr=atr,
            )
            # Si tenemos snapshot SMC actual, propagamos eventos clave
            if smc_snapshot is not None:
                try:
                    self._knowledge_propagate_smc(symbol, timeframe, smc_snapshot)
                except Exception as exc:
                    logger.debug("[KNOWLEDGE] propagate smc falló: %s", exc)
            # Evaluar reentrenamiento (no bloquea: el cb hace su trabajo o no)
            try:
                ke.trigger_retrain_if_needed()
            except Exception as exc:
                logger.debug("[KNOWLEDGE] retrain check falló: %s", exc)
        except Exception as exc:
            logger.debug("[KNOWLEDGE] observe_bar falló: %s", exc)

    def _knowledge_propagate_smc(self, symbol: str, timeframe: str, snapshot) -> list[str]:
        """Vuelca eventos SMC del snapshot al knowledge engine y devuelve
        IDs persistidos para vincular al trade si se abre."""
        ke = getattr(self, "_xau_knowledge", None)
        if ke is None:
            return []
        ids: list[str] = []
        try:
            tf_blocks = getattr(snapshot, "timeframes", {}) or {}
            tf_block = tf_blocks.get(timeframe)
            if tf_block is None:
                return []
            now = self._now_utc() if hasattr(self, "_now_utc") else __import__("datetime").datetime.utcnow()
            for ev in (getattr(tf_block, "bos", None) or []):
                eid = ke.observe_smc_event(
                    event_kind="BOS",
                    timeframe=timeframe,
                    timestamp=getattr(ev, "timestamp", now) or now,
                    direction=getattr(getattr(ev, "direction", None), "name", None),
                    strength=getattr(ev, "bos_score", None),
                    quality=getattr(ev, "bos_quality", None),
                    symbol=symbol,
                )
                if eid:
                    ids.append(eid)
            for ev in (getattr(tf_block, "choch", None) or []):
                eid = ke.observe_smc_event(
                    event_kind="CHOCH",
                    timeframe=timeframe,
                    timestamp=getattr(ev, "timestamp", now) or now,
                    direction=getattr(getattr(ev, "direction", None), "name", None),
                    strength=getattr(ev, "choch_score", None),
                    quality=getattr(ev, "choch_quality", None),
                    symbol=symbol,
                )
                if eid:
                    ids.append(eid)
            for ob in (getattr(tf_block, "order_blocks", None) or []):
                eid = ke.observe_smc_event(
                    event_kind="OB",
                    timeframe=timeframe,
                    timestamp=getattr(ob, "timestamp", now) or now,
                    direction=getattr(getattr(ob, "direction", None), "name", None),
                    strength=getattr(ob, "ob_strength", None),
                    quality=getattr(ob, "ob_quality", None),
                    symbol=symbol,
                )
                if eid:
                    ids.append(eid)
            for fvg in (getattr(tf_block, "fvgs", None) or []):
                eid = ke.observe_smc_event(
                    event_kind="FVG",
                    timeframe=timeframe,
                    timestamp=getattr(fvg, "timestamp", now) or now,
                    direction=getattr(getattr(fvg, "direction", None), "name", None),
                    strength=getattr(fvg, "fvg_strength", None),
                    symbol=symbol,
                )
                if eid:
                    ids.append(eid)
            # Liquidity sweeps
            for sw in (getattr(tf_block, "sweeps", None) or []):
                eid = ke.observe_smc_event(
                    event_kind="SWEEP",
                    timeframe=timeframe,
                    timestamp=getattr(sw, "timestamp", now) or now,
                    strength=getattr(sw, "sweep_strength", None),
                    symbol=symbol,
                )
                if eid:
                    ids.append(eid)
            # Patrones combinados (frecuencia)
            present: list[str] = []
            if getattr(tf_block, "order_blocks", None):
                present.append("OB")
            if getattr(tf_block, "fvgs", None):
                present.append("FVG")
            if getattr(tf_block, "choch", None):
                present.append("CHOCH")
            if getattr(tf_block, "bos", None):
                present.append("BOS")
            if getattr(tf_block, "sweeps", None):
                present.append("SWEEP")
            if len(present) >= 2:
                ke.register_pattern_combo(
                    names=present,
                    timeframe=timeframe,
                    direction=None,
                    when_utc=now,
                    symbol=symbol,
                )
        except Exception as exc:
            logger.debug("[KNOWLEDGE] propagate snapshot falló: %s", exc)
        return ids

    def _bootstrap_ml_from_database(self) -> None:
        """Entrenamiento inicial al arrancar usando histórico en PostgreSQL."""
        if not self._bootstrap_ml_on_start:
            return
        ml_cfg = self._config.get("ml", {}) if isinstance(self._config, dict) else {}
        if not bool(ml_cfg.get("enabled", True)):
            return
        model_path = self._ml_model_path
        if model_path.is_file() and not self._bootstrap_ml_force_on_start:
            try:
                self._ml_closed_baseline = self._trade_repo.count_closed_trades()
            except Exception:
                self._ml_closed_baseline = 0
            logger.info(
                "bootstrap ML: modelo existente %s (baseline_closed=%s)",
                model_path,
                self._ml_closed_baseline,
            )
            return
        try:
            closed = self._trade_repo.count_closed_trades()
        except Exception as exc:
            logger.warning("bootstrap ML: sin acceso a BD (%s)", exc)
            return
        if closed < self._bootstrap_ml_min_closed:
            logger.warning(
                "bootstrap ML: trades cerrados=%s < mínimo %s",
                closed,
                self._bootstrap_ml_min_closed,
            )
            return
        metrics = self._run_ml_retrain(reason="bootstrap")
        if metrics is not None:
            self._auto_retrain_last_mono = monotonic()

    def _maybe_auto_retrain_ml(self) -> None:
        if self._institutional_engine is not None and self._institutional_engine.enabled:
            return
        if self._quant_engine is not None and self._quant_engine.enabled:
            return
        if not self._auto_retrain_enabled:
            return
        now_m = monotonic()
        if (now_m - self._auto_retrain_last_mono) < self._auto_retrain_min_interval_s:
            return
        try:
            closed_db = self._trade_repo.count_closed_trades()
        except Exception:
            closed_db = self._closed_trades_counter
        new_since = max(0, closed_db - self._ml_closed_baseline)
        if new_since < self._auto_retrain_every_closed:
            return
        if closed_db < self._auto_retrain_min_closed and self._ml_closed_baseline == 0:
            return
        metrics = self._run_ml_retrain(reason="auto")
        if metrics is not None:
            self._auto_retrain_last_mono = now_m

    def _profile_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Devuelve perfil de calibración por activo (si existe)."""
        return dict(self._asset_profiles.get(symbol.upper(), {}))

    def _dynamic_exit_cfg_for_symbol(self, symbol: str) -> dict[str, Any]:
        """``execution.dynamic_exit`` fusionado con ``asset_profiles.<SYM>.dynamic_exit`` (si existe)."""
        base = dict(getattr(self, "_dynamic_exit_cfg", {}) or {})
        prof = self._profile_for_symbol(symbol)
        ov = prof.get("dynamic_exit")
        return merge_dynamic_exit_cfg(base, ov if isinstance(ov, dict) else None)

    def _predictor_for_symbol(self, symbol: str) -> Predictor:
        """Predictor específico del activo o el global."""
        return self._predictor_by_symbol.get(symbol.upper(), self._predictor)

    def _load_per_symbol_ml_models(self) -> None:
        """Carga PKL opcionales desde ``asset_profiles.<SYM>.ml_model_path``."""
        self._predictor_by_symbol.clear()
        profiles = self._asset_profiles
        if not isinstance(profiles, dict):
            return
        for sym_key, prof in profiles.items():
            if not isinstance(prof, dict):
                continue
            raw = prof.get("ml_model_path")
            if raw is None or str(raw).strip() == "":
                continue
            p = Path(str(raw).strip())
            s = str(sym_key).strip().upper()
            if not p.is_file():
                logger.warning("ML por símbolo %s: archivo inexistente %s", s, p)
                continue
            try:
                model = TradingModel(artifact_path=p)
                model.load()
                pred = Predictor()
                pred.set_model(model)
                self._predictor_by_symbol[s] = pred
                logger.info(
                    "ML por símbolo %s cargado desde %s (n_features=%s)",
                    s,
                    p,
                    model.n_features_expected(),
                )
            except Exception as exc:
                logger.warning("ML por símbolo %s omitido: %s", s, exc)

    def _scalper_engine_for_symbol(self, symbol: str) -> ScalperEngine:
        s = symbol.upper()
        if s in self._scalper_by_symbol:
            return self._scalper_by_symbol[s]
        profile = self._profile_for_symbol(s)
        strat_scalper = dict((self._config.get("strategy", {}) or {}).get("scalper", {}) or {})
        prof_scalper = profile.get("scalper", {})
        if isinstance(prof_scalper, dict) and prof_scalper:
            strat_scalper = {**strat_scalper, **prof_scalper}
        rq_root = self._config.get("range_quality")
        prof_rq = profile.get("range_quality")
        if isinstance(rq_root, dict):
            strat_scalper["range_quality"] = {**rq_root, **(prof_rq if isinstance(prof_rq, dict) else {})}
        elif isinstance(prof_rq, dict):
            strat_scalper["range_quality"] = prof_rq
        if strat_scalper:
            exec_cfg = self._config.get("execution") or {}
            strat_scalper["skip_scalper_extreme_m5_range"] = bool(
                exec_cfg.get("skip_scalper_extreme_m5_range", False)
            )
            strat_scalper["scalper_extreme_m5_min_confidence"] = float(
                exec_cfg.get("scalper_extreme_m5_min_confidence", 0.88)
            )
            eng = ScalperEngine(strat_scalper)
            self._scalper_by_symbol[s] = eng
            return eng
        self._scalper_by_symbol[s] = self._scalper
        return self._scalper

    def _pre_entry_momentum_boost(self, symbol: str, direction: str) -> int:
        """Bonus de score para decisiones más rápidas con desplazamiento real."""
        score = self._continuation_strength(symbol, direction)
        if score >= 0.75:
            return 8
        if score >= 0.60:
            return 5
        if score >= 0.50:
            return 3
        return 0

    def _is_ny_turbo_window(self, symbol: str, when_utc: datetime) -> bool:
        if not self._ny_turbo_enabled or symbol.upper() not in self._ny_turbo_symbols:
            return False
        local = when_utc.astimezone(ZoneInfo("America/New_York")) if when_utc.tzinfo else when_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York"))
        t = local.timetz().replace(tzinfo=None)
        sym_u = symbol.upper()
        win = self._ny_turbo_symbol_windows.get(sym_u)
        start_t, end_t = win if win is not None else (self._ny_turbo_start, self._ny_turbo_end)
        if start_t <= end_t:
            return start_t <= t <= end_t
        return t >= start_t or t <= end_t

    def _learn_from_decision(
        self,
        *,
        symbol: str,
        blocked_reason: str | None,
        strong_impulse_entry: bool,
        continuation_now: float,
        effective_conf: float,
        threshold: float,
    ) -> None:
        s = symbol.upper()
        bias = float(self._adaptive_entry_bias.get(s, 0.0))
        # Aprende de "errores de omisión": si bloquea impulsos fuertes repetidamente, se vuelve más permisivo.
        if blocked_reason in {"low_confidence", "validator"} and strong_impulse_entry and continuation_now >= 0.62:
            bias -= 0.015
        # Si aun con impulso la confianza está muy por debajo, vuelve a endurecer ligeramente.
        if blocked_reason == "low_confidence" and effective_conf < max(self._ml_threshold_floor, threshold - 0.10):
            bias += 0.01
        self._adaptive_entry_bias[s] = max(-0.10, min(0.06, bias))

    def _publish_chart_hint(
        self,
        *,
        symbol: str,
        direction: str,
        confidence: float,
        score: int,
        rr: float,
        stage: str,
        setup_type: str,
        when_utc: datetime,
    ) -> None:
        """Exporta dirección live para dibujar flecha en chart (puente Python→MQL5)."""
        if (not self._chart_hints_enabled) or confidence < self._chart_hints_min_conf:
            return
        min_sc = int(self._chart_hints_min_score)
        if min_sc > 0 and int(score) < min_sc:
            return
        try:
            p = self._chart_hints_path
            p.parent.mkdir(parents=True, exist_ok=True)
            sym = symbol.upper()
            rd = max(0, min(int(self._chart_hints_conf_round_decimals), 8))
            conf_key = round(float(confidence), rd) if rd > 0 else float(confidence)
            when_iso = when_utc.astimezone(timezone.utc).isoformat()
            dedupe_tail = self._m1_candle_id(when_utc) if self._chart_hints_dedupe_by_m1_bar else when_iso
            row = [
                sym,
                direction.upper(),
                f"{confidence:.4f}",
                str(int(score)),
                f"{max(rr, 0.0):.4f}",
                stage.upper(),
                setup_type,
                when_iso,
            ]
            payload = (
                row[1],
                f"{conf_key:.4f}",
                row[3],
                row[4],
                row[5],
                row[6],
                dedupe_tail,
            )
            if self._chart_hints_last_payload.get(sym) == payload:
                return
            self._chart_hints_last_payload[sym] = payload
            self._chart_hints_rows_cache[sym] = row
            now_m = monotonic()
            flush_interval_s = max(0.0, self._chart_hints_write_interval_ms / 1000.0)
            if (now_m - self._chart_hints_last_flush_mono) < flush_interval_s:
                return
            tmp = p.with_suffix(p.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8", newline="") as f:
                wr = csv.writer(f)
                for s in sorted(self._chart_hints_rows_cache.keys()):
                    wr.writerow(self._chart_hints_rows_cache[s])
            tmp.replace(p)
            self._chart_hints_last_flush_mono = now_m
        except Exception as exc:
            logger.debug("chart-hints publish fallo: %s", exc)

    def _compute_entry_score(
        self,
        *,
        direction: str,
        trends: dict[str, Trend],
        structure: StructureResult,
        volatility: VolatilityRegime,
        spread_points: float,
        symbol: str,
        signal: Mapping[str, Any],
        state: dict[str, list[float]],
    ) -> tuple[int, dict[str, int]]:
        parts: dict[str, int] = {}
        d = direction.upper()
        setup_type = str(signal.get("setup_type", "")).upper()
        is_range_fade = (
            bool(signal.get("range_fade"))
            or "RANGE_FADE" in setup_type
            or "RANGE_STRUCTURE" in setup_type
        )
        lateral_mtf = trends.get("M5", Trend.RANGE) == Trend.RANGE and trends.get("M15", Trend.RANGE) == Trend.RANGE
        m1 = trends.get("M1", Trend.RANGE)
        m5 = trends.get("M5", Trend.RANGE)

        if not is_range_fade:
            if (d == "BUY" and m1 == Trend.UP) or (d == "SELL" and m1 == Trend.DOWN):
                parts["m1_aligned"] = 20
            if (d == "BUY" and m5 == Trend.UP) or (d == "SELL" and m5 == Trend.DOWN):
                parts["m5_aligned"] = 20
        else:
            parts["range_fade_engine"] = 15
            if lateral_mtf:
                parts["lateral_mtf_double_range"] = 14
            if (d == "BUY" and m1 == Trend.DOWN) or (d == "SELL" and m1 == Trend.UP):
                parts["lateral_exhaustion_m1"] = 12
            elif (d == "BUY" and m1 == Trend.UP) or (d == "SELL" and m1 == Trend.DOWN):
                parts["m1_aligned"] = 14
            if (d == "BUY" and m5 == Trend.UP) or (d == "SELL" and m5 == Trend.DOWN):
                parts["m5_aligned"] = 14
            elif lateral_mtf:
                parts["m5_range_context"] = 10

        if structure.trend != Trend.RANGE and len(structure.key_levels) >= 3:
            parts["clean_structure"] = 15
        elif structure.trend == Trend.RANGE and structure.range_width_pct is not None:
            rw = float(structure.range_width_pct)
            if 0.00004 <= rw <= 0.007:
                parts["defined_range_box"] = 12
        if (
            is_range_fade
            and structure.position_in_range is not None
            and structure.trend == Trend.RANGE
        ):
            pin = float(structure.position_in_range)
            if d == "BUY" and pin <= 0.28:
                parts["at_range_support"] = 12
            elif d == "SELL" and pin >= 0.72:
                parts["at_range_resistance"] = 12

        vols = state.get("vols", [])
        if len(vols) >= 20:
            recent = vols[-20:]
            current = recent[-1]
            baseline = sorted(recent[:-1])[len(recent[:-1]) // 2] if len(recent) > 1 else current
            if current >= baseline * 1.15:
                parts["volume_surge_at_entry"] = 15
        if bool(signal.get("pullback", False)):
            parts["absorption"] = 10
            if not is_range_fade:
                h4 = trends.get("H4", Trend.RANGE)
                if d == "BUY" and h4 == Trend.UP:
                    parts["htf_pullback_stack"] = 8
                elif d == "SELL" and h4 == Trend.DOWN:
                    parts["htf_pullback_stack"] = 8
        if "BREAKOUT" in setup_type:
            parts["valid_breakout"] = 10
        if bool(signal.get("liquidity_sweep", False)):
            parts["liquidity_swept"] = 10
        if is_range_fade:
            rq_s = float(signal.get("range_quality_score", 0.0))
            if rq_s >= 68.0:
                parts["range_quality_pro"] = 16
            elif rq_s >= 55.0:
                parts["range_quality_ok"] = 10
            elif rq_s > 0.0 and rq_s < 46.0:
                parts["penalty_range_quality_weak"] = -10
            bo_p = float(signal.get("range_breakout_probability", 0.0))
            if bo_p >= 68.0:
                parts["penalty_range_near_breakout"] = -8
        volume_score = float(signal.get("volume_score", 0.0))
        volume_ratio = float(signal.get("volume_ratio", 0.0))
        if volume_score >= 0.75 and volume_ratio >= 1.35:
            parts["volume_surge_boost"] = 12
        elif volume_score >= 0.55 and volume_ratio >= 1.10:
            parts["healthy_volume"] = 6
        elif volume_ratio < 0.90 and not is_range_fade:
            parts["penalty_low_volume"] = -8
        elif volume_ratio < 0.82 and is_range_fade:
            parts["penalty_low_volume"] = -4

        if volatility == VolatilityRegime.HIGH:
            parts["penalty_high_vol"] = -5 if is_range_fade else -7
        profile = self._profile_for_symbol(symbol)
        spread_high = float(profile.get("spread_high_points", self._spread_high_points))
        if spread_points > spread_high:
            parts["penalty_high_spread"] = -8 if is_range_fade else -10
        m15 = trends.get("M15", Trend.RANGE)
        h4 = trends.get("H4", Trend.RANGE)
        d1 = trends.get("D1", Trend.RANGE)
        if (d == "BUY" and m15 == Trend.DOWN) or (d == "SELL" and m15 == Trend.UP):
            if is_range_fade or lateral_mtf:
                parts["penalty_m15_conflict"] = -3
            else:
                parts["penalty_m15_conflict"] = -10
        if not is_range_fade:
            if d == "BUY" and h4 == Trend.UP and d1 in (Trend.UP, Trend.RANGE):
                parts["macro_h4_d1_bull"] = 10
            elif d == "SELL" and h4 == Trend.DOWN and d1 in (Trend.DOWN, Trend.RANGE):
                parts["macro_h4_d1_bear"] = 10
            elif (d == "BUY" and h4 == Trend.DOWN) or (d == "SELL" and h4 == Trend.UP):
                parts["penalty_macro_h4_fight"] = -12
        if bool(signal.get("macro_news_near", False)):
            parts["penalty_news"] = -20
        momentum_bonus = self._pre_entry_momentum_boost(symbol, d)
        if momentum_bonus > 0:
            parts["real_momentum_boost"] = momentum_bonus
        if self._edge_fusion_enabled and not is_range_fade:
            h1 = trends.get("H1", Trend.RANGE)
            if volatility in (VolatilityRegime.NORMAL, VolatilityRegime.HIGH):
                stack_buy = (
                    d == "BUY"
                    and m1 == Trend.UP
                    and m5 == Trend.UP
                    and m15 == Trend.UP
                    and h1 == Trend.UP
                )
                stack_sell = (
                    d == "SELL"
                    and m1 == Trend.DOWN
                    and m5 == Trend.DOWN
                    and m15 == Trend.DOWN
                    and h1 == Trend.DOWN
                )
                if stack_buy or stack_sell:
                    parts["edge_fusion_mtf_stack"] = max(0, int(self._edge_fusion_mtf_stack_bonus))
        return sum(parts.values()), parts

    def _maybe_refresh_equity_from_broker(self) -> None:
        """Actualiza equity desde MT5 para sizing/riesgo alineados con la cuenta real."""
        if not self._use_live_equity or self._broker_connector is None:
            return
        try:
            eq = self._broker_connector.account_equity()
        except Exception:
            return
        if eq is not None and eq > 0:
            self._equity = Decimal(str(eq))

    def _build_market_state(self, symbol: str, tick: Tick, structure: StructureResult) -> dict[str, Any]:
        state = self._ensure_symbol_state(symbol)
        drawdown = float((self._risk._current_drawdown_pct(self._equity) / Decimal("100")) * Decimal("-1"))
        if structure.trend == Trend.UP:
            direction_hint = 1
        elif structure.trend == Trend.DOWN:
            direction_hint = -1
        elif structure.trend == Trend.RANGE and structure.position_in_range is not None:
            pin = float(structure.position_in_range)
            if pin <= 0.35:
                direction_hint = 1
            elif pin >= 0.65:
                direction_hint = -1
            else:
                direction_hint = 0
        else:
            direction_hint = 0
        mh, ml, mc = self._mtf.get_ohlc_series(symbol, "M1")
        if len(mc) >= 10:
            closes_m1 = mc
            highs_m1 = mh
            lows_m1 = ml
        else:
            closes_m1 = list(state["closes"])
            highs_m1 = list(state["highs"])
            lows_m1 = list(state["lows"])
        return {
            "closes_m1": closes_m1,
            "highs_m1": highs_m1,
            "lows_m1": lows_m1,
            "tick_volume_proxy": state["vols"][-1] if state["vols"] else 0.0,
            "direction": direction_hint,
            "drawdown": drawdown,
            "timestamp_utc": tick.time_utc,
        }

    def _maybe_log_impulses(self, symbol: str) -> None:
        """Registra impulsos históricos fuertes (fechas UTC) con throttle por símbolo."""
        cfg = self._impulse_log_cfg
        if not cfg.get("enabled", False):
            return
        interval = max(60.0, float(cfg.get("log_interval_s", 600)))
        now_m = monotonic()
        if now_m - self._last_impulse_log_mono.get(symbol, 0.0) < interval:
            return
        tf = str(cfg.get("timeframe", "H4")).strip().upper()
        if tf not in SUPPORTED_TIMEFRAMES:
            tf = "H4"
        times, highs, lows, closes = self._mtf.get_ohlc_with_times(symbol, tf)
        if cfg.get("exclude_forming_bar", True) and len(closes) > 1:
            times, highs, lows, closes = times[:-1], highs[:-1], lows[:-1], closes[:-1]
        min_pct = float(cfg.get("min_leg_return_pct", 3.0))
        max_legs = int(cfg.get("max_legs", 6))
        legs = scan_impulse_legs(highs, lows, closes, times, min_leg_return_pct=min_pct, max_legs=max_legs)
        logger.info("%s | %s", symbol, format_impulse_summary(legs, label=f"impulsos {tf}"))
        self._last_impulse_log_mono[symbol] = now_m

    def _continuation_strength(self, symbol: str, direction: str) -> float:
        """Score [0,1] de continuidad del impulso para gestión dinámica SL/TP."""
        state = self._ensure_symbol_state(symbol)
        closes = state.get("closes", [])
        vols = state.get("vols", [])
        if len(closes) < 8:
            return 0.0
        c = closes
        d1 = c[-1] - c[-2]
        d2 = c[-2] - c[-3]
        d3 = c[-3] - c[-4]
        same_dir = (
            (direction == "BUY" and d1 > 0 and d2 > 0 and d3 > 0)
            or (direction == "SELL" and d1 < 0 and d2 < 0 and d3 < 0)
        )
        velocity = abs(c[-1] - c[-4]) / max(abs(c[-4]), 1e-12)
        velocity_score = min(1.0, velocity * 250.0)
        volume_score = 0.5
        if len(vols) >= 20:
            recent = vols[-20:]
            baseline = sum(recent[:-1]) / max(len(recent[:-1]), 1)
            volume_score = min(1.0, (recent[-1] / max(baseline, 1e-9)))
        return max(0.0, min(1.0, (0.45 if same_dir else 0.0) + 0.35 * velocity_score + 0.20 * volume_score))

    def _sync_closed_trades(self, symbol: str, tick: Tick) -> None:
        """Sincroniza cierres reales con broker/DB y actualiza equity/riesgo."""
        exec_cfg = self._config.get("execution", {})
        pip_value_per_lot = float(exec_cfg.get("pip_value_per_lot", 1.0))
        to_remove: list[str] = []
        for ticket, st in list(self._open_trades.items()):
            if st.symbol != symbol:
                continue
            if self._order_manager.is_position_open(ticket):
                continue
            summary = self._order_manager.get_closed_summary(ticket) or {}
            exit_price = float(summary.get("exit_price", self._mid_price(tick)))
            profit = float(summary.get("profit", 0.0))
            close_time = summary.get("close_time", tick.time_utc)
            try:
                close_time_utc = close_time if isinstance(close_time, datetime) else tick.time_utc
            except Exception:
                close_time_utc = tick.time_utc
            if isinstance(close_time_utc, datetime) and close_time_utc.tzinfo is None:
                close_time_utc = close_time_utc.replace(tzinfo=timezone.utc)
            risk_money = max(st.lot * st.sl_pips * pip_value_per_lot, 1e-9)
            r_multiple = profit / risk_money
            self._equity += Decimal(str(profit))
            self._risk.register_trade_result(
                Decimal(str(profit)),
                closed_at=close_time_utc if isinstance(close_time_utc, datetime) else None,
                setup_type=str(st.setup_type or ""),
                symbol=str(st.symbol or ""),
            )
            if isinstance(close_time_utc, datetime):
                self._accumulate_symbol_daily_pnl(
                    st.symbol,
                    Decimal(str(profit)),
                    close_time_utc,
                )
            self._register_online_setup_outcome(st.symbol, st.setup_type, r_multiple)
            self._closed_trades_counter += 1
            # AURION X 4.1 — self-learning SMC: cierra el ciclo del trade-link.
            try:
                _mfe = float(getattr(st, "max_favorable_r", 0.0) or 0.0)
                _mae = float(getattr(st, "max_adverse_r", 0.0) or 0.0)
                self._smc_on_trade_closed(
                    symbol=st.symbol,
                    close_time=close_time_utc if isinstance(close_time_utc, datetime) else None,
                    r_multiple=float(r_multiple),
                    mfe_r=_mfe if _mfe != 0.0 else None,
                    mae_r=_mae if _mae != 0.0 else None,
                    win=bool(profit > 0),
                    trade_id=getattr(st, "db_id", None),
                )
            except Exception as exc:
                logger.debug("[SMC] on_trade_closed hook falló: %s", exc)
            # AURION X 4.1 — XAU Knowledge Engine: cierre del trade (memoria
            # + evaluación de retrain triggers).
            try:
                ke = getattr(self, "_xau_knowledge", None)
                if ke is not None and ke.enabled and (st.symbol or "").upper() == "XAUUSD":
                    tuid = (
                        getattr(getattr(st, "ticket", None), "trade_uuid", None)
                        or getattr(getattr(st, "ticket", None), "uuid", None)
                        or getattr(st, "db_id", None)
                        or str(getattr(st, "ticket", "?"))
                    )
                    ke.on_trade_close(
                        trade_id=str(tuid),
                        symbol=st.symbol,
                        close_time=close_time_utc if isinstance(close_time_utc, datetime) else datetime.utcnow(),
                        r_multiple=float(r_multiple),
                        win=bool(profit > 0),
                    )
                    # Evaluar y disparar reentrenamiento si corresponde.
                    try:
                        ke.trigger_retrain_if_needed()
                    except Exception as exc:
                        logger.debug("[KNOWLEDGE] retrain check post-close falló: %s", exc)
            except Exception as exc:
                logger.debug("[KNOWLEDGE] on_trade_close hook falló: %s", exc)
            if self._institutional_engine is not None and self._institutional_engine.enabled:
                close_ctx = {
                    "setup_type": st.setup_type,
                    "volatility_regime": self._volatility.regime().value.upper(),
                    "hour_slot": (
                        close_time_utc.astimezone(self._sniper_tz).hour
                        if isinstance(close_time_utc, datetime)
                        else tick.time_utc.hour
                    ),
                    "sniper_key": st.sniper_key,
                }
                self._institutional_engine.on_trade_closed(
                    symbol=st.symbol,
                    features_json=close_ctx,
                    profit=profit,
                    r_multiple=r_multiple,
                    score=0,
                    closed_counter=self._closed_trades_counter,
                    reload_cb=lambda: self._reload_ml_models(self._ml_model_path),
                )
            elif self._quant_engine is not None and self._quant_engine.enabled:
                close_ctx = {
                    "setup_type": st.setup_type,
                    "volatility_regime": self._volatility.regime().value.upper(),
                    "hour_slot": (
                        close_time_utc.astimezone(self._sniper_tz).hour
                        if isinstance(close_time_utc, datetime)
                        else tick.time_utc.hour
                    ),
                    "sniper_key": st.sniper_key,
                }
                self._quant_engine.on_trade_closed(
                    symbol=st.symbol,
                    features_json=close_ctx,
                    profit=profit,
                    r_multiple=r_multiple,
                    score=0,
                    closed_counter=self._closed_trades_counter,
                    reload_cb=lambda: self._reload_ml_models(self._ml_model_path),
                )
            else:
                self._maybe_auto_retrain_ml()
            self._record_intraday_close_r(st.symbol, r_multiple, close_time_utc)
            stats = self._telemetry_symbol(st.symbol)
            stats["trades_closed"] = stats.get("trades_closed", 0) + 1
            if profit > 0:
                stats["wins"] = stats.get("wins", 0) + 1
            else:
                stats["losses"] = stats.get("losses", 0) + 1
            stats["sum_pnl_cents"] = stats.get("sum_pnl_cents", 0) + int(round(profit * 100.0))
            stats["sum_r_milli"] = stats.get("sum_r_milli", 0) + int(round(r_multiple * 1000.0))
            # AURION X 4.3: trade journal - detectar razón de cierre por heurística.
            close_reason = self._infer_close_reason(st, exit_price, profit, r_multiple)
            if st.db_id:
                try:
                    self._trade_repo.update_trade_close(
                        trade_id=st.db_id,
                        exit_price=exit_price,
                        profit=profit,
                        r_multiple=r_multiple,
                        close_reason=close_reason,
                    )
                except Exception as exc:
                    logger.warning("No se pudo persistir trade_close id=%s: %s", st.db_id, exc)
            # AURION X 4.3: drift detection hook.
            if self._drift_detector is not None and self._drift_detector.enabled:
                try:
                    drift_ev = self._drift_detector.on_trade_closed(
                        r_multiple=float(r_multiple),
                        win=bool(profit > 0),
                        ts=close_time_utc if isinstance(close_time_utc, datetime) else None,
                        symbol=st.symbol,
                    )
                    if drift_ev.detected:
                        self._drift_detector.execute_action(
                            drift_ev,
                            retrain_callback=self._maybe_auto_retrain_ml,
                        )
                except Exception as exc:
                    logger.debug("[DRIFT] hook fallo: %s", exc)
            logger.info(
                "[TRADE_CLOSED] symbol=%s ticket=%s dir=%s lot=%.3f exit=%.5f pnl=%.2f R=%.3f reason=%s close=%s",
                st.symbol,
                ticket,
                st.direction,
                st.lot,
                exit_price,
                profit,
                r_multiple,
                close_reason,
                close_time_utc,
            )
            to_remove.append(ticket)
        for t in to_remove:
            self._open_trades.pop(t, None)

    @staticmethod
    def _infer_close_reason(
        st: "OpenTradeState",
        exit_price: float,
        profit: float,
        r_multiple: float,
    ) -> str:
        """Heuristica para inferir razon de cierre desde estado + exit price.

        AURION X 4.3: trade journal mejorado. Compara exit_price con SL/TP
        teoricos del state y la magnitud de R-multiple resultante. Permite
        post-mortems claros sin tracking explicito en cada decision branch.

        Valores devueltos: tp_hit, sl_hit, breakeven, profit_lock_soft,
        profit_lock_hard, trailing_stop, secure_profit, time_exit, unknown.
        """
        try:
            entry = float(st.entry_price)
            risk_px = float(st.risk_price)
            if entry <= 0 or risk_px <= 0:
                return "unknown"
            is_buy = (st.direction or "").upper() in ("BUY", "LONG")
            sign = 1.0 if is_buy else -1.0
            sl_price = entry - sign * risk_px
            tp_distance_px = (risk_px * float(st.tp_pips) / float(st.sl_pips)) if st.sl_pips > 0 else risk_px
            tp_price = entry + sign * tp_distance_px
            # Tolerancia: 10% de la distancia de riesgo (mas estricto para BE).
            tol = max(0.10 * risk_px, 1e-9)
            be_tol = max(0.05 * risk_px, 1e-9)
            # Comparaciones por proximidad de exit_price.
            near_sl = abs(exit_price - sl_price) <= tol
            near_tp = abs(exit_price - tp_price) <= tol
            near_be = abs(exit_price - entry) <= be_tol
            if near_tp:
                return "tp_hit"
            if near_sl and r_multiple < -0.85:
                return "sl_hit"
            if near_be or abs(profit) < 0.50:
                return "breakeven"
            # No esta cerca de SL/TP/BE => cerro dinamicamente.
            if r_multiple >= 0.50:
                return "secure_profit"
            if 0.0 < r_multiple < 0.50:
                return "profit_lock_soft"
            if -0.85 < r_multiple <= 0.0:
                return "trailing_stop"
            if r_multiple <= -0.85:
                return "sl_hit"
            return "unknown"
        except (ValueError, TypeError, AttributeError):
            return "unknown"

    def _execute_management(self, symbol: str, tick: Tick) -> None:
        """Gestión de salidas: TP parcial RANGE, salida dinámica opcional, o legado BE/trailing/TP adapt."""
        mid = self._mid_price(tick)
        regime = self._volatility.regime()
        spread_points = float(tick.ask - tick.bid) / max(self._pip_size(symbol), 1e-9)
        structure = self._mtf.get_structure(symbol, self._structure_timeframe)
        trend_s = str(structure.trend.value)
        state = self._ensure_symbol_state(symbol)
        mh, ml, mc = self._mtf.get_ohlc_series(symbol, "M1")
        if len(mc) >= 10:
            highs_m1, lows_m1, closes_m1 = list(mh), list(ml), list(mc)
        else:
            highs_m1, lows_m1, closes_m1 = list(state["highs"]), list(state["lows"]), list(state["closes"])
        n_c = len(closes_m1)
        vols_raw = list(state.get("vols", []) or [])
        if len(vols_raw) >= n_c and n_c > 0:
            vols_m1 = vols_raw[-n_c:]
        elif n_c > 0:
            vols_m1 = [0.0] * max(0, n_c - len(vols_raw)) + vols_raw[-min(len(vols_raw), n_c) :]
        else:
            vols_m1 = [0.0]

        to_add: list[OpenTradeState] = []

        for ticket, st in list(self._open_trades.items()):
            if st.symbol != symbol:
                continue
            move = (mid - st.entry_price) if st.direction == "BUY" else (st.entry_price - mid)
            r = move / max(st.risk_price, 1e-12)
            st.max_r = max(st.max_r, r)
            # AURION X 4.1 — XAU Knowledge: actualizar MFE/MAE en vivo (no
            # persiste por tick; sólo memoria interna del engine).
            try:
                ke = getattr(self, "_xau_knowledge", None)
                if ke is not None and ke.enabled and symbol.upper() == "XAUUSD":
                    tuid = (
                        getattr(ticket, "trade_uuid", None)
                        or getattr(ticket, "uuid", None)
                        or str(ticket)
                    )
                    ke.on_trade_tick(
                        trade_id=str(tuid),
                        current_price=float(mid),
                        when_utc=tick.time_utc,
                    )
            except Exception:
                pass
            continuation = self._continuation_strength(symbol, st.direction)

            itm_cfg = (self._config.get("execution", {}) or {}).get("intraday_trade_management") or {}
            if (
                bool(itm_cfg.get("enabled", True))
                and str(st.setup_type).upper().startswith("INTRADAY")
            ):
                pip_sz_i = self._pip_size(st.symbol)
                tp_dist_px = (
                    st.initial_tp_distance_price
                    if st.initial_tp_distance_price > 0.0
                    else float(st.tp_pips) * pip_sz_i
                )
                favorable_tp = (
                    (mid - st.entry_price) if st.direction == "BUY" else (st.entry_price - mid)
                )
                tp_prog = (
                    max(0.0, favorable_tp / max(tp_dist_px, 1e-12)) if tp_dist_px > 0.0 else 0.0
                )
                t_from = float(itm_cfg.get("tighten_from_tp_progress", 0.30))
                be_at = float(itm_cfg.get("breakeven_at_tp_progress", 0.50))
                be_buf = max(float(itm_cfg.get("be_buffer_pips", 2.0)) * pip_sz_i, pip_sz_i * 2.0)
                sl_target: float | None = None
                if tp_prog >= t_from:
                    if tp_prog >= be_at:
                        u = 1.0
                    else:
                        u = min(
                            1.0,
                            max(0.0, (tp_prog - t_from) / max(be_at - t_from, 1e-9)),
                        )
                    if st.direction == "SELL":
                        sl_hi = st.entry_price + st.risk_price
                        sl_lo = st.entry_price + be_buf
                        sl_target = sl_hi - u * (sl_hi - sl_lo)
                    else:
                        sl_lo = st.entry_price - st.risk_price
                        sl_hi = st.entry_price - be_buf
                        sl_target = sl_lo + u * (sl_hi - sl_lo)
                if sl_target is not None and st.current_sl_price is not None:
                    ok_move = (
                        st.direction == "BUY" and sl_target >= st.current_sl_price + pip_sz_i * 0.25
                    ) or (
                        st.direction == "SELL" and sl_target <= st.current_sl_price - pip_sz_i * 0.25
                    )
                    if ok_move and self._order_manager.modify_order(ticket, sl=sl_target):
                        st.current_sl_price = sl_target
                        if tp_prog >= be_at - 1e-9:
                            st.be_done = True
                        logger.info(
                            "[INTRADAY_TP_MGMT] ticket=%s tp_prog=%.2f sl→%.5f",
                            ticket,
                            tp_prog,
                            sl_target,
                        )

            stm_cfg = (self._config.get("execution", {}) or {}).get("scalper_trade_management") or {}
            if bool(stm_cfg.get("enabled", True)) and (
                not str(st.setup_type).upper().startswith("INTRADAY")
            ):
                pip_sz_s = self._pip_size(st.symbol)
                tp_dist_px = (
                    st.initial_tp_distance_price
                    if st.initial_tp_distance_price > 0.0
                    else float(st.tp_pips) * pip_sz_s
                )
                favorable_sc = (
                    (mid - st.entry_price) if st.direction == "BUY" else (st.entry_price - mid)
                )
                favorable_sc = max(0.0, favorable_sc)
                tp_prog_s = (
                    favorable_sc / max(tp_dist_px, 1e-12) if tp_dist_px > 0.0 else 0.0
                )
                arm_at = float(stm_cfg.get("arm_at_tp_progress", 0.35))
                lock_frac = float(stm_cfg.get("lock_max_favorable_excursion_frac", 0.50))
                lock_frac = min(0.95, max(0.0, lock_frac))
                be_floor_at = float(stm_cfg.get("breakeven_floor_at_tp_progress", 0.50))
                be_buf_s = max(
                    float(stm_cfg.get("be_buffer_pips", 2.0)) * pip_sz_s,
                    pip_sz_s * 2.0,
                )
                if (
                    tp_prog_s >= arm_at
                    and st.current_sl_price is not None
                    and float(st.max_r) > 0.0
                ):
                    mfe_px = float(st.max_r) * st.risk_price
                    sl_target_s: float | None = None
                    if st.direction == "BUY":
                        sl_mfe = st.entry_price - st.risk_price + lock_frac * mfe_px
                        sl_mfe = max(sl_mfe, st.entry_price - st.risk_price)
                        sl_target_s = sl_mfe
                        if tp_prog_s >= be_floor_at:
                            sl_be = st.entry_price - be_buf_s
                            sl_target_s = max(sl_mfe, sl_be)
                        sl_ord = min(sl_target_s, mid - pip_sz_s * 1.5)
                        if (
                            sl_ord >= st.current_sl_price + pip_sz_s * 0.25
                            and self._order_manager.modify_order(ticket, sl=sl_ord)
                        ):
                            st.current_sl_price = sl_ord
                            if tp_prog_s >= be_floor_at - 1e-9:
                                st.be_done = True
                            logger.info(
                                "[SCALPER_MFE_LOCK] ticket=%s tp_prog=%.2f max_r=%.3f lock=%.0f%% sl→%.5f",
                                ticket,
                                tp_prog_s,
                                st.max_r,
                                lock_frac * 100.0,
                                sl_ord,
                            )
                    else:
                        sl_mfe = st.entry_price + st.risk_price - lock_frac * mfe_px
                        sl_mfe = min(sl_mfe, st.entry_price + st.risk_price)
                        sl_target_s = sl_mfe
                        if tp_prog_s >= be_floor_at:
                            sl_be = st.entry_price + be_buf_s
                            sl_target_s = min(sl_mfe, sl_be)
                        sl_ord = max(sl_target_s, mid + pip_sz_s * 1.5)
                        if (
                            sl_ord <= st.current_sl_price - pip_sz_s * 0.25
                            and self._order_manager.modify_order(ticket, sl=sl_ord)
                        ):
                            st.current_sl_price = sl_ord
                            if tp_prog_s >= be_floor_at - 1e-9:
                                st.be_done = True
                            logger.info(
                                "[SCALPER_MFE_LOCK] ticket=%s tp_prog=%.2f max_r=%.3f lock=%.0f%% sl→%.5f",
                                ticket,
                                tp_prog_s,
                                st.max_r,
                                lock_frac * 100.0,
                                sl_ord,
                            )

            rtm = (self._config.get("execution", {}) or {}).get("range_trade_management") or {}
            if (
                bool(rtm.get("partial_tp_enabled", True))
                and (not st.range_partial_tp_done)
                and st.initial_tp_pips_planned > 0.0
                and "RANGE_FADE" in st.setup_type.upper()
                and r >= float(rtm.get("partial_tp_at_r", 0.42))
            ):
                pip_sz = self._pip_size(st.symbol)
                frac = float(rtm.get("partial_tp_fraction", 0.56))
                new_tp_pips = max(float(st.sl_pips) * 0.95, float(st.initial_tp_pips_planned) * frac)
                if st.direction == "BUY":
                    new_tp = st.entry_price + new_tp_pips * pip_sz
                else:
                    new_tp = st.entry_price - new_tp_pips * pip_sz
                if self._order_manager.modify_order(ticket, tp=new_tp):
                    st.tp_pips = float(new_tp_pips)
                    st.range_partial_tp_done = True
                    logger.info(
                        "[RANGE_MGMT] ticket=%s partial_tp r=%.2f tp_pips %.2f→%.2f",
                        ticket,
                        r,
                        st.initial_tp_pips_planned,
                        new_tp_pips,
                    )

            dyn_cfg = self._dynamic_exit_cfg_for_symbol(st.symbol)
            use_dyn = bool(dyn_cfg.get("enabled", False))
            if use_dyn:
                cont = self._continuation_strength(symbol, st.direction)
                exit_ms = build_exit_market_state(
                    highs_m1=highs_m1,
                    lows_m1=lows_m1,
                    closes_m1=closes_m1,
                    vols=vols_m1,
                    continuation_score=cont,
                    structure_trend=trend_s,
                    trade_direction=st.direction,
                    regime=regime.value.upper(),
                    spread_points=spread_points,
                    mid=mid,
                )
                trade_map: dict[str, Any] = {
                    "ticket": ticket,
                    "direction": st.direction,
                    "entry_price": st.entry_price,
                    "risk_price": st.risk_price,
                    "sl_pips": st.sl_pips,
                    "tp_pips": st.tp_pips,
                    "lot": st.lot,
                    "current_r": r,
                    "max_r": st.max_r,
                    "be_done": st.be_done,
                    "dynamic_exit_partial_done": st.dynamic_exit_partial_done,
                }
                if self._institutional_engine is not None:
                    exit_ai = self._institutional_engine.evaluate_exit(
                        trade=trade_map,
                        market_state=exit_ms,
                        signal_context={"setup_type": st.setup_type},
                    )
                    if exit_ai.should_exit and r >= 0.12:
                        if exit_ai.partial_fraction > 0 and not st.dynamic_exit_partial_done:
                            vol_close = max(st.lot * exit_ai.partial_fraction, 1e-9)
                            if self._order_manager.close_position_partial(ticket, vol_close):
                                st.lot = max(st.lot - vol_close, 1e-6)
                                st.dynamic_exit_partial_done = True
                                logger.info(
                                    "[EXIT_AI_PARTIAL] %s ticket=%s urgency=%.2f r=%.2f %s",
                                    symbol,
                                    ticket,
                                    exit_ai.urgency,
                                    r,
                                    exit_ai.reason,
                                )
                        elif self._order_manager.close_position(ticket):
                            logger.info(
                                "[EXIT_AI_FULL] %s ticket=%s urgency=%.2f r=%.2f %s",
                                symbol,
                                ticket,
                                exit_ai.urgency,
                                r,
                                exit_ai.reason,
                            )
                            continue
                dec = decide_dynamic_exit(trade_map, exit_ms, dyn_cfg, mid=mid)
                if dec.close_full:
                    if self._order_manager.close_position(ticket):
                        logger.info(
                            "[EXIT_FULL] reason=%s ticket=%s symbol=%s r=%.2f max_r=%.2f",
                            dec.reason_full,
                            ticket,
                            symbol,
                            r,
                            st.max_r,
                        )
                    continue
                if dec.close_partial_fraction > 0:
                    vol_close = max(st.lot * dec.close_partial_fraction, 1e-9)
                    if self._order_manager.close_position_partial(ticket, vol_close):
                        st.lot = max(st.lot - vol_close, 1e-6)
                        st.dynamic_exit_partial_done = True
                        logger.info(
                            "[EXIT_PARTIAL] reason=%s ticket=%s symbol=%s fraction=%.2f r=%.2f",
                            dec.reason_partial,
                            ticket,
                            symbol,
                            dec.close_partial_fraction,
                            r,
                        )
                pip_sz = self._pip_size(st.symbol)
                try:
                    min_trail = float(dyn_cfg.get("min_trail_improve_pips", 1.0)) * pip_sz
                except (TypeError, ValueError):
                    min_trail = pip_sz
                sl_candidates: list[float] = []
                if dec.move_be and not st.be_done:
                    be_sl = st.entry_price + (
                        0.03 * st.risk_price if st.direction == "BUY" else -0.03 * st.risk_price
                    )
                    sl_candidates.append(be_sl)
                if dec.trail_sl_price is not None:
                    sl_candidates.append(float(dec.trail_sl_price))
                if sl_candidates:
                    if st.direction == "BUY":
                        final_sl = max(sl_candidates)
                    else:
                        final_sl = min(sl_candidates)
                    cur_sl = st.current_sl_price
                    improved = cur_sl is None
                    if not improved and cur_sl is not None:
                        if st.direction == "BUY":
                            improved = final_sl >= cur_sl + min_trail
                        else:
                            improved = final_sl <= cur_sl - min_trail
                    if improved and self._order_manager.modify_order(ticket, sl=final_sl):
                        st.current_sl_price = final_sl
                        be_line = st.entry_price + (
                            0.03 * st.risk_price if st.direction == "BUY" else -0.03 * st.risk_price
                        )
                        if st.direction == "BUY" and final_sl >= be_line - 1e-9:
                            st.be_done = True
                        elif st.direction == "SELL" and final_sl <= be_line + 1e-9:
                            st.be_done = True
                        if dec.move_be and dec.reason_be:
                            logger.info(
                                "[MOVE_BE] reason=%s ticket=%s symbol=%s r=%.2f sl=%.5f",
                                dec.reason_be,
                                ticket,
                                symbol,
                                r,
                                final_sl,
                            )
                        if dec.trail_sl_price is not None and dec.reason_trail:
                            st.last_dynamic_trail_sl = final_sl
                            logger.info(
                                "[TRAILING_ON] reason=%s ticket=%s symbol=%s sl=%.5f",
                                dec.reason_trail,
                                ticket,
                                symbol,
                                final_sl,
                            )
                if dec.log_keep:
                    logger.debug("%s ticket=%s r=%.2f", dec.log_keep, ticket, r)
            else:
                exec_cfg_l = self._config.get("execution", {}) or {}
                itm_on = bool((exec_cfg_l.get("intraday_trade_management") or {}).get("enabled", True))
                stm_on = bool((exec_cfg_l.get("scalper_trade_management") or {}).get("enabled", True))
                is_id_setup = str(st.setup_type).upper().startswith("INTRADAY")
                scalper_stm = stm_on and (not is_id_setup)
                legacy_mgmt = not ((is_id_setup and itm_on) or scalper_stm)
                if legacy_mgmt and (not st.be_done) and r >= 0.3:
                    be_sl = st.entry_price + (
                        0.03 * st.risk_price if st.direction == "BUY" else -0.03 * st.risk_price
                    )
                    if self._order_manager.modify_order(ticket, sl=be_sl):
                        st.current_sl_price = be_sl
                        st.be_done = True

                if legacy_mgmt and (not st.trailing_done) and r >= 0.7:
                    trail_factor = 0.35 if regime == VolatilityRegime.HIGH else 0.5
                    trail_price = (
                        mid - st.risk_price * trail_factor
                        if st.direction == "BUY"
                        else mid + st.risk_price * trail_factor
                    )
                    if self._order_manager.modify_order(ticket, sl=trail_price):
                        st.current_sl_price = trail_price
                        st.trailing_done = True

                base_tp_dist = st.tp_pips * self._pip_size(st.symbol)
                if (not st.tp_extended) and continuation >= 0.65 and r >= 0.6 and regime in (
                    VolatilityRegime.NORMAL,
                    VolatilityRegime.HIGH,
                ):
                    new_tp = (
                        st.entry_price + base_tp_dist * 1.35
                        if st.direction == "BUY"
                        else st.entry_price - base_tp_dist * 1.35
                    )
                    lock_sl = (
                        st.entry_price + st.risk_price * 0.25
                        if st.direction == "BUY"
                        else st.entry_price - st.risk_price * 0.25
                    )
                    if self._order_manager.modify_order(ticket, sl=lock_sl, tp=new_tp):
                        st.current_sl_price = lock_sl
                    st.tp_extended = True
                    logger.info(
                        "[MGMT_ADAPT] ticket=%s mode=TP_EXTEND continuation=%.2f regime=%s",
                        ticket,
                        continuation,
                        regime.value.upper(),
                    )

                if (not st.tp_tightened) and regime == VolatilityRegime.LOW and r >= 0.35 and continuation < 0.45:
                    new_tp = (
                        st.entry_price + base_tp_dist * 0.75
                        if st.direction == "BUY"
                        else st.entry_price - base_tp_dist * 0.75
                    )
                    protect_sl = (
                        st.entry_price + st.risk_price * 0.10
                        if st.direction == "BUY"
                        else st.entry_price - st.risk_price * 0.10
                    )
                    if self._order_manager.modify_order(ticket, sl=protect_sl, tp=new_tp):
                        st.current_sl_price = protect_sl
                    st.tp_tightened = True
                    logger.info(
                        "[MGMT_ADAPT] ticket=%s mode=TP_TIGHTEN continuation=%.2f regime=%s",
                        ticket,
                        continuation,
                        regime.value.upper(),
                    )

            last_signal = self._last_signal_by_symbol.get(symbol, "")
            if (not st.reentry_done) and st.max_r > 0.5 and last_signal == st.direction:
                corr_block, corr_msg = self._correlation_blocks_new_entry(symbol)
                if corr_block:
                    self._maybe_log_entry_skip(
                        symbol,
                        "correlation_cap",
                        "%s skip re-entry: correlación (%s)",
                        symbol,
                        corr_msg,
                    )
                    continue
                kill_block, kill_msg = self._intraday_kill_blocks_new_entry(symbol, tick.time_utc)
                if kill_block:
                    self._maybe_log_entry_skip(
                        symbol,
                        "intraday_kill",
                        "%s skip re-entry: intraday kill-switch (%s)",
                        symbol,
                        kill_msg,
                    )
                    continue
                sym_loss_block, sym_loss_msg = self._symbol_daily_loss_blocks(symbol, tick.time_utc)
                if sym_loss_block:
                    self._maybe_log_entry_skip(
                        symbol,
                        "symbol_daily_loss",
                        "%s skip re-entry: circuit breaker pérdida diaria (%s)",
                        symbol,
                        sym_loss_msg,
                    )
                    continue
                spread_block, spread_msg = self._spread_guard_blocks(symbol, spread_points)
                if spread_block:
                    self._maybe_log_entry_skip(
                        symbol,
                        "spread_guard",
                        "%s skip re-entry: spread_guard (%s)",
                        symbol,
                        spread_msg,
                    )
                    continue
                same_pending = sum(1 for x in to_add if x.symbol.upper() == symbol.upper())
                if self._reconcile_before_new_entry:
                    self._reconcile_positions_with_broker()
                cap_block, cap_msg = self._position_caps_block(
                    symbol,
                    pending_same_symbol=same_pending,
                    pending_total_extra=len(to_add),
                )
                if cap_block:
                    self._maybe_log_entry_skip(
                        symbol,
                        "position_cap",
                        "%s skip re-entry: %s",
                        symbol,
                        cap_msg,
                    )
                    continue
                # AURION X 4.4 (10-jun-2026): MODO OBSERVADOR (re-entry tambien).
                _obs_cfg = self._config.get("execution", {}) or {}
                if bool(_obs_cfg.get("observer_mode", False)):
                    logger.info(
                        "[OBSERVER] %s would_re_enter dir=%s lot=%.2f "
                        "(observer_mode=ON, no order sent)",
                        symbol,
                        st.direction.upper(),
                        max(st.lot * 0.5, 0.01),
                    )
                    continue
                re_ticket = self._order_manager.place_market_order(
                    symbol=symbol,
                    volume=max(st.lot * 0.5, 0.01),
                    side=st.direction.lower(),
                )
                if re_ticket:
                    re = OpenTradeState(
                        ticket=re_ticket,
                        symbol=symbol,
                        direction=st.direction,
                        entry_price=mid,
                        risk_price=st.risk_price,
                        sl_pips=st.sl_pips,
                        tp_pips=st.tp_pips,
                        lot=max(st.lot * 0.5, 0.01),
                        setup_type=st.setup_type,
                        sniper_key=st.sniper_key,
                        initial_tp_distance_price=st.initial_tp_distance_price,
                    )
                    to_add.append(re)
                    st.reentry_done = True
        for t in to_add:
            self._open_trades[t.ticket] = t

    def on_tick(self, symbol: str, tick: Tick) -> None:
        """Pipeline on_tick:
        1) session_filter, 2) emergency, 3) risk, 4) market_state, 5) features,
        6) ML score, 7) strategy, 8) validator, 9) execute, 10) gestión, 11) log+DB.
        """
        self._append_tick_to_bars(symbol, tick)
        self._maybe_refresh_equity_from_broker()
        self._maybe_run_daily_ml_optimization(tick.time_utc)
        self._maybe_reconcile_broker()
        m1_h, m1_l, m1_c, m1_v = self._mtf.get_ohlcv_series(symbol, "M1")
        self._intraday.set_m1_ohlc(m1_h, m1_l, m1_c)
        self._volatility.update(tick)
        self._sync_closed_trades(symbol, tick)
        self._learn_from_closed_trades()
        now_m = monotonic()
        interval_s = max(0.0, float(self._decision_interval_ms) / 1000.0)
        if interval_s > 0.0:
            last_decision = self._last_decision_mono.get(symbol, 0.0)
            if (now_m - last_decision) < interval_s:
                self._execute_management(symbol, tick)
                return
            self._last_decision_mono[symbol] = now_m

        # 1) Session filter (antes de MTF / estructura para ahorrar CPU fuera de horario)
        if not self._session_filter.allow_new_entries(tick.time_utc):
            self._maybe_log_entry_skip(
                symbol,
                "session",
                "%s skip: fuera de ventana de sesión (%s)",
                symbol,
                self._session_filter.hint,
            )
            return

        # Recalcula tendencias en todos los TFs del motor en cada decisión (máxima frescura para scalper).
        mtf_trends = self._mtf.get_trends(symbol, min_interval_s=0.0)
        self._maybe_log_mtf(symbol, mtf_trends)
        self._maybe_log_impulses(symbol)
        self._maybe_log_telemetry(symbol)
        # AURION X 4.1 — SMC institucional: compute periódico desacoplado del flujo
        # de señales para que dashboards / EAs en MT5 reciban OB/BOS/CHOCH/FVG/Liq
        # aunque el scalper aún esté en warm-up.
        self._maybe_compute_smc(symbol)

        # 2) Emergency controller
        state = self._ensure_symbol_state(symbol)
        structure = self._mtf.get_structure(symbol, self._structure_timeframe)
        market_state = self._build_market_state(symbol, tick, structure)
        features_vec, features_dict = build_features(market_state)
        atr_now = float(features_dict.get("atr_m1", 0.0))
        self._atr_feature_hist.append(atr_now)
        if len(self._atr_feature_hist) > 200:
            self._atr_feature_hist = self._atr_feature_hist[-200:]
        atr_ready = len(self._atr_feature_hist) >= self._atr_min_samples
        baseline = (
            sum(self._atr_feature_hist[:-1]) / max(len(self._atr_feature_hist) - 1, 1)
            if atr_ready
            else max(atr_now, 1e-9)
        )
        spread_now = float(tick.ask - tick.bid)
        spread_points = spread_now / max(self._pip_size(symbol), 1e-9)
        atr_current_for_emergency = max(atr_now, 1e-9) if (atr_ready and self._atr_spike_enabled) else max(
            baseline,
            1e-9,
        )
        if self._emergency.evaluate_and_act(
            atr_current=atr_current_for_emergency,
            atr_baseline=max(baseline, 1e-9),
            spread=spread_points,
            executor=self._order_manager,
        ):
            now_ts = datetime.now(timezone.utc).timestamp()
            reason = self._emergency.last_reason
            if reason != self._last_emergency_reason or (now_ts - self._last_emergency_warn_ts >= 30.0):
                logger.warning("on_tick: emergencia activa (%s)", reason)
                self._last_emergency_warn_ts = now_ts
                self._last_emergency_reason = reason
            return

        # 3) Risk manager
        risk_probe = Decimal("0")
        if not self._risk.can_open_new_trade(self._equity, risk_probe, now_utc=tick.time_utc):
            self._maybe_log_entry_skip(
                symbol,
                "risk_pause",
                "%s skip: RiskManager bloquea nuevas entradas (drawdown, cooldown o límite diario)",
                symbol,
            )
            self._execute_management(symbol, tick)
            return

        # 5) Features ya construidas arriba; ML cuantitativo tras tener señal (paso 6b)
        prob_buy = 0.5
        prob_sell = 0.5
        ml_conf = 0.0

        # 7) Elegir estrategia
        structure_fuerte = (
            structure.trend != Trend.RANGE
            and len(structure.key_levels) >= self._intraday_min_key_levels
        )
        strategy_name = "SCALPER"
        signal: Any = None
        if structure_fuerte:
            intraday_sig = self._intraday.evaluate_intraday(structure_hint=structure)
            if intraday_sig is not None:
                signal = dict(intraday_sig)
                strategy_name = "INTRADAY"
        elif (
            structure.trend == Trend.RANGE
            and structure.position_in_range is not None
            and len(structure.key_levels) >= 2
        ):
            rng_sig = self._intraday.evaluate_range_structure_reversion(structure)
            if rng_sig is not None:
                signal = dict(rng_sig)
                strategy_name = "INTRADAY_RANGE"
        scalper_min_m1 = 0
        if signal is None:
            symbol_scalper = self._scalper_engine_for_symbol(symbol)
            scalper_min_m1 = symbol_scalper.min_bars + 1
            if len(m1_c) < scalper_min_m1:
                self._maybe_log_entry_skip(
                    symbol,
                    "m1_warmup",
                    "%s skip: scalper aún sin historia M1 (%d/%d velas; se necesita ~%d min de ticks por símbolo)",
                    symbol,
                    len(m1_c),
                    scalper_min_m1,
                    max(0, scalper_min_m1 - len(m1_c)),
                )
                signal = None
            else:
                signal = symbol_scalper.evaluate_m1(
                    m1_h[:-1], m1_l[:-1], m1_c[:-1], m1_v[:-1]
                )
                strategy_name = "SCALPER"
        if signal is None:
            if scalper_min_m1 <= 0 or len(m1_c) >= scalper_min_m1:
                self._maybe_log_entry_skip(
                    symbol,
                    "no_strategy_signal",
                    "%s skip: sin señal de estrategia (intraday/scalper)",
                    symbol,
                )
            self._execute_management(symbol, tick)
            return
        self._telemetry_inc(symbol, "signals")
        continuation_now = self._continuation_strength(symbol, str(signal["direction"]))
        momentum_score = float(signal.get("momentum_score", 0.0))
        volume_score = float(signal.get("volume_score", 0.0))
        volume_ratio = float(signal.get("volume_ratio", 0.0))
        setup_type = str(signal.get("setup_type", "")).upper()
        regime = self._volatility.regime()
        profile = self._profile_for_symbol(symbol)
        spread_cap = float(profile.get("spread_extreme_points", self._spread_guard_max_points))
        feat_ctx = {
            "direction": signal["direction"],
            "atr_m1": float(features_dict.get("atr_m1", 0.0)),
            "continuation_strength": continuation_now,
            "momentum_score": momentum_score,
            "volume_score": volume_score,
            "volume_ratio": volume_ratio,
            "spread_points": spread_points,
            "spread_cap_points": spread_cap,
            "mtf_trends": {k: v.value for k, v in mtf_trends.items()},
            "position_in_range": structure.position_in_range,
            "liquidity_sweep": signal.get("liquidity_sweep"),
            "pullback": signal.get("pullback"),
            "range_breakout_probability": signal.get("range_breakout_probability"),
            "absorption_score": signal.get("absorption_score"),
            "breakout_quality": volume_score,
            "equity": float(self._equity),
            "session_type": self._session_filter.classify_profile(tick.time_utc),
        }
        # AURION X 4.1 — Capa SMC institucional (XAUUSD-only por defecto).
        # Inyectamos features SMC en features_dict (no en el vector, para no
        # romper el schema de modelos ya entrenados) y en feat_ctx para que el
        # institucional las vea en `context`. Soft-fail: si falla, sigue todo.
        smc_snapshot = None
        smc_features: dict[str, float] = {}
        if getattr(self, "_smc_analyzer", None) is not None:
            try:
                smc_snapshot = self._smc_analyze(symbol)
                if smc_snapshot is not None:
                    smc_features = self._smc_analyzer.features(
                        symbol,
                        direction_hint=str(signal.get("direction", "")),
                    )
                    for k, v in smc_features.items():
                        # prefijo smc_ para que coexistan con keys ML existentes
                        # (algunas como bos_strength ya son únicas; las dejamos
                        # tal cual para que el dashboard / learner las consuma
                        # directamente y además duplicamos con prefijo).
                        features_dict[k] = float(v)
                        features_dict[f"smc_{k}"] = float(v)
                    feat_ctx["smc_features"] = dict(smc_features)
                    feat_ctx["smc_macro_bias"] = smc_snapshot.macro_bias.value
                    feat_ctx["smc_inst_bias"] = smc_snapshot.institutional_bias.value
                    feat_ctx["smc_op_bias"] = smc_snapshot.operational_bias.value
                    feat_ctx["smc_confluence"] = float(
                        smc_snapshot.confluence.confluence_score
                    )
            except Exception as exc:
                logger.debug("[SMC] features no aplicadas (%s)", exc)
        # AURION X 4.1 — XAU Knowledge: observación pasiva de velas cerradas
        # (sólo XAUUSD). La función es idempotente: detecta cuándo cambia el
        # último timestamp por TF supervisado y entonces ingesta.
        if getattr(self, "_xau_knowledge", None) is not None and symbol.upper() == "XAUUSD":
            try:
                for _tf in self._xau_knowledge.config.observe_timeframes:
                    self._knowledge_observe_bar(symbol, _tf, smc_snapshot=smc_snapshot)
            except Exception as exc:
                logger.debug("[KNOWLEDGE] observe_bar loop falló: %s", exc)
        rule_conf_base = max(float(signal.get("confidence_base", 0.0)), 0.0)
        quant_dec: QuantDecision | None = None
        inst_dec: InstitutionalDecision | None = None
        dd_pct = float(self._risk._current_drawdown_pct(self._equity))
        recent_wr = self._recent_winrate(symbol)

        # ─── AURION X 4.5 (11-jun-2026) — SMC DECISION ENGINE ──────────────
        # Capa institucional SMC que actua ANTES de Institutional/Apex.
        # Filtra setups mediocres y concentra entradas en estructuras fuertes.
        # Soft-fail: si el engine es None o lanza, no rompe el pipeline.
        smc_decision = None
        if getattr(self, "_smc_decision_engine", None) is not None and \
                self._smc_decision_engine.enabled and smc_snapshot is not None:
            try:
                # ml_confidence preliminar (sin Apex todavia): usamos el
                # confidence_base de la senal si existe; el filtro definitivo
                # con ml_conf real lo aplican Apex/Institutional aguas abajo.
                pre_ml_conf = max(float(signal.get("confidence_base", 0.0)), 0.0)
                # Calcular flag de post-event (30 min post-evento macro) usando
                # el news_calendar guard si esta disponible.
                _post_event = False
                _news_blackout = False
                try:
                    nc = getattr(self, "_news_calendar", None)
                    if nc is not None and hasattr(nc, "in_post_event_window"):
                        _post_event = bool(nc.in_post_event_window(
                            tick.time_utc, symbol=symbol, minutes=30,
                        ))
                    if nc is not None and hasattr(nc, "in_blackout"):
                        _news_blackout = bool(nc.in_blackout(
                            tick.time_utc, symbol=symbol,
                        ))
                except Exception:
                    _post_event = False
                    _news_blackout = False

                # AURION X 4.6 — VOLUMEN, SESSIONS, SMART TARGETS, SMART STOPS
                _entry_px = self._mid_price(tick)
                _vol_class: str | None = None
                _vol_delta: float = 0.0
                _session_pools: dict[str, Any] | None = None
                _smart_targets = None
                _smart_stops = None
                _rr_estimated: float | None = None
                try:
                    from market.volume_classifier import classify_volume
                    from market.session_liquidity import (
                        compute_session_ranges, session_pools_for_targets,
                        current_session,
                    )
                    from risk.smart_targets import compute_targets
                    from risk.smart_stops import compute_stop_plan
                    # Velas recientes para vol/sessions (M15)
                    _candles = []
                    try:
                        # ``self._mtf`` o ``market_data`` puede exponer get_candles
                        for src_attr in ("_mtf", "_market_data", "market_data"):
                            src = getattr(self, src_attr, None)
                            if src is None:
                                continue
                            for meth in ("get_candles", "get_recent_candles",
                                         "candles_for"):
                                if hasattr(src, meth):
                                    _candles = list(
                                        getattr(src, meth)(symbol, "M15", 80) or []
                                    )
                                    if _candles:
                                        break
                            if _candles:
                                break
                    except Exception:
                        _candles = []
                    # Clasificacion de volumen
                    try:
                        _vol = classify_volume(
                            candles=_candles,
                            has_recent_bos=bool(
                                feat_ctx.get("smc_bos_recent")
                                or feat_ctx.get("smc_bos_count_h1", 0)
                            ),
                            has_recent_choch=bool(
                                feat_ctx.get("smc_choch_recent")
                                or feat_ctx.get("smc_choch_count_h1", 0)
                            ),
                        )
                        _vol_class = _vol.classification.value
                        _vol_delta = float(_vol.score_delta + _vol.bos_modifier)
                    except Exception as exc:
                        logger.debug("[VOL_CLS] fallo (soft): %s", exc)
                    # Sessions Asia/London/NY (siempre que tengamos velas)
                    try:
                        _ranges = compute_session_ranges(
                            candles=_candles, now=tick.time_utc,
                            last_close=_entry_px,
                        )
                        _session_pools = session_pools_for_targets(_ranges)
                    except Exception as exc:
                        logger.debug("[SESS_LIQ] fallo (soft): %s", exc)
                    # ATR(14) M15: lo tomamos de feat_ctx si lo hay, sino estimamos
                    _atr = float(
                        feat_ctx.get("atr_14_m15")
                        or feat_ctx.get("atr_m15")
                        or feat_ctx.get("atr")
                        or 1.0
                    )
                    if _atr <= 0:
                        _atr = 1.0
                    _direction = str(signal.get("direction", "")).upper()
                    # Smart Targets TP1..TP4
                    try:
                        _smart_targets = compute_targets(
                            direction=_direction, entry=_entry_px,
                            snapshot=smc_snapshot, atr=_atr,
                            sessions=_session_pools,
                        )
                    except Exception as exc:
                        logger.debug("[SMART_TGT] fallo (soft): %s", exc)
                    # Smart Stops (necesita grade preliminar -> usamos B)
                    try:
                        _tp_prices = (
                            _smart_targets.prices() if _smart_targets else []
                        )
                        _smart_stops = compute_stop_plan(
                            direction=_direction, entry=_entry_px,
                            snapshot=smc_snapshot, atr=_atr,
                            primary_tf=getattr(smc_snapshot, "primary_timeframe", "M15"),
                            setup_grade="B",  # preliminar; SMCDecision reajustara
                            smc_score=50.0,   # preliminar
                            targets_prices=_tp_prices,
                        )
                        # RR estimado = RR balanced vs TP1 (referencia)
                        bal_rr = _smart_stops.rr_table.get("balanced", {}).get("tp1")
                        if bal_rr is not None:
                            _rr_estimated = float(bal_rr)
                    except Exception as exc:
                        logger.debug("[SMART_SL] fallo (soft): %s", exc)
                except Exception as exc:
                    logger.debug("[SMC_46] precomp fallo (soft): %s", exc)

                smc_decision = self._smc_decision_engine.evaluate(
                    symbol=symbol,
                    direction=str(signal.get("direction", "")),
                    snapshot=smc_snapshot,
                    ml_confidence=pre_ml_conf,
                    mtf_trends={tf: t.value if hasattr(t, "value") else str(t)
                                for tf, t in (mtf_trends or {}).items()},
                    last_close=_entry_px,
                    primary_tf=getattr(smc_snapshot, "primary_timeframe", "M15"),
                    ts_utc=tick.time_utc,
                    in_post_event_window=_post_event,
                    volume_class=_vol_class,
                    volume_score_delta=_vol_delta,
                    rr_estimated=_rr_estimated,
                    news_blackout=_news_blackout,
                )
                # Mergeamos el veredicto en feat_ctx y features_dict ANTES de
                # llamar a Institutional/Apex. Asi el motor institucional ve
                # smc_score/setup_grade/pd_state y puede usarlos como context.
                feat_ctx["smc_decision"] = smc_decision.to_dict()
                feat_ctx["smc_decision_score"] = float(smc_decision.smc_score)
                feat_ctx["smc_decision_grade"] = smc_decision.setup_grade
                feat_ctx["smc_decision_status"] = getattr(smc_decision, "setup_status", "NO_TRADE")
                feat_ctx["smc_pd_state"] = smc_decision.pd_state.state.value
                feat_ctx["smc_sweep_detected"] = bool(smc_decision.sweep_detected)
                feat_ctx["smc_quality_mode"] = getattr(smc_decision, "quality_mode", "normal")
                feat_ctx["smc_volume_class"] = _vol_class or "UNKNOWN"
                # Smart Targets / Stops (AURION X 4.6) — para panel + DB
                if _smart_targets is not None:
                    feat_ctx["smart_targets"] = _smart_targets.to_dict()
                if _smart_stops is not None:
                    feat_ctx["smart_stops"] = _smart_stops.to_dict()
                if _session_pools is not None:
                    feat_ctx["session_pools"] = _session_pools
                # Cache atomico en disco para que el dashboard server lo lea
                # y lo exponga al panel MT5 sin tocar la BD ni crear tablas.
                try:
                    self._write_smc_decision_cache(
                        symbol, smc_decision,
                        smart_targets=_smart_targets,
                        smart_stops=_smart_stops,
                        session_pools=_session_pools,
                        volume_class=_vol_class,
                        entry_price=_entry_px,
                    )
                except Exception as exc:
                    logger.debug("[SMC_DECISION] cache write fallo (soft): %s", exc)
                features_dict["smc_decision_score"] = float(smc_decision.smc_score)
                features_dict["smc_decision_lot_mult"] = float(smc_decision.lot_multiplier)
                features_dict["smc_pd_pct_in_range"] = float(
                    smc_decision.pd_state.pct_in_range
                )
                features_dict["smc_ob_quality_score"] = float(
                    smc_decision.ob_quality.score
                )
                features_dict["smc_fvg_quality_score"] = float(
                    smc_decision.fvg_quality.score
                )
                features_dict["smc_mtf_aligned_count"] = float(
                    smc_decision.mtf.aligned_count
                )
                features_dict["smc_sweep_detected"] = 1.0 if smc_decision.sweep_detected else 0.0
                # Hard filter: si el motor SMC bloquea, no abrimos.
                if not smc_decision.allow_trade:
                    self._maybe_log_entry_skip(
                        symbol,
                        "smc_decision_block",
                        "%s skip: SMC_DECISION block score=%.1f grade=%s reason=%s pd=%s",
                        symbol,
                        smc_decision.smc_score,
                        smc_decision.setup_grade,
                        smc_decision.block_reason,
                        smc_decision.pd_state.state.value,
                    )
                    self._execute_management(symbol, tick)
                    return
            except Exception as exc:
                logger.debug("[SMC_DECISION] evaluate fallo (soft): %s", exc)
                smc_decision = None

        if self._institutional_engine is not None and self._institutional_engine.enabled:
            self._institutional_engine.periodic_learn()
            inst_dec = self._institutional_engine.evaluate_entry(
                symbol=symbol,
                direction=str(signal["direction"]),
                market_state=market_state,
                signal=signal,
                regime=regime,
                mtf_trends=mtf_trends,
                structure=structure,
                continuation_strength=continuation_now,
                spread_points=spread_points,
                spread_cap=spread_cap,
                rule_score=0,
                rule_confidence=rule_conf_base,
                drawdown_pct=dd_pct,
                recent_winrate=recent_wr,
                context=feat_ctx,
            )
            self._inst_decision_by_symbol[symbol.upper()] = inst_dec
            quant_dec = inst_dec.quant
            self._quant_decision_by_symbol[symbol.upper()] = quant_dec
            features_vec = quant_dec.feature_vector
            features_dict = quant_dec.features_dict
            prob_buy = quant_dec.prob_buy
            prob_sell = quant_dec.prob_sell
            ml_conf = quant_dec.ml_confidence
            if not inst_dec.allow_trade:
                # v4.1 HÍBRIDO: bypass de setup_grade_C en SCALPER cuando hay
                # convicción ML alta + dirección alineada + ejecución limpia.
                # Sin esto el motor institucional rechazaba grades B-/C que
                # tenían ml=0.66-0.79 y micro alineado (visible en logs como
                # "motor institucional (setup_grade_C)"). El downgrade de lote
                # se aplica más abajo a través de inst_dec.lot_multiplier.
                grade_val = str(inst_dec.setup_rank.grade.value or "").upper()
                block_reason_l = str(inst_dec.block_reason or "").lower()
                is_grade_c_block = (
                    grade_val == "C"
                    and block_reason_l.startswith("setup_grade_")
                )
                ml_strong_for_bypass = float(ml_conf or 0.0) >= 0.62
                d_sig = str(signal.get("direction", "")).upper()
                ml_dir_for_bypass = "BUY" if float(prob_buy) >= float(prob_sell) else "SELL"
                dir_aligned_for_bypass = (
                    ml_dir_for_bypass == d_sig
                    or abs(float(prob_buy) - float(prob_sell)) < 0.10
                )
                scalper_like = (
                    strategy_name == "SCALPER"
                    or "SCALPER" in setup_type
                    or "MOMENTUM" in setup_type
                    or "BREAKOUT" in setup_type
                )
                if (
                    is_grade_c_block
                    and scalper_like
                    and ml_strong_for_bypass
                    and dir_aligned_for_bypass
                ):
                    logger.info(
                        "%s | institutional grade_C bypass (ml=%.3f dir=%s ml_dir=%s gap=%.3f setup=%s)",
                        symbol,
                        float(ml_conf or 0.0),
                        d_sig,
                        ml_dir_for_bypass,
                        abs(float(prob_buy) - float(prob_sell)),
                        setup_type,
                    )
                else:
                    self._maybe_log_entry_skip(
                        symbol,
                        "institutional_block",
                        "%s skip: motor institucional (%s) grade=%s",
                        symbol,
                        inst_dec.block_reason,
                        inst_dec.setup_rank.grade.value,
                        setup_type=setup_type,
                    )
                    self._execute_management(symbol, tick)
                    return
        elif self._quant_engine is not None and self._quant_engine.enabled:
            self._quant_engine.periodic_learn()
            self._quant_engine.on_end_of_day(
                tick.time_utc,
                reload_cb=lambda: self._reload_ml_models(self._ml_model_path),
            )
            quant_dec = self._quant_engine.evaluate_entry(
                symbol=symbol,
                direction=str(signal["direction"]),
                market_state=market_state,
                signal=signal,
                regime=regime.value,
                mtf_trends={k: v.value for k, v in mtf_trends.items()},
                structure_trend=structure.trend.value,
                continuation_strength=continuation_now,
                spread_points=spread_points,
                spread_cap=spread_cap,
                rule_score=0,
                rule_confidence=rule_conf_base,
                context=feat_ctx,
            )
            self._quant_decision_by_symbol[symbol.upper()] = quant_dec
            features_vec = quant_dec.feature_vector
            features_dict = quant_dec.features_dict
            prob_buy = quant_dec.prob_buy
            prob_sell = quant_dec.prob_sell
            ml_conf = quant_dec.ml_confidence
        elif self._ml_enabled:
            try:
                pred = self._predictor_for_symbol(symbol)
                if pred.model is not None:
                    ml_score = pred.predict(features_vec)
                    prob_buy = float(ml_score["prob_buy"])
                    prob_sell = float(ml_score["prob_sell"])
                    ml_conf = float(ml_score["confidence"])
            except Exception as exc:
                logger.debug("predictor no disponible (%s)", exc)
        # v4.1 HÍBRIDO: calibración anti-bias en runtime. Si el modelo viene
        # devolviendo >=80% predicciones en un solo sentido en las últimas N
        # muestras, recalibramos prob_buy/prob_sell para reducir el peso del
        # sentido dominante. Esto evita que un modelo con bias extremo (caso
        # XAUUSD que predice 100% BUY) bloquee constantemente la señal
        # contraria por "ML contradice señal". La calibración es conservadora
        # (factor 0.5-0.7 según severidad) y nunca invierte el ranking de
        # probabilidades (sólo lo acerca), preservando la información del ML.
        try:
            br_buy, br_sell = self._ml_direction_bias_ratio(symbol)
            bias_dominant = max(br_buy, br_sell)
            if bias_dominant >= 0.80 and float(prob_buy) >= 0.0 and float(prob_sell) >= 0.0:
                if bias_dominant >= 0.95:
                    shrink = 0.50
                elif bias_dominant >= 0.90:
                    shrink = 0.60
                else:
                    shrink = 0.75
                pb = float(prob_buy)
                ps = float(prob_sell)
                mean_p = (pb + ps) / 2.0 if (pb + ps) > 0.0 else 0.5
                if br_buy >= 0.80 and pb > ps:
                    pb_adj = mean_p + (pb - mean_p) * shrink
                    ps_adj = mean_p + (ps - mean_p) * shrink
                elif br_sell >= 0.80 and ps > pb:
                    pb_adj = mean_p + (pb - mean_p) * shrink
                    ps_adj = mean_p + (ps - mean_p) * shrink
                else:
                    pb_adj, ps_adj = pb, ps
                tot = pb_adj + ps_adj
                if tot > 0.0:
                    pb_adj /= tot
                    ps_adj /= tot
                if abs(pb_adj - pb) > 1e-3 or abs(ps_adj - ps) > 1e-3:
                    logger.info(
                        "%s | anti-bias calibración runtime: bias=(%.2f/%.2f) "
                        "prob_buy %.3f->%.3f prob_sell %.3f->%.3f (shrink=%.2f)",
                        symbol,
                        br_buy,
                        br_sell,
                        pb,
                        pb_adj,
                        ps,
                        ps_adj,
                        shrink,
                    )
                    prob_buy = pb_adj
                    prob_sell = ps_adj
        except Exception as exc:
            logger.debug("anti-bias calibración runtime falló (%s)", exc)
        wd_pol = self._weekday_policy_for(tick.time_utc)
        allow_wd, wd_reason = self._evaluate_weekday_policy(
            symbol,
            setup_type,
            wd_pol,
            mtf_trends=mtf_trends,
            base_ml_threshold=self._ml_threshold_floor,
            trade_count_today=self._weekday_trade_count_today(symbol, tick.time_utc),
        )
        if not allow_wd:
            self._maybe_log_entry_skip(
                symbol,
                wd_reason,
                "%s skip: política día %s (%s)",
                symbol,
                tick.time_utc.strftime("%A"),
                wd_reason,
            )
            self._execute_management(symbol, tick)
            return
        ny_turbo = self._is_ny_turbo_window(symbol, tick.time_utc)
        blocked_reason: str | None = None

        scalper_mtf_ok = self._mtf_scalper_allows(str(signal["direction"]), mtf_trends)
        mtf_ok = scalper_mtf_ok or self._mtf_allows(str(signal["direction"]), mtf_trends)
        turbo_mtf_override = False
        if (not mtf_ok) and ny_turbo:
            m1 = mtf_trends.get("M1", Trend.RANGE)
            h1 = mtf_trends.get("H1", Trend.RANGE)
            d = str(signal["direction"]).upper()
            if (d == "SELL" and h1 == Trend.UP) or (d == "BUY" and h1 == Trend.DOWN):
                turbo_mtf_override = False
            elif (d == "BUY" and m1 == Trend.UP) or (d == "SELL" and m1 == Trend.DOWN):
                turbo_mtf_override = (
                    continuation_now >= 0.52
                    and momentum_score >= 0.42
                )
            mtf_ok = turbo_mtf_override
        ml_strong_mtf_override = False
        if (not mtf_ok) and float(ml_conf or 0.0) >= self._mtf_ml_strong_override_th:
            d = str(signal["direction"]).upper()
            # CRÍTICO: el override sólo aplica si ML "vota" en la misma dirección
            # que la señal. Si el ML está fuerte en sentido opuesto, no
            # forzamos la entrada (lo dejamos para que ml_veto/decision_hierarchy decidan).
            ml_dir_pref = "BUY" if float(prob_buy) >= float(prob_sell) else "SELL"
            ml_gap_pref = abs(float(prob_buy) - float(prob_sell))
            # v4.1: relajamos el gap (0.05 -> 0.10) y aplicamos detección de
            # sesgo extremo del modelo. Si en las últimas N predicciones el
            # modelo solo predice un sentido, ya no podemos confiar en
            # ml_dir_pref como filtro de override y dejamos pasar el override
            # siempre que la señal de estrategia tenga otras confirmaciones.
            bias_ratio_buy, bias_ratio_sell = self._ml_direction_bias_ratio(symbol)
            # v4.1 HÍBRIDO: severe_bias se dispara al 0.80 (igual que el
            # bias_monitor.warn_ratio) en lugar de 0.85, y además marcamos
            # `bias_against_signal` para ignorar `ml_dir_pref` cuando el
            # modelo está claramente sesgado contra la señal. Esto evita los
            # "ML strong override descartado" cuando XAUUSD predice 100% BUY
            # contra una señal de estrategia SELL (caso visto en los logs).
            severe_bias = (
                bias_ratio_buy >= 0.80 and d == "SELL"
            ) or (
                bias_ratio_sell >= 0.80 and d == "BUY"
            )
            bias_against_signal = (
                (d == "SELL" and bias_ratio_buy >= 0.70)
                or (d == "BUY" and bias_ratio_sell >= 0.70)
            )
            ml_aligned = (
                (ml_dir_pref == d)
                or (ml_gap_pref < 0.10)
                or severe_bias
                or bias_against_signal
            )
            m1 = mtf_trends.get("M1", Trend.RANGE)
            m5 = mtf_trends.get("M5", Trend.RANGE)
            h1 = mtf_trends.get("H1", Trend.RANGE)
            h4 = mtf_trends.get("H4", Trend.RANGE)
            d1 = mtf_trends.get("D1", Trend.RANGE)
            macro_contra = False
            if d == "BUY" and (h4 == Trend.DOWN or d1 == Trend.DOWN):
                macro_contra = True
            elif d == "SELL" and (h4 == Trend.UP or d1 == Trend.UP):
                macro_contra = True
            short_term_ok = False
            if d == "BUY":
                short_term_ok = (m1 == Trend.UP) or (m5 == Trend.UP) or (h1 == Trend.UP)
            elif d == "SELL":
                short_term_ok = (m1 == Trend.DOWN) or (m5 == Trend.DOWN) or (h1 == Trend.DOWN)
            if ml_aligned and (not macro_contra) and short_term_ok:
                ml_strong_mtf_override = True
                mtf_ok = True
            elif not ml_aligned:
                logger.info(
                    "%s | ML strong override descartado: ML contradice señal "
                    "(señal=%s ml_dir=%s gap=%.3f ml=%.3f)",
                    symbol,
                    d,
                    ml_dir_pref,
                    ml_gap_pref,
                    float(ml_conf or 0.0),
                )
        # AURION X 4.1 — SMC override del filtro MTF para operar pullbacks
        # institucionales. Permite entrar contra-tendencia inmediata SI el
        # bias institucional SMC confirma la dirección (caso pullback dentro
        # de estructura BULLISH/BEARISH macro). Requiere confluencia alta y
        # short_term_ok para no abrir trades aleatorios.
        if (not mtf_ok) and smc_snapshot is not None:
            try:
                d = str(signal.get("direction", "")).upper()
                macro_b = getattr(smc_snapshot.macro_bias, "value", "NEUTRAL")
                inst_b = getattr(smc_snapshot.institutional_bias, "value", "NEUTRAL")
                conf_score = float(
                    getattr(getattr(smc_snapshot, "confluence", None), "score", 0.0)
                )
                smc_dir_aligned = (
                    (d == "BUY" and macro_b == "BULLISH" and inst_b in ("BULLISH", "NEUTRAL"))
                    or (d == "SELL" and macro_b == "BEARISH" and inst_b in ("BEARISH", "NEUTRAL"))
                )
                m1 = mtf_trends.get("M1", Trend.RANGE)
                m5 = mtf_trends.get("M5", Trend.RANGE)
                h1 = mtf_trends.get("H1", Trend.RANGE)
                short_term_ok2 = False
                if d == "BUY":
                    short_term_ok2 = (m1 == Trend.UP) or (m5 == Trend.UP) or (h1 == Trend.UP)
                elif d == "SELL":
                    short_term_ok2 = (m1 == Trend.DOWN) or (m5 == Trend.DOWN) or (h1 == Trend.DOWN)
                # Verificamos que haya AL MENOS UN OB activo del lado de la
                # señal en CUALQUIER TF del snapshot (entry institucional real).
                ob_aligned = False
                ob_tf_found = ""
                try:
                    for tf_name, tfc in (getattr(smc_snapshot, "per_tf", {}) or {}).items():
                        for ob in getattr(tfc, "active_obs", []) or []:
                            ob_dir = getattr(ob, "direction", None)
                            ob_dir_val = getattr(ob_dir, "value", str(ob_dir))
                            if (d == "BUY" and ob_dir_val == "BULLISH") or (
                                d == "SELL" and ob_dir_val == "BEARISH"
                            ):
                                ob_aligned = True
                                ob_tf_found = tf_name
                                break
                        if ob_aligned:
                            break
                except Exception:
                    ob_aligned = False
                if (
                    smc_dir_aligned
                    and short_term_ok2
                    and ob_aligned
                    and float(ml_conf or 0.0) >= 0.60
                ):
                    mtf_ok = True
                    # También marcamos ml_strong_mtf_override=True para que los
                    # filtros posteriores (htf_counter_scalper, sniper_dead_hour,
                    # etc.) consideren este override igual que un ML strong.
                    ml_strong_mtf_override = True
                    logger.info(
                        "%s | SMC pullback override: macro=%s inst=%s OB@%s "
                        "ml_conf=%.2f → permite %s en MTF mixto (operando pullback "
                        "dentro de estructura institucional)",
                        symbol, macro_b, inst_b, ob_tf_found,
                        float(ml_conf or 0.0), d,
                    )
                else:
                    # Diagnóstico throttled (1 cada 60s por símbolo) para entender
                    # por qué el override no se dispara y poder afinar.
                    now_ts = datetime.now(timezone.utc).timestamp()
                    if not hasattr(self, "_smc_ovr_diag_ts"):
                        self._smc_ovr_diag_ts: dict[str, float] = {}
                    last = self._smc_ovr_diag_ts.get(symbol, 0.0)
                    if (now_ts - last) >= 60.0:
                        self._smc_ovr_diag_ts[symbol] = now_ts
                        logger.info(
                            "%s | SMC override no aplicable: dir_aligned=%s "
                            "short_term_ok=%s ob_aligned=%s ml_conf=%.2f "
                            "macro=%s inst=%s",
                            symbol, smc_dir_aligned, short_term_ok2,
                            ob_aligned, float(ml_conf or 0.0), macro_b, inst_b,
                        )
            except Exception as exc:
                logger.debug("[SMC] override mtf falló: %s", exc)
        # AURION X 4.2 — segundo override MTF: mayoría estructural + operational
        # bias. Permite operar cuando macro_bias no apoya pero la mayoría de TFs
        # estructurales (M15+M30+H1+D1) Y el bias operativo SMC sí coinciden con
        # la señal. Caso típico: pullback de fin de impulso H4 alcista cuando el
        # nuevo impulso bajista ya consolidó en M15/M30/D1 con OB institucional.
        if (not mtf_ok) and smc_snapshot is not None:
            try:
                d = str(signal.get("direction", "")).upper()
                op_b = getattr(smc_snapshot.operational_bias, "value", "NEUTRAL")
                conf_score = float(
                    getattr(getattr(smc_snapshot, "confluence", None), "score", 0.0)
                )
                op_aligned = (
                    (d == "BUY" and op_b == "BULLISH")
                    or (d == "SELL" and op_b == "BEARISH")
                )
                structural_tfs = ("M15", "M30", "H1", "D1")
                target = Trend.UP if d == "BUY" else Trend.DOWN if d == "SELL" else None
                aligned_count = 0
                snap_str = []
                for tf in structural_tfs:
                    tf_trend = mtf_trends.get(tf, Trend.RANGE)
                    snap_str.append(f"{tf}:{tf_trend.value}")
                    if tf_trend == target:
                        aligned_count += 1
                ob_aligned2 = False
                try:
                    for _tf_name, tfc in (getattr(smc_snapshot, "per_tf", {}) or {}).items():
                        for ob in getattr(tfc, "active_obs", []) or []:
                            ob_dir_val = getattr(getattr(ob, "direction", None), "value", "")
                            if (d == "BUY" and ob_dir_val == "BULLISH") or (
                                d == "SELL" and ob_dir_val == "BEARISH"
                            ):
                                ob_aligned2 = True
                                break
                        if ob_aligned2:
                            break
                except Exception:
                    ob_aligned2 = False
                # Nota: NO usamos conf_score (a menudo es 0 fuera de setups
                # explícitos). Los 4 filtros restantes ya garantizan calidad:
                # mayoría estructural + bias operativo + OB activo + ML alto.
                ok_all = (
                    op_aligned
                    and aligned_count >= 3                  # ≥3 de 4 TFs estructurales
                    and ob_aligned2
                    and float(ml_conf or 0.0) >= 0.65       # ML confianza alta
                )
                if ok_all:
                    mtf_ok = True
                    ml_strong_mtf_override = True
                    logger.info(
                        "%s | SMC structural-majority override: op_bias=%s "
                        "aligned_tfs=%d/4 [%s] ml_conf=%.2f conf=%.2f → permite %s",
                        symbol, op_b, aligned_count, ", ".join(snap_str),
                        float(ml_conf or 0.0), conf_score, d,
                    )
                else:
                    now_ts2 = datetime.now(timezone.utc).timestamp()
                    if not hasattr(self, "_smc_smaj_diag_ts"):
                        self._smc_smaj_diag_ts: dict[str, float] = {}
                    last2 = self._smc_smaj_diag_ts.get(symbol, 0.0)
                    if (now_ts2 - last2) >= 60.0:
                        self._smc_smaj_diag_ts[symbol] = now_ts2
                        logger.info(
                            "%s | SMC structural-majority NO aplicable: "
                            "op_aligned=%s (op_b=%s d=%s) aligned_tfs=%d/4 [%s] "
                            "ob_aligned=%s ml_conf=%.2f conf_score=%.2f",
                            symbol, op_aligned, op_b, d, aligned_count,
                            ", ".join(snap_str), ob_aligned2,
                            float(ml_conf or 0.0), conf_score,
                        )
            except Exception as exc:
                logger.warning("[SMC] structural-majority override falló: %s", exc)
        if self._skip_mtf_filter:
            mtf_ok = True
        if not mtf_ok:
            ordered = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
            snap = ", ".join(f"{tf}:{mtf_trends.get(tf, Trend.RANGE).value}" for tf in ordered)
            self._maybe_log_entry_skip(
                symbol,
                "mtf_align",
                "%s skip: MTF no alineado con %s | %s",
                symbol,
                signal["direction"],
                snap,
            )
            blocked_reason = "mtf_align"
            self._learn_from_decision(
                symbol=symbol,
                blocked_reason=blocked_reason,
                strong_impulse_entry=False,
                continuation_now=continuation_now,
                effective_conf=0.0,
                threshold=self._ml_threshold_floor,
            )
            self._execute_management(symbol, tick)
            return
        if turbo_mtf_override:
            logger.info("%s | override MTF por NY turbo (cont=%.2f mom=%.2f)", symbol, continuation_now, momentum_score)
        if ml_strong_mtf_override:
            logger.info(
                "%s | override MTF por ML strong (ml=%.3f th=%.2f dir=%s)",
                symbol,
                float(ml_conf or 0.0),
                self._mtf_ml_strong_override_th,
                signal["direction"],
            )

        if self._skip_scalper_extreme_m5_range:
            regime_chop = self._volatility.regime()
            if (
                strategy_name == "SCALPER"
                and regime_chop == VolatilityRegime.EXTREME
                and mtf_trends.get("M5", Trend.RANGE) == Trend.RANGE
                and "RANGE_FADE" not in setup_type
            ):
                self._maybe_log_entry_skip(
                    symbol,
                    "scalper_extreme_m5_range",
                    (
                        "%s skip: scalper en volatilidad EXTREME con M5:RANGE y setup=%s "
                        "(chop; execution.skip_scalper_extreme_m5_range=false para permitir)"
                    ),
                    symbol,
                    setup_type,
                )
                self._learn_from_decision(
                    symbol=symbol,
                    blocked_reason="scalper_extreme_m5_range",
                    strong_impulse_entry=False,
                    continuation_now=continuation_now,
                    effective_conf=0.0,
                    threshold=self._ml_threshold_floor,
                )
                self._execute_management(symbol, tick)
                return

        # Guard estructural: RANGE_BREAKOUT_MOMENTUM en chop multi-TF.
        # En los logs nocturnos este setup acumuló avgR=-3.17 en XAUUSD (4
        # trades, 25% wr). Si M5+M15+M30 están todos en RANGE el "breakout"
        # casi siempre es ruido y se revierte. Bloqueamos a no ser que el ML
        # confirme con strong override alineado (donde sí podemos asumir un
        # impulso real corto plazo).
        if (
            "RANGE_BREAKOUT_MOMENTUM" in setup_type
            and not ml_strong_mtf_override
        ):
            _m5 = mtf_trends.get("M5", Trend.RANGE)
            _m15 = mtf_trends.get("M15", Trend.RANGE)
            _m30 = mtf_trends.get("M30", Trend.RANGE)
            _h1 = mtf_trends.get("H1", Trend.RANGE)
            _h4 = mtf_trends.get("H4", Trend.RANGE)
            _range_count = sum(
                1 for t in (_m5, _m15, _m30, _h1, _h4) if t == Trend.RANGE
            )
            if _range_count >= 4:
                self._maybe_log_entry_skip(
                    symbol,
                    "range_breakout_in_chop",
                    (
                        "%s skip: RANGE_BREAKOUT_MOMENTUM en chop multi-TF "
                        "(M5=%s M15=%s M30=%s H1=%s H4=%s) — setup tóxico"
                    ),
                    symbol,
                    _m5.value,
                    _m15.value,
                    _m30.value,
                    _h1.value,
                    _h4.value,
                    setup_type=setup_type,
                )
                self._learn_from_decision(
                    symbol=symbol,
                    blocked_reason="range_breakout_in_chop",
                    strong_impulse_entry=False,
                    continuation_now=continuation_now,
                    effective_conf=0.0,
                    threshold=self._ml_threshold_floor,
                )
                self._execute_management(symbol, tick)
                return

        # Guard "chop persistente macro": cuando H1+H4+D1+W1 están TODOS en
        # RANGE durante un período prolongado (configurable, def. 30 min) el
        # mercado está en consolidación estructural. En los logs nocturnos
        # H1+H4+D1+W1+MN1 estaban todos RANGE durante horas y los setups
        # momentum/breakout perdieron consistentemente. Permitimos solo setups
        # mean-reverting (RANGE_FADE_REVERSAL, MEAN_REVERSION) y bloqueamos el
        # resto, a no ser que ML strong override esté activo con grade A+.
        _h1m = mtf_trends.get("H1", Trend.RANGE)
        _h4m = mtf_trends.get("H4", Trend.RANGE)
        _d1m = mtf_trends.get("D1", Trend.RANGE)
        _w1m = mtf_trends.get("W1", Trend.RANGE)
        _macro_all_range = (
            _h1m == Trend.RANGE
            and _h4m == Trend.RANGE
            and _d1m == Trend.RANGE
            and _w1m == Trend.RANGE
        )
        if _macro_all_range:
            sym_state = self._chop_macro_state.setdefault(
                symbol.upper(), {"since_mono": None, "active": False}
            )
            if sym_state["since_mono"] is None:
                sym_state["since_mono"] = monotonic()
            duration = monotonic() - float(sym_state["since_mono"] or monotonic())
            sym_state["active"] = duration >= self._chop_macro_min_duration_s
        else:
            self._chop_macro_state[symbol.upper()] = {
                "since_mono": None,
                "active": False,
            }

        chop_macro_active = self._chop_macro_state.get(
            symbol.upper(), {}
        ).get("active", False)
        # En chop persistente solo permitimos setups mean-reverting. Si el
        # ML tiene strong override permitimos también momentum (alta convicción
        # combinada que sí filtró señales mediocres).
        if chop_macro_active and not ml_strong_mtf_override:
            mean_revert_setups = (
                "RANGE_FADE_REVERSAL",
                "RANGE_FADE",
                "MEAN_REVERSION",
                "RANGE_REVERSAL",
            )
            is_mean_revert = any(s in setup_type for s in mean_revert_setups)
            if not is_mean_revert:
                self._maybe_log_entry_skip(
                    symbol,
                    "chop_macro_persistente",
                    (
                        "%s skip: chop macro persistente "
                        "(H1=%s H4=%s D1=%s W1=%s, dur>=%ds) — solo fade reversal"
                    ),
                    symbol,
                    _h1m.value,
                    _h4m.value,
                    _d1m.value,
                    _w1m.value,
                    int(self._chop_macro_min_duration_s),
                    setup_type=setup_type,
                )
                self._learn_from_decision(
                    symbol=symbol,
                    blocked_reason="chop_macro_persistente",
                    strong_impulse_entry=False,
                    continuation_now=continuation_now,
                    effective_conf=0.0,
                    threshold=self._ml_threshold_floor,
                )
                self._execute_management(symbol, tick)
                return

        if (
            self._volatility.regime() == VolatilityRegime.EXTREME
            and volume_score <= 0.0
            and volume_ratio <= 0.0
            and (
                "RANGE_BREAKOUT" in setup_type
                or "MICRO_PULLBACK" in setup_type
                or "RANGE_FADE" in setup_type
            )
        ):
            # ML strong escape: muchos brokers MT5 reportan tickvolume=0 en demos /
            # condiciones de mercado lentas. Si el modelo tiene convicción alta
            # en la dirección, o el orquestador ya activó ml_strong_mtf_override
            # (con alineación short-term + sin contradicción macro), la falta de
            # volumen tick es ruido y no debe bloquear. Apex sigue siendo la
            # última red de seguridad. Umbral configurable vía
            # execution.extreme_no_volume_ml_strong_th (def. 0.58).
            ml_raw_for_vol = float(ml_conf) if ml_conf else 0.0
            extreme_no_vol_th = float(
                (self._config.get("execution", {}) or {}).get(
                    "extreme_no_volume_ml_strong_th", 0.58
                )
            )
            if (ml_raw_for_vol < extreme_no_vol_th) and (not ml_strong_mtf_override):
                self._maybe_log_entry_skip(
                    symbol,
                    "extreme_chop_no_volume",
                    (
                        "%s skip: EXTREME sin volumen (volScore=%.2f volRatio=%.2f) setup=%s "
                        "(chop disfrazado; momentum/breakout bloqueado)"
                    ),
                    symbol,
                    volume_score,
                    volume_ratio,
                    setup_type,
                )
                self._learn_from_decision(
                    symbol=symbol,
                    blocked_reason="extreme_chop_no_volume",
                    strong_impulse_entry=False,
                    continuation_now=continuation_now,
                    effective_conf=0.0,
                    threshold=self._ml_threshold_floor,
                )
                self._execute_management(symbol, tick)
                return

        if (
            self._predictor_range is not None
            and getattr(self._predictor_range, "model", None) is not None
            and (bool(signal.get("range_fade")) or "RANGE_FADE" in setup_type)
        ):
            try:
                ml_range = self._predictor_range.predict(features_vec)
                prob_buy = float(ml_range["prob_buy"])
                prob_sell = float(ml_range["prob_sell"])
                ml_conf = float(ml_range["confidence"])
            except Exception as exc:
                logger.debug("%s ML RANGE omitido: %s", symbol, exc)

        # Alinear lado de señal con ML mayoritario (opcional según ml.veto_contradictions).
        # Refinamientos:
        #   * Si `ml_strong_mtf_override` está activo, ya validamos alineación short-term
        #     (M1/M5/H1) y ausencia de contradicción macro (H4/D1) en la dirección de la
        #     SEÑAL. En ese caso el veto se suprime (antes el override permitía pasar el
        #     filtro MTF pero el veto mataba la entrada inmediatamente).
        #   * Adicionalmente exigimos un gap mínimo |prob_buy - prob_sell| para que el
        #     veto se aplique: evita vetar por empates (0.50/0.50) cuando ml_conf alta
        #     es producto de probabilidades casi equivalentes.
        if self._ml_veto_enabled:
            ml_dir = "BUY" if prob_buy >= prob_sell else "SELL"
            self._track_ml_dir(symbol, prob_buy, prob_sell)
            ml_cfg = self._config.get("ml", {}) or {}
            ml_veto_threshold = float(ml_cfg.get("veto_confidence", 0.7))
            ml_veto_min_gap = float(ml_cfg.get("veto_min_gap", 0.12))
            ml_gap = abs(float(prob_buy) - float(prob_sell))
            direction_contradicts = signal["direction"] != ml_dir
            if (
                direction_contradicts
                and ml_conf >= ml_veto_threshold
                and ml_gap >= ml_veto_min_gap
                and not ml_strong_mtf_override
            ):
                self._maybe_log_entry_skip(
                    symbol,
                    "ml_direction",
                    "%s skip: señal %s vs ML %s con conf ML %.2f >= %.2f (gap=%.3f >= %.3f)",
                    symbol,
                    signal["direction"],
                    ml_dir,
                    ml_conf,
                    ml_veto_threshold,
                    ml_gap,
                    ml_veto_min_gap,
                )
                self._execute_management(symbol, tick)
                return
            if direction_contradicts and ml_strong_mtf_override:
                logger.info(
                    "%s | ml_veto suprimido por ml_strong_mtf_override (ml=%.2f gap=%.3f señal=%s ml_dir=%s)",
                    symbol,
                    ml_conf,
                    ml_gap,
                    signal["direction"],
                    ml_dir,
                )
        self._last_signal_by_symbol[symbol] = str(signal["direction"])

        # 8) Validar entrada
        prof_val = self._profile_for_symbol(symbol)
        spread_cap_pts = float(prof_val.get("spread_high_points", self._spread_high_points))
        max_spread_abs = spread_cap_pts * max(self._pip_size(symbol), 1e-12) * 1.08
        is_valid, reason = self._entry_validator.validate(
            signal={
                "direction": signal["direction"],
                "price": self._mid_price(tick),
                "setup_type": signal.get("setup_type", "UNKNOWN"),
                "pullback": signal.get("pullback", False),
                "range_fade": signal.get("range_fade", False),
            },
            structure=structure,
            volatility=regime,
            spread=spread_now,
            mtf_filter_passed=True,
            max_spread_abs=float(max_spread_abs),
            ml_strong_override=bool(ml_strong_mtf_override),
        )
        strong_impulse_entry = (
            ("BREAKOUT" in setup_type or "MOMENTUM" in setup_type)
            and continuation_now >= 0.62
            and momentum_score >= 0.55
            and volume_score >= 0.50
        )
        if self._block_scalper_counter_htf and strategy_name in ("SCALPER", "INTRADAY_SCALPER", "INTRADAY"):
            ch_block, ch_reason = self._counter_htf_scalper_block(signal, mtf_trends)
            # v4.1 HÍBRIDO: si el ML strong override ya validó alineación de
            # corto plazo (M1/M5/H1) y M5+M15 confirman la dirección de la
            # señal, no bloqueamos por H1 contrario en scalper. Esto resuelve
            # los repetidos "bloqueo HTF: venta con H1/H4 alcista" en
            # contextos lateralizados donde la pierna de scalper sigue siendo
            # válida (mean-reversion / retrocesos en rango macro).
            d_sig_htf = str(signal.get("direction", "")).upper()
            _m5 = mtf_trends.get("M5", Trend.RANGE)
            _m15 = mtf_trends.get("M15", Trend.RANGE)
            short_term_confirms = (
                (d_sig_htf == "BUY" and _m5 == Trend.UP and _m15 == Trend.UP)
                or (d_sig_htf == "SELL" and _m5 == Trend.DOWN and _m15 == Trend.DOWN)
            )
            # v4.1 SMC: la relajación también admite alineación SOLO con M5
            # cuando hay confluencia SMC fuerte (override SMC ya validó OB).
            short_term_partial = (
                (d_sig_htf == "BUY" and _m5 == Trend.UP)
                or (d_sig_htf == "SELL" and _m5 == Trend.DOWN)
            )
            ml_override_relax = bool(ml_strong_mtf_override) and (
                short_term_confirms or short_term_partial
            )
            if ch_block and not strong_impulse_entry and not ml_override_relax:
                self._maybe_log_entry_skip(
                    symbol,
                    "htf_counter_scalper",
                    "%s skip: %s",
                    symbol,
                    ch_reason,
                )
                self._learn_from_decision(
                    symbol=symbol,
                    blocked_reason="htf_counter_scalper",
                    strong_impulse_entry=False,
                    continuation_now=continuation_now,
                    effective_conf=0.0,
                    threshold=self._ml_threshold_floor,
                )
                self._execute_management(symbol, tick)
                return
            if ch_block and ml_override_relax:
                logger.info(
                    "%s | HTF counter-scalper relajado por ML strong + M5/M15 alineados (dir=%s)",
                    symbol,
                    d_sig_htf,
                )
        if self._sniper_enabled and self._sniper_blocks_dead_hour(tick.time_utc):
            # v4.1 HÍBRIDO: además del strong_impulse_entry clásico, ahora
            # también permitimos el bypass de "sniper dead hour" cuando hay
            # ML strong override + dirección alineada + M5 confirma. Logs
            # mostraban bloqueos en UTC=22 con ml~0.70-0.79 alineado al
            # micro: esos contextos no son hora muerta real para XAUUSD.
            _d_sn = str(signal.get("direction", "")).upper()
            _m5_sn = mtf_trends.get("M5", Trend.RANGE)
            _ml_dir_sn = "BUY" if float(prob_buy) >= float(prob_sell) else "SELL"
            sniper_ml_bypass = (
                bool(ml_strong_mtf_override)
                and (
                    (_d_sn == "BUY" and _m5_sn == Trend.UP)
                    or (_d_sn == "SELL" and _m5_sn == Trend.DOWN)
                )
                and (
                    _ml_dir_sn == _d_sn
                    or abs(float(prob_buy) - float(prob_sell)) < 0.10
                )
            )
            if not (
                (self._sniper_allow_strong_override and strong_impulse_entry)
                or sniper_ml_bypass
            ):
                blocked_reason = "sniper_dead_hour"
                self._maybe_log_entry_skip(
                    symbol,
                    "sniper_dead_hour",
                    "%s skip: sniper bloquea hora muerta UTC=%02d (setup=%s)",
                    symbol,
                    tick.time_utc.hour,
                    setup_type,
                )
                self._execute_management(symbol, tick)
                return
            if sniper_ml_bypass:
                logger.info(
                    "%s | sniper dead hour bypass por ML strong + dir aligned (UTC=%02d ml=%.3f)",
                    symbol,
                    tick.time_utc.hour,
                    float(ml_conf or 0.0),
                )
        ml_cfg_gate = self._config.get("ml", {}) or {}
        if (
            bool(ml_cfg_gate.get("skip_range_fade_on_extreme_vol", True))
            and regime == VolatilityRegime.EXTREME
            and "RANGE_FADE" in setup_type
        ):
            blocked_reason = "extreme_vol_range_fade"
            pre_eff = max(float(signal.get("confidence_base", 0.0)), ml_conf)
            self._maybe_log_entry_skip(
                symbol,
                blocked_reason,
                "%s skip: volatilidad EXTREME + setup RANGE_FADE (telemetría débil; ml.skip_range_fade_on_extreme_vol=false para permitir)",
                symbol,
            )
            self._learn_from_decision(
                symbol=symbol,
                blocked_reason=blocked_reason,
                strong_impulse_entry=strong_impulse_entry,
                continuation_now=continuation_now,
                effective_conf=pre_eff,
                threshold=self._ml_threshold_floor,
            )
            self._execute_management(symbol, tick)
            return
        if (not is_valid) and ("volatilidad EXTREME" in reason) and strong_impulse_entry and "RANGE_FADE" not in setup_type:
            logger.info(
                "%s | override validador por impulso fuerte (setup=%s cont=%.2f mom=%.2f)",
                symbol,
                setup_type,
                continuation_now,
                momentum_score,
            )
            is_valid = True
            reason = "ok_impulse_override"

        # 9) threshold + valid -> lot + order
        mode = self._detect_mode(
            tick=tick,
            symbol=symbol,
            volatility=regime,
            trends=mtf_trends,
            direction=str(signal["direction"]),
        )
        profile = self._profile_for_symbol(symbol)
        threshold_map = profile.get("ml_thresholds", {})
        if isinstance(threshold_map, dict):
            if mode == "CONSERVATIVE":
                threshold = max(self._ml_threshold_floor, float(threshold_map.get("conservative", 0.58)))
            elif mode == "AGGRESSIVE_CONTROLLED":
                threshold = max(self._ml_threshold_floor, float(threshold_map.get("aggressive", 0.42)))
            else:
                threshold = max(self._ml_threshold_floor, float(threshold_map.get("normal", 0.48)))
        else:
            threshold = self._adaptive_ml_threshold(mode)
        if inst_dec is not None:
            effective_conf = inst_dec.quant.effective_confidence
            threshold = max(
                self._ml_threshold_floor,
                threshold + inst_dec.threshold_delta,
            )
            score_bias_inst = inst_dec.score_bias
        elif quant_dec is not None:
            effective_conf = quant_dec.effective_confidence
            threshold = max(self._ml_threshold_floor, threshold + quant_dec.threshold_delta)
            score_bias_inst = 0
        else:
            score_bias_inst = 0
            effective_conf = max(float(signal.get("confidence_base", 0.0)), ml_conf)
        if not self._ml_enabled and quant_dec is None:
            boost = float(self._ml_disabled_confidence_boost)
            effective_conf = min(0.95, effective_conf + boost)
        adaptive_bias = float(self._adaptive_entry_bias.get(symbol.upper(), 0.0))
        perf_bias = float(self._adaptive_perf_bias.get(symbol.upper(), 0.0))
        threshold = max(self._ml_threshold_floor, threshold + adaptive_bias + perf_bias)
        sniper_key = self._sniper_key(setup_type, regime, tick.time_utc)
        bucket_thr_bias, bucket_score_bias, bucket_risk_mult = self._sniper_bucket_adjustments(symbol, sniper_key)
        online_thr_bias, online_score_bias, online_risk_mult = self._online_setup_adjustments(symbol, setup_type)
        threshold = max(self._ml_threshold_floor, threshold + bucket_thr_bias + online_thr_bias)
        if not self._ml_enabled:
            relax = float(self._ml_disabled_threshold_relax)
            threshold = max(self._ml_threshold_floor, threshold - relax)
            # Sin modelo ML, no dejar que el aprendizaje online suba demasiado el umbral.
            cap = float((self._config.get("ml", {}) or {}).get("disabled_max_threshold_cap", 0.36))
            if cap > self._ml_threshold_floor:
                threshold = min(threshold, cap)
        if ny_turbo and ("BREAKOUT" in setup_type or "MOMENTUM" in setup_type):
            threshold = max(self._ml_threshold_floor, threshold - 0.03)
        if strong_impulse_entry and mode == "CONSERVATIVE":
            threshold = max(self._ml_threshold_floor, threshold - 0.05)
        ext_add = float((self._config.get("ml", {}) or {}).get("extreme_vol_ml_threshold_add", 0.0))
        if regime == VolatilityRegime.EXTREME and ext_add > 0:
            threshold = min(0.94, threshold + ext_add)
        rb_m5_add = float((self._config.get("ml", {}) or {}).get("extreme_range_breakout_m5_range_ml_threshold_add", 0.0))
        if (
            regime == VolatilityRegime.EXTREME
            and rb_m5_add > 0
            and "RANGE_BREAKOUT" in setup_type
            and mtf_trends.get("M5", Trend.RANGE) == Trend.RANGE
        ):
            threshold = min(0.94, threshold + rb_m5_add)
        wd_thr = self._weekday_policy_for(tick.time_utc)
        if wd_thr.ml_threshold_increase > 0:
            threshold = min(0.99, threshold + wd_thr.ml_threshold_increase)
        # Cap absoluto al umbral adaptativo por modo para evitar que la suma de
        # biases (institutional + adaptive + perf + bucket + online + weekday +
        # extreme_add) ahogue la operativa cuando el bot está aprendiendo.
        # Valores anteriores producían umbrales de 0.71+ en CONSERVATIVE,
        # impidiendo cualquier entrada incluso con ML=0.70.
        _ml_cfg_caps = (self._config.get("ml", {}) or {}).get("adaptive_threshold_caps", {}) or {}
        if mode == "CONSERVATIVE":
            _cap = float(_ml_cfg_caps.get("conservative", 0.58))
        elif mode == "AGGRESSIVE_CONTROLLED":
            _cap = float(_ml_cfg_caps.get("aggressive", 0.48))
        else:
            _cap = float(_ml_cfg_caps.get("normal", 0.52))
        threshold = min(threshold, max(self._ml_threshold_floor, _cap))
        if self._edge_fusion_enabled and self._ml_enabled:
            d_f = str(signal["direction"]).upper()
            p_al = prob_buy if d_f == "BUY" else prob_sell
            if p_al >= self._edge_fusion_ml_prob_min:
                effective_conf = min(0.95, effective_conf + float(self._edge_fusion_ml_prob_aligned_boost))
        # Cuando ML strong override está activo (y, por Fix #1, ya validamos
        # que ML está alineado con la señal) damos un pequeño empujón a la
        # confianza efectiva para que no se quede colgada justo por debajo
        # del cap adaptativo. Tope conservador (+0.04) para no inflar señales
        # mediocres.
        if ml_strong_mtf_override and self._ml_enabled:
            effective_conf = min(0.95, effective_conf + 0.04)
        # AURION X 4.1 — Modulación SMC sobre effective_confidence (soft).
        # Sólo se aplica si la capa SMC está activa y publicó snapshot para
        # este símbolo. NUNCA bloquea: a lo sumo modula +/- max_conf_delta.
        smc_delta_conf = 0.0
        smc_grade_bump = 0
        if getattr(self, "_smc_analyzer", None) is not None and smc_snapshot is not None:
            try:
                smc_delta_conf, smc_grade_bump = self._smc_analyzer.score_modifier(
                    symbol,
                    str(signal.get("direction", "")),
                )
                if abs(smc_delta_conf) > 1e-9 or smc_grade_bump != 0:
                    effective_conf = max(
                        0.0,
                        min(0.95, effective_conf + float(smc_delta_conf)),
                    )
                    logger.info(
                        "[SMC] %s modulación: conf %+.3f grade_bump=%+d "
                        "(confluence=%.3f macro=%s inst=%s op=%s)",
                        symbol,
                        smc_delta_conf,
                        smc_grade_bump,
                        float(smc_snapshot.confluence.confluence_score),
                        smc_snapshot.macro_bias.value,
                        smc_snapshot.institutional_bias.value,
                        smc_snapshot.operational_bias.value,
                    )
                    features_dict["smc_delta_conf"] = float(smc_delta_conf)
                    features_dict["smc_grade_bump"] = int(smc_grade_bump)
            except Exception as exc:
                logger.debug("[SMC] score_modifier falló: %s", exc)
        score, score_parts = self._compute_entry_score(
            direction=str(signal["direction"]),
            trends=mtf_trends,
            structure=structure,
            volatility=regime,
            spread_points=spread_points,
            symbol=symbol,
            signal=signal,
            state=state,
        )
        if bucket_score_bias != 0:
            score += bucket_score_bias
            score_parts["sniper_bucket_bias"] = bucket_score_bias
        if online_score_bias != 0:
            score += online_score_bias
            score_parts["online_setup_bias"] = online_score_bias
        if score_bias_inst != 0:
            score += score_bias_inst
            score_parts["institutional_score_bias"] = score_bias_inst
        if inst_dec is not None and self._institutional_engine is not None:
            score = self._institutional_engine.blend_rule_score(
                score,
                composite=inst_dec.quant.composite_score,
            )
            score_parts["setup_grade"] = int(
                {"A+": 95, "A": 80, "B": 60, "C": 40, "X": 0}.get(inst_dec.setup_rank.grade.value, 50),
            )
            score_parts["trading_mode"] = hash(inst_dec.trading_mode) % 100
        elif quant_dec is not None and self._quant_engine is not None:
            score = self._quant_engine.blend_rule_score(score, composite=quant_dec.composite_score)
            score_parts["quant_composite"] = int(quant_dec.composite_score * 100)
            mem = quant_dec.memory_insight
            if mem.get("why") == "historical_drag":
                threshold = min(0.99, threshold + 0.03)
                score_parts["quant_memory_penalty"] = -5
            elif mem.get("why") == "historical_edge":
                score_parts["quant_memory_boost"] = 4
        score_min_normal = int(profile.get("score_min_normal", self._score_min_normal))
        score_min_strong = int(profile.get("score_min_strong", self._score_min_strong))
        score_floor = 30
        if adaptive_bias < 0:
            relax = int(abs(adaptive_bias) * 100)
            score_min_normal = max(score_floor, score_min_normal - relax)
            score_min_strong = max(score_min_normal + 4, score_min_strong - max(2, relax // 2))
        if perf_bias < 0:
            score_min_normal = max(score_floor, score_min_normal - 2)
            score_min_strong = max(score_min_normal + 4, score_min_strong - 2)
        elif perf_bias > 0:
            score_min_normal = min(70, score_min_normal + 2)
            score_min_strong = min(85, score_min_strong + 2)
        if ny_turbo and ("BREAKOUT" in setup_type or "MOMENTUM" in setup_type):
            score_min_normal = max(score_floor, score_min_normal - 3)
            score_min_strong = max(score_min_normal + 4, score_min_strong - 2)
        if strong_impulse_entry:
            score_min_normal = max(score_floor, score_min_normal - 4)
            score_min_strong = max(score_min_normal + 4, score_min_strong - 3)
        if self._entry_score_floor > 0:
            score = max(score, self._entry_score_floor)
        score_bucket = "STRONG" if score >= score_min_strong else "NORMAL" if score >= score_min_normal else "REJECT"
        rr_preview = max(float(signal.get("tp_pips", 0.0)), 1e-9) / max(float(signal.get("sl_pips", 0.0)), 1e-9)
        hint_stage = "CONFIRMED" if score_bucket == "STRONG" else "EARLY"
        allowed_keys = self._sniper_whitelist.get(symbol.upper(), set())
        if (
            self._sniper_enabled
            and self._sniper_require_whitelist
            and allowed_keys
            and (sniper_key not in allowed_keys)
        ):
            if not (self._sniper_allow_strong_override and strong_impulse_entry and score_bucket == "STRONG"):
                blocked_reason = "sniper_whitelist"
                self._maybe_log_entry_skip(
                    symbol,
                    "sniper_whitelist",
                    "%s skip: sniper no autoriza bucket=%s (clave=%s)",
                    symbol,
                    score_bucket,
                    sniper_key,
                )
                self._learn_from_decision(
                    symbol=symbol,
                    blocked_reason=blocked_reason,
                    strong_impulse_entry=strong_impulse_entry,
                    continuation_now=continuation_now,
                    effective_conf=effective_conf,
                    threshold=threshold,
                )
                self._execute_management(symbol, tick)
                return
        # Hints "tempranos" solo en modo diagnóstico; por defecto el gráfico refleja órdenes que sí se envían.
        if not self._chart_hints_require_trade_ready:
            self._publish_chart_hint(
                symbol=symbol,
                direction=str(signal["direction"]),
                confidence=effective_conf,
                score=score,
                rr=rr_preview,
                stage=hint_stage,
                setup_type=setup_type,
                when_utc=tick.time_utc,
            )
        if not is_valid:
            blocked_reason = "validator"
            self._maybe_log_entry_skip(
                symbol,
                "validator",
                "%s skip: validador rechaza (%s)",
                symbol,
                reason,
            )

        if score < score_min_normal:
            blocked_reason = "score_low"
            self._maybe_log_entry_skip(
                symbol,
                "score_low",
                "%s skip: score=%s (<%s) parts=%s",
                symbol,
                score,
                score_min_normal,
                score_parts,
            )
            self._learn_from_decision(
                symbol=symbol,
                blocked_reason=blocked_reason,
                strong_impulse_entry=strong_impulse_entry,
                continuation_now=continuation_now,
                effective_conf=effective_conf,
                threshold=threshold,
            )
            self._execute_management(symbol, tick)
            return

        if (effective_conf >= threshold) and is_valid:
            if self._reconcile_before_new_entry:
                self._reconcile_positions_with_broker()
            cap_block, cap_msg = self._position_caps_block(symbol)
            if cap_block:
                self._maybe_log_entry_skip(
                    symbol,
                    "position_cap",
                    "%s skip: %s",
                    symbol,
                    cap_msg,
                )
                self._execute_management(symbol, tick)
                return
            # Anti-spam: no abrir más de 1 trade por vela M1 (minuto UTC) y símbolo.
            m1_id = self._m1_candle_id(tick.time_utc)
            sym_u = symbol.upper()
            last_m1_id = self._last_entry_m1_candle_id_by_symbol.get(sym_u)
            if last_m1_id == m1_id:
                self._maybe_log_entry_skip(
                    symbol,
                    "m1_candle_cap",
                    "%s skip: tope 1 entrada por vela M1 (candle=%s)",
                    symbol,
                    m1_id,
                )
                self._execute_management(symbol, tick)
                return
            spacing = self._min_entry_spacing_for_symbol(symbol)
            if spacing > 0:
                last_m = self._last_new_entry_mono_by_symbol.get(sym_u)
                if last_m is not None and (monotonic() - last_m) < spacing:
                    self._maybe_log_entry_skip(
                        symbol,
                        "entry_spacing",
                        "%s skip: espaciado entre entradas (%.2fs < %.2fs)",
                        symbol,
                        monotonic() - last_m,
                        spacing,
                    )
                    self._execute_management(symbol, tick)
                    return
            sl_pips = float(signal["sl_pips"])
            tp_pips = float(signal["tp_pips"])
            exec_cfg = self._config.get("execution", {}) or {}
            risk_cfg = self._config.get("risk", {}) or {}
            prof = self._profile_for_symbol(symbol)
            pip_size = self._pip_size(symbol)
            entry_ref = self._mid_price(tick)
            sl_pips, tp_pips = self._normalize_entry_sl_tp(
                symbol,
                strategy_name,
                signal,
                sl_pips,
                tp_pips,
                atr_now=atr_now,
                entry_price=entry_ref,
            )
            sl_wide, sl_wide_msg = self._sl_exceeds_hard_cap(symbol, sl_pips, entry_ref)
            if sl_wide:
                self._maybe_log_entry_skip(
                    symbol,
                    "risk",
                    "%s skip: stop demasiado amplio (%s) sl=%.1f tp=%.1f",
                    symbol,
                    sl_wide_msg,
                    sl_pips,
                    tp_pips,
                )
                self._execute_management(symbol, tick)
                return
            pip_val = prof.get("pip_value_per_lot")
            if pip_val is None:
                pip_val = exec_cfg.get("pip_value_per_lot", 1.0)
            lot_cfg = {
                    "equity": float(self._equity),
                    "sl_pips": sl_pips,
                    "pip_value_per_lot": float(pip_val),
                    "risk_pct": risk_cfg.get("risk_per_trade", risk_cfg.get("risk_pct", 1.0)),
                    "risk_pct_min": risk_cfg.get("risk_pct_min", 0.40),
                    "risk_pct_max": risk_cfg.get("risk_pct_max", 2.50),
                    "capital_multiplier": self._capital_multiplier(),
                    # Usamos racha por símbolo: position_sizer reduce el lot
                    # cuando el activo lleva pérdidas, sin que las pérdidas de
                    # otros activos achiquen el tamaño aquí.
                    "loss_streak": self._risk.loss_streak_for(symbol),
                    "drawdown_pct": float(self._risk._current_drawdown_pct(self._equity)),
                    "edge_multiplier": self._edge_multiplier(effective_conf, signal, score),
                    "performance_multiplier": self._adaptive_risk_multiplier.get(symbol.upper(), 1.0) * bucket_risk_mult * online_risk_mult,
                    "lot_step": float(exec_cfg.get("lot_step", 0.01)),
                    "min_lot": float(exec_cfg.get("min_lot", 0.01)),
                    "max_lot": float(exec_cfg.get("max_lot", 1.0)),
            }
            mc_factor = 1.0
            setup_grade = "B"
            setup_lot_mult = 1.0
            regime_mode = ""
            if inst_dec is not None:
                setup_grade = inst_dec.setup_rank.grade.value
                setup_lot_mult = inst_dec.lot_multiplier
                regime_mode = inst_dec.trading_mode
                if self._institutional_engine is not None:
                    mc_factor = getattr(self._institutional_engine, "_mc_lot_factor", 1.0)
                day_start = self._risk.session_start_equity
                if day_start <= 0:
                    day_start = self._equity
                # Cooldown TEMPORAL escalonado post-pérdida POR SÍMBOLO.
                # Antes del hard block (loss_streak>=5) damos un descanso de
                # 15/45/90 min según racha (2/3/4+). Evita revenge trading
                # silencioso del MISMO símbolo entre pérdida 2 y pérdida 4
                # sin tocar otros activos. Si el setup es A+ con ML strong
                # override permitimos saltarlo (alta convicción combinada).
                cd_remaining = self._risk.post_loss_cooldown_remaining_for(
                    symbol, now_utc=tick.time_utc
                )
                if cd_remaining > 0 and not (
                    ml_strong_override and setup_grade == "A+"
                ):
                    self._maybe_log_entry_skip(
                        symbol,
                        "post_loss_cooldown",
                        "%s skip: post_loss_cooldown (%d s restantes, streak=%d)",
                        symbol,
                        cd_remaining,
                        self._risk.loss_streak_for(symbol),
                        setup_type=setup_type,
                    )
                    self._execute_management(symbol, tick)
                    return
                fp = self._institutional_engine.fundingpips_check(
                    equity=self._equity,
                    day_start_equity=day_start,
                    total_dd_pct=dd_pct,
                    trades_today=self._risk.session_trade_count,
                    trades_last_hour=self._risk.trades_in_last_hour(tick.time_utc),
                    # CRÍTICO: usamos racha POR SÍMBOLO. Antes el revenge_guard
                    # global (loss_streak) hacía que pérdidas en XAUUSD/BTCUSD
                    # bloquearan NDX100 que apenas operaba.
                    consecutive_losses=self._risk.loss_streak_for(symbol),
                    seconds_since_last_entry=(
                        monotonic() - self._last_new_entry_mono_by_symbol.get(symbol.upper(), 0.0)
                    ),
                    setup_grade=setup_grade,
                    setup_score=inst_dec.setup_rank.score,
                    winrate=recent_wr,
                    drawdown_pct=dd_pct,
                ) if self._institutional_engine else None
                if fp is not None and not fp.allow_new_trade:
                    self._maybe_log_entry_skip(symbol, "fundingpips", "%s skip: %s", symbol, fp.reason)
                    self._execute_management(symbol, tick)
                    return
                if fp is not None:
                    setup_lot_mult *= fp.lot_cap_factor
            if self._institutional_engine is not None and inst_dec is not None:
                lot = self._position_sizer.compute_lot_intelligent(
                    symbol,
                    effective_conf,
                    lot_cfg,
                    setup_grade=setup_grade,
                    setup_lot_multiplier=setup_lot_mult,
                    regime_mode=regime_mode,
                    montecarlo_factor=mc_factor,
                )
            else:
                lot = self._position_sizer.compute_lot(symbol, effective_conf, lot_cfg)
            wd_lot = self._weekday_policy_for(tick.time_utc).lot_factor
            if wd_lot < 1.0 and lot > 0:
                lot_step = float(exec_cfg.get("lot_step", 0.01))
                min_lot = float(exec_cfg.get("min_lot", 0.01))
                lot = max(min_lot, math.floor(lot * wd_lot / lot_step) * lot_step)
            if lot > 0:
                risk_price = sl_pips * pip_size
                pip_val_risk = float(
                    prof.get("pip_value_per_lot", exec_cfg.get("pip_value_per_lot", 1.0)),
                )
                proposed_risk_money = Decimal(str(lot * sl_pips * pip_val_risk))
                if self._risk.can_open_new_trade(self._equity, proposed_risk_money, now_utc=tick.time_utc):
                    corr_block, corr_msg = self._correlation_blocks_new_entry(symbol)
                    if corr_block:
                        self._maybe_log_entry_skip(
                            symbol,
                            "correlation_cap",
                            "%s skip: correlación (%s)",
                            symbol,
                            corr_msg,
                        )
                        self._execute_management(symbol, tick)
                        return
                    kill_block, kill_msg = self._intraday_kill_blocks_new_entry(symbol, tick.time_utc)
                    if kill_block:
                        self._maybe_log_entry_skip(
                            symbol,
                            "intraday_kill",
                            "%s skip: intraday kill-switch (%s)",
                            symbol,
                            kill_msg,
                        )
                        self._execute_management(symbol, tick)
                        return
                    # AURION X 4.3: filtro de calendario economico (NFP por default).
                    # Pasa ml_conf para que el override por alta confluencia funcione.
                    if self._news_calendar is not None and self._news_calendar.enabled:
                        try:
                            ml_conf_for_news = float(ml_conf) if ml_conf is not None else None
                        except (TypeError, ValueError):
                            ml_conf_for_news = None
                        in_news, event_name = self._news_calendar.in_blackout(
                            tick.time_utc, symbol=symbol, ml_conf=ml_conf_for_news,
                        )
                        if in_news:
                            self._maybe_log_entry_skip(
                                symbol,
                                "news_calendar_blackout",
                                "%s skip: news blackout activo (event=%s)",
                                symbol,
                                event_name,
                            )
                            self._execute_management(symbol, tick)
                            return
                    # ChopGuard: post-mortem 2-jun-2026.
                    if self._chop_guard is not None and self._chop_guard.enabled:
                        cd_block, cd_msg = self._chop_guard.in_cooldown(symbol, tick.time_utc)
                        if cd_block:
                            self._maybe_log_entry_skip(
                                symbol,
                                "chop_guard_cooldown",
                                "%s skip: chop_guard cooldown (%s)",
                                symbol,
                                cd_msg,
                            )
                            self._execute_management(symbol, tick)
                            return
                        rp_val: float | None = None
                        try:
                            if isinstance(features_dict, dict):
                                rp_val = features_dict.get("range_position")
                            if rp_val is None and isinstance(signal, dict):
                                ctx = signal.get("context") or signal.get("features") or {}
                                if isinstance(ctx, dict):
                                    rp_val = ctx.get("range_position") or ctx.get("position_in_range")
                        except Exception:
                            rp_val = None
                        rp_block, rp_msg = self._chop_guard.block_by_range_position(
                            signal.get("direction", ""), rp_val
                        )
                        if rp_block:
                            self._maybe_log_entry_skip(
                                symbol,
                                "chop_guard_range_pos",
                                "%s skip: chop_guard range_position (%s)",
                                symbol,
                                rp_msg,
                            )
                            self._execute_management(symbol, tick)
                            return
                        # Regla 6: blacklist de setups (RANGE_STRUCTURE_M15_FADE etc.)
                        setup_label = (
                            signal.get("setup_type")
                            or signal.get("setup")
                            or signal.get("setup_label")
                            or ""
                        )
                        bl_block, bl_msg = self._chop_guard.block_by_setup_blacklist(
                            str(setup_label or "")
                        )
                        if bl_block:
                            self._maybe_log_entry_skip(
                                symbol,
                                "chop_guard_setup_blacklist",
                                "%s skip: chop_guard setup blacklist (%s)",
                                symbol,
                                bl_msg,
                            )
                            self._execute_management(symbol, tick)
                            return
                        # Regla 5: bloquea SCALPER cuando ml_conf < min_ml_conf_scalper.
                        # FIX (4-jun): usar strategy_name (local, fiable) además de
                        # signal["strategy"]; este último podía venir vacío en algunos
                        # paths (RANGE_BREAKOUT_MOMENTUM via INTRADAY_RANGE).
                        ml_conf_val: float | None = None
                        try:
                            ml_conf_val = float(ml_conf) if ml_conf is not None else None
                        except (TypeError, ValueError):
                            ml_conf_val = None
                        strat_val = (
                            str(strategy_name or "")
                            or str(signal.get("strategy", "") or "")
                            or str(signal.get("setup", "") or "")
                        )
                        if not strat_val and isinstance(features_dict, dict):
                            strat_val = str(features_dict.get("strategy", ""))
                        ml_block, ml_msg = self._chop_guard.block_by_ml_conf_scalper(
                            strat_val, ml_conf_val, ts_utc=tick.time_utc
                        )
                        if ml_block:
                            self._maybe_log_entry_skip(
                                symbol,
                                "chop_guard_ml_conf_scalper",
                                "%s skip: chop_guard ml_conf scalper (strategy=%s ml_conf=%.3f %s)",
                                symbol,
                                strat_val,
                                ml_conf_val or 0.0,
                                ml_msg,
                            )
                            self._execute_management(symbol, tick)
                            return
                        # AURION X 4.4 (10-jun-2026): bloqueo categorico de horas
                        # con WR historico catastrofico (XAUUSD h=09 UTC: WR 0%).
                        h_block, h_msg = self._chop_guard.block_by_blocked_hour(tick.time_utc)
                        if h_block:
                            self._maybe_log_entry_skip(
                                symbol,
                                "chop_guard_blocked_hour",
                                "%s skip: %s",
                                symbol,
                                h_msg,
                            )
                            self._execute_management(symbol, tick)
                            return
                    sym_loss_block, sym_loss_msg = self._symbol_daily_loss_blocks(symbol, tick.time_utc)
                    if sym_loss_block:
                        self._maybe_log_entry_skip(
                            symbol,
                            "symbol_daily_loss",
                            "%s skip: circuit breaker pérdida diaria (%s)",
                            symbol,
                            sym_loss_msg,
                        )
                        self._execute_management(symbol, tick)
                        return
                    spread_block, spread_msg = self._spread_guard_blocks(symbol, spread_points)
                    if spread_block:
                        self._maybe_log_entry_skip(
                            symbol,
                            "spread_guard",
                            "%s skip: spread_guard bloquea entrada (%s)",
                            symbol,
                            spread_msg,
                        )
                        self._execute_management(symbol, tick)
                        return
                    # TP hunter: ajustar TP según impulso/régimen para exprimir entradas de calidad.
                    tp_mult = 1.0
                    if self._tp_hunter_enabled:
                        if strong_impulse_entry and continuation_now >= 0.70:
                            tp_mult = self._tp_hunter_mult_impulse_strong
                        elif strong_impulse_entry:
                            tp_mult = self._tp_hunter_mult_impulse
                        elif regime == VolatilityRegime.LOW:
                            tp_mult = self._tp_hunter_mult_low_vol
                        if ny_turbo and ("BREAKOUT" in setup_type or "MOMENTUM" in setup_type):
                            tp_mult = max(tp_mult, self._tp_hunter_mult_impulse)
                    tp_pips *= tp_mult
                    is_intraday = str(strategy_name).upper().startswith("INTRADAY")
                    try:
                        prof_max_rr = prof.get("max_tp_rr")
                        if prof_max_rr is not None:
                            cap_rr = float(prof_max_rr)
                        elif is_intraday:
                            cap_rr = float(self._max_tp_rr_intraday)
                        else:
                            cap_rr = float(self._max_tp_rr)
                    except (TypeError, ValueError):
                        cap_rr = float(self._max_tp_rr_intraday if is_intraday else self._max_tp_rr)
                    tp_pips = min(tp_pips, sl_pips * max(1.05, cap_rr))
                    try:
                        prof_max_tp = prof.get("max_tp_pips")
                        if prof_max_tp is not None:
                            tp_pips = min(tp_pips, float(prof_max_tp))
                    except (TypeError, ValueError):
                        pass
                    rr_now = tp_pips / max(sl_pips, 1e-9)
                    if rr_now < self._tp_hunter_min_rr:
                        self._maybe_log_entry_skip(
                            symbol,
                            "score_low",
                            "%s skip: RR insuficiente rr=%.2f sl=%.2f tp=%.2f",
                            symbol,
                            rr_now,
                            sl_pips,
                            tp_pips,
                        )
                        self._execute_management(symbol, tick)
                        return
                    if self._log_trade_attempts:
                        logger.info(
                            "[TRADE_ATTEMPT] symbol=%s side=%s lot=%.4f conf=%.3f thr=%.3f score=%s rr=%.2f setup=%s",
                            symbol,
                            signal["direction"],
                            lot,
                            effective_conf,
                            threshold,
                            score,
                            rr_now,
                            setup_type,
                        )
                    if str(signal["direction"]) == "BUY":
                        sl_price = entry_ref - risk_price
                        tp_price = entry_ref + tp_pips * pip_size
                    else:
                        sl_price = entry_ref + risk_price
                        tp_price = entry_ref - tp_pips * pip_size
                    sane, sane_msg = self._order_sl_tp_sane(
                        symbol,
                        str(signal["direction"]),
                        entry_ref,
                        sl_price,
                        tp_price,
                    )
                    if not sane:
                        self._maybe_log_entry_skip(
                            symbol,
                            "risk",
                            "%s skip: SL/TP incoherentes (%s) entry=%.5f sl=%.5f tp=%.5f",
                            symbol,
                            sane_msg,
                            entry_ref,
                            sl_price,
                            tp_price,
                        )
                        self._execute_management(symbol, tick)
                        return
                    # AURION X 4.4 (10-jun-2026): MODO OBSERVADOR
                    # Si execution.observer_mode = true, el bot analiza el
                    # mercado normalmente pero NO envia ordenes al broker.
                    # Loggea como "[OBSERVER]" la entrada que se habria abierto.
                    _obs_cfg = self._config.get("execution", {}) or {}
                    if bool(_obs_cfg.get("observer_mode", False)):
                        logger.info(
                            "[OBSERVER] %s would_open dir=%s lot=%.2f entry=%.5f "
                            "sl=%.5f tp=%.5f rr=%.2f ml_conf=%s score=%s setup=%s "
                            "(observer_mode=ON, no order sent)",
                            symbol,
                            str(signal["direction"]).upper(),
                            lot,
                            entry_ref,
                            sl_price,
                            tp_price,
                            float(rr_now),
                            f"{effective_conf:.3f}" if effective_conf is not None else "n/a",
                            f"{score:.1f}" if score is not None else "n/a",
                            setup_type or "n/a",
                        )
                        self._execute_management(symbol, tick)
                        return
                    ticket = self._order_manager.place_market_order(
                        symbol=symbol,
                        volume=lot,
                        side=str(signal["direction"]).lower(),
                        sl=sl_price,
                        tp=tp_price,
                    )
                    if ticket:
                        self._increment_weekday_trade(symbol, tick.time_utc)
                        self._last_new_entry_mono_by_symbol[symbol.upper()] = monotonic()
                        self._last_entry_m1_candle_id_by_symbol[symbol.upper()] = m1_id
                        if self._chart_hints_require_trade_ready:
                            self._publish_chart_hint(
                                symbol=symbol,
                                direction=str(signal["direction"]),
                                confidence=effective_conf,
                                score=score,
                                rr=float(rr_now),
                                stage="CONFIRMED",
                                setup_type=setup_type,
                                when_utc=tick.time_utc,
                            )
                        self._telemetry_inc(symbol, "trades_opened")
                        self._risk.register_trade_open(opened_at=tick.time_utc)
                        self._notify_ml_entry_opened(tick.time_utc, symbol=symbol)
                        # AURION X 4.1 — link SMC snapshot ↔ trade (self-learning).
                        self._smc_on_trade_opened(
                            symbol=symbol,
                            direction=str(signal["direction"]),
                            open_time=tick.time_utc,
                            trade_id=getattr(ticket, "trade_uuid", None) or getattr(ticket, "uuid", None),
                            signal=signal,
                        )
                        # AURION X 4.1 — XAU Knowledge: alta de trade vivo para MFE/MAE
                        try:
                            ke = getattr(self, "_xau_knowledge", None)
                            if ke is not None and ke.enabled and symbol.upper() == "XAUUSD":
                                tuid = (
                                    getattr(ticket, "trade_uuid", None)
                                    or getattr(ticket, "uuid", None)
                                    or str(ticket)
                                )
                                sl_dist = max(abs(float(entry_ref) - float(sl_price)), 1e-9)
                                # IDs de eventos SMC del snapshot del primary TF para vincular outcomes.
                                smc_ids: list[str] = []
                                try:
                                    snap = self._smc_analyze(symbol)
                                    if snap is not None:
                                        smc_ids = self._knowledge_propagate_smc(
                                            symbol, snap.primary_timeframe, snap
                                        )
                                except Exception:
                                    smc_ids = []
                                ke.on_trade_open(
                                    trade_id=str(tuid),
                                    symbol=symbol,
                                    direction=str(signal["direction"]),
                                    entry_price=float(entry_ref),
                                    sl_distance_price=sl_dist,
                                    open_time=tick.time_utc,
                                    smc_event_ids=smc_ids,
                                    pattern_keys=[],
                                    setup_type=setup_type,
                                )
                        except Exception as exc:
                            logger.debug("[KNOWLEDGE] on_trade_open falló: %s", exc)
                        is_rf = "RANGE_FADE" in setup_type
                        st = OpenTradeState(
                            ticket=ticket,
                            symbol=symbol,
                            direction=str(signal["direction"]),
                            entry_price=entry_ref,
                            risk_price=risk_price,
                            sl_pips=sl_pips,
                            tp_pips=tp_pips,
                            lot=lot,
                            setup_type=setup_type,
                            sniper_key=sniper_key,
                            range_partial_tp_done=False,
                            initial_tp_pips_planned=float(tp_pips) if is_rf else 0.0,
                            initial_tp_distance_price=float(tp_pips) * pip_size,
                            current_sl_price=float(sl_price),
                        )
                        self._open_trades[ticket] = st
                        # 11) log + DB
                        local_entry = tick.time_utc.astimezone(self._sniper_tz) if tick.time_utc.tzinfo else tick.time_utc.replace(tzinfo=timezone.utc).astimezone(self._sniper_tz)
                        features_for_db = dict(features_dict)
                        qd = self._quant_decision_by_symbol.get(symbol.upper())
                        idec = self._inst_decision_by_symbol.get(symbol.upper())
                        if idec is not None and self._institutional_engine is not None:
                            features_for_db = self._institutional_engine.enrich_features_for_db(
                                features_for_db,
                                idec.quant,
                            )
                            features_for_db["setup_grade"] = idec.setup_rank.grade.value
                            features_for_db["trading_mode"] = idec.trading_mode
                        elif qd is not None and self._quant_engine is not None:
                            features_for_db = self._quant_engine.enrich_features_for_db(features_for_db, qd)
                        features_for_db.update(
                            {
                                "setup_type": setup_type,
                                "volatility_regime": regime.value.upper(),
                                "hour_slot": int(local_entry.hour),
                                "score": int(score),
                                "score_bucket": score_bucket,
                                "rr": float(rr_now),
                                "strategy_detail": strategy_name,
                                "ml_dataset_context": "RANGE" if is_rf else "MOMENTUM",
                                "range_quality_score": signal.get("range_quality_score"),
                                "range_breakout_probability": signal.get("range_breakout_probability"),
                                "range_market_mode": signal.get("range_market_mode"),
                                "continuation_strength": float(continuation_now),
                                "mtf_M1": mtf_trends.get("M1", Trend.RANGE).value,
                                "mtf_M5": mtf_trends.get("M5", Trend.RANGE).value,
                                "mtf_M15": mtf_trends.get("M15", Trend.RANGE).value,
                                "mtf_H1": mtf_trends.get("H1", Trend.RANGE).value,
                                "mtf_H4": mtf_trends.get("H4", Trend.RANGE).value,
                                "mtf_D1": mtf_trends.get("D1", Trend.RANGE).value,
                            },
                        )
                        # AURION X 4.5 (11-jun-2026) — persistir SMC Decision
                        # en features_json (sin migracion SQL). El ML aprendera
                        # que configuraciones SMC producen mas ganancia.
                        _smc_dec_obj = feat_ctx.get("smc_decision")
                        if isinstance(_smc_dec_obj, dict):
                            features_for_db.update({
                                "smc_decision_score": _smc_dec_obj.get("smc_score"),
                                "smc_decision_grade": _smc_dec_obj.get("setup_grade"),
                                "smc_ob_quality_grade": _smc_dec_obj.get("ob_quality", {}).get("grade"),
                                "smc_ob_quality_score": _smc_dec_obj.get("ob_quality", {}).get("score"),
                                "smc_ob_institutional": _smc_dec_obj.get("ob_quality", {}).get("institutional"),
                                "smc_fvg_quality_grade": _smc_dec_obj.get("fvg_quality", {}).get("grade"),
                                "smc_fvg_quality_score": _smc_dec_obj.get("fvg_quality", {}).get("score"),
                                "smc_fvg_htf": _smc_dec_obj.get("fvg_quality", {}).get("htf"),
                                "smc_pd_state": _smc_dec_obj.get("pd_state", {}).get("state"),
                                "smc_pd_pct_in_range": _smc_dec_obj.get("pd_state", {}).get("pct_in_range"),
                                "smc_pd_tf_source": _smc_dec_obj.get("pd_state", {}).get("tf_source"),
                                "smc_sweep_detected": _smc_dec_obj.get("sweep_detected"),
                                "smc_sweep_kind": _smc_dec_obj.get("sweep_kind"),
                                "smc_mtf_full_aligned": _smc_dec_obj.get("mtf", {}).get("full_aligned"),
                                "smc_mtf_aligned_count": _smc_dec_obj.get("mtf", {}).get("aligned_count"),
                                "smc_lot_multiplier": _smc_dec_obj.get("lot_multiplier"),
                                "smc_aggressive_execution": _smc_dec_obj.get("aggressive_execution"),
                            })
                        db_strategy = (
                            "INTRADAY" if str(strategy_name).upper().startswith("INTRADAY") else strategy_name
                        )
                        try:
                            db_id = self._trade_repo.insert_trade_open(
                                {
                                    "symbol": symbol,
                                    "direction": st.direction,
                                    "lot": lot,
                                    "entry_price": st.entry_price,
                                    "open_time": tick.time_utc,
                                    "strategy": db_strategy,
                                    "sl": st.entry_price - risk_price if st.direction == "BUY" else st.entry_price + risk_price,
                                    "tp": st.entry_price + st.tp_pips * pip_size if st.direction == "BUY" else st.entry_price - st.tp_pips * pip_size,
                                    "ml_confidence": effective_conf,
                                    "features_json": features_for_db,
                                },
                            )
                            st.db_id = db_id
                        except Exception as exc:
                            logger.warning("No se pudo persistir trade_open: %s", exc)
                        logger.info(
                            "[TRADE_ALLOWED] symbol=%s side=%s lot=%.3f ml_conf_raw=%.3f effective_conf=%.3f threshold=%.3f mode=%s "
                            "score=%s score_bucket=%s volatility=%s volScore=%.2f volRatio=%.2f mtf=M1_%s_M5_%s setup=%s strategy=%s parts=%s",
                            symbol,
                            st.direction,
                            lot,
                            float(ml_conf or 0.0),
                            effective_conf,
                            threshold,
                            mode,
                            score,
                            score_bucket,
                            regime.value.upper(),
                            volume_score,
                            volume_ratio,
                            mtf_trends.get("M1", Trend.RANGE).value.upper(),
                            mtf_trends.get("M5", Trend.RANGE).value.upper(),
                            str(signal.get("setup_type", "UNKNOWN")),
                            strategy_name,
                            score_parts,
                        )
                    else:
                        self._maybe_log_entry_skip(
                            symbol,
                            "order_none",
                            "%s skip: broker no devolvió ticket (orden no ejecutada)",
                            symbol,
                        )
                else:
                    self._maybe_log_entry_skip(
                        symbol,
                        "risk_sizing",
                        "%s skip: riesgo propuesto rechazado tras sizing (lot=%.4f)",
                        symbol,
                        lot,
                    )
            else:
                self._maybe_log_entry_skip(
                    symbol,
                    "zero_lot",
                    "%s skip: PositionSizer devolvió lot<=0 (conf=%.3f)",
                    symbol,
                    effective_conf,
                )
        elif effective_conf < threshold:
            blocked_reason = "low_confidence"
            self._maybe_log_entry_skip(
                symbol,
                "low_confidence",
                "%s skip: conf efectiva %.3f < umbral ML adaptativo %.3f (mode=%s score=%s)",
                symbol,
                effective_conf,
                threshold,
                mode,
                score,
            )
        else:
            logger.debug(
                "on_tick skip: conf=%.3f thr=%.3f invalid=%s (%s)",
                effective_conf,
                threshold,
                is_valid,
                reason,
            )

        self._learn_from_decision(
            symbol=symbol,
            blocked_reason=blocked_reason,
            strong_impulse_entry=strong_impulse_entry,
            continuation_now=continuation_now,
            effective_conf=effective_conf,
            threshold=threshold,
        )

        # 10) Gestión de trades abiertos
        self._execute_management(symbol, tick)

    def load_config(self) -> dict[str, Any]:
        """Carga ``parameters.yaml`` si existe y fusiona con defaults.

        Returns:
            Diccionario de configuración resuelto (copia superficial mezclada).
        """
        merged: dict[str, Any] = dict(_DEFAULT_CONFIG)
        if self._config_path is not None and self._config_path.is_file():
            with self._config_path.open(encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                merged.update(loaded)
            else:
                logger.warning("load_config: YAML no es dict, se ignoran claves externas")
        self._config = merged
        app_cfg = merged.get("app", {}) or {}
        self._environment = str(app_cfg.get("environment", "demo")).strip().lower()
        self._order_manager.set_environment(self._environment)
        logger.info("load_config: claves top-level=%s", list(merged.keys()))
        return dict(merged)

    def start(self) -> None:
        """Arranca el orquestador (carga config si aún no está en memoria)."""
        if not self._config:
            self.load_config()
        self._equity = Decimal(str(self._config.get("equity", 10000)))
        exec_cfg = self._config.get("execution", {})
        risk_cfg = self._config.get("risk", {})
        ml_cfg = self._config.get("ml", {})
        self._ml_threshold_floor = float(ml_cfg.get("adaptive_threshold_floor", 0.40))
        self._ml_disabled_threshold_relax = float(ml_cfg.get("disabled_threshold_relax", 0.08))
        self._ml_disabled_confidence_boost = float(ml_cfg.get("disabled_confidence_boost", 0.06))
        self._lenient_extreme_volatility = bool(exec_cfg.get("lenient_extreme_volatility", False))
        self._skip_mtf_filter = bool(exec_cfg.get("skip_mtf_filter", False))
        self._use_live_equity = bool(exec_cfg.get("use_live_equity", True))
        self._entry_score_floor = max(0, int(exec_cfg.get("entry_score_floor", 0)))
        self._log_trade_attempts = bool(exec_cfg.get("log_trade_attempts", False))
        app_cfg = self._config.get("app", {}) or {}
        self._environment = str(app_cfg.get("environment", "demo")).strip().lower()
        self._order_manager.set_environment(self._environment)
        sdl = risk_cfg.get("symbol_daily_loss") or {}
        if not isinstance(sdl, dict):
            sdl = {}
        self._symbol_daily_loss_enabled = bool(sdl.get("enabled", False))
        try:
            self._symbol_daily_loss_limit = Decimal(str(sdl.get("max_loss_money", 0)))
        except Exception:
            self._symbol_daily_loss_limit = Decimal("0")
        self._symbol_daily_pnl.clear()
        self._symbol_daily_pnl_day = None
        cg_raw = exec_cfg.get("chop_guard") if isinstance(exec_cfg, dict) else None
        cg_cfg = ChopGuardConfig.from_mapping(cg_raw if isinstance(cg_raw, dict) else None)
        self._chop_guard = ChopGuard(cg_cfg)
        # AURION X 4.3: filtro de calendario economico (NFP, CPI, FOMC, etc).
        try:
            from risk.economic_calendar import EconomicCalendarGuard
            risk_cfg = self._config.get("risk", {}) if isinstance(self._config.get("risk"), dict) else {}
            cal_cfg = risk_cfg.get("news_calendar") if isinstance(risk_cfg, dict) else None
            self._news_calendar = EconomicCalendarGuard.from_config(cal_cfg)
            if self._news_calendar.enabled:
                logger.info(
                    "[NEWS_CALENDAR] activo: blackout=-%dmin/+%dmin custom_events=%d "
                    "include=%s override_ml>=%.2f",
                    self._news_calendar.default_before_min,
                    self._news_calendar.default_after_min,
                    len(self._news_calendar.custom_events),
                    list(self._news_calendar.include_recurring),
                    self._news_calendar.override_high_ml_conf,
                )
        except Exception as exc:
            logger.warning("[NEWS_CALENDAR] no se pudo cargar: %s", exc)
            self._news_calendar = None
        # AURION X 4.3: drift detection con auto-retrain.
        try:
            from ml.drift_detector import ModelDriftDetector
            ml_cfg = self._config.get("ml", {}) if isinstance(self._config.get("ml"), dict) else {}
            drift_cfg = ml_cfg.get("drift") if isinstance(ml_cfg, dict) else None
            self._drift_detector = ModelDriftDetector.from_config(drift_cfg)
            if self._drift_detector.enabled:
                logger.info(
                    "[DRIFT] activo: window=%d baseline_WR=%.2f baseline_R=%+.3f "
                    "wr_thr=%.2f r_thr=%.2f action=%s",
                    self._drift_detector.window_size,
                    self._drift_detector.baseline_win_rate,
                    self._drift_detector.baseline_avg_r,
                    self._drift_detector.drift_wr_threshold,
                    self._drift_detector.drift_r_threshold,
                    self._drift_detector.action_on_critical,
                )
        except Exception as exc:
            logger.warning("[DRIFT] no se pudo cargar: %s", exc)
            self._drift_detector = None
        if cg_cfg.enabled:
            logger.info(
                "[CHOP_GUARD] activo: range_pos_long_max=%.2f range_pos_short_min=%.2f "
                "micro_window=%d micro_avgR_max=%.2f cooldown_min=%d asia_sl_x=%.2f "
                "consec_loss_win=%d consec_min_absR=%.2f consec_cooldown=%d "
                "ml_conf_scalper>=%s%s blacklist=%s weak_hours=%s(+%.2f) blocked_hours=%s",
                cg_cfg.long_max_range_position,
                cg_cfg.short_min_range_position,
                cg_cfg.micro_trade_window,
                cg_cfg.micro_trade_avg_abs_r_max,
                cg_cfg.micro_trade_cooldown_min,
                cg_cfg.asia_sl_multiplier,
                cg_cfg.consecutive_loss_window,
                cg_cfg.consecutive_loss_min_abs_r,
                cg_cfg.consecutive_loss_cooldown_min,
                f"{cg_cfg.min_ml_conf_scalper:.2f}",
                " (enforce=False)" if not cg_cfg.enforce_ml_conf_scalper else "",
                list(cg_cfg.blocked_setups) or "[]",
                list(cg_cfg.weak_hours_utc) or "[]",
                cg_cfg.weak_hours_ml_boost,
                list(cg_cfg.blocked_hours_utc) or "[]",
            )
        raw_trading = self._config.get("symbols")
        if isinstance(raw_trading, list):
            self._symbols_trading = [str(s).strip().upper() for s in raw_trading if str(s).strip()]
        else:
            self._symbols_trading = []
        edge_cfg = self._config.get("edge_fusion", {}) or {}
        self._edge_fusion_enabled = bool(edge_cfg.get("enabled", True))
        self._edge_fusion_mtf_stack_bonus = int(edge_cfg.get("mtf_stack_bonus", 12))
        self._edge_fusion_ml_prob_aligned_boost = float(edge_cfg.get("ml_prob_aligned_boost", 0.04))
        self._edge_fusion_ml_prob_min = float(edge_cfg.get("ml_prob_min", 0.56))
        self._order_magic = int(exec_cfg.get("order_magic", 300000))
        self._reconcile_filter_magic = bool(exec_cfg.get("reconcile_filter_magic", True))
        self._reconcile_interval_s = max(0.0, float(exec_cfg.get("reconcile_positions_interval_s", 90)))
        self._reconcile_before_new_entry = bool(exec_cfg.get("reconcile_before_new_entry", True))
        self._scalper_require_h4_align = bool(exec_cfg.get("scalper_require_h4_alignment", True))
        self._block_scalper_counter_htf = bool(exec_cfg.get("block_scalper_counter_htf", False))
        self._skip_scalper_extreme_m5_range = bool(exec_cfg.get("skip_scalper_extreme_m5_range", True))
        try:
            self._min_sl_atr_m1_mult = float(exec_cfg.get("min_sl_atr_m1_mult", 0.0) or 0.0)
        except (TypeError, ValueError):
            self._min_sl_atr_m1_mult = 0.0
        try:
            self._min_tp_rr_after_sl_floor = float(exec_cfg.get("min_tp_rr_after_sl_floor", 1.12))
        except (TypeError, ValueError):
            self._min_tp_rr_after_sl_floor = 1.12
        try:
            self._max_sl_atr_m1_mult = float(exec_cfg.get("max_sl_atr_m1_mult", 1.75) or 1.75)
        except (TypeError, ValueError):
            self._max_sl_atr_m1_mult = 1.75
        try:
            self._max_sl_price_pct = float(exec_cfg.get("max_sl_price_pct", 0.0012) or 0.0012)
        except (TypeError, ValueError):
            self._max_sl_price_pct = 0.0012
        try:
            self._max_tp_rr = float(exec_cfg.get("max_tp_rr", 2.35) or 2.35)
        except (TypeError, ValueError):
            self._max_tp_rr = 2.35
        try:
            self._max_tp_rr_intraday = float(exec_cfg.get("max_tp_rr_intraday", 2.55) or 2.55)
        except (TypeError, ValueError):
            self._max_tp_rr_intraday = 2.55
        self._enforce_broker_position_caps = bool(exec_cfg.get("enforce_broker_position_caps", True))
        self._last_reconcile_mono = 0.0
        vol_cfg = self._config.get("volatility", {}) or {}
        self._volatility = VolatilityDetector(
            int(vol_cfg.get("window", 120)),
            low_cv=float(vol_cfg.get("low_cv", 0.00008)),
            high_cv=float(vol_cfg.get("high_cv", 0.00035)),
            extreme_cv=float(vol_cfg.get("extreme_cv", 0.00070)),
        )
        self._spread_high_points = float(exec_cfg.get("spread_high_points", 120))
        self._spread_extreme_points = float(exec_cfg.get("spread_extreme_points", 220))
        self._score_min_normal = int(exec_cfg.get("score_min_normal", 55))
        self._score_min_strong = int(exec_cfg.get("score_min_strong", 70))
        self._asset_profiles = self._config.get("asset_profiles", {}) or {}
        self._scalper_by_symbol.clear()
        telemetry_cfg = self._config.get("telemetry", {}) or {}
        self._telemetry_enabled = bool(telemetry_cfg.get("enabled", True))
        self._telemetry_interval_s = max(30.0, float(telemetry_cfg.get("interval_s", 120)))
        self._telemetry_last_mono.clear()
        self._atr_spike_enabled = bool(exec_cfg.get("atr_spike_enabled", False))
        self._atr_min_samples = int(exec_cfg.get("atr_min_samples", 400))
        stf = str(exec_cfg.get("structure_timeframe", "M15")).strip().upper()
        if stf not in SUPPORTED_TIMEFRAMES:
            logger.warning("structure_timeframe inválido %r, se usa M15", stf)
            stf = "M15"
        self._structure_timeframe = stf
        self._mtf.set_max_bars(int(exec_cfg.get("mtf_max_bars", 600)))
        self._mtf.set_trends_cache_ttl(float(exec_cfg.get("mtf_trends_cache_ttl_s", 0.0)))
        self._entry_skip_log_interval_s = max(5.0, float(exec_cfg.get("entry_skip_log_interval_s", 45)))
        self._block_summary_window_s = max(30.0, float(exec_cfg.get("block_summary_window_s", 300)))
        self._chop_macro_min_duration_s = max(
            60.0, float(exec_cfg.get("chop_macro_min_duration_s", 1800.0))
        )
        ml_cfg_root = self._config.get("ml", {}) or {}
        bias_cfg = ml_cfg_root.get("bias_monitor", {}) or {}
        try:
            self._ml_bias_window = max(20, int(bias_cfg.get("window", 200)))
        except (TypeError, ValueError):
            self._ml_bias_window = 200
        try:
            self._ml_bias_warn_ratio = max(
                0.5, min(0.99, float(bias_cfg.get("warn_ratio", 0.80)))
            )
        except (TypeError, ValueError):
            self._ml_bias_warn_ratio = 0.80
        try:
            self._ml_bias_min_samples = max(
                10, int(bias_cfg.get("min_samples", 80))
            )
        except (TypeError, ValueError):
            self._ml_bias_min_samples = 80
        try:
            self._ml_bias_warn_interval_s = max(
                60.0, float(bias_cfg.get("warn_interval_s", 1800.0))
            )
        except (TypeError, ValueError):
            self._ml_bias_warn_interval_s = 1800.0
        self._min_entry_spacing_s = max(0.0, float(exec_cfg.get("min_seconds_between_entries_per_symbol", 0)))
        self._last_new_entry_mono_by_symbol.clear()
        self._last_entry_m1_candle_id_by_symbol.clear()
        self._mtf_entry_min_support = max(1, min(3, int(exec_cfg.get("mtf_entry_min_support", 2))))
        self._mtf_allow_all_entry_range = bool(exec_cfg.get("mtf_allow_when_all_entry_range", False))
        try:
            self._mtf_ml_strong_override_th = float(exec_cfg.get("mtf_ml_strong_override_th", 0.80))
        except (TypeError, ValueError):
            self._mtf_ml_strong_override_th = 0.80
        strat_cfg = self._config.get("strategy", {}) or {}
        scalper_ov = dict(strat_cfg.get("scalper", {}) or {})
        rq_root = self._config.get("range_quality")
        if isinstance(rq_root, dict) and rq_root:
            scalper_ov["range_quality"] = rq_root
        scalper_min_conf = float((scalper_ov or {}).get("scalper_extreme_m5_min_confidence", 0.88))
        scalper_ov.setdefault("scalper_extreme_m5_min_confidence", scalper_min_conf)
        self._scalper = ScalperEngine(scalper_ov if scalper_ov else None)
        intraday_ov = dict(strat_cfg.get("intraday", {}) or {})
        self._intraday = IntradayEngine(intraday_ov if intraday_ov else None)
        try:
            self._intraday_min_key_levels = max(1, int(intraday_ov.get("min_structure_key_levels", 3)))
        except (TypeError, ValueError):
            self._intraday_min_key_levels = 3
        struct_ov = strat_cfg.get("structure", {}) or {}
        self._mtf.set_structure_detector_params(struct_ov if struct_ov else None)
        self._impulse_log_cfg = strat_cfg.get("impulse_log", {}) or {}
        self._last_impulse_log_mono.clear()
        sfe_raw = exec_cfg.get("session_filter_enabled", False)
        if isinstance(sfe_raw, str):
            session_filter_enabled = sfe_raw.strip().lower() not in ("false", "0", "no", "")
        else:
            session_filter_enabled = bool(sfe_raw)
        sess_tz = str(exec_cfg.get("session_timezone", "UTC")).strip() or "UTC"
        sess_start = parse_session_time(exec_cfg.get("session_start"), fallback=time(7, 0))
        sess_end = parse_session_time(exec_cfg.get("session_end"), fallback=time(20, 0))
        self._session_filter = SessionFilter(
            sess_tz,
            start=sess_start,
            end=sess_end,
            enabled=session_filter_enabled,
        )
        ny_turbo_cfg = exec_cfg.get("ny_turbo", {}) or {}
        self._ny_turbo_enabled = bool(ny_turbo_cfg.get("enabled", True))
        symbols_raw = ny_turbo_cfg.get("symbols", ["XAUUSD"])
        if isinstance(symbols_raw, (list, tuple, set)):
            self._ny_turbo_symbols = {str(s).upper() for s in symbols_raw if str(s).strip()}
        else:
            self._ny_turbo_symbols = {"XAUUSD"}
        self._ny_turbo_start = parse_session_time(ny_turbo_cfg.get("start"), fallback=time(9, 30))
        self._ny_turbo_end = parse_session_time(ny_turbo_cfg.get("end"), fallback=time(11, 30))
        self._ny_turbo_symbol_windows.clear()
        sw_raw = ny_turbo_cfg.get("symbol_windows")
        if isinstance(sw_raw, dict):
            for sym_key, win in sw_raw.items():
                sym_k = str(sym_key).strip().upper()
                if not sym_k or not isinstance(win, dict):
                    continue
                st = parse_session_time(win.get("start"), fallback=self._ny_turbo_start)
                en = parse_session_time(win.get("end"), fallback=self._ny_turbo_end)
                self._ny_turbo_symbol_windows[sym_k] = (st, en)
        self._adaptive_entry_bias.clear()
        wd_raw = self._config.get("weekday_filters", {}) or {}
        self._weekday_policies = build_weekday_policies_from_config(wd_raw)
        tph_cfg = exec_cfg.get("tp_hunter", {}) or {}
        self._tp_hunter_enabled = bool(tph_cfg.get("enabled", True))
        self._tp_hunter_mult_impulse = float(tph_cfg.get("impulse_mult", 1.22))
        self._tp_hunter_mult_impulse_strong = float(tph_cfg.get("impulse_strong_mult", 1.35))
        self._tp_hunter_mult_low_vol = float(tph_cfg.get("low_vol_mult", 0.90))
        self._tp_hunter_min_rr = float(tph_cfg.get("min_rr", 1.15))
        hints_cfg = exec_cfg.get("chart_hints", {}) or {}
        self._chart_hints_enabled = bool(hints_cfg.get("enabled", True))
        self._chart_hints_min_conf = float(hints_cfg.get("min_confidence", 0.45))
        self._chart_hints_require_trade_ready = bool(hints_cfg.get("require_trade_ready", True))
        hints_path_raw = str(hints_cfg.get("path", "AURION_X_3_0/mql5/aurion_live_hints.csv")).strip()
        self._chart_hints_path = Path(hints_path_raw) if hints_path_raw else Path("AURION_X_3_0/mql5/aurion_live_hints.csv")
        self._chart_hints_write_interval_ms = max(50.0, float(hints_cfg.get("write_interval_ms", 250)))
        self._chart_hints_dedupe_by_m1_bar = bool(hints_cfg.get("dedupe_by_m1_bar", True))
        try:
            self._chart_hints_conf_round_decimals = max(0, min(int(hints_cfg.get("confidence_round_decimals", 3)), 8))
        except (TypeError, ValueError):
            self._chart_hints_conf_round_decimals = 3
        try:
            self._chart_hints_min_score = max(0, int(hints_cfg.get("min_score", 0)))
        except (TypeError, ValueError):
            self._chart_hints_min_score = 0
        self._chart_hints_rows_cache.clear()
        self._chart_hints_last_payload.clear()
        self._chart_hints_last_flush_mono = 0.0
        spread_guard_cfg = exec_cfg.get("spread_guard", {}) or {}
        self._spread_guard_enabled = bool(spread_guard_cfg.get("enabled", True))
        self._spread_guard_lookback = max(8, int(spread_guard_cfg.get("lookback_ticks", 30)))
        self._spread_guard_spike_mult = max(1.1, float(spread_guard_cfg.get("spike_mult", 1.8)))
        self._spread_guard_max_points = float(spread_guard_cfg.get("max_points", 260))
        de_raw = exec_cfg.get("dynamic_exit") or {}
        self._dynamic_exit_cfg = dict(de_raw) if isinstance(de_raw, dict) else {}
        self._dynamic_exit_enabled = bool(self._dynamic_exit_cfg.get("enabled", False))
        logger.info(
            "dynamic_exit: raíz enabled=%s (por símbolo: asset_profiles.<SYM>.dynamic_exit fusiona claves)",
            self._dynamic_exit_enabled,
        )
        sniper_cfg = exec_cfg.get("sniper", {}) or {}
        self._sniper_enabled = bool(sniper_cfg.get("enabled", True))
        dead_hours_raw = sniper_cfg.get("dead_hours_utc", [])
        dead_hours: set[int] = set()
        if isinstance(dead_hours_raw, (list, tuple, set)):
            for h in dead_hours_raw:
                try:
                    hv = int(h)
                except Exception:
                    continue
                if 0 <= hv <= 23:
                    dead_hours.add(hv)
        self._sniper_dead_hours_utc = dead_hours
        tz_name = str(sniper_cfg.get("learning_timezone", "America/New_York")).strip() or "America/New_York"
        try:
            self._sniper_tz = ZoneInfo(tz_name)
        except Exception:
            logger.warning("sniper.learning_timezone inválida=%r, usando America/New_York", tz_name)
            self._sniper_tz = ZoneInfo("America/New_York")
        self._sniper_min_samples = max(4, int(sniper_cfg.get("min_samples_per_bucket", 10)))
        self._sniper_min_expectancy = float(sniper_cfg.get("min_expectancy_r", 0.05))
        self._sniper_min_winrate = float(sniper_cfg.get("min_winrate", 0.48))
        self._sniper_allow_strong_override = bool(sniper_cfg.get("allow_strong_impulse_override", True))
        self._sniper_require_whitelist = bool(sniper_cfg.get("require_learning_whitelist", False))
        self._sniper_whitelist.clear()
        self._sniper_bucket_threshold_bias.clear()
        self._sniper_bucket_score_bias.clear()
        self._sniper_bucket_risk_mult.clear()
        self._online_setup_perf.clear()
        self._online_setup_min_samples = max(3, int(sniper_cfg.get("online_min_samples", 5)))
        self._online_setup_bad_expectancy = float(sniper_cfg.get("online_bad_expectancy_r", -0.08))
        self._online_setup_good_expectancy = float(sniper_cfg.get("online_good_expectancy_r", 0.10))
        self._online_setup_good_winrate = float(sniper_cfg.get("online_good_winrate", 0.52))
        learn_cfg = self._config.get("adaptive_learning", {}) or {}
        self._perf_learning_enabled = bool(learn_cfg.get("enabled", True))
        self._perf_learning_interval_s = max(60.0, float(learn_cfg.get("interval_s", 180)))
        self._perf_learning_min_samples = max(6, int(learn_cfg.get("min_samples", 12)))
        self._perf_learning_lookback = max(50, int(learn_cfg.get("lookback_trades", 250)))
        self._adaptive_perf_bias.clear()
        self._adaptive_risk_multiplier.clear()
        self._last_perf_learn_mono = 0.0
        self._decision_interval_ms = max(0.0, float(exec_cfg.get("decision_interval_ms", 35)))
        self._last_decision_mono.clear()
        want_ml = bool(ml_cfg.get("enabled", True))
        self._ml_veto_enabled = bool(ml_cfg.get("veto_contradictions", True))
        self._auto_retrain_enabled = bool(ml_cfg.get("auto_retrain_enabled", False))
        self._auto_retrain_min_closed = max(30, int(ml_cfg.get("auto_retrain_min_closed_trades", 60)))
        self._auto_retrain_every_closed = max(5, int(ml_cfg.get("auto_retrain_every_closed_trades", 25)))
        self._auto_retrain_limit = max(500, int(ml_cfg.get("auto_retrain_limit", 5000)))
        self._auto_retrain_batch = bool(ml_cfg.get("auto_retrain_batch", True))
        self._auto_retrain_min_interval_s = max(120.0, float(ml_cfg.get("auto_retrain_min_interval_s", 1800)))
        self._auto_retrain_last_mono = 0.0
        self._closed_trades_counter = 0
        self._bootstrap_ml_on_start = bool(ml_cfg.get("bootstrap_on_start", True))
        self._bootstrap_ml_min_closed = max(30, int(ml_cfg.get("bootstrap_min_closed_trades", 30)))
        self._bootstrap_ml_force_on_start = bool(ml_cfg.get("bootstrap_force_on_start", False))
        self._ml_walk_forward_enabled = bool(ml_cfg.get("walk_forward_enabled", True))
        self._ml_walk_forward_min_samples = max(50, int(ml_cfg.get("walk_forward_min_samples", 80)))
        self._ml_walk_forward_train_ratio = float(ml_cfg.get("walk_forward_train_ratio", 0.65))
        self._ml_walk_forward_val_ratio = float(ml_cfg.get("walk_forward_val_ratio", 0.15))
        model_path_raw = str(ml_cfg.get("model_path", "AURION_X_3_0/ml/model.pkl")).strip()
        model_path = Path(model_path_raw) if model_path_raw else Path("AURION_X_3_0/ml/model.pkl")
        self._ml_model_path = model_path
        self._predictor = Predictor()
        loaded_global = False
        if want_ml:
            try:
                if model_path.is_file():
                    model = TradingModel(artifact_path=model_path)
                    model.load()
                    self._predictor.set_model(model)
                    loaded_global = True
                    logger.info(
                        "ML cargado desde %s (n_features=%s schema=%s)",
                        model_path,
                        model.n_features_expected(),
                        model.feature_schema_version(),
                    )
                else:
                    logger.warning("model_path global no encontrado: %s", model_path)
            except Exception as exc:
                logger.warning("Error cargando ML global: %s", exc)
            self._load_per_symbol_ml_models()
            any_model = loaded_global or bool(self._predictor_by_symbol)
            self._ml_enabled = any_model
            if not any_model:
                logger.warning(
                    "ML deshabilitado: sin PKL global válido ni ml_model_path por activo",
                )
            elif not loaded_global and self._predictor_by_symbol:
                logger.info("ML activo con modelos solo por símbolo (sin PKL global)")
        else:
            self._ml_enabled = False
            self._predictor_by_symbol.clear()
        self._predictor_range = None
        if want_ml:
            range_path_raw = str(ml_cfg.get("range_model_path", "") or "").strip()
            if range_path_raw:
                rp = Path(range_path_raw)
                if rp.is_file():
                    try:
                        rmod = TradingModel(artifact_path=rp)
                        rmod.load()
                        prng = Predictor()
                        prng.set_model(rmod)
                        self._predictor_range = prng
                        logger.info(
                            "ML RANGE cargado desde %s (n_features=%s)",
                            rp,
                            rmod.n_features_expected(),
                        )
                    except Exception as exc:
                        logger.warning("ML RANGE no cargado (%s): %s", rp, exc)
                elif range_path_raw:
                    logger.warning("ml.range_model_path no encontrado: %s", rp)
        self._bootstrap_ml_from_database()
        qcfg = self._config.get("quant_ml", {}) or {}
        icfg = self._config.get("institutional", {}) or {}
        acfg = self._config.get("apex", {}) or {}
        if bool(icfg.get("enabled", False)) and bool(qcfg.get("enabled", False)) and want_ml:
            try:
                if bool(acfg.get("enabled", False)):
                    self._institutional_engine = ApexEngine(
                        trade_repo=self._trade_repo,
                        ml_cfg=ml_cfg if isinstance(ml_cfg, dict) else {},
                        quant_cfg=qcfg if isinstance(qcfg, dict) else {},
                        institutional_cfg=icfg if isinstance(icfg, dict) else {},
                        asset_profiles=self._asset_profiles,
                        apex_cfg=acfg if isinstance(acfg, dict) else {},
                    )
                    logger.info("[APEX] motor cuantitativo auto-evolutivo activo (institucional + micro + meta-riesgo)")
                else:
                    self._institutional_engine = InstitutionalEngine(
                        trade_repo=self._trade_repo,
                        ml_cfg=ml_cfg if isinstance(ml_cfg, dict) else {},
                        quant_cfg=qcfg if isinstance(qcfg, dict) else {},
                        institutional_cfg=icfg if isinstance(icfg, dict) else {},
                        asset_profiles=self._asset_profiles,
                    )
                    logger.info("[INST] motor institucional activo (PG quant tables + ranking + funded)")
                self._quant_engine = self._institutional_engine.quant_engine
                try:
                    self._quant_engine.set_closed_baseline(self._trade_repo.count_closed_trades())
                except Exception:
                    self._quant_engine.set_closed_baseline(0)
                if any(
                    self._institutional_engine.has_model(s)
                    for s in (self._config.get("symbols") or ["XAUUSD"])
                ):
                    self._ml_enabled = True
                self._warn_ml_schema_mismatch()
            except Exception as exc:
                logger.warning("Motor institucional/Apex no iniciado: %s", exc)
                self._institutional_engine = None
        elif bool(qcfg.get("enabled", False)) and want_ml:
            try:
                self._quant_engine = QuantMLEngine(
                    trade_repo=self._trade_repo,
                    ml_cfg=ml_cfg if isinstance(ml_cfg, dict) else {},
                    quant_cfg=qcfg if isinstance(qcfg, dict) else {},
                    asset_profiles=self._asset_profiles,
                    config_symbols=self._config.get("symbols") if isinstance(self._config, dict) else None,
                )
                try:
                    self._quant_engine.set_closed_baseline(self._trade_repo.count_closed_trades())
                except Exception:
                    self._quant_engine.set_closed_baseline(0)
                any_quant = any(
                    self._quant_engine.has_model(s) for s in (self._config.get("symbols") or ["XAUUSD"])
                )
                if any_quant:
                    self._ml_enabled = True
                logger.info(
                    "[QUANT] motor cuantitativo activo ensemble=%s features_v2=%s peso_decision=%.2f",
                    qcfg.get("use_ensemble", True),
                    qcfg.get("use_extended_features", True),
                    float(qcfg.get("ml_decision_weight", 0.65)),
                )
            except Exception as exc:
                logger.warning("QuantMLEngine no iniciado: %s", exc)
                self._quant_engine = None
        if not self._ml_enabled and self._ml_model_path.is_file():
            try:
                self._reload_ml_models(self._ml_model_path)
                logger.info("ML recargado tras bootstrap desde %s", self._ml_model_path)
            except Exception as exc:
                logger.warning("No se pudo recargar ML tras bootstrap: %s", exc)

        self._wire_v4_hybrid()
        self._wire_smc_layer()
        self._wire_xau_knowledge()

        ignore_spread = bool(exec_cfg.get("ignore_spread", False))
        self._emergency = EmergencyController(
            atr_spike_threshold=float(exec_cfg.get("atr_spike_threshold", 8.0)),
            spread_max=float(exec_cfg.get("spread_max_points", 300)),
            pause_seconds=int(exec_cfg.get("cooldown_sec", 120)),
            spread_emergency_enabled=not ignore_spread,
        )
        raw_max_entries = risk_cfg.get("max_entries_per_day", 50)
        try:
            max_entries_resolved = int(raw_max_entries) if raw_max_entries is not None else 0
        except (TypeError, ValueError):
            max_entries_resolved = 50
        self._risk = RiskManager(
            limits={
                "max_drawdown_pause_pct": risk_cfg.get("max_dd_pause", 5),
                "max_drawdown_hard_stop_pct": risk_cfg.get("max_dd_stop", 8),
                "max_trades_per_session": max_entries_resolved,
                "cooldown_seconds": exec_cfg.get("cooldown_sec", 120),
                "loss_streak_pause": risk_cfg.get("max_consecutive_losses", 20),
                "loss_streak_reset_after_cooldown": bool(
                    risk_cfg.get("loss_streak_reset_after_cooldown", False)
                ),
                "risk_per_trade_pct": risk_cfg.get("risk_per_trade", 1.0),
                "post_loss_cooldown_seconds": risk_cfg.get(
                    "post_loss_cooldown_seconds", {2: 900, 3: 2700, 4: 5400}
                ),
            }
        )
        # Sincroniza la equity con el bróker real ANTES de fijar el baseline del RiskManager.
        # Si no, un mismatch entre el default ($10k) y la cuenta real (p.ej. FundingPips $5k) dispara
        # un "drawdown" ficticio del 50% y activa el hard stop al arrancar.
        self._maybe_refresh_equity_from_broker()
        self._risk.start_session(self._equity)
        logger.info("RiskManager.start_session equity_baseline=%s (live_equity=%s)",
                    str(self._equity), self._use_live_equity)
        # Persistencia ligera de loss_streak: tras reiniciar el proceso
        # reconstruimos las rachas por símbolo leyendo los últimos trades
        # cerrados desde el TradeRepository. Evita que el bot empiece "limpio"
        # (sin cooldown y sin tope de revenge_guard) si justo veníamos de una
        # mala racha en producción.
        try:
            recent = self._trade_repo.get_training_dataset(limit=50)
            self._risk.hydrate_loss_streaks(recent)
        except Exception as exc:
            logger.debug("hydrate_loss_streaks no disponible: %s", exc)
        corr_raw = risk_cfg.get("correlation_groups") or []
        parsed_groups: list[frozenset[str]] = []
        if isinstance(corr_raw, list):
            for item in corr_raw:
                if isinstance(item, (list, tuple, set)):
                    syms = {str(s).strip().upper() for s in item if str(s).strip()}
                    if len(syms) >= 2:
                        parsed_groups.append(frozenset(syms))
        self._correlation_groups = parsed_groups
        self._max_open_per_correlation_group = max(0, int(risk_cfg.get("max_open_per_correlation_group", 1)))
        self._max_open_positions_total = max(0, int(risk_cfg.get("max_open_positions_total", 0)))
        self._max_open_positions_per_symbol = max(0, int(risk_cfg.get("max_open_positions_per_symbol", 0)))
        ik = risk_cfg.get("intraday_kill_switch") or {}
        self._intraday_kill_enabled = bool(ik.get("enabled", True))
        self._intraday_kill_window = max(5, int(ik.get("window_trades", 12)))
        self._intraday_kill_min_trades = max(4, int(ik.get("min_trades", 8)))
        self._intraday_kill_bad_winrate = float(ik.get("bad_winrate", 0.38))
        self._intraday_kill_bad_avg_r = float(ik.get("bad_avg_r", -0.12))
        self._intraday_kill_require_both = bool(ik.get("require_both", True))
        self._intraday_kill_cooldown_s = max(60, int(ik.get("cooldown_s", 1800)))
        # Fix #11: early kill switch (más sensible, pausa día UTC entero)
        ek = risk_cfg.get("early_kill_switch") or {}
        self._early_kill_enabled = bool(ek.get("enabled", True))
        self._early_kill_min_trades = max(2, int(ek.get("min_trades", 3)))
        self._early_kill_bad_winrate = float(ek.get("bad_winrate", 0.10))
        self._early_kill_bad_avg_r = float(ek.get("bad_avg_r", -3.0))
        self._early_kill_require_both = bool(ek.get("require_both", True))
        self._intraday_recent_r.clear()
        self._intraday_roll_day.clear()
        self._intraday_kill_until.clear()
        ev_cfg = exec_cfg.get("entry_validator", {}) or {}
        if not isinstance(ev_cfg, dict):
            ev_cfg = {}
        self._entry_validator = EntryValidator(
            rules={
                "relax_range_when_mtf_aligned": bool(
                    exec_cfg.get("relax_structure_range_when_mtf_aligned", True),
                ),
                "ignore_spread": ignore_spread,
                "allow_low_vol_breakout_when_mtf_aligned": bool(
                    ev_cfg.get("allow_low_vol_breakout_when_mtf_aligned", False),
                ),
                "allow_extreme_when_mtf_aligned": bool(ev_cfg.get("allow_extreme_when_mtf_aligned", False)),
                "scalper_ignore_structure_trend": bool(ev_cfg.get("scalper_ignore_structure_trend", False)),
            },
        )
        if self._environment == "live":
            if self._skip_mtf_filter:
                logger.warning(
                    "[CONFIG] app.environment=live con skip_mtf_filter=true: mayor riesgo de falsos setups",
                )
            if bool(risk_cfg.get("loss_streak_reset_after_cooldown", False)):
                logger.warning(
                    "[CONFIG] live con loss_streak_reset_after_cooldown: pensado para demo; revisa riesgo",
                )
        self._running = True
        logger.info(
            "[HEALTH] ml_enabled=%s ml_veto=%s model_path=%s model_exists=%s auto_retrain=%s walk_forward=%s "
            "min_closed=%s every_closed=%s hints_path=%s hints_enabled=%s hints_require_trade=%s "
            "decision_interval_ms=%.1f corr_groups=%s max_open_group=%s intraday_kill=%s "
            "sniper_whitelist_gate=%s low_vol_breakout_ok=%s extreme_mtf_ok=%s lenient_extreme_vol=%s "
            "ml_no_model_relax=%.2f ml_no_model_conf_boost=%.2f skip_mtf=%s",
            self._ml_enabled,
            self._ml_veto_enabled,
            self._ml_model_path,
            self._ml_model_path.is_file(),
            self._auto_retrain_enabled,
            self._ml_walk_forward_enabled,
            self._auto_retrain_min_closed,
            self._auto_retrain_every_closed,
            self._chart_hints_path,
            self._chart_hints_enabled,
            self._chart_hints_require_trade_ready,
            self._decision_interval_ms,
            len(self._correlation_groups),
            self._max_open_per_correlation_group,
            self._intraday_kill_enabled,
            self._sniper_require_whitelist,
            bool((exec_cfg.get("entry_validator", {}) or {}).get("allow_low_vol_breakout_when_mtf_aligned", False)),
            bool((exec_cfg.get("entry_validator", {}) or {}).get("allow_extreme_when_mtf_aligned", False)),
            self._lenient_extreme_volatility,
            self._ml_disabled_threshold_relax,
            self._ml_disabled_confidence_boost,
            self._skip_mtf_filter,
        )
        logger.info(
            "Orchestrator.start (running=True) structure_tf=%s ignore_spread=%s session=%s "
            "mtf_min_support=%s mtf_allow_all_entry_range=%s skip_mtf_filter=%s",
            self._structure_timeframe,
            ignore_spread,
            self._session_filter.hint,
            self._mtf_entry_min_support,
            self._mtf_allow_all_entry_range,
            self._skip_mtf_filter,
        )
        logger.info(
            "[EDGE] fusion=%s mtf_stack_bonus=%s ml_boost=%s reconcile_s=%s magic_filter=%s env=%s",
            self._edge_fusion_enabled,
            self._edge_fusion_mtf_stack_bonus,
            self._edge_fusion_ml_prob_aligned_boost,
            self._reconcile_interval_s,
            self._reconcile_filter_magic,
            self._environment,
        )
        # AURION X 4.1 — Hidratación inicial OHLC desde MT5 con timeout duro.
        # Resuelve cold-start del scalper M1 (necesita 25 velas, antes tardaba
        # ~25 min en arrancar tras cada reinicio) y warm-up de detectores
        # estructurales. Si MT5 no responde en 15s la abortamos y dejamos que
        # el bot se hidrate por ticks como antes.
        try:
            raw_syms = self._config.get("symbols") if isinstance(self._config, dict) else None
            if not raw_syms:
                raw_syms = ["XAUUSD"]
            syms = [str(s).strip().upper() for s in raw_syms if str(s).strip()]
            essential_tfs = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")

            def _do_hydrate() -> None:
                for sym in syms:
                    try:
                        self._mtf.hydrate_from_mt5(
                            sym, timeframes=essential_tfs, bars_per_tf=120
                        )
                    except Exception as exc:
                        logger.debug("[HYDRATE] %s falló: %s", sym, exc)

            import threading as _th
            th = _th.Thread(target=_do_hydrate, daemon=True, name="aurion-hydrate")
            th.start()
            th.join(timeout=15.0)
            if th.is_alive():
                logger.warning(
                    "[HYDRATE] timeout 15s (MT5 lento); continuando sin pre-carga "
                    "— el bot se hidratará por ticks vivos"
                )
        except Exception as exc:
            logger.debug("[HYDRATE] arranque falló: %s", exc)
        self._reconcile_positions_with_broker()

    def stop(self) -> None:
        """Detiene de forma ordenada."""
        self._running = False
        logger.info("Orchestrator.stop (running=False)")

    def run_once(self) -> None:
        """Una iteración del ciclo principal (hook; sin pipeline completo aún)."""
        if not self._config:
            self.load_config()
        logger.debug("Orchestrator.run_once")

    @property
    def config(self) -> dict[str, Any]:
        """Configuración cargada (puede estar vacía hasta ``load_config``/``start``)."""
        return dict(self._config)

    @property
    def running(self) -> bool:
        """Indica si ``start`` se llamó y ``stop`` aún no."""
        return self._running

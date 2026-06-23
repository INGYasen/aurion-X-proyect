"""AURION X 4.1 — XAU Knowledge Engine.

**Filosofía**: el motor NO reentrena modelos continuamente. Observa el mercado
en vivo, acumula memoria estructurada en PostgreSQL, y dispara
reentrenamientos sólo bajo condiciones controladas (4 triggers).

Componentes:

1. **Observer**          — Recibe ticks/velas; ingesta cada vela cerrada a
                           ``xau_market_memory`` con regime/trend/range/vol/
                           session/spread.
2. **PatternRegistry**   — Detecta y registra combinaciones SMC (OB+FVG,
                           CHOCH+BOS, SSL Sweep, etc.) con frecuencia. Cuando
                           un trade cierra, propaga el outcome al pattern.
3. **OpenTradeTracker**  — Sigue MFE/MAE de trades vivos (no espera al
                           cierre) y emite snapshots periódicos.
4. **RegimeMemory**      — Episodios de régimen (TREND/RANGE/EXPANSION/
                           COMPRESSION/REVERSAL) con duración y winrate.
5. **SessionMemory**     — Estadísticas por sesión Asia/London/NY/Overlap.
6. **RetrainCoordinator** — Decide *cuándo* reentrenar el modelo XAUUSD
                           aislado. Triggers: ``closed_trades>=N``,
                           ``end_of_day``, ``regime_change``, ``degradation``.
                           Pesos temporales: últimos 6 meses (50%), 6-12 m
                           (30%), 12-36 m (20%). Walk-forward obligatorio.

Aislamiento XAUUSD: el modelo de oro se entrena **sólo** con trades XAUUSD;
BTCUSD/USTEC se ignoran al construir el dataset de entrenamiento. El resto
de motores siguen viendo el ensemble multi-símbolo intacto.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from analytics.logger import get_logger
from analytics.xau_knowledge_repository import XAUKnowledgeRepository

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


DEFAULT_TIME_WEIGHTS: dict[str, dict[str, float]] = {
    "recent_6m":   {"max_age_days": 183.0, "weight": 0.50},
    "mid_6_12m":   {"max_age_days": 365.0, "weight": 0.30},
    "long_12_36m": {"max_age_days": 1095.0, "weight": 0.20},
}


@dataclass(slots=True)
class XAUKnowledgeConfig:
    enabled: bool = True
    symbol: str = "XAUUSD"
    # Observer
    observe_timeframes: tuple[str, ...] = ("M15", "M30", "H1", "H4", "D1")
    market_memory_flush_size: int = 64
    # Patterns
    pattern_min_confluence: float = 0.40
    # Retrain triggers
    retrain_closed_trades_n: int = 10
    retrain_end_of_day_utc_hour: int = 23
    retrain_regime_change_min_duration_min: int = 30
    retrain_degradation_winrate_drop: float = 0.10
    retrain_degradation_window_trades: int = 20
    retrain_min_interval_minutes: int = 60
    # Modelo aislado
    model_path: Path = field(default_factory=lambda: Path("D:/AURION_DATA/models/model_XAUUSD.pkl"))
    walk_forward_folds: int = 5
    # Pesos temporales
    time_weights: dict[str, dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_TIME_WEIGHTS))
    # Sesiones (rangos UTC; XAUUSD práctica común)
    sessions_utc: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "ASIA":   (0, 7),    # 00:00 - 07:00 UTC
        "LONDON": (7, 12),
        "OVERLAP_LDN_NY": (12, 16),
        "NY":     (16, 21),
        "AFTER_HOURS": (21, 24),
    })


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _classify_session(when_utc: datetime, sessions_utc: Mapping[str, tuple[int, int]]) -> str:
    h = when_utc.astimezone(timezone.utc).hour if when_utc.tzinfo else when_utc.replace(tzinfo=timezone.utc).hour
    for name, (start, end) in sessions_utc.items():
        if start <= h < end:
            return name
    return "AFTER_HOURS"


def _atr_basic(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> float:
    n = len(closes)
    if n < 2 or len(highs) != n or len(lows) != n:
        return 0.0
    trs = []
    for i in range(1, n):
        h, l_, cp = float(highs[i]), float(lows[i]), float(closes[i - 1])
        trs.append(max(h - l_, abs(h - cp), abs(l_ - cp)))
    if not trs:
        return 0.0
    if len(trs) < period:
        return sum(trs) / len(trs)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


# --------------------------------------------------------------------------- #
# Open trade tracking (MFE / MAE)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _OpenTradeState:
    trade_id: str
    symbol: str
    direction: str           # "BUY" / "SELL"
    entry_price: float
    open_time: datetime
    sl_distance: float       # distancia precio→SL (positiva)
    mfe_r: float = 0.0       # max favorable excursion (en R)
    mae_r: float = 0.0       # max adverse excursion (en R) — guardada como valor positivo
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    smc_event_ids: list[str] = field(default_factory=list)
    pattern_keys: list[str] = field(default_factory=list)
    setup_type: str | None = None
    session: str | None = None
    regime: str | None = None


# --------------------------------------------------------------------------- #
# Engine principal
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _RegimeEpisode:
    regime: str
    started_at: datetime
    episode_id: str | None = None


class XAUKnowledgeEngine:
    """Observador + memoria + coordinador de reentrenamiento XAUUSD.

    El Orchestrator lo instancia (opcional, vía ``parameters.yaml``) y le
    pasa hooks. El engine **nunca** ejecuta órdenes ni gestiona riesgo: sólo
    aprende.
    """

    def __init__(
        self,
        config: XAUKnowledgeConfig | None = None,
        *,
        repository: XAUKnowledgeRepository | None = None,
        retrain_callback: Callable[..., dict[str, float] | None] | None = None,
        dataset_provider: Callable[..., list[Mapping[str, Any]]] | None = None,
    ) -> None:
        """Args:
        config: configuración (defaults seguros para XAUUSD).
        repository: DAO; si ``None`` se crea uno nuevo.
        retrain_callback: función ``(samples, sample_weights, **kwargs) ->
            dict métricas`` que ejecuta el reentrenamiento real (típicamente
            ``ml.trainer.Trainer.train_walk_forward_from_repository`` envuelto
            con filtro XAUUSD-only y pesos temporales).
        dataset_provider: función ``(symbol, limit) -> [rows]`` que devuelve
            el dataset filtrado XAUUSD-only (no contamina con BTC/USTEC).
        """
        self._cfg = config or XAUKnowledgeConfig()
        self._repo = repository or XAUKnowledgeRepository()
        self._retrain_cb = retrain_callback
        self._dataset_provider = dataset_provider

        self._market_buffer: list[dict[str, Any]] = []
        self._last_bar_key: dict[str, tuple[str, datetime]] = {}  # (tf, last_bucket)
        self._closed_trades_since_retrain: int = 0
        self._last_retrain_at: datetime | None = None
        self._last_eod_retrain_date: date | None = None
        self._current_regime_episode: _RegimeEpisode | None = None
        self._recent_outcomes: deque[float] = deque(maxlen=self._cfg.retrain_degradation_window_trades)
        self._baseline_winrate: float | None = None
        self._open_trades: dict[str, _OpenTradeState] = {}
        # Atajos
        self._pip_size: float = 0.10  # XAUUSD
        logger.info(
            "[KNOWLEDGE] XAUKnowledgeEngine inicializado (model_path=%s, triggers=%s/%s/%s/%s)",
            self._cfg.model_path,
            "closed_trades>=" + str(self._cfg.retrain_closed_trades_n),
            "EOD@%02dh" % self._cfg.retrain_end_of_day_utc_hour,
            "regime_change>=" + str(self._cfg.retrain_regime_change_min_duration_min) + "m",
            "winrate_drop>" + str(self._cfg.retrain_degradation_winrate_drop),
        )

    # ------------------------------------------------------------------ #
    # Properties / status
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled)

    @property
    def config(self) -> XAUKnowledgeConfig:
        return self._cfg

    def stats(self) -> dict[str, Any]:
        return {
            "open_trades": len(self._open_trades),
            "closed_since_retrain": self._closed_trades_since_retrain,
            "last_retrain_at": self._last_retrain_at.isoformat() if self._last_retrain_at else None,
            "current_regime": self._current_regime_episode.regime if self._current_regime_episode else None,
            "baseline_winrate": self._baseline_winrate,
            "buffer_size": len(self._market_buffer),
        }

    # ------------------------------------------------------------------ #
    # Observer: vela cerrada
    # ------------------------------------------------------------------ #

    def observe_bar_close(
        self,
        *,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        ohlc: tuple[float, float, float, float],   # (o, h, l, c)
        tick_volume: float | None = None,
        spread_points: float | None = None,
        regime: str | None = None,
        trend: str | None = None,
        range_pct: float | None = None,
        atr: float | None = None,
        volatility_zscore: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Registra una vela cerrada para el knowledge memory.

        Se llama desde el Orchestrator cuando una vela del TF supervisado
        cierra (no por tick). El engine bufferiza y flushea cada
        ``market_memory_flush_size`` filas.
        """
        if not self.enabled or symbol.upper() != self._cfg.symbol.upper():
            return
        tf = timeframe.upper()
        if tf not in self._cfg.observe_timeframes:
            return
        # Evitar duplicados si el mismo bucket llega dos veces.
        key = (tf, timestamp)
        last = self._last_bar_key.get(tf)
        if last == key:
            return
        self._last_bar_key[tf] = key

        o, h, l, c = ohlc
        session = _classify_session(timestamp, self._cfg.sessions_utc)
        row = {
            "timestamp": timestamp,
            "symbol": symbol.upper(),
            "timeframe": tf,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "tick_volume": float(tick_volume) if tick_volume is not None else None,
            "spread_points": float(spread_points) if spread_points is not None else None,
            "regime": regime,
            "trend": trend,
            "range_pct": float(range_pct) if range_pct is not None else None,
            "volatility_atr": float(atr) if atr is not None else None,
            "volatility_zscore": float(volatility_zscore) if volatility_zscore is not None else None,
            "session": session,
            "hour_utc": int(timestamp.astimezone(timezone.utc).hour if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc).hour),
            "extra": dict(extra or {}),
        }
        self._market_buffer.append(row)

        # Acumular session memory daily (granularidad diaria).
        try:
            self._repo.upsert_session_day(
                session=session,
                day=timestamp,
                bars_n_delta=1,
                atr_sample=float(atr) if atr is not None else None,
                range_pct_sample=float(range_pct) if range_pct is not None else None,
                spread_sample=float(spread_points) if spread_points is not None else None,
                symbol=symbol,
            )
        except Exception as exc:
            logger.debug("[KNOWLEDGE] session upsert falló: %s", exc)

        # Track regime episodes — sólo nos quedamos con el régimen del TF
        # principal para evitar fragmentación. Usamos H1 como referencia
        # (en su defecto el primero disponible).
        if tf in ("H1", "M30", "M15"):
            self._maybe_transition_regime(regime, timestamp)

        if len(self._market_buffer) >= self._cfg.market_memory_flush_size:
            self.flush_market_memory()

    def flush_market_memory(self) -> int:
        if not self._market_buffer:
            return 0
        rows = self._market_buffer
        self._market_buffer = []
        try:
            return self._repo.insert_market_memory_batch(rows)
        except Exception as exc:
            logger.warning("[KNOWLEDGE] flush_market_memory falló: %s", exc)
            return 0

    def _maybe_transition_regime(self, regime: str | None, when_utc: datetime) -> None:
        if not regime:
            return
        new_regime = regime.upper()
        current = self._current_regime_episode
        if current is None:
            ep_id = self._repo.start_regime_episode(regime=new_regime, started_at=when_utc, symbol=self._cfg.symbol)
            self._current_regime_episode = _RegimeEpisode(
                regime=new_regime,
                started_at=when_utc,
                episode_id=str(ep_id) if ep_id is not None else None,
            )
            return
        if current.regime != new_regime:
            self._repo.close_regime_episode(
                episode_id=current.episode_id,
                ended_at=when_utc,
            )
            duration_min = int((when_utc - current.started_at).total_seconds() // 60)
            if duration_min >= self._cfg.retrain_regime_change_min_duration_min:
                # Trigger candidato anotado para que el coordinador lo evalúe.
                self._flag_regime_change_trigger = True  # type: ignore[attr-defined]
            ep_id = self._repo.start_regime_episode(regime=new_regime, started_at=when_utc, symbol=self._cfg.symbol)
            self._current_regime_episode = _RegimeEpisode(
                regime=new_regime,
                started_at=when_utc,
                episode_id=str(ep_id) if ep_id is not None else None,
            )

    # ------------------------------------------------------------------ #
    # SMC events
    # ------------------------------------------------------------------ #

    def observe_smc_event(
        self,
        *,
        event_kind: str,
        timeframe: str,
        timestamp: datetime,
        direction: str | None = None,
        strength: float | None = None,
        quality: float | None = None,
        regime: str | None = None,
        context: Mapping[str, Any] | None = None,
        symbol: str = "XAUUSD",
    ) -> str | None:
        if not self.enabled or symbol.upper() != self._cfg.symbol.upper():
            return None
        session = _classify_session(timestamp, self._cfg.sessions_utc)
        hour = timestamp.astimezone(timezone.utc).hour if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc).hour
        try:
            event_id = self._repo.insert_smc_event(
                timestamp=timestamp,
                event_kind=event_kind,
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                quality=quality,
                session=session,
                regime=regime,
                hour_utc=int(hour),
                context=context,
                symbol=symbol,
            )
            return str(event_id) if event_id is not None else None
        except Exception as exc:
            logger.debug("[KNOWLEDGE] observe_smc_event falló: %s", exc)
            return None

    def register_pattern_combo(
        self,
        *,
        names: Sequence[str],
        timeframe: str,
        direction: str | None,
        when_utc: datetime,
        score: float | None = None,
        symbol: str = "XAUUSD",
    ) -> str:
        """Registra una combinación (e.g. ``["OB","FVG"]`` → ``OB+FVG``) en
        ``xau_patterns`` incrementando ``occurrences``.
        """
        if not names:
            return ""
        key = "+".join(sorted({n.upper() for n in names}))
        try:
            self._repo.upsert_pattern(
                pattern_key=key,
                timeframe=timeframe,
                direction=direction,
                when_utc=when_utc,
                increment_occurrence=True,
                score=score,
                symbol=symbol,
            )
        except Exception as exc:
            logger.debug("[KNOWLEDGE] register_pattern_combo falló: %s", exc)
        return key

    # ------------------------------------------------------------------ #
    # Trade lifecycle: open / live update / close
    # ------------------------------------------------------------------ #

    def on_trade_open(
        self,
        *,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_distance_price: float,
        open_time: datetime,
        smc_event_ids: Iterable[str] | None = None,
        pattern_keys: Iterable[str] | None = None,
        setup_type: str | None = None,
        regime: str | None = None,
    ) -> None:
        if not self.enabled or symbol.upper() != self._cfg.symbol.upper():
            return
        session = _classify_session(open_time, self._cfg.sessions_utc)
        st = _OpenTradeState(
            trade_id=str(trade_id),
            symbol=symbol.upper(),
            direction=direction.upper(),
            entry_price=float(entry_price),
            open_time=open_time,
            sl_distance=max(float(sl_distance_price), 1e-9),
            smc_event_ids=[str(x) for x in (smc_event_ids or []) if x],
            pattern_keys=[str(x) for x in (pattern_keys or []) if x],
            setup_type=setup_type,
            session=session,
            regime=regime,
        )
        self._open_trades[str(trade_id)] = st

    def on_trade_tick(
        self,
        *,
        trade_id: str,
        current_price: float,
        when_utc: datetime,
    ) -> None:
        """Actualiza MFE/MAE en R-multiples de un trade vivo.

        No persiste por tick — los valores quedan en memoria hasta
        ``on_trade_close``. (Mantener un MFE/MAE en PG por tick saturaría la
        DB; el snapshot se guarda al cierre).
        """
        if not self.enabled:
            return
        st = self._open_trades.get(str(trade_id))
        if st is None:
            return
        if st.symbol != self._cfg.symbol.upper():
            return
        dist = current_price - st.entry_price
        if st.direction == "BUY":
            favorable = max(dist, 0.0)
            adverse = max(-dist, 0.0)
        else:
            favorable = max(-dist, 0.0)
            adverse = max(dist, 0.0)
        favorable_r = favorable / st.sl_distance
        adverse_r = adverse / st.sl_distance
        if favorable_r > st.mfe_r:
            st.mfe_r = favorable_r
        if adverse_r > st.mae_r:
            st.mae_r = adverse_r
        st.last_update = when_utc

    def on_trade_close(
        self,
        *,
        trade_id: str,
        symbol: str,
        close_time: datetime,
        r_multiple: float,
        win: bool,
        exit_price: float | None = None,
    ) -> None:
        if not self.enabled or symbol.upper() != self._cfg.symbol.upper():
            return
        st = self._open_trades.pop(str(trade_id), None)
        mfe = float(st.mfe_r) if st is not None else 0.0
        mae = float(st.mae_r) if st is not None else 0.0
        # Acumulamos para detección de degradación.
        self._recent_outcomes.append(1.0 if win else 0.0)
        self._closed_trades_since_retrain += 1
        # Session memory: outcome del trade
        try:
            sess = (st.session if st is not None else _classify_session(close_time, self._cfg.sessions_utc))
            self._repo.upsert_session_day(
                session=sess,
                day=close_time,
                trade_outcome=(bool(win), float(r_multiple)),
                symbol=symbol,
            )
        except Exception as exc:
            logger.debug("[KNOWLEDGE] session trade upsert: %s", exc)
        # Patterns outcome
        if st is not None and st.pattern_keys:
            for pkey in st.pattern_keys:
                try:
                    # Inferimos timeframe del primer evento SMC; si no hay, M15 default.
                    tf_hint = "M15"
                    self._repo.upsert_pattern(
                        pattern_key=pkey,
                        timeframe=tf_hint,
                        direction=st.direction,
                        when_utc=close_time,
                        increment_occurrence=False,
                        outcome_win=bool(win),
                        r_multiple=float(r_multiple),
                        mfe_r=mfe,
                        mae_r=mae,
                        symbol=symbol,
                    )
                except Exception as exc:
                    logger.debug("[KNOWLEDGE] pattern outcome upsert: %s", exc)
        # SMC events outcome
        if st is not None and st.smc_event_ids:
            try:
                self._repo.attach_smc_event_outcome(
                    event_ids=st.smc_event_ids,
                    r_multiple=float(r_multiple),
                    win=bool(win),
                )
            except Exception as exc:
                logger.debug("[KNOWLEDGE] smc event outcome attach: %s", exc)

    # ------------------------------------------------------------------ #
    # Retrain coordinator (4 triggers)
    # ------------------------------------------------------------------ #

    def evaluate_retrain_triggers(self, *, now_utc: datetime | None = None) -> tuple[bool, str]:
        """Evalúa los 4 triggers y devuelve ``(should_retrain, reason)``.

        Reglas:

        * ``closed_trades`` ≥ ``retrain_closed_trades_n``.
        * ``end_of_day``: hora UTC ≥ ``retrain_end_of_day_utc_hour`` y aún no
          se ha reentrenado hoy.
        * ``regime_change``: flag interno seteado al cerrar un episodio de
          régimen con duración ≥ umbral.
        * ``degradation``: caída de winrate de la ventana reciente vs baseline
          (``baseline_winrate`` se aprende del primer batch suficiente).
        """
        now = now_utc or datetime.now(timezone.utc)
        # Cooldown global
        if self._last_retrain_at is not None:
            elapsed_min = (now - self._last_retrain_at).total_seconds() / 60.0
            if elapsed_min < self._cfg.retrain_min_interval_minutes:
                return (False, f"cooldown_{int(elapsed_min)}m")

        # 1) Closed trades
        if self._closed_trades_since_retrain >= self._cfg.retrain_closed_trades_n:
            return (True, "closed_trades")

        # 2) End of day (sólo una vez por día UTC)
        if now.hour >= self._cfg.retrain_end_of_day_utc_hour:
            if self._last_eod_retrain_date != now.date():
                return (True, "end_of_day")

        # 3) Regime change (flag seteado por `_maybe_transition_regime`)
        if getattr(self, "_flag_regime_change_trigger", False):
            self._flag_regime_change_trigger = False  # consumir
            return (True, "regime_change")

        # 4) Degradation
        if len(self._recent_outcomes) >= self._cfg.retrain_degradation_window_trades:
            recent_wr = sum(self._recent_outcomes) / float(len(self._recent_outcomes))
            if self._baseline_winrate is None:
                # Establecemos baseline si aún no existe (después de cubrir ventana).
                self._baseline_winrate = recent_wr
            else:
                drop = self._baseline_winrate - recent_wr
                if drop >= self._cfg.retrain_degradation_winrate_drop:
                    return (True, f"degradation(drop={drop:.3f})")
        return (False, "")

    def trigger_retrain_if_needed(self, *, now_utc: datetime | None = None, force_reason: str | None = None) -> dict[str, Any] | None:
        """Si toca reentrenar, ejecuta el callback con dataset XAUUSD-only +
        sample_weights temporales y registra el resultado en
        ``xau_model_learning``. Devuelve resumen o ``None``.
        """
        if self._retrain_cb is None:
            logger.debug("[KNOWLEDGE] no hay retrain_callback configurado; skip.")
            return None
        now = now_utc or datetime.now(timezone.utc)
        if force_reason:
            should, reason = True, force_reason
        else:
            should, reason = self.evaluate_retrain_triggers(now_utc=now)
        if not should:
            return None
        logger.info("[KNOWLEDGE] reentrenamiento XAUUSD disparado por trigger=%s", reason)

        # 1) Cargar dataset filtrado XAUUSD-only.
        rows: list[Mapping[str, Any]] = []
        if self._dataset_provider is not None:
            try:
                rows = list(self._dataset_provider(self._cfg.symbol, 5000) or [])
            except Exception as exc:
                logger.warning("[KNOWLEDGE] dataset_provider falló: %s", exc)
        rows = [r for r in rows if str(r.get("symbol", "")).upper() == self._cfg.symbol.upper()]
        if not rows:
            logger.warning("[KNOWLEDGE] reentrenamiento abortado: sin filas XAUUSD en dataset.")
            return None

        # 2) Sample weights temporales (50/30/20).
        weights = self._time_weighted_sample_weights(rows, now=now)

        pre_metrics = self._estimate_metrics(rows)
        started = now
        try:
            metrics_post = self._retrain_cb(rows=rows, sample_weights=weights, symbol=self._cfg.symbol, reason=reason)
        except Exception as exc:
            logger.warning("[KNOWLEDGE] retrain_cb falló: %s", exc)
            metrics_post = None
        finished = datetime.now(timezone.utc)

        # 3) Registrar auditoría
        try:
            self._repo.record_model_retrain(
                trigger=reason,
                started_at=started,
                finished_at=finished,
                samples_used=len(rows),
                walk_forward_folds=self._cfg.walk_forward_folds,
                metrics_pre=pre_metrics,
                metrics_post=dict(metrics_post or {}),
                model_path=str(self._cfg.model_path),
                weights_used={
                    k: v["weight"] for k, v in self._cfg.time_weights.items()
                },
                notes=f"closed_since={self._closed_trades_since_retrain} baseline_wr={self._baseline_winrate}",
                symbol=self._cfg.symbol,
            )
        except Exception as exc:
            logger.debug("[KNOWLEDGE] record_model_retrain falló: %s", exc)

        # 4) Reset contadores
        self._closed_trades_since_retrain = 0
        self._last_retrain_at = finished
        if reason == "end_of_day":
            self._last_eod_retrain_date = finished.date()
        # Actualiza baseline_winrate con la ventana reciente como nueva referencia.
        if len(self._recent_outcomes) >= self._cfg.retrain_degradation_window_trades:
            self._baseline_winrate = sum(self._recent_outcomes) / float(len(self._recent_outcomes))

        return {
            "trigger": reason,
            "samples": len(rows),
            "metrics_pre": pre_metrics,
            "metrics_post": metrics_post,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Time-weighted sample weights (50/30/20)
    # ------------------------------------------------------------------ #

    def _time_weighted_sample_weights(
        self,
        rows: list[Mapping[str, Any]],
        *,
        now: datetime,
    ) -> list[float]:
        """Calcula pesos por muestra según el bucket temporal (más reciente
        pesa más).

        Política por defecto::

            últimos 6 meses     → 0.50
            entre 6-12 meses    → 0.30
            entre 12-36 meses   → 0.20

        Cada muestra recibe el peso de su bucket como peso *por muestra*
        (consistente con la convención ``sample_weight`` de scikit-learn:
        muestras más recientes pesan más individualmente). Tras asignar pesos
        absolutos por bucket, se re-normalizan a media 1 sobre todo el dataset
        para que el trainer interprete pesos relativos sin escalas raras.
        Muestras > 36 meses reciben peso minúsculo (0.01) — no se descartan
        pero contribuyen marginalmente.
        """
        tw = self._cfg.time_weights or DEFAULT_TIME_WEIGHTS
        # Construir lista de cutoffs ordenados de menor a mayor edad.
        sorted_buckets = sorted(tw.items(), key=lambda kv: kv[1]["max_age_days"])
        # Clasificar cada fila
        bucket_indices: dict[str, list[int]] = {k: [] for k, _ in sorted_buckets}
        overflow: list[int] = []
        for i, r in enumerate(rows):
            ts_raw = r.get("timestamp") or r.get("open_time") or r.get("close_time")
            if isinstance(ts_raw, datetime):
                ts = ts_raw
            elif isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    ts = now
            else:
                ts = now
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).total_seconds() / 86400.0
            placed = False
            for name, cfg in sorted_buckets:
                if age_days <= float(cfg["max_age_days"]):
                    bucket_indices[name].append(i)
                    placed = True
                    break
            if not placed:
                overflow.append(i)
        weights = [0.0] * len(rows)
        for name, cfg in sorted_buckets:
            ixs = bucket_indices[name]
            if not ixs:
                continue
            # Peso ABSOLUTO por muestra (sklearn convention).
            per = float(cfg["weight"])
            for i in ixs:
                weights[i] = per
        if overflow:
            for i in overflow:
                weights[i] = 0.01
        total = sum(weights)
        if total > 0:
            scale = float(len(weights)) / total
            weights = [w * scale for w in weights]
        return weights

    # ------------------------------------------------------------------ #
    # Métricas pre/post baseline
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
        """Estima winrate / avg_r / expectancy desde el dataset cargado.

        Estas métricas son la *baseline* del modelo previa al retrain (no las
        métricas del retrain en sí — éstas las devuelve el callback)."""
        n = len(rows)
        if n == 0:
            return {"winrate": 0.0, "avg_r": 0.0, "expectancy": 0.0, "sharpe": 0.0, "n": 0}
        wins = 0
        r_sum = 0.0
        r_sq_sum = 0.0
        r_count = 0
        for r in rows:
            try:
                p = float(r.get("profit") or 0.0)
                if p > 0:
                    wins += 1
            except Exception:
                pass
            raw = r.get("r_multiple")
            if raw is None:
                continue
            try:
                rv = float(raw)
            except Exception:
                continue
            r_sum += rv
            r_sq_sum += rv * rv
            r_count += 1
        winrate = float(wins) / float(n)
        avg_r = (r_sum / r_count) if r_count > 0 else 0.0
        if r_count >= 2:
            mean = avg_r
            var = max(0.0, (r_sq_sum / r_count) - mean * mean)
            std = math.sqrt(var) if var > 0 else 0.0
            sharpe = (mean / std) if std > 0 else 0.0
        else:
            sharpe = 0.0
        expectancy = avg_r  # simplificación: ya es por trade
        return {
            "winrate": winrate,
            "avg_r": avg_r,
            "expectancy": expectancy,
            "sharpe": sharpe,
            "n": float(n),
        }


__all__ = [
    "XAUKnowledgeEngine",
    "XAUKnowledgeConfig",
    "DEFAULT_TIME_WEIGHTS",
]

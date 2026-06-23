"""Límites agregados de riesgo (pérdida diaria, exposición, correlación)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from time import monotonic
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Resultado detallado de decisión de riesgo."""

    allowed: bool
    reason: str


class RiskManager:
    """Evalúa si una operación candidata respeta el marco de riesgo global."""

    def __init__(self, limits: dict[str, Any] | None = None) -> None:
        """Define límites numéricos y flags de política.

        Args:
            limits: Por ejemplo drawdown máximo, riesgo por trade, etc.
        """
        self._limits = dict(limits or {})
        self._max_drawdown_pause_pct = Decimal(str(self._limits.get("max_drawdown_pause_pct", "5")))
        self._max_drawdown_hard_stop_pct = Decimal(str(self._limits.get("max_drawdown_hard_stop_pct", "8")))
        # 0 o ausente explícito como 0 = sin tope de entradas por día/sesión (solo drawdown/racha/riesgo).
        _mt = self._limits.get("max_trades_per_session", 50)
        try:
            self._max_trades_per_session = int(_mt) if _mt is not None else 0
        except (TypeError, ValueError):
            self._max_trades_per_session = 50
        self._cooldown_seconds = int(self._limits.get("cooldown_seconds", 120))
        self._loss_streak_pause = int(self._limits.get("loss_streak_pause", 20))
        self._loss_streak_reset_after_cooldown = bool(
            self._limits.get("loss_streak_reset_after_cooldown", False)
        )
        self._risk_per_trade_pct = Decimal(str(self._limits.get("risk_per_trade_pct", "2")))

        self._session_start_equity: Decimal | None = None
        self._session_realized_pnl = Decimal("0")
        self._session_trade_count = 0
        self._loss_streak = 0
        # Loss streak POR SÍMBOLO. Las rachas de un símbolo no deben bloquear
        # otros símbolos (antes una racha de BTCUSD bloqueaba NDX100 y XAUUSD
        # vía fundingpips_guard.consecutive_losses).
        self._loss_streak_by_symbol: dict[str, int] = {}
        # Timestamp UTC de la última pérdida POR SÍMBOLO. Se usa para el
        # cooldown temporal escalonado tras N pérdidas consecutivas en el
        # mismo símbolo (anti-revenge fino antes de llegar al hard block).
        self._last_loss_time_by_symbol: dict[str, datetime] = {}
        # Escalado en segundos en función del loss_streak por símbolo.
        # 0/1 pérdida: sin cooldown. 2: 15 min. 3: 45 min. 4+: 90 min.
        # Permite enfriar el símbolo entre racha y el hard block (=5) sin
        # bloquearlo definitivamente; el resto de símbolos sigue operando.
        raw_post_loss = self._limits.get("post_loss_cooldown_seconds")
        if isinstance(raw_post_loss, Mapping):
            try:
                self._post_loss_cooldown_seconds: dict[int, int] = {
                    int(k): int(v) for k, v in raw_post_loss.items()
                }
            except (TypeError, ValueError):
                self._post_loss_cooldown_seconds = {2: 900, 3: 2700, 4: 5400}
        else:
            self._post_loss_cooldown_seconds = {2: 900, 3: 2700, 4: 5400}
        self._paused_until: datetime | None = None
        self._pause_due_to_loss_streak = False
        self._hard_stopped = False
        self._current_trading_day: date | None = None
        self._last_block_log_reason: str | None = None
        self._last_block_log_mono: float = 0.0
        self._block_log_interval_s = 90.0
        self._entry_times_utc: deque[datetime] = deque(maxlen=200)
        logger.debug("RiskManager límites: %s", list(self._limits))

    @property
    def session_start_equity(self) -> Decimal:
        """Equity al inicio de la sesión/día UTC (para límites diarios funded)."""
        if self._session_start_equity is None or self._session_start_equity <= 0:
            return Decimal("0")
        return self._session_start_equity

    @property
    def session_trade_count(self) -> int:
        """Entradas registradas en el día de trading UTC actual."""
        return self._session_trade_count

    @property
    def hard_stopped(self) -> bool:
        """Indica si se alcanzó hard stop de drawdown."""
        return self._hard_stopped

    @property
    def paused_until(self) -> datetime | None:
        """Instante UTC hasta el cual no se admiten entradas."""
        return self._paused_until

    @property
    def loss_streak(self) -> int:
        """Racha actual de pérdidas (GLOBAL, agregada de todos los símbolos).

        Para tomar decisiones por símbolo (e.g. fundingpips revenge guard) usar
        ``loss_streak_for(symbol)``.
        """
        return self._loss_streak

    def loss_streak_for(self, symbol: str | None) -> int:
        """Racha actual de pérdidas para un símbolo concreto.

        Si ``symbol`` es None o no se ha visto antes, devuelve 0 (sin racha).
        """
        if not symbol:
            return 0
        return int(self._loss_streak_by_symbol.get(symbol.upper(), 0))

    def hydrate_loss_streaks(self, recent_closed_trades: list[Mapping[str, Any]]) -> None:
        """Reconstruye ``_loss_streak_by_symbol`` desde trades cerrados recientes.

        Para cada símbolo cuenta consecutivamente cuántas pérdidas seguidas
        hubo hasta llegar a una victoria, desde el trade más reciente hacia
        atrás. Si el trade más reciente fue ganador, racha=0. Trades con
        ``setup_type == BROKER_SYNC`` se ignoran (no son decisiones del bot).

        Espera filas con al menos: ``symbol``, ``profit`` y opcional
        ``close_time`` (orden descendente) y ``features_json.setup_type`` o
        ``setup_type`` plano. Es defensivo ante claves faltantes.

        Tras un reinicio del proceso, esto evita que el bot opere a
        ``risk_per_trade`` completo si el último trade real fue una pérdida
        de una racha en curso.
        """
        if not recent_closed_trades:
            return
        by_symbol: dict[str, list[Mapping[str, Any]]] = {}
        for row in recent_closed_trades:
            sym = str(row.get("symbol") or "").upper().strip()
            if not sym:
                continue
            by_symbol.setdefault(sym, []).append(row)
        rebuilt_streak: dict[str, int] = {}
        rebuilt_last_loss: dict[str, datetime] = {}
        for sym, rows in by_symbol.items():
            streak = 0
            for row in rows:
                setup = str(
                    (row.get("features_json") or {}).get("setup_type")
                    if isinstance(row.get("features_json"), Mapping)
                    else row.get("setup_type") or ""
                ).upper()
                if setup == "BROKER_SYNC":
                    continue
                try:
                    profit = float(row.get("profit") or 0.0)
                except (TypeError, ValueError):
                    profit = 0.0
                if profit < 0:
                    streak += 1
                    ct = row.get("close_time")
                    if streak == 1 and ct is not None:
                        try:
                            if isinstance(ct, str):
                                rebuilt_last_loss[sym] = datetime.fromisoformat(
                                    ct.replace("Z", "+00:00")
                                )
                            elif isinstance(ct, datetime):
                                dt = ct if ct.tzinfo else ct.replace(tzinfo=timezone.utc)
                                rebuilt_last_loss[sym] = dt
                        except (TypeError, ValueError):
                            pass
                else:
                    break
            if streak > 0:
                rebuilt_streak[sym] = streak
        if rebuilt_streak:
            self._loss_streak_by_symbol.update(rebuilt_streak)
            self._last_loss_time_by_symbol.update(rebuilt_last_loss)
            logger.info(
                "RiskManager.hydrate_loss_streaks: rachas reconstruidas %s",
                rebuilt_streak,
            )

    def post_loss_cooldown_remaining_for(
        self, symbol: str | None, now_utc: datetime | None = None
    ) -> int:
        """Segundos restantes de cooldown temporal post-pérdida para un símbolo.

        Devuelve 0 si no aplica (sin racha, sin pérdida reciente, o ya cumplido).
        Lógica: dado el ``loss_streak`` del símbolo se mira la tabla
        ``_post_loss_cooldown_seconds`` y se calcula cuánto falta desde la
        última pérdida. Una victoria reinicia la racha (y por tanto el cooldown).
        """
        if not symbol:
            return 0
        key = symbol.upper()
        streak = int(self._loss_streak_by_symbol.get(key, 0))
        if streak < 2:
            return 0
        bucket_key = streak if streak in self._post_loss_cooldown_seconds else max(
            (k for k in self._post_loss_cooldown_seconds if k <= streak),
            default=0,
        )
        if bucket_key <= 0:
            return 0
        cooldown_s = int(self._post_loss_cooldown_seconds.get(bucket_key, 0))
        if cooldown_s <= 0:
            return 0
        last_loss = self._last_loss_time_by_symbol.get(key)
        if last_loss is None:
            return 0
        now = now_utc if now_utc is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed = (now - last_loss).total_seconds()
        remaining = cooldown_s - elapsed
        return int(max(0.0, remaining))

    def _refresh_trading_day(self, now_utc: datetime) -> None:
        """Avanza el día contable solo si el tick es POSTERIOR al día actual.

        Antes, durante el warm-up con ticks históricos (ordenados o no por símbolo),
        cualquier `tick.time_utc.date()` distinto reseteaba los contadores y disparaba
        un log. Ahora exigimos que `trading_day > _current_trading_day` (avance real
        de fecha UTC) para considerar "nuevo día". Tick con fecha pasada se ignora.
        """
        trading_day = now_utc.date()
        if self._current_trading_day is None:
            self._current_trading_day = trading_day
            return
        if trading_day <= self._current_trading_day:
            return
        self._current_trading_day = trading_day
        self._session_trade_count = 0
        self._session_realized_pnl = Decimal("0")
        self._entry_times_utc.clear()
        self._loss_streak = 0
        self._loss_streak_by_symbol.clear()
        self._last_loss_time_by_symbol.clear()
        self._paused_until = None
        self._pause_due_to_loss_streak = False
        self._last_block_log_reason = None
        self._last_block_log_mono = 0.0
        logger.info("RiskManager: nuevo día de mercado UTC, contadores reiniciados")

    def start_session(self, equity: Decimal, now_utc: datetime | None = None) -> None:
        """Inicializa contadores de sesión con equity base."""
        now = now_utc if now_utc is not None else datetime.now(timezone.utc)
        self._session_start_equity = equity
        self._session_realized_pnl = Decimal("0")
        self._session_trade_count = 0
        self._loss_streak = 0
        self._loss_streak_by_symbol.clear()
        self._last_loss_time_by_symbol.clear()
        self._paused_until = None
        self._pause_due_to_loss_streak = False
        self._hard_stopped = False
        self._current_trading_day = now.date()
        self._last_block_log_reason = None
        self._last_block_log_mono = 0.0
        logger.info("RiskManager.start_session equity=%s", equity)

    def _current_drawdown_pct(self, equity: Decimal) -> Decimal:
        if self._session_start_equity is None or self._session_start_equity <= 0:
            self._session_start_equity = equity
            return Decimal("0")
        dd = (self._session_start_equity - equity) / self._session_start_equity * Decimal("100")
        return max(Decimal("0"), dd)

    def assess(self, equity: Decimal, proposed_risk_money: Decimal, now_utc: datetime | None = None) -> RiskDecision:
        """Evalúa reglas de drawdown, cooldown, racha y límite de trades."""
        now = now_utc if now_utc is not None else datetime.now(timezone.utc)
        self._refresh_trading_day(now)

        if self._hard_stopped:
            return RiskDecision(False, "hard stop activo por drawdown")

        if self._paused_until is not None and now >= self._paused_until:
            if (
                self._loss_streak_reset_after_cooldown
                and self._pause_due_to_loss_streak
                and self._loss_streak_pause > 0
                and self._loss_streak >= self._loss_streak_pause
            ):
                self._loss_streak = 0
                logger.info(
                    "RiskManager: cooldown por racha finalizado; contador de racha reiniciado "
                    "(loss_streak_reset_after_cooldown)"
                )
            self._paused_until = None
            self._pause_due_to_loss_streak = False

        if self._paused_until is not None and now < self._paused_until:
            return RiskDecision(False, f"en cooldown hasta {self._paused_until.isoformat()}")

        dd_pct = self._current_drawdown_pct(equity)
        if dd_pct >= self._max_drawdown_hard_stop_pct:
            self._hard_stopped = True
            logger.error("Hard stop por drawdown %.2f%%", float(dd_pct))
            return RiskDecision(False, "drawdown >= hard stop")

        if dd_pct >= self._max_drawdown_pause_pct:
            self._paused_until = now + timedelta(seconds=self._cooldown_seconds)
            self._pause_due_to_loss_streak = False
            logger.warning("Pausa por drawdown %.2f%%", float(dd_pct))
            return RiskDecision(False, "drawdown >= pause, sesión en pausa")

        if self._max_trades_per_session > 0 and self._session_trade_count >= self._max_trades_per_session:
            return RiskDecision(False, f"máximo diario alcanzado ({self._max_trades_per_session})")

        if self._loss_streak_pause > 0 and self._loss_streak >= self._loss_streak_pause:
            self._paused_until = now + timedelta(seconds=self._cooldown_seconds)
            self._pause_due_to_loss_streak = True
            return RiskDecision(False, f"racha de pérdidas >= {self._loss_streak_pause}, pausa activada")

        risk_cap = equity * (self._risk_per_trade_pct / Decimal("100"))
        if proposed_risk_money > risk_cap:
            return RiskDecision(False, "riesgo por trade excede límite permitido")

        return RiskDecision(True, "ok")

    def can_open_new_trade(
        self,
        equity: Decimal,
        proposed_risk_money: Decimal,
        now_utc: datetime | None = None,
    ) -> bool:
        """Decide si se admite abrir una nueva operación dado el estado de cuenta.

        Args:
            equity: Capital actual de la cuenta.
            proposed_risk_money: Riesgo monetario de la operación candidata.

        Returns:
            ``True`` si la operación está permitida por riesgo (stub).
        """
        decision = self.assess(equity, proposed_risk_money, now_utc=now_utc)
        if not decision.allowed:
            now_m = monotonic()
            if (
                decision.reason != self._last_block_log_reason
                or (now_m - self._last_block_log_mono) >= self._block_log_interval_s
            ):
                logger.info("can_open_new_trade: bloqueado (%s)", decision.reason)
                self._last_block_log_reason = decision.reason
                self._last_block_log_mono = now_m
            else:
                logger.debug("can_open_new_trade: bloqueado (%s) [throttle]", decision.reason)
        return decision.allowed

    def trades_in_last_hour(self, now_utc: datetime | None = None) -> int:
        """Cuenta entradas abiertas en la última hora (UTC)."""
        now = now_utc if now_utc is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cutoff = now - timedelta(hours=1)
        return sum(1 for t in self._entry_times_utc if t >= cutoff)

    def register_trade_open(self, opened_at: datetime | None = None) -> None:
        """Registra una entrada ejecutada para límites diarios."""
        now = opened_at if opened_at is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        self._refresh_trading_day(now)
        self._session_trade_count += 1
        self._entry_times_utc.append(now)

    def register_trade_result(
        self,
        pnl_money: Decimal,
        closed_at: datetime | None = None,
        setup_type: str | None = None,
        symbol: str | None = None,
    ) -> None:
        """Actualiza acumulados intradía con el PnL de una operación cerrada.

        Si ``setup_type == "BROKER_SYNC"`` la operación fue reconciliada del bróker y NO
        proviene de una decisión propia del bot: contabiliza PnL para equity/sesión pero
        no contamina la racha (``_loss_streak``) ni dispara el ``loss_streak_revenge_guard``.

        Si ``symbol`` se proporciona, también actualiza el contador de racha por símbolo
        (``_loss_streak_by_symbol``), que es lo que consume ``fundingpips_guard`` para
        evitar que una racha de un instrumento bloquee la operativa de los demás.
        """
        now = closed_at if closed_at is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        self._refresh_trading_day(now)
        self._session_realized_pnl += pnl_money

        is_broker_sync = (setup_type or "").upper() == "BROKER_SYNC"
        if is_broker_sync:
            logger.info(
                "RiskManager: trade BROKER_SYNC pnl=%.2f no afecta loss_streak (racha=%d)",
                float(pnl_money),
                self._loss_streak,
            )
            return

        if pnl_money < 0:
            self._loss_streak += 1
            if symbol:
                key = symbol.upper()
                self._loss_streak_by_symbol[key] = self._loss_streak_by_symbol.get(key, 0) + 1
                self._last_loss_time_by_symbol[key] = now
        else:
            self._loss_streak = 0
            if symbol:
                key = symbol.upper()
                self._loss_streak_by_symbol[key] = 0
                self._last_loss_time_by_symbol.pop(key, None)

        if self._loss_streak_pause > 0 and self._loss_streak >= self._loss_streak_pause:
            self._paused_until = now + timedelta(seconds=self._cooldown_seconds)
            self._pause_due_to_loss_streak = True
            logger.warning(
                "Pausa activada por racha de pérdidas (%s)",
                self._loss_streak,
            )

"""ChopGuard — protecciones quirúrgicas contra el patrón de fallo del 2-jun-2026.

Tres reglas independientes y activables por config (`chop_guard` en parameters.yaml):

1. **Asia SL multiplier**: en horas de sesión asiática (00–07 UTC por defecto)
   el ATR M1 absoluto del XAU es bajo pero la volatilidad relativa al spread
   es alta. SLs muy ajustados son barridos por el primer impulso. Aplicamos
   un multiplicador adicional al SL en ese horario.

2. **Range-position filter**: no abrir LONG en la parte alta del rango actual
   (range_position > 0.65) ni SHORT en la baja (< 0.35). Evita comprar techos
   y vender suelos cuando hay consolidación.

3. **Micro-trade cooldown**: si los últimos N trades cerrados tienen |R|
   promedio por debajo del umbral configurado, el bot está absorbiendo
   spread sin edge real → pausa preventiva breve.

Diseño:
- 100% configurable, off por default → opt-in vía YAML.
- No toca el sizer ni el risk_manager existentes; expone helpers puros.
- Logging explícito de bloqueos para diagnóstico.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger("risk.chop_guard")


@dataclass(frozen=True)
class ChopGuardConfig:
    enabled: bool = False
    asia_sl_multiplier: float = 1.0
    asia_hours_utc: tuple[int, ...] = ()
    enforce_range_position: bool = False
    long_max_range_position: float = 1.0
    short_min_range_position: float = 0.0
    micro_trade_window: int = 0
    micro_trade_avg_abs_r_max: float = 0.0
    micro_trade_cooldown_min: int = 0
    # AURION X 4.2.2 — nuevas reglas
    consecutive_loss_window: int = 0
    consecutive_loss_min_abs_r: float = 0.0
    consecutive_loss_cooldown_min: int = 0
    min_ml_conf_scalper: float = 0.0
    enforce_ml_conf_scalper: bool = False
    blocked_setups: tuple[str, ...] = ()
    # AURION X 4.4 (10-jun-2026): boost dinamico al min_ml_conf en horas debiles.
    # Diagnostico mostro WR << 50% en h=03/05/09/17 UTC (Asia/lunch NY).
    # En esas horas exigimos ml_conf >= min_ml_conf_scalper + weak_hours_ml_boost.
    weak_hours_utc: tuple[int, ...] = ()
    weak_hours_ml_boost: float = 0.0
    blocked_hours_utc: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ChopGuardConfig":
        raw = raw or {}
        hours_raw = raw.get("asia_hours_utc") or ()
        try:
            hours = tuple(int(h) for h in hours_raw)
        except Exception:
            hours = ()
        weak_raw = raw.get("weak_hours_utc") or ()
        try:
            weak_hours = tuple(int(h) for h in weak_raw)
        except Exception:
            weak_hours = ()
        blocked_raw = raw.get("blocked_hours_utc") or ()
        try:
            blocked_hours = tuple(int(h) for h in blocked_raw)
        except Exception:
            blocked_hours = ()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            asia_sl_multiplier=float(raw.get("asia_sl_multiplier", 1.0) or 1.0),
            asia_hours_utc=hours,
            enforce_range_position=bool(raw.get("enforce_range_position", False)),
            long_max_range_position=float(raw.get("long_max_range_position", 1.0) or 1.0),
            short_min_range_position=float(raw.get("short_min_range_position", 0.0) or 0.0),
            micro_trade_window=int(raw.get("micro_trade_window", 0) or 0),
            micro_trade_avg_abs_r_max=float(raw.get("micro_trade_avg_abs_r_max", 0.0) or 0.0),
            micro_trade_cooldown_min=int(raw.get("micro_trade_cooldown_min", 0) or 0),
            consecutive_loss_window=int(raw.get("consecutive_loss_window", 0) or 0),
            consecutive_loss_min_abs_r=float(raw.get("consecutive_loss_min_abs_r", 0.0) or 0.0),
            consecutive_loss_cooldown_min=int(raw.get("consecutive_loss_cooldown_min", 0) or 0),
            min_ml_conf_scalper=float(raw.get("min_ml_conf_scalper", 0.0) or 0.0),
            enforce_ml_conf_scalper=bool(raw.get("enforce_ml_conf_scalper", False)),
            blocked_setups=tuple(
                str(s).strip().upper()
                for s in (raw.get("blocked_setups") or ())
                if str(s).strip()
            ),
            weak_hours_utc=weak_hours,
            weak_hours_ml_boost=float(raw.get("weak_hours_ml_boost", 0.0) or 0.0),
            blocked_hours_utc=blocked_hours,
        )


class ChopGuard:
    """Singleton-friendly guard. Mantiene un único cooldown_until en memoria."""

    def __init__(self, cfg: ChopGuardConfig) -> None:
        self._cfg = cfg
        self._cooldown_until: dict[str, datetime] = {}

    @property
    def config(self) -> ChopGuardConfig:
        return self._cfg

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def adjusted_sl_multiplier(self, ts_utc: datetime | None = None) -> float:
        """Devuelve el multiplicador a aplicar al SL final según la hora UTC."""
        if not self._cfg.enabled or self._cfg.asia_sl_multiplier <= 1.0:
            return 1.0
        ts = ts_utc or datetime.now(timezone.utc)
        if ts.hour in self._cfg.asia_hours_utc:
            return self._cfg.asia_sl_multiplier
        return 1.0

    def ml_conf_boost_for_hour(self, ts_utc: datetime | None = None) -> float:
        """Boost al ``min_ml_conf_scalper`` en horas debiles (h=03/05/09/17 UTC).

        AURION X 4.4 (10-jun-2026): diagnostico de 100 trades mostro WR << 50% en
        horas de baja liquidez (Asia + lunch NY). Subimos el umbral ML en esas
        horas para exigir mas conviccion del modelo.
        """
        if not self._cfg.enabled or self._cfg.weak_hours_ml_boost <= 0.0:
            return 0.0
        ts = ts_utc or datetime.now(timezone.utc)
        if ts.hour in self._cfg.weak_hours_utc:
            return self._cfg.weak_hours_ml_boost
        return 0.0

    def block_by_blocked_hour(self, ts_utc: datetime | None = None) -> tuple[bool, str]:
        """Bloqueo categorico de horas con WR=0% historico (h=09 UTC en XAU)."""
        if not self._cfg.enabled or not self._cfg.blocked_hours_utc:
            return False, ""
        ts = ts_utc or datetime.now(timezone.utc)
        if ts.hour in self._cfg.blocked_hours_utc:
            return True, f"hora UTC h={ts.hour:02d} bloqueada (WR historico <30%)"
        return False, ""

    def block_by_range_position(
        self,
        direction: str,
        range_position: float | None,
    ) -> tuple[bool, str]:
        """True si el filtro de posición en rango bloquea la entrada."""
        if not self._cfg.enabled or not self._cfg.enforce_range_position:
            return False, ""
        if range_position is None:
            return False, ""
        rp = float(range_position)
        d = (direction or "").upper()
        if d in ("BUY", "LONG") and rp > self._cfg.long_max_range_position:
            return True, (
                f"range_position={rp:.2f} > "
                f"long_max={self._cfg.long_max_range_position:.2f} "
                "(no comprar cerca del techo del rango)"
            )
        if d in ("SELL", "SHORT") and rp < self._cfg.short_min_range_position:
            return True, (
                f"range_position={rp:.2f} < "
                f"short_min={self._cfg.short_min_range_position:.2f} "
                "(no vender cerca del suelo del rango)"
            )
        return False, ""

    def maybe_arm_micro_cooldown(
        self,
        symbol: str,
        recent_r_multiples: Sequence[float],
        ts_utc: datetime | None = None,
    ) -> tuple[bool, str]:
        """Evalúa micro-trades sin edge Y losses consecutivos para armar cooldown.

        Aplica dos reglas independientes (se queda con la pausa más larga):
        1) micro-trade window: últimos N trades con |R| promedio bajo el umbral.
        2) consecutive-loss: últimos M trades todos con R < 0 y |R|>=min_abs.
        """
        if not self._cfg.enabled:
            return False, ""

        ts = ts_utc or datetime.now(timezone.utc)
        triggers: list[tuple[datetime, str]] = []

        # Regla 1 — micro trades
        if (
            self._cfg.micro_trade_window > 0
            and self._cfg.micro_trade_cooldown_min > 0
            and len(recent_r_multiples) >= self._cfg.micro_trade_window
        ):
            w = list(recent_r_multiples)[-self._cfg.micro_trade_window:]
            avg_abs_r = sum(abs(float(r)) for r in w) / len(w)
            if avg_abs_r <= self._cfg.micro_trade_avg_abs_r_max:
                until = ts + timedelta(minutes=self._cfg.micro_trade_cooldown_min)
                triggers.append((
                    until,
                    f"micro-trade cooldown: last {self._cfg.micro_trade_window} "
                    f"avg|R|={avg_abs_r:.3f} <= {self._cfg.micro_trade_avg_abs_r_max:.2f}",
                ))

        # Regla 2 — consecutive losses
        if (
            self._cfg.consecutive_loss_window > 0
            and self._cfg.consecutive_loss_cooldown_min > 0
            and self._cfg.consecutive_loss_min_abs_r > 0
            and len(recent_r_multiples) >= self._cfg.consecutive_loss_window
        ):
            w = list(recent_r_multiples)[-self._cfg.consecutive_loss_window:]
            all_losing = all(float(r) < 0 for r in w)
            all_big = all(abs(float(r)) >= self._cfg.consecutive_loss_min_abs_r for r in w)
            if all_losing and all_big:
                until = ts + timedelta(minutes=self._cfg.consecutive_loss_cooldown_min)
                triggers.append((
                    until,
                    f"{self._cfg.consecutive_loss_window} losses seguidos "
                    f"(|R|>={self._cfg.consecutive_loss_min_abs_r:.1f}) R={[round(x,2) for x in w]}",
                ))

        if not triggers:
            return False, ""

        # Tomamos la pausa más larga
        until, reason = max(triggers, key=lambda x: x[0])
        prev = self._cooldown_until.get(symbol)
        if prev is None or until > prev:
            self._cooldown_until[symbol] = until
        msg = f"{reason} → pausa hasta {until.isoformat()}"
        logger.warning("[CHOP_GUARD] %s %s", symbol, msg)
        return True, msg

    def block_by_setup_blacklist(
        self,
        setup_label: str | None,
    ) -> tuple[bool, str]:
        """Bloquea setups con edge claramente negativo en backtest histórico."""
        if not self._cfg.enabled or not self._cfg.blocked_setups:
            return False, ""
        if not setup_label:
            return False, ""
        label = str(setup_label).strip().upper()
        if not label:
            return False, ""
        # Coincidencia exacta o parcial (el setup puede venir con sufijos).
        for blocked in self._cfg.blocked_setups:
            if label == blocked or label.startswith(blocked) or blocked in label:
                return True, f"setup '{label}' en blacklist ({blocked})"
        return False, ""

    def block_by_ml_conf_scalper(
        self,
        strategy: str | None,
        ml_conf: float | None,
        ts_utc: datetime | None = None,
    ) -> tuple[bool, str]:
        """Bloquea entradas SCALPER cuando el ML está cerca de neutro."""
        if not self._cfg.enabled or not self._cfg.enforce_ml_conf_scalper:
            return False, ""
        if self._cfg.min_ml_conf_scalper <= 0:
            return False, ""
        strat = str(strategy or "").upper()
        if "SCALPER" not in strat:
            return False, ""
        if ml_conf is None:
            return False, ""
        try:
            v = float(ml_conf)
        except (TypeError, ValueError):
            return False, ""
        boost = self.ml_conf_boost_for_hour(ts_utc)
        threshold = self._cfg.min_ml_conf_scalper + boost
        if v < threshold:
            extra = f" (+{boost:.2f} weak hour)" if boost > 0 else ""
            return True, (
                f"SCALPER ml_conf={v:.3f} < min={threshold:.2f}{extra} "
                "(modelo sin convicción para scalping)"
            )
        return False, ""

    def in_cooldown(self, symbol: str, ts_utc: datetime | None = None) -> tuple[bool, str]:
        if not self._cfg.enabled:
            return False, ""
        until = self._cooldown_until.get(symbol)
        if not until:
            return False, ""
        ts = ts_utc or datetime.now(timezone.utc)
        if ts < until:
            return True, f"micro_trade_cooldown until {until.isoformat()}"
        # Cooldown vencido: limpiamos
        self._cooldown_until.pop(symbol, None)
        return False, ""

    def clear_cooldown(self, symbol: str) -> None:
        self._cooldown_until.pop(symbol, None)


__all__ = ["ChopGuard", "ChopGuardConfig"]

"""
Políticas de trading por día de la semana (lunes=0 … domingo=6, ``datetime.weekday()``).

Configuración: ``config/parameters.yaml`` → ``weekday_filters``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from strategy.structure_detector import Trend


@dataclass(frozen=True)
class WeekdayPolicy:
    """Reglas de operación para un día de la semana."""

    enabled: bool = True
    lot_factor: float = 1.0
    ml_threshold_increase: float = 0.0
    max_trades_per_symbol: int = 99
    require_mtf_full_align: bool = False
    require_m15_alignment: bool = False
    skip_setups: tuple[str, ...] = ()

    def effective_ml_threshold(self, base_threshold: float) -> float:
        return min(0.99, max(0.01, base_threshold + self.ml_threshold_increase))

    def allows_setup(self, setup_type: str) -> bool:
        if not self.skip_setups:
            return True
        if "ALL" in self.skip_setups:
            return False
        upper = setup_type.upper()
        return upper not in {s.upper() for s in self.skip_setups}


_DEFAULT_POLICY = WeekdayPolicy()


def load_weekday_policies(raw: Mapping[str, Any] | None) -> dict[int, WeekdayPolicy]:
    if not raw:
        return {}
    out: dict[int, WeekdayPolicy] = {}
    for key, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        try:
            wd = int(key)
        except (TypeError, ValueError):
            continue
        if wd < 0 or wd > 6:
            continue
        skip = cfg.get("skip_setups") or []
        if isinstance(skip, str):
            skip_tuple = (skip,)
        elif isinstance(skip, (list, tuple)):
            skip_tuple = tuple(str(x) for x in skip)
        else:
            skip_tuple = ()
        max_trades = int(cfg.get("max_trades_per_symbol", 99))
        out[wd] = WeekdayPolicy(
            enabled=bool(cfg.get("enabled", True)),
            lot_factor=float(cfg.get("lot_factor", 1.0)),
            ml_threshold_increase=float(cfg.get("ml_threshold_increase", 0.0)),
            max_trades_per_symbol=max_trades,
            require_mtf_full_align=bool(cfg.get("require_mtf_full_align", False)),
            require_m15_alignment=bool(cfg.get("require_m15_alignment", False)),
            skip_setups=skip_tuple,
        )
    return out


build_weekday_policies_from_config = load_weekday_policies


def get_config_for_day(weekday: int, raw: Mapping[str, Any] | None = None) -> WeekdayPolicy:
    policies = load_weekday_policies(raw)
    return policies.get(weekday, _DEFAULT_POLICY)


def evaluate_weekday_policy(
    symbol: str,
    setup_type: str,
    policy: WeekdayPolicy,
    *,
    mtf_trends: Optional[Mapping[str, Trend]] = None,
    base_ml_threshold: float = 0.0,
    trade_count_today: int = 0,
) -> tuple[bool, str]:
    """Devuelve ``(permitir_entrada, motivo_si_bloquea)``."""
    _ = symbol, base_ml_threshold  # reservado para telemetría futura
    if not policy.enabled:
        return False, "weekday_disabled"
    setup_u = setup_type.upper()
    if not policy.allows_setup(setup_u):
        return False, f"weekday_setup_blocked:{setup_u}"
    if policy.max_trades_per_symbol < 99 and trade_count_today >= policy.max_trades_per_symbol:
        return False, "weekday_max_trades"
    if mtf_trends:
        if policy.require_m15_alignment:
            m15 = mtf_trends.get("M15", Trend.RANGE)
            if m15 == Trend.RANGE:
                return False, "weekday_m15_align"
        if policy.require_mtf_full_align:
            for tf in ("M1", "M5", "M15"):
                if mtf_trends.get(tf, Trend.RANGE) == Trend.RANGE:
                    return False, "weekday_mtf_align"
    return True, "ok"

"""Pesos ML específicos por activo (oro, índice, crypto)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 1.0,
    "liquidity_sweep": 1.0,
    "absorption": 1.0,
    "momentum": 1.0,
    "range": 1.0,
    "session": 1.0,
    "volatility": 1.0,
    "fake_breakout": 1.0,
    "displacement": 1.0,
    "manipulation": 1.0,
}

ASSET_PRESETS: dict[str, dict[str, float]] = {
    "XAUUSD": {
        "liquidity_sweep": 1.45,
        "absorption": 1.35,
        "manipulation": 1.30,
        "volatility": 1.25,
        "session": 1.20,
        "range": 1.15,
        "momentum": 0.90,
        "displacement": 0.85,
        "fake_breakout": 1.10,
        "trend": 0.95,
    },
    "NDX100": {
        "momentum": 1.45,
        "displacement": 1.40,
        "trend": 1.30,
        "session": 1.15,
        "volatility": 1.10,
        "liquidity_sweep": 0.95,
        "absorption": 0.90,
        "range": 0.85,
        "fake_breakout": 0.90,
        "manipulation": 0.85,
    },
    "BTCUSD": {
        "volatility": 1.50,
        "fake_breakout": 1.40,
        "displacement": 1.35,
        "momentum": 1.25,
        "liquidity_sweep": 1.15,
        "range": 1.10,
        "absorption": 1.05,
        "trend": 1.0,
        "session": 1.0,
        "manipulation": 1.20,
    },
}


@dataclass(frozen=True)
class AssetMLProfile:
    symbol: str
    weights: dict[str, float]
    ml_decision_weight: float
    min_setup_grade_for_aggressive: str


def resolve_asset_ml_profile(
    symbol: str,
    asset_profiles: Mapping[str, Any] | None = None,
    institutional_cfg: Mapping[str, Any] | None = None,
) -> AssetMLProfile:
    s = symbol.upper().strip()
    base = dict(DEFAULT_WEIGHTS)
    base.update(ASSET_PRESETS.get(s, {}))
    prof = (asset_profiles or {}).get(s, {}) if isinstance(asset_profiles, dict) else {}
    if isinstance(prof, dict):
        ow = prof.get("ml_weights") or prof.get("quant_weights")
        if isinstance(ow, dict):
            for k, v in ow.items():
                try:
                    base[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
        inst = prof.get("institutional")
        if isinstance(inst, dict):
            ow2 = inst.get("ml_weights")
            if isinstance(ow2, dict):
                for k, v in ow2.items():
                    try:
                        base[str(k)] = float(v)
                    except (TypeError, ValueError):
                        pass
    dec_w = 0.65
    if isinstance(prof, dict):
        dec_w = float(prof.get("ml_decision_weight", dec_w))
    if isinstance(institutional_cfg, dict):
        per = institutional_cfg.get("asset_decision_weights")
        if isinstance(per, dict) and s in per:
            dec_w = float(per[s])
    min_grade = "B"
    if isinstance(prof, dict):
        min_grade = str(prof.get("min_setup_grade_aggressive", min_grade))
    return AssetMLProfile(
        symbol=s,
        weights=base,
        ml_decision_weight=max(0.35, min(0.85, dec_w)),
        min_setup_grade_for_aggressive=min_grade,
    )

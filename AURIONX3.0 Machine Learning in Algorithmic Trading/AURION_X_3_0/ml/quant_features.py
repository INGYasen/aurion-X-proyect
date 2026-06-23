"""Features cuantitativas extendidas (esquema v2) para ensemble ML."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping

from analytics.logger import get_logger

from .feature_engineering import FEATURE_ORDER, build_features

logger = get_logger(__name__)

QUANT_EXTENDED_ORDER: tuple[str, ...] = (
    "displacement_speed",
    "candle_imbalance",
    "rejection_ratio",
    "volatility_compression",
    "breakout_quality",
    "session_aggression",
    "directional_conviction",
    "sweep_score",
    "absorption_score",
    "fake_breakout_prob",
    "liquidity_density",
    "spread_norm",
    "momentum_strength",
    "range_position",
    "mtf_alignment",
)

FEATURE_ORDER_V2: tuple[str, ...] = FEATURE_ORDER + QUANT_EXTENDED_ORDER
FEATURE_SCHEMA_VERSION_V2: int = 2
FEATURE_VECTOR_LEN_V2: int = len(FEATURE_ORDER_V2)

# AURION X 4.3 (7-jun-2026): features macro para schema v3.
# Correlaciones criticas para XAUUSD:
#   - DXY: correlacion inversa ~-0.7 (oro como anti-USD)
#   - US10Y: yield real = costo oportunidad de holdear oro
#   - VIX: risk-off => vuelo a oro como safe-haven
MACRO_FEATURES_ORDER: tuple[str, ...] = (
    "dxy_return_24h",
    "dxy_zscore_5d",
    "us10y_change_24h",
    "vix_level",
    "vix_change_24h",
)

FEATURE_ORDER_V3: tuple[str, ...] = FEATURE_ORDER_V2 + MACRO_FEATURES_ORDER
FEATURE_SCHEMA_VERSION_V3: int = 3
FEATURE_VECTOR_LEN_V3: int = len(FEATURE_ORDER_V3)


def _parse_ts(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str):
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _body_imbalance(o: float, h: float, l: float, c: float) -> float:
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    return max(-1.0, min(1.0, (c - o) / rng)) if rng > 0 else 0.0


def _compute_extended(
    market_state: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> dict[str, float]:
    ctx = dict(context or {})
    closes = list(market_state.get("closes_m1") or [])
    highs = list(market_state.get("highs_m1") or closes)
    lows = list(market_state.get("lows_m1") or closes)
    n = len(closes)
    out: dict[str, float] = {k: 0.0 for k in QUANT_EXTENDED_ORDER}

    if n < 2:
        return out

    c_last = float(closes[-1])
    c_prev = float(closes[-2])
    h_last = float(highs[-1])
    l_last = float(lows[-1])
    o_last = float(closes[-2]) if n >= 2 else c_prev

    atr = float(ctx.get("atr_m1") or market_state.get("atr_m1") or 0.0)
    if atr <= 0 and n >= 14:
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, n)]
        atr = sum(trs[-14:]) / min(14, len(trs)) if trs else 0.0

    ret_1 = (c_last - c_prev) / c_prev if c_prev else 0.0
    out["displacement_speed"] = float(ret_1 / max(atr, 1e-9)) if atr > 0 else float(ret_1 * 100.0)
    out["candle_imbalance"] = _body_imbalance(o_last, h_last, l_last, c_last)

    upper_wick = h_last - max(o_last, c_last)
    lower_wick = min(o_last, c_last) - l_last
    rng = max(h_last - l_last, 1e-12)
    out["rejection_ratio"] = float((upper_wick + lower_wick) / rng)

    if n >= 20:
        recent_atr = sum(
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            for i in range(n - 14, n)
        ) / 14.0
        baseline_atr = sum(
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            for i in range(max(1, n - 40), n - 14)
        ) / max(1, min(26, n - 15))
        if baseline_atr > 0:
            out["volatility_compression"] = float(min(2.0, recent_atr / baseline_atr))

    out["breakout_quality"] = float(ctx.get("breakout_quality", ctx.get("volume_score", 0.0)) or 0.0)
    ts = _parse_ts(market_state.get("timestamp_utc"))
    hour = ts.hour + ts.minute / 60.0
    # Sesiones aproximadas UTC: Londres 7-16, NY 13-21
    london = 1.0 if 7.0 <= hour <= 16.0 else 0.0
    ny = 1.0 if 13.0 <= hour <= 21.0 else 0.0
    out["session_aggression"] = float(min(1.0, 0.5 * london + 0.5 * ny))

    mtf_align = float(ctx.get("mtf_alignment", 0.0) or 0.0)
    if mtf_align <= 0:
        trends = ctx.get("mtf_trends") or {}
        if isinstance(trends, dict):
            d = str(ctx.get("direction", "")).upper()
            score = 0.0
            for tf in ("M1", "M5", "M15", "H1"):
                t = str(trends.get(tf, "RANGE")).upper()
                if d == "BUY" and t == "UP":
                    score += 0.25
                elif d == "SELL" and t == "DOWN":
                    score += 0.25
            mtf_align = score
    out["mtf_alignment"] = float(min(1.0, mtf_align))
    out["directional_conviction"] = float(
        min(1.0, abs(float(market_state.get("direction", 0) or 0)) * 0.4 + mtf_align * 0.6),
    )

    out["sweep_score"] = float(ctx.get("liquidity_sweep_score", ctx.get("sweep_score", 0.0)) or 0.0)
    if bool(ctx.get("liquidity_sweep", False)):
        out["sweep_score"] = max(out["sweep_score"], 0.65)
    out["absorption_score"] = float(ctx.get("absorption_score", 0.0) or 0.0)
    if bool(ctx.get("pullback", False)):
        out["absorption_score"] = max(out["absorption_score"], 0.55)

    bo_p = float(ctx.get("range_breakout_probability", ctx.get("fake_breakout_probability", 0.0)) or 0.0)
    out["fake_breakout_prob"] = float(min(1.0, bo_p / 100.0 if bo_p > 1.0 else bo_p))

    vol_proxy = float(market_state.get("tick_volume_proxy", 0.0) or 0.0)
    out["liquidity_density"] = float(min(1.0, math.log1p(max(0.0, vol_proxy)) / 8.0))

    spread_pts = float(ctx.get("spread_points", 0.0) or 0.0)
    spread_cap = float(ctx.get("spread_cap_points", 280.0) or 280.0)
    out["spread_norm"] = float(min(1.0, spread_pts / max(spread_cap, 1.0)))

    out["momentum_strength"] = float(
        min(1.0, abs(ctx.get("continuation_strength", ctx.get("momentum_score", 0.0)) or 0.0)),
    )
    out["range_position"] = float(ctx.get("position_in_range", ctx.get("range_position", 0.5)) or 0.5)

    return out


def build_quant_features(
    market_state: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Vector v1+v2 y dict unificado para persistencia en BD."""
    base_vec, base_dict = build_features(market_state)
    ext = _compute_extended(market_state, context)
    vector = base_vec + [float(ext[k]) for k in QUANT_EXTENDED_ORDER]
    features_dict: dict[str, Any] = {**base_dict, **ext, "feature_schema_version": FEATURE_SCHEMA_VERSION_V2}
    return vector, features_dict


def vector_from_features_json(raw: Any, *, schema: str = "v2") -> list[float] | None:
    """Reconstruye vector desde ``features_json`` guardado en trade."""
    if schema == "v1":
        order = FEATURE_ORDER
    elif schema == "v3":
        order = FEATURE_ORDER_V3
    else:
        order = FEATURE_ORDER_V2
    if isinstance(raw, dict):
        try:
            return [float(raw.get(k, 0.0) or 0.0) for k in order]
        except (TypeError, ValueError):
            return None
    if isinstance(raw, list):
        try:
            vals = [float(v) for v in raw]
        except (TypeError, ValueError):
            return None
        if len(vals) >= len(order):
            return vals[: len(order)]
        return vals + [0.0] * (len(order) - len(vals))
    return None

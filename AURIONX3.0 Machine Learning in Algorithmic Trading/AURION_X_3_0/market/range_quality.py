"""Calidad de rango M1: chop vs rango operable, absorción/sweeps y probabilidad de ruptura.

Sin datos de libro de órdenes: absorción y delta se aproximan con mechas, cierre y volumen tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)


class RangeMarketMode(StrEnum):
    """Modo operativo inferido en la ventana M1."""

    CHOP = "CHOP"
    RANGE_TRADEABLE = "RANGE_TRADEABLE"
    BREAKOUT_WATCH = "BREAKOUT_WATCH"


@dataclass(frozen=True, slots=True)
class RangeQualityReport:
    """Salida de :func:`assess_range_at_index`."""

    range_score: float
    """0–100: mayor = rango más limpio / institucional."""

    breakout_probability: float
    """0–100: mayor = más presión hacia ruptura."""

    mode: RangeMarketMode
    chop_score: float
    """0–100: mayor = más caótico (overlap, cambios de signo, mechas erráticas)."""

    support_resistance_clarity: float
    rejection_touches: int
    overlap_ratio: float
    stability_score: float
    absorption_buy: float
    absorption_sell: float
    compression_ratio: float
    components: dict[str, float] = field(default_factory=dict)


def _true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))


def _bar_tr(highs: list[float], lows: list[float], closes: list[float], j: int) -> float:
    if j <= 0:
        return max(highs[j] - lows[j], 1e-12)
    return _true_range(highs[j], lows[j], closes[j - 1])


def _open_proxy(highs: list[float], lows: list[float], closes: list[float], j: int) -> float:
    if j <= 0:
        return float(closes[j])
    return float((highs[j - 1] + lows[j - 1] + closes[j - 1]) / 3.0)


def _default_rq_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "min_range_score": 42.0,
        "max_chop_score": 58.0,
        "breakout_watch_min": 62.0,
        "skip_fade_on_breakout_watch": True,
        "center_deadband_low": 0.38,
        "center_deadband_high": 0.62,
        "touch_zone_frac": 0.06,
        "min_rejections": 1,
    }


def assess_range_at_index(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    i: int,
    lookback: int,
    rh: float,
    rl: float,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> RangeQualityReport:
    """Evalúa calidad de rango en ``[i-lookback, i]`` con soporte/resistencia ya definidos (rh, rl)."""
    merged = {**_default_rq_cfg(), **dict(cfg or {})}
    if not bool(merged.get("enabled", True)):
        return RangeQualityReport(
            range_score=70.0,
            breakout_probability=25.0,
            mode=RangeMarketMode.RANGE_TRADEABLE,
            chop_score=30.0,
            support_resistance_clarity=0.5,
            rejection_touches=2,
            overlap_ratio=0.25,
            stability_score=0.5,
            absorption_buy=0.5,
            absorption_sell=0.5,
            compression_ratio=0.5,
            components={"disabled": 1.0},
        )

    left = max(1, i - max(8, lookback))
    rng = max(rh - rl, 1e-12)
    width_pct = rng / max(abs(closes[i]), 1e-12)
    touch_frac = float(merged.get("touch_zone_frac", 0.06))
    tol = touch_frac * rng

    n = i - left + 1
    touches_hi = 0
    touches_lo = 0
    reject_up = 0
    reject_dn = 0
    overlaps: list[float] = []
    signs: list[int] = []

    for j in range(left, i + 1):
        h, l, c = highs[j], lows[j], closes[j]
        o = _open_proxy(highs, lows, closes, j)
        body = abs(c - o) + 1e-9
        if h >= rh - tol:
            touches_hi += 1
            upper_wick = h - max(c, o)
            if upper_wick / max(h - l, 1e-12) >= 0.38 and c < o + body * 0.15:
                reject_up += 1
        if l <= rl + tol:
            touches_lo += 1
            lower_wick = min(c, o) - l
            if lower_wick / max(h - l, 1e-12) >= 0.38 and c > o - body * 0.15:
                reject_dn += 1

        if j > left:
            h0, l0 = highs[j - 1], lows[j - 1]
            inner = min(h, h0) - max(l, l0)
            r0 = max(h0 - l0, h - l, 1e-12)
            overlaps.append(max(0.0, inner) / r0)
        if j >= left + 2:
            d1 = closes[j] - closes[j - 1]
            d2 = closes[j - 1] - closes[j - 2]
            s1 = 1 if d1 > 0 else (-1 if d1 < 0 else 0)
            s2 = 1 if d2 > 0 else (-1 if d2 < 0 else 0)
            if s1 != 0 and s2 != 0:
                signs.append(1 if s1 != s2 else 0)

    overlap_ratio = sum(overlaps) / max(len(overlaps), 1)
    flip_rate = sum(signs) / max(len(signs), 1)

    mids = [(highs[j] + lows[j]) / 2.0 for j in range(left, i + 1)]
    mid_mean = sum(mids) / len(mids)
    mid_std = (sum((m - mid_mean) ** 2 for m in mids) / max(len(mids), 1)) ** 0.5
    stability = 1.0 - min(1.0, (mid_std / max(rng, 1e-12)) * 4.0)

    clarity = min(1.0, (touches_hi + touches_lo) / max(n * 0.22, 1.0))
    rejection_touches = reject_up + reject_dn
    rej_norm = min(1.0, rejection_touches / max(n * 0.12, 1.0))

    # Mechas erráticas vs cuerpo (chop proxy)
    wick_chaos = 0.0
    for j in range(left + 1, i + 1):
        o = _open_proxy(highs, lows, closes, j)
        h, l, c = highs[j], lows[j], closes[j]
        br = max(h - l, 1e-12)
        wick_ratio = (h - max(c, o) + min(c, o) - l) / br
        wick_chaos += abs(wick_ratio - 0.5)
    wick_chaos /= max(n - 1, 1)

    chop_core = (
        0.32 * flip_rate
        + 0.28 * overlap_ratio
        + 0.22 * min(1.0, wick_chaos * 1.4)
        + 0.18 * (1.0 - clarity)
    )
    chop_score = float(max(0.0, min(100.0, chop_core * 100.0)))

    range_core = (
        0.26 * clarity
        + 0.24 * rej_norm
        + 0.22 * stability
        + 0.18 * (1.0 - overlap_ratio)
        + 0.10 * min(1.0, width_pct * 800.0)
    )
    range_score = float(max(0.0, min(100.0, range_core * 100.0)))

    # Absorción (última vela y contexto)
    o_i = _open_proxy(highs, lows, closes, i)
    hi, lo, ci = highs[i], lows[i], closes[i]
    br_i = max(hi - lo, 1e-12)
    vols_seg = [max(0.0, float(volumes[k])) for k in range(left, i + 1)] if volumes else []
    vol_base = sum(vols_seg[:-1]) / max(len(vols_seg) - 1, 1) if len(vols_seg) > 2 else 1.0
    vol_i = max(0.0, float(volumes[i])) if i < len(volumes) else 0.0
    vol_ratio = vol_i / max(vol_base, 1e-9)

    lower_w = (min(ci, o_i) - lo) / br_i
    upper_w = (hi - max(ci, o_i)) / br_i
    close_pos = (ci - lo) / br_i
    near_lo = (ci - rl) / rng <= 0.28
    near_hi = (rh - ci) / rng <= 0.28

    absorption_buy = float(
        max(
            0.0,
            min(
                1.0,
                0.45 * (1.0 if lower_w >= 0.42 else lower_w * 1.2)
                + 0.30 * (1.0 if close_pos >= 0.62 else 0.0)
                + 0.25 * min(1.0, (vol_ratio - 0.9) / 0.8 if vol_ratio > 0.9 else 0.0),
            ),
        )
        * (1.0 if near_lo else 0.55),
    )
    absorption_sell = float(
        max(
            0.0,
            min(
                1.0,
                0.45 * (1.0 if upper_w >= 0.42 else upper_w * 1.2)
                + 0.30 * (1.0 if close_pos <= 0.38 else 0.0)
                + 0.25 * min(1.0, (vol_ratio - 0.9) / 0.8 if vol_ratio > 0.9 else 0.0),
            ),
        )
        * (1.0 if near_hi else 0.55),
    )

    # Compresión: rango reciente vs ventana anterior
    half = max(4, (i - left) // 2)
    r1 = max(highs[i - half + 1 : i + 1]) - min(lows[i - half + 1 : i + 1])
    r2 = max(highs[left : i - half + 1]) - min(lows[left : i - half + 1]) if i - half > left else rng
    compression_ratio = float(min(1.5, max(0.2, r2 / max(r1, 1e-12))))

    # ATR corto vs largo (expansión)
    trs_short: list[float] = []
    trs_long: list[float] = []
    for j in range(max(1, i - 5), i + 1):
        trs_short.append(_bar_tr(highs, lows, closes, j))
    for j in range(max(1, i - 20), i + 1):
        trs_long.append(_bar_tr(highs, lows, closes, j))
    atr_s = sum(trs_short) / len(trs_short)
    atr_l = sum(trs_long) / len(trs_long)
    atr_ratio = atr_s / max(atr_l, 1e-12)

    impulse_seq = 0
    for j in range(max(left, i - 4), i + 1):
        if closes[j] > closes[j - 1]:
            impulse_seq = impulse_seq + 1 if impulse_seq >= 0 else 1
        elif closes[j] < closes[j - 1]:
            impulse_seq = impulse_seq - 1 if impulse_seq <= 0 else -1
        else:
            impulse_seq = 0
    impulse_score = min(1.0, abs(impulse_seq) / 4.0)

    bo_core = (
        0.28 * min(1.0, max(0.0, atr_ratio - 0.85) * 2.5)
        + 0.26 * min(1.0, (compression_ratio - 0.55) * 2.0) * (1.0 if compression_ratio > 0.55 else 0.0)
        + 0.22 * impulse_score
        + 0.14 * min(1.0, vol_ratio / 2.0 if vol_ratio > 1.0 else 0.0)
        + 0.10 * (1.0 - stability)
    )
    breakout_probability = float(max(0.0, min(100.0, bo_core * 100.0)))

    bwm = float(merged.get("breakout_watch_min", 62.0))
    max_chop = float(merged.get("max_chop_score", 58.0))
    min_rs = float(merged.get("min_range_score", 42.0))

    if chop_score >= max_chop and breakout_probability < bwm - 8.0:
        mode = RangeMarketMode.CHOP
    elif breakout_probability >= bwm and atr_ratio >= 1.02:
        mode = RangeMarketMode.BREAKOUT_WATCH
    elif range_score >= min_rs and chop_score < max_chop + 5.0:
        mode = RangeMarketMode.RANGE_TRADEABLE
    elif chop_score >= max_chop - 5.0:
        mode = RangeMarketMode.CHOP
    else:
        mode = RangeMarketMode.RANGE_TRADEABLE

    comps = {
        "clarity": clarity,
        "rejection_norm": rej_norm,
        "stability": stability,
        "overlap": overlap_ratio,
        "flip_rate": flip_rate,
        "wick_chaos": wick_chaos,
        "atr_ratio": atr_ratio,
        "compression": compression_ratio,
        "vol_ratio_last": vol_ratio,
    }
    logger.debug("range_quality i=%s mode=%s range=%.1f chop=%.1f bo=%.1f", i, mode, range_score, chop_score, breakout_probability)
    return RangeQualityReport(
        range_score=range_score,
        breakout_probability=breakout_probability,
        mode=mode,
        chop_score=chop_score,
        support_resistance_clarity=clarity,
        rejection_touches=rejection_touches,
        overlap_ratio=overlap_ratio,
        stability_score=stability,
        absorption_buy=absorption_buy,
        absorption_sell=absorption_sell,
        compression_ratio=compression_ratio,
        components=comps,
    )

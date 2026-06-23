"""Tests del motor de calidad de rango M1."""

from __future__ import annotations

from market.range_quality import RangeMarketMode, assess_range_at_index


def _flat_range_series(n: int = 40) -> tuple[list[float], list[float], list[float], list[float]]:
    """Serie lateral con toques claros a soporte/resistencia 100 / 102."""
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    vols: list[float] = []
    for k in range(n):
        if k % 4 == 0:
            h, l, c = 101.4, 100.0, 100.35
        elif k % 4 == 1:
            h, l, c = 102.0, 100.8, 101.65
        elif k % 4 == 2:
            h, l, c = 101.2, 100.05, 100.5
        else:
            h, l, c = 101.9, 100.9, 101.4
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vols.append(120.0 + float(k))
    return highs, lows, closes, vols


def test_assess_disabled_returns_neutral_tradeable() -> None:
    h, l, c, v = _flat_range_series()
    i = len(c) - 1
    rep = assess_range_at_index(h, l, c, v, i, 30, 102.0, 100.0, cfg={"enabled": False})
    assert rep.mode == RangeMarketMode.RANGE_TRADEABLE
    assert rep.range_score >= 65.0
    assert "disabled" in rep.components


def test_assess_clean_range_prefers_tradeable_or_watch() -> None:
    h, l, c, v = _flat_range_series(45)
    i = len(c) - 1
    rep = assess_range_at_index(h, l, c, v, i, 32, 102.0, 100.0, cfg={"enabled": True})
    assert rep.mode in (RangeMarketMode.RANGE_TRADEABLE, RangeMarketMode.BREAKOUT_WATCH)
    assert 0.0 <= rep.range_score <= 100.0
    assert 0.0 <= rep.chop_score <= 100.0
    assert 0.0 <= rep.breakout_probability <= 100.0


def test_choppy_overlapping_series_raises_chop_score() -> None:
    n = 45
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    vols = [100.0] * n
    base = 100.0
    for k in range(n):
        # Alternancia fuerte y velas muy solapadas (overlap alto).
        step = 0.08 if k % 2 == 0 else -0.08
        c = base + step * (k % 7)
        br = 0.05
        highs.append(c + br * 0.55)
        lows.append(c - br * 0.45)
        closes.append(c)
    i = n - 1
    rep = assess_range_at_index(highs, lows, closes, vols, i, 30, 101.5, 99.5, cfg={"enabled": True})
    assert rep.chop_score >= rep.range_score * 0.5 or rep.mode == RangeMarketMode.CHOP

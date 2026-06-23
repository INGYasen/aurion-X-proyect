from __future__ import annotations

from decimal import Decimal

from strategy.intraday_engine import IntradayEngine
from strategy.structure_detector import StructureResult, Trend


def test_evaluate_intraday_breakout_with_m15_hint() -> None:
    """M15 direccional + ruptura M1 (parámetros relajados solo para aislar la ruta con hint)."""
    n = 52
    base = 1.0
    c = [base + i * 0.00001 for i in range(n)]
    h = [x + 0.000005 for x in c]
    l = [x - 0.000005 for x in c]
    c[-2] = 1.0195
    c[-1] = 1.0225
    h[-1] = c[-1] + 0.00001
    l[-1] = c[-2] - 0.00002

    eng = IntradayEngine(
        {
            "min_m1_bars": 50,
            "ema_fast_span": 5,
            "ema_slow_span": 13,
            "break_buffer": 0.00004,
            "break_extension_mult": 1.0,
            "min_structure_key_levels": 3,
            "enable_pullback_entries": False,
            "min_ema_separation_ratio": 0.0,
            "require_m1_bias_align": False,
            "min_breakout_body_ratio": 0.35,
            "min_bar_momentum_pct": 0.0,
        },
    )
    eng.set_m1_ohlc(h, l, c)
    hint = StructureResult(
        trend=Trend.UP,
        key_levels=(Decimal("1.0180"), Decimal("1.0190"), Decimal("1.0218")),
    )
    sig = eng.evaluate_intraday(structure_hint=hint)
    assert sig is not None
    assert sig["direction"] == "BUY"
    assert sig.get("setup_type") == "INTRADAY_BREAKOUT"


def test_evaluate_intraday_sell_breakout_with_m15_hint() -> None:
    """M15 bajista + ruptura M1 a la baja (parámetros relajados, simétrico al BUY)."""
    n = 52
    c = [1.035 - i * 0.00012 for i in range(n)]
    c[-2] = 1.0195
    c[-1] = 1.0170
    h = [x + 0.00002 for x in c]
    l = [x - 0.00002 for x in c]
    h[-1] = c[-2] + 0.00003
    l[-1] = c[-1] - 0.000025

    eng = IntradayEngine(
        {
            "min_m1_bars": 50,
            "ema_fast_span": 5,
            "ema_slow_span": 13,
            "break_buffer": 0.00004,
            "break_extension_mult": 1.0,
            "min_structure_key_levels": 3,
            "enable_pullback_entries": False,
            "min_ema_separation_ratio": 0.0,
            "require_m1_bias_align": False,
            "min_breakout_body_ratio": 0.35,
            "min_bar_momentum_pct": 0.0,
        },
    )
    eng.set_m1_ohlc(h, l, c)
    hint = StructureResult(
        trend=Trend.DOWN,
        key_levels=(Decimal("1.0160"), Decimal("1.0180"), Decimal("1.0210")),
    )
    sig = eng.evaluate_intraday(structure_hint=hint)
    assert sig is not None
    assert sig["direction"] == "SELL"
    assert sig.get("setup_type") == "INTRADAY_BREAKOUT"


def test_quality_filters_reject_chop() -> None:
    """Serie casi plana: con defaults estrictos no debe haber señal intradía."""
    n = 60
    c = [1.0 + (i % 3) * 0.000001 for i in range(n)]
    h = [x + 0.000002 for x in c]
    l = [x - 0.000002 for x in c]
    eng = IntradayEngine()
    eng.set_m1_ohlc(h, l, c)
    hint = StructureResult(
        trend=Trend.UP,
        key_levels=(Decimal("0.9995"), Decimal("1.0002"), Decimal("1.0008")),
    )
    assert eng.evaluate_intraday(structure_hint=hint) is None


def test_quality_filters_reject_chop_down_hint() -> None:
    """Serie casi plana con hint bajista: defaults estrictos → sin señal."""
    n = 60
    c = [1.0 + (i % 3) * 0.000001 for i in range(n)]
    h = [x + 0.000002 for x in c]
    l = [x - 0.000002 for x in c]
    eng = IntradayEngine()
    eng.set_m1_ohlc(h, l, c)
    hint = StructureResult(
        trend=Trend.DOWN,
        key_levels=(Decimal("0.9992"), Decimal("1.0001"), Decimal("1.0009")),
    )
    assert eng.evaluate_intraday(structure_hint=hint) is None

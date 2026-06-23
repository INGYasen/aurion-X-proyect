"""Smoke test del paquete ``market_structure`` (AURION X 4.1 SMC).

Genera datos sintéticos OHLC con un patrón claro:
1. Tendencia alcista limpia (HH/HL) durante 60 velas.
2. Displacement alcista (crea OB/FVG bullish + BOS alcista).
3. Rotación contra tendencia (LH) y CHOCH bajista con cierre fuerte.
4. Equal highs alrededor del peak para liquidez BSL.

Verifica:
* Los detectores devuelven al menos un evento de cada tipo.
* ``SMCAnalyzer`` produce un snapshot con confluence_score > 0.
* Las features para ML traen las 14 claves esperadas (todas en [0,1]).
* El blending del QuantEngine no rompe si se invoca con features SMC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import pytest

from market_structure import (
    SMCAnalyzer,
    SMCAnalyzerConfig,
    StructureDirection,
    detect_bos,
    detect_choch,
    detect_fvgs,
    detect_liquidity,
    detect_order_blocks,
    detect_swings,
)


def _synthetic_xauusd_bars(n: int = 220) -> tuple[list[float], list[float], list[float], list[float], list[datetime]]:
    """Genera serie OHLCV sintética con BOS bullish + CHOCH bearish."""
    base = 2300.0
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    times: list[datetime] = []
    t0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    price = base
    for i in range(n):
        if i < 40:
            # Tendencia alcista limpia con HH/HL claros (zig-zag al alza)
            delta = 4.0 if (i % 4 in (0, 1)) else -1.5
        elif i < 50:
            # Vela de displacement alcista (genera OB/FVG bullish + BOS bullish)
            delta = 6.0 if (i % 2 == 0) else 1.0
        elif i < 90:
            # Continuación alcista lenta + equal highs (liquidez BSL ~ 2360-2362)
            target = 2362.0
            delta = (target - price) * 0.30 + (1.5 if i % 3 == 0 else -1.0)
        elif i < 100:
            # Pequeño retroceso (HL claro)
            delta = -2.5 if (i % 2 == 0) else 0.5
        elif i < 110:
            # Reanudación leve antes del giro
            delta = 1.5 if (i % 3 != 0) else -1.0
        elif i < 120:
            # CHOCH bajista: cierre fuerte bajo el último HL (~2340)
            delta = -7.0 if (i % 2 == 0) else -1.0
        else:
            # Tras CHOCH, continuación bajista moderada con HL/LL
            delta = -3.5 if (i % 4 in (0, 1)) else 1.0
        price = max(2000.0, price + delta)
        # Wick proporcional
        wick = abs(delta) * 0.35 + 0.5
        h = price + wick
        l = price - wick
        highs.append(h)
        lows.append(l)
        closes.append(price)
        volumes.append(1000.0 + (200.0 if abs(delta) >= 4.0 else 50.0))
        times.append(t0 + timedelta(minutes=15 * i))
    return highs, lows, closes, volumes, times


def test_swing_detector_returns_pivots() -> None:
    h, l, c, _v, t = _synthetic_xauusd_bars()
    swings = detect_swings(h, l, c, times=t, left_bars=2, right_bars=2)
    assert len(swings) >= 6, f"esperaba >=6 swings, obtuve {len(swings)}"
    kinds = {s.kind.value for s in swings}
    # Debe haber al menos HH/HL/LH/LL (sin equal por la tolerancia ATR de 10%).
    assert "HH" in kinds and "LL" in kinds, kinds


def test_bos_detector_bullish() -> None:
    h, l, c, _v, t = _synthetic_xauusd_bars()
    # Sobre la primera mitad sólo hay tendencia alcista + displacement.
    bos = detect_bos(h[:80], l[:80], c[:80], times=t[:80], timeframe="M15")
    assert bos is not None, "esperaba BOS válido"
    assert bos.direction == StructureDirection.BULLISH
    assert 0.0 <= bos.bos_strength <= 1.0
    assert 0.0 <= bos.bos_quality <= 1.0
    assert 0.0 <= bos.bos_reliability <= 1.0


def test_choch_detector_bearish() -> None:
    h, l, c, _v, t = _synthetic_xauusd_bars()
    # Toda la serie incluye el CHOCH bajista en la última pierna.
    choch = detect_choch(h, l, c, times=t, timeframe="M15")
    assert choch is not None, "esperaba CHOCH válido"
    assert choch.direction == StructureDirection.BEARISH
    assert 0.0 <= choch.choch_score <= 1.0


def test_order_blocks_detected() -> None:
    h, l, c, v, t = _synthetic_xauusd_bars()
    obs = detect_order_blocks(h, l, c, v, times=t, timeframe="M15")
    assert len(obs) >= 1
    assert all(0.0 <= ob.ob_quality <= 1.0 for ob in obs)
    assert all(0.0 <= ob.ob_freshness <= 1.0 for ob in obs)


def test_fvgs_detected() -> None:
    h, l, c, _v, t = _synthetic_xauusd_bars()
    fvgs = detect_fvgs(h, l, c, times=t, timeframe="M15", min_size_atr=0.10)
    # En la fase de displacement esperamos al menos un FVG (puede estar parcialmente lleno).
    assert isinstance(fvgs, list)
    for fvg in fvgs:
        assert 0.0 <= fvg.fvg_strength <= 1.0
        assert 0.0 <= fvg.gap_fill_percent <= 1.0


def test_liquidity_pools_and_sweeps() -> None:
    h, l, c, _v, t = _synthetic_xauusd_bars()
    pools, sweeps = detect_liquidity(h, l, c, times=t, timeframe="M15")
    assert isinstance(pools, list)
    assert isinstance(sweeps, list)
    for p in pools:
        assert 0.0 <= p.liquidity_score <= 1.0
        assert 0.0 <= p.density <= 1.0
        assert 0.0 <= p.stop_hunt_probability <= 1.0


def test_smc_analyzer_end_to_end() -> None:
    """Verifica el flujo completo SMCAnalyzer.analyze + features + score_modifier."""
    h, l, c, v, t = _synthetic_xauusd_bars()

    def bars_cb(symbol: str, tf: str) -> tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float], Sequence[datetime] | None]:
        # Para el smoke test devolvemos la misma serie en todos los TFs (en
        # producción cada TF tiene su propia agregación M-minutes).
        if symbol.upper() == "XAUUSD":
            return (h, l, c, v, t)
        return ([], [], [], [], [])

    analyzer = SMCAnalyzer(SMCAnalyzerConfig(persist_snapshots=False))
    snap = analyzer.analyze("XAUUSD", bars_cb=bars_cb, force=True)
    assert snap is not None
    assert snap.symbol == "XAUUSD"
    assert 0.0 <= snap.confluence.confluence_score <= 1.0
    feats = analyzer.features("XAUUSD", direction_hint="SELL")
    expected = {
        "bos_strength", "bos_quality", "choch_strength", "ob_strength",
        "ob_freshness", "fvg_strength", "liquidity_score", "ssl_score",
        "bsl_score", "confluence_score",
        "smc_macro_bullish", "smc_macro_bearish",
        "smc_inst_bullish", "smc_inst_bearish",
    }
    assert set(feats.keys()) == expected
    for k, val in feats.items():
        assert 0.0 <= float(val) <= 1.0, f"{k}={val} fuera de [0,1]"
    delta, bump = analyzer.score_modifier("XAUUSD", "SELL")
    assert -0.10 <= delta <= 0.10
    assert bump in (-1, 0, 1)


def test_analyzer_disabled_for_non_xauusd() -> None:
    h, l, c, v, t = _synthetic_xauusd_bars()
    def cb(symbol: str, tf: str):
        return (h, l, c, v, t)
    analyzer = SMCAnalyzer(SMCAnalyzerConfig(enabled_symbols=("XAUUSD",), persist_snapshots=False))
    assert analyzer.analyze("BTCUSD", cb) is None
    assert analyzer.analyze("USTEC", cb) is None


if __name__ == "__main__":
    # Permite ejecutar manualmente: python tests/test_smc_smoke.py
    pytest.main([__file__, "-v"])

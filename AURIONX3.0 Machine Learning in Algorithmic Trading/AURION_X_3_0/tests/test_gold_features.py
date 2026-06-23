"""AURION X 4.0 — Tests para GoldFeatureProvider + schema v3."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from ml.gold_specific_features import (
    GOLD_FEATURE_ORDER,
    GOLD_FEATURE_LEN,
    GoldFeatureProvider,
)
from ml.quant_features_v3 import (
    FEATURE_ORDER_V3,
    FEATURE_VECTOR_LEN_V3,
    build_quant_features_v3,
)


def test_bundle_returns_finite_values() -> None:
    provider = GoldFeatureProvider(cfg={
        "cache_ttl_s": {k: 60 for k in GOLD_FEATURE_ORDER},
    })
    ms = {
        "eurusd_closes": list(np.linspace(1.05, 1.10, 100)),
        "sp500_atr_m5": list(np.linspace(20.0, 25.0, 80)),
        "last_price": 2050.0,
        "xagusd_last": 23.5,
        "wtiusd_last": 78.0,
        "asia_vol_vs_20dma": 1.2,
    }
    ctx = {"cb_event_minutes_to": 120}
    bundle = provider.compute(market_state=ms, context=ctx, now_utc=datetime(2026, 5, 28, 14, 30, tzinfo=timezone.utc))
    vec = bundle.as_vector()
    assert vec.size == GOLD_FEATURE_LEN
    assert np.isfinite(vec).all(), f"Algún feature NaN: {vec}"


def test_cache_ttl_respected() -> None:
    """Una source que cambia su retorno no se reflejará hasta que expire el TTL."""
    state = {"v": 5.0}

    def stub_dxy():
        return state["v"]

    provider = GoldFeatureProvider(
        cfg={"cache_ttl_s": {"dxy_zscore": 60}},
        data_sources={"dxy_zscore": stub_dxy},
    )
    v1 = provider.compute().as_dict()["dxy_zscore"]
    state["v"] = 99.0
    v2 = provider.compute().as_dict()["dxy_zscore"]
    assert v1 == v2, "Caché debe devolver mismo valor antes de expirar TTL."


def test_fallback_on_source_error() -> None:
    def bad_dxy():
        raise RuntimeError("network down")

    provider = GoldFeatureProvider(
        cfg={"cache_ttl_s": {"dxy_zscore": 60}},
        data_sources={"dxy_zscore": bad_dxy},
    )
    bundle = provider.compute()
    # Sin fuente válida: imputa con 0.0 (no debe romper).
    assert bundle.as_dict()["dxy_zscore"] == 0.0


def test_gold_silver_ratio_calculation() -> None:
    provider = GoldFeatureProvider()
    bundle = provider.compute(
        market_state={"last_price": 2050.0, "xagusd_last": 25.0}
    )
    assert bundle.as_dict()["gold_silver_ratio"] == pytest.approx(82.0)


def test_v3_vector_length_and_order() -> None:
    market_state = {
        "closes_m1": list(np.linspace(2000.0, 2010.0, 50)),
        "highs_m1": list(np.linspace(2001.0, 2011.0, 50)),
        "lows_m1": list(np.linspace(1999.0, 2009.0, 50)),
        "volumes_m1": [100.0] * 50,
        "last_price": 2010.0,
        "xagusd_last": 25.0,
        "wtiusd_last": 78.0,
    }
    vec, fdict = build_quant_features_v3(market_state)
    assert len(vec) == FEATURE_VECTOR_LEN_V3
    assert FEATURE_ORDER_V3[-1] == "london_fix_proximity"
    assert "schema_version" in fdict
    assert fdict["schema_version"] == 3

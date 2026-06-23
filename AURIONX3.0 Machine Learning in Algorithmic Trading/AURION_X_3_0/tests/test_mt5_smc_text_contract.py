"""AURION X 4.1 — Validación del contrato endpoint /api/smc/<sym>.txt ↔ EA MQL5.

Asegura que cada línea pipe-separated emitida por ``_smc_data_as_mt5_text``
tiene EXACTAMENTE el número de campos que el parser del EA exige
(verificable visualmente en ``AURION_SMC_Visualizer.mq5`` — las funciones
``DrawOBRect`` requieren np>=12, ``DrawBOSObj`` np>=11, etc.).

Esto previene regresiones futuras (e.g. añadir un campo al modelo SMC
sin actualizar el text dumper rompería el EA en silencio).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Número de campos esperados por kind (DEBE coincidir con la lógica del EA).
EXPECTED_FIELDS = {
    "OB": 12,     # OB|id|tf|dir|open_time|mitigation_time|bottom|top|state|strength|quality|freshness
    "BOS": 11,    # BOS|id|tf|dir|broken_swing_time|break_time|broken_swing_price|break_price|strength|quality|reliability
    "CHOCH": 11,  # CHOCH|id|tf|dir|kind|broken_swing_time|break_time|broken_swing_price|break_price|strength|quality
    "FVG": 11,    # FVG|id|tf|dir|open_time|inverted_at|bottom|top|state|strength|fill_pct
    "LIQ": 11,    # LIQ|id|tf|kind|open_time|swept_at|price|bound_low|bound_high|score|swept
    "CONF": 9,    # CONF|||+5 campos (4 vacíos para alinear)
}


def _fake_smc_data():
    return {
        "symbol": "XAUUSD",
        "order_blocks": [{
            "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "timeframe": "M15",
            "direction": "BULLISH",
            "open_time": "2026-05-31T20:00:00+00:00",
            "mitigation_time": None,                    # FRESH
            "bottom": 2400.5,
            "top": 2402.3,
            "state": "FRESH",
            "ob_strength": 0.85,
            "ob_quality": 0.91,
            "ob_freshness": 0.95,
        }, {
            "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "timeframe": "H1",
            "direction": "BEARISH",
            "open_time": "2026-05-30T15:00:00+00:00",
            "mitigation_time": "2026-05-31T12:00:00+00:00",
            "bottom": 2430.0,
            "top": 2433.5,
            "state": "MITIGATED",
            "ob_strength": 0.62,
            "ob_quality": 0.55,
            "ob_freshness": 0.10,
        }],
        "bos": [{
            "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "timeframe": "M15",
            "direction": "BULLISH",
            "broken_swing_time": "2026-05-31T19:30:00+00:00",
            "break_time": "2026-05-31T20:15:00+00:00",
            "broken_swing_price": 2412.0,
            "break_price": 2412.6,
            "bos_strength": 0.80,
            "bos_quality": 0.78,
            "bos_reliability": 0.84,
        }],
        "choch": [{
            "id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
            "timeframe": "M15",
            "direction": "BEARISH",
            "kind": "MAJOR",
            "broken_swing_time": "2026-05-31T18:00:00+00:00",
            "break_time": "2026-05-31T18:45:00+00:00",
            "broken_swing_price": 2418.0,
            "break_price": 2417.3,
            "choch_strength": 0.72,
            "choch_quality": 0.69,
        }],
        "fvg": [{
            "id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
            "timeframe": "M15",
            "direction": "BULLISH",
            "open_time": "2026-05-31T19:45:00+00:00",
            "inverted_at": None,
            "bottom": 2408.0,
            "top": 2409.5,
            "state": "OPEN",
            "fvg_strength": 0.65,
            "gap_fill_percent": 0.0,
        }],
        "liquidity": [{
            "id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
            "timeframe": "H1",
            "kind": "SSL",
            "open_time": "2026-05-30T18:00:00+00:00",
            "swept_at": None,
            "price": 2395.0,
            "bound_low": 2394.5,
            "bound_high": 2395.5,
            "liquidity_score": 0.71,
            "swept": False,
        }, {
            "id": "11111111-1111-1111-1111-111111111111",
            "timeframe": "H1",
            "kind": "BSL",
            "open_time": "2026-05-30T19:00:00+00:00",
            "swept_at": "2026-05-31T15:30:00+00:00",
            "price": 2440.0,
            "bound_low": 2439.5,
            "bound_high": 2440.5,
            "liquidity_score": 0.66,
            "swept": True,
        }],
        "confluence": {
            "direction_bias": "BULLISH",
            "confluence_score": 0.74,
            "macro_bias": "BULLISH",
            "institutional_bias": "BULLISH",
            "operational_bias": "NEUTRAL",
            "per_tf": {"M15": 0.6, "H1": 0.5},
        },
    }


def test_mt5_text_field_counts():
    """Cada línea debe tener EXACTAMENTE el número de campos que el EA espera."""
    from analytics import dashboard_server as ds

    with patch.object(ds, "_smc_data", return_value=_fake_smc_data()):
        body = ds._smc_data_as_mt5_text("XAUUSD", limit_per_tf=10)

    lines = [ln for ln in body.splitlines() if ln and not ln.startswith("#") and ln != "END"]
    assert lines, "endpoint vacío"
    found_kinds: set[str] = set()
    for ln in lines:
        parts = ln.split("|")
        kind = parts[0]
        found_kinds.add(kind)
        expected = EXPECTED_FIELDS.get(kind)
        assert expected is not None, f"kind desconocido emitido por endpoint: {kind}"
        assert len(parts) == expected, (
            f"{kind} → emitió {len(parts)} campos, esperaba {expected}. "
            f"Línea: {ln!r}"
        )
    # Verificar que TODOS los kinds están presentes
    for k in ("OB", "BOS", "CHOCH", "FVG", "LIQ", "CONF"):
        assert k in found_kinds, f"falta kind {k} en el output"


def test_iso_timestamps_present_in_critical_fields():
    """Los timestamps deben estar en ISO 8601 (incluyen 'T' y '+' / 'Z')."""
    from analytics import dashboard_server as ds

    with patch.object(ds, "_smc_data", return_value=_fake_smc_data()):
        body = ds._smc_data_as_mt5_text("XAUUSD", limit_per_tf=10)

    # OB FRESH debe tener open_time ISO + mitigation_time vacío
    for ln in body.splitlines():
        if ln.startswith("OB|"):
            parts = ln.split("|")
            assert "T" in parts[4] and ("+" in parts[4] or "Z" in parts[4]), \
                f"open_time no es ISO 8601: {parts[4]!r}"
            if "FRESH" in ln:
                assert parts[5] == "", f"OB FRESH debe tener mitigation_time vacío, got {parts[5]!r}"


def test_swept_flag_as_01_for_easy_mql5_parse():
    """El campo swept debe estar serializado como '0' o '1' (no True/False).
    El EA MQL5 espera ese formato."""
    from analytics import dashboard_server as ds

    with patch.object(ds, "_smc_data", return_value=_fake_smc_data()):
        body = ds._smc_data_as_mt5_text("XAUUSD", limit_per_tf=10)

    liq_lines = [ln for ln in body.splitlines() if ln.startswith("LIQ|")]
    assert len(liq_lines) == 2
    for ln in liq_lines:
        parts = ln.split("|")
        assert parts[10] in ("0", "1"), f"swept esperado '0'/'1', got {parts[10]!r}"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if callable(v) and k.startswith("test_")]
    ok = fail = 0
    for fn in tests:
        try:
            fn()
            print(f"OK    {fn.__name__}")
            ok += 1
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            fail += 1
        except Exception as exc:
            print(f"ERROR {fn.__name__}: {exc!r}")
            fail += 1
    print(f"\n{ok} OK | {fail} FAIL")
    sys.exit(0 if fail == 0 else 1)

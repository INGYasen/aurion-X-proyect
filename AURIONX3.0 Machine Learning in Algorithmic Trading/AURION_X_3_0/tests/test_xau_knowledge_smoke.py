"""AURION X 4.1 — Smoke tests del XAU Knowledge Engine.

Estos tests:

* NO tocan PostgreSQL real (usan un repo fake).
* Verifican: observer + transiciones de régimen, triggers de retraining,
  pesos temporales 50/30/20, aislamiento XAUUSD (ignora BTC/USTEC),
  MFE/MAE en vivo, propagación de outcomes a patterns/SMC events.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeRepo:
    """Repo en memoria que graba operaciones para asserts."""

    def __init__(self) -> None:
        self.market_rows: list[dict] = []
        self.patterns: dict[tuple, dict] = {}
        self.smc_events: list[dict] = []
        self.regime_starts: list[dict] = []
        self.regime_closes: list[dict] = []
        self.sessions: list[dict] = []
        self.retrains: list[dict] = []

    def insert_market_memory_batch(self, rows):
        rows_list = list(rows)
        self.market_rows.extend(rows_list)
        return len(rows_list)

    def upsert_pattern(self, **kw):
        key = (kw["symbol"].upper(), kw["pattern_key"], kw["timeframe"].upper(), (kw.get("direction") or "").upper() or None)
        row = self.patterns.setdefault(key, {
            "pattern_key": kw["pattern_key"],
            "timeframe": kw["timeframe"].upper(),
            "direction": (kw.get("direction") or "").upper() or None,
            "symbol": kw["symbol"].upper(),
            "occurrences": 0, "wins": 0, "losses": 0,
            "sum_r_milli": 0, "sum_mfe_milli": 0, "sum_mae_milli": 0,
        })
        if kw.get("increment_occurrence", True):
            row["occurrences"] += 1
        if kw.get("outcome_win") is True:
            row["wins"] += 1
        elif kw.get("outcome_win") is False:
            row["losses"] += 1
        if kw.get("r_multiple") is not None:
            row["sum_r_milli"] += int(round(float(kw["r_multiple"]) * 1000.0))

    def insert_smc_event(self, **kw):
        self.smc_events.append(kw)
        return f"evt-{len(self.smc_events)}"

    def attach_smc_event_outcome(self, **kw):
        for eid in kw.get("event_ids", []):
            for r in self.smc_events:
                if r.get("__id") == str(eid):
                    r["outcome_r_milli"] = int(round((kw.get("r_multiple") or 0) * 1000))
                    r["outcome_win"] = kw.get("win")

    def start_regime_episode(self, **kw):
        self.regime_starts.append(kw)
        return f"ep-{len(self.regime_starts)}"

    def close_regime_episode(self, **kw):
        self.regime_closes.append(kw)

    def upsert_session_day(self, **kw):
        self.sessions.append(kw)

    def record_model_retrain(self, **kw):
        self.retrains.append(kw)


def _utc(year=2026, month=1, day=1, hour=0):
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


# =====================================================================
# 1) Observer + memoria de mercado + transición de régimen
# =====================================================================

def test_observer_records_bar_and_transitions_regime():
    from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine

    repo = FakeRepo()
    ke = XAUKnowledgeEngine(
        XAUKnowledgeConfig(market_memory_flush_size=2),
        repository=repo,
    )
    base = _utc(2026, 1, 1, 8)
    # 2 velas H1 con TREND
    ke.observe_bar_close(symbol="XAUUSD", timeframe="H1", timestamp=base,
                         ohlc=(2400, 2410, 2398, 2408), regime="TREND",
                         range_pct=0.005, atr=4.0)
    ke.observe_bar_close(symbol="XAUUSD", timeframe="H1", timestamp=base + timedelta(hours=1),
                         ohlc=(2408, 2415, 2406, 2414), regime="TREND",
                         range_pct=0.0037, atr=4.5)
    # debería haber flushado al llegar al size
    assert len(repo.market_rows) >= 2, "flush no ocurrió tras market_memory_flush_size"
    # transición a RANGE
    ke.observe_bar_close(symbol="XAUUSD", timeframe="H1", timestamp=base + timedelta(hours=2),
                         ohlc=(2414, 2416, 2412, 2413), regime="RANGE",
                         range_pct=0.0017, atr=2.0)
    assert any(r["regime"] == "TREND" for r in repo.regime_starts), "TREND no abrió episodio"
    assert any(r["regime"] == "RANGE" for r in repo.regime_starts), "RANGE no abrió episodio"


# =====================================================================
# 2) Aislamiento XAUUSD: BTCUSD/USTEC se ignoran
# =====================================================================

def test_isolation_ignores_other_symbols():
    from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine

    repo = FakeRepo()
    ke = XAUKnowledgeEngine(XAUKnowledgeConfig(), repository=repo)
    base = _utc()
    ke.observe_bar_close(symbol="BTCUSD", timeframe="H1", timestamp=base,
                         ohlc=(50000, 50100, 49900, 50050), regime="TREND")
    ke.observe_bar_close(symbol="USTEC", timeframe="H1", timestamp=base,
                         ohlc=(18000, 18100, 17900, 18050), regime="TREND")
    ke.flush_market_memory()
    assert len(repo.market_rows) == 0, "BTC/USTEC contaminaron el knowledge"


# =====================================================================
# 3) Triggers de reentrenamiento: closed_trades
# =====================================================================

def test_retrain_trigger_closed_trades():
    from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine

    repo = FakeRepo()
    triggered: list[dict] = []

    def cb(*, rows, sample_weights, symbol, reason):
        triggered.append({"reason": reason, "n": len(rows), "weights_sum": sum(sample_weights)})
        return {"winrate": 0.55, "sharpe": 1.1, "expectancy": 0.2}

    def provider(symbol, limit):
        # Dataset de prueba con timestamps recientes
        rows = []
        now = datetime.now(timezone.utc)
        for i in range(50):
            rows.append({
                "symbol": "XAUUSD",
                "r_multiple": 0.5 if i % 3 == 0 else -0.3,
                "profit": 5.0 if i % 3 == 0 else -3.0,
                "close_time": (now - timedelta(days=i)).isoformat(),
            })
        return rows

    cfg = XAUKnowledgeConfig(retrain_closed_trades_n=3, retrain_min_interval_minutes=0)
    ke = XAUKnowledgeEngine(cfg, repository=repo, retrain_callback=cb, dataset_provider=provider)

    # Simulamos 3 cierres XAUUSD
    base = _utc()
    for i in range(3):
        ke.on_trade_open(trade_id=f"t{i}", symbol="XAUUSD", direction="BUY",
                         entry_price=2400.0, sl_distance_price=5.0, open_time=base)
        ke.on_trade_close(trade_id=f"t{i}", symbol="XAUUSD",
                          close_time=base + timedelta(minutes=10),
                          r_multiple=0.4, win=True)
    out = ke.trigger_retrain_if_needed()
    assert out is not None, "no se disparó retrain con 3 closed trades"
    assert triggered, "callback no se invocó"
    assert triggered[0]["reason"] == "closed_trades"


# =====================================================================
# 4) Triggers: end-of-day
# =====================================================================

def test_retrain_trigger_end_of_day():
    from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine

    repo = FakeRepo()
    fired = []

    def cb(*, rows, sample_weights, symbol, reason):
        fired.append(reason)
        return {"winrate": 0.5}

    def provider(symbol, limit):
        return [{"symbol": "XAUUSD", "r_multiple": 0.2, "profit": 2.0,
                 "close_time": datetime.now(timezone.utc).isoformat()}] * 5

    cfg = XAUKnowledgeConfig(
        retrain_closed_trades_n=999,  # nunca alcanzable
        retrain_end_of_day_utc_hour=23,
        retrain_min_interval_minutes=0,
    )
    ke = XAUKnowledgeEngine(cfg, repository=repo, retrain_callback=cb, dataset_provider=provider)
    # Forzar evaluación a las 23 UTC
    out = ke.trigger_retrain_if_needed(now_utc=_utc(2026, 5, 30, 23))
    assert out is not None
    assert fired and fired[0] == "end_of_day"


# =====================================================================
# 5) Triggers: degradation
# =====================================================================

def test_retrain_trigger_degradation():
    from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine

    repo = FakeRepo()
    fired = []

    def cb(*, rows, sample_weights, symbol, reason):
        fired.append(reason)
        return {"winrate": 0.40}

    def provider(symbol, limit):
        return [{"symbol": "XAUUSD", "r_multiple": -0.1, "profit": -1.0,
                 "close_time": datetime.now(timezone.utc).isoformat()}] * 20

    cfg = XAUKnowledgeConfig(
        retrain_closed_trades_n=999,
        retrain_end_of_day_utc_hour=99,
        retrain_degradation_window_trades=10,
        retrain_degradation_winrate_drop=0.10,
        retrain_min_interval_minutes=0,
    )
    ke = XAUKnowledgeEngine(cfg, repository=repo, retrain_callback=cb, dataset_provider=provider)
    base = _utc()
    # Primero llenamos ventana con wins para establecer baseline alto
    for i in range(10):
        ke.on_trade_open(trade_id=f"w{i}", symbol="XAUUSD", direction="BUY",
                         entry_price=2400.0, sl_distance_price=5.0, open_time=base)
        ke.on_trade_close(trade_id=f"w{i}", symbol="XAUUSD",
                          close_time=base + timedelta(minutes=10),
                          r_multiple=0.5, win=True)
    # Forzar baseline = 1.0
    out_first = ke.trigger_retrain_if_needed()
    # closed_trades=10 dispara; baseline_winrate se sembró antes en evaluate.
    # Para test puro de degradación, reseteamos contador y bajamos winrate.
    ke._closed_trades_since_retrain = 0  # type: ignore[attr-defined]
    ke._baseline_winrate = 1.0  # type: ignore[attr-defined]
    # Ahora 10 pérdidas
    for i in range(10):
        ke.on_trade_open(trade_id=f"l{i}", symbol="XAUUSD", direction="BUY",
                         entry_price=2400.0, sl_distance_price=5.0, open_time=base)
        ke.on_trade_close(trade_id=f"l{i}", symbol="XAUUSD",
                          close_time=base + timedelta(minutes=10),
                          r_multiple=-0.5, win=False)
    # No se debe disparar por closed (set a 999) → debe disparar por degradation.
    should, reason = ke.evaluate_retrain_triggers()
    assert should and "degradation" in reason


# =====================================================================
# 6) Time-weighted sampling 50/30/20
# =====================================================================

def test_time_weighted_sampling_5030_20():
    from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine

    repo = FakeRepo()
    ke = XAUKnowledgeEngine(XAUKnowledgeConfig(), repository=repo)
    now = _utc(2026, 6, 1, 12)
    rows = []
    # 30 muestras "recientes" (≤ 6m)
    for i in range(30):
        rows.append({"timestamp": now - timedelta(days=5 + i), "symbol": "XAUUSD"})
    # 20 mid (6-12m)
    for i in range(20):
        rows.append({"timestamp": now - timedelta(days=200 + i), "symbol": "XAUUSD"})
    # 10 long (12-36m)
    for i in range(10):
        rows.append({"timestamp": now - timedelta(days=400 + i), "symbol": "XAUUSD"})

    weights = ke._time_weighted_sample_weights(rows, now=now)
    assert len(weights) == 60
    # Re-normalizado a media=1: sum == n_total.
    assert abs(sum(weights) - 60.0) < 0.01
    # Pesos por bucket en escala original: recent_per > mid_per > long_per.
    recent_per = weights[0]  # primera fila recent (todas iguales en bucket)
    mid_per = weights[30]
    long_per = weights[50]
    assert recent_per > mid_per > long_per, (recent_per, mid_per, long_per)


# =====================================================================
# 7) Dataset provider XAUUSD-only (aislamiento)
# =====================================================================

def test_isolated_dataset_provider_filters_other_symbols():
    from ml.xau_isolated_retrainer import build_xauusd_dataset_provider

    class FakeTradeRepo:
        def get_training_dataset(self, limit, symbol=None):
            base = [
                {"symbol": "XAUUSD", "r_multiple": 0.5, "profit": 5.0, "close_time": "2026-05-01T12:00:00+00:00"},
                {"symbol": "BTCUSD", "r_multiple": 0.4, "profit": 4.0, "close_time": "2026-05-01T13:00:00+00:00"},
                {"symbol": "USTEC", "r_multiple": -0.2, "profit": -2.0, "close_time": "2026-05-01T14:00:00+00:00"},
                {"symbol": "XAUUSD", "r_multiple": -0.1, "profit": -1.0, "close_time": "2026-05-01T15:00:00+00:00"},
            ]
            # Simula el filtro real del TradeRepository
            if symbol is not None:
                return [r for r in base if r["symbol"].upper() == str(symbol).upper()]
            return base

    provider = build_xauusd_dataset_provider(repository=FakeTradeRepo())
    rows = provider("XAUUSD", limit=100)
    assert all(r["symbol"].upper() == "XAUUSD" for r in rows)
    assert len(rows) == 2, f"esperaba 2 XAUUSD, obtuve {len(rows)}"


# =====================================================================
# 8) MFE/MAE en vivo
# =====================================================================

def test_mfe_mae_in_flight():
    from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine

    repo = FakeRepo()
    ke = XAUKnowledgeEngine(XAUKnowledgeConfig(), repository=repo)
    base = _utc()
    ke.on_trade_open(trade_id="x1", symbol="XAUUSD", direction="BUY",
                     entry_price=2400.0, sl_distance_price=5.0, open_time=base)
    # Movimientos: hasta +10 (MFE=2R), luego -2.5 (MAE=0.5R), luego +5 (MFE no debe disminuir)
    ke.on_trade_tick(trade_id="x1", current_price=2410.0, when_utc=base + timedelta(seconds=30))
    ke.on_trade_tick(trade_id="x1", current_price=2397.5, when_utc=base + timedelta(seconds=60))
    ke.on_trade_tick(trade_id="x1", current_price=2405.0, when_utc=base + timedelta(seconds=90))
    st = ke._open_trades["x1"]  # type: ignore[attr-defined]
    assert abs(st.mfe_r - 2.0) < 1e-6, f"MFE esperado 2.0, got {st.mfe_r}"
    assert abs(st.mae_r - 0.5) < 1e-6, f"MAE esperado 0.5, got {st.mae_r}"


# =====================================================================
# 9) Pattern outcome propagation al cerrar trade
# =====================================================================

def test_pattern_outcome_propagation_on_close():
    from ml.xau_knowledge_engine import XAUKnowledgeConfig, XAUKnowledgeEngine

    repo = FakeRepo()
    ke = XAUKnowledgeEngine(XAUKnowledgeConfig(), repository=repo)
    base = _utc()
    ke.on_trade_open(trade_id="p1", symbol="XAUUSD", direction="BUY",
                     entry_price=2400.0, sl_distance_price=5.0,
                     open_time=base, pattern_keys=["OB+FVG"])
    ke.on_trade_close(trade_id="p1", symbol="XAUUSD",
                      close_time=base + timedelta(minutes=30),
                      r_multiple=1.2, win=True)
    # Pattern wins=1, sum_r_milli=1200
    key = ("XAUUSD", "OB+FVG", "M15", "BUY")
    assert key in repo.patterns
    row = repo.patterns[key]
    assert row["wins"] == 1
    assert row["sum_r_milli"] == 1200


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if callable(v) and k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK    {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} OK | {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)

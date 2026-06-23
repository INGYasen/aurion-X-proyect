"""Preflight: verifica DB + configuracion antes de arrancar bot."""
import sys
from pathlib import Path

import yaml
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.db_connection import init_db, new_session

print("[DB]")
init_db()
s = new_session()
try:
    total = s.execute(text("SELECT COUNT(*) FROM trades")).scalar()
    xau_total = s.execute(text("SELECT COUNT(*) FROM trades WHERE symbol = 'XAUUSD'")).scalar()
    xau_closed = s.execute(text(
        "SELECT COUNT(*) FROM trades WHERE symbol = 'XAUUSD' AND close_time IS NOT NULL"
    )).scalar()
    xau_v3 = s.execute(text(
        "SELECT COUNT(*) FROM trades WHERE symbol = 'XAUUSD' "
        "AND (features_json->>'macro_v3')::text = 'true'"
    )).scalar()
    print(f"  trades total:     {total}")
    print(f"  trades XAUUSD:    {xau_total}  (closed: {xau_closed})")
    print(f"  trades macro_v3:  {xau_v3}")
finally:
    s.close()

print("\n[CONFIG]")
cfg = yaml.safe_load(open("config/parameters.yaml", encoding="utf-8"))
ml = cfg.get("ml", {}) or {}
risk = cfg.get("risk", {}) or {}
mt5 = cfg.get("mt5", {}) or {}
print(f"  ml.enabled:         {ml.get('enabled')}")
print(f"  ml.model_path:      {ml.get('model_path')}")
print(f"  ml.drift.enabled:   {(ml.get('drift') or {}).get('enabled')}")
print(f"  risk.news.enabled:  {(risk.get('news_calendar') or {}).get('enabled')}")
print(f"  mt5.symbols:        {mt5.get('symbols')}")
print(f"  mt5.timeframe:      {mt5.get('timeframe')}")

print("\n[MODELS]")
for sym in ("XAUUSD",):
    p = Path("D:/AURION_DATA/models") / f"model_{sym}.pkl"
    if p.exists():
        size_kb = p.stat().st_size / 1024
        print(f"  model_{sym}.pkl  exists ({size_kb:.1f} KB)")
    else:
        print(f"  model_{sym}.pkl  MISSING")

print("\n[OK] Preflight pasado.")

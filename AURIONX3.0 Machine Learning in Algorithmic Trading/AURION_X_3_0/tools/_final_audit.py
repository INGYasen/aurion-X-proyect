"""Auditoría final tras mejoras: dataset, Knowledge Engine, modelo, bot LIVE."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from analytics.db_connection import init_db, new_session

init_db()
s = new_session()
try:
    print("\n" + "=" * 76)
    print(" AUDITORÍA FINAL — Mejoras aplicadas")
    print("=" * 76)

    # === DATASET ===
    print("\n[1] DATASET ENTRENABLE")
    total = s.execute(text("SELECT COUNT(*) FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL")).fetchone()[0]
    print(f"  Trades totales (cerrados)             : {total}")
    real = s.execute(text("""
        SELECT COUNT(*) FROM trades
        WHERE symbol='XAUUSD' AND close_time IS NOT NULL
          AND (features_json::jsonb->>'synthetic' IS NULL OR features_json::jsonb->>'synthetic'='false')
    """)).fetchone()[0]
    synth = s.execute(text("""
        SELECT COUNT(*) FROM trades
        WHERE symbol='XAUUSD' AND close_time IS NOT NULL
          AND (features_json::jsonb->>'synthetic')='true'
    """)).fetchone()[0]
    print(f"  Trades reales (bot LIVE)              : {real}")
    print(f"  Trades sintéticos (3a M15)            : {synth}")

    # WR por origen
    r = s.execute(text("""
        SELECT
          CASE WHEN (features_json::jsonb->>'synthetic')='true' THEN 'SINT'
               ELSE 'REAL' END AS origin,
          COUNT(*),
          AVG(r_multiple),
          SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END)::float / COUNT(*) AS wr,
          SUM(profit)
        FROM trades
        WHERE symbol='XAUUSD' AND close_time IS NOT NULL
        GROUP BY origin
    """)).fetchall()
    print(f"  {'origen':10s} {'n':>6s} {'WR%':>7s} {'avgR':>8s} {'pnl_$':>10s}")
    for row in r:
        print(f"  {row[0]:10s} {row[1]:>6d} {row[3]*100:>7.2f} {float(row[2]):>8.3f} {float(row[4]):>10.2f}")

    # === KNOWLEDGE ENGINE ===
    print("\n[2] XAU KNOWLEDGE ENGINE")
    for tbl, label in [
        ("xau_market_memory", "market memory (cada vela)"),
        ("xau_regime_memory", "episodios de régimen"),
        ("xau_session_memory", "días-sesión"),
        ("xau_patterns", "patrones SMC"),
        ("xau_smc_events", "eventos SMC denormalizados"),
        ("xau_model_learning", "registros retrain"),
    ]:
        cnt = s.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()[0]
        print(f"  {label:35s} : {cnt:>8d}")

    # Distribución por régimen
    print("\n  Top regímenes (xau_regime_memory):")
    r = s.execute(text("""
        SELECT regime, COUNT(*) AS n,
               AVG(duration_minutes)::int AS avg_dur_min,
               AVG(avg_atr)::numeric(10,3) AS avg_atr
        FROM xau_regime_memory
        GROUP BY regime ORDER BY n DESC LIMIT 10
    """)).fetchall()
    print(f"  {'régimen':25s} {'n':>6s} {'avgDur(min)':>13s} {'avgATR':>10s}")
    for row in r:
        print(f"  {row[0]:25s} {row[1]:>6d} {row[2]:>13d} {row[3]:>10.3f}")

    # Sesiones top
    print("\n  Top sesiones (xau_session_memory):")
    r = s.execute(text("""
        SELECT session, COUNT(*) AS n,
               AVG(bars_n)::numeric(10,1) AS avg_bars,
               AVG(avg_atr)::numeric(10,3) AS avg_atr
        FROM xau_session_memory
        GROUP BY session ORDER BY n DESC
    """)).fetchall()
    print(f"  {'sesión':25s} {'días':>6s} {'avgBars':>10s} {'avgATR':>10s}")
    for row in r:
        print(f"  {row[0]:25s} {row[1]:>6d} {row[2]:>10.1f} {row[3]:>10.3f}")

    # === MODELO ===
    print("\n[3] MODELO NUEVO")
    import joblib
    from pathlib import Path
    p = Path("D:/AURION_DATA/models/model_XAUUSD.pkl")
    if p.exists():
        info_size = p.stat().st_size / 1024
        info_mod = p.stat().st_mtime
        import datetime as dt
        print(f"  path     : {p}")
        print(f"  size     : {info_size:.1f} KB")
        print(f"  modified : {dt.datetime.fromtimestamp(info_mod)}")
        try:
            art = joblib.load(p)
            if isinstance(art, dict):
                print(f"  keys     : {list(art.keys())}")
                if "model" in art:
                    m = art["model"]
                    print(f"  type     : {type(m).__name__}")
                if "feature_names" in art:
                    print(f"  features : {len(art['feature_names'])}")
                if "metadata" in art:
                    md = art["metadata"]
                    if isinstance(md, dict):
                        for k, v in md.items():
                            if not isinstance(v, (list, dict)):
                                print(f"    {k:25s}: {v}")
        except Exception as exc:
            print(f"  load err : {exc}")

    # === BOT LIVE ===
    print("\n[4] BOT LIVE")
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/api/smc/XAUUSD", timeout=5) as resp:
            j = json.loads(resp.read())
        print(f"  SMC endpoint OK")
        print(f"    macro_bias       : {j.get('macro_bias')}")
        print(f"    institutional    : {j.get('institutional_bias')}")
        conf = j.get('confluence', {})
        print(f"    confluence_score : {conf.get('confluence_score')}")
        print(f"    primary TF       : {j.get('primary_timeframe')}")
    except Exception as exc:
        print(f"  SMC endpoint err: {exc}")

    print("\n" + "=" * 76)
finally:
    s.close()

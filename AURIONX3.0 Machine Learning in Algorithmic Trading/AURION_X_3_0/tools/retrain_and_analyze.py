"""AURION X 4.1 — Retraining + análisis completo XAUUSD.

Lanza un walk-forward retraining usando TODOS los trades XAUUSD acumulados,
con muestreo time-weighted 50/30/20 (último mes / últimos 3 meses / resto)
y reporta métricas detalladas + comparativa con el modelo previo.
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from analytics.db_connection import init_db, new_session
from analytics.trade_repository import TradeRepository

MODEL_DIR = Path("D:/AURION_DATA/models")
MODEL_XAU = MODEL_DIR / "model_XAUUSD.pkl"
MODEL_GLOBAL = MODEL_DIR / "model.pkl"
BACKUP_DIR = MODEL_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. ANÁLISIS PRE-RETRAIN: estado del dataset
# --------------------------------------------------------------------------- #
def analyze_dataset() -> dict:
    init_db()
    s = new_session()
    out: dict = {}
    try:
        # Total trades XAUUSD
        n_total = s.execute(text(
            "SELECT count(*) FROM trades WHERE symbol='XAUUSD'"
        )).scalar() or 0
        n_closed = s.execute(text(
            "SELECT count(*) FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL"
        )).scalar() or 0
        out["n_total"] = n_total
        out["n_closed"] = n_closed

        # Distribución temporal por bucket (último mes / últimos 3 meses / resto)
        now = datetime.now(timezone.utc)
        for label, days in [("últimos 30d", 30), ("últimos 90d", 90), ("últimos 365d", 365)]:
            since = now - timedelta(days=days)
            n = s.execute(text(
                "SELECT count(*) FROM trades WHERE symbol='XAUUSD' "
                "AND close_time IS NOT NULL AND close_time > :since"
            ), {"since": since}).scalar() or 0
            out[f"n_{label}"] = n

        # Winrate y avgR global
        row = s.execute(text(
            "SELECT count(*), "
            "       sum(case when profit > 0 then 1 else 0 end), "
            "       avg(r_multiple), "
            "       avg(profit) "
            "FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL"
        )).fetchone()
        if row and row[0]:
            n, wins, avgR, avgP = row
            out["winrate"] = (wins / n) if n else 0.0
            out["avg_r"] = float(avgR or 0.0)
            out["avg_pnl"] = float(avgP or 0.0)

        # Distribución por setup
        rows = s.execute(text(
            "SELECT features_json->>'setup_type' AS s, count(*), "
            "       avg(r_multiple), sum(case when profit > 0 then 1 else 0 end)::float / count(*) AS wr "
            "FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL "
            "GROUP BY features_json->>'setup_type' ORDER BY count(*) DESC LIMIT 12"
        )).fetchall()
        out["setups"] = [
            {"setup": r[0], "n": int(r[1]), "avg_r": float(r[2] or 0.0), "winrate": float(r[3] or 0.0)}
            for r in rows
        ]

        # SMC features presentes (cast json -> jsonb para usar operador ?)
        try:
            with_smc = s.execute(text(
                "SELECT count(*) FROM trades WHERE symbol='XAUUSD' AND close_time IS NOT NULL "
                "AND (features_json::jsonb) ? 'smc_delta_conf'"
            )).scalar() or 0
            out["with_smc_features"] = with_smc
        except Exception:
            s.rollback()
            out["with_smc_features"] = -1

        # SMC tables
        for tbl in ("smc_bos", "smc_choch", "smc_order_blocks", "smc_fvg", "smc_liquidity", "smc_confluence"):
            try:
                c = s.execute(text(f"SELECT count(*) FROM {tbl}")).scalar() or 0
                out[f"smc_{tbl.replace('smc_', '')}"] = int(c)
            except Exception:
                s.rollback()

    finally:
        s.close()
    return out


# --------------------------------------------------------------------------- #
# 2. BACKUP del modelo actual
# --------------------------------------------------------------------------- #
def backup_existing() -> dict:
    info = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for src, label in [(MODEL_XAU, "model_XAUUSD"), (MODEL_GLOBAL, "model_global")]:
        if src.exists():
            dst = BACKUP_DIR / f"{label}_{ts}.pkl.bak"
            shutil.copy2(src, dst)
            info[label] = {"path": str(dst), "size_kb": round(dst.stat().st_size / 1024, 1)}
        else:
            info[label] = None
    return info


# --------------------------------------------------------------------------- #
# 3. RETRAIN walk-forward (con time-weighted sampling 50/30/20)
# --------------------------------------------------------------------------- #
def run_retrain():
    print("[RETRAIN] importando Trainer...")
    from ml.trainer import Trainer

    # IMPORTANTE: feature_schema="v2" (25 features) — debe coincidir con el
    # pipeline live (QuantMLEngine). Schema v1 (10) provoca:
    #   "Predictor: len(features)=25 != modelo n_features=10; salida neutral"
    # y el bot opera con ML en NEUTRO (sin señal predictiva real).
    # AURION X 4.3: use_ensemble=True activa RF+LR+XGB+LGBM voting soft (mejor
    # generalizacion vs RF solo). Si la mejora OOS no es significativa, dejar
    # use_ensemble=False para volver al modelo simple.
    use_ensemble = os.environ.get("AURION_USE_ENSEMBLE", "1").strip() not in ("0", "false", "no", "off")
    schema = os.environ.get("AURION_FEATURE_SCHEMA", "v2").strip().lower()
    if schema not in ("v1", "v2", "v3"):
        schema = "v2"
    n_features = {"v1": 10, "v2": 25, "v3": 30}.get(schema, 25)
    # AURION X 4.4 (10-jun-2026): leer calibration_method del yaml en vez de
    # default sigmoid hardcoded. Audit mostro overconfidence del ensemble que
    # solo isotonic corrige (no-monotonicidad de probabilidades).
    cal_method = "isotonic"
    cal_holdout = 0.18
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "parameters.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as fp:
                cfg = yaml.safe_load(fp) or {}
            ml_cfg = cfg.get("ml", {}) or {}
            cal_method = str(ml_cfg.get("calibration_method", cal_method) or cal_method)
            cal_holdout = float(ml_cfg.get("calibration_holdout_ratio", cal_holdout))
    except Exception as exc:
        print(f"[RETRAIN] warn: no se pudo leer parameters.yaml ({exc}); usando defaults")
    print(
        f"[RETRAIN] instanciando trainer con schema {schema} ({n_features} features) "
        f"+ calibration={cal_method} + ensemble={use_ensemble}..."
    )
    trainer = Trainer(
        feature_schema=schema,
        output_dir=MODEL_DIR,
        use_ensemble=use_ensemble,
        calibration_method=cal_method,
        calibration_holdout_ratio=cal_holdout,
    )

    print("[RETRAIN] dataset XAUUSD (limit=5000)...")
    t0 = time.time()
    try:
        model, metrics = trainer.train_walk_forward_from_repository(
            limit=5000,
            symbol="XAUUSD",
            model_filename="model_XAUUSD.pkl",
            dataset_context="all",
            train_ratio=0.65,
            val_ratio=0.15,
        )
        elapsed = time.time() - t0
        return {"ok": True, "metrics": dict(metrics), "elapsed_s": round(elapsed, 1)}
    except Exception as exc:
        elapsed = time.time() - t0
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "elapsed_s": round(elapsed, 1)}


# --------------------------------------------------------------------------- #
# 4. Análisis del modelo entrenado
# --------------------------------------------------------------------------- #
def analyze_model() -> dict:
    info = {}
    if not MODEL_XAU.exists():
        return {"exists": False}
    info["exists"] = True
    info["path"] = str(MODEL_XAU)
    info["size_kb"] = round(MODEL_XAU.stat().st_size / 1024, 1)
    info["modified"] = datetime.fromtimestamp(MODEL_XAU.stat().st_mtime).isoformat()
    try:
        import joblib
        artifact = joblib.load(MODEL_XAU)
        if isinstance(artifact, dict):
            info["keys"] = list(artifact.keys())
            if "feature_names" in artifact:
                info["n_features"] = len(artifact["feature_names"])
                info["features_sample"] = artifact["feature_names"][:8]
            if "metadata" in artifact:
                info["metadata"] = artifact["metadata"]
            if "model" in artifact:
                m = artifact["model"]
                info["model_type"] = type(m).__name__
                if hasattr(m, "feature_importances_"):
                    fi = m.feature_importances_
                    if "feature_names" in artifact and len(fi) == len(artifact["feature_names"]):
                        pairs = sorted(zip(artifact["feature_names"], fi), key=lambda x: -float(x[1]))
                        info["top_features"] = [
                            {"name": n, "importance": round(float(v), 4)}
                            for n, v in pairs[:15]
                        ]
        else:
            info["model_type"] = type(artifact).__name__
    except Exception as exc:
        info["load_error"] = str(exc)
    return info


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def main() -> int:
    print("\n" + "=" * 70)
    print(" AURION X 4.1 — RETRAIN + ANÁLISIS COMPLETO XAUUSD")
    print("=" * 70)

    # 1. Análisis pre
    print("\n[1/5] Analizando dataset actual...")
    pre = analyze_dataset()
    print(f"  Trades XAUUSD total       : {pre.get('n_total', 0)}")
    print(f"  Trades XAUUSD cerrados    : {pre.get('n_closed', 0)}")
    print(f"  Últimos 30 días           : {pre.get('n_últimos 30d', 0)}")
    print(f"  Últimos 90 días           : {pre.get('n_últimos 90d', 0)}")
    print(f"  Últimos 365 días          : {pre.get('n_últimos 365d', 0)}")
    if pre.get("winrate") is not None:
        print(f"  Winrate global            : {fmt_pct(pre['winrate'])}")
        print(f"  Avg R                     : {pre['avg_r']:.3f}")
        print(f"  Avg PnL                   : ${pre['avg_pnl']:.2f}")
    print(f"  Trades con features SMC   : {pre.get('with_smc_features', 0)}")

    if pre.get("setups"):
        print(f"\n  Setups XAUUSD top:")
        for s in pre["setups"][:8]:
            name = s.get('setup') or '(sin setup_type)'
            try:
                print(f"    {name[:30]:30s} n={s['n']:4d} "
                      f"wr={fmt_pct(s['winrate']):8s} avgR={s['avg_r']:.2f}")
            except Exception:
                print(f"    {name} n={s['n']} wr={s['winrate']} avgR={s['avg_r']}")

    print(f"\n  Universo SMC en PostgreSQL:")
    for k in ("bos", "choch", "order_blocks", "fvg", "liquidity", "confluence"):
        v = pre.get(f"smc_{k}", 0)
        print(f"    {k:20s} {v:>9d} eventos")

    # 2. Backup
    print("\n[2/5] Backup del modelo actual...")
    bk = backup_existing()
    for k, v in bk.items():
        if v:
            print(f"  {k:20s} -> {v['path']} ({v['size_kb']} KB)")
        else:
            print(f"  {k:20s} (no existía)")

    # 3. Retrain
    print("\n[3/5] Lanzando walk-forward retraining...")
    print("  Configuración: train 65% / val 15% / test 20% OOS temporal")
    print("  Embargo: 2% del dataset entre train y val (anti-leakage)")
    print("  Calibración: lee de parameters.yaml > ml.calibration_method")
    print("  Pesos: 1+|R| cap 3 (énfasis en trades con R extremo)")
    res = run_retrain()

    if not res["ok"]:
        print(f"\n  ❌ FAIL: {res.get('error_type')}: {res.get('error')}")
        print(f"  (elapsed: {res.get('elapsed_s')} s)")
        return 1

    print(f"  ✅ OK ({res['elapsed_s']} s)")
    print(f"\n  Métricas resultantes:")
    m = res["metrics"]
    for k in sorted(m.keys()):
        v = m[k]
        if isinstance(v, float):
            print(f"    {k:30s} {v:.4f}")
        else:
            print(f"    {k:30s} {v}")

    # 4. Análisis del modelo
    print("\n[4/5] Analizando modelo entrenado...")
    info = analyze_model()
    if not info.get("exists"):
        print("  ⚠️ model_XAUUSD.pkl no encontrado tras retrain")
    else:
        print(f"  Path       : {info['path']}")
        print(f"  Size       : {info['size_kb']} KB")
        print(f"  Modified   : {info['modified']}")
        print(f"  Type       : {info.get('model_type', 'n/a')}")
        if info.get("n_features"):
            print(f"  Features   : {info['n_features']}")
            print(f"  Sample     : {info.get('features_sample', [])}")
        if info.get("top_features"):
            print(f"\n  Top features por importancia:")
            for f in info["top_features"]:
                bar = "█" * int(f["importance"] * 50)
                print(f"    {f['name']:30s} {f['importance']:.4f}  {bar}")

    # 5. Veredicto y guía siguiente
    print("\n[5/5] Veredicto:")
    oos_wr = m.get("oos_winrate", m.get("winrate", 0))
    oos_sh = m.get("oos_sharpe", m.get("sharpe", 0))
    oos_ex = m.get("oos_expectancy", m.get("expectancy", 0))

    print(f"\n  OOS winrate      : {fmt_pct(oos_wr)}")
    print(f"  OOS Sharpe       : {oos_sh:.3f}")
    print(f"  OOS Expectancy   : {oos_ex:.4f}")

    verdict_lines = []
    if oos_wr >= 0.55:
        verdict_lines.append("  ✅ Winrate OOS >= 55% — modelo apto para producción")
    elif oos_wr >= 0.50:
        verdict_lines.append("  🟡 Winrate OOS 50-55% — aceptable, vigilar deriva")
    else:
        verdict_lines.append("  🔴 Winrate OOS < 50% — modelo conservador; se beneficia de más data")

    if oos_sh >= 0.5:
        verdict_lines.append("  ✅ Sharpe OOS >= 0.5 — relación riesgo/recompensa sólida")
    elif oos_sh >= 0.0:
        verdict_lines.append("  🟡 Sharpe OOS positivo pero bajo — necesita más historia para estabilizar")
    else:
        verdict_lines.append("  🔴 Sharpe OOS negativo — el bot operativo debe filtrar más con SMC + Knowledge")

    for line in verdict_lines:
        print(line)

    # Save report
    report_path = MODEL_DIR / "last_retrain_report.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset_pre": pre,
        "backup": bk,
        "retrain": res,
        "model_info": info,
    }
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n  Reporte completo guardado en: {report_path}")
    except Exception as exc:
        print(f"\n  (no se pudo guardar reporte JSON: {exc})")

    print("\n" + "=" * 70)
    print(" RETRAIN COMPLETO. El bot recargará el modelo en el próximo retrain cycle")
    print(" (auto cada 5 trades cerrados, o reinicia el bot para usarlo inmediatamente).")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

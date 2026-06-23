"""AURION X 4.0 — Entrenamiento offline del PPO Exit Agent para XAUUSD.

Pipeline:
    1. Carga últimos N trades cerrados de PostgreSQL.
    2. Carga M1 sincronizado del mismo periodo.
    3. Construye ``ExitManagementEnv`` con esos trades.
    4. Entrena PPO con 1M timesteps (n_steps=2048, batch_size=64, n_epochs=10).
    5. Evalúa cada 50k steps en validación temporal (10% final).
    6. Métrica de éxito: % R≈0 baja del 88% al ≤30% Y win-rate ≥55%.
    7. Si pasa, guarda ZIP en D:/AURION_DATA/models/ppo_exit_xauusd_v4.zip.

Uso:
    python -m scripts.train_drl_exit_xauusd --total-timesteps 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _ensure_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena PPO Exit Agent XAUUSD.")
    p.add_argument("--config", type=Path, default=Path("AURION_X_3_0/config/parameters.yaml"))
    p.add_argument("--symbol", type=str, default="XAUUSD")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--min-trades", type=int, default=300)
    p.add_argument("--months", type=int, default=6, help="Meses históricos M1 a cargar.")
    p.add_argument("--eval-interval", type=int, default=50_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_trades_and_bars(
    symbol: str, min_trades: int, months: int,
) -> tuple[list[Any], np.ndarray]:
    """Carga trades cerrados + barras M1 alineadas (Postgres + MT5).

    Args:
        symbol: símbolo (XAUUSD).
        min_trades: mínimo de trades cerrados requeridos tras alinear con M1.
        months: meses de histórico M1 a descargar de MT5.
    """
    from datetime import timezone as _tz
    from analytics.trade_repository import TradeRepository
    from analytics.m1_loader import load_m1_bars
    from ml.envs.exit_env import TradeRecord

    repo = TradeRepository()
    print(f"  Cargando trades cerrados desde Postgres (last 5000)...")
    rows_all = repo.get_last_trades(limit=5000)
    rows = [r for r in rows_all if r.get("symbol") == symbol.upper() and r.get("close_time")]
    if len(rows) < min_trades:
        raise SystemExit(
            f"Insuficientes trades cerrados {symbol}: {len(rows)} < {min_trades}. "
            "Deja correr el bot más tiempo o reduce --min-trades."
        )
    print(f"  Trades {symbol} cerrados encontrados: {len(rows)}")

    print(f"  Cargando M1 {symbol} ({months} meses) desde MT5 / cache parquet...")
    bars = load_m1_bars(symbol, months=months)
    print(f"  Bars M1: {bars.shape[0]:,}")

    # Para alinear trades con barras necesitamos timestamps de las barras.
    # m1_loader devuelve (T, 5) sin tiempo. Lo recuperamos del último download
    # de MT5 vía un pase auxiliar (rápido, vuelve del cache).
    # Estrategia: como las barras vienen ordenadas y son M1, podemos construir
    # un eje temporal sintético a partir del rango total disponible.
    # Solución más robusta: pedir timestamps al loader.
    times_idx = _build_times_index(symbol, months, bars.shape[0])

    trades: list[TradeRecord] = []
    for r in rows:
        ot = r.get("open_time")
        ct = r.get("close_time")
        if not ot or not ct:
            continue
        ot_utc = ot.astimezone(_tz.utc) if hasattr(ot, "astimezone") else None
        ct_utc = ct.astimezone(_tz.utc) if hasattr(ct, "astimezone") else None
        if ot_utc is None or ct_utc is None:
            continue
        ot_idx = _nearest_idx(times_idx, ot_utc)
        ct_idx = _nearest_idx(times_idx, ct_utc)
        if ot_idx is None or ct_idx is None or ct_idx <= ot_idx:
            continue
        feats = r.get("features_json") or {}
        trades.append(TradeRecord(
            direction=str(r.get("direction", "BUY")).upper(),
            entry_price=float(r.get("entry_price", 0.0)),
            sl_price=float(r.get("sl", 0.0) or 0.0),
            tp_price=float(r.get("tp", 0.0) or 0.0),
            open_time_idx=int(ot_idx),
            close_time_idx=int(ct_idx),
            setup_grade=str(feats.get("setup_grade", "B")).upper() if feats else "B",
            ml_confidence_at_open=float(r.get("ml_confidence") or feats.get("ml_confidence", 0.5)),
            vol_regime=str(feats.get("volatility_regime", "HIGH")).upper() if feats else "HIGH",
            hour_utc=int(ot_utc.hour),
            drawdown_at_open_pct=float(feats.get("drawdown_pct", 0.0)) if feats else 0.0,
            loss_streak_at_open=int(feats.get("loss_streak", 0)) if feats else 0,
            distance_to_sr_pct=float(feats.get("distance_to_sr_pct", 0.0)) if feats else 0.0,
        ))

    if len(trades) < min_trades:
        raise SystemExit(
            f"Trades alineados con M1: {len(trades)} < {min_trades}. "
            f"Posiblemente el rango de barras ({months} meses) no cubre la fecha de los trades."
        )
    return trades, bars


def _build_times_index(symbol: str, months: int, expected_len: int) -> list[Any]:
    """Reconstruye los timestamps de cada barra M1 cacheada.

    Re-descarga rápida desde MT5 (o cache parquet con time column si existe)
    para tener un eje temporal alineado al array de barras.
    """
    from datetime import datetime, timedelta, timezone as _tz2
    import MetaTrader5 as mt5  # type: ignore[import-not-found]
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() falló al reconstruir timestamps: {mt5.last_error()}")
    try:
        utc_end = datetime.now(_tz2.utc)
        utc_start = utc_end - timedelta(days=months * 31)
        cursor = utc_end
        chunks: list[Any] = []
        while cursor > utc_start:
            chunk_from = max(utc_start, cursor - timedelta(days=30))
            r = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, chunk_from, cursor)
            if r is not None and len(r) > 0:
                chunks.append(r)
            cursor = chunk_from
        if not chunks:
            raise SystemExit("No se pudo reconstruir índice temporal de M1.")
        import numpy as _np
        full = _np.concatenate(chunks[::-1], axis=0)
        # dedup por time
        _, idx = _np.unique(full["time"], return_index=True)
        full = full[_np.sort(idx)]
        times = [datetime.fromtimestamp(int(t), tz=_tz2.utc) for t in full["time"]]
        if abs(len(times) - expected_len) > expected_len * 0.05:
            # Permite hasta 5% de discrepancia entre dedup variantes
            print(f"  [WARN] timestamps={len(times)} vs bars={expected_len} (discrepancia tolerable)")
        return times[:expected_len]
    finally:
        mt5.shutdown()


def _nearest_idx(times: list[Any], target: Any) -> int | None:
    """Busca el índice de la barra cuyo timestamp es más cercano a ``target``."""
    if not times:
        return None
    if target < times[0] or target > times[-1]:
        return None
    # Búsqueda binaria
    lo, hi = 0, len(times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main() -> int:
    args = parse_args()
    _ensure_paths()

    cfg_root = load_yaml(args.config)
    v4 = (cfg_root.get("v4_hybrid") or {}).get("drl_exit") or {}
    model_path = Path(str(v4.get("model_path", "D:/AURION_DATA/models/ppo_exit_xauusd_v4.zip")))
    reward_cfg = v4.get("reward") or {}

    from ml.drl_exit_agent import DRLExitAgent, is_available as _drl_ok
    from ml.envs.exit_env import ExitManagementEnv

    if not _drl_ok():
        raise SystemExit("stable-baselines3 + torch no instalados. pip install -e .[v4]")

    print(f"Cargando trades y barras M1 para {args.symbol}...")
    trades, bars = load_trades_and_bars(args.symbol, args.min_trades, args.months)
    print(f"Trades alineados: {len(trades)}  Bars M1: {bars.shape[0]:,}")

    # Split temporal: 90% train / 10% val
    split = int(len(trades) * 0.9)
    train_trades = trades[:split]
    val_trades = trades[split:]
    print(f"Train trades: {len(train_trades)}  Val trades: {len(val_trades)}")

    train_env = ExitManagementEnv(
        trades=train_trades,
        m1_bars=bars,
        reward_action_penalty=float(reward_cfg.get("action_penalty", 0.02)),
        reward_action_lambda=float(reward_cfg.get("action_lambda", 0.10)),
        reward_drawdown_mu=float(reward_cfg.get("drawdown_mu", 0.50)),
    )

    agent = DRLExitAgent(model_path=model_path, device=args.device)
    agent.build(train_env)

    history: list[dict[str, float]] = []
    timesteps_done = 0
    steps_per_eval = int(args.eval_interval)
    iterations = max(1, int(args.total_timesteps // steps_per_eval))

    for it in range(iterations):
        agent.learn(total_timesteps=steps_per_eval)
        timesteps_done += steps_per_eval

        val_env = ExitManagementEnv(
            trades=val_trades,
            m1_bars=bars,
            reward_action_penalty=float(reward_cfg.get("action_penalty", 0.02)),
            reward_action_lambda=float(reward_cfg.get("action_lambda", 0.10)),
            reward_drawdown_mu=float(reward_cfg.get("drawdown_mu", 0.50)),
        )
        wins = 0
        zeros = 0
        losses = 0
        ep_rewards = []
        n_eval_episodes = min(200, len(val_trades))
        for _ in range(n_eval_episodes):
            obs, _ = val_env.reset()
            done = False
            total_r = 0.0
            final_r = 0.0
            while not done:
                action = agent.predict_action(obs)
                obs, reward, term, trunc, info = val_env.step(action)
                total_r += float(reward)
                if "realized_r" in info:
                    final_r = float(info["realized_r"])
                done = term or trunc
            ep_rewards.append(total_r)
            if final_r > 0.05:
                wins += 1
            elif final_r < -0.05:
                losses += 1
            else:
                zeros += 1

        winrate = wins / max(1, n_eval_episodes)
        zero_rate = zeros / max(1, n_eval_episodes)
        avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        metrics = {
            "iter": float(it + 1),
            "timesteps_done": float(timesteps_done),
            "win_rate": winrate,
            "zero_rate": zero_rate,
            "loss_rate": losses / max(1, n_eval_episodes),
            "avg_reward": avg_reward,
        }
        history.append(metrics)
        print(f"[eval {it+1}/{iterations}] ts={timesteps_done:,}  "
              f"win={winrate:.3f}  zero={zero_rate:.3f}  avg_r={avg_reward:.4f}")

    final = history[-1] if history else {}
    success = (
        final.get("zero_rate", 1.0) <= 0.30
        and final.get("win_rate", 0.0) >= 0.55
    )

    if success and not args.dry_run:
        agent.save(model_path)
        print(f"[OK] Modelo promovido: {model_path}")
    else:
        print(f"[INFO] Modelo NO promovido (zero_rate={final.get('zero_rate', 1):.3f}, "
              f"win_rate={final.get('win_rate', 0):.3f}).")

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "total_timesteps": args.total_timesteps,
        "n_trades_train": len(train_trades),
        "n_trades_val": len(val_trades),
        "history": history,
        "promoted": bool(success) and not args.dry_run,
    }
    out = Path("D:/AURION_DATA/reports/phase3_drl_exit_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Reporte: {out}")
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())

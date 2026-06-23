"""AURION X 4.0 — Tests para DRLExitAgent (PPO wrapper)."""

from __future__ import annotations

import numpy as np
import pytest


def _has_sb3() -> bool:
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_sb3(), reason="stable-baselines3+torch no instalados")


def _build_bars(n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = 2000.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    high = base + rng.uniform(0.5, 1.5, size=n)
    low = base - rng.uniform(0.5, 1.5, size=n)
    vol = rng.uniform(100, 1000, size=n)
    return np.column_stack([base, high, low, vol])


def _make_env():
    from ml.envs.exit_env import ExitManagementEnv, TradeRecord
    bars = _build_bars(400)
    trades = [
        TradeRecord(
            direction="BUY", entry_price=2000.0, sl_price=1998.0, tp_price=2003.0,
            open_time_idx=50, close_time_idx=150, setup_grade="A",
            ml_confidence_at_open=0.7, vol_regime="HIGH", hour_utc=14,
            drawdown_at_open_pct=0.0, loss_streak_at_open=0, distance_to_sr_pct=0.5,
        ),
        TradeRecord(
            direction="SELL", entry_price=2005.0, sl_price=2007.0, tp_price=2002.0,
            open_time_idx=180, close_time_idx=280, setup_grade="B",
            ml_confidence_at_open=0.65, vol_regime="HIGH", hour_utc=15,
            drawdown_at_open_pct=0.5, loss_streak_at_open=1, distance_to_sr_pct=0.3,
        ),
    ]
    return ExitManagementEnv(trades=trades, m1_bars=bars)


def test_build_agent_cpu() -> None:
    from ml.drl_exit_agent import DRLExitAgent
    env = _make_env()
    agent = DRLExitAgent(device="cpu")
    agent.build(env, seed=1)
    assert agent.is_loaded


def test_predict_action_in_range() -> None:
    from ml.drl_exit_agent import DRLExitAgent
    from ml.envs.exit_env import STATE_DIM
    env = _make_env()
    agent = DRLExitAgent(device="cpu")
    agent.build(env, seed=1)
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    action = agent.predict_action(obs)
    assert isinstance(action, int)
    assert 0 <= action <= 4


def test_save_and_load(tmp_path) -> None:
    from ml.drl_exit_agent import DRLExitAgent
    from ml.envs.exit_env import STATE_DIM
    env = _make_env()
    agent = DRLExitAgent(device="cpu")
    agent.build(env, seed=1)
    path = tmp_path / "ppo.zip"
    agent.save(path)
    assert path.is_file()

    agent2 = DRLExitAgent(model_path=path, device="cpu")
    ok = agent2.load()
    assert ok
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    a1 = agent.predict_action(obs)
    a2 = agent2.predict_action(obs)
    # Política determinística → misma acción para misma observación
    assert a1 == a2


def test_inference_fast(tmp_path) -> None:
    import time
    from ml.drl_exit_agent import DRLExitAgent
    from ml.envs.exit_env import STATE_DIM
    env = _make_env()
    agent = DRLExitAgent(device="cpu")
    agent.build(env, seed=1)
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    # Warmup
    for _ in range(5):
        agent.predict_action(obs)
    t0 = time.perf_counter()
    for _ in range(100):
        agent.predict_action(obs)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 100.0
    # Objetivo <5ms; permitimos hasta 15ms en CI sin AVX-512.
    assert avg_ms < 15.0, f"Inferencia PPO promedio = {avg_ms:.2f}ms"

"""AURION X 4.0 — Tests para ExitManagementEnv (Gymnasium)."""

from __future__ import annotations

import numpy as np
import pytest


def _has_gym() -> bool:
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_gym(), reason="gymnasium no instalado")


def _build_bars(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 2000.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    high = base + rng.uniform(0.5, 1.5, size=n)
    low = base - rng.uniform(0.5, 1.5, size=n)
    vol = rng.uniform(100, 1000, size=n)
    return np.column_stack([base, high, low, vol])


def _make_trade(open_idx: int, close_idx: int, direction: str = "BUY") -> "ml.envs.exit_env.TradeRecord":
    from ml.envs.exit_env import TradeRecord
    return TradeRecord(
        direction=direction,
        entry_price=2000.0,
        sl_price=1998.0 if direction == "BUY" else 2002.0,
        tp_price=2003.0 if direction == "BUY" else 1997.0,
        open_time_idx=open_idx,
        close_time_idx=close_idx,
        setup_grade="A",
        ml_confidence_at_open=0.7,
        vol_regime="HIGH",
        hour_utc=14,
        drawdown_at_open_pct=0.0,
        loss_streak_at_open=0,
        distance_to_sr_pct=0.5,
    )


def test_reset_returns_correct_shape() -> None:
    from ml.envs.exit_env import ExitManagementEnv, STATE_DIM
    bars = _build_bars(500)
    env = ExitManagementEnv(trades=[_make_trade(50, 150)], m1_bars=bars)
    obs, info = env.reset(seed=0)
    assert obs.shape == (STATE_DIM,)
    assert obs.dtype == np.float32
    assert "trade_idx" in info


def test_step_with_all_actions_valid() -> None:
    from ml.envs.exit_env import ExitManagementEnv, ExitAction
    bars = _build_bars(500)
    env = ExitManagementEnv(trades=[_make_trade(50, 150)], m1_bars=bars)
    for action in ExitAction:
        env.reset(seed=int(action))
        obs, reward, term, trunc, info = env.step(int(action))
        assert np.isfinite(reward)
        assert isinstance(term, bool)


def test_episode_terminates_at_close() -> None:
    from ml.envs.exit_env import ExitManagementEnv, ExitAction
    bars = _build_bars(200)
    env = ExitManagementEnv(trades=[_make_trade(20, 50)], m1_bars=bars)
    env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated and steps < 200:
        _obs, _r, terminated, truncated, _info = env.step(int(ExitAction.HOLD))
        steps += 1
    assert terminated or truncated, "Episodio debe terminar antes de 200 steps."


def test_full_close_terminates_immediately() -> None:
    from ml.envs.exit_env import ExitManagementEnv, ExitAction
    bars = _build_bars(500)
    env = ExitManagementEnv(trades=[_make_trade(50, 300)], m1_bars=bars)
    env.reset(seed=0)
    _obs, reward, term, _trunc, info = env.step(int(ExitAction.FULL_CLOSE))
    assert term is True
    assert "realized_r" in info
    assert np.isfinite(reward)


def test_deterministic_with_seed() -> None:
    from ml.envs.exit_env import ExitManagementEnv, ExitAction
    bars = _build_bars(500)
    trades = [_make_trade(50, 200), _make_trade(60, 220), _make_trade(70, 240)]

    def run_episode(seed: int) -> list[float]:
        env = ExitManagementEnv(trades=trades, m1_bars=bars, seed=seed)
        obs, _ = env.reset(seed=seed)
        rewards: list[float] = []
        done = False
        while not done:
            _obs, r, term, trunc, _info = env.step(int(ExitAction.HOLD))
            rewards.append(float(r))
            done = term or trunc
        return rewards

    a = run_episode(42)
    b = run_episode(42)
    assert a == pytest.approx(b, abs=1e-9), "Mismo seed → mismas recompensas."

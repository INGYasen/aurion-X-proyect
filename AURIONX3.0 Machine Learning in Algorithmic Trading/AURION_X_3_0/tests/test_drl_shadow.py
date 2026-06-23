"""AURION X 4.0 — Tests para DRLExitShadow (modo sombra)."""

from __future__ import annotations

import numpy as np
import pytest


def _has_deps() -> bool:
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_deps(), reason="DRL deps no instaladas")


def test_shadow_disabled_returns_none() -> None:
    from ml.drl_exit_shadow import DRLExitShadow
    shadow = DRLExitShadow(cfg={"enabled": False, "shadow_mode": True})
    out = shadow.observe(trade_ticket="t1", state_vec=np.zeros(18), actual_action="HOLD")
    assert out is None


def test_shadow_without_model_returns_none(tmp_path) -> None:
    from ml.drl_exit_shadow import DRLExitShadow
    shadow = DRLExitShadow(cfg={
        "enabled": True,
        "shadow_mode": True,
        "model_path": str(tmp_path / "no_existe.zip"),
    })
    assert shadow.is_active
    out = shadow.observe(trade_ticket="t1", state_vec=np.zeros(18), actual_action="HOLD")
    # Modelo no existe → debe ser None y no romper
    assert out is None


def test_shadow_observe_invalid_shape_returns_none(tmp_path) -> None:
    """Validamos que un state_vec de tamaño incorrecto no rompa."""
    from ml.drl_exit_shadow import DRLExitShadow
    shadow = DRLExitShadow(cfg={
        "enabled": True,
        "shadow_mode": True,
        "model_path": str(tmp_path / "no_existe.zip"),
    })
    # Sin modelo cargado tampoco haría falta, pero forzamos shape inválido
    out = shadow.observe(trade_ticket="t1", state_vec=np.zeros(5), actual_action="HOLD")
    assert out is None


def test_shadow_records_actions_with_trained_agent(tmp_path) -> None:
    """E2E corto: entrena PPO minúsculo, lo guarda, shadow lo carga y observa."""
    import numpy as np
    from ml.envs.exit_env import ExitManagementEnv, TradeRecord, STATE_DIM
    from ml.drl_exit_agent import DRLExitAgent
    from ml.drl_exit_shadow import DRLExitShadow

    # Mini-env para entrenar 1000 timesteps (rápido)
    rng = np.random.default_rng(0)
    n = 400
    base = 2000.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    bars = np.column_stack([
        base,
        base + rng.uniform(0.5, 1.5, size=n),
        base - rng.uniform(0.5, 1.5, size=n),
        rng.uniform(100, 1000, size=n),
    ])
    trades = [
        TradeRecord(
            direction="BUY", entry_price=2000.0, sl_price=1998.0, tp_price=2003.0,
            open_time_idx=50, close_time_idx=120, setup_grade="A",
            ml_confidence_at_open=0.7, vol_regime="HIGH", hour_utc=14,
            drawdown_at_open_pct=0.0, loss_streak_at_open=0, distance_to_sr_pct=0.5,
        ),
    ]
    env = ExitManagementEnv(trades=trades, m1_bars=bars)
    agent = DRLExitAgent(device="cpu")
    agent.build(env, seed=1)
    agent.learn(total_timesteps=500)
    model_path = tmp_path / "ppo.zip"
    agent.save(model_path)

    shadow = DRLExitShadow(cfg={
        "enabled": True,
        "shadow_mode": True,
        "model_path": str(model_path),
    })
    action = shadow.observe(
        trade_ticket="t1",
        state_vec=np.zeros(STATE_DIM, dtype=np.float32),
        actual_action="HOLD",
    )
    assert action is not None
    assert 0 <= int(action) <= 4
    # Sin BD configurada → debe haber acumulado en memoria
    decisions = shadow.get_in_memory_decisions()
    assert len(decisions) == 1
    assert decisions[0].drl_action == int(action)
    assert decisions[0].actual_action == "HOLD"

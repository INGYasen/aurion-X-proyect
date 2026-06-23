"""AURION X 4.0 — DRL Exit Agent (PPO) para gestión post-entrada.

Wrap sobre ``stable_baselines3.PPO`` que ofrece:
    - ``train`` (offline, sobre ``ExitManagementEnv``).
    - ``predict_action`` (single-step, batch=1, sin gradiente).
    - ``save`` / ``load`` (ZIP nativo de SB3).
    - Modo ``shadow`` (sin actuar, solo registrar).

stable-baselines3 + gymnasium son **opcionales**: si no están instalados,
``is_available()`` devuelve False y el orchestrator omite el agente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analytics.logger import get_logger

logger = get_logger(__name__)


try:
    from stable_baselines3 import PPO  # type: ignore[import-not-found]
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore[import-not-found]
    _HAS_SB3 = True
except ImportError:
    PPO = None  # type: ignore[assignment]
    DummyVecEnv = None  # type: ignore[assignment]
    _HAS_SB3 = False

try:
    import torch  # type: ignore[import-not-found]
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def is_available() -> bool:
    return _HAS_SB3 and _HAS_TORCH


class DRLExitAgent:
    """Agente PPO con política actor-critic compartida [128, 64] tanh."""

    DEFAULT_POLICY_KW = {
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        "activation_fn": "tanh",  # se mapea a torch.nn.Tanh en _resolve_policy
    }

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        """Agente PPO.

        Args:
            model_path: ruta del .zip a cargar/guardar.
            device: "cpu" (default, recomendado para MlpPolicy) | "cuda" | "auto".
                PPO con redes MLP pequeñas no se beneficia de GPU (SB3 lo
                avisa explícitamente); recomendamos CPU salvo experimentos.
        """
        self._model_path = Path(str(model_path)) if model_path else None
        self._device = device
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_policy_kwargs(self) -> dict[str, Any]:
        if not _HAS_TORCH:
            return {}
        import torch.nn as nn  # type: ignore[import-not-found]

        return {
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
            "activation_fn": nn.Tanh,
        }

    def build(self, env: Any, *, seed: int = 42, **ppo_kwargs: Any) -> None:
        """Construye un PPO nuevo (sin pesos pre-entrenados)."""
        if not is_available():
            raise RuntimeError("stable-baselines3 + torch no instalados (grupo [v4]).")
        wrapped = env if hasattr(env, "step_async") else DummyVecEnv([lambda: env])
        policy_kwargs = self._resolve_policy_kwargs()
        defaults = dict(
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed,
            device=self._device,
            policy_kwargs=policy_kwargs,
        )
        defaults.update(ppo_kwargs)
        self._model = PPO("MlpPolicy", wrapped, **defaults)

    def learn(self, *, total_timesteps: int, log_interval: int = 10) -> None:
        if self._model is None:
            raise RuntimeError("Llama build() o load() antes de learn().")
        self._model.learn(total_timesteps=int(total_timesteps), log_interval=log_interval)

    def load(self, path: Path | str | None = None) -> bool:
        if not is_available():
            return False
        target = Path(str(path)) if path else self._model_path
        if target is None or not target.is_file():
            logger.warning("DRLExitAgent: modelo no encontrado en %s", target)
            return False
        try:
            self._model = PPO.load(target, device=self._device)
            logger.info("DRLExitAgent: cargado %s", target)
            return True
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            logger.error("DRLExitAgent: error al cargar %s: %s", target, exc)
            return False

    def save(self, path: Path | str | None = None) -> None:
        if self._model is None:
            raise RuntimeError("DRLExitAgent: nada que guardar.")
        target = Path(str(path)) if path else self._model_path
        if target is None:
            raise ValueError("DRLExitAgent.save() requiere path.")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(target)

    def predict_action(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        deterministic: bool = True,
    ) -> int:
        """Devuelve la acción (int 0..4) para un único estado."""
        if self._model is None:
            raise RuntimeError("DRLExitAgent: modelo no cargado.")
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action[0]) if hasattr(action, "__len__") else int(action)

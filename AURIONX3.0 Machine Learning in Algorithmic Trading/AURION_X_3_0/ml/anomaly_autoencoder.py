"""AURION X 4.0 — Autoencoder denoising para detectar régimen anómalo.

Detecta toxic flow / eventos macro inesperados sobre features de
microestructura ya capturadas en ``market/microstructure_engine.py``:

    [bid_ask_spread, tick_volume_z, cancel_fill_ratio, imbalance_score,
     aggressor_ratio, sweep_score, displacement_speed, liquidity_density]

Arquitectura denoising:
    encoder: Linear(8→32) → ReLU → Linear(32→16) → ReLU → Linear(16→8)
    decoder: Linear(8→16) → ReLU → Linear(16→32) → ReLU → Linear(32→8)

Score de anomalía: reconstrucción L2 normalizada por el percentil 99 del
training set. Si ``score > block_threshold`` (default 0.80), el orchestrator
bloquea nuevas entradas durante ``block_cooldown_s`` segundos.

Referencias:
    Poutré, C., Chételat, D., & Morales, M. (2024). Deep unsupervised anomaly
    detection in HF markets. JFDS 10, 100129. https://doi.org/10.1016/j.jfds.2024.100129
    Choi, K. et al. (2021). Deep learning for anomaly detection in time-series
    data. IEEE Access 9, 120043–120065. https://doi.org/10.1109/ACCESS.2021.3107975
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analytics.logger import get_logger

logger = get_logger(__name__)


try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


def is_available() -> bool:
    return _HAS_TORCH


ANOMALY_FEATURE_NAMES: tuple[str, ...] = (
    "bid_ask_spread_norm",
    "tick_volume_z",
    "cancel_fill_ratio",
    "imbalance_score",
    "aggressor_ratio",
    "sweep_score",
    "displacement_speed",
    "liquidity_density",
)


def build_anomaly_input(micro: Mapping[str, Any], market_state: Mapping[str, Any]) -> np.ndarray:
    """Construye vector 8-D desde ``micro`` y ``market_state`` con saneamiento."""
    spread_pts = float(market_state.get("spread_points") or 0.0)
    spread_cap = float(market_state.get("spread_cap") or 1.0)
    spread_norm = spread_pts / max(spread_cap, 1e-9)

    vols = list(market_state.get("volumes_m1") or [])
    if len(vols) >= 2:
        mu = float(np.mean(vols))
        sd = float(np.std(vols)) or 1.0
        vol_z = (float(vols[-1]) - mu) / sd
    else:
        vol_z = 0.0

    cancel_fill = float(micro.get("intention_loss") or 0.0)  # proxy si broker no expone
    imbalance = float(micro.get("absorption_pressure") or 0.0)
    aggressor = float(micro.get("directional_exhaustion") or 0.0)
    sweep_s = float((market_state.get("signal") or {}).get("sweep_score", 0.0) or 0.0) if isinstance(market_state.get("signal"), dict) else 0.0
    displ = float(micro.get("institutional_velocity") or 0.0)
    liquidity_d = float((market_state.get("range_quality") or {}).get("support_resistance_clarity", 0.0)) if isinstance(market_state.get("range_quality"), dict) else 0.0

    vec = np.array(
        [spread_norm, vol_z, cancel_fill, imbalance, aggressor, sweep_s, displ, liquidity_d],
        dtype=np.float32,
    )
    return np.where(np.isfinite(vec), vec, 0.0).astype(np.float32)


if _HAS_TORCH:

    class AnomalyAutoencoder(nn.Module):
        """Denoising AE 8 → 32 → 16 → 8 → 16 → 32 → 8."""

        def __init__(
            self,
            input_dim: int = 8,
            encoder_dims: Sequence[int] = (32, 16, 8),
            dropout: float = 0.10,
            noise_std: float = 0.02,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.encoder_dims = tuple(int(d) for d in encoder_dims)
            self._dropout = float(dropout)
            self._noise_std = float(noise_std)

            enc_layers: list[Any] = []
            prev = self.input_dim
            for d in self.encoder_dims:
                enc_layers += [nn.Linear(prev, d), nn.ReLU(inplace=False), nn.Dropout(self._dropout)]
                prev = d
            self.encoder = nn.Sequential(*enc_layers)

            dec_layers: list[Any] = []
            for d in reversed(self.encoder_dims[:-1]):
                dec_layers += [nn.Linear(prev, d), nn.ReLU(inplace=False), nn.Dropout(self._dropout)]
                prev = d
            dec_layers += [nn.Linear(prev, self.input_dim)]
            self.decoder = nn.Sequential(*dec_layers)

        def forward(self, x: "torch.Tensor", *, add_noise: bool = False) -> "torch.Tensor":
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if add_noise and self.training and self._noise_std > 0:
                x = x + torch.randn_like(x) * self._noise_std
            z = self.encoder(x)
            return self.decoder(z)

        @torch.no_grad()  # type: ignore[misc]
        def reconstruction_error(self, x: "torch.Tensor") -> "torch.Tensor":
            """L2 normalizado (raíz de MSE por muestra)."""
            self.eval()
            xh = self.forward(x, add_noise=False)
            err = (xh - x).pow(2).mean(dim=-1).sqrt()
            return err

else:

    class AnomalyAutoencoder:  # type: ignore[no-redef]
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            raise RuntimeError("AnomalyAutoencoder requiere PyTorch (grupo [v4]).")


class AnomalyDetector:
    """Wrapper de inferencia con threshold persistido y cooldown."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
    ) -> None:
        c = dict(cfg or {})
        self._enabled = bool(c.get("enabled", False))
        self._model_path = Path(str(c.get("model_path", "")))
        self._threshold_path = Path(str(c.get("threshold_path", "")))
        self._block_threshold = float(c.get("block_threshold", 0.80))
        self._cooldown_s = float(c.get("block_cooldown_s", 90.0))
        self._input_dim = int(c.get("input_features", 8))
        self._encoder_dims = list(c.get("encoder_dims", [32, 16, 8]))
        self._dropout = float(c.get("dropout", 0.10))

        self._model: AnomalyAutoencoder | None = None
        self._p99_error: float = 1.0
        self._loaded = False
        self._auto_disabled = False
        self._cooldown_until_mono = 0.0

    @property
    def is_active(self) -> bool:
        return self._enabled and is_available() and not self._auto_disabled

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._model is not None
        if not self.is_active:
            return False
        if not self._model_path.is_file():
            logger.warning("AnomalyDetector: modelo no encontrado en %s.", self._model_path)
            self._auto_disabled = True
            return False
        try:
            model = AnomalyAutoencoder(
                input_dim=self._input_dim,
                encoder_dims=self._encoder_dims,
                dropout=self._dropout,
            )
            state = torch.load(self._model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            self._model = model
            self._loaded = True

            if self._threshold_path.is_file():
                with self._threshold_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                    self._p99_error = float(payload.get("p99_error", 1.0))
            else:
                logger.warning("AnomalyDetector: threshold no encontrado, usando p99=1.0.")
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("AnomalyDetector load fail: %s", exc)
            self._auto_disabled = True
            return False

    def score(
        self,
        *,
        micro: Mapping[str, Any],
        market_state: Mapping[str, Any],
    ) -> float | None:
        """Devuelve ``anomaly_score ∈ [0, 1+]`` o ``None`` si inactivo."""
        if not self._ensure_loaded() or self._model is None:
            return None
        vec = build_anomaly_input(micro, market_state)
        x = torch.from_numpy(vec).float().unsqueeze(0)
        with torch.no_grad():
            err = float(self._model.reconstruction_error(x).item())
        if self._p99_error <= 1e-9:
            return None
        return err / self._p99_error

    def should_block(self, *, now_mono: float, micro: Mapping[str, Any], market_state: Mapping[str, Any]) -> tuple[bool, str, float]:
        """Devuelve ``(block, reason, score)``.

        Persiste cooldown interno: si una vez bloqueó, los siguientes
        ``cooldown_s`` segundos también bloquean sin re-evaluar.
        """
        if now_mono < self._cooldown_until_mono:
            return True, "anomaly_autoencoder:cooldown", float("nan")
        score = self.score(micro=micro, market_state=market_state)
        if score is None:
            return False, "", float("nan")
        if score >= self._block_threshold:
            self._cooldown_until_mono = now_mono + self._cooldown_s
            return True, "anomaly_autoencoder:regime_break", score
        return False, "", score

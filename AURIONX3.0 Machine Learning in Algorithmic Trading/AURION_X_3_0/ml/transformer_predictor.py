"""AURION X 4.0 — Wrapper de inferencia PatchTST consumible por el orchestrator.

Encapsula:
    - Carga lazy del modelo .pt.
    - Construcción de la ventana de 256 ticks M1 desde ``market_state``.
    - Normalización online (z-score por canal, ventana móvil).
    - Manejo de NaN / latencia (con fallback configurable).

Interfaz pública:
    ``TransformerPredictor.predict(market_state) -> {prob_buy, prob_sell,
    confidence, latency_ms}`` o ``None`` si no aplica.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analytics.logger import get_logger

from .transformer_branch import PatchTSTBranch, is_available as _torch_available

logger = get_logger(__name__)

try:
    import torch  # type: ignore[import-not-found]
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


class TransformerPrediction(dict):
    """Compatible con ``PredictionResult`` de ``predictor.py``."""

    prob_buy: float
    prob_sell: float
    confidence: float
    latency_ms: float


class TransformerPredictor:
    """Inferencia PatchTST con fallback a ``None`` si:
        - PyTorch no instalado.
        - Modelo .pt no carga.
        - Ventana incompleta (< window_len ticks).
        - Cualquier NaN en input.
        - Latencia > ``max_latency_ms`` (auto-desactiva tras 3 strikes).
    """

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
    ) -> None:
        c = dict(cfg or {})
        self._enabled = bool(c.get("enabled", False))
        self._model_path = Path(str(c.get("model_path", ""))) if c.get("model_path") else None
        self._n_channels = int(c.get("n_channels", 5))
        self._patch_len = int(c.get("patch_len", 16))
        self._n_patches = int(c.get("n_patches", 16))
        self._d_model = int(c.get("d_model", 64))
        self._n_heads = int(c.get("n_heads", 4))
        self._n_layers = int(c.get("n_layers", 3))
        self._dropout = float(c.get("dropout", 0.15))
        self._fallback_on_nan = bool(c.get("fallback_on_nan", True))
        self._max_latency_ms = float(c.get("max_latency_ms", 50.0))
        device_pref = str(c.get("device", "auto")).lower()
        self._device = self._resolve_device(device_pref)

        self._model: PatchTSTBranch | None = None
        self._loaded = False
        self._latency_strikes = 0
        self._auto_disabled = False

    @staticmethod
    def _resolve_device(pref: str) -> str:
        if not _HAS_TORCH:
            return "cpu"
        if pref == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if pref == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def window_len(self) -> int:
        return self._patch_len * self._n_patches

    @property
    def is_active(self) -> bool:
        return self._enabled and _torch_available() and not self._auto_disabled

    def _ensure_loaded(self) -> bool:
        if self._loaded or self._auto_disabled:
            return self._loaded
        if not self.is_active:
            return False
        if self._model_path is None or not self._model_path.is_file():
            logger.warning(
                "TransformerPredictor: modelo no encontrado en %s (desactivado).",
                self._model_path,
            )
            self._auto_disabled = True
            return False
        try:
            model = PatchTSTBranch(
                n_channels=self._n_channels,
                patch_len=self._patch_len,
                n_patches=self._n_patches,
                d_model=self._d_model,
                n_heads=self._n_heads,
                n_layers=self._n_layers,
                dropout=self._dropout,
            )
            state = torch.load(self._model_path, map_location=self._device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.to(self._device)
            model.eval()
            self._model = model
            self._loaded = True
            logger.info(
                "TransformerPredictor: cargado %s en %s (%d params).",
                self._model_path.name,
                self._device,
                model.num_params(),
            )
            return True
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            logger.error("TransformerPredictor: error al cargar %s: %s", self._model_path, exc)
            self._auto_disabled = True
            return False

    def _build_window(self, market_state: Mapping[str, Any]) -> np.ndarray | None:
        """Construye matriz ``(window_len, n_channels)`` desde ``market_state``."""
        closes = list(market_state.get("closes_m1") or [])
        highs = list(market_state.get("highs_m1") or closes)
        lows = list(market_state.get("lows_m1") or closes)
        volumes = list(market_state.get("volumes_m1") or [])
        spreads = list(market_state.get("spreads_m1") or [])

        wl = self.window_len
        if len(closes) < wl or len(highs) < wl or len(lows) < wl:
            return None

        closes_a = np.asarray(closes[-wl:], dtype=np.float64)
        highs_a = np.asarray(highs[-wl:], dtype=np.float64)
        lows_a = np.asarray(lows[-wl:], dtype=np.float64)

        if len(volumes) >= wl:
            vol_a = np.asarray(volumes[-wl:], dtype=np.float64)
        else:
            vol_a = np.full(wl, np.nanmean(volumes) if volumes else 0.0, dtype=np.float64)

        if len(spreads) >= wl:
            spr_a = np.asarray(spreads[-wl:], dtype=np.float64)
        else:
            spr = float(market_state.get("spread_points") or 0.0)
            spr_a = np.full(wl, spr, dtype=np.float64)

        mat = np.column_stack([closes_a, highs_a, lows_a, vol_a, spr_a])
        if not np.isfinite(mat).all():
            if self._fallback_on_nan:
                return None
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score por canal (numéricamente estable).
        mu = mat.mean(axis=0, keepdims=True)
        sigma = mat.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        z = (mat - mu) / sigma
        return z.astype(np.float32)

    def _maybe_auto_disable(self, latency_ms: float) -> None:
        if latency_ms > self._max_latency_ms:
            self._latency_strikes += 1
            if self._latency_strikes >= 3:
                logger.warning(
                    "TransformerPredictor auto-desactivado: latencia P95 %.1fms > %.1fms.",
                    latency_ms, self._max_latency_ms,
                )
                self._auto_disabled = True
        else:
            self._latency_strikes = max(0, self._latency_strikes - 1)

    def predict(self, market_state: Mapping[str, Any]) -> TransformerPrediction | None:
        """Inferencia.

        Returns:
            Diccionario con ``prob_buy``, ``prob_sell``, ``confidence`` y
            ``latency_ms``, o ``None`` si la predicción no se pudo realizar.
        """
        if not self.is_active or not self._ensure_loaded() or self._model is None:
            return None

        window = self._build_window(market_state)
        if window is None:
            return None

        t0 = time.perf_counter()
        try:
            x = torch.from_numpy(window).unsqueeze(0).to(self._device)
            with torch.no_grad():
                p_buy_t = self._model(x)
            p_buy = float(p_buy_t.detach().cpu().item())
        except RuntimeError as exc:
            logger.warning("TransformerPredictor.predict: %s", exc)
            return None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not math.isfinite(p_buy):
            return None

        p_buy = max(0.0, min(1.0, p_buy))
        p_sell = 1.0 - p_buy
        confidence = max(p_buy, p_sell)

        self._maybe_auto_disable(latency_ms)

        return TransformerPrediction(
            prob_buy=p_buy,
            prob_sell=p_sell,
            confidence=confidence,
            latency_ms=latency_ms,
        )

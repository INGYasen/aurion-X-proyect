"""AURION X 4.0 — Rama Transformer (PatchTST) para señal direccional intradía.

Implementación minimalista de PatchTST (Nie et al., ICLR 2023) adaptada a
XAUUSD scalper M1.

    Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
    A time series is worth 64 words: Long-term forecasting with Transformers.
    ICLR 2023.

Arquitectura:
    - Input: ventana de 256 ticks M1 con 5 canales
      (close, high, low, volume_proxy, spread).
    - Patches de 16 ticks → 16 patches (cubre 4 horas de M1).
    - Embedding lineal + positional embedding aprendido.
    - 3 TransformerEncoderLayer (d_model=64, n_heads=4, GELU, batch_first).
    - Head: LayerNorm → Linear(d_model, 1) → Sigmoid.
    - Output: prob_buy ∈ [0, 1].

PyTorch es **opcional**. Si no está instalado, este módulo permanece importable
pero ``is_available()`` devuelve ``False`` y ningún caller intentará usarlo.
"""

from __future__ import annotations

from typing import Any

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
    """``True`` si PyTorch está instalado (grupo opcional ``[v4]``)."""
    return _HAS_TORCH


if _HAS_TORCH:

    class PatchTSTBranch(nn.Module):
        """PatchTST minimalista para predicción direccional XAUUSD M1.

        Args:
            n_channels: número de canales por tick (default 5).
            patch_len: longitud de cada patch (default 16).
            n_patches: cantidad de patches por ventana (default 16 → ventana 256).
            d_model: dimensión latente (default 64).
            n_heads: cabezas de atención (default 4).
            n_layers: capas del encoder (default 3).
            dropout: dropout en attention + FF (default 0.15).
            activation: activación ``"gelu"`` (default) o ``"relu"``.
        """

        def __init__(
            self,
            n_channels: int = 5,
            patch_len: int = 16,
            n_patches: int = 16,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 3,
            dropout: float = 0.15,
            activation: str = "gelu",
        ) -> None:
            super().__init__()
            self.n_channels = int(n_channels)
            self.patch_len = int(patch_len)
            self.n_patches = int(n_patches)
            self.d_model = int(d_model)

            self.proj = nn.Linear(self.patch_len * self.n_channels, self.d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, self.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=int(n_heads),
                dim_feedforward=self.d_model * 4,
                dropout=float(dropout),
                batch_first=True,
                activation=activation,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
            self.head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, 1),
                nn.Sigmoid(),
            )

        @property
        def window_len(self) -> int:
            """Longitud total de la ventana de entrada (patches × patch_len)."""
            return self.patch_len * self.n_patches

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward.

            Args:
                x: tensor ``(B, T, C)`` con ``T = patch_len * n_patches``.

            Returns:
                Tensor ``(B,)`` con probabilidades de ``buy``.
            """
            if x.dim() != 3:
                raise ValueError(f"x debe ser 3D (B,T,C); recibido {tuple(x.shape)}")
            B, T, C = x.shape
            expected_T = self.window_len
            if T != expected_T:
                raise ValueError(f"T={T} != window_len={expected_T}")
            if C != self.n_channels:
                raise ValueError(f"C={C} != n_channels={self.n_channels}")

            x = x.reshape(B, self.n_patches, self.patch_len * self.n_channels)
            x = self.proj(x) + self.pos_emb
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.head(x).squeeze(-1)

        @torch.no_grad()  # type: ignore[misc]
        def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
            """Inferencia con gradientes desactivados."""
            self.eval()
            return self.forward(x)

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:

    class PatchTSTBranch:  # type: ignore[no-redef]
        """Stub cuando PyTorch no está instalado.

        Cualquier instanciación lanza ``RuntimeError`` con un mensaje claro;
        los callers deben consultar ``is_available()`` antes de construir.
        """

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PatchTSTBranch requiere PyTorch. Instala dependencias v4 con:\n"
                "  pip install -e .[v4] --index-url https://download.pytorch.org/whl/cu124"
            )

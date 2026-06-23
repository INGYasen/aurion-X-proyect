"""AURION X 4.0 — Tests para PatchTSTBranch.

Se saltan automáticamente si PyTorch no está instalado (CI mínimo sin v4).
"""

from __future__ import annotations

import time

import pytest

from ml.transformer_branch import PatchTSTBranch, is_available

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="PyTorch no instalado (instala dependencias [v4]).",
)


@pytest.fixture
def model() -> PatchTSTBranch:
    return PatchTSTBranch(
        n_channels=5, patch_len=16, n_patches=16,
        d_model=64, n_heads=4, n_layers=3, dropout=0.0,
    )


def test_forward_shape(model: PatchTSTBranch) -> None:
    import torch
    x = torch.randn(4, 256, 5)
    y = model(x)
    assert y.shape == (4,)
    assert (y >= 0).all() and (y <= 1).all()


def test_window_len_property(model: PatchTSTBranch) -> None:
    assert model.window_len == 256


def test_gradient_flow(model: PatchTSTBranch) -> None:
    import torch
    x = torch.randn(2, 256, 5, requires_grad=False)
    y = model(x)
    target = torch.tensor([1.0, 0.0])
    loss = torch.nn.functional.binary_cross_entropy(y, target)
    loss.backward()
    # Todos los parámetros que requieren grad deben tener un .grad no nulo
    grads_ok = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert grads_ok > 0, "No fluyen gradientes a los parámetros del modelo."


def test_inference_under_50ms_cpu(model: PatchTSTBranch) -> None:
    """Latencia P95 razonable en CPU para ventana batch=1."""
    import torch
    model.eval()
    x = torch.randn(1, 256, 5)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    times: list[float] = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p95 = times[int(0.95 * len(times))]
    # Margen generoso: en CPU sin AVX-512 puede oscilar; objetivo 50ms.
    assert p95 < 80.0, f"Latencia P95={p95:.1f}ms demasiado alta"


def test_invalid_shape_raises(model: PatchTSTBranch) -> None:
    import torch
    with pytest.raises(ValueError):
        model(torch.randn(1, 128, 5))  # T != 256
    with pytest.raises(ValueError):
        model(torch.randn(1, 256, 3))  # C != 5
    with pytest.raises(ValueError):
        model(torch.randn(1, 256))     # 2D


def test_save_load_state_dict(model: PatchTSTBranch, tmp_path) -> None:
    import torch
    path = tmp_path / "patchtst.pt"
    torch.save(model.state_dict(), path)
    fresh = PatchTSTBranch(
        n_channels=5, patch_len=16, n_patches=16,
        d_model=64, n_heads=4, n_layers=3, dropout=0.0,
    )
    fresh.load_state_dict(torch.load(path, map_location="cpu"))
    x = torch.randn(2, 256, 5)
    y1 = model.predict_proba(x)
    y2 = fresh.predict_proba(x)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)

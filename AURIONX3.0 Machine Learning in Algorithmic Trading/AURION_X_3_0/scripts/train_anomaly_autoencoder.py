"""AURION X 4.0 — Entrena Anomaly Autoencoder sobre microestructura XAUUSD.

Pipeline:
    1. Carga snapshots de microestructura del último mes (filtrando eventos
       macro conocidos vía calendario o ventana ``ny_turbo``).
    2. Entrena el AE en denoising mode (BCE/MSE + ruido gaussiano).
    3. Computa el percentil 99 del error de reconstrucción → threshold.
    4. Guarda modelo .pt + threshold JSON.

Uso:
    python -m scripts.train_anomaly_autoencoder --days 30
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
    p = argparse.ArgumentParser(description="Entrena Anomaly Autoencoder XAUUSD.")
    p.add_argument("--config", type=Path, default=Path("AURION_X_3_0/config/parameters.yaml"))
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--symbol", type=str, default="XAUUSD")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_paths()

    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch no instalado (pip install -e .[v4])")

    from ml.anomaly_autoencoder import AnomalyAutoencoder, ANOMALY_FEATURE_NAMES
    from analytics.quant_repository import QuantAnalyticsRepository

    cfg_root = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    ae_cfg = (cfg_root.get("v4_hybrid") or {}).get("anomaly_autoencoder") or {}
    model_path = Path(str(ae_cfg.get("model_path", "D:/AURION_DATA/models/anomaly_ae_xauusd_v4.pt")))
    threshold_path = Path(str(ae_cfg.get("threshold_path", "D:/AURION_DATA/models/anomaly_ae_xauusd_v4_threshold.json")))

    print(f"Cargando microestructura {args.symbol} ({args.days} días)...")
    repo = QuantAnalyticsRepository()
    rows = repo.fetch_microstructure_snapshots(symbol=args.symbol, last_days=args.days)  # type: ignore[attr-defined]
    if not rows or len(rows) < 1000:
        raise SystemExit(f"Insuficientes snapshots: {len(rows) if rows else 0}")

    X = np.array(
        [[r.get(c, 0.0) for c in ANOMALY_FEATURE_NAMES] for r in rows],
        dtype=np.float64,  # float64 para evitar pérdida en winsorize
    )
    X = np.where(np.isfinite(X), X, 0.0)
    # Winsorize [1%, 99%] por columna antes de cualquier estadística — protege
    # contra outliers extremos almacenados en BD (divisiones por cero, overflow).
    lo = np.percentile(X, 1, axis=0)
    hi = np.percentile(X, 99, axis=0)
    X = np.clip(X, lo, hi).astype(np.float32)
    print(f"Dataset: X={X.shape}  (winsorized 1/99% por columna)")
    print(f"  min por col: {X.min(axis=0).round(4)}")
    print(f"  max por col: {X.max(axis=0).round(4)}")

    # Normalización robusta: median + MAD (Median Absolute Deviation × 1.4826)
    mu = np.median(X, axis=0).astype(np.float32)
    mad = np.median(np.abs(X - mu), axis=0).astype(np.float32) * 1.4826
    sigma = np.where(mad < 1e-6, 1.0, mad).astype(np.float32)
    Xn = ((X - mu) / sigma).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AnomalyAutoencoder(
        input_dim=len(ANOMALY_FEATURE_NAMES),
        encoder_dims=tuple(ae_cfg.get("encoder_dims", [32, 16, 8])),
        dropout=float(ae_cfg.get("dropout", 0.10)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.MSELoss()

    X_t = torch.from_numpy(Xn).float().to(device)
    n = X_t.shape[0]
    val_n = int(n * 0.10)
    perm = torch.randperm(n, device=device)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    X_tr, X_va = X_t[tr_idx], X_t[val_idx]

    print(f"Training: tr={X_tr.shape[0]}, val={X_va.shape[0]}, device={device}")
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    bs = int(args.batch_size)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        perm_e = torch.randperm(X_tr.shape[0], device=device)
        for start in range(0, X_tr.shape[0], bs):
            idx = perm_e[start : start + bs]
            xb = X_tr[idx]
            opt.zero_grad()
            xh = model(xb, add_noise=True)
            loss = loss_fn(xh, xb)
            loss.backward()
            opt.step()
            running += float(loss.detach().item()) * xb.shape[0]
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_va), X_va).item())
        print(f"epoch {epoch:02d}  train_mse={running/X_tr.shape[0]:.5f}  val_mse={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        errs = model.reconstruction_error(X_t).cpu().numpy()
    p99 = float(np.percentile(errs, 99))
    p999 = float(np.percentile(errs, 99.9))
    print(f"p99 error reconstrucción = {p99:.5f}  (p99.9 = {p999:.5f})")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with threshold_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "p99_error": p99,
                "p99_9_error": p999,
                "norm_mu": mu.tolist(),
                "norm_sigma": sigma.tolist(),
                "winsorize_low": lo.tolist(),
                "winsorize_high": hi.tolist(),
                "feature_names": list(ANOMALY_FEATURE_NAMES),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_samples": int(X.shape[0]),
            },
            fh,
            indent=2,
        )
    print(f"[OK] Modelo guardado: {model_path}")
    print(f"[OK] Threshold:       {threshold_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

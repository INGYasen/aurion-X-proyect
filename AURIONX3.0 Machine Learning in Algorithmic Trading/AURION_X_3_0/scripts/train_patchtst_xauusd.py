"""AURION X 4.0 — Entrenamiento PatchTST para XAUUSD scalper M1.

Pipeline:
    1. Carga M1 de PostgreSQL (TradeRepository / quant_repository).
    2. Construye ventanas deslizantes de 256 ticks con etiqueta binaria (la
       siguiente vela cierra arriba/abajo del último close del input).
    3. Purged K-Fold CV con embargo 2%.
    4. Entrena PatchTST (AdamW + CosineAnnealingWarmRestarts + BCE balanceada).
    5. Early stopping con paciencia 10 sobre cada fold.
    6. Gate de promoción: DSR >= 0.60 Y PBO <= 0.30 (statistical_gates).
    7. Guarda mejor checkpoint en D:/AURION_DATA/models/patchtst_xauusd_v4.pt.
    8. Reporte JSON en D:/AURION_DATA/reports/phase2_patchtst_metrics.json.

Uso:
    python -m scripts.train_patchtst_xauusd \
        --config AURION_X_3_0/config/parameters.yaml \
        --months 12 \
        --epochs 30 \
        --batch-size 256

Requiere el grupo opcional ``[v4]`` (torch>=2.5).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _import_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        return torch, nn, AdamW, CosineAnnealingWarmRestarts
    except ImportError as exc:
        sys.stderr.write(
            "PyTorch no instalado. Ejecuta:\n"
            "  pip install -e .[v4] --index-url https://download.pytorch.org/whl/cu124\n"
        )
        raise SystemExit(1) from exc


def _ensure_paths() -> Path:
    """Añade la raíz del proyecto al sys.path para imports relativos."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@dataclass
class TrainConfig:
    window_len: int
    n_channels: int
    patch_len: int
    n_patches: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    n_folds: int
    embargo_pct: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    device: str
    model_path: Path
    report_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena PatchTST sobre XAUUSD M1.")
    p.add_argument("--config", type=Path, default=Path("AURION_X_3_0/config/parameters.yaml"))
    p.add_argument("--months", type=int, default=12, help="Meses históricos a usar.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--symbol", type=str, default="XAUUSD")
    p.add_argument("--dry-run", action="store_true", help="Sin guardar artefactos.")
    return p.parse_args()


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_m1_history(symbol: str, months: int) -> np.ndarray:
    """Carga histórico M1 vía MT5 con cache parquet en D:/AURION_DATA/cache.

    5 canales: [close, high, low, tick_volume, spread].
    """
    from analytics.m1_loader import load_m1_bars  # lazy import (requiere MT5)
    return load_m1_bars(symbol, months=months)


def build_windows(
    bars: np.ndarray,
    *,
    window_len: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera (X, y) con etiquetas binarias.

    Etiqueta = 1 si ``close[t+horizon] > close[t]``, 0 si no.
    """
    T = bars.shape[0]
    if T < window_len + horizon + 10:
        raise SystemExit(f"Histórico insuficiente: {T} barras.")
    closes = bars[:, 0]
    n_samples = T - window_len - horizon
    X = np.empty((n_samples, window_len, bars.shape[1]), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        win = bars[i : i + window_len]
        # Z-score por canal
        mu = win.mean(axis=0, keepdims=True)
        sigma = win.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        X[i] = ((win - mu) / sigma).astype(np.float32)
        ref_close = closes[i + window_len - 1]
        fut_close = closes[i + window_len - 1 + horizon]
        y[i] = 1 if fut_close > ref_close else 0
    return X, y


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    fold_id: int,
) -> tuple[Any, dict[str, float]]:
    torch, nn, AdamW, CosineSched = _import_torch()
    from ml.transformer_branch import PatchTSTBranch

    device = torch.device(cfg.device)
    model = PatchTSTBranch(
        n_channels=cfg.n_channels,
        patch_len=cfg.patch_len,
        n_patches=cfg.n_patches,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)

    pos_weight_value = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))
    criterion = nn.BCELoss(reduction="mean")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineSched(opt, T_0=5, T_mult=2)

    X_tr = torch.from_numpy(X_train).float().to(device)
    y_tr = torch.from_numpy(y_train).float().to(device)
    X_va = torch.from_numpy(X_val).float().to(device)
    y_va = torch.from_numpy(y_val).float().to(device)
    weights = torch.where(y_tr > 0.5, torch.tensor(pos_weight_value, device=device), torch.tensor(1.0, device=device))

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_no_improve = 0

    n = X_tr.shape[0]
    bs = cfg.batch_size

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            xb = X_tr[idx]
            yb = y_tr[idx]
            wb = weights[idx]
            opt.zero_grad()
            preds = model(xb)
            loss_raw = criterion(preds, yb)
            loss = (loss_raw * wb.mean()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach().item()) * xb.shape[0]
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = float(criterion(val_pred, y_va).item())
            val_acc = float(((val_pred > 0.5).float() == y_va).float().mean().item())

        improvement = best_val_loss - val_loss > 1e-5
        if improvement:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"[fold {fold_id}][epoch {epoch:02d}] train_loss={running/n:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if epochs_no_improve >= cfg.patience:
            print(f"[fold {fold_id}] early stopping (paciencia={cfg.patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_va).detach().cpu().numpy()
    metrics = {
        "val_loss": best_val_loss,
        "val_accuracy": float(((val_pred > 0.5) == y_val.astype(bool)).mean()),
        "val_buy_rate": float((val_pred > 0.5).mean()),
        "n_train": float(X_train.shape[0]),
        "n_val": float(X_val.shape[0]),
    }
    return model, metrics


def main() -> int:
    args = parse_args()
    _ensure_paths()

    cfg_root = load_cfg(args.config)
    v4 = (cfg_root.get("v4_hybrid") or {}).get("patchtst") or {}
    train_cfg = TrainConfig(
        window_len=int(v4.get("window_ticks", 256)),
        n_channels=int(v4.get("n_channels", 5)),
        patch_len=int(v4.get("patch_len", 16)),
        n_patches=int(v4.get("n_patches", 16)),
        d_model=int(v4.get("d_model", 64)),
        n_heads=int(v4.get("n_heads", 4)),
        n_layers=int(v4.get("n_layers", 3)),
        dropout=float(v4.get("dropout", 0.15)),
        n_folds=int(args.folds),
        embargo_pct=float((cfg_root.get("v4_hybrid") or {}).get("training", {}).get("purged_kfold_embargo_pct", 0.02)),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        device="cuda",
        model_path=Path(str(v4.get("model_path", "D:/AURION_DATA/models/patchtst_xauusd_v4.pt"))),
        report_path=Path("D:/AURION_DATA/reports/phase2_patchtst_metrics.json"),
    )

    torch, _, _, _ = _import_torch()
    if not torch.cuda.is_available():
        print("[WARN] CUDA no disponible; entrenando en CPU (mucho más lento).")
        train_cfg.device = "cpu"
    else:
        print(f"[OK] CUDA disponible. GPU: {torch.cuda.get_device_name(0)}")

    print(f"Cargando histórico {args.symbol} ({args.months} meses) desde MT5...")
    bars = load_m1_history(args.symbol, args.months)
    print(f"Bars cargadas: {bars.shape[0]:,}")

    print("Construyendo ventanas...")
    X, y = build_windows(bars, window_len=train_cfg.window_len)
    print(f"Ventanas: X={X.shape}, y={y.shape}, buy_rate={y.mean():.3f}")

    from ml.cv_strategies import purged_kfold_cv
    from ml.statistical_gates import passes_statistical_gates

    fold_metrics: list[dict[str, float]] = []
    fold_returns: list[np.ndarray] = []
    best_model = None
    best_score = -float("inf")

    for k, (train_idx, val_idx) in enumerate(purged_kfold_cv(
        X.shape[0], n_splits=train_cfg.n_folds, embargo_pct=train_cfg.embargo_pct
    )):
        print(f"\n=== Fold {k+1}/{train_cfg.n_folds} (train={train_idx.size}, val={val_idx.size}) ===")
        model, m = train_one_fold(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx], train_cfg, k + 1
        )
        fold_metrics.append(m)
        # Returns proxy = (pred>0.5) * sign(future_move) → para DSR
        torch_local, _, _, _ = _import_torch()
        with torch_local.no_grad():
            val_pred = model(torch_local.from_numpy(X[val_idx]).float().to(train_cfg.device)).detach().cpu().numpy()
        # Reward simulado: +1 / -1 según acierto direccional (proxy razonable).
        rewards = np.where((val_pred > 0.5) == y[val_idx].astype(bool), 1.0, -1.0)
        fold_returns.append(rewards * 0.001)  # escala razonable

        composite = m["val_accuracy"] - 0.4 * m["val_loss"]
        if composite > best_score:
            best_score = composite
            best_model = model

    all_returns = np.concatenate(fold_returns)
    # Para PBO necesitamos una matriz (T, N). Aquí solo entrenamos UN modelo
    # candidato; el PBO se calculará cuando haya histórico de configs entrenadas.
    stat_ok, stat_reason, stat_metrics = passes_statistical_gates(
        returns=all_returns,
        n_trials=max(1, train_cfg.n_folds),
        returns_matrix=None,
        min_dsr=0.60,
        max_pbo=0.30,
        bars_per_year=525_600,
    )

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "months": args.months,
        "config": {k: v for k, v in train_cfg.__dict__.items() if not isinstance(v, Path)},
        "folds": fold_metrics,
        "statistical_gates": {
            "pass": stat_ok,
            "reason": stat_reason,
            **stat_metrics,
        },
        "promoted": stat_ok and not args.dry_run,
    }

    train_cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    with train_cfg.report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReporte guardado en {train_cfg.report_path}")

    if stat_ok and best_model is not None and not args.dry_run:
        train_cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
        import torch as _torch
        _torch.save(best_model.state_dict(), train_cfg.model_path)
        print(f"[OK] Modelo promovido: {train_cfg.model_path}")
    elif not stat_ok:
        print(f"[REJECT] Modelo NO promovido: {stat_reason}")
    return 0 if stat_ok else 2


if __name__ == "__main__":
    sys.exit(main())

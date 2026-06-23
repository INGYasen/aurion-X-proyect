# AURION X 4.0 HYBRID — Plan de Migración

> Documento vivo. Actualizar al cerrar cada fase con métricas reales.

## Resumen

AURION X 3.0 (ML supervisado + ensemble) → AURION X 4.0 HYBRID (DL + DRL + estadística avanzada),
conservando 100% de la arquitectura 3.0 detrás de feature flags.

Stack runtime confirmado:
- **OS**: Windows 10 (build 26200)
- **Python**: 3.13.13
- **GPU**: NVIDIA RTX 5060 Laptop (CUDA 12.4)
- **PostgreSQL**: Docker (`docker-compose.yml` + `docker-compose.db2.yml`)
- **Símbolo**: XAUUSD (oro)
- **Estrategia**: Scalper M1

## Justificación científica (SLR PRISMA 2020)

Cutipa Mayhua & Vizcarra Ramos, 2026 — Revisión sistemática de ML/DL/DRL en trading algorítmico.

| Capa nueva | Paradigma | Fuente principal |
|------------|-----------|------------------|
| Statistical gates (DSR + PBO) | Estadística inferencial | Bailey & López de Prado (2014, 2016) |
| Purged K-Fold CV | Validación temporal | López de Prado (2018) |
| PatchTST | Transformer atención | Nie et al. (ICLR 2023) |
| PPO Exit Agent | Deep Reinforcement Learning | Schulman et al. (2017); Zhang, Zohren & Roberts (2020) |
| Anomaly Autoencoder | Detección no supervisada | Poutré, Chételat & Morales (2024) |
| Gold-specific features | Variables exógenas | López de Prado (2018) — feature engineering |

## Fases y entregables

### Fase 0 — Pre-flight ✅
- [x] `pyproject.toml` con grupo `[v4]`
- [x] `requirements-v4.txt`
- [x] Bloque `v4_hybrid:` en `config/parameters.yaml` (todos `false`)
- [x] `docs/v4_migration.md`
- [ ] Backup PostgreSQL (script en `scripts/backup_pre_v4.ps1` — ejecutar manualmente)

### Fase 1 — Statistical gates + Purged K-Fold ⬜
Archivos creados:
- `ml/statistical_gates.py` — DSR + PBO
- `ml/cv_strategies.py` — purged k-fold con embargo
- `tests/test_statistical_gates.py` (6 tests)
- `tests/test_cv_strategies.py` (4 tests)

Modificados (sin romper 3.0):
- `ml/oos_validator.py` — añade `passes_statistical_gates()`
- `ml/trainer.py` — parámetro `cv_strategy`
- `ml/daily_retrainer.py` — propaga `cv_strategy`

Activación:
```yaml
v4_hybrid:
  enabled: true
  statistical_gates:
    enabled: true
  training:
    cv_strategy: "purged_kfold"
```

### Fase 2 — PatchTST + MetaStacker ⬜
Archivos:
- `ml/transformer_branch.py`
- `ml/transformer_predictor.py`
- `ml/meta_stacker.py`
- `scripts/train_patchtst_xauusd.py`
- `scripts/train_meta_stacker.py`
- `tests/test_patchtst_branch.py` (5 tests)
- `tests/test_meta_stacker.py` (4 tests)

### Fase 3 — DRL Exit Agent (PPO) ⬜
Archivos:
- `ml/envs/__init__.py`
- `ml/envs/exit_env.py` — Gym env personalizado
- `ml/drl_exit_agent.py` — wrapper PPO
- `ml/drl_exit_shadow.py` — shadow mode logger
- `analytics/drl_shadow_models.py` — modelo SQLAlchemy
- `scripts/train_drl_exit_xauusd.py`
- `scripts/compare_drl_vs_baseline.py`
- `tests/test_drl_exit_env.py` (5 tests)
- `tests/test_drl_exit_agent.py` (4 tests)
- `tests/test_drl_shadow.py` (4 tests)

### Fase 4 — Anomaly AE + Gold features ⬜
Archivos:
- `ml/anomaly_autoencoder.py`
- `scripts/train_anomaly_autoencoder.py`
- `ml/gold_specific_features.py`
- `scripts/cache_economic_calendar.py`
- `ml/quant_features_v3.py`
- `tests/test_anomaly_autoencoder.py` (4 tests)
- `tests/test_gold_features.py` (5 tests)

## Métricas objetivo (rollout v4 → producción)

| Métrica | Umbral 3.0 actual | Umbral 4.0 objetivo |
|---------|-------------------|---------------------|
| Win rate | ≥ 40% | **≥ 60%** |
| Trades/día | 5–15 | 5–15 |
| RR promedio | ≥ 1.3 | **≥ 1.5** |
| Sharpe anualizado neto | — | **≥ 1.3** |
| DSR | — | **≥ 0.60** |
| PBO | — | **≤ 0.30** |
| % cierres R≈0 | 88% | **≤ 30%** |
| Drawdown máx | ≤ 8% | ≤ 8% |
| Latencia P95 / tick | < 1 s | **< 250 ms** |

## Rollback

Cualquier capa nueva tiene flag `enabled: false` por defecto. Para volver a comportamiento 3.0:

```yaml
v4_hybrid:
  enabled: false
```

Los modelos PKL/PT v4 se guardan con sufijo `_v4` y nunca sobrescriben los del 3.0.

## Comandos clave

```powershell
# Instalación deps v4 (con CUDA)
pip install -e .[v4] --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple

# Tests de cada fase
pytest tests/test_statistical_gates.py tests/test_cv_strategies.py -v  # Fase 1
pytest tests/test_patchtst_branch.py tests/test_meta_stacker.py -v     # Fase 2
pytest tests/test_drl_exit_env.py tests/test_drl_exit_agent.py tests/test_drl_shadow.py -v  # Fase 3
pytest tests/test_anomaly_autoencoder.py tests/test_gold_features.py -v  # Fase 4

# Suite completa con coverage
pytest --cov=ml --cov-report=html

# Entrenamiento (offline, GPU)
python -m scripts.train_patchtst_xauusd
python -m scripts.train_drl_exit_xauusd
python -m scripts.train_anomaly_autoencoder
python -m scripts.train_meta_stacker
```

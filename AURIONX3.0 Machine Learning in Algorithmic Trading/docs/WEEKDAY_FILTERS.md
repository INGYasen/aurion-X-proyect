# Filtros por día de la semana

Ver `config/parameters.yaml` → `weekday_filters`.

- **Domingo**: sin trading (`allow_trading: false`).
- **Lunes**: lotaje ×0.25, umbral ML ×1.12, scalper min conf ×1.15.
- **Viernes**: lotaje ×0.85.

El orquestador aplica `lot_factor` y umbral ML por día al abrir y en `RANGE_FADE`.
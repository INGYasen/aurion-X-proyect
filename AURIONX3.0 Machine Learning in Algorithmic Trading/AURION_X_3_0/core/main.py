"""Punto de entrada CLI para AURION X 3.0."""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from pathlib import Path

# Antes de importar sklearn (vía orchestrator): evita spam UserWarning joblib en Py3.13+.
# Incluye el warning de LGBM/sklearn "X does not have valid feature names" que
# inundaba los logs en cada predicción de inferencia (no es un error: el modelo
# fue entrenado con DataFrame y se predice con ndarray; las features son las
# mismas y ya están alineadas por el feature_pipeline).
for _pat in (
    r".*sklearn\.utils\.parallel\.delayed.*",
    r".*propagate the scikit-learn configuration.*",
    r".*does not have valid feature names.*",
    r".*X has feature names.*",
):
    warnings.filterwarnings("ignore", message=_pat, category=UserWarning)

from analytics.db_connection import init_db
from analytics.logger import configure_root_logger, get_logger
from core.orchestrator import Orchestrator
from execution.mt5_connector import MT5Connector
from market.tick_handler import TickHandler

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Carga variables estilo .env en `os.environ` sin sobrescribir existentes."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_int(name: str) -> int | None:
    val = os.getenv(name)
    if val is None or not val.strip():
        return None
    try:
        return int(val.strip())
    except ValueError:
        logger.warning("Variable %s no es entero válido", name)
        return None


def parse_args() -> argparse.Namespace:
    """Define y parsea argumentos de línea de comandos.

    Returns:
        Namespace con opciones del usuario.
    """
    parser = argparse.ArgumentParser(description="AURION X 3.0 — motor de trading")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Ruta a config/parameters.yaml",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging raíz",
    )
    return parser.parse_args()


def main() -> None:
    """Arranca runtime: config, DB, MT5, TickHandler y loop resiliente."""
    args = parse_args()
    configure_root_logger(getattr(logging, args.log_level))
    logger.info("Iniciando AURION X 3.0")
    load_dotenv(_PROJECT_ROOT / ".env")

    config_path = args.config
    if config_path is None:
        candidate = _PROJECT_ROOT / "config" / "parameters.yaml"
        if candidate.is_file():
            config_path = candidate
    elif not config_path.is_absolute():
        config_path = (_PROJECT_ROOT / config_path).resolve()
    orchestrator = Orchestrator(config_path=config_path)
    config = orchestrator.load_config()

    # AURION X 4.4 (10-jun-2026): banner explicito si esta en modo observador.
    if isinstance(config, dict):
        _exec = (config.get("execution") or {}) if isinstance(config, dict) else {}
        if bool(_exec.get("observer_mode", False)):
            logger.warning(
                "==================== MODO OBSERVADOR ACTIVO ===================="
            )
            logger.warning(
                "  El bot consume datos, calcula features/SMC/ML y decide,"
            )
            logger.warning(
                "  pero NO envia ordenes al broker. Cada entrada potencial"
            )
            logger.warning(
                "  se loguea como '[OBSERVER] would_open ...'."
            )
            logger.warning(
                "  Para reactivar trading: poner execution.observer_mode: false"
            )
            logger.warning(
                "  en parameters.yaml y reiniciar el bot."
            )
            logger.warning(
                "================================================================"
            )

    # 1) init DB
    init_db()
    logger.info("DB inicializada")

    # 2) conectar MT5
    mt5_cfg = config.get("mt5", {}) if isinstance(config, dict) else {}
    mt5_login = _env_int("MT5_LOGIN")
    mt5_password = os.getenv("MT5_PASSWORD")
    mt5_server = os.getenv("MT5_SERVER")

    connector = MT5Connector(
        login=mt5_login if mt5_login is not None else mt5_cfg.get("login"),
        password=mt5_password if mt5_password else mt5_cfg.get("password"),
        server=mt5_server if mt5_server else mt5_cfg.get("server"),
        path=os.getenv("MT5_PATH"),
    )
    if not connector.initialize():
        logger.warning("No se pudo inicializar MT5 al inicio; se continuará con auto-reintentos")

    raw_symbols = config.get("symbols") if isinstance(config, dict) else None
    if isinstance(raw_symbols, list) and raw_symbols:
        symbols = [str(s).strip().upper() for s in raw_symbols if str(s).strip()]
    else:
        symbols = ["XAUUSD", "NDX100"]
    for s in symbols:
        connector.subscribe_symbol(s)

    orchestrator.bind_connector(connector)
    orchestrator.start()
    exec_cfg = config.get("execution", {}) if isinstance(config, dict) else {}
    poll_s = float(exec_cfg.get("tick_poll_interval_s", 0.0))
    tick_handler = TickHandler(connector=connector, orchestrator=orchestrator, poll_interval_s=poll_s)

    # 3) loop infinito con manejo de errores y backoff acotado
    backoff_s = 2.0
    while True:
        try:
            tick_handler.run(symbols)
            backoff_s = 2.0
        except KeyboardInterrupt:
            logger.info("Interrupción manual recibida, cerrando...")
            connector.shutdown()
            orchestrator.stop()
            break
        except Exception as exc:
            logger.exception("Fallo en loop principal (%s). Reintentando en %.1fs.", exc, backoff_s)
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 1.5, 60.0)


if __name__ == "__main__":
    main()

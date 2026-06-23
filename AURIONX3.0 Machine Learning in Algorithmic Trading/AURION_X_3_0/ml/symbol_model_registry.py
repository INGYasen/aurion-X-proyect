"""Rutas y símbolos activos para modelos ML por activo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYMBOLS = ("XAUUSD", "NDX100", "BTCUSD")


class SymbolModelRegistry:
    """Estructura models/<SYMBOL>/best_model.pkl + deploy legacy model_<SYM>.pkl."""

    def __init__(
        self,
        models_dir: Path,
        *,
        asset_profiles: Mapping[str, Any] | None = None,
        config_symbols: list[str] | None = None,
        legacy_global: Path | None = None,
    ) -> None:
        self._root = Path(models_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._profiles = asset_profiles if isinstance(asset_profiles, dict) else {}
        self._config_symbols = [str(s).upper() for s in (config_symbols or [])]
        self._legacy_global = legacy_global

    def symbol_dir(self, symbol: str) -> Path:
        d = self._root / str(symbol).upper().strip()
        d.mkdir(parents=True, exist_ok=True)
        (d / "versions").mkdir(parents=True, exist_ok=True)
        return d

    def best_path(self, symbol: str) -> Path:
        return self.symbol_dir(symbol) / "best_model.pkl"

    def rollback_path(self, symbol: str) -> Path:
        return self.symbol_dir(symbol) / "rollback_model.pkl"

    def candidate_path(self, symbol: str) -> Path:
        return self.symbol_dir(symbol) / "candidate_model.pkl"

    def deploy_path(self, symbol: str) -> Path:
        """Ruta que usa inferencia (legacy ml_model_path o model_<SYM>.pkl)."""
        s = symbol.upper()
        prof = self._profiles.get(s, {})
        if isinstance(prof, dict) and prof.get("ml_model_path"):
            return Path(str(prof["ml_model_path"]).strip())
        return self._root / f"model_{s}.pkl"

    def memory_path(self, symbol: str, base_memory: Path) -> Path:
        return base_memory.parent / f"trade_memory_{symbol.upper()}.json"

    def version_manager_root(self, symbol: str | None) -> Path:
        if symbol:
            return self.symbol_dir(symbol)
        return self._root

    def configured_symbols(self) -> list[str]:
        out: set[str] = set(self._config_symbols)
        for k in self._profiles:
            out.add(str(k).upper())
        for s in DEFAULT_SYMBOLS:
            out.add(s)
        return sorted(out)

    def active_symbols(
        self,
        trade_repo: Any | None = None,
        *,
        min_closed_trades: int = 8,
    ) -> list[str]:
        """Símbolos con datos suficientes para retrain contextual."""
        candidates = self.configured_symbols()
        if trade_repo is None:
            return candidates
        active: list[str] = []
        for sym in candidates:
            try:
                n = trade_repo.count_closed_trades(sym)
            except Exception:
                n = 0
            if n >= min_closed_trades:
                active.append(sym)
            else:
                logger.debug("SymbolModelRegistry: %s omitido retrain (n=%s < %s)", sym, n, min_closed_trades)
        return active or candidates

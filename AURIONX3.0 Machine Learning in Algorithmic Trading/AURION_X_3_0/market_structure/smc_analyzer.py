"""AURION X 4.1 SMC — Analizador orquestador (entry point del módulo).

Esta clase es **la** que el Orchestrator instancia. Encapsula:

* Configuración (parámetros por TF, pesos, símbolos habilitados).
* Cache temporal del último ``SMCSnapshot`` por símbolo (TTL configurable
  para evitar recálculo en cada tick — el SMC es caro en TFs altos).
* Persistencia opcional vía ``SMCRepository`` (no rompe si PG no está).
* API pública mínima:
    - ``analyze(symbol, mtf_engine)`` → ``SMCSnapshot``
    - ``snapshot(symbol)`` → último cacheado (sin recálculo)
    - ``features(symbol, direction)`` → dict listo para inyectar en ML
    - ``score_modifier(symbol, direction)`` → tuple ``(delta_conf, grade_bump)``
      que el scalper / institutional puede sumar a su ``effective_confidence``
      y/o usar para promover/degradar ``setup_grade`` (soft, no bloquea).

Por diseño está limitado a **XAUUSD** (configurable). Si se pasa otro símbolo
``analyze()`` devuelve un snapshot vacío sin coste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Mapping, Sequence

from .confluence_engine import (
    DEFAULT_COMPONENT_WEIGHTS,
    DEFAULT_FIB_WEIGHTS,
    compute_confluence,
    macro_institutional_op_bias,
)
from .mtf_engine import (
    DEFAULT_TF_PARAMS_XAUUSD,
    TFParams,
    evaluate_symbol_mtf,
)
from .types import (
    ConfluenceResult,
    SMCSnapshot,
    StructureDirection,
)


DEFAULT_SMC_TFS: tuple[str, ...] = ("M15", "M30", "H1", "H4", "D1", "W1")


@dataclass(slots=True)
class SMCAnalyzerConfig:
    enabled_symbols: tuple[str, ...] = ("XAUUSD",)
    timeframes: tuple[str, ...] = DEFAULT_SMC_TFS
    primary_timeframe: str = "M15"
    cache_ttl_s: float = 8.0           # recalcular máx cada 8 s por símbolo
    persist_snapshots: bool = True
    fib_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FIB_WEIGHTS))
    component_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COMPONENT_WEIGHTS))
    tf_params: dict[str, TFParams] = field(default_factory=lambda: dict(DEFAULT_TF_PARAMS_XAUUSD))
    # Soft integration knobs
    max_conf_delta: float = 0.06       # +/-0.06 a effective_confidence
    grade_bump_threshold: float = 0.70 # confluence >= 0.70 → posible bump
    grade_demote_threshold: float = 0.25  # confluence < 0.25 → posible demote
    log_each_snapshot: bool = False


class SMCAnalyzer:
    """Entry point del módulo SMC. Cache + persistencia + API soft de scores."""

    def __init__(
        self,
        config: SMCAnalyzerConfig | None = None,
        *,
        repository: Any = None,
    ) -> None:
        self._cfg = config or SMCAnalyzerConfig()
        self._repo = repository
        self._cache: dict[str, tuple[float, SMCSnapshot]] = {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def is_enabled_for(self, symbol: str) -> bool:
        return symbol.upper() in {s.upper() for s in self._cfg.enabled_symbols}

    @property
    def config(self) -> SMCAnalyzerConfig:
        return self._cfg

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        symbol: str,
        bars_cb,
        *,
        force: bool = False,
    ) -> SMCSnapshot | None:
        """Calcula (o reusa cache) el snapshot SMC del símbolo.

        Args:
            symbol: ticker.
            bars_cb: callback ``(symbol, tf) -> (highs, lows, closes, volumes, times)``.
            force: si True ignora cache.

        Returns:
            ``SMCSnapshot`` o ``None`` si el símbolo no está habilitado.
        """
        sym = symbol.upper()
        if not self.is_enabled_for(sym):
            return None
        now_m = monotonic()
        cached = self._cache.get(sym)
        if (not force) and cached is not None and (now_m - cached[0]) < self._cfg.cache_ttl_s:
            return cached[1]

        per_tf = evaluate_symbol_mtf(
            symbol=sym,
            timeframes=self._cfg.timeframes,
            bars_cb=bars_cb,
            params_by_tf=self._cfg.tf_params,
        )
        confluence: ConfluenceResult = compute_confluence(
            symbol=sym,
            per_tf=per_tf,
            fib_weights=self._cfg.fib_weights,
            component_weights=self._cfg.component_weights,
            direction_hint=None,
        )
        macro_b, inst_b, op_b = macro_institutional_op_bias(per_tf)
        snapshot = SMCSnapshot(
            symbol=sym,
            generated_at=datetime.now(timezone.utc),
            per_tf=per_tf,
            confluence=confluence,
            primary_timeframe=self._cfg.primary_timeframe,
            macro_bias=macro_b,
            institutional_bias=inst_b,
            operational_bias=op_b,
        )
        self._cache[sym] = (now_m, snapshot)

        if self._cfg.persist_snapshots and self._repo is not None:
            try:
                self._repo.persist_snapshot(snapshot)
            except Exception:
                # nunca propaga
                pass

        return snapshot

    def snapshot(self, symbol: str) -> SMCSnapshot | None:
        cached = self._cache.get(symbol.upper())
        if cached is None:
            return None
        return cached[1]

    # ------------------------------------------------------------------ #
    # ML feature surface
    # ------------------------------------------------------------------ #

    def features(
        self,
        symbol: str,
        direction_hint: str | None = None,
    ) -> dict[str, float]:
        """Devuelve features SMC para el pipeline ML (todas las claves siempre
        presentes, valor 0.0 si no hay snapshot).
        """
        snap = self.snapshot(symbol)
        empty: dict[str, float] = {
            "bos_strength": 0.0,
            "bos_quality": 0.0,
            "choch_strength": 0.0,
            "ob_strength": 0.0,
            "ob_freshness": 0.0,
            "fvg_strength": 0.0,
            "liquidity_score": 0.0,
            "ssl_score": 0.0,
            "bsl_score": 0.0,
            "confluence_score": 0.0,
            "smc_macro_bullish": 0.0,
            "smc_macro_bearish": 0.0,
            "smc_inst_bullish": 0.0,
            "smc_inst_bearish": 0.0,
        }
        if snap is None:
            return empty
        return snap.features_for_ml(direction_hint=direction_hint)

    # ------------------------------------------------------------------ #
    # Soft integration with InstitutionalEngine / Scalper
    # ------------------------------------------------------------------ #

    def score_modifier(
        self,
        symbol: str,
        direction: str,
    ) -> tuple[float, int]:
        """Devuelve ``(delta_conf, grade_bump)``:

        * ``delta_conf``: número en ``[-max_conf_delta, +max_conf_delta]`` que
          se SUMA a ``effective_confidence``.
        * ``grade_bump``: ``+1``, ``0`` o ``-1`` para promover/degradar
          ``setup_grade`` (A+/A/B/C). Nunca bloquea.

        Reglas:

        * Si confluence_score >= ``grade_bump_threshold`` *y* sesgo institucional
          alineado con la señal: ``delta = +max_conf_delta``, ``bump = +1``.
        * Si confluence_score < ``grade_demote_threshold`` *o* sesgo macro
          contradice señal: ``delta = -max_conf_delta * 0.5``, ``bump = -1``.
        * En cualquier otro caso: delta proporcional al confluence_score (sin
          bump).
        """
        snap = self.snapshot(symbol)
        if snap is None:
            return (0.0, 0)
        d = (direction or "").upper()
        signal_bias = StructureDirection.NEUTRAL
        if d in ("BUY", "BULLISH", "LONG"):
            signal_bias = StructureDirection.BULLISH
        elif d in ("SELL", "BEARISH", "SHORT"):
            signal_bias = StructureDirection.BEARISH

        conf = float(snap.confluence.confluence_score)
        macro_ok = (
            signal_bias != StructureDirection.NEUTRAL
            and snap.macro_bias != StructureDirection.NEUTRAL
            and snap.macro_bias == signal_bias
        )
        inst_ok = (
            signal_bias != StructureDirection.NEUTRAL
            and snap.institutional_bias != StructureDirection.NEUTRAL
            and snap.institutional_bias == signal_bias
        )
        macro_contra = (
            signal_bias != StructureDirection.NEUTRAL
            and snap.macro_bias != StructureDirection.NEUTRAL
            and snap.macro_bias != signal_bias
        )

        delta_max = float(self._cfg.max_conf_delta)
        bump = 0
        if conf >= self._cfg.grade_bump_threshold and inst_ok and not macro_contra:
            delta = +delta_max
            bump = +1
        elif conf < self._cfg.grade_demote_threshold or macro_contra:
            delta = -delta_max * 0.5
            bump = -1
        else:
            # Lineal: conf 0.4 → +0.01, conf 0.6 → +0.04 (aprox)
            base = (conf - 0.4) * (delta_max / 0.3)
            delta = max(-delta_max, min(delta_max, base))
            if inst_ok:
                delta *= 1.15
            if macro_ok:
                delta *= 1.10
            delta = max(-delta_max, min(delta_max, delta))
        return (float(delta), int(bump))


__all__ = ["SMCAnalyzer", "SMCAnalyzerConfig", "DEFAULT_SMC_TFS"]

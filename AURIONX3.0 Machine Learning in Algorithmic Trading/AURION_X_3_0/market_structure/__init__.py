"""AURION X 4.1 — Capa SMC institucional (XAUUSD-focused).

Detección, scoring y persistencia de los conceptos Smart Money:

* BOS (Break of Structure)
* CHOCH (Change of Character)
* Order Blocks (Bullish/Bearish, Fresh/Mitigated/Invalidated)
* Fair Value Gaps (Bullish/Bearish, Open/Partial/Filled/Inverted)
* Liquidity Pools (SSL/BSL/EqH/EqL/POOL) + Liquidity Sweeps
* Multi-Timeframe (MN1 → M1) con jerarquía Macro / Institucional / Operativa / Timing
* Confluence Engine con pesos Fibonacci (M15=1, M30=2, H1=3, H4=5, D1=8, W1=13)

Diseño:

* **No-blocking**: el módulo nunca bloquea operaciones. Sólo ajusta confidence
  / setup_grade vía ``SMCAnalyzer.score_modifier``.
* **Soft-fail**: cualquier excepción en detectores se traga internamente; el
  resto del bot sigue funcionando idéntico al estado previo.
* **XAUUSD-first**: parámetros calibrados para oro. Para otros símbolos basta
  con dejar ``enabled_symbols`` por defecto (``("XAUUSD",)``).

Entry point recomendado para integradores:

    >>> from market_structure import SMCAnalyzer, SMCAnalyzerConfig
    >>> analyzer = SMCAnalyzer(SMCAnalyzerConfig(), repository=smc_repo)
    >>> snap = analyzer.analyze("XAUUSD", bars_cb)
    >>> feats = analyzer.features("XAUUSD", direction_hint="SELL")
    >>> delta_conf, grade_bump = analyzer.score_modifier("XAUUSD", "SELL")
"""

from .bos_detector import detect_bos
from .choch_detector import detect_choch
from .confluence_engine import (
    DEFAULT_COMPONENT_WEIGHTS,
    DEFAULT_FIB_WEIGHTS,
    compute_confluence,
    macro_institutional_op_bias,
)
from .fvg_engine import detect_fvgs
from .liquidity_engine import detect_liquidity
from .mtf_engine import (
    DEFAULT_TF_PARAMS_XAUUSD,
    TFParams,
    evaluate_symbol_mtf,
    evaluate_timeframe,
)
from .order_block_engine import detect_order_blocks
from .smc_analyzer import DEFAULT_SMC_TFS, SMCAnalyzer, SMCAnalyzerConfig
from .swing_detector import detect_swings
from .types import (
    BOSEvent,
    BOSStrength,
    CHOCHEvent,
    CHOCHKind,
    ConfluenceResult,
    FairValueGap,
    FVGState,
    LiquidityKind,
    LiquidityPool,
    LiquiditySweep,
    OBState,
    OrderBlock,
    SMCSnapshot,
    StructureDirection,
    SweepKind,
    SwingKind,
    SwingPoint,
    TFContext,
)

__all__ = [
    # Types
    "StructureDirection",
    "SwingKind",
    "SwingPoint",
    "BOSEvent",
    "BOSStrength",
    "CHOCHEvent",
    "CHOCHKind",
    "OBState",
    "OrderBlock",
    "FVGState",
    "FairValueGap",
    "LiquidityKind",
    "LiquidityPool",
    "LiquiditySweep",
    "SweepKind",
    "TFContext",
    "ConfluenceResult",
    "SMCSnapshot",
    # Detectors
    "detect_swings",
    "detect_bos",
    "detect_choch",
    "detect_order_blocks",
    "detect_fvgs",
    "detect_liquidity",
    # MTF + Confluence
    "TFParams",
    "DEFAULT_TF_PARAMS_XAUUSD",
    "evaluate_timeframe",
    "evaluate_symbol_mtf",
    "DEFAULT_FIB_WEIGHTS",
    "DEFAULT_COMPONENT_WEIGHTS",
    "compute_confluence",
    "macro_institutional_op_bias",
    # Orchestrator
    "SMCAnalyzer",
    "SMCAnalyzerConfig",
    "DEFAULT_SMC_TFS",
]

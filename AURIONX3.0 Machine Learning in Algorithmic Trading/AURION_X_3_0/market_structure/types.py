"""AURION X 4.1 SMC — Tipos compartidos del paquete ``market_structure``.

Todas las estructuras son ``dataclass(slots=True, frozen=False)`` para evitar
costes de hashing en el hot-path del Orchestrator y permitir mutación de
campos derivados (``mitigated``, ``fill_percent``…) sin recrear instancias.

Convenciones:

* Los tiempos se almacenan como ``datetime`` UTC-aware (``tzinfo`` obligatorio).
  Cuando un detector recibe arrays sin timestamps caemos a ``datetime.now(UTC)``
  para no romper el contrato.
* Los precios siempre son ``float`` (no ``Decimal``) — el motor SMC es analítico,
  no contable.
* Cada elemento expone ``to_dict()`` para serializar a JSON/PG sin código extra.
* ``score`` siempre vive en ``[0.0, 1.0]`` (clip explícito en cada detector).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class StructureDirection(str, Enum):
    """Dirección institucional de un elemento estructural."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SwingKind(str, Enum):
    """Clasificación de un swing point en la secuencia."""

    HH = "HH"   # Higher High
    HL = "HL"   # Higher Low
    LH = "LH"   # Lower High
    LL = "LL"   # Lower Low
    EQH = "EQH" # Equal Highs (liquidez)
    EQL = "EQL" # Equal Lows  (liquidez)


class BOSStrength(str, Enum):
    """Calidad de un Break of Structure."""

    WEAK = "WEAK"
    STANDARD = "STANDARD"
    STRONG = "STRONG"


class CHOCHKind(str, Enum):
    """Major: cambia el sesgo macro; Minor: cambia el sub-sesgo intra-pierna."""

    MAJOR = "MAJOR"
    MINOR = "MINOR"


class OBState(str, Enum):
    """Estado de un Order Block frente al precio actual."""

    FRESH = "FRESH"           # nunca mitigado
    MITIGATED = "MITIGATED"   # precio entró en la zona y reaccionó
    INVALIDATED = "INVALIDATED"  # precio rompió la zona en dirección contraria


class FVGState(str, Enum):
    """Estado de un Fair Value Gap."""

    OPEN = "OPEN"           # sin fill alguno
    PARTIAL = "PARTIAL"     # llenado parcial
    FILLED = "FILLED"       # totalmente cubierto
    INVERTED = "INVERTED"   # IFVG: fue rebasado y ahora actúa como contrario


class LiquidityKind(str, Enum):
    """Tipo de bolsa/pool de liquidez."""

    SSL = "SSL"              # Sell-Side Liquidity (bajo equal lows / swing lows)
    BSL = "BSL"              # Buy-Side Liquidity  (sobre equal highs / swing highs)
    EQUAL_HIGHS = "EQUAL_HIGHS"
    EQUAL_LOWS = "EQUAL_LOWS"
    POOL = "POOL"            # cluster denso


class SweepKind(str, Enum):
    """Cómo se tomó la liquidez."""

    STOP_HUNT_UP = "STOP_HUNT_UP"      # barrida arriba + rechazo bajista
    STOP_HUNT_DOWN = "STOP_HUNT_DOWN"  # barrida abajo + rechazo alcista
    CLEAN_SWEEP = "CLEAN_SWEEP"        # barrida sin rechazo inmediato


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _enum_value(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v


def _serialize_field(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_serialize_field(x) for x in v]
    if isinstance(v, dict):
        return {k: _serialize_field(x) for k, x in v.items()}
    return v


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SwingPoint:
    """Punto pivote detectado con left/right bars de confirmación."""

    index: int
    price: float
    time: datetime
    kind: SwingKind
    is_high: bool
    strength: float = 0.0  # [0,1] — distancia normalizada vs ATR

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["time"] = self.time.isoformat()
        return d


@dataclass(slots=True)
class BOSEvent:
    """Break of Structure: ruptura confirmada de un swing previo en la dirección de la tendencia.

    El BOS confirma continuación: un swing HH/LL es rebasado por la siguiente
    pierna (cierre por encima/debajo del nivel del swing previo).
    """

    timeframe: str
    direction: StructureDirection
    strength: BOSStrength
    broken_swing_price: float
    broken_swing_time: datetime
    break_time: datetime
    break_price: float
    bos_strength: float       # [0,1]
    bos_quality: float        # [0,1]
    bos_age_bars: int
    bos_distance_pips: float  # distancia entre swing roto y precio de cierre del break
    bos_reliability: float    # combinación strength*quality (cacheada)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "broken_swing_price": self.broken_swing_price,
            "broken_swing_time": self.broken_swing_time.isoformat(),
            "break_time": self.break_time.isoformat(),
            "break_price": self.break_price,
            "bos_strength": self.bos_strength,
            "bos_quality": self.bos_quality,
            "bos_age_bars": self.bos_age_bars,
            "bos_distance_pips": self.bos_distance_pips,
            "bos_reliability": self.bos_reliability,
            "extra": _serialize_field(self.extra),
        }


@dataclass(slots=True)
class CHOCHEvent:
    """Change of Character: cambio de sesgo estructural (LL→LH→HH o HH→HL→LL)."""

    timeframe: str
    direction: StructureDirection   # nuevo sesgo tras el CHOCH
    kind: CHOCHKind
    broken_swing_price: float
    broken_swing_time: datetime
    break_time: datetime
    break_price: float
    choch_strength: float    # [0,1]
    choch_quality: float     # [0,1]
    choch_reliability: float
    choch_age_bars: int
    choch_score: float       # [0,1]  combinación final usada por confluence
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "kind": self.kind.value,
            "broken_swing_price": self.broken_swing_price,
            "broken_swing_time": self.broken_swing_time.isoformat(),
            "break_time": self.break_time.isoformat(),
            "break_price": self.break_price,
            "choch_strength": self.choch_strength,
            "choch_quality": self.choch_quality,
            "choch_reliability": self.choch_reliability,
            "choch_age_bars": self.choch_age_bars,
            "choch_score": self.choch_score,
            "extra": _serialize_field(self.extra),
        }


@dataclass(slots=True)
class OrderBlock:
    """Order Block: última vela contraria antes de un desplazamiento institucional.

    Representamos la OB como una zona ``[low, high]`` extendida hacia el futuro
    hasta que mitigue/invalide.
    """

    timeframe: str
    direction: StructureDirection
    top: float
    bottom: float
    open_time: datetime          # tiempo de la vela OB
    open_index: int              # índice de la vela OB en la serie original
    state: OBState = OBState.FRESH
    ob_strength: float = 0.0     # [0,1]
    ob_volume_score: float = 0.0
    ob_displacement_score: float = 0.0
    ob_reaction_score: float = 0.0
    ob_freshness: float = 1.0    # 1.0 fresh, decae con cada toque
    ob_age_bars: int = 0
    ob_quality: float = 0.0      # [0,1]  agregado final
    mitigation_time: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def height(self) -> float:
        return max(self.top - self.bottom, 1e-9)

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "top": self.top,
            "bottom": self.bottom,
            "open_time": self.open_time.isoformat(),
            "open_index": self.open_index,
            "state": self.state.value,
            "ob_strength": self.ob_strength,
            "ob_volume_score": self.ob_volume_score,
            "ob_displacement_score": self.ob_displacement_score,
            "ob_reaction_score": self.ob_reaction_score,
            "ob_freshness": self.ob_freshness,
            "ob_age_bars": self.ob_age_bars,
            "ob_quality": self.ob_quality,
            "mitigation_time": self.mitigation_time.isoformat() if self.mitigation_time else None,
            "extra": _serialize_field(self.extra),
        }


@dataclass(slots=True)
class FairValueGap:
    """Imbalance entre 3 velas consecutivas (gap formado por low[i+1]>high[i-1] o viceversa)."""

    timeframe: str
    direction: StructureDirection
    top: float
    bottom: float
    open_time: datetime
    open_index: int
    state: FVGState = FVGState.OPEN
    gap_size: float = 0.0        # tamaño en precio puro
    gap_size_pips: float = 0.0   # tamaño en pips (XAUUSD: 1 pip = 0.10$)
    gap_quality: float = 0.0     # [0,1]
    gap_age_bars: int = 0
    gap_fill_percent: float = 0.0  # 0 si está abierto, 1 si full fill
    fvg_strength: float = 0.0    # [0,1]
    inverted_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "top": self.top,
            "bottom": self.bottom,
            "open_time": self.open_time.isoformat(),
            "open_index": self.open_index,
            "state": self.state.value,
            "gap_size": self.gap_size,
            "gap_size_pips": self.gap_size_pips,
            "gap_quality": self.gap_quality,
            "gap_age_bars": self.gap_age_bars,
            "gap_fill_percent": self.gap_fill_percent,
            "fvg_strength": self.fvg_strength,
            "inverted_at": self.inverted_at.isoformat() if self.inverted_at else None,
            "extra": _serialize_field(self.extra),
        }


@dataclass(slots=True)
class LiquidityPool:
    """Bolsa de liquidez: SSL/BSL o equal highs/lows o pool denso."""

    timeframe: str
    kind: LiquidityKind
    price: float                 # nivel principal del pool
    bound_low: float             # extensión inferior (clusters)
    bound_high: float            # extensión superior
    open_time: datetime
    open_index: int
    touches: int = 1             # cuántas veces fue tocado (sin romper)
    density: float = 0.0         # [0,1] — concentración de toques
    liquidity_score: float = 0.0 # [0,1] — score agregado
    stop_hunt_probability: float = 0.0
    swept: bool = False
    swept_at: datetime | None = None
    sweep_kind: SweepKind | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "kind": self.kind.value,
            "price": self.price,
            "bound_low": self.bound_low,
            "bound_high": self.bound_high,
            "open_time": self.open_time.isoformat(),
            "open_index": self.open_index,
            "touches": self.touches,
            "density": self.density,
            "liquidity_score": self.liquidity_score,
            "stop_hunt_probability": self.stop_hunt_probability,
            "swept": self.swept,
            "swept_at": self.swept_at.isoformat() if self.swept_at else None,
            "sweep_kind": self.sweep_kind.value if self.sweep_kind else None,
            "extra": _serialize_field(self.extra),
        }


@dataclass(slots=True)
class LiquiditySweep:
    """Evento de barrida de liquidez: precio rompe un pool y vuelve dentro."""

    timeframe: str
    kind: SweepKind
    pool_price: float
    sweep_high: float
    sweep_low: float
    sweep_time: datetime
    rejection_strength: float = 0.0  # [0,1] velocidad de vuelta dentro del rango
    sweep_strength: float = 0.0      # [0,1] amplitud relativa a ATR
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "kind": self.kind.value,
            "pool_price": self.pool_price,
            "sweep_high": self.sweep_high,
            "sweep_low": self.sweep_low,
            "sweep_time": self.sweep_time.isoformat(),
            "rejection_strength": self.rejection_strength,
            "sweep_strength": self.sweep_strength,
            "extra": _serialize_field(self.extra),
        }


@dataclass(slots=True)
class TFContext:
    """Contexto agregado de un timeframe específico (qué hay y cuán fuerte)."""

    timeframe: str
    bias: StructureDirection
    last_bos: BOSEvent | None = None
    last_choch: CHOCHEvent | None = None
    active_obs: list[OrderBlock] = field(default_factory=list)
    active_fvgs: list[FairValueGap] = field(default_factory=list)
    liquidity_pools: list[LiquidityPool] = field(default_factory=list)
    recent_sweeps: list[LiquiditySweep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "bias": self.bias.value,
            "last_bos": self.last_bos.to_dict() if self.last_bos else None,
            "last_choch": self.last_choch.to_dict() if self.last_choch else None,
            "active_obs": [x.to_dict() for x in self.active_obs],
            "active_fvgs": [x.to_dict() for x in self.active_fvgs],
            "liquidity_pools": [x.to_dict() for x in self.liquidity_pools],
            "recent_sweeps": [x.to_dict() for x in self.recent_sweeps],
        }


@dataclass(slots=True)
class ConfluenceResult:
    """Score final SMC + breakdown por componente y por timeframe.

    Pesos institucionales (configurables, defaults del prompt):

        BOS 10% · CHOCH 15% · OB 20% · FVG 15% · LIQUIDITY 10%

    Sumando 70 %; el 30 % restante lo aporta el ML existente al integrarse
    con el ``composite_score`` del QuantEngine.
    """

    symbol: str
    direction_bias: StructureDirection
    confluence_score: float          # [0,1] capa SMC pura (sin ML)
    bos_component: float = 0.0
    choch_component: float = 0.0
    ob_component: float = 0.0
    fvg_component: float = 0.0
    liquidity_component: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    per_tf: dict[str, float] = field(default_factory=dict)   # tf -> score [0,1]
    fib_weighted_sum: float = 0.0
    fib_weights_total: float = 0.0
    notes: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction_bias": self.direction_bias.value,
            "confluence_score": self.confluence_score,
            "bos_component": self.bos_component,
            "choch_component": self.choch_component,
            "ob_component": self.ob_component,
            "fvg_component": self.fvg_component,
            "liquidity_component": self.liquidity_component,
            "weights": dict(self.weights),
            "per_tf": dict(self.per_tf),
            "fib_weighted_sum": self.fib_weighted_sum,
            "fib_weights_total": self.fib_weights_total,
            "notes": list(self.notes),
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass(slots=True)
class SMCSnapshot:
    """Snapshot completo SMC del símbolo (lo que el Orchestrator consume)."""

    symbol: str
    generated_at: datetime
    per_tf: dict[str, TFContext]
    confluence: ConfluenceResult
    primary_timeframe: str = "M15"
    macro_bias: StructureDirection = StructureDirection.NEUTRAL
    institutional_bias: StructureDirection = StructureDirection.NEUTRAL
    operational_bias: StructureDirection = StructureDirection.NEUTRAL

    def features_for_ml(self, direction_hint: str | None = None) -> dict[str, float]:
        """Devuelve las features SMC que entran al pipeline ML.

        ``direction_hint`` permite orientar los scores al lado de la señal:
        si la señal es ``SELL`` los OB/FVG bajistas valen más que los alcistas.
        Si es ``None`` no se reorienta (score directional-neutro).
        """
        c = self.confluence
        # Liquidez separada en lado SSL/BSL para que el ML pueda discriminar.
        ssl_score = 0.0
        bsl_score = 0.0
        for tfc in self.per_tf.values():
            for pool in tfc.liquidity_pools:
                if pool.kind == LiquidityKind.SSL or pool.kind == LiquidityKind.EQUAL_LOWS:
                    ssl_score = max(ssl_score, pool.liquidity_score)
                elif pool.kind == LiquidityKind.BSL or pool.kind == LiquidityKind.EQUAL_HIGHS:
                    bsl_score = max(bsl_score, pool.liquidity_score)
        # OB freshness: media ponderada por strength entre OBs activas alineadas
        # con la dirección sugerida (o todas si no hay hint).
        ob_freshness = 0.0
        ob_total = 0.0
        for tfc in self.per_tf.values():
            for ob in tfc.active_obs:
                if ob.state == OBState.FRESH:
                    if direction_hint is None or ob.direction.value == _bias_for_dir(direction_hint).value:
                        ob_freshness += ob.ob_freshness * ob.ob_quality
                        ob_total += ob.ob_quality
        ob_freshness = ob_freshness / ob_total if ob_total > 0 else 0.0
        return {
            "bos_strength": float(c.bos_component),
            "bos_quality": float(_safe(c.bos_component * 1.0)),
            "choch_strength": float(c.choch_component),
            "ob_strength": float(c.ob_component),
            "ob_freshness": float(ob_freshness),
            "fvg_strength": float(c.fvg_component),
            "liquidity_score": float(c.liquidity_component),
            "ssl_score": float(ssl_score),
            "bsl_score": float(bsl_score),
            "confluence_score": float(c.confluence_score),
            "smc_macro_bullish": 1.0 if self.macro_bias == StructureDirection.BULLISH else 0.0,
            "smc_macro_bearish": 1.0 if self.macro_bias == StructureDirection.BEARISH else 0.0,
            "smc_inst_bullish": 1.0 if self.institutional_bias == StructureDirection.BULLISH else 0.0,
            "smc_inst_bearish": 1.0 if self.institutional_bias == StructureDirection.BEARISH else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat(),
            "primary_timeframe": self.primary_timeframe,
            "macro_bias": self.macro_bias.value,
            "institutional_bias": self.institutional_bias.value,
            "operational_bias": self.operational_bias.value,
            "per_tf": {tf: ctx.to_dict() for tf, ctx in self.per_tf.items()},
            "confluence": self.confluence.to_dict(),
        }


def _safe(v: float) -> float:
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(v)))


def _bias_for_dir(direction: str) -> StructureDirection:
    d = (direction or "").upper()
    if d in ("BUY", "BULLISH", "LONG", "UP"):
        return StructureDirection.BULLISH
    if d in ("SELL", "BEARISH", "SHORT", "DOWN"):
        return StructureDirection.BEARISH
    return StructureDirection.NEUTRAL

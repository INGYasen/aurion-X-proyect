"""Premium / Discount / Equilibrium engine (AURION X 4.5).

Concepto SMC clasico: dado un swing institucional relevante (preferentemente
H4 -> D1 -> W1), el rango se divide en tres zonas:

    PREMIUM      [0.618, 1.000]   → favorece SHORTS
    EQUILIBRIUM  [0.382, 0.618]   → zona neutra (50% Fib)
    DISCOUNT     [0.000, 0.382]   → favorece LONGS

Donde 0.000 es ``range_low`` y 1.000 es ``range_high``.

Reglas operacionales emitidas por ``evaluate_direction_bias``:

    BUY  → ideal en DISCOUNT, aceptable en EQUILIBRIUM (lower half),
           penalizada en PREMIUM.
    SELL → ideal en PREMIUM,  aceptable en EQUILIBRIUM (upper half),
           penalizada en DISCOUNT.

Este modulo es *standalone*: no depende del orchestrator ni de la BD. Se
alimenta del ``SMCSnapshot`` (last_bos / last_choch / liquidity_pools por TF)
y opcionalmente de barras OHLC crudas. No persiste estado.

Compatibilidad: si no hay snapshot ni datos, devuelve ``PremiumDiscountState.
UNDEFINED`` (no bloquea nada upstream).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class PremiumDiscountState(str, Enum):
    DISCOUNT = "DISCOUNT"
    EQUILIBRIUM = "EQUILIBRIUM"
    PREMIUM = "PREMIUM"
    UNDEFINED = "UNDEFINED"


@dataclass(frozen=True)
class PremiumDiscountResult:
    """Estado P/D operacional + metadatos del rango usado."""

    state: PremiumDiscountState
    range_high: float
    range_low: float
    equilibrium: float
    pct_in_range: float           # 0.0 = low, 1.0 = high
    tf_source: str                # "H4" | "D1" | "W1" | "FALLBACK"
    discount_threshold: float     # default 0.382
    premium_threshold: float      # default 0.618
    notes: list[str] = field(default_factory=list)

    @property
    def is_premium(self) -> bool:
        return self.state == PremiumDiscountState.PREMIUM

    @property
    def is_discount(self) -> bool:
        return self.state == PremiumDiscountState.DISCOUNT

    @property
    def is_equilibrium(self) -> bool:
        return self.state == PremiumDiscountState.EQUILIBRIUM

    @property
    def is_undefined(self) -> bool:
        return self.state == PremiumDiscountState.UNDEFINED

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "range_high": float(self.range_high),
            "range_low": float(self.range_low),
            "equilibrium": float(self.equilibrium),
            "pct_in_range": float(self.pct_in_range),
            "tf_source": self.tf_source,
            "discount_threshold": float(self.discount_threshold),
            "premium_threshold": float(self.premium_threshold),
            "notes": list(self.notes),
        }


class PremiumDiscountEngine:
    """Calculador del estado Premium/Discount.

    Estrategia de busqueda del swing (prioridad descendente):

    1. ``snapshot.per_tf["H4"].last_bos``    (rango = broken_swing_price ↔ break_price)
    2. ``snapshot.per_tf["D1"].last_bos``
    3. ``snapshot.per_tf["W1"].last_bos``
    4. ``snapshot.per_tf["H4"].last_choch``
    5. ``snapshot.per_tf["D1"].last_choch``
    6. ``snapshot.per_tf["W1"].last_choch``
    7. liquidity_pools mas altos/bajos en cada TF (H4 -> D1 -> W1)
    8. UNDEFINED (no bloquea, solo no aporta)

    El modulo es defensivo: cualquier excepcion devuelve UNDEFINED.
    """

    _TF_PRIORITY: tuple[str, ...] = ("H4", "D1", "W1")

    def __init__(
        self,
        *,
        discount_threshold: float = 0.382,
        premium_threshold: float = 0.618,
        min_range_width_pct: float = 0.001,
        tf_priority: Sequence[str] | None = None,
    ) -> None:
        self._discount_thr = float(max(0.0, min(0.49, discount_threshold)))
        self._premium_thr = float(max(0.51, min(1.0, premium_threshold)))
        self._min_range = float(max(1e-6, min_range_width_pct))
        if tf_priority:
            self._tfs = tuple(str(t).upper() for t in tf_priority)
        else:
            self._tfs = self._TF_PRIORITY

    # ─── API publica ─────────────────────────────────────────────────────

    def calculate(
        self,
        *,
        snapshot: Any | None,
        last_close: float,
    ) -> PremiumDiscountResult:
        """Devuelve el estado P/D del precio actual frente al swing macro.

        ``snapshot`` puede ser ``None`` o un ``SMCSnapshot``; se accede via
        ``getattr`` para no acoplar duro al tipo concreto.
        """
        try:
            close = float(last_close)
            if close <= 0:
                return self._undefined("close_invalid")
            if snapshot is None:
                return self._undefined("no_snapshot")

            for tf in self._tfs:
                rng = self._range_from_tf(snapshot, tf)
                if rng is None:
                    continue
                high, low, src_label = rng
                if high <= low:
                    continue
                width_pct = (high - low) / max(close, 1e-9)
                if width_pct < self._min_range:
                    continue
                return self._classify(
                    high=high, low=low, close=close, tf_source=f"{tf}/{src_label}",
                )
            return self._undefined("no_valid_swing_in_priority_tfs")
        except Exception as exc:  # defensivo: nunca rompe upstream
            return self._undefined(f"exception:{type(exc).__name__}")

    # ─── helpers internos ───────────────────────────────────────────────

    def _range_from_tf(
        self,
        snapshot: Any,
        tf: str,
    ) -> tuple[float, float, str] | None:
        """Intenta extraer (high, low, source_label) del TF dado."""
        per_tf = getattr(snapshot, "per_tf", None) or {}
        ctx = per_tf.get(tf)
        if ctx is None:
            return None

        # 1) last_bos: broken_swing_price ↔ break_price
        bos = getattr(ctx, "last_bos", None)
        if bos is not None:
            high, low = self._minmax(
                getattr(bos, "broken_swing_price", None),
                getattr(bos, "break_price", None),
            )
            if high is not None and low is not None and high > low:
                return high, low, "bos"

        # 2) last_choch: broken_swing_price ↔ break_price
        choch = getattr(ctx, "last_choch", None)
        if choch is not None:
            high, low = self._minmax(
                getattr(choch, "broken_swing_price", None),
                getattr(choch, "break_price", None),
            )
            if high is not None and low is not None and high > low:
                return high, low, "choch"

        # 3) liquidity_pools: top BSL price ↔ bottom SSL price
        pools = getattr(ctx, "liquidity_pools", None) or []
        if pools:
            highs: list[float] = []
            lows: list[float] = []
            for p in pools:
                try:
                    price = float(getattr(p, "price"))
                    kind_obj = getattr(p, "kind", None)
                    kind = kind_obj.value if hasattr(kind_obj, "value") else str(kind_obj)
                    kind_u = (kind or "").upper()
                    if "BSL" in kind_u or "HIGH" in kind_u:
                        highs.append(price)
                    elif "SSL" in kind_u or "LOW" in kind_u:
                        lows.append(price)
                    else:
                        highs.append(price)
                        lows.append(price)
                except (TypeError, ValueError):
                    continue
            if highs and lows:
                hi, lo = max(highs), min(lows)
                if hi > lo:
                    return hi, lo, "liquidity"

        # 4) active_obs: rango entre OB mas alto y mas bajo
        obs = getattr(ctx, "active_obs", None) or []
        if obs:
            try:
                tops = [float(getattr(o, "top")) for o in obs]
                bottoms = [float(getattr(o, "bottom")) for o in obs]
                if tops and bottoms:
                    hi, lo = max(tops), min(bottoms)
                    if hi > lo:
                        return hi, lo, "obs"
            except (TypeError, ValueError):
                pass

        return None

    @staticmethod
    def _minmax(a: Any, b: Any) -> tuple[float | None, float | None]:
        try:
            af = float(a) if a is not None else None
            bf = float(b) if b is not None else None
        except (TypeError, ValueError):
            return None, None
        if af is None or bf is None:
            return None, None
        return max(af, bf), min(af, bf)

    def _classify(
        self,
        *,
        high: float,
        low: float,
        close: float,
        tf_source: str,
    ) -> PremiumDiscountResult:
        denom = max(high - low, 1e-9)
        pct = float((close - low) / denom)
        pct = max(0.0, min(1.0, pct))
        eq = (high + low) / 2.0
        if pct >= self._premium_thr:
            state = PremiumDiscountState.PREMIUM
        elif pct <= self._discount_thr:
            state = PremiumDiscountState.DISCOUNT
        else:
            state = PremiumDiscountState.EQUILIBRIUM
        notes = [f"pct={pct:.3f}", f"src={tf_source}"]
        return PremiumDiscountResult(
            state=state,
            range_high=float(high),
            range_low=float(low),
            equilibrium=float(eq),
            pct_in_range=pct,
            tf_source=tf_source,
            discount_threshold=self._discount_thr,
            premium_threshold=self._premium_thr,
            notes=notes,
        )

    def _undefined(self, reason: str) -> PremiumDiscountResult:
        return PremiumDiscountResult(
            state=PremiumDiscountState.UNDEFINED,
            range_high=0.0,
            range_low=0.0,
            equilibrium=0.0,
            pct_in_range=0.5,
            tf_source="none",
            discount_threshold=self._discount_thr,
            premium_threshold=self._premium_thr,
            notes=[reason],
        )

    # ─── reglas de bias direccional ─────────────────────────────────────

    def evaluate_direction_bias(
        self,
        *,
        direction: str,
        pd_result: PremiumDiscountResult,
    ) -> tuple[float, str]:
        """Devuelve (delta_score, reason).

        ``delta_score`` se suma al SMC score:
            BUY  en DISCOUNT     → +12
            BUY  en EQUILIBRIUM  → +4 (lower half) / -4 (upper half)
            BUY  en PREMIUM      → -20
            SELL en PREMIUM      → +12
            SELL en EQUILIBRIUM  → +4 (upper half) / -4 (lower half)
            SELL en DISCOUNT     → -20
            cualquiera UNDEFINED → 0 (no penaliza ni premia)
        """
        d = (direction or "").strip().upper()
        if pd_result.is_undefined or d not in ("BUY", "SELL", "LONG", "SHORT"):
            return 0.0, f"pd_undefined_or_invalid_dir({d})"
        is_buy = d in ("BUY", "LONG")
        pct = pd_result.pct_in_range

        if pd_result.is_discount:
            if is_buy:
                return 12.0, "buy_in_discount"
            return -20.0, "sell_in_discount_penalty"

        if pd_result.is_premium:
            if is_buy:
                return -20.0, "buy_in_premium_penalty"
            return 12.0, "sell_in_premium"

        # EQUILIBRIUM: subdivide en lower (<0.5) y upper (>=0.5)
        if pct < 0.5:
            return (4.0 if is_buy else -4.0), (
                "buy_eq_lower_half" if is_buy else "sell_eq_lower_half_mild"
            )
        return (-4.0 if is_buy else 4.0), (
            "buy_eq_upper_half_mild" if is_buy else "sell_eq_upper_half"
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "PremiumDiscountEngine":
        cfg = cfg or {}
        return cls(
            discount_threshold=float(cfg.get("discount_threshold", 0.382)),
            premium_threshold=float(cfg.get("premium_threshold", 0.618)),
            min_range_width_pct=float(cfg.get("min_range_width_pct", 0.001)),
            tf_priority=cfg.get("tf_priority"),
        )

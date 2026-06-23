"""Cálculo de tamaño de posición en función de confianza y riesgo por SL."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any, Mapping

from analytics.logger import get_logger

logger = get_logger(__name__)

_ONE = Decimal("1")
_TWO = Decimal("2")
_HUNDRED = Decimal("100")
_ZERO = Decimal("0")


class PositionSizer:
    """Convierte reglas de sizing por símbolo + riesgo en volumen final seguro."""

    def __init__(self, risk_per_trade_pct: Decimal = Decimal("1.0")) -> None:
        """Porcentaje por defecto de riesgo por trade (acotado a 1-2%)."""
        self._risk_pct = risk_per_trade_pct
        logger.debug("PositionSizer risk_pct=%s", risk_per_trade_pct)

    @staticmethod
    def _to_decimal(value: Any, *, field: str) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"{field} inválido: {value!r}") from exc

    @staticmethod
    def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        units = (value / step).quantize(Decimal("1"), rounding=ROUND_DOWN)
        return (units * step).normalize()

    @staticmethod
    def _base_lot_by_confidence(symbol: str, confidence: Decimal) -> Decimal:
        s = symbol.upper().strip()
        if s == "XAUUSD":
            base = Decimal("0.03")
        elif s in {"USTEC", "NDX100", "NAS100", "US100"}:
            base = Decimal("0.09")
        elif s in {"BTCUSD", "BITCOIN"}:
            base = Decimal("0.05")
        else:
            base = Decimal("0.01")

        if confidence >= Decimal("0.65"):
            mult = Decimal("1.8")
        elif confidence >= Decimal("0.50"):
            mult = Decimal("1.5")
        elif confidence >= Decimal("0.42"):
            mult = Decimal("1.0")
        else:
            mult = Decimal("0.85")
        return (base * mult).quantize(Decimal("0.0001"))

    @staticmethod
    def _legacy_base_lot_by_confidence(symbol: str, confidence: Decimal) -> Decimal:
        s = symbol.upper().strip()
        if s == "XAUUSD":
            if confidence >= Decimal("0.85"):
                return Decimal("0.03")
            if confidence >= Decimal("0.75"):
                return Decimal("0.02")
            return Decimal("0.01")
        if s in {"USTEC", "NDX100", "NAS100", "US100"}:
            if confidence >= Decimal("0.85"):
                return Decimal("0.15")
            if confidence >= Decimal("0.75"):
                return Decimal("0.12")
            return Decimal("0.09")
        # Fallback prudente para otros símbolos
        return Decimal("0.01")

    def size_for_fixed_risk(
        self,
        equity: Decimal,
        stop_distance_points: Decimal,
        point_value_per_lot: Decimal,
    ) -> Decimal:
        """Volumen máximo que respeta riesgo por trade (self._risk_pct en %)."""
        if equity <= 0 or stop_distance_points <= 0 or point_value_per_lot <= 0:
            return Decimal("0")
        risk_pct = min(_TWO, max(_ONE, self._risk_pct))
        risk_money = equity * (risk_pct / _HUNDRED)
        loss_per_lot = stop_distance_points * point_value_per_lot
        if loss_per_lot <= 0:
            return Decimal("0")
        return risk_money / loss_per_lot

    def compute_lot(
        self,
        symbol: str,
        confidence: float | Decimal,
        config: Mapping[str, Any],
    ) -> float:
        """Calcula lote final por símbolo respetando riesgo (1-2% equity) y SL.

        `config` esperado (claves):
        - `equity` (obligatoria)
        - `sl_pips` (obligatoria)
        - `pip_value_per_lot` (obligatoria, monetario por pip/lote)
        - `risk_pct` (opcional, se acota entre 1 y 2)
        - `max_lot` (opcional, límite duro adicional)
        - `lot_step` (opcional, default 0.01)
        - `min_lot` (opcional, default base por símbolo)
        """
        conf = self._to_decimal(confidence, field="confidence")
        equity = self._to_decimal(config.get("equity"), field="equity")
        sl_pips = self._to_decimal(config.get("sl_pips"), field="sl_pips")
        pip_value = self._to_decimal(config.get("pip_value_per_lot"), field="pip_value_per_lot")

        if equity <= 0 or sl_pips <= 0 or pip_value <= 0:
            logger.warning("compute_lot: equity/sl/pip_value inválidos; lote=0")
            return 0.0

        # Escalado base por confianza
        table_lot = self._base_lot_by_confidence(symbol, conf)

        # Riesgo por trade dinámico (capas: configuración + capital + racha/dd).
        risk_pct_cfg = config.get("risk_pct", self._risk_pct)
        risk_pct = self._to_decimal(risk_pct_cfg, field="risk_pct")
        risk_pct_min = self._to_decimal(config.get("risk_pct_min", "0.40"), field="risk_pct_min")
        risk_pct_max = self._to_decimal(config.get("risk_pct_max", "2.50"), field="risk_pct_max")
        if risk_pct_min > risk_pct_max:
            risk_pct_min, risk_pct_max = risk_pct_max, risk_pct_min
        risk_pct = min(risk_pct_max, max(risk_pct_min, risk_pct))

        capital_multiplier = self._to_decimal(config.get("capital_multiplier", "1.0"), field="capital_multiplier")
        capital_multiplier = min(Decimal("1.80"), max(Decimal("0.50"), capital_multiplier))
        risk_pct = risk_pct * capital_multiplier

        loss_streak = int(config.get("loss_streak", 0) or 0)
        if loss_streak >= 5:
            risk_pct *= Decimal("0.60")
        elif loss_streak >= 3:
            risk_pct *= Decimal("0.75")

        dd_pct = self._to_decimal(config.get("drawdown_pct", "0"), field="drawdown_pct")
        if dd_pct >= Decimal("6.0"):
            risk_pct *= Decimal("0.60")
        elif dd_pct >= Decimal("3.0"):
            risk_pct *= Decimal("0.80")

        edge_multiplier = self._to_decimal(config.get("edge_multiplier", "1.0"), field="edge_multiplier")
        edge_multiplier = min(Decimal("1.40"), max(Decimal("0.70"), edge_multiplier))
        perf_multiplier = self._to_decimal(config.get("performance_multiplier", "1.0"), field="performance_multiplier")
        perf_multiplier = min(Decimal("1.30"), max(Decimal("0.70"), perf_multiplier))
        risk_pct = min(risk_pct_max, max(risk_pct_min, risk_pct * edge_multiplier * perf_multiplier))

        risk_money_cap = equity * (risk_pct / _HUNDRED)
        loss_per_lot = sl_pips * pip_value
        max_lot_by_risk = risk_money_cap / loss_per_lot if loss_per_lot > 0 else Decimal("0")

        lot_step = self._to_decimal(config.get("lot_step", "0.01"), field="lot_step")
        # Mínimo ejecutable del bróker (paso), no el lote "ideal" de la tabla: si no,
        # el riesgo capa a 0.01 lotes pero min_lot=0.027 exige más y compute_lot devuelve 0 siempre.
        min_lot_raw = config.get("min_lot")
        if min_lot_raw is None:
            min_lot = lot_step
        else:
            min_lot = self._to_decimal(min_lot_raw, field="min_lot")
        if min_lot < lot_step:
            min_lot = lot_step
        max_lot_cfg = self._to_decimal(config.get("max_lot", "999"), field="max_lot")
        if max_lot_cfg <= _ZERO:
            return 0.0

        # Nunca exceder límites: tabla, riesgo y hard-cap
        raw = min(table_lot, max_lot_by_risk, max_lot_cfg)
        lot = self._floor_to_step(raw, lot_step)

        # Si el riesgo permitido no alcanza el mínimo ejecutable, no abrir.
        if lot < min_lot:
            logger.info(
                "compute_lot: lote por riesgo (%s) menor que min_lot (%s), devuelve 0",
                lot,
                min_lot,
            )
            return 0.0

        # Guardrail final: recomprobar riesgo monetario para evitar exceso por redondeos.
        est_risk = lot * loss_per_lot
        if est_risk > risk_money_cap:
            lot = self._floor_to_step(risk_money_cap / loss_per_lot, lot_step)
            if lot < min_lot:
                return 0.0

        return float(lot)

    def compute_lot_intelligent(
        self,
        symbol: str,
        confidence: float | Decimal,
        config: Mapping[str, Any],
        *,
        setup_grade: str = "B",
        setup_lot_multiplier: float = 1.0,
        regime_mode: str = "",
        montecarlo_factor: float = 1.0,
    ) -> float:
        """Lote dinámico: confianza + grado setup + régimen + Monte Carlo."""
        base = self.compute_lot(symbol, confidence, config)
        if base <= 0:
            return 0.0
        conf = float(confidence) if not isinstance(confidence, Decimal) else float(confidence)
        grade_mult = {
            "A+": 1.22,
            "A": 1.10,
            "B": 0.92,
            "C": 0.72,
            "X": 0.0,
        }.get(str(setup_grade).upper(), 0.85)
        if grade_mult <= 0:
            return 0.0
        regime_mult = 1.0
        if regime_mode == "defensive":
            regime_mult = 0.65
        elif regime_mode == "high_volatility":
            regime_mult = 0.78
        elif regime_mode == "trend_following" and conf >= 0.72:
            regime_mult = 1.08
        mc = max(0.45, min(1.15, float(montecarlo_factor)))
        lot = base * grade_mult * max(0.5, setup_lot_multiplier) * regime_mult * mc
        if conf >= 0.90 and symbol.upper() == "XAUUSD":
            lot = max(lot, 0.03 * grade_mult)
        elif conf >= 0.65 and symbol.upper() == "XAUUSD":
            lot = min(lot, max(lot, 0.01))
        lot_step = float(config.get("lot_step", 0.01) or 0.01)
        max_lot = float(config.get("max_lot", 999) or 999)
        lot = self._floor_to_step(self._to_decimal(lot, field="lot"), self._to_decimal(lot_step, field="lot_step"))
        return float(min(lot, self._to_decimal(max_lot, field="max_lot")))

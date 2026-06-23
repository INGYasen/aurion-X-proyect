"""Validación final de entradas antes de enviar órdenes al broker."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from analytics.logger import get_logger
from market.volatility_detector import VolatilityRegime

from .structure_detector import StructureResult, Trend

logger = get_logger(__name__)

_DEFAULT_MAX_SPREAD = Decimal("0.00050")
_DEFAULT_LEVEL_TOLERANCE = Decimal("0.00030")


class EntryValidator:
    """Valida ``signal`` frente a estructura, volatilidad y spread antes de ejecutar."""

    def __init__(self, rules: dict[str, Any] | None = None) -> None:
        """Define límites (extensibles vía ``rules``).

        Claves reconocidas:
            * ``max_spread``: spread máximo permitido (``Decimal`` o convertible).
            * ``level_tolerance``: distancia máxima al nivel para considerar pullback
              cercano a ``key_levels``.
            * ``relax_range_when_mtf_aligned``: si es verdadero y ``mtf_filter_passed``,
              no rechazar entradas solo por ``Trend.RANGE``.
            * ``ignore_spread``: si es verdadero, no se compara el spread con ``max_spread``.
            * ``allow_low_vol_breakout_when_mtf_aligned``: si es verdadero y ``mtf_filter_passed``,
              permite BREAKOUT/MOMENTUM en régimen LOW (por defecto se rechazan).
            * ``allow_extreme_when_mtf_aligned``: si es verdadero y ``mtf_filter_passed``, no bloquea
              solo por régimen EXTREME (ticks muy nerviosos); el orquestador ya filtró MTF.
            * ``scalper_ignore_structure_trend``: con MTF ya validado, acepta señales scalper
              aunque contradigan la tendencia de la estructura M15 (micro vs macro).
            * ``allow_against_trend_when_ml_strong``: si es verdadero y el orquestador
              indica ``ml_strong_override=True`` (ml_conf alto + alineación short-term +
              sin contradicción macro), no rechaza por tendencia M15 opuesta. Apex/Hyper
              y los caps de SL siguen siendo la red de seguridad.
        """
        self._rules = dict(rules or {})
        logger.debug("EntryValidator reglas: %s", list(self._rules))

    def _max_spread(self) -> Decimal:
        raw = self._rules.get("max_spread", _DEFAULT_MAX_SPREAD)
        return Decimal(str(raw))

    def _level_tolerance(self) -> Decimal:
        raw = self._rules.get("level_tolerance", _DEFAULT_LEVEL_TOLERANCE)
        return Decimal(str(raw))

    @staticmethod
    def _is_buy(signal: Mapping[str, Any]) -> bool:
        side = signal.get("side", signal.get("direction"))
        if side is None:
            raise ValueError("signal debe incluir 'side' o 'direction'")
        if isinstance(side, bool):
            return side
        if isinstance(side, int | float):
            return int(side) > 0
        s = str(side).strip().lower()
        if s in {"buy", "long", "b", "l", "1"}:
            return True
        if s in {"sell", "short", "s", "-1"}:
            return False
        raise ValueError(f"Lado de señal no reconocido: {side!r}")

    @staticmethod
    def _to_decimal(value: Any, *, field: str) -> Decimal:
        try:
            return Decimal(str(value))
        except Exception as exc:
            raise ValueError(f"{field} no numérico: {value!r}") from exc

    def _price_near_key_levels(self, signal: Mapping[str, Any], levels: tuple[Decimal, ...]) -> bool:
        if not levels:
            return False
        px_raw = signal.get("price", signal.get("entry_price"))
        if px_raw is None:
            return False
        px = self._to_decimal(px_raw, field="price")
        tol = self._level_tolerance()
        return any(abs(px - lv) <= tol for lv in levels)

    def validate(
        self,
        signal: Mapping[str, Any],
        structure: StructureResult,
        volatility: VolatilityRegime,
        spread: Any,
        *,
        mtf_filter_passed: bool = False,
        max_spread_abs: Any | None = None,
        ml_strong_override: bool = False,
    ) -> tuple[bool, str]:
        """Valida entrada según spread, volatilidad, estructura y alineación.

        Reglas:
            * Spread dentro del límite configurado (omitido si ``ignore_spread``).
            * Volatilidad adaptativa:
              - ``LOW``: evita setups de breakout/momentum, permite reversals limpios.
              - ``NORMAL``: comportamiento estándar.
              - ``HIGH``: no bloquea, solo penaliza el score en capas superiores.
              - ``EXTREME``: bloquea.
            * ``Trend.RANGE``: rechazo por defecto; si ``relax_range_when_mtf_aligned``
              está activo y ``mtf_filter_passed`` es verdadero, se acepta la entrada
              (el filtro multi-timeframe ya validó la dirección).
            * Alineación: a favor de tendencia (compra en UP, venta en DOWN) **o**
              operación marcada como pullback con precio cercano a ``key_levels``.

        Args:
            signal: Debe incluir ``side``/``direction``; opcional ``pullback`` (bool),
                ``price``/``entry_price`` para comprobar proximidad a niveles.
            structure: Resultado del detector de estructura.
            volatility: Régimen actual de volatilidad.
            spread: Spread actual (mismo tipo de unidad que ``max_spread`` en reglas).
            mtf_filter_passed: ``True`` si ya pasó el filtro MTF del orquestador.
            max_spread_abs: Tope de spread en unidades de precio (ask−bid); si es ``None`` usa la regla global.

        Returns:
            ``(True, "ok")`` o ``(False, motivo)`` en español.
        """
        if not bool(self._rules.get("ignore_spread", False)):
            try:
                sp = self._to_decimal(spread, field="spread")
            except ValueError as exc:
                logger.warning("validate: spread inválido: %s", exc)
                return False, "spread inválido"

            max_sp = (
                Decimal(str(max_spread_abs))
                if max_spread_abs is not None
                else self._max_spread()
            )
            if sp > max_sp:
                logger.debug("validate: spread fuera de límite %s > %s", sp, max_sp)
                return False, f"spread fuera de límite ({sp} > {max_sp})"

        if isinstance(volatility, VolatilityRegime):
            regime = volatility
        else:
            try:
                regime = VolatilityRegime(str(volatility).strip().lower())
            except ValueError:
                return False, f"volatilidad no reconocida: {volatility!r}"
        setup_type = str(signal.get("setup_type", "UNKNOWN")).upper()
        if regime == VolatilityRegime.EXTREME:
            if bool(self._rules.get("allow_extreme_when_mtf_aligned", False)) and mtf_filter_passed:
                logger.debug("validate: EXTREME ignorado (MTF validado + allow_extreme_when_mtf_aligned)")
            else:
                return False, "volatilidad EXTREME: bloqueo de seguridad"
        if regime == VolatilityRegime.LOW:
            if "BREAKOUT" in setup_type or "MOMENTUM" in setup_type:
                if bool(self._rules.get("allow_low_vol_breakout_when_mtf_aligned", False)) and mtf_filter_passed:
                    pass
                else:
                    return False, "LOW volatility: evitar breakout/momentum"

        relax_range = bool(self._rules.get("relax_range_when_mtf_aligned", False))
        if structure.trend == Trend.RANGE:
            if not (relax_range and mtf_filter_passed):
                return False, "evitar rango extremo: estructura RANGE"

        try:
            buy = self._is_buy(signal)
        except ValueError as exc:
            return False, str(exc)

        if structure.trend == Trend.RANGE and relax_range and mtf_filter_passed:
            with_trend = True
        else:
            with_trend = (structure.trend == Trend.UP and buy) or (structure.trend == Trend.DOWN and not buy)
        if (
            not with_trend
            and bool(self._rules.get("scalper_ignore_structure_trend", False))
            and mtf_filter_passed
        ):
            tags = ("MICRO_PULLBACK", "RANGE_FADE", "RANGE_BREAKOUT", "BREAKOUT", "MOMENTUM")
            if any(tag in setup_type for tag in tags):
                with_trend = True
                logger.debug("validate: scalper ignora contradicción estructura M15 vs señal")
        if (
            not with_trend
            and ml_strong_override
            and bool(self._rules.get("allow_against_trend_when_ml_strong", False))
        ):
            with_trend = True
            logger.debug(
                "validate: bypass trend M15 por ml_strong_override (trend=%s buy=%s setup=%s)",
                structure.trend.value,
                buy,
                setup_type,
            )
        if not with_trend:
            t = structure.trend.value
            side = "compra" if buy else "venta"
            return False, f"sin alineación con tendencia ({t} vs señal {side})"

        pullback_flag = bool(signal.get("pullback", False))
        range_fade = bool(signal.get("range_fade", False))
        if pullback_flag and not range_fade and not self._price_near_key_levels(signal, structure.key_levels):
            return False, "pullback sin proximidad a nivel clave"

        logger.debug("validate: aprobado (trend=%s pullback=%s)", structure.trend.value, pullback_flag)
        return True, "ok"

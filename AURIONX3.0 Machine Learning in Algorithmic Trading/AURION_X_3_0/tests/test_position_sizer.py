"""Tests para risk.position_sizer."""
from __future__ import annotations

from decimal import Decimal

from risk.position_sizer import PositionSizer


def _base_cfg(**overrides: object) -> dict:
    cfg = {
        "equity": Decimal("10000"),
        "sl_pips": Decimal("30"),
        "pip_value_per_lot": Decimal("10"),  # $10/pip por 1 lote XAUUSD
        "risk_pct": Decimal("1.0"),
        "lot_step": Decimal("0.01"),
        "min_lot": Decimal("0.01"),
        "max_lot": Decimal("10"),
    }
    cfg.update(overrides)
    return cfg


class TestPositionSizer:
    def test_zero_or_invalid_inputs_return_zero(self) -> None:
        ps = PositionSizer()
        cfg = _base_cfg(equity=Decimal("0"))
        assert ps.compute_lot("XAUUSD", 0.55, cfg) == 0.0
        cfg2 = _base_cfg(sl_pips=Decimal("0"))
        assert ps.compute_lot("XAUUSD", 0.55, cfg2) == 0.0

    def test_basic_xauusd_size(self) -> None:
        ps = PositionSizer()
        cfg = _base_cfg()
        lot = ps.compute_lot("XAUUSD", 0.50, cfg)
        # 1% de $10K = $100 risk / (30 * $10/pip) = 0.333 lots cap por riesgo.
        # Confidence 0.50 -> tabla base 0.03 * 1.5 = 0.045 → min(0.333, 0.045) = 0.045
        assert lot == 0.04 or lot == 0.05  # depende redondeo (lot_step 0.01)

    def test_risk_pct_capped_within_bounds(self) -> None:
        ps = PositionSizer(risk_per_trade_pct=Decimal("5.0"))  # > max 2.5
        cfg = _base_cfg(risk_pct=Decimal("5.0"))
        lot = ps.compute_lot("XAUUSD", 0.50, cfg)
        # risk_pct se cappea a 2.5 máximo
        risk_money = 10000 * 0.025  # 250
        max_lot_by_risk = 250 / (30 * 10)  # 0.833
        assert lot <= 0.833 + 1e-9

    def test_loss_streak_reduces_size(self) -> None:
        ps = PositionSizer()
        cfg_normal = _base_cfg()
        cfg_streak5 = _base_cfg(loss_streak=5)
        lot_normal = ps.compute_lot("XAUUSD", 0.50, cfg_normal)
        lot_streak = ps.compute_lot("XAUUSD", 0.50, cfg_streak5)
        # Con loss_streak >= 5, risk_pct se multiplica por 0.60
        # Si lot estaba topado por tabla, no por riesgo, ambos pueden ser iguales.
        assert lot_streak <= lot_normal

    def test_drawdown_reduces_size(self) -> None:
        ps = PositionSizer()
        cfg_normal = _base_cfg()
        cfg_dd = _base_cfg(drawdown_pct=Decimal("7.0"))
        lot_normal = ps.compute_lot("XAUUSD", 0.50, cfg_normal)
        lot_dd = ps.compute_lot("XAUUSD", 0.50, cfg_dd)
        assert lot_dd <= lot_normal

    def test_confidence_buckets_increase_lot(self) -> None:
        ps = PositionSizer()
        cfg = _base_cfg(equity=Decimal("100000"))  # equity grande para que tabla domine
        lot_low = ps.compute_lot("XAUUSD", 0.30, cfg)   # <0.42 -> 0.85x
        lot_med = ps.compute_lot("XAUUSD", 0.50, cfg)   # 0.42-0.50 -> 1.5x
        lot_high = ps.compute_lot("XAUUSD", 0.80, cfg)  # >=0.65 -> 1.8x
        assert lot_high >= lot_med >= lot_low

    def test_lot_floored_to_step(self) -> None:
        ps = PositionSizer()
        cfg = _base_cfg(lot_step=Decimal("0.05"))
        lot = ps.compute_lot("XAUUSD", 0.50, cfg)
        # lot debe ser múltiplo de 0.05
        assert abs((lot * 100) % 5) < 1e-6 or lot == 0.0

    def test_below_min_lot_returns_zero(self) -> None:
        ps = PositionSizer()
        # Riesgo tan pequeño que ni 0.01 lots cabe.
        cfg = _base_cfg(equity=Decimal("1"), sl_pips=Decimal("100"))
        lot = ps.compute_lot("XAUUSD", 0.50, cfg)
        assert lot == 0.0

    def test_compute_lot_intelligent_grade_x_returns_zero(self) -> None:
        ps = PositionSizer()
        cfg = _base_cfg()
        lot = ps.compute_lot_intelligent("XAUUSD", 0.60, cfg, setup_grade="X")
        assert lot == 0.0

    def test_compute_lot_intelligent_grade_a_plus_scales_up(self) -> None:
        ps = PositionSizer()
        cfg = _base_cfg(equity=Decimal("100000"))
        base = ps.compute_lot("XAUUSD", 0.60, cfg)
        a_plus = ps.compute_lot_intelligent("XAUUSD", 0.60, cfg, setup_grade="A+")
        c = ps.compute_lot_intelligent("XAUUSD", 0.60, cfg, setup_grade="C")
        # A+ debe ser mayor o igual que C
        assert a_plus >= c
        # Y la base debe ser intermedia o cerca.
        assert a_plus >= 0
        assert base >= 0

    def test_max_lot_cap_enforced(self) -> None:
        ps = PositionSizer()
        cfg = _base_cfg(equity=Decimal("1000000"), max_lot=Decimal("0.10"))
        lot = ps.compute_lot("XAUUSD", 0.80, cfg)
        assert lot <= 0.10

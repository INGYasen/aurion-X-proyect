"""Tests para risk.chop_guard."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from risk.chop_guard import ChopGuard, ChopGuardConfig


def _utc(h: int, mi: int = 0) -> datetime:
    return datetime(2026, 6, 5, h, mi, tzinfo=timezone.utc)


class TestChopGuardSLMultiplier:
    def test_disabled_returns_1(self) -> None:
        cg = ChopGuard(ChopGuardConfig(enabled=False, asia_sl_multiplier=1.5))
        assert cg.adjusted_sl_multiplier(_utc(2)) == 1.0

    def test_asia_hours_apply_multiplier(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, asia_sl_multiplier=1.5,
            asia_hours_utc=(0, 1, 2, 3, 4, 5, 6, 7),
        ))
        assert cg.adjusted_sl_multiplier(_utc(3)) == 1.5
        assert cg.adjusted_sl_multiplier(_utc(10)) == 1.0  # fuera de asia


class TestChopGuardRangePosition:
    def test_disabled_does_not_block(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=False, enforce_range_position=True,
            long_max_range_position=0.65,
        ))
        blocked, _ = cg.block_by_range_position("BUY", 0.90)
        assert blocked is False

    def test_long_at_top_blocked(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, enforce_range_position=True,
            long_max_range_position=0.65,
        ))
        blocked, msg = cg.block_by_range_position("BUY", 0.90)
        assert blocked is True
        assert "range_position" in msg

    def test_long_at_middle_allowed(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, enforce_range_position=True,
            long_max_range_position=0.65,
        ))
        blocked, _ = cg.block_by_range_position("BUY", 0.40)
        assert blocked is False

    def test_short_at_bottom_blocked(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, enforce_range_position=True,
            short_min_range_position=0.35,
        ))
        blocked, _ = cg.block_by_range_position("SELL", 0.10)
        assert blocked is True

    def test_none_range_position_allowed(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, enforce_range_position=True,
            long_max_range_position=0.65,
        ))
        blocked, _ = cg.block_by_range_position("BUY", None)
        assert blocked is False


class TestChopGuardMicroCooldown:
    def test_disabled_no_cooldown(self) -> None:
        cg = ChopGuard(ChopGuardConfig(enabled=False))
        armed, _ = cg.maybe_arm_micro_cooldown("XAUUSD", [0.0, 0.0, 0.0])
        assert armed is False

    def test_micro_trades_arm_cooldown(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, micro_trade_window=5,
            micro_trade_avg_abs_r_max=0.10,
            micro_trade_cooldown_min=30,
        ))
        # 5 trades con |R| promedio < 0.10
        armed, _ = cg.maybe_arm_micro_cooldown(
            "XAUUSD", [0.05, -0.03, 0.04, 0.02, -0.05], _utc(10),
        )
        assert armed is True

    def test_consecutive_losses_arm_cooldown(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, consecutive_loss_window=3,
            consecutive_loss_min_abs_r=0.5,
            consecutive_loss_cooldown_min=60,
        ))
        # 3 perdidas grandes consecutivas
        armed, _ = cg.maybe_arm_micro_cooldown(
            "XAUUSD", [-0.8, -1.0, -1.2], _utc(10),
        )
        assert armed is True


class TestChopGuardCooldownCheck:
    def test_no_cooldown_no_block(self) -> None:
        cg = ChopGuard(ChopGuardConfig(enabled=True))
        blocked, _ = cg.in_cooldown("XAUUSD", _utc(10))
        assert blocked is False

    def test_active_cooldown_blocks(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, micro_trade_window=3,
            micro_trade_avg_abs_r_max=0.10,
            micro_trade_cooldown_min=30,
        ))
        # Armar cooldown a las 10 UTC con 30min de duracion
        cg.maybe_arm_micro_cooldown(
            "XAUUSD", [0.01, 0.02, 0.03], _utc(10),
        )
        # 5min despues debe seguir en cooldown
        blocked, _ = cg.in_cooldown("XAUUSD", _utc(10, 5))
        assert blocked is True
        # Tras 31min ya no
        blocked2, _ = cg.in_cooldown("XAUUSD", _utc(10, 31))
        assert blocked2 is False


class TestChopGuardMLConf:
    def test_disabled_no_block(self) -> None:
        cg = ChopGuard(ChopGuardConfig(enabled=False, min_ml_conf_scalper=0.60))
        blocked, _ = cg.block_by_ml_conf_scalper("SCALPER", 0.30)
        assert blocked is False

    def test_scalper_low_conf_blocked(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, min_ml_conf_scalper=0.58,
            enforce_ml_conf_scalper=True,
        ))
        blocked, _ = cg.block_by_ml_conf_scalper("SCALPER", 0.45)
        assert blocked is True

    def test_scalper_high_conf_allowed(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, min_ml_conf_scalper=0.58,
            enforce_ml_conf_scalper=True,
        ))
        blocked, _ = cg.block_by_ml_conf_scalper("SCALPER", 0.70)
        assert blocked is False

    def test_intraday_not_affected(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True, min_ml_conf_scalper=0.58,
            enforce_ml_conf_scalper=True,
        ))
        blocked, _ = cg.block_by_ml_conf_scalper("INTRADAY", 0.45)
        assert blocked is False


class TestChopGuardBlacklist:
    def test_blocked_setup_blocked(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True,
            blocked_setups=("RANGE_STRUCTURE_M15_FADE",),
        ))
        blocked, msg = cg.block_by_setup_blacklist("RANGE_STRUCTURE_M15_FADE")
        assert blocked is True
        assert "blacklist" in msg.lower() or "blocked" in msg.lower() or msg

    def test_not_blocked_setup_allowed(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=True,
            blocked_setups=("RANGE_STRUCTURE_M15_FADE",),
        ))
        blocked, _ = cg.block_by_setup_blacklist("BOS_RETEST")
        assert blocked is False

    def test_disabled_no_block(self) -> None:
        cg = ChopGuard(ChopGuardConfig(
            enabled=False,
            blocked_setups=("RANGE_STRUCTURE_M15_FADE",),
        ))
        blocked, _ = cg.block_by_setup_blacklist("RANGE_STRUCTURE_M15_FADE")
        assert blocked is False


class TestChopGuardConfigFromMapping:
    def test_from_mapping_full_shape(self) -> None:
        cfg = ChopGuardConfig.from_mapping({
            "enabled": True,
            "asia_sl_multiplier": 1.6,
            "asia_hours_utc": [0, 1, 2, 3, 4, 5, 6, 7],
            "enforce_range_position": True,
            "long_max_range_position": 0.62,
            "short_min_range_position": 0.38,
            "micro_trade_window": 5,
            "micro_trade_avg_abs_r_max": 0.10,
            "micro_trade_cooldown_min": 30,
            "consecutive_loss_window": 3,
            "consecutive_loss_min_abs_r": 0.5,
            "consecutive_loss_cooldown_min": 60,
            "min_ml_conf_scalper": 0.58,
            "enforce_ml_conf_scalper": True,
            "blocked_setups": ["RANGE_STRUCTURE_M15_FADE"],
        })
        assert cfg.enabled is True
        assert cfg.asia_sl_multiplier == 1.6
        assert 4 in cfg.asia_hours_utc
        assert cfg.long_max_range_position == 0.62
        assert "RANGE_STRUCTURE_M15_FADE" in cfg.blocked_setups

    def test_from_mapping_none_returns_disabled(self) -> None:
        cfg = ChopGuardConfig.from_mapping(None)
        assert cfg.enabled is False

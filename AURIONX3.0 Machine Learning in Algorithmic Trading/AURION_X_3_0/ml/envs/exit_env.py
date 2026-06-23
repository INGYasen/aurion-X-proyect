"""AURION X 4.0 — Gymnasium environment para entrenar el DRL Exit Agent.

Replica offline la vida de un trade abierto en XAUUSD M1, paso a paso. El
agente observa estado en 18 dimensiones y elige una de 5 acciones:

    0 = HOLD             (no hace nada)
    1 = TIGHTEN_SL_10    (apretar SL un 10%)
    2 = MOVE_TO_BE       (mover SL a precio de entrada)
    3 = PARTIAL_CLOSE_40 (cerrar 40% de la posición)
    4 = FULL_CLOSE       (cerrar todo)

Reward (Schulman et al., 2017; Zhang, Zohren & Roberts, 2020):

    r_t = ΔR_t  -  λ · 1[a ≠ HOLD] · 0.02  -  μ · drawdown_t

donde ΔR_t es el cambio en R-múltiplos del MFE, λ = 0.10 penaliza acciones
costosas (slippage simulado), y μ = 0.50 penaliza drawdown intra-trade.

Gymnasium es **opcional**. Si no está instalado el módulo declara stubs y los
tests se saltan automáticamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Sequence

import numpy as np

from analytics.logger import get_logger

logger = get_logger(__name__)

try:
    import gymnasium as gym  # type: ignore[import-not-found]
    from gymnasium import spaces  # type: ignore[import-not-found]
    _HAS_GYM = True
except ImportError:
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _HAS_GYM = False


class ExitAction(IntEnum):
    HOLD = 0
    TIGHTEN_SL_10 = 1
    MOVE_TO_BE = 2
    PARTIAL_CLOSE_40 = 3
    FULL_CLOSE = 4


STATE_DIM = 18


@dataclass
class TradeRecord:
    """Representa un trade cerrado para replay offline."""

    direction: str  # "BUY" | "SELL"
    entry_price: float
    sl_price: float
    tp_price: float
    open_time_idx: int  # índice en m1_bars donde abrió
    close_time_idx: int
    setup_grade: str
    ml_confidence_at_open: float
    vol_regime: str  # LOW | HIGH | EXTREME
    hour_utc: int
    drawdown_at_open_pct: float
    loss_streak_at_open: int
    distance_to_sr_pct: float


def build_state_vector(
    *,
    r_current: float,
    mfe_r: float,
    mae_r: float,
    minutes_in_trade: int,
    atr_norm_momentum: float,
    ema_fast_slow_spread: float,
    volume_z: float,
    spread_norm: float,
    hour_utc: int,
    distance_to_sr_pct: float,
    vol_regime: str,
    setup_grade: str,
    ml_confidence_open: float,
    drawdown_today_pct: float,
    loss_streak: int,
) -> np.ndarray:
    """Construye el vector de estado canónico (18-D).

    Cualquier feature inválida (NaN/inf) se imputa con 0.
    """
    hour_sin = float(np.sin(2 * np.pi * (hour_utc % 24) / 24.0))
    hour_cos = float(np.cos(2 * np.pi * (hour_utc % 24) / 24.0))
    vol_one_hot_low = 1.0 if vol_regime == "LOW" else 0.0
    vol_one_hot_high = 1.0 if vol_regime == "HIGH" else 0.0
    vol_one_hot_extreme = 1.0 if vol_regime == "EXTREME" else 0.0
    grade_score = {"A+": 1.0, "A": 0.85, "B": 0.65, "C": 0.40, "X": 0.10}.get(
        str(setup_grade).upper(), 0.5
    )
    state = np.array(
        [
            float(r_current),
            float(mfe_r),
            float(mae_r),
            float(minutes_in_trade),
            float(atr_norm_momentum),
            float(ema_fast_slow_spread),
            float(volume_z),
            float(spread_norm),
            hour_sin,
            hour_cos,
            float(distance_to_sr_pct),
            vol_one_hot_low,
            vol_one_hot_high,
            vol_one_hot_extreme,
            grade_score,
            float(ml_confidence_open),
            float(drawdown_today_pct),
            float(loss_streak),
        ],
        dtype=np.float32,
    )
    # Saneamiento
    state = np.where(np.isfinite(state), state, 0.0).astype(np.float32)
    assert state.shape == (STATE_DIM,)
    return state


@dataclass
class _StepState:
    """Estado mutable del trade durante el episodio."""

    entry_price: float
    sl_price: float
    tp_price: float
    direction: int   # +1 BUY, -1 SELL
    current_idx: int
    open_idx: int
    close_idx: int
    initial_risk: float
    initial_lots: float = 1.0
    open_lots: float = 1.0
    mfe_price: float = 0.0
    mae_price: float = 0.0
    partial_done: bool = False
    closed: bool = False
    realized_r: float = 0.0


if _HAS_GYM:

    class ExitManagementEnv(gym.Env):
        """Env Gym para entrenamiento offline del agente PPO de salida."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            *,
            trades: Sequence[TradeRecord],
            m1_bars: np.ndarray,  # shape (T, ≥4): close, high, low, [volume]
            atr_per_bar: np.ndarray | None = None,
            reward_action_penalty: float = 0.02,
            reward_action_lambda: float = 0.10,
            reward_drawdown_mu: float = 0.50,
            partial_close_fraction: float = 0.40,
            tighten_fraction: float = 0.10,
            seed: int | None = None,
        ) -> None:
            super().__init__()
            self.trades: list[TradeRecord] = list(trades)
            if not self.trades:
                raise ValueError("ExitManagementEnv: lista de trades vacía.")
            bars = np.asarray(m1_bars, dtype=np.float64)
            if bars.ndim != 2 or bars.shape[1] < 3:
                raise ValueError(f"m1_bars debe ser (T, ≥3); recibido {bars.shape}.")
            self.m1 = bars
            if atr_per_bar is None:
                atr_per_bar = self._compute_atr_proxy(bars)
            self.atr = np.asarray(atr_per_bar, dtype=np.float64)

            self.action_space = spaces.Discrete(len(ExitAction))
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
            )

            self._action_penalty = float(reward_action_penalty)
            self._action_lambda = float(reward_action_lambda)
            self._dd_mu = float(reward_drawdown_mu)
            self._partial_fraction = float(partial_close_fraction)
            self._tighten_fraction = float(tighten_fraction)

            self._rng = np.random.default_rng(seed)
            self._trade_idx = 0
            self._st: _StepState | None = None

        # ---- helpers --------------------------------------------------------

        @staticmethod
        def _compute_atr_proxy(bars: np.ndarray, n: int = 14) -> np.ndarray:
            """ATR proxy basado en true range; sin OHLC abre asume close anterior."""
            T = bars.shape[0]
            tr = np.zeros(T, dtype=np.float64)
            h = bars[:, 1]
            l = bars[:, 2]
            c = bars[:, 0]
            for i in range(1, T):
                tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            atr = np.zeros(T, dtype=np.float64)
            for i in range(1, T):
                window = tr[max(0, i - n + 1) : i + 1]
                atr[i] = float(window.mean()) if window.size else 1e-9
            return np.clip(atr, 1e-9, None)

        def _r_multiple(self, price: float) -> float:
            assert self._st is not None
            st = self._st
            if abs(st.initial_risk) < 1e-12:
                return 0.0
            return st.direction * (price - st.entry_price) / st.initial_risk

        def _build_obs(self) -> np.ndarray:
            assert self._st is not None
            st = self._st
            trade = self.trades[self._trade_idx]
            cur_bar = self.m1[st.current_idx]
            close = float(cur_bar[0])
            high = float(cur_bar[1])
            low = float(cur_bar[2])

            r_cur = self._r_multiple(close)
            # MFE/MAE en R
            mfe_r = st.direction * (st.mfe_price - st.entry_price) / max(st.initial_risk, 1e-12)
            mae_r = st.direction * (st.mae_price - st.entry_price) / max(st.initial_risk, 1e-12)

            atr = float(self.atr[st.current_idx])
            momentum = (close - float(self.m1[max(0, st.current_idx - 5), 0])) / max(atr, 1e-9)

            # EMA proxy con ventana corta de closes recientes
            window = self.m1[max(0, st.current_idx - 21) : st.current_idx + 1, 0]
            ema_fast = float(window[-8:].mean()) if window.size >= 1 else close
            ema_slow = float(window.mean()) if window.size >= 1 else close
            ema_spread = (ema_fast - ema_slow) / max(close, 1e-9)

            vols = self.m1[max(0, st.current_idx - 20) : st.current_idx + 1, 3] if self.m1.shape[1] >= 4 else np.zeros(1)
            vol_mu = float(vols.mean())
            vol_sd = float(vols.std(ddof=0)) or 1.0
            vol_z = (float(vols[-1]) - vol_mu) / vol_sd if vols.size > 1 else 0.0

            spread_norm = (high - low) / max(atr, 1e-9)
            minutes_in_trade = st.current_idx - st.open_idx

            return build_state_vector(
                r_current=r_cur,
                mfe_r=mfe_r,
                mae_r=mae_r,
                minutes_in_trade=minutes_in_trade,
                atr_norm_momentum=momentum,
                ema_fast_slow_spread=ema_spread,
                volume_z=vol_z,
                spread_norm=spread_norm,
                hour_utc=trade.hour_utc,
                distance_to_sr_pct=trade.distance_to_sr_pct,
                vol_regime=trade.vol_regime,
                setup_grade=trade.setup_grade,
                ml_confidence_open=trade.ml_confidence_at_open,
                drawdown_today_pct=trade.drawdown_at_open_pct,
                loss_streak=trade.loss_streak_at_open,
            )

        def _check_sl_tp_hit(self) -> tuple[bool, float]:
            """Comprueba si SL o TP fueron tocados en la vela actual."""
            assert self._st is not None
            st = self._st
            bar = self.m1[st.current_idx]
            high = float(bar[1])
            low = float(bar[2])

            if st.direction > 0:  # BUY
                if low <= st.sl_price:
                    return True, st.sl_price
                if high >= st.tp_price:
                    return True, st.tp_price
            else:  # SELL
                if high >= st.sl_price:
                    return True, st.sl_price
                if low <= st.tp_price:
                    return True, st.tp_price
            return False, 0.0

        def _apply_action(self, action: int) -> float:
            """Aplica la acción y devuelve la penalización si la hay."""
            assert self._st is not None
            st = self._st
            act = ExitAction(action)
            close = float(self.m1[st.current_idx, 0])

            if act == ExitAction.HOLD:
                return 0.0

            penalty = self._action_lambda * self._action_penalty

            if act == ExitAction.TIGHTEN_SL_10:
                # Acerca SL un 10% de la distancia restante hacia el precio actual.
                delta = close - st.sl_price
                st.sl_price += delta * self._tighten_fraction
            elif act == ExitAction.MOVE_TO_BE:
                # Solo si está en profit; si no, no hace nada
                if self._r_multiple(close) > 0:
                    st.sl_price = st.entry_price
            elif act == ExitAction.PARTIAL_CLOSE_40:
                if not st.partial_done and st.open_lots > 0:
                    lots_to_close = st.open_lots * self._partial_fraction
                    st.realized_r += self._r_multiple(close) * lots_to_close
                    st.open_lots -= lots_to_close
                    st.partial_done = True
            elif act == ExitAction.FULL_CLOSE:
                if st.open_lots > 0:
                    st.realized_r += self._r_multiple(close) * st.open_lots
                    st.open_lots = 0.0
                    st.closed = True
            return penalty

        # ---- gym API --------------------------------------------------------

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            # Selección aleatoria del próximo trade
            self._trade_idx = int(self._rng.integers(0, len(self.trades)))
            trade = self.trades[self._trade_idx]
            direction = 1 if trade.direction.upper() == "BUY" else -1
            initial_risk = abs(trade.entry_price - trade.sl_price)
            if initial_risk < 1e-12:
                initial_risk = max(1e-9, abs(trade.entry_price) * 0.0001)
            self._st = _StepState(
                entry_price=float(trade.entry_price),
                sl_price=float(trade.sl_price),
                tp_price=float(trade.tp_price),
                direction=direction,
                current_idx=int(trade.open_time_idx),
                open_idx=int(trade.open_time_idx),
                close_idx=int(trade.close_time_idx),
                initial_risk=float(initial_risk),
                mfe_price=float(trade.entry_price),
                mae_price=float(trade.entry_price),
            )
            obs = self._build_obs()
            return obs, {"trade_idx": self._trade_idx}

        def step(self, action: int):
            assert self._st is not None, "Llama reset() antes de step()."
            st = self._st

            penalty = self._apply_action(int(action))
            prev_r = self._r_multiple(float(self.m1[st.current_idx, 0]))

            # Avance temporal
            st.current_idx += 1

            terminated = False
            truncated = False

            if st.current_idx >= len(self.m1) or st.current_idx > st.close_idx:
                terminated = True
                # Cierre forzoso a último close
                last_idx = min(st.current_idx, len(self.m1) - 1)
                last_close = float(self.m1[last_idx, 0])
                if st.open_lots > 0:
                    st.realized_r += self._r_multiple(last_close) * st.open_lots
                    st.open_lots = 0.0
                st.closed = True
            else:
                bar = self.m1[st.current_idx]
                high = float(bar[1])
                low = float(bar[2])
                # Actualiza MFE/MAE
                st.mfe_price = max(st.mfe_price, high) if st.direction > 0 else min(st.mfe_price, low)
                st.mae_price = min(st.mae_price, low) if st.direction > 0 else max(st.mae_price, high)

                hit, fill = self._check_sl_tp_hit()
                if hit:
                    terminated = True
                    st.realized_r += self._r_multiple(float(fill)) * st.open_lots
                    st.open_lots = 0.0
                    st.closed = True

            if st.closed:
                obs = self._build_obs()
                # Reward terminal = delta_r del último tick (no realized_r entero,
                # que ya fue acumulado tick-a-tick via reward shaping). Esto evita
                # el doble-conteo que inflaba episode_reward x2.
                last_idx = min(st.current_idx, len(self.m1) - 1)
                cur_r_final = self._r_multiple(float(self.m1[last_idx, 0]))
                delta_final = cur_r_final - prev_r
                reward = float(delta_final - penalty)
                return obs, reward, terminated, truncated, {"realized_r": st.realized_r}

            obs = self._build_obs()
            cur_r = self._r_multiple(float(self.m1[st.current_idx, 0]))
            delta_r = cur_r - prev_r
            drawdown_pct = max(0.0, -cur_r)
            reward = float(delta_r - penalty - self._dd_mu * 0.01 * drawdown_pct)
            return obs, reward, terminated, truncated, {}

else:

    class ExitManagementEnv:  # type: ignore[no-redef]
        """Stub cuando Gymnasium no está instalado."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "ExitManagementEnv requiere gymnasium. Instala dependencias v4 con:\n"
                "  pip install -e .[v4]"
            )

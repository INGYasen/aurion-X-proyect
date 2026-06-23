"""Microestructura de mercado: velocidad, impulsos, absorción, manipulación."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MicrostructureSnapshot:
    micro_velocity: float
    tick_acceleration: float
    impulse_decay: float
    rejection_velocity: float
    absorption_pressure: float
    directional_exhaustion: float
    liquidity_reaction_speed: float
    spoofing_probability: float
    institutional_velocity: float
    intention_loss: float
    manipulation_score: float
    quality: float

    def as_dict(self) -> dict[str, float]:
        return {
            "micro_velocity": self.micro_velocity,
            "tick_acceleration": self.tick_acceleration,
            "impulse_decay": self.impulse_decay,
            "rejection_velocity": self.rejection_velocity,
            "absorption_pressure": self.absorption_pressure,
            "directional_exhaustion": self.directional_exhaustion,
            "liquidity_reaction_speed": self.liquidity_reaction_speed,
            "spoofing_probability": self.spoofing_probability,
            "institutional_velocity": self.institutional_velocity,
            "intention_loss": self.intention_loss,
            "manipulation_score": self.manipulation_score,
            "micro_quality": self.quality,
        }


class MicrostructureEngine:
    """Analiza ticks/velas M1 para micro-comportamiento."""

    def analyze(
        self,
        *,
        closes: Sequence[float],
        highs: Sequence[float] | None = None,
        lows: Sequence[float] | None = None,
        volumes: Sequence[float] | None = None,
        spread_points: float = 0.0,
        direction: str = "BUY",
    ) -> MicrostructureSnapshot:
        c = list(closes)
        n = len(c)
        h = list(highs) if highs else c
        l = list(lows) if lows else c
        v = list(volumes) if volumes else [1.0] * n

        if n < 4:
            return MicrostructureSnapshot(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        rets = [(c[i] - c[i - 1]) / max(abs(c[i - 1]), 1e-12) for i in range(1, n)]
        micro_vel = abs(rets[-1]) * 1000.0
        accel = (rets[-1] - rets[-2]) if len(rets) >= 2 else 0.0
        tick_accel = min(1.0, abs(accel) * 500.0)

        impulse = max(abs(r) for r in rets[-4:]) if len(rets) >= 4 else abs(rets[-1])
        decay = 1.0 - (abs(rets[-1]) / max(impulse, 1e-12)) if impulse > 0 else 0.0
        impulse_decay = max(0.0, min(1.0, decay))

        rng = max(h[-1] - l[-1], 1e-12)
        body = abs(c[-1] - c[-2])
        wick_upper = h[-1] - max(c[-1], c[-2])
        wick_lower = min(c[-1], c[-2]) - l[-1]
        rejection = (wick_upper + wick_lower) / rng
        rejection_velocity = min(1.0, rejection * (1.0 + abs(accel) * 10.0))

        vol_recent = v[-1] if v else 1.0
        vol_base = sum(v[-min(8, n) :]) / max(min(8, n), 1)
        vol_spike = vol_recent / max(vol_base, 1e-9)
        absorption = min(1.0, body / rng * vol_spike * 0.5)

        d = direction.upper()
        against = 0
        for i in range(max(1, n - 4), n):
            if d == "BUY" and c[i] < c[i - 1]:
                against += 1
            elif d == "SELL" and c[i] > c[i - 1]:
                against += 1
        exhaustion = min(1.0, against / 3.0)

        liq_speed = min(1.0, vol_spike / 2.0)

        wick_dom = rejection > 0.65 and body / rng < 0.25
        spoof = min(1.0, (0.4 if wick_dom else 0.0) + (0.3 if vol_spike > 2.5 and abs(rets[-1]) < impulse * 0.3 else 0.0))

        inst_vel = min(1.0, micro_vel * 0.4 + liq_speed * 0.35 + (1.0 - impulse_decay) * 0.25)

        aligned = sum(
            1 for i in range(max(1, n - 3), n)
            if (d == "BUY" and c[i] > c[i - 1]) or (d == "SELL" and c[i] < c[i - 1])
        )
        intention_loss = 1.0 - aligned / 3.0

        manip = min(
            1.0,
            spoof * 0.35 + rejection_velocity * 0.25 + exhaustion * 0.2 + (spread_points / 200.0) * 0.2,
        )

        quality = max(
            0.0,
            min(
                1.0,
                inst_vel * 0.35
                + (1.0 - manip) * 0.30
                + (1.0 - exhaustion) * 0.20
                + liq_speed * 0.15,
            ),
        )

        return MicrostructureSnapshot(
            micro_velocity=float(micro_vel),
            tick_acceleration=float(tick_accel),
            impulse_decay=float(impulse_decay),
            rejection_velocity=float(rejection_velocity),
            absorption_pressure=float(absorption),
            directional_exhaustion=float(exhaustion),
            liquidity_reaction_speed=float(liq_speed),
            spoofing_probability=float(spoof),
            institutional_velocity=float(inst_vel),
            intention_loss=float(intention_loss),
            manipulation_score=float(manip),
            quality=float(quality),
        )

    def analyze_market_state(self, market_state: Mapping[str, Any], *, direction: str = "BUY") -> MicrostructureSnapshot:
        return self.analyze(
            closes=market_state.get("closes_m1") or [],
            highs=market_state.get("highs_m1"),
            lows=market_state.get("lows_m1"),
            volumes=[market_state.get("tick_volume_proxy", 0.0)] * max(len(market_state.get("closes_m1") or []), 1),
            spread_points=float(market_state.get("spread_points", 0.0)),
            direction=direction,
        )

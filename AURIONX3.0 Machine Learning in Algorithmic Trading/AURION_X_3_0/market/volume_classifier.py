"""Volume Classifier (AURION X 4.6) — clasificación discreta de volumen.

Compara el volumen actual contra promedios MA(20) y MA(50) y devuelve una
clasificación discreta:

    LOW       : current < 0.50 × MA20
    NORMAL    : 0.50 ≤ current < 1.50 × MA20
    HIGH      : 1.50 ≤ current < 2.50 × MA20
    EXTREME   : current ≥ 2.50 × MA20

Más un delta de score sugerido para integrar al SMCDecisionEngine:

    LOW + dirección momentum  → -8
    HIGH                      → +6
    EXTREME                   → +12
    NORMAL                    → 0

También evalúa "BOS sin volumen" → penaliza fuerte; "BOS con volumen
extremo" → bonifica.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class VolumeClass(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VolumeAssessment:
    """Resultado del análisis de volumen."""
    classification: VolumeClass
    current: float
    ma20: float
    ma50: float
    ratio_ma20: float        # current / ma20
    ratio_ma50: float        # current / ma50
    score_delta: float       # sugerencia para sumar/restar al smc_score
    bos_modifier: float      # delta extra si hay BOS reciente
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "current": float(self.current),
            "ma20": float(self.ma20),
            "ma50": float(self.ma50),
            "ratio_ma20": float(self.ratio_ma20),
            "ratio_ma50": float(self.ratio_ma50),
            "score_delta": float(self.score_delta),
            "bos_modifier": float(self.bos_modifier),
            "notes": list(self.notes),
        }


# ─── helpers ────────────────────────────────────────────────────────────


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _vol_of(c: Any) -> float | None:
    """Extrae volumen de una vela (tick_volume preferido, real_volume fallback)."""
    for attr in ("tick_volume", "tickvol", "volume", "real_volume"):
        v = getattr(c, attr, None)
        if v is not None:
            try:
                vf = float(v)
                if vf > 0:
                    return vf
            except (TypeError, ValueError):
                continue
    return None


# ─── api publica ─────────────────────────────────────────────────────────


def classify_volume(
    *,
    candles: Iterable[Any],
    n_recent: int = 1,
    has_recent_bos: bool = False,
    has_recent_choch: bool = False,
) -> VolumeAssessment:
    """Clasifica el volumen actual y devuelve delta sugerido para el score.

    Args:
        candles: iterable de velas ordenadas ascendente (más antigua primero).
        n_recent: cuántas velas finales considerar como "actual" (default 1).
        has_recent_bos: si hubo BOS en últimas N velas (modificador).
        has_recent_choch: idem CHOCH.

    Returns:
        VolumeAssessment con clase + ratios + delta sugerido.
    """
    vols: list[float] = []
    for c in candles or []:
        v = _vol_of(c)
        if v is not None:
            vols.append(v)
    if len(vols) < 20:
        return VolumeAssessment(
            classification=VolumeClass.UNKNOWN,
            current=0.0, ma20=0.0, ma50=0.0,
            ratio_ma20=0.0, ratio_ma50=0.0,
            score_delta=0.0, bos_modifier=0.0,
            notes=[f"insufficient_data (n={len(vols)})"],
        )

    n_recent = max(1, int(n_recent))
    current = _avg(vols[-n_recent:])
    ma20 = _avg(vols[-20:])
    ma50 = _avg(vols[-50:]) if len(vols) >= 50 else _avg(vols[-len(vols):])
    r20 = current / ma20 if ma20 > 0 else 0.0
    r50 = current / ma50 if ma50 > 0 else 0.0

    if r20 < 0.50:
        cls = VolumeClass.LOW
        score_delta = -8.0
        note = "volumen_bajo"
    elif r20 < 1.50:
        cls = VolumeClass.NORMAL
        score_delta = 0.0
        note = "volumen_normal"
    elif r20 < 2.50:
        cls = VolumeClass.HIGH
        score_delta = +6.0
        note = "volumen_alto"
    else:
        cls = VolumeClass.EXTREME
        score_delta = +12.0
        note = "volumen_extremo"

    # BOS modifier: BOS sin volumen es señal débil; BOS con volumen extremo es muy fuerte
    bos_modifier = 0.0
    if has_recent_bos:
        if cls == VolumeClass.LOW:
            bos_modifier = -10.0
        elif cls == VolumeClass.EXTREME:
            bos_modifier = +8.0
        elif cls == VolumeClass.HIGH:
            bos_modifier = +4.0
    if has_recent_choch:
        if cls == VolumeClass.LOW:
            bos_modifier += -6.0
        elif cls == VolumeClass.EXTREME:
            bos_modifier += +5.0

    notes = [note, f"r20={r20:.2f}", f"r50={r50:.2f}"]
    if has_recent_bos:
        notes.append("bos_active")
    if has_recent_choch:
        notes.append("choch_active")

    return VolumeAssessment(
        classification=cls, current=current, ma20=ma20, ma50=ma50,
        ratio_ma20=r20, ratio_ma50=r50,
        score_delta=score_delta, bos_modifier=bos_modifier,
        notes=notes,
    )

"""SMC Decision Engine (AURION X 4.5) — capa institucional encima del sistema.

Filtro institucional SMC que actua ANTES de Institutional/Apex/Execution.
NO reemplaza ningun motor existente; los amplifica con un veredicto basado
en estructura pura (BOS / CHOCH / OB / FVG / Liquidity Sweeps / MTF / P-D).

Filosofia:

  * No mas indicadores. **Mejores decisiones**.
  * Filtra setups mediocres y concentra entradas en estructuras fuertes.
  * Si el SMC score < ``min_score_to_trade`` (default 70), bloquea aunque
    el ML diga BUY/SELL.
  * Compatible 100% con Apex/Institutional/Quant/Auto-retrain/Ensemble.

Salida principal: ``SMCDecisionEngine.evaluate(...) -> SMCDecision``.

El motor es defensivo: cualquier excepcion o snapshot vacio devuelve
``allow_trade=True`` con score 0 y grade ``X`` (no bloquea por defecto si
no esta habilitado en config; si lo esta, ``X`` si bloqueara salvo override
de ML > ``c_grade_ml_override`` para grade C).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from market.premium_discount_engine import (
    PremiumDiscountEngine,
    PremiumDiscountResult,
    PremiumDiscountState,
)
from ml.smc_heuristics import (
    HeuristicResult,
    evaluate_heuristics,
    heuristics_from_config,
)

logger = logging.getLogger("ml.smc_decision_engine")


# ─── Resultados estructurados ───────────────────────────────────────────


@dataclass(frozen=True)
class OBQuality:
    """Calidad de Order Block A+/A/B/C/X."""
    grade: str            # "A+", "A", "B", "C", "X"
    score: float          # [0, 1]
    fresh: bool
    institutional: bool   # OB en HTF + impulso fuerte + freshness alta
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "score": float(self.score),
            "fresh": bool(self.fresh),
            "institutional": bool(self.institutional),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class FVGQuality:
    """Calidad de FVG A+/A/B/C."""
    grade: str            # "A+", "A", "B", "C"
    score: float          # [0, 1]
    aligned: bool         # FVG en la direccion del trade
    htf: bool             # FVG en H1/H4/D1/W1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "score": float(self.score),
            "aligned": bool(self.aligned),
            "htf": bool(self.htf),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MTFAlignment:
    """Resultado del MTF check (H4, H1, M30, M15)."""
    full_aligned: bool
    htf_contradicts: bool   # H4 contradice direccion → penalizacion fuerte
    per_tf: dict[str, str]  # {tf: "BULLISH"/"BEARISH"/"NEUTRAL"}
    aligned_count: int      # cuantos TF apuntan a la direccion

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_aligned": bool(self.full_aligned),
            "htf_contradicts": bool(self.htf_contradicts),
            "per_tf": dict(self.per_tf),
            "aligned_count": int(self.aligned_count),
        }


@dataclass(frozen=True)
class SMCDecision:
    """Veredicto final del SMC Decision Engine."""
    allow_trade: bool
    block_reason: str          # vacio si allow
    smc_score: float           # 0..100
    setup_grade: str           # "A+", "A", "B", "C", "X"
    ob_quality: OBQuality
    fvg_quality: FVGQuality
    pd_state: PremiumDiscountResult
    sweep_detected: bool       # True si hay SSL o BSL barrido reciente
    sweep_kind: str            # "SSL" | "BSL" | "BOTH" | "NONE"
    mtf: MTFAlignment
    components: dict[str, float]  # detalle de puntos por categoria
    lot_multiplier: float      # 0.0 (X prohibido) / 0.3 (C) / 0.5 (B) / 1.0 (A) / 1.2 (A+)
    aggressive_execution: bool # True si A+ → permite agresividad maxima
    notes: list[str]
    heuristics: HeuristicResult | None = None  # detalle reglas institucionales
    setup_status: str = "NO_TRADE"  # NO_TRADE | WATCHLIST | VALID | PREMIUM
    quality_mode: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_trade": bool(self.allow_trade),
            "block_reason": self.block_reason,
            "smc_score": float(self.smc_score),
            "setup_grade": self.setup_grade,
            "setup_status": self.setup_status,
            "quality_mode": self.quality_mode,
            "ob_quality": self.ob_quality.to_dict(),
            "fvg_quality": self.fvg_quality.to_dict(),
            "pd_state": self.pd_state.to_dict(),
            "sweep_detected": bool(self.sweep_detected),
            "sweep_kind": self.sweep_kind,
            "mtf": self.mtf.to_dict(),
            "components": {k: float(v) for k, v in self.components.items()},
            "lot_multiplier": float(self.lot_multiplier),
            "aggressive_execution": bool(self.aggressive_execution),
            "notes": list(self.notes),
            "heuristics": self.heuristics.to_dict() if self.heuristics else None,
        }


# ─── Constantes (puntos por componente, segun spec) ─────────────────────

# Maximos por categoria (no apilan dentro de una misma categoria).
_PTS_SWEEP_SSL = 20.0
_PTS_SWEEP_BSL = 20.0
_PTS_CHOCH_VALID = 15.0
_PTS_CHOCH_HTF = 20.0   # excluye CHOCH valido (toma el mejor)
_PTS_BOS_VALID = 20.0
_PTS_BOS_HTF = 25.0     # excluye BOS valido
_PTS_OB_FRESH = 25.0
_PTS_OB_MIT = 10.0
_PTS_OB_INST = 30.0     # excluye fresh/mitigated
_PTS_FVG_ALIGNED = 10.0
_PTS_FVG_HTF = 15.0     # excluye FVG aligned
_PTS_CONFL_CB = 10.0    # CHOCH+BOS
_PTS_CONFL_CBO = 20.0   # CHOCH+BOS+OB
_PTS_CONFL_CBOF = 25.0  # CHOCH+BOS+OB+FVG

# Suma maxima teorica con seleccion optima:
#   sweep(40) + choch_htf(20) + bos_htf(25) + ob_inst(30) + fvg_htf(15)
#   + confluencia_full(25) = 155
# Normalizamos linealmente a [0,100] dividiendo por 1.55 SOLO en el cap final.
_THEORETICAL_MAX = 155.0

_HTF_SET = {"H1", "H4", "D1", "W1", "MN1"}


# ─── Engine ──────────────────────────────────────────────────────────────


class SMCDecisionEngine:
    """Filtro institucional SMC. Calcula score 0-100 y emite veredicto.

    Uso:
        engine = SMCDecisionEngine.from_config(cfg)
        dec = engine.evaluate(
            symbol="XAUUSD",
            direction="BUY",
            snapshot=smc_snapshot,
            ml_confidence=0.62,
            mtf_trends={"H4":"BULLISH",...},
            last_close=4060.5,
            primary_tf="M15",
        )
        if not dec.allow_trade:
            log_skip(dec.block_reason); return
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_score_to_trade: float = 70.0,
        require_sweep: bool = False,
        require_mtf_full_align: bool = False,
        c_grade_ml_override: float = 0.80,
        grade_lot_multipliers: Mapping[str, float] | None = None,
        pd_engine: PremiumDiscountEngine | None = None,
        # Pesos calibrables (defaults segun spec)
        weights: Mapping[str, float] | None = None,
        # Heuristicas institucionales (AURION X 4.5.2)
        heuristics_enabled: bool = True,
        heuristics_weight: float = 1.0,
        heuristics_rules: set[str] | None = None,
        # AURION X 4.6 — Quality Mode (filtro PREMIUM unificado seccion 14)
        quality_mode: str = "normal",  # "normal" | "high_quality" | "premium_only"
        min_rr_high_quality: float = 2.5,
        min_rr_premium: float = 3.0,
    ) -> None:
        self._enabled = bool(enabled)
        self._min_score = float(max(0.0, min(100.0, min_score_to_trade)))
        self._require_sweep = bool(require_sweep)
        self._require_mtf = bool(require_mtf_full_align)
        self._c_grade_ml_override = float(c_grade_ml_override)
        self._lot_mults: dict[str, float] = {
            "A+": 1.2, "A": 1.0, "B": 0.5, "C": 0.3, "X": 0.0,
        }
        if grade_lot_multipliers:
            for k, v in grade_lot_multipliers.items():
                self._lot_mults[str(k).upper()] = float(v)
        self._pd_engine = pd_engine or PremiumDiscountEngine()
        self._w = dict(weights or {})
        self._heur_enabled = bool(heuristics_enabled)
        self._heur_weight = float(heuristics_weight)
        self._heur_rules = heuristics_rules
        self._quality_mode = str(quality_mode or "normal").lower()
        if self._quality_mode not in ("normal", "high_quality", "premium_only"):
            self._quality_mode = "normal"
        self._min_rr_hq = float(min_rr_high_quality)
        self._min_rr_pr = float(min_rr_premium)

    # ─── API publica ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate(
        self,
        *,
        symbol: str,
        direction: str,
        snapshot: Any | None,
        ml_confidence: float | None,
        mtf_trends: Mapping[str, Any] | None,
        last_close: float,
        primary_tf: str = "M15",
        ts_utc: datetime | None = None,
        in_post_event_window: bool = False,
        # AURION X 4.6 — contexto extra para quality_mode + score adjustment
        volume_class: str | None = None,
        volume_score_delta: float = 0.0,
        rr_estimated: float | None = None,
        news_blackout: bool = False,
        session_quality: float | None = None,
    ) -> SMCDecision:
        """Calcula score, grade y allow/deny. Defensivo: nunca lanza."""
        try:
            d = (direction or "").strip().upper()
            if d in ("LONG",):
                d = "BUY"
            elif d in ("SHORT",):
                d = "SELL"

            if not self._enabled:
                return self._noop_decision(
                    "engine_disabled", smc_score=0.0, grade="A",
                    allow=True, ml=ml_confidence,
                )

            if snapshot is None:
                return self._noop_decision(
                    "no_snapshot", smc_score=0.0, grade="X",
                    allow=False, ml=ml_confidence,
                )

            # 1) Premium / Discount
            pd_res = self._pd_engine.calculate(snapshot=snapshot, last_close=last_close)
            pd_delta, pd_note = self._pd_engine.evaluate_direction_bias(
                direction=d, pd_result=pd_res,
            )

            # 2) Componentes SMC (puntos brutos)
            comps: dict[str, float] = {}

            # 2.1) Liquidity sweep
            swept, sweep_kind, sweep_pts = self._score_sweep(snapshot, d)
            comps["sweep"] = sweep_pts

            # 2.2) CHOCH (best en cualquier TF; HTF excluye Valid)
            choch_pts, choch_notes = self._score_choch(snapshot, d)
            comps["choch"] = choch_pts

            # 2.3) BOS (best en cualquier TF; HTF excluye Valid)
            bos_pts, bos_notes = self._score_bos(snapshot, d)
            comps["bos"] = bos_pts

            # 2.4) OB best + calidad detallada
            ob_q = self._evaluate_ob_quality(snapshot, d)
            comps["ob"] = self._ob_points(ob_q)

            # 2.5) FVG best + calidad detallada
            fvg_q = self._evaluate_fvg_quality(snapshot, d)
            comps["fvg"] = self._fvg_points(fvg_q)

            # 2.6) Confluencia (combos)
            confl_pts = self._score_confluence(
                has_choch=choch_pts > 0,
                has_bos=bos_pts > 0,
                has_ob=ob_q.grade not in ("X",),
                has_fvg=fvg_q.grade not in ("X",),  # FVG no usa X pero por consistencia
            )
            comps["confluence"] = confl_pts

            # 2.7) MTF check
            mtf = self._evaluate_mtf(mtf_trends, d)

            # 3) Suma bruta y normalizacion a 0-100
            raw = sum(comps.values())
            score_norm = float(min(100.0, max(0.0, raw * 100.0 / _THEORETICAL_MAX)))

            # 4) Ajustes (P/D + MTF)
            score_after_pd = float(max(0.0, min(100.0, score_norm + pd_delta)))
            mtf_delta = 0.0
            if mtf.full_aligned:
                mtf_delta = +6.0
            elif mtf.htf_contradicts:
                mtf_delta = -20.0
            elif mtf.aligned_count >= 2:
                mtf_delta = +3.0
            score_after_mtf = float(max(0.0, min(100.0, score_after_pd + mtf_delta)))

            # 4.5) Heuristicas institucionales (AURION X 4.5.2)
            heur_result: HeuristicResult | None = None
            heur_delta = 0.0
            if self._heur_enabled:
                heur_result = evaluate_heuristics(
                    ts=ts_utc,
                    direction=d,
                    last_close=float(last_close or 0.0),
                    pd_pct=float(pd_res.pct_in_range),
                    mtf_aligned=int(mtf.aligned_count),
                    sweep_detected=bool(swept),
                    has_choch=bool(choch_pts > 0),
                    snapshot=snapshot,
                    in_post_event_window=bool(in_post_event_window),
                    enabled_rules=self._heur_rules,
                    weight=self._heur_weight,
                )
                heur_delta = float(heur_result.delta_total)

            # 4.6) Volumen (AURION X 4.6): clasificacion discreta + delta
            vol_delta = float(volume_score_delta or 0.0)
            score_final = float(max(0.0, min(100.0,
                score_after_mtf + heur_delta + vol_delta)))

            # 5) Grade
            grade = self._setup_grade(
                smc_score=score_final, ml=ml_confidence,
                ob_grade=ob_q.grade, fvg_grade=fvg_q.grade, mtf=mtf, pd=pd_res,
            )

            # 6) Reglas de ejecucion (con contexto extra para quality_mode)
            allow, block_reason = self._enforce_rules(
                smc_score=score_final, grade=grade, ml=ml_confidence,
                sweep_detected=swept, mtf=mtf, pd=pd_res,
                ob_grade=ob_q.grade, fvg_grade=fvg_q.grade,
                volume_class=volume_class, rr_estimated=rr_estimated,
                news_blackout=news_blackout,
            )
            status = self._setup_status(grade=grade, allow=allow)

            lot_mult = float(self._lot_mults.get(grade, 0.0))
            aggressive = grade == "A+"
            notes = [
                pd_note,
                f"raw={raw:.1f}",
                f"norm={score_norm:.1f}",
                f"pd_delta={pd_delta:+.1f}",
                f"mtf_delta={mtf_delta:+.1f}",
                f"heur_delta={heur_delta:+.1f}",
                f"vol_delta={vol_delta:+.1f}",
                f"qmode={self._quality_mode}",
                f"status={status}",
                *choch_notes,
                *bos_notes,
            ]
            if heur_result is not None and heur_result.applied_rules:
                rule_ids = [r["id"] for r in heur_result.applied_rules]
                notes.append("heur_rules=" + ",".join(rule_ids))
            return SMCDecision(
                allow_trade=allow,
                block_reason=block_reason,
                smc_score=score_final,
                setup_grade=grade,
                ob_quality=ob_q,
                fvg_quality=fvg_q,
                pd_state=pd_res,
                sweep_detected=swept,
                sweep_kind=sweep_kind,
                mtf=mtf,
                components=comps,
                lot_multiplier=lot_mult,
                aggressive_execution=aggressive,
                notes=notes,
                heuristics=heur_result,
                setup_status=status,
                quality_mode=self._quality_mode,
            )

        except Exception as exc:
            logger.debug("[SMCDecision] evaluate fallo: %s", exc, exc_info=True)
            return self._noop_decision(
                f"exception:{type(exc).__name__}",
                smc_score=0.0, grade="A", allow=True, ml=ml_confidence,
            )

    # ─── Scoring por componente ─────────────────────────────────────────

    def _score_sweep(
        self, snapshot: Any, direction: str,
    ) -> tuple[bool, str, float]:
        """Detecta si hay SSL/BSL barridos recientes. +20 cada uno (max 40).

        Para BUY: nos beneficia que SSL haya sido barrido recientemente
                  (stop-hunt de longs liquidados = inducement bajista a comprar).
        Para SELL: nos beneficia BSL barrido (stop-hunt de shorts).
        Si ningun lado: 0 puntos (y si require_sweep esta on, allow=False).
        """
        per_tf = getattr(snapshot, "per_tf", None) or {}
        ssl_swept = False
        bsl_swept = False
        for ctx in per_tf.values():
            # 1) liquidity_pools con swept=True
            for pool in (getattr(ctx, "liquidity_pools", None) or []):
                if not getattr(pool, "swept", False):
                    continue
                kind_obj = getattr(pool, "kind", None)
                kind = (kind_obj.value if hasattr(kind_obj, "value") else str(kind_obj or "")).upper()
                if "SSL" in kind or "LOW" in kind:
                    ssl_swept = True
                elif "BSL" in kind or "HIGH" in kind:
                    bsl_swept = True
            # 2) recent_sweeps explicitos
            for sw in (getattr(ctx, "recent_sweeps", None) or []):
                kind_obj = getattr(sw, "kind", None)
                kind = (kind_obj.value if hasattr(kind_obj, "value") else str(kind_obj or "")).upper()
                if "DOWN" in kind or "SSL" in kind:
                    ssl_swept = True
                elif "UP" in kind or "BSL" in kind:
                    bsl_swept = True

        pts = 0.0
        if ssl_swept:
            pts += _PTS_SWEEP_SSL
        if bsl_swept:
            pts += _PTS_SWEEP_BSL

        if ssl_swept and bsl_swept:
            kind = "BOTH"
        elif ssl_swept:
            kind = "SSL"
        elif bsl_swept:
            kind = "BSL"
        else:
            kind = "NONE"

        # Direccion-aware: si los sweeps NO favorecen la direccion, descuenta.
        d = direction.upper()
        if pts > 0 and kind != "BOTH":
            if d == "BUY" and kind == "BSL":
                pts *= 0.4   # BSL barrido contradice un BUY (suelen seguir shorts)
            elif d == "SELL" and kind == "SSL":
                pts *= 0.4

        return bool(pts > 0), kind, float(pts)

    def _score_choch(
        self, snapshot: Any, direction: str,
    ) -> tuple[float, list[str]]:
        """Devuelve (pts, notes). HTF excluye Valid (toma el max)."""
        per_tf = getattr(snapshot, "per_tf", None) or {}
        notes: list[str] = []
        max_pts = 0.0
        for tf, ctx in per_tf.items():
            choch = getattr(ctx, "last_choch", None)
            if choch is None:
                continue
            bias_dir = self._dir_value(getattr(choch, "direction", None))
            if not self._bias_matches(direction, bias_dir):
                continue
            is_htf = str(tf).upper() in _HTF_SET
            pts = _PTS_CHOCH_HTF if is_htf else _PTS_CHOCH_VALID
            if pts > max_pts:
                max_pts = pts
                notes = [f"choch:{tf}:{'HTF' if is_htf else 'V'}"]
        return float(max_pts), notes

    def _score_bos(
        self, snapshot: Any, direction: str,
    ) -> tuple[float, list[str]]:
        per_tf = getattr(snapshot, "per_tf", None) or {}
        notes: list[str] = []
        max_pts = 0.0
        for tf, ctx in per_tf.items():
            bos = getattr(ctx, "last_bos", None)
            if bos is None:
                continue
            bias_dir = self._dir_value(getattr(bos, "direction", None))
            if not self._bias_matches(direction, bias_dir):
                continue
            is_htf = str(tf).upper() in _HTF_SET
            pts = _PTS_BOS_HTF if is_htf else _PTS_BOS_VALID
            if pts > max_pts:
                max_pts = pts
                notes = [f"bos:{tf}:{'HTF' if is_htf else 'V'}"]
        return float(max_pts), notes

    def _evaluate_ob_quality(self, snapshot: Any, direction: str) -> OBQuality:
        """Clasifica el mejor OB alineado a la direccion → A+/A/B/C/X.

        Combina: ob_strength * 0.30 + volume * 0.20 + displacement * 0.20
                 + reaction * 0.15 + freshness * 0.15 (todos en [0,1]).
        HTF + state=FRESH + score >= 0.85 → ``institutional=True``.
        """
        per_tf = getattr(snapshot, "per_tf", None) or {}
        best_score = -1.0
        best_grade = "X"
        best_fresh = False
        best_inst = False
        best_notes: list[str] = []
        for tf, ctx in per_tf.items():
            obs = getattr(ctx, "active_obs", None) or []
            for ob in obs:
                bias_dir = self._dir_value(getattr(ob, "direction", None))
                if not self._bias_matches(direction, bias_dir):
                    continue
                strength = self._clamp01(getattr(ob, "ob_strength", 0.0))
                volume = self._clamp01(getattr(ob, "ob_volume_score", 0.0))
                displ = self._clamp01(getattr(ob, "ob_displacement_score", 0.0))
                react = self._clamp01(getattr(ob, "ob_reaction_score", 0.0))
                fresh = self._clamp01(getattr(ob, "ob_freshness", 1.0))
                score = (
                    0.30 * strength + 0.20 * volume + 0.20 * displ
                    + 0.15 * react + 0.15 * fresh
                )
                if score > best_score:
                    best_score = score
                    state_obj = getattr(ob, "state", None)
                    state = (state_obj.value if hasattr(state_obj, "value") else str(state_obj or "")).upper()
                    is_fresh = "FRESH" in state
                    is_htf = str(tf).upper() in _HTF_SET
                    is_inst = bool(is_htf and is_fresh and score >= 0.80)
                    best_fresh = is_fresh
                    best_inst = is_inst
                    best_notes = [
                        f"tf={tf}", f"state={state}",
                        f"str={strength:.2f}", f"vol={volume:.2f}",
                        f"disp={displ:.2f}", f"react={react:.2f}",
                    ]
                    if score >= 0.90:
                        best_grade = "A+"
                    elif score >= 0.75:
                        best_grade = "A"
                    elif score >= 0.60:
                        best_grade = "B"
                    elif score >= 0.40:
                        best_grade = "C"
                    else:
                        best_grade = "X"
        if best_score < 0:
            return OBQuality(grade="X", score=0.0, fresh=False,
                             institutional=False, notes=["no_aligned_ob"])
        return OBQuality(
            grade=best_grade, score=float(max(0.0, best_score)),
            fresh=best_fresh, institutional=best_inst, notes=best_notes,
        )

    def _evaluate_fvg_quality(self, snapshot: Any, direction: str) -> FVGQuality:
        """Clasifica el mejor FVG → A+/A/B/C.

        Combina: fvg_strength * 0.35 + gap_quality * 0.35
                 + (1 - fill_pct) * 0.30. HTF detectado por el tf donde vive.
        """
        per_tf = getattr(snapshot, "per_tf", None) or {}
        best_score = -1.0
        best_grade = "X"
        best_htf = False
        best_aligned = False
        best_notes: list[str] = []
        for tf, ctx in per_tf.items():
            fvgs = getattr(ctx, "active_fvgs", None) or []
            for fvg in fvgs:
                bias_dir = self._dir_value(getattr(fvg, "direction", None))
                aligned = self._bias_matches(direction, bias_dir)
                if not aligned:
                    continue
                strength = self._clamp01(getattr(fvg, "fvg_strength", 0.0))
                quality = self._clamp01(getattr(fvg, "gap_quality", 0.0))
                fill_pct = self._clamp01(getattr(fvg, "gap_fill_percent", 0.0))
                unfilled = 1.0 - fill_pct
                score = 0.35 * strength + 0.35 * quality + 0.30 * unfilled
                if score > best_score:
                    best_score = score
                    is_htf = str(tf).upper() in _HTF_SET
                    best_htf = is_htf
                    best_aligned = aligned
                    best_notes = [
                        f"tf={tf}", f"str={strength:.2f}",
                        f"qual={quality:.2f}", f"fill={fill_pct:.2f}",
                    ]
                    if score >= 0.85:
                        best_grade = "A+"
                    elif score >= 0.70:
                        best_grade = "A"
                    elif score >= 0.55:
                        best_grade = "B"
                    else:
                        best_grade = "C"
        if best_score < 0:
            return FVGQuality(grade="X", score=0.0, aligned=False,
                              htf=False, notes=["no_aligned_fvg"])
        return FVGQuality(
            grade=best_grade, score=float(max(0.0, best_score)),
            aligned=best_aligned, htf=best_htf, notes=best_notes,
        )

    @staticmethod
    def _ob_points(ob: OBQuality) -> float:
        """OB inst (30) > fresh (25) > mit (10). Excluyentes."""
        if ob.grade == "X":
            return 0.0
        if ob.institutional:
            return _PTS_OB_INST
        if ob.fresh:
            return _PTS_OB_FRESH
        return _PTS_OB_MIT

    @staticmethod
    def _fvg_points(fvg: FVGQuality) -> float:
        """HTF (15) > aligned (10). Excluyentes."""
        if fvg.grade == "X" or not fvg.aligned:
            return 0.0
        return _PTS_FVG_HTF if fvg.htf else _PTS_FVG_ALIGNED

    @staticmethod
    def _score_confluence(
        *, has_choch: bool, has_bos: bool, has_ob: bool, has_fvg: bool,
    ) -> float:
        if has_choch and has_bos and has_ob and has_fvg:
            return _PTS_CONFL_CBOF
        if has_choch and has_bos and has_ob:
            return _PTS_CONFL_CBO
        if has_choch and has_bos:
            return _PTS_CONFL_CB
        return 0.0

    def _evaluate_mtf(
        self, mtf_trends: Mapping[str, Any] | None, direction: str,
    ) -> MTFAlignment:
        """MTF check sobre D1, H4, H1, M30, M15 (AURION X 4.6: D1 incluido).

        D1 y H4 son HTF: si contradicen, ``htf_contradicts=True`` activa
        penalizacion fuerte. ``full_aligned`` requiere los 5 TFs.
        """
        check_tfs = ("D1", "H4", "H1", "M30", "M15")
        d = direction.upper()
        per_tf: dict[str, str] = {}
        aligned = 0
        htf_contra = False
        for tf in check_tfs:
            trend = (mtf_trends or {}).get(tf)
            val = self._dir_value(trend)
            per_tf[tf] = val
            if self._bias_matches(d, val):
                aligned += 1
            elif tf in ("H4", "D1") and val in ("BULLISH", "BEARISH"):
                if (d == "BUY" and val == "BEARISH") or (d == "SELL" and val == "BULLISH"):
                    htf_contra = True
        full = (aligned == len(check_tfs))
        return MTFAlignment(
            full_aligned=full,
            htf_contradicts=htf_contra,
            per_tf=per_tf,
            aligned_count=aligned,
        )

    # ─── Grading y reglas ───────────────────────────────────────────────

    @staticmethod
    def _setup_grade(
        *,
        smc_score: float,
        ml: float | None,
        ob_grade: str,
        fvg_grade: str,
        mtf: MTFAlignment,
        pd: PremiumDiscountResult,
    ) -> str:
        """Combina SMC + ML + execution quality → A+/A/B/C/X.

        AURION X 4.5.1 — Si ``ml`` no esta disponible aun (None o 0.0) se
        asume "pending" (neutro 0.55) para que el grade pre-Apex refleje
        solo la estructura SMC. El gate definitivo de ml lo aplica Apex /
        Institutional aguas abajo (no nos compete duplicarlo aqui).

        Reglas:
          A+ : smc>=90 AND ml>=0.65 AND ob in {A+,A} AND no htf_contradicts
          A  : smc>=80 AND ml>=0.55 AND ob in {A+,A,B} AND no htf_contradicts
          B  : smc>=70 AND ml>=0.50
          C  : smc>=60 AND ml>=0.45
          X  : el resto
        """
        ml_known = ml is not None and float(ml) > 0.0
        ml_v = float(ml) if ml_known else 0.55   # neutro mientras Apex no responda
        if mtf.htf_contradicts:
            # Bajamos automaticamente: maximo "B" si H4 contradice
            if smc_score >= 70 and ml_v >= 0.50:
                return "B"
            return "C" if smc_score >= 60 else "X"
        if smc_score >= 90 and ml_v >= 0.65 and ob_grade in ("A+", "A"):
            return "A+"
        if smc_score >= 80 and ml_v >= 0.55 and ob_grade in ("A+", "A", "B"):
            return "A"
        if smc_score >= 70 and ml_v >= 0.50:
            return "B"
        if smc_score >= 60 and ml_v >= 0.45:
            return "C"
        return "X"

    def _enforce_rules(
        self,
        *,
        smc_score: float,
        grade: str,
        ml: float | None,
        sweep_detected: bool,
        mtf: MTFAlignment,
        pd: PremiumDiscountResult,
        ob_grade: str = "B",
        fvg_grade: str = "B",
        volume_class: str | None = None,
        rr_estimated: float | None = None,
        news_blackout: bool = False,
    ) -> tuple[bool, str]:
        """Aplica las reglas de filtrado segun ``quality_mode``.

        Modos:
          * normal       → como hasta ahora (smc>=min_score, grade!=X, sweep/mtf opcionales).
          * high_quality → grade>=A + sweep + MTF aligned>=3 + RR>=min_rr_hq + no news.
          * premium_only → grade=A+ + score>=85 + sweep + MTF==5 + OB en {A+,A} +
                            FVG en {A+,A} + RR>=min_rr_pr + volumen HIGH/EXTREME + no news.
        """
        ml_known = ml is not None and float(ml) > 0.0
        ml_v = float(ml) if ml_known else 0.0

        # ── Gate base (todos los modos)
        if smc_score < self._min_score:
            return False, f"smc_score={smc_score:.1f}<min={self._min_score:.0f}"
        if grade == "X":
            return False, "setup_grade=X_prohibido"
        if grade == "C" and ml_known and ml_v < self._c_grade_ml_override:
            return False, (
                f"grade=C_requiere_ml>={self._c_grade_ml_override:.2f}_actual={ml_v:.2f}"
            )
        if self._require_sweep and not sweep_detected:
            return False, "require_sweep_no_sweep_detected"
        if self._require_mtf and not mtf.full_aligned:
            return False, f"require_mtf_full_align_aligned={mtf.aligned_count}/5"

        # ── HIGH_QUALITY mode
        if self._quality_mode == "high_quality":
            if grade not in ("A+", "A"):
                return False, f"hq_mode_grade={grade}_requires>=A"
            if not sweep_detected:
                return False, "hq_mode_no_sweep"
            if mtf.aligned_count < 3:
                return False, f"hq_mode_mtf_aligned={mtf.aligned_count}<3"
            if rr_estimated is not None and rr_estimated < self._min_rr_hq:
                return False, f"hq_mode_rr={rr_estimated:.2f}<{self._min_rr_hq:.2f}"
            if news_blackout:
                return False, "hq_mode_news_blackout"

        # ── PREMIUM_ONLY mode (seccion 14 del prompt del usuario)
        if self._quality_mode == "premium_only":
            if grade != "A+":
                return False, f"premium_grade={grade}_requires=A+"
            if smc_score < 85.0:
                return False, f"premium_score={smc_score:.1f}<85"
            if not sweep_detected:
                return False, "premium_no_sweep"
            if mtf.aligned_count < 5:
                return False, f"premium_mtf={mtf.aligned_count}<5"
            if ob_grade not in ("A+", "A"):
                return False, f"premium_ob={ob_grade}_requires>=A"
            if fvg_grade not in ("A+", "A"):
                return False, f"premium_fvg={fvg_grade}_requires>=A"
            if rr_estimated is not None and rr_estimated < self._min_rr_pr:
                return False, f"premium_rr={rr_estimated:.2f}<{self._min_rr_pr:.2f}"
            if volume_class is not None and volume_class.upper() not in ("HIGH", "EXTREME"):
                return False, f"premium_volume={volume_class}_requires>=HIGH"
            if news_blackout:
                return False, "premium_news_blackout"
        return True, ""

    def _setup_status(self, *, grade: str, allow: bool) -> str:
        """Mapea grade + allow a estado discreto para el panel."""
        if not allow or grade == "X":
            return "NO_TRADE"
        if grade == "A+":
            return "PREMIUM"
        if grade == "A":
            return "VALID"
        if grade == "B":
            return "WATCHLIST"
        return "NO_TRADE"  # C bloqueado o downgrade

    # ─── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dir_value(obj: Any) -> str:
        """Normaliza un valor a 'BULLISH' / 'BEARISH' / 'NEUTRAL'."""
        if obj is None:
            return "NEUTRAL"
        if hasattr(obj, "value"):
            obj = obj.value
        s = str(obj).strip().upper()
        if s in ("BULLISH", "UP", "LONG", "BUY"):
            return "BULLISH"
        if s in ("BEARISH", "DOWN", "SHORT", "SELL"):
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _bias_matches(direction: str, bias: str) -> bool:
        d = (direction or "").strip().upper()
        b = bias.upper()
        if d in ("BUY", "LONG"):
            return b == "BULLISH"
        if d in ("SELL", "SHORT"):
            return b == "BEARISH"
        return False

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        if v != v:  # NaN
            return 0.0
        return max(0.0, min(1.0, v))

    def _noop_decision(
        self, reason: str, *, smc_score: float, grade: str,
        allow: bool, ml: float | None,
    ) -> SMCDecision:
        empty_pd = PremiumDiscountResult(
            state=PremiumDiscountState.UNDEFINED,
            range_high=0.0, range_low=0.0, equilibrium=0.0,
            pct_in_range=0.5, tf_source="none",
            discount_threshold=0.382, premium_threshold=0.618,
            notes=[reason],
        )
        empty_ob = OBQuality("X", 0.0, False, False, [reason])
        empty_fvg = FVGQuality("X", 0.0, False, False, [reason])
        mtf = MTFAlignment(False, False, {}, 0)
        return SMCDecision(
            allow_trade=allow,
            block_reason="" if allow else reason,
            smc_score=float(smc_score),
            setup_grade=grade,
            ob_quality=empty_ob,
            fvg_quality=empty_fvg,
            pd_state=empty_pd,
            sweep_detected=False,
            sweep_kind="NONE",
            mtf=mtf,
            components={"sweep": 0, "choch": 0, "bos": 0, "ob": 0, "fvg": 0, "confluence": 0},
            lot_multiplier=float(self._lot_mults.get(grade, 1.0)),
            aggressive_execution=False,
            notes=[reason],
            setup_status=self._setup_status(grade=grade, allow=allow),
            quality_mode=self._quality_mode,
        )

    # ─── factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "SMCDecisionEngine":
        cfg = cfg or {}
        pd_cfg = cfg.get("premium_discount") or {}
        heur_cfg = heuristics_from_config(cfg.get("heuristics"))
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            min_score_to_trade=float(cfg.get("min_score_to_trade", 70.0)),
            require_sweep=bool(cfg.get("require_sweep", False)),
            require_mtf_full_align=bool(cfg.get("require_mtf_full_align", False)),
            c_grade_ml_override=float(cfg.get("c_grade_ml_override", 0.80)),
            grade_lot_multipliers=cfg.get("grade_lot_multipliers"),
            pd_engine=PremiumDiscountEngine.from_config(pd_cfg),
            weights=cfg.get("weights"),
            heuristics_enabled=bool(heur_cfg["enabled"]),
            heuristics_weight=float(heur_cfg["weight"]),
            heuristics_rules=heur_cfg["enabled_rules"],
            quality_mode=str(cfg.get("quality_mode", "normal")),
            min_rr_high_quality=float(cfg.get("min_rr_high_quality", 2.5)),
            min_rr_premium=float(cfg.get("min_rr_premium", 3.0)),
        )

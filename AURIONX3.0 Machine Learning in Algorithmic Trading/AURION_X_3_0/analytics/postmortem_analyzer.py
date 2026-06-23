"""Postmortem Analyzer (AURION X 4.5.2) — analista determinista local.

Analiza trades cerrados de XAUUSD desde PostgreSQL y produce:

    * Estadisticas agregadas: WR, total R, total PnL, expectancy.
    * Slices: por hora UTC, dia de semana, setup_type, smc_decision_grade,
      ob_quality_grade, fvg_quality_grade, smc_pd_state, sweep_kind,
      heur_rules_applied.
    * Peores trades (top losers por R-multiple) con contexto SMC.
    * Mejores trades (top winners).
    * Patrones recurrentes ganadores vs perdedores.
    * Recomendaciones automatizadas en lenguaje natural (no LLM, plantillas
      derivadas estadisticamente).

Sin coste de API, sin internet, sin alucinaciones. Determinista 100%.

Se invoca desde:
    * ``tools/_postmortem_local.py`` (CLI manual).
    * (futuro) cron interno EOD del orchestrator.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from analytics.db_connection import get_session
from analytics.models import Trade

logger = logging.getLogger("postmortem")


# ─── dataclasses de salida ──────────────────────────────────────────────


@dataclass
class TradeRow:
    """Fila normalizada para analisis (con features SMC ya extraidas)."""
    id: str
    symbol: str
    direction: str
    open_time: datetime
    close_time: datetime | None
    profit: float
    r_multiple: float
    ml_confidence: float | None
    setup_type: str
    smc_score: float | None
    smc_grade: str | None
    ob_grade: str | None
    fvg_grade: str | None
    pd_state: str | None
    sweep_kind: str | None
    mtf_aligned: int | None
    heur_rules: list[str]
    close_reason: str | None
    hour_utc: int
    weekday: int   # 0=Mon ... 6=Sun


@dataclass
class SliceStats:
    """Estadisticas de un grupo de trades."""
    n: int
    n_wins: int
    n_losses: int
    win_rate: float       # 0-1
    avg_r: float
    total_r: float
    total_pnl: float
    expectancy: float     # avg_r ponderado (proxy del Kelly)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "n_wins": self.n_wins, "n_losses": self.n_losses,
            "win_rate": round(self.win_rate, 4),
            "avg_r": round(self.avg_r, 3),
            "total_r": round(self.total_r, 2),
            "total_pnl": round(self.total_pnl, 2),
            "expectancy": round(self.expectancy, 3),
        }


@dataclass
class Recommendation:
    """Sugerencia accionable derivada de los datos."""
    severity: str   # "info" | "warning" | "critical"
    target: str     # "parameters.yaml > path.to.key" / setup / hour / regla
    action: str     # "raise" | "lower" | "blacklist" | "review"
    current_value: Any
    suggested_value: Any
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "target": self.target,
            "action": self.action,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class PostMortemReport:
    """Informe completo del periodo analizado."""
    symbol: str
    period_start: datetime
    period_end: datetime
    n_trades: int
    overall: SliceStats
    by_hour: dict[int, SliceStats]
    by_weekday: dict[int, SliceStats]
    by_setup_type: dict[str, SliceStats]
    by_smc_grade: dict[str, SliceStats]
    by_ob_grade: dict[str, SliceStats]
    by_fvg_grade: dict[str, SliceStats]
    by_pd_state: dict[str, SliceStats]
    by_sweep_kind: dict[str, SliceStats]
    by_heur_rule: dict[str, SliceStats]
    top_losers: list[TradeRow]
    top_winners: list[TradeRow]
    recommendations: list[Recommendation]

    def to_dict(self) -> dict[str, Any]:
        def _slices(d: dict) -> dict:
            return {str(k): v.to_dict() for k, v in d.items()}
        return {
            "symbol": self.symbol,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "n_trades": self.n_trades,
            "overall": self.overall.to_dict(),
            "by_hour": _slices(self.by_hour),
            "by_weekday": _slices(self.by_weekday),
            "by_setup_type": _slices(self.by_setup_type),
            "by_smc_grade": _slices(self.by_smc_grade),
            "by_ob_grade": _slices(self.by_ob_grade),
            "by_fvg_grade": _slices(self.by_fvg_grade),
            "by_pd_state": _slices(self.by_pd_state),
            "by_sweep_kind": _slices(self.by_sweep_kind),
            "by_heur_rule": _slices(self.by_heur_rule),
            "top_losers": [_trade_to_dict(t) for t in self.top_losers],
            "top_winners": [_trade_to_dict(t) for t in self.top_winners],
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


def _trade_to_dict(t: TradeRow) -> dict[str, Any]:
    return {
        "id": t.id, "direction": t.direction,
        "open_time": t.open_time.isoformat() if t.open_time else None,
        "close_time": t.close_time.isoformat() if t.close_time else None,
        "profit": round(t.profit, 2), "r_multiple": round(t.r_multiple, 2),
        "setup_type": t.setup_type, "smc_grade": t.smc_grade,
        "smc_score": t.smc_score, "ob_grade": t.ob_grade, "fvg_grade": t.fvg_grade,
        "pd_state": t.pd_state, "sweep_kind": t.sweep_kind,
        "heur_rules": t.heur_rules, "close_reason": t.close_reason,
        "hour_utc": t.hour_utc, "weekday": t.weekday,
    }


# ─── extraccion ──────────────────────────────────────────────────────────


_WEEKDAY_LABELS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]


def _feat(t: Trade, key: str, default: Any = None) -> Any:
    """Acceso defensivo a t.features_json[key]."""
    try:
        fj = t.features_json or {}
        if not isinstance(fj, dict):
            return default
        return fj.get(key, default)
    except Exception:
        return default


def _to_trade_row(t: Trade) -> TradeRow:
    open_t = t.open_time
    if isinstance(open_t, datetime) and open_t.tzinfo is None:
        open_t = open_t.replace(tzinfo=timezone.utc)
    close_t = t.close_time
    if isinstance(close_t, datetime) and close_t.tzinfo is None:
        close_t = close_t.replace(tzinfo=timezone.utc)
    fj = t.features_json or {}
    # heur_rules viene como list[dict] dentro de smc_decision.heuristics
    # pero nosotros los persistimos individualmente; aqui consolidamos.
    heur_rules: list[str] = []
    smc_dec = fj.get("smc_decision") if isinstance(fj, dict) else None
    if isinstance(smc_dec, dict):
        heur = smc_dec.get("heuristics") or {}
        for r in (heur.get("applied_rules") or []):
            if isinstance(r, dict) and r.get("id"):
                heur_rules.append(str(r["id"]))
    return TradeRow(
        id=str(t.id),
        symbol=t.symbol,
        direction=t.direction.value if hasattr(t.direction, "value") else str(t.direction),
        open_time=open_t,
        close_time=close_t,
        profit=float(t.profit or 0.0),
        r_multiple=float(t.r_multiple or 0.0),
        ml_confidence=float(t.ml_confidence) if t.ml_confidence is not None else None,
        setup_type=str(_feat(t, "setup_type") or _feat(t, "strategy_detail") or
                       (t.strategy.value if hasattr(t.strategy, "value") else "UNKNOWN")),
        smc_score=float(_feat(t, "smc_decision_score")) if _feat(t, "smc_decision_score") is not None else None,
        smc_grade=_feat(t, "smc_decision_grade"),
        ob_grade=_feat(t, "smc_ob_quality_grade"),
        fvg_grade=_feat(t, "smc_fvg_quality_grade"),
        pd_state=_feat(t, "smc_pd_state"),
        sweep_kind=_feat(t, "smc_sweep_kind"),
        mtf_aligned=int(_feat(t, "smc_mtf_aligned_count", 0) or 0),
        heur_rules=heur_rules,
        close_reason=t.close_reason,
        hour_utc=open_t.hour if open_t else 0,
        weekday=open_t.weekday() if open_t else 0,
    )


def _fetch_trades(
    session: Session,
    symbol: str,
    period_start: datetime,
    period_end: datetime,
) -> list[TradeRow]:
    """Lee trades CERRADOS del periodo."""
    stmt = (
        select(Trade)
        .where(Trade.symbol == symbol.upper())
        .where(Trade.close_time.is_not(None))
        .where(Trade.open_time >= period_start)
        .where(Trade.open_time < period_end)
        .order_by(Trade.open_time.asc())
    )
    rows: list[TradeRow] = []
    for t in session.scalars(stmt):
        try:
            rows.append(_to_trade_row(t))
        except Exception as exc:
            logger.debug("postmortem: skip trade %s (%s)", t.id, exc)
    return rows


# ─── estadisticas ────────────────────────────────────────────────────────


def _stats_for(rows: Iterable[TradeRow]) -> SliceStats:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return SliceStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wins = [r for r in rows if r.r_multiple > 0]
    losses = [r for r in rows if r.r_multiple <= 0]
    total_r = sum(r.r_multiple for r in rows)
    total_pnl = sum(r.profit for r in rows)
    win_rate = len(wins) / n
    avg_r = total_r / n
    # Expectancy: WR * avg(win_R) + (1-WR) * avg(loss_R)
    avg_win_r = sum(r.r_multiple for r in wins) / max(1, len(wins))
    avg_loss_r = sum(r.r_multiple for r in losses) / max(1, len(losses))
    expectancy = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r
    return SliceStats(
        n=n, n_wins=len(wins), n_losses=len(losses),
        win_rate=win_rate, avg_r=avg_r, total_r=total_r,
        total_pnl=total_pnl, expectancy=expectancy,
    )


def _group_by(
    rows: list[TradeRow], key_fn: Any, label_fn: Any = str,
) -> dict[Any, SliceStats]:
    """Agrupa por clave + calcula stats."""
    buckets: dict[Any, list[TradeRow]] = {}
    for r in rows:
        k = key_fn(r)
        if k is None or k == "" or k == "None":
            k = "UNKNOWN"
        k = label_fn(k)
        buckets.setdefault(k, []).append(r)
    return {k: _stats_for(v) for k, v in buckets.items()}


def _group_by_heur_rule(rows: list[TradeRow]) -> dict[str, SliceStats]:
    """Cada trade puede tener varias reglas: contamos cada combinacion."""
    buckets: dict[str, list[TradeRow]] = {}
    for r in rows:
        if not r.heur_rules:
            buckets.setdefault("NONE", []).append(r)
            continue
        for rule in r.heur_rules:
            buckets.setdefault(rule, []).append(r)
    return {k: _stats_for(v) for k, v in buckets.items()}


# ─── recomendaciones automatizadas ───────────────────────────────────────


def _build_recommendations(
    overall: SliceStats,
    by_hour: dict[int, SliceStats],
    by_setup: dict[str, SliceStats],
    by_smc_grade: dict[str, SliceStats],
    by_heur_rule: dict[str, SliceStats],
    by_pd_state: dict[str, SliceStats],
    by_sweep_kind: dict[str, SliceStats],
    rows: list[TradeRow],
) -> list[Recommendation]:
    """Reglas duras sobre los slices para sugerir ajustes."""
    recs: list[Recommendation] = []
    MIN_N = 5      # umbral minimo de muestras
    BAD_WR = 0.35  # debajo de esto: subset perdedor claro
    GOOD_WR = 0.65
    BAD_AVG_R = -0.30

    # ── horas perdedoras
    for hr, s in sorted(by_hour.items()):
        if s.n >= MIN_N and s.win_rate < BAD_WR and s.avg_r < BAD_AVG_R:
            recs.append(Recommendation(
                severity="warning",
                target=f"chop_guard.blocked_hours_utc (hora={hr:02d})",
                action="blacklist",
                current_value="(no blocked)",
                suggested_value=f"agregar {hr}",
                reason=(
                    f"Hora {hr:02d} UTC: WR {s.win_rate:.1%} avgR {s.avg_r:+.2f} "
                    f"en {s.n} trades. Significativamente peor que overall "
                    f"(WR {overall.win_rate:.1%}, avgR {overall.avg_r:+.2f})."
                ),
                evidence={"hour": hr, **s.to_dict()},
            ))

    # ── setups perdedores
    for setup, s in by_setup.items():
        if s.n >= MIN_N and s.win_rate < BAD_WR and s.expectancy < -0.20:
            recs.append(Recommendation(
                severity="warning",
                target=f"chop_guard.blocked_setups (setup={setup})",
                action="blacklist",
                current_value="(no blacklisted)",
                suggested_value=f"agregar '{setup}'",
                reason=(
                    f"Setup {setup}: WR {s.win_rate:.1%} avgR {s.avg_r:+.2f} "
                    f"expectancy {s.expectancy:+.2f} en {s.n} trades. Sangrando."
                ),
                evidence={"setup": setup, **s.to_dict()},
            ))

    # ── SMC grade rentabilidad
    if "X" in by_smc_grade and by_smc_grade["X"].n >= 3:
        # Trades grado X que se ejecutaron (no deberian) → revisar
        recs.append(Recommendation(
            severity="critical",
            target="smc.decision_engine.min_score_to_trade",
            action="review",
            current_value=70.0,
            suggested_value="revisar logs: trades grade=X ejecutados",
            reason=(
                f"Hay {by_smc_grade['X'].n} trades con SMC grade=X (prohibido). "
                f"Esto solo puede pasar si el filtro estaba off o legacy. "
                f"Verificar que SMCDecisionEngine.enabled=true en config."
            ),
            evidence=by_smc_grade["X"].to_dict(),
        ))

    # Sugerir subir min_score si los grados bajos (B/C) tienen WR muy malo
    bc_n = sum(by_smc_grade.get(g, SliceStats(0,0,0,0,0,0,0,0)).n for g in ("B", "C"))
    bc_wins = sum(by_smc_grade.get(g, SliceStats(0,0,0,0,0,0,0,0)).n_wins for g in ("B", "C"))
    bc_wr = (bc_wins / bc_n) if bc_n > 0 else 0.0
    a_n = sum(by_smc_grade.get(g, SliceStats(0,0,0,0,0,0,0,0)).n for g in ("A+", "A"))
    a_wins = sum(by_smc_grade.get(g, SliceStats(0,0,0,0,0,0,0,0)).n_wins for g in ("A+", "A"))
    a_wr = (a_wins / a_n) if a_n > 0 else 0.0
    if bc_n >= 10 and a_n >= 5 and bc_wr < BAD_WR and a_wr > GOOD_WR:
        recs.append(Recommendation(
            severity="warning",
            target="smc.decision_engine.min_score_to_trade",
            action="raise",
            current_value=70.0,
            suggested_value=80.0,
            reason=(
                f"Grades B/C (n={bc_n}) WR {bc_wr:.1%} vs A+/A (n={a_n}) WR {a_wr:.1%}. "
                f"Subir min_score=80 filtraria B/C y concentraria en setups premium."
            ),
            evidence={"bc_wr": bc_wr, "a_wr": a_wr, "bc_n": bc_n, "a_n": a_n},
        ))

    # ── Heuristicas: identificar reglas con buen/mal track record
    for rule, s in by_heur_rule.items():
        if rule == "NONE" or s.n < 5:
            continue
        if s.win_rate < BAD_WR and s.expectancy < -0.20:
            recs.append(Recommendation(
                severity="info",
                target=f"smc.decision_engine.heuristics.rules (regla={rule})",
                action="review",
                current_value="enabled (default)",
                suggested_value="aumentar penalizacion o desactivar",
                reason=(
                    f"Regla {rule}: WR {s.win_rate:.1%} avgR {s.avg_r:+.2f} en "
                    f"{s.n} trades. La regla esta detectando situaciones malas "
                    f"pero el bot esta entrando igual; aumentar penalty."
                ),
                evidence={"rule": rule, **s.to_dict()},
            ))
        elif s.win_rate > GOOD_WR and s.expectancy > 0.20 and s.avg_r > 0.4:
            recs.append(Recommendation(
                severity="info",
                target=f"smc.decision_engine.heuristics.weight (regla={rule})",
                action="raise",
                current_value=1.0,
                suggested_value="considerar weight=1.3 si la regla aporta bonus",
                reason=(
                    f"Regla {rule}: WR {s.win_rate:.1%} avgR {s.avg_r:+.2f}. "
                    f"Setups que disparan esta regla ganan; reforzar su peso."
                ),
                evidence={"rule": rule, **s.to_dict()},
            ))

    # ── P/D state direccional
    # PREMIUM = mejor para SELL, DISCOUNT = mejor para BUY
    # Si trades en PREMIUM van mal (mayoria BUYs?), advertir
    if "PREMIUM" in by_pd_state and by_pd_state["PREMIUM"].n >= 5:
        s = by_pd_state["PREMIUM"]
        # cuantos son BUY
        buys_premium = [r for r in rows if r.pd_state == "PREMIUM" and r.direction == "BUY"]
        if len(buys_premium) >= 3:
            buys_wins = sum(1 for r in buys_premium if r.r_multiple > 0)
            buys_wr = buys_wins / len(buys_premium)
            if buys_wr < 0.40:
                recs.append(Recommendation(
                    severity="warning",
                    target="smc.decision_engine (BUY en PREMIUM)",
                    action="review",
                    current_value="pd_delta=-20 (penalty)",
                    suggested_value="aumentar penalty a -25 o bloquear directamente",
                    reason=(
                        f"BUYs en PREMIUM: {len(buys_premium)} trades, WR "
                        f"{buys_wr:.1%}. La penalty actual no es suficiente."
                    ),
                    evidence={"n": len(buys_premium), "wr": buys_wr},
                ))

    # ── overall sanity
    if overall.n >= 20 and overall.expectancy < -0.10:
        recs.append(Recommendation(
            severity="critical",
            target="overall_strategy",
            action="review",
            current_value=f"expectancy={overall.expectancy:+.2f}",
            suggested_value="pausar / reentrenar / subir thresholds",
            reason=(
                f"Expectancy global {overall.expectancy:+.2f} en {overall.n} trades. "
                f"El sistema esta perdiendo en agregado. Revisar urgentemente."
            ),
            evidence=overall.to_dict(),
        ))

    return recs


# ─── API publica ─────────────────────────────────────────────────────────


def analyze(
    *,
    symbol: str = "XAUUSD",
    days_back: int = 1,
    until: datetime | None = None,
    top_n: int = 5,
) -> PostMortemReport:
    """Analiza los ultimos ``days_back`` dias de trading.

    Args:
        symbol: simbolo a analizar (default XAUUSD).
        days_back: cuantos dias hacia atras desde ``until``.
        until: fecha tope (default: ahora UTC).
        top_n: cuantos top winners/losers extraer.

    Returns:
        PostMortemReport con stats, slices y recomendaciones.
    """
    end = until or datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    with get_session() as session:
        rows = _fetch_trades(session, symbol, start, end)
    logger.info(
        "postmortem: %s trades %s..%s symbol=%s",
        len(rows), start.isoformat(), end.isoformat(), symbol,
    )
    overall = _stats_for(rows)
    by_hour = _group_by(rows, lambda r: r.hour_utc, label_fn=int)
    by_weekday = _group_by(rows, lambda r: r.weekday, label_fn=int)
    by_setup = _group_by(rows, lambda r: r.setup_type)
    by_smc_grade = _group_by(rows, lambda r: r.smc_grade or "n/a")
    by_ob_grade = _group_by(rows, lambda r: r.ob_grade or "n/a")
    by_fvg_grade = _group_by(rows, lambda r: r.fvg_grade or "n/a")
    by_pd_state = _group_by(rows, lambda r: r.pd_state or "n/a")
    by_sweep_kind = _group_by(rows, lambda r: r.sweep_kind or "n/a")
    by_heur_rule = _group_by_heur_rule(rows)
    sorted_rows = sorted(rows, key=lambda r: r.r_multiple)
    top_losers = sorted_rows[:top_n]
    top_winners = sorted_rows[::-1][:top_n]
    recs = _build_recommendations(
        overall=overall, by_hour=by_hour, by_setup=by_setup,
        by_smc_grade=by_smc_grade, by_heur_rule=by_heur_rule,
        by_pd_state=by_pd_state, by_sweep_kind=by_sweep_kind, rows=rows,
    )
    return PostMortemReport(
        symbol=symbol, period_start=start, period_end=end,
        n_trades=len(rows), overall=overall, by_hour=by_hour,
        by_weekday=by_weekday, by_setup_type=by_setup,
        by_smc_grade=by_smc_grade, by_ob_grade=by_ob_grade,
        by_fvg_grade=by_fvg_grade, by_pd_state=by_pd_state,
        by_sweep_kind=by_sweep_kind, by_heur_rule=by_heur_rule,
        top_losers=top_losers, top_winners=top_winners,
        recommendations=recs,
    )


# ─── render HTML ─────────────────────────────────────────────────────────


_HTML_TPL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AURION X — Postmortem __SYM__</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1419;color:#e6edf3;margin:24px;line-height:1.5}
h1{color:#58a6ff;border-bottom:2px solid #30363d;padding-bottom:8px}
h2{color:#9da7b3;margin-top:32px;border-bottom:1px solid #30363d;padding-bottom:4px}
.kpi{display:inline-block;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 20px;margin:6px}
.kpi .v{font-size:24px;font-weight:bold;color:#58a6ff}.kpi .l{font-size:12px;color:#8b949e}
table{border-collapse:collapse;width:100%;margin:8px 0;background:#161b22;border:1px solid #30363d;border-radius:6px;overflow:hidden}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid #30363d;font-family:Consolas,monospace;font-size:13px}
th{background:#21262d;color:#9da7b3;text-transform:uppercase;font-size:11px;letter-spacing:1px}
.win{color:#56d364}.loss{color:#f85149}.warn{color:#f0b41f}.crit{color:#f85149;font-weight:bold}
.rec{background:#161b22;border-left:4px solid #58a6ff;padding:10px 16px;margin:8px 0;border-radius:4px}
.rec.warning{border-left-color:#f0b41f}.rec.critical{border-left-color:#f85149}.rec.info{border-left-color:#58a6ff}
.rec h4{margin:0 0 4px;color:#58a6ff;font-size:14px}
.rec .meta{font-size:11px;color:#8b949e}
.rec p{margin:4px 0;font-size:13px}
.muted{color:#8b949e;font-size:12px}
</style></head>
<body>
<h1>AURION X — Postmortem analisis <span class="muted">__SYM__ &nbsp; __PERIOD__</span></h1>
__KPIS__
<h2>Recomendaciones automaticas (__NREC__)</h2>
__RECS__
__SECTIONS__
<p class="muted">Generado: __GEN__ &nbsp; · &nbsp; Determinista, sin LLM &nbsp; · &nbsp; AURION X 4.5.2</p>
</body></html>
"""


def _slice_table(title: str, slices: dict[Any, SliceStats], sort_by_n: bool = True) -> str:
    if not slices:
        return f"<h2>{title}</h2><p class='muted'>Sin datos.</p>"
    items = sorted(slices.items(), key=lambda kv: kv[1].n, reverse=True) if sort_by_n \
        else sorted(slices.items(), key=lambda kv: str(kv[0]))
    rows_html = []
    for k, s in items:
        wr_cls = "win" if s.win_rate >= 0.55 else ("loss" if s.win_rate < 0.40 else "")
        r_cls = "win" if s.avg_r > 0 else "loss"
        rows_html.append(
            f"<tr><td>{k}</td><td>{s.n}</td>"
            f"<td class='{wr_cls}'>{s.win_rate:.1%}</td>"
            f"<td class='{r_cls}'>{s.avg_r:+.2f}</td>"
            f"<td>{s.total_r:+.1f}</td><td>{s.total_pnl:+.2f}</td>"
            f"<td>{s.expectancy:+.2f}</td></tr>"
        )
    return (
        f"<h2>{title}</h2>"
        "<table><tr><th>Grupo</th><th>N</th><th>WR</th><th>avgR</th><th>totR</th><th>PnL$</th><th>Exp</th></tr>"
        + "".join(rows_html) + "</table>"
    )


def _trades_table(title: str, trades: list[TradeRow]) -> str:
    if not trades:
        return f"<h2>{title}</h2><p class='muted'>Sin trades.</p>"
    rows_html = []
    for t in trades:
        r_cls = "win" if t.r_multiple > 0 else "loss"
        rows_html.append(
            f"<tr><td>{t.open_time.strftime('%m-%d %H:%M') if t.open_time else 'n/a'}</td>"
            f"<td>{t.direction}</td><td>{t.setup_type}</td>"
            f"<td class='{r_cls}'>{t.r_multiple:+.2f}</td>"
            f"<td>{t.profit:+.2f}$</td>"
            f"<td>{t.smc_grade or 'n/a'}</td><td>{t.smc_score:.0f}</td>"
            f"<td>{t.pd_state or 'n/a'}</td><td>{t.sweep_kind or 'n/a'}</td>"
            f"<td>{t.ob_grade or 'n/a'}</td><td>{t.fvg_grade or 'n/a'}</td>"
            f"<td>{','.join(t.heur_rules[:3])}</td>"
            f"<td class='muted'>{t.close_reason or '-'}</td></tr>"
            if t.smc_score is not None else
            f"<tr><td>{t.open_time.strftime('%m-%d %H:%M') if t.open_time else 'n/a'}</td>"
            f"<td>{t.direction}</td><td>{t.setup_type}</td>"
            f"<td class='{r_cls}'>{t.r_multiple:+.2f}</td>"
            f"<td>{t.profit:+.2f}$</td>"
            f"<td colspan='7' class='muted'>(sin smc_decision en features_json)</td>"
            f"<td class='muted'>{t.close_reason or '-'}</td></tr>"
        )
    return (
        f"<h2>{title}</h2>"
        "<table><tr><th>Open</th><th>Dir</th><th>Setup</th><th>R</th><th>PnL</th>"
        "<th>SMC</th><th>Score</th><th>P/D</th><th>Sweep</th><th>OB</th><th>FVG</th>"
        "<th>Heur</th><th>CloseReason</th></tr>"
        + "".join(rows_html) + "</table>"
    )


def render_html(report: PostMortemReport) -> str:
    """Renderiza HTML pretty del informe."""
    o = report.overall
    kpis = (
        f"<div class='kpi'><div class='l'>Trades</div><div class='v'>{o.n}</div></div>"
        f"<div class='kpi'><div class='l'>Win Rate</div><div class='v'>{o.win_rate:.1%}</div></div>"
        f"<div class='kpi'><div class='l'>Avg R</div><div class='v'>{o.avg_r:+.2f}</div></div>"
        f"<div class='kpi'><div class='l'>Total R</div><div class='v'>{o.total_r:+.1f}</div></div>"
        f"<div class='kpi'><div class='l'>Total PnL</div><div class='v'>{o.total_pnl:+.2f}$</div></div>"
        f"<div class='kpi'><div class='l'>Expectancy</div><div class='v'>{o.expectancy:+.2f}</div></div>"
    )
    if report.recommendations:
        recs_html = []
        for r in report.recommendations:
            recs_html.append(
                f"<div class='rec {r.severity}'>"
                f"<h4>[{r.severity.upper()}] {r.target} → {r.action.upper()}</h4>"
                f"<div class='meta'>actual: {r.current_value} &nbsp; sugerido: {r.suggested_value}</div>"
                f"<p>{r.reason}</p></div>"
            )
        recs_block = "".join(recs_html)
    else:
        recs_block = "<p class='muted'>Sin recomendaciones criticas (sistema operando dentro de parametros).</p>"
    weekday_view = {f"{wd}-{_WEEKDAY_LABELS[wd]}": v for wd, v in report.by_weekday.items()}
    sections = (
        _slice_table("Por hora UTC", report.by_hour, sort_by_n=False)
        + _slice_table("Por dia de semana", weekday_view, sort_by_n=False)
        + _slice_table("Por setup_type", report.by_setup_type)
        + _slice_table("Por SMC grade", report.by_smc_grade)
        + _slice_table("Por OB grade", report.by_ob_grade)
        + _slice_table("Por FVG grade", report.by_fvg_grade)
        + _slice_table("Por Premium/Discount state", report.by_pd_state)
        + _slice_table("Por sweep kind", report.by_sweep_kind)
        + _slice_table("Por regla heuristica disparada", report.by_heur_rule)
        + _trades_table("Top losers", report.top_losers)
        + _trades_table("Top winners", report.top_winners)
    )
    period = (
        f"{report.period_start.strftime('%Y-%m-%d %H:%M')} → "
        f"{report.period_end.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    html = (_HTML_TPL
        .replace("__SYM__", report.symbol)
        .replace("__PERIOD__", period)
        .replace("__KPIS__", kpis)
        .replace("__NREC__", str(len(report.recommendations)))
        .replace("__RECS__", recs_block)
        .replace("__SECTIONS__", sections)
        .replace("__GEN__", datetime.now(timezone.utc).isoformat())
    )
    return html

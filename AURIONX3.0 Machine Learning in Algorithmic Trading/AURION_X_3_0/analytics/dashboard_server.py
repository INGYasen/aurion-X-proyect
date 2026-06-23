"""Dashboard analítico HTTP (stdlib) — métricas ML y trading."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from analytics.db_connection import init_db
from analytics.logger import get_logger
from analytics.quant_repository import QuantAnalyticsRepository
from analytics.trade_repository import TradeRepository

try:
    from analytics.smc_repository import SMCRepository
except Exception:  # pragma: no cover - import opcional
    SMCRepository = None  # type: ignore[assignment]

logger = get_logger(__name__)


def _aggregate_trades(limit: int = 2000) -> dict[str, Any]:
    repo = TradeRepository()
    rows = repo.get_training_dataset(limit=limit)
    if not rows:
        return {"total": 0}
    wins = sum(1 for r in rows if float(r.get("profit") or 0) > 0)
    n = len(rows)
    rs = [float(r["r_multiple"]) for r in rows if r.get("r_multiple") is not None]
    by_setup: dict[str, dict[str, float]] = {}
    by_hour: dict[int, dict[str, float]] = {}
    for r in rows:
        fx = r.get("features_json")
        if not isinstance(fx, dict):
            continue
        st = str(fx.get("setup_type", "UNK"))
        b = by_setup.setdefault(st, {"n": 0, "w": 0})
        b["n"] += 1
        if float(r.get("profit") or 0) > 0:
            b["w"] += 1
        hr = int(fx.get("hour_slot", -1))
        if 0 <= hr <= 23:
            h = by_hour.setdefault(hr, {"n": 0, "w": 0})
            h["n"] += 1
            if float(r.get("profit") or 0) > 0:
                h["w"] += 1
    equity = 10000.0
    curve = [equity]
    for r in reversed(rows):
        equity += float(r.get("profit") or 0)
        curve.append(equity)
    return {
        "total": n,
        "winrate": wins / n,
        "avg_r": sum(rs) / len(rs) if rs else 0.0,
        "by_setup": {k: {"winrate": v["w"] / max(v["n"], 1), "n": v["n"]} for k, v in by_setup.items()},
        "by_hour": {str(k): {"winrate": v["w"] / max(v["n"], 1), "n": v["n"]} for k, v in by_hour.items()},
        "equity_curve": curve[-120:],
        "recent_predictions": QuantAnalyticsRepository().recent_predictions(limit=50),
    }


HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AURION X Analytics</title>
<style>
body{font-family:system-ui,sans-serif;background:#0f1419;color:#e6edf3;margin:24px}
h1{color:#58a6ff} h2{color:#9da7b3;margin-top:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
pre{overflow:auto;font-size:12px}
a{color:#58a6ff}
.smc-ob{background:rgba(255,193,7,0.10);border-left:3px solid #ffc107;padding:6px;margin:4px 0}
.smc-ob.bear{background:rgba(244,67,54,0.10);border-left-color:#f44336}
.smc-fvg{background:rgba(102,187,106,0.12);border-left:3px solid #66bb6a;padding:6px;margin:4px 0}
.smc-fvg.bear{background:rgba(239,83,80,0.12);border-left-color:#ef5350}
.tag{display:inline-block;padding:2px 6px;background:#30363d;border-radius:4px;font-size:11px;margin-right:6px}
</style></head><body>
<h1>AURION X — Dashboard cuantitativo</h1>
<p>API: <a href="/api/summary">/api/summary</a> · <a href="/api/smc/XAUUSD">/api/smc/XAUUSD</a> · <a href="/smc/XAUUSD">/smc/XAUUSD</a> (visual) · <a href="/api/knowledge/XAUUSD">/api/knowledge/XAUUSD</a> · <a href="/knowledge/XAUUSD">XAU Knowledge</a></p>
<div id="out" class="card"><pre>Cargando...</pre></div>
<script>
fetch('/api/summary').then(r=>r.json()).then(d=>{
  document.getElementById('out').innerHTML='<pre>'+JSON.stringify(d,null,2)+'</pre>';
});
</script></body></html>"""


SMC_HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AURION X — SMC __SYM__</title>
<style>
body{font-family:system-ui,sans-serif;background:#0f1419;color:#e6edf3;margin:24px}
h1{color:#58a6ff} h2{color:#9da7b3;margin-top:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.tag{display:inline-block;padding:2px 6px;background:#30363d;border-radius:4px;font-size:11px;margin-right:6px}
.bull{color:#66bb6a}
.bear{color:#ef5350}
.neutral{color:#fcc419}
.kv{display:flex;justify-content:space-between;padding:2px 0;font-size:13px}
.kv span:first-child{color:#9da7b3}
pre{overflow:auto;font-size:11px}
</style></head><body>
<h1>SMC — <span id="sym">__SYM__</span></h1>
<p><a href="/" style="color:#58a6ff">← volver al dashboard</a></p>
<div class="card" id="conf"><strong>Confluence</strong><pre id="conf-pre">Cargando...</pre></div>
<div class="grid">
  <div class="card"><h2>Order Blocks</h2><div id="ob"></div></div>
  <div class="card"><h2>Fair Value Gaps</h2><div id="fvg"></div></div>
  <div class="card"><h2>BOS</h2><div id="bos"></div></div>
  <div class="card"><h2>CHOCH</h2><div id="choch"></div></div>
  <div class="card"><h2>Liquidity (SSL / BSL / EQ / POOL)</h2><div id="liq"></div></div>
</div>
<script>
const sym = document.getElementById('sym').innerText;
function tag(t,cls){return '<span class="tag '+(cls||'')+'">'+t+'</span>';}
function biasCls(b){return b==='BULLISH'?'bull':(b==='BEARISH'?'bear':'neutral');}
function row(o, fields){
  let html='<div class="card" style="margin:8px 0;padding:10px">';
  fields.forEach(f=>{html+='<div class="kv"><span>'+f+'</span><span>'+(o[f]??'-')+'</span></div>';});
  return html+'</div>';
}
fetch('/api/smc/'+sym).then(r=>r.json()).then(d=>{
  if(d && d.confluence){
    const c=d.confluence;
    document.getElementById('conf-pre').innerHTML =
      tag(c.direction_bias, biasCls(c.direction_bias))+
      tag('confluence '+(c.confluence_score||0).toFixed(3))+
      tag('macro '+(c.macro_bias||'-'), biasCls(c.macro_bias))+
      tag('inst '+(c.institutional_bias||'-'), biasCls(c.institutional_bias))+
      tag('op '+(c.operational_bias||'-'), biasCls(c.operational_bias))+
      '\\nBOS:'+(c.bos_component||0).toFixed(3)+
      '  CHOCH:'+(c.choch_component||0).toFixed(3)+
      '  OB:'+(c.ob_component||0).toFixed(3)+
      '  FVG:'+(c.fvg_component||0).toFixed(3)+
      '  Liq:'+(c.liquidity_component||0).toFixed(3)+
      '\\nper_tf:'+JSON.stringify(c.per_tf||{});
  } else {
    document.getElementById('conf-pre').innerText='Sin confluencia disponible aún.';
  }
  document.getElementById('ob').innerHTML  = (d.order_blocks||[]).map(o=>row(o,['timeframe','direction','top','bottom','state','ob_quality','ob_freshness','ob_strength','ob_age_bars'])).join('');
  document.getElementById('fvg').innerHTML = (d.fvg||[]).map(o=>row(o,['timeframe','direction','top','bottom','state','fvg_strength','gap_fill_percent','gap_size_pips','gap_age_bars'])).join('');
  document.getElementById('bos').innerHTML = (d.bos||[]).map(o=>row(o,['timeframe','direction','strength','bos_strength','bos_quality','bos_reliability','bos_distance_pips','bos_age_bars'])).join('');
  document.getElementById('choch').innerHTML=(d.choch||[]).map(o=>row(o,['timeframe','direction','kind','choch_strength','choch_quality','choch_reliability','choch_score','choch_age_bars'])).join('');
  document.getElementById('liq').innerHTML = (d.liquidity||[]).map(o=>row(o,['timeframe','kind','price','bound_low','bound_high','touches','density','liquidity_score','stop_hunt_probability','swept','sweep_kind'])).join('');
});
</script></body></html>"""


def _smc_data(symbol: str, limit_per_tf: int = 30) -> dict[str, Any]:
    if SMCRepository is None:
        return {"error": "smc_repository_unavailable", "symbol": symbol.upper()}
    try:
        repo = SMCRepository()
        data = repo.recent_for_dashboard(symbol.upper(), limit_per_tf=limit_per_tf)
        # AURION X 4.5 — anexa la ultima decision SMC desde el cache file-based
        # alimentado por el orchestrator. No tocamos la BD.
        try:
            data["smc_decision"] = _read_smc_decision_cache(symbol)
        except Exception as exc:
            logger.debug("smc_decision cache read fallo: %s", exc)
        return data
    except Exception as exc:
        logger.warning("smc_data %s: %s", symbol, exc)
        return {"error": str(exc), "symbol": symbol.upper()}


def _read_smc_decision_cache(symbol: str) -> dict[str, Any] | None:
    """Lee runtime/smc_decisions/<SYM>.json. Devuelve None si no existe."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "runtime" / "smc_decisions" / f"{symbol.upper()}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _smc_data_as_mt5_text(symbol: str, limit_per_tf: int = 30) -> str:
    """Formato pipe-separated v2 para el EA MQL5 ``AURION_SMC_Visualizer``.

    Todos los timestamps son **ISO 8601 con timezone UTC** (no aproximaciones
    por ``age_bars``), de forma que el EA pueda anclar los objetos al
    **candle exacto** vía ``StringToTime`` / ``iBarShift``.

    Cada línea representa un elemento SMC con un **ID estable** (UUID de PG):
    el EA usa ese ID para *tracking incremental* (no borrar/redibujar todo en
    cada refresh, evita flicker).

    Formato por kind::

        OB    |id|tf|dir|open_time|mitigation_time|bottom|top|state|strength|quality|freshness
        BOS   |id|tf|dir|broken_swing_time|break_time|broken_swing_price|break_price|strength|quality|reliability
        CHOCH |id|tf|dir|kind|broken_swing_time|break_time|broken_swing_price|break_price|strength|quality
        FVG   |id|tf|dir|open_time|inverted_at|bottom|top|state|strength|fill_pct
        LIQ   |id|tf|kind|open_time|swept_at|price|bound_low|bound_high|score|swept(0/1)
        CONF  ||||dir_bias|confluence_score|macro_bias|inst_bias|op_bias
        SMC_DECISION |ts_iso|smc_score|setup_grade|pd_state|pd_pct|ob_grade|fvg_grade|sweep_kind|mtf_aligned|allow|block_reason
        (v=3 añade SMC_DECISION; los EAs v2 ignoran lineas con prefijo desconocido)

    Campos vacíos van como ``""`` (sin valor). Headers ``#`` y terminador
    ``END`` permiten al parser MQL5 saber dónde acabar.
    """
    from datetime import datetime, timezone
    lines: list[str] = []
    lines.append(f"# AURION SMC {symbol.upper()} @ {datetime.now(timezone.utc).isoformat()}")
    lines.append("# v=2  fields: ID + ISO timestamps + state + strength")
    data = _smc_data(symbol, limit_per_tf=limit_per_tf)
    if data.get("error"):
        lines.append(f"# ERROR: {data['error']}")
        lines.append("END")
        return "\n".join(lines) + "\n"

    def _f(v: Any, default: str = "") -> str:
        if v is None:
            return default
        if isinstance(v, bool):
            return "1" if v else "0"
        return str(v)

    # Order Blocks
    for ob in (data.get("order_blocks") or []):
        lines.append("|".join([
            "OB",
            _f(ob.get("id")),
            _f(ob.get("timeframe")),
            _f(ob.get("direction")),
            _f(ob.get("open_time")),                  # ISO del candle origen
            _f(ob.get("mitigation_time")),            # ISO si está mitigado, "" si fresh
            _f(ob.get("bottom")),
            _f(ob.get("top")),
            _f(ob.get("state")),                      # FRESH / MITIGATED / INVALIDATED
            _f(ob.get("ob_strength")),
            _f(ob.get("ob_quality")),
            _f(ob.get("ob_freshness")),
        ]))
    # BOS
    for bos in (data.get("bos") or []):
        lines.append("|".join([
            "BOS",
            _f(bos.get("id")),
            _f(bos.get("timeframe")),
            _f(bos.get("direction")),
            _f(bos.get("broken_swing_time")),         # ISO swing roto
            _f(bos.get("break_time")),                # ISO punto de break
            _f(bos.get("broken_swing_price")),
            _f(bos.get("break_price")),
            _f(bos.get("bos_strength")),
            _f(bos.get("bos_quality")),
            _f(bos.get("bos_reliability")),
        ]))
    # CHOCH
    for ch in (data.get("choch") or []):
        lines.append("|".join([
            "CHOCH",
            _f(ch.get("id")),
            _f(ch.get("timeframe")),
            _f(ch.get("direction")),
            _f(ch.get("kind")),                       # MAJOR / MINOR
            _f(ch.get("broken_swing_time")),
            _f(ch.get("break_time")),
            _f(ch.get("broken_swing_price")),
            _f(ch.get("break_price")),
            _f(ch.get("choch_strength")),
            _f(ch.get("choch_quality")),
        ]))
    # FVG
    for fv in (data.get("fvg") or []):
        lines.append("|".join([
            "FVG",
            _f(fv.get("id")),
            _f(fv.get("timeframe")),
            _f(fv.get("direction")),
            _f(fv.get("open_time")),
            _f(fv.get("inverted_at")),                # "" si OPEN/PARTIAL
            _f(fv.get("bottom")),
            _f(fv.get("top")),
            _f(fv.get("state")),                      # OPEN / PARTIAL / FILLED / INVERTED
            _f(fv.get("fvg_strength")),
            _f(fv.get("gap_fill_percent")),
        ]))
    # Liquidity
    for lq in (data.get("liquidity") or []):
        lines.append("|".join([
            "LIQ",
            _f(lq.get("id")),
            _f(lq.get("timeframe")),
            _f(lq.get("kind")),                       # SSL/BSL/EQH/EQL/POOL
            _f(lq.get("open_time")),
            _f(lq.get("swept_at")),
            _f(lq.get("price")),
            _f(lq.get("bound_low")),
            _f(lq.get("bound_high")),
            _f(lq.get("liquidity_score")),
            _f(lq.get("swept")),
        ]))
    # Confluence (1 línea)
    conf = data.get("confluence")
    if conf:
        lines.append("|".join([
            "CONF",
            "", "", "",  # placeholders para alinear posiciones de los demás
            _f(conf.get("direction_bias")),
            _f(conf.get("confluence_score")),
            _f(conf.get("macro_bias")),
            _f(conf.get("institutional_bias")),
            _f(conf.get("operational_bias")),
        ]))
    # AURION X 4.5/4.6 — SMC Decision + Smart Targets + Smart Stops + Sessions.
    # Si no hay cache aun, se omite (el EA tolera ausencia).
    # v4 formato:
    #   SMC_DECISION|ts|score|grade|pd|pct|ob_q|fvg_q|sweep|mtf|allow|reason|status|qmode|vol_class|entry
    #   TP|rank|price|prob|src|tf|dist_pips|dist_atr
    #   SL|mode|price|dist_pips|dist_atr|stop_hunt_p|reason
    #   RR|sl_mode|tp1|tp2|tp3|tp4
    #   SESS|name|high|low|swept_hi|swept_lo|active
    # AURION X 4.6 — Sessions y volume_class se emiten SIEMPRE que estén en el
    # cache (ambient_only=True acepta sessions+volume aun sin decision).
    sdc = data.get("smc_decision")
    # AMBIENT: emitir SESS + VOL line aun sin decision (solo cuando hay sessions)
    if isinstance(sdc, dict) and not sdc.get("decision"):
        sess_pools_amb = sdc.get("session_pools") or {}
        vc_amb = sdc.get("volume_class")
        ep_amb = sdc.get("entry_price")
        if sess_pools_amb or vc_amb:
            # Linea AMBIENT|ts|vol_class|entry para que el EA pueda usar volumen
            lines.append("|".join([
                "AMBIENT",
                _f(sdc.get("generated_at")),
                _f(vc_amb),
                _f(ep_amb),
            ]))
            for sname, sinfo in sess_pools_amb.items():
                if not isinstance(sinfo, dict):
                    continue
                lines.append("|".join([
                    "SESS",
                    _f(sname),
                    _f(sinfo.get("high")),
                    _f(sinfo.get("low")),
                    _f(sinfo.get("swept_high")),
                    _f(sinfo.get("swept_low")),
                    _f(sinfo.get("is_active")),
                ]))
    if isinstance(sdc, dict) and sdc.get("decision"):
        dec = sdc["decision"]
        ob = dec.get("ob_quality") or {}
        fvg = dec.get("fvg_quality") or {}
        pd_s = dec.get("pd_state") or {}
        mtf = dec.get("mtf") or {}
        lines.append("|".join([
            "SMC_DECISION",
            _f(sdc.get("generated_at")),
            _f(dec.get("smc_score")),
            _f(dec.get("setup_grade")),
            _f(pd_s.get("state")),
            _f(pd_s.get("pct_in_range")),
            _f(ob.get("grade")),
            _f(fvg.get("grade")),
            _f(dec.get("sweep_kind")),
            _f(mtf.get("aligned_count")),
            _f(dec.get("allow_trade")),
            _f(dec.get("block_reason")),
            # AURION X 4.6: campos extra
            _f(dec.get("setup_status")),
            _f(dec.get("quality_mode")),
            _f(sdc.get("volume_class")),
            _f(sdc.get("entry_price")),
        ]))
        # Smart Targets TP1..TP4
        st_targets = sdc.get("smart_targets") or {}
        for tp in (st_targets.get("tps") or []):
            lines.append("|".join([
                "TP",
                _f(tp.get("rank")),
                _f(tp.get("price")),
                _f(tp.get("probability")),
                _f(tp.get("source")),
                _f(tp.get("timeframe")),
                _f(tp.get("distance_pips")),
                _f(tp.get("distance_atr")),
            ]))
        # Smart Stops 3 modos
        st_stops = sdc.get("smart_stops") or {}
        for mode_key in ("conservative", "balanced", "aggressive"):
            sl_m = st_stops.get(mode_key)
            if not isinstance(sl_m, dict):
                continue
            lines.append("|".join([
                "SL",
                _f(mode_key),
                _f(sl_m.get("price")),
                _f(sl_m.get("distance_pips")),
                _f(sl_m.get("distance_atr")),
                _f(sl_m.get("stop_hunt_probability")),
                _f(sl_m.get("reason")),
            ]))
        # RR table por modo (1 linea por modo, con tp1..tp4)
        rr_tbl = st_stops.get("rr_table") or {}
        for mode_key, rr_d in rr_tbl.items():
            if not isinstance(rr_d, dict):
                continue
            lines.append("|".join([
                "RR",
                _f(mode_key),
                _f(rr_d.get("tp1")),
                _f(rr_d.get("tp2")),
                _f(rr_d.get("tp3")),
                _f(rr_d.get("tp4")),
            ]))
        # SL recomendado
        rec_mode = st_stops.get("recommended_mode")
        if rec_mode:
            lines.append("|".join(["SL_REC", _f(rec_mode)]))
        # Sessions
        sess_pools = sdc.get("session_pools") or {}
        for sname, sinfo in sess_pools.items():
            if not isinstance(sinfo, dict):
                continue
            lines.append("|".join([
                "SESS",
                _f(sname),
                _f(sinfo.get("high")),
                _f(sinfo.get("low")),
                _f(sinfo.get("swept_high")),
                _f(sinfo.get("swept_low")),
                _f(sinfo.get("is_active")),
            ]))
    lines.append("END")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# AURION X 4.1 — XAU Knowledge Engine endpoints
# --------------------------------------------------------------------------- #


KNOWLEDGE_HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AURION X — XAU Knowledge __SYM__</title>
<style>
body{font-family:system-ui,sans-serif;background:#0f1419;color:#e6edf3;margin:24px}
h1{color:#58a6ff} h2{color:#9da7b3;margin-top:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.tag{display:inline-block;padding:2px 6px;background:#30363d;border-radius:4px;font-size:11px;margin-right:6px}
table{width:100%;border-collapse:collapse;font-size:12px;color:#e6edf3}
th{text-align:left;color:#9da7b3;border-bottom:1px solid #30363d;padding:4px 6px}
td{border-bottom:1px solid #21262d;padding:4px 6px}
.win{color:#66bb6a}
.loss{color:#ef5350}
.kv{display:flex;justify-content:space-between;padding:2px 0;font-size:13px}
.kv span:first-child{color:#9da7b3}
pre{overflow:auto;font-size:11px}
</style></head><body>
<h1>XAU Knowledge — <span id="sym">__SYM__</span></h1>
<p><a href="/" style="color:#58a6ff">← volver al dashboard</a> · <a href="/smc/__SYM__" style="color:#58a6ff">SMC view →</a></p>
<div class="grid">
  <div class="card"><h2>Patrones más rentables</h2><div id="patterns"></div></div>
  <div class="card"><h2>Eventos SMC por kind</h2><div id="smc"></div></div>
  <div class="card"><h2>Sesiones (winrate / avg-R)</h2><div id="sessions"></div></div>
  <div class="card"><h2>Regímenes</h2><div id="regimes"></div></div>
</div>
<div class="card" style="margin-top:16px"><h2>Reentrenamientos recientes</h2><div id="retrains"></div></div>
<script>
const sym=document.getElementById('sym').innerText;
function tbl(rows, cols){
  if(!rows||!rows.length){return '<em style="color:#9da7b3">sin datos</em>';}
  let h='<table><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
  rows.forEach(r=>{
    h+='<tr>';
    cols.forEach(c=>{
      let v=r[c];
      if(typeof v==='number'){v=v.toFixed(c==='winrate'?3:(c.indexOf('r')>=0?3:2));}
      if(v===null||v===undefined){v='-';}
      h+='<td>'+v+'</td>';
    });
    h+='</tr>';
  });
  return h+'</tbody></table>';
}
fetch('/api/knowledge/'+sym).then(r=>r.json()).then(d=>{
  document.getElementById('patterns').innerHTML=tbl(d.top_patterns||[], ['pattern_key','timeframe','direction','occurrences','wins','losses','winrate','avg_r']);
  document.getElementById('smc').innerHTML=tbl(d.top_smc_events||[], ['event_kind','timeframe','direction','n','avg_strength','avg_quality','winrate','avg_r']);
  document.getElementById('sessions').innerHTML=tbl(d.sessions||[], ['session','trades_n','winrate','avg_r','avg_atr','avg_range_pct','avg_spread_pts']);
  document.getElementById('regimes').innerHTML=tbl(d.regimes||[], ['regime','episodes','avg_duration_min','trades_n','winrate','avg_r']);
  document.getElementById('retrains').innerHTML=tbl(d.retrains||[], ['started_at','trigger','samples_used','walk_forward_folds','post_winrate','post_sharpe','post_expectancy','notes']);
});
</script></body></html>"""


def _knowledge_data(symbol: str, limit: int = 20) -> dict[str, Any]:
    try:
        from analytics.xau_knowledge_repository import XAUKnowledgeRepository
    except Exception as exc:
        return {"error": f"knowledge_repo_unavailable: {exc}", "symbol": symbol.upper()}
    try:
        repo = XAUKnowledgeRepository()
        return {
            "symbol": symbol.upper(),
            "top_patterns": repo.top_patterns(symbol=symbol, limit=limit, by="winrate"),
            "top_smc_events": repo.top_smc_events(symbol=symbol, limit=limit),
            "sessions": repo.session_summary(symbol=symbol, days_lookback=60),
            "regimes": repo.regime_summary(symbol=symbol),
            "retrains": repo.recent_retrains(symbol=symbol, limit=limit),
        }
    except Exception as exc:
        logger.warning("knowledge_data %s: %s", symbol, exc)
        return {"error": str(exc), "symbol": symbol.upper()}


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("dashboard %s", format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML_PAGE.encode())
            return
        if parsed.path == "/api/summary":
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", ["2000"])[0])
            data = _aggregate_trades(limit=limit)
            self._send(200, "application/json", json.dumps(data, default=str).encode())
            return
        # /api/smc/<SYMBOL> — JSON con todos los elementos SMC recientes
        if parsed.path.startswith("/api/smc/"):
            sym = parsed.path[len("/api/smc/"):].strip().upper() or "XAUUSD"
            # Formato MT5-friendly (pipe-separated, line-based).
            # Ejemplo: GET /api/smc/XAUUSD.txt
            if sym.endswith(".TXT"):
                sym = sym[:-4]
                qs = parse_qs(parsed.query)
                lim = int(qs.get("limit", ["20"])[0])
                body = _smc_data_as_mt5_text(sym, limit_per_tf=lim).encode()
                self._send(200, "text/plain; charset=utf-8", body)
                return
            qs = parse_qs(parsed.query)
            lim = int(qs.get("limit", ["50"])[0])
            data = _smc_data(sym, limit_per_tf=lim)
            self._send(200, "application/json", json.dumps(data, default=str).encode())
            return
        # /smc/<SYMBOL> — vista visual (HTML)
        if parsed.path.startswith("/smc/"):
            sym = parsed.path[len("/smc/"):].strip().upper() or "XAUUSD"
            html = SMC_HTML_PAGE.replace("__SYM__", sym)
            self._send(200, "text/html; charset=utf-8", html.encode())
            return
        # AURION X 4.1 — XAU Knowledge Engine
        if parsed.path.startswith("/api/knowledge/"):
            sym = parsed.path[len("/api/knowledge/"):].strip().upper() or "XAUUSD"
            qs = parse_qs(parsed.query)
            lim = int(qs.get("limit", ["20"])[0])
            data = _knowledge_data(sym, limit=lim)
            self._send(200, "application/json", json.dumps(data, default=str).encode())
            return
        if parsed.path.startswith("/knowledge/"):
            sym = parsed.path[len("/knowledge/"):].strip().upper() or "XAUUSD"
            html = KNOWLEDGE_HTML_PAGE.replace("__SYM__", sym)
            self._send(200, "text/html; charset=utf-8", html.encode())
            return
        self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_dashboard(host: str = "127.0.0.1", port: int = 8765) -> None:
    init_db()
    server = HTTPServer((host, port), DashboardHandler)
    logger.info("Dashboard en http://%s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    from analytics.logger import configure_root_logger

    configure_root_logger()
    run_dashboard()

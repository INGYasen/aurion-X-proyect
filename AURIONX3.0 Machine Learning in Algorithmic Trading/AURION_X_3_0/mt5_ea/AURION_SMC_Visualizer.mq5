//+------------------------------------------------------------------+
//|                                       AURION_SMC_Visualizer.mq5  |
//|        AURION X 4.1 - v3 PROFESSIONAL                              |
//|                                                                    |
//|   Nueva filosofía visual:                                           |
//|   - "Menos es más": solo lo institucionalmente relevante.           |
//|   - BOS / CHOCH: solo el más reciente por TF (1 por TF).            |
//|   - OB: top 3 GLOBAL por strength (entre TFs).                      |
//|   - FVG: top 5 GLOBAL no llenos.                                    |
//|   - LIQ: solo no-swept del TF primario (configurable).              |
//|   - Panel lateral resumen en esquina derecha-superior con           |
//|     bias, confluence, conteos por TF y últimos eventos.             |
//|                                                                    |
//|   Características técnicas:                                         |
//|   - Anchors por timestamp ISO -> precisión candle.                  |
//|   - Tracking incremental (anti-flicker).                            |
//|   - Estado-color: FRESH saturado, MITIGATED claro.                  |
//|   - Width por strength.                                             |
//|                                                                    |
//|   INSTRUCCIONES MT5                                                 |
//|   1) Copia a <MT5>\MQL5\Experts\ y compila (F7)                     |
//|   2) Tools > Options > Expert Advisors: Allow WebRequest            |
//|      + añadir URL http://127.0.0.1:8765                             |
//|   3) Arrastrar al chart XAUUSD (M15 recomendado)                    |
//+------------------------------------------------------------------+
#property copyright "AURION X 4.6.1"
#property version   "4.61"
#property strict
#property description "Visualizador SMC v4.6.1 - Decision + Heur + Smart TP/SL + Sessions ambient + FVG/OB far-filter."

//================ Inputs =====================================================
input string  DashboardHost      = "127.0.0.1";
input int     DashboardPort      = 8765;
input string  Symbol_            = "XAUUSD";
input int     RefreshSeconds     = 6;
input int     RequestTimeoutMs   = 3500;
input int     LimitPerTF         = 25;

// What to draw (categorías)
input bool    DrawOB             = true;
input bool    DrawBOS            = true;
input bool    DrawCHOCH          = true;
input bool    DrawFVG            = true;
input bool    DrawLIQ            = true;

// === NUEVOS LÍMITES PROFESIONALES (top-K filtrado) ====================
input int     TopOB_Global       = 3;     // 3 OBs más fuertes en total (entre todos los TFs)
input int     TopBOS_PerTF       = 1;     // 1 BOS más reciente por TF
input int     TopCHOCH_PerTF     = 1;     // 1 CHOCH más reciente por TF
input int     TopFVG_Global      = 5;     // 5 FVGs más fuertes en total (no llenos)
input int     TopLIQ_PrimaryTF   = 4;     // 4 niveles de liquidez del TF primario
input string  PrimaryTF_Hint     = "M15"; // TF preferente para liquidez y resumen
input bool    DrawInvalidated    = false; // mostrar invalidados/llenos
input bool    DrawMitigated      = true;  // mostrar mitigados (color tenue)

// Filters
input double  MinStrength        = 0.30;
input int     MaxAgeBars         = 600;
input int     ExtendBars         = 80;
// AURION X 4.6.1: filtro de FVG/OB lejanos al precio (anti-clutter visual)
// 0.05 = 5% de distancia maxima entre el nivel y el precio actual.
// 0.0 desactiva el filtro (mantiene todos los niveles, comportamiento previo).
input double  MaxFVG_DistancePct = 0.05;
input double  MaxOB_DistancePct  = 0.10;

// Labels visibilidad (anti-clutter)
input bool    ShowLabels         = true;  // si false, solo dibuja figuras sin texto
input bool    ShortLabels        = true;  // "OB M15 ↑ 0.92" en vez de label largo
input int     LabelFontSize      = 8;

// Colors - FRESH (saturados)
input color   OBBullColor        = clrSeaGreen;
input color   OBBearColor        = clrCrimson;
input color   FVGBullColor       = clrTeal;
input color   FVGBearColor       = clrMaroon;
input color   BOSColor           = clrDodgerBlue;
input color   CHOCHColor         = clrGold;
input color   LIQColor           = clrViolet;
// Colors - MITIGATED (claros)
input color   OBBullMitColor     = clrDarkSeaGreen;
input color   OBBearMitColor     = clrLightSalmon;
input color   FVGBullMitColor    = clrLightSeaGreen;
input color   FVGBearMitColor    = clrDarkSalmon;

// Panel lateral
input bool    ShowSummaryPanel   = true;
input int     PanelXDistance     = 12;
input int     PanelYDistance     = 24;
input ENUM_BASE_CORNER PanelCorner = CORNER_RIGHT_UPPER;
input color   PanelTitleColor    = clrGold;
input color   PanelTextColor     = clrWhiteSmoke;
input color   PanelBullColor     = clrLimeGreen;
input color   PanelBearColor     = clrTomato;
// AURION X 4.4 (10-jun-2026): legibilidad mejorada — font 10, line-height
// extra, fondo opaco para destacar sobre velas, y gaps mas amplios entre
// secciones (antes el texto se veia "pegado" sobre las velas oscuras).
// AURION X 4.5.3 (11-jun-2026): legibilidad mejorada del panel
// - fuente Courier New (siempre disponible en Windows, garantizado monospace)
// - tamano +1, line spacing +2, padding +4, ancho panel +60px
// - mas espacio entre columnas hdr/rows
input int     PanelFontSize      = 11;         // antes 10
input int     PanelLineSpacing   = 10;         // antes 8 (extra px entre lineas)
input int     PanelSectionGap    = 14;         // antes 10 (gap entre secciones)
input bool    PanelShowBackground = true;      // dibujar fondo del panel
input color   PanelBgColor       = C'18,18,22'; // gris carbon casi negro
input color   PanelBorderColor   = C'70,70,90'; // borde sutil
input int     PanelPadding       = 16;         // antes 12 (padding interno)
input int     PanelWidth         = 400;        // antes 340 (ancho fijo del fondo)
input string  PanelFont          = "Courier New"; // monoespaciada estandar de Windows

//================ Globals ====================================================
const string  OBJ_PREFIX     = "AURION_SMC_";
const string  PANEL_PREFIX   = "AURION_PANEL_";
datetime      g_lastFetch    = 0;
int           g_drawn        = 0;
int           g_pruned       = 0;

string        g_active_ids[];
int           g_active_count = 0;

// ==== Storage de elementos parseados (parallel arrays) ====================
// OB
string g_ob_id[];   string g_ob_tf[];    string g_ob_dir[];   string g_ob_state[];
datetime g_ob_t0[]; datetime g_ob_tm[];  double g_ob_bot[];   double g_ob_top[];
double g_ob_st[];   double g_ob_qu[];    double g_ob_fr[];
int g_ob_n = 0;
// BOS
string g_bos_id[];  string g_bos_tf[];   string g_bos_dir[];
datetime g_bos_tsw[]; datetime g_bos_tbk[];
double g_bos_psw[]; double g_bos_pbk[];  double g_bos_st[]; double g_bos_qu[]; double g_bos_rl[];
int g_bos_n = 0;
// CHOCH
string g_cho_id[];  string g_cho_tf[];   string g_cho_dir[];  string g_cho_kind[];
datetime g_cho_tsw[]; datetime g_cho_tbk[];
double g_cho_psw[]; double g_cho_pbk[];  double g_cho_st[]; double g_cho_qu[];
int g_cho_n = 0;
// FVG
string g_fvg_id[];  string g_fvg_tf[];   string g_fvg_dir[];  string g_fvg_state[];
datetime g_fvg_t0[]; datetime g_fvg_tinv[];
double g_fvg_bot[]; double g_fvg_top[];  double g_fvg_st[]; double g_fvg_fp[];
int g_fvg_n = 0;
// LIQ
string g_liq_id[];  string g_liq_tf[];   string g_liq_kind[];
datetime g_liq_t0[]; datetime g_liq_tsw[];
double g_liq_p[];   double g_liq_bl[];   double g_liq_bh[]; double g_liq_sc[];
bool g_liq_sw[];
int g_liq_n = 0;
// CONF
bool   g_has_conf       = false;
string g_conf_dirBias   = "NEUTRAL";
double g_conf_score     = 0.0;
string g_conf_macro     = "NEUTRAL";
string g_conf_inst      = "NEUTRAL";
string g_conf_op        = "NEUTRAL";
// AURION X 4.5 - SMC_DECISION (engine institucional)
bool   g_has_decision    = false;
string g_dec_ts          = "";
double g_dec_score       = 0.0;
string g_dec_grade       = "X";
string g_dec_pd_state    = "UNDEFINED";
double g_dec_pd_pct      = 0.5;
string g_dec_ob_grade    = "X";
string g_dec_fvg_grade   = "X";
string g_dec_sweep_kind  = "NONE";
int    g_dec_mtf_aligned = 0;
bool   g_dec_allow       = false;
string g_dec_block_reason= "";
// AURION X 4.6 - SMC_DECISION extra fields
string g_dec_status      = "NO_TRADE";
string g_dec_qmode       = "normal";
string g_dec_vol_class   = "UNKNOWN";
double g_dec_entry_px    = 0.0;

// AURION X 4.6 - Smart Targets (max 4 TPs)
int    g_tp_n            = 0;
int    g_tp_rank[4];
double g_tp_price[4];
double g_tp_prob[4];
string g_tp_src[4];
string g_tp_tf[4];
double g_tp_dist_pips[4];

// AURION X 4.6 - Smart Stops (3 modes)
bool   g_has_sl_cons     = false;
bool   g_has_sl_bal      = false;
bool   g_has_sl_agg      = false;
double g_sl_cons_price   = 0.0;
double g_sl_bal_price    = 0.0;
double g_sl_agg_price    = 0.0;
double g_sl_cons_pips    = 0.0;
double g_sl_bal_pips     = 0.0;
double g_sl_agg_pips     = 0.0;
double g_sl_cons_hunt    = 0.0;
double g_sl_bal_hunt     = 0.0;
double g_sl_agg_hunt     = 0.0;
string g_sl_recommended  = "balanced";

// AURION X 4.6 - RR table (3 modes x 4 TPs)
double g_rr_cons[4];
double g_rr_bal[4];
double g_rr_agg[4];

// AURION X 4.6 - Sessions (Asia/London/NY/Overlap)
string g_sess_name[6];
double g_sess_high[6];
double g_sess_low[6];
bool   g_sess_sw_hi[6];
bool   g_sess_sw_lo[6];
bool   g_sess_active[6];
int    g_sess_n         = 0;

//+------------------------------------------------------------------+
//| OnInit / OnDeinit / OnTimer                                       |
//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(MathMax(1, RefreshSeconds));
   PrintFormat("[AURION SMC v4.6.1] http://%s:%d sym=%s refresh=%ds Top: OB=%d BOS/TF=%d CHOCH/TF=%d FVG=%d LIQ@%s=%d",
               DashboardHost, DashboardPort, Symbol_, RefreshSeconds,
               TopOB_Global, TopBOS_PerTF, TopCHOCH_PerTF, TopFVG_Global, PrimaryTF_Hint, TopLIQ_PrimaryTF);
   PrintFormat("[AURION SMC v4.6.1] panel: font='%s' (min CourierNew) size>=11 width>=460 padding>=16",
               PanelFont);
   PrintFormat("[AURION SMC v4.6.1] far-filter: FVG<=%.1f%% OB<=%.1f%% del precio actual",
               MaxFVG_DistancePct*100.0, MaxOB_DistancePct*100.0);
   Print("[AURION SMC v4.6.1] ambient: Sessions+Volume visibles SIEMPRE (sin esperar signal)");
   FetchAndDraw();
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ClearAll();
   PrintFormat("[AURION SMC v3] detenido (reason=%d).", reason);
  }

void OnTimer() { FetchAndDraw(); }

//+------------------------------------------------------------------+
//| FetchAndDraw                                                      |
//+------------------------------------------------------------------+
void FetchAndDraw()
  {
   string url = StringFormat("http://%s:%d/api/smc/%s.txt?limit=%d",
                             DashboardHost, DashboardPort, Symbol_, LimitPerTF);
   string body = "";
   if(!HttpGet(url, body))
     {
      PrintFormat("[AURION SMC v3] WebRequest FAIL url=%s", url);
      return;
     }
   if(StringLen(body) < 5)
     {
      Print("[AURION SMC v3] respuesta vacía.");
      return;
     }

   ArrayResize(g_active_ids, 0);
   g_active_count = 0;

   // 1) Parse: NO dibujar todavía, solo poblar arrays
   ResetStorage();
   ParseAllLines(body);

   // 2) Filter + sort: aplicar top-K
   //    - OBs: ordenar por strength desc, tomar TopOB_Global
   //    - BOS/CHOCH: por TF, tomar el más reciente (TopBOS_PerTF / TopCHOCH_PerTF)
   //    - FVG: por strength desc, no llenos, tomar TopFVG_Global
   //    - LIQ: solo del TF primario, no-swept, top score
   bool sel_ob[];    ApplyTopK_OB(sel_ob);
   bool sel_bos[];   ApplyTopK_PerTF(g_bos_n, g_bos_tf, g_bos_tsw, TopBOS_PerTF, sel_bos);
   bool sel_choch[]; ApplyTopK_PerTF(g_cho_n, g_cho_tf, g_cho_tsw, TopCHOCH_PerTF, sel_choch);
   bool sel_fvg[];   ApplyTopK_FVG(sel_fvg);
   bool sel_liq[];   ApplyTopK_LIQ(sel_liq);

   // 3) Anti-solape buckets compartidos entre todas las categorías
   double bucketsPx[];
   int    bucketsN = 0;
   double atr = ChartATRApprox();
   double bucketStep = MathMax(atr * 0.20, _Point * 50);

   int drawn = 0;
   if(DrawOB)    drawn += DrawAllOBs(sel_ob, bucketsPx, bucketsN, bucketStep);
   if(DrawBOS)   drawn += DrawAllBOS(sel_bos, bucketsPx, bucketsN, bucketStep);
   if(DrawCHOCH) drawn += DrawAllCHOCH(sel_choch, bucketsPx, bucketsN, bucketStep);
   if(DrawFVG)   drawn += DrawAllFVG(sel_fvg, bucketsPx, bucketsN, bucketStep);
   if(DrawLIQ)   drawn += DrawAllLIQ(sel_liq, bucketsPx, bucketsN, bucketStep);

   // 4) Panel lateral resumen
   if(ShowSummaryPanel) RenderSummaryPanel();

   // 5) Prune: borrar lo que ya no esté activo
   int pruned = PruneStaleObjects();
   g_drawn  = drawn;
   g_pruned = pruned;
   g_lastFetch = TimeCurrent();
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
//| Parser                                                            |
//+------------------------------------------------------------------+
void ParseAllLines(const string body)
  {
   string lines[];
   int n = StringSplit(body, '\n', lines);
   for(int i = 0; i < n; i++)
     {
      string ln = TrimAll(lines[i]);
      if(StringLen(ln) == 0) continue;
      if(StringFind(ln, "#") == 0) continue;
      if(ln == "END") break;
      string p[];
      int np = StringSplit(ln, '|', p);
      if(np < 3) continue;
      string kind = p[0];
      if(kind == "OB")    StoreOB(p, np);
      else if(kind == "BOS")   StoreBOS(p, np);
      else if(kind == "CHOCH") StoreCHOCH(p, np);
      else if(kind == "FVG")   StoreFVG(p, np);
      else if(kind == "LIQ")   StoreLIQ(p, np);
      else if(kind == "CONF")  StoreCONF(p, np);
      else if(kind == "SMC_DECISION") StoreDECISION(p, np);
      else if(kind == "TP")    StoreTP(p, np);
      else if(kind == "SL")    StoreSL(p, np);
      else if(kind == "RR")    StoreRR(p, np);
      else if(kind == "SL_REC") StoreSLREC(p, np);
      else if(kind == "SESS")  StoreSESS(p, np);
      else if(kind == "AMBIENT") StoreAMBIENT(p, np);
     }
  }

void ResetStorage()
  {
   ArrayResize(g_ob_id,0); ArrayResize(g_ob_tf,0); ArrayResize(g_ob_dir,0); ArrayResize(g_ob_state,0);
   ArrayResize(g_ob_t0,0); ArrayResize(g_ob_tm,0); ArrayResize(g_ob_bot,0); ArrayResize(g_ob_top,0);
   ArrayResize(g_ob_st,0); ArrayResize(g_ob_qu,0); ArrayResize(g_ob_fr,0);
   g_ob_n = 0;
   ArrayResize(g_bos_id,0); ArrayResize(g_bos_tf,0); ArrayResize(g_bos_dir,0);
   ArrayResize(g_bos_tsw,0); ArrayResize(g_bos_tbk,0);
   ArrayResize(g_bos_psw,0); ArrayResize(g_bos_pbk,0); ArrayResize(g_bos_st,0);
   ArrayResize(g_bos_qu,0); ArrayResize(g_bos_rl,0);
   g_bos_n = 0;
   ArrayResize(g_cho_id,0); ArrayResize(g_cho_tf,0); ArrayResize(g_cho_dir,0); ArrayResize(g_cho_kind,0);
   ArrayResize(g_cho_tsw,0); ArrayResize(g_cho_tbk,0);
   ArrayResize(g_cho_psw,0); ArrayResize(g_cho_pbk,0); ArrayResize(g_cho_st,0); ArrayResize(g_cho_qu,0);
   g_cho_n = 0;
   ArrayResize(g_fvg_id,0); ArrayResize(g_fvg_tf,0); ArrayResize(g_fvg_dir,0); ArrayResize(g_fvg_state,0);
   ArrayResize(g_fvg_t0,0); ArrayResize(g_fvg_tinv,0);
   ArrayResize(g_fvg_bot,0); ArrayResize(g_fvg_top,0); ArrayResize(g_fvg_st,0); ArrayResize(g_fvg_fp,0);
   g_fvg_n = 0;
   ArrayResize(g_liq_id,0); ArrayResize(g_liq_tf,0); ArrayResize(g_liq_kind,0);
   ArrayResize(g_liq_t0,0); ArrayResize(g_liq_tsw,0);
   ArrayResize(g_liq_p,0); ArrayResize(g_liq_bl,0); ArrayResize(g_liq_bh,0); ArrayResize(g_liq_sc,0);
   ArrayResize(g_liq_sw,0);
   g_liq_n = 0;
   g_has_conf = false;
   g_has_decision = false;
   g_dec_grade = "X";
   g_dec_pd_state = "UNDEFINED";
   g_dec_ob_grade = "X";
   g_dec_fvg_grade = "X";
   g_dec_sweep_kind = "NONE";
   g_dec_mtf_aligned = 0;
   g_dec_allow = false;
   g_dec_block_reason = "";
   // AURION X 4.6 - reset extra storage
   g_dec_status = "NO_TRADE";
   g_dec_qmode = "normal";
   g_dec_vol_class = "UNKNOWN";
   g_dec_entry_px = 0.0;
   g_tp_n = 0;
   for(int k = 0; k < 4; k++)
     {
      g_tp_rank[k] = 0; g_tp_price[k] = 0; g_tp_prob[k] = 0;
      g_tp_src[k] = ""; g_tp_tf[k] = ""; g_tp_dist_pips[k] = 0;
      g_rr_cons[k] = 0; g_rr_bal[k] = 0; g_rr_agg[k] = 0;
     }
   g_has_sl_cons = false; g_has_sl_bal = false; g_has_sl_agg = false;
   g_sl_cons_price = 0; g_sl_bal_price = 0; g_sl_agg_price = 0;
   g_sl_cons_pips = 0;  g_sl_bal_pips = 0;  g_sl_agg_pips = 0;
   g_sl_cons_hunt = 0;  g_sl_bal_hunt = 0;  g_sl_agg_hunt = 0;
   g_sl_recommended = "balanced";
   g_sess_n = 0;
   for(int s = 0; s < 6; s++)
     {
      g_sess_name[s] = ""; g_sess_high[s] = 0; g_sess_low[s] = 0;
      g_sess_sw_hi[s] = false; g_sess_sw_lo[s] = false; g_sess_active[s] = false;
     }
  }

//+------------------------------------------------------------------+
//| Store: parse cada línea al storage                                |
//+------------------------------------------------------------------+
void StoreOB(const string &p[], int np)
  {
   if(np < 12) return;
   string state = p[8];
   if(state == "INVALIDATED" && !DrawInvalidated) return;
   if(state == "MITIGATED" && !DrawMitigated) return;
   double st = StrToDoubleOrZero(p[9]);
   if(st < MinStrength) return;
   datetime t0 = IsoToTime(p[4]);
   if(!AgeOk(p[2], t0)) return;
   // AURION X 4.6.1: descartar OBs muy lejanos al precio actual (anti-clutter)
   if(MaxOB_DistancePct > 0.0)
     {
      double bot = StrToDoubleOrZero(p[6]);
      double top = StrToDoubleOrZero(p[7]);
      double mid = (bot + top) * 0.5;
      double cur = SymbolInfoDouble(Symbol_, SYMBOL_BID);
      if(cur > 0.0 && mid > 0.0)
        {
         double pctDiff = MathAbs(mid - cur) / cur;
         if(pctDiff > MaxOB_DistancePct) return;
        }
     }
   int i = g_ob_n;
   ArrayResize(g_ob_id,i+1);   g_ob_id[i]=p[1];
   ArrayResize(g_ob_tf,i+1);   g_ob_tf[i]=p[2];
   ArrayResize(g_ob_dir,i+1);  g_ob_dir[i]=p[3];
   ArrayResize(g_ob_state,i+1);g_ob_state[i]=state;
   ArrayResize(g_ob_t0,i+1);   g_ob_t0[i]=t0;
   ArrayResize(g_ob_tm,i+1);   g_ob_tm[i]=IsoToTime(p[5]);
   ArrayResize(g_ob_bot,i+1);  g_ob_bot[i]=StrToDoubleOrZero(p[6]);
   ArrayResize(g_ob_top,i+1);  g_ob_top[i]=StrToDoubleOrZero(p[7]);
   ArrayResize(g_ob_st,i+1);   g_ob_st[i]=st;
   ArrayResize(g_ob_qu,i+1);   g_ob_qu[i]=StrToDoubleOrZero(p[10]);
   ArrayResize(g_ob_fr,i+1);   g_ob_fr[i]=StrToDoubleOrZero(p[11]);
   g_ob_n++;
  }
void StoreBOS(const string &p[], int np)
  {
   if(np < 11) return;
   double st = StrToDoubleOrZero(p[8]);
   if(st < MinStrength) return;
   datetime tSw = IsoToTime(p[4]);
   if(!AgeOk(p[2], tSw)) return;
   int i = g_bos_n;
   ArrayResize(g_bos_id,i+1);  g_bos_id[i]=p[1];
   ArrayResize(g_bos_tf,i+1);  g_bos_tf[i]=p[2];
   ArrayResize(g_bos_dir,i+1); g_bos_dir[i]=p[3];
   ArrayResize(g_bos_tsw,i+1); g_bos_tsw[i]=tSw;
   ArrayResize(g_bos_tbk,i+1); g_bos_tbk[i]=IsoToTime(p[5]);
   ArrayResize(g_bos_psw,i+1); g_bos_psw[i]=StrToDoubleOrZero(p[6]);
   ArrayResize(g_bos_pbk,i+1); g_bos_pbk[i]=StrToDoubleOrZero(p[7]);
   ArrayResize(g_bos_st,i+1);  g_bos_st[i]=st;
   ArrayResize(g_bos_qu,i+1);  g_bos_qu[i]=StrToDoubleOrZero(p[9]);
   ArrayResize(g_bos_rl,i+1);  g_bos_rl[i]=StrToDoubleOrZero(p[10]);
   g_bos_n++;
  }
void StoreCHOCH(const string &p[], int np)
  {
   if(np < 11) return;
   double st = StrToDoubleOrZero(p[9]);
   if(st < MinStrength) return;
   datetime tSw = IsoToTime(p[5]);
   if(!AgeOk(p[2], tSw)) return;
   int i = g_cho_n;
   ArrayResize(g_cho_id,i+1);  g_cho_id[i]=p[1];
   ArrayResize(g_cho_tf,i+1);  g_cho_tf[i]=p[2];
   ArrayResize(g_cho_dir,i+1); g_cho_dir[i]=p[3];
   ArrayResize(g_cho_kind,i+1);g_cho_kind[i]=p[4];
   ArrayResize(g_cho_tsw,i+1); g_cho_tsw[i]=tSw;
   ArrayResize(g_cho_tbk,i+1); g_cho_tbk[i]=IsoToTime(p[6]);
   ArrayResize(g_cho_psw,i+1); g_cho_psw[i]=StrToDoubleOrZero(p[7]);
   ArrayResize(g_cho_pbk,i+1); g_cho_pbk[i]=StrToDoubleOrZero(p[8]);
   ArrayResize(g_cho_st,i+1);  g_cho_st[i]=st;
   ArrayResize(g_cho_qu,i+1);  g_cho_qu[i]=StrToDoubleOrZero(p[10]);
   g_cho_n++;
  }
void StoreFVG(const string &p[], int np)
  {
   if(np < 11) return;
   string state = p[8];
   if(state == "FILLED" && !DrawInvalidated) return;
   double st = StrToDoubleOrZero(p[9]);
   if(st < MinStrength) return;
   datetime t0 = IsoToTime(p[4]);
   if(!AgeOk(p[2], t0)) return;
   // AURION X 4.6.1: descartar FVGs muy lejanos al precio actual (anti-clutter)
   if(MaxFVG_DistancePct > 0.0)
     {
      double bot = StrToDoubleOrZero(p[6]);
      double top = StrToDoubleOrZero(p[7]);
      double mid = (bot + top) * 0.5;
      double cur = SymbolInfoDouble(Symbol_, SYMBOL_BID);
      if(cur > 0.0 && mid > 0.0)
        {
         double pctDiff = MathAbs(mid - cur) / cur;
         if(pctDiff > MaxFVG_DistancePct) return;
        }
     }
   int i = g_fvg_n;
   ArrayResize(g_fvg_id,i+1);   g_fvg_id[i]=p[1];
   ArrayResize(g_fvg_tf,i+1);   g_fvg_tf[i]=p[2];
   ArrayResize(g_fvg_dir,i+1);  g_fvg_dir[i]=p[3];
   ArrayResize(g_fvg_state,i+1);g_fvg_state[i]=state;
   ArrayResize(g_fvg_t0,i+1);   g_fvg_t0[i]=t0;
   ArrayResize(g_fvg_tinv,i+1); g_fvg_tinv[i]=IsoToTime(p[5]);
   ArrayResize(g_fvg_bot,i+1);  g_fvg_bot[i]=StrToDoubleOrZero(p[6]);
   ArrayResize(g_fvg_top,i+1);  g_fvg_top[i]=StrToDoubleOrZero(p[7]);
   ArrayResize(g_fvg_st,i+1);   g_fvg_st[i]=st;
   ArrayResize(g_fvg_fp,i+1);   g_fvg_fp[i]=StrToDoubleOrZero(p[10]);
   g_fvg_n++;
  }
void StoreLIQ(const string &p[], int np)
  {
   if(np < 11) return;
   double sc = StrToDoubleOrZero(p[9]);
   if(sc < MinStrength) return;
   datetime t0 = IsoToTime(p[4]);
   if(!AgeOk(p[2], t0)) return;
   bool sw = (p[10] == "1" || p[10] == "True" || p[10] == "true");
   int i = g_liq_n;
   ArrayResize(g_liq_id,i+1);  g_liq_id[i]=p[1];
   ArrayResize(g_liq_tf,i+1);  g_liq_tf[i]=p[2];
   ArrayResize(g_liq_kind,i+1);g_liq_kind[i]=p[3];
   ArrayResize(g_liq_t0,i+1);  g_liq_t0[i]=t0;
   ArrayResize(g_liq_tsw,i+1); g_liq_tsw[i]=IsoToTime(p[5]);
   ArrayResize(g_liq_p,i+1);   g_liq_p[i]=StrToDoubleOrZero(p[6]);
   ArrayResize(g_liq_bl,i+1);  g_liq_bl[i]=StrToDoubleOrZero(p[7]);
   ArrayResize(g_liq_bh,i+1);  g_liq_bh[i]=StrToDoubleOrZero(p[8]);
   ArrayResize(g_liq_sc,i+1);  g_liq_sc[i]=sc;
   ArrayResize(g_liq_sw,i+1);  g_liq_sw[i]=sw;
   g_liq_n++;
  }
void StoreCONF(const string &p[], int np)
  {
   if(np < 9) return;
   g_has_conf       = true;
   g_conf_dirBias   = p[4];
   g_conf_score     = StrToDoubleOrZero(p[5]);
   g_conf_macro     = p[6];
   g_conf_inst      = p[7];
   g_conf_op        = p[8];
  }

// AURION X 4.5/4.6
// v3: SMC_DECISION|ts|score|grade|pd_state|pd_pct|ob|fvg|sweep|mtf|allow|reason
// v4: + status|qmode|vol_class|entry
void StoreDECISION(const string &p[], int np)
  {
   if(np < 11) return;
   g_has_decision     = true;
   g_dec_ts           = p[1];
   g_dec_score        = StrToDoubleOrZero(p[2]);
   g_dec_grade        = p[3];
   g_dec_pd_state     = p[4];
   g_dec_pd_pct       = StrToDoubleOrZero(p[5]);
   g_dec_ob_grade     = p[6];
   g_dec_fvg_grade    = p[7];
   g_dec_sweep_kind   = p[8];
   g_dec_mtf_aligned  = (int)StrToDoubleOrZero(p[9]);
   string allow_s     = p[10];
   g_dec_allow        = (allow_s == "1" || allow_s == "True" || allow_s == "true");
   if(np >= 12) g_dec_block_reason = p[11]; else g_dec_block_reason = "";
   // AURION X 4.6 (campos extra v4, tolerantes a versiones viejas)
   if(np >= 13) g_dec_status    = p[12];
   if(np >= 14) g_dec_qmode     = p[13];
   if(np >= 15) g_dec_vol_class = p[14];
   if(np >= 16) g_dec_entry_px  = StrToDoubleOrZero(p[15]);
  }

// AURION X 4.6 - TP|rank|price|prob|src|tf|dist_pips|dist_atr
void StoreTP(const string &p[], int np)
  {
   if(np < 4) return;
   if(g_tp_n >= 4) return;
   int rank = (int)StrToDoubleOrZero(p[1]);
   if(rank < 1 || rank > 4) return;
   int i = g_tp_n;
   g_tp_rank[i]      = rank;
   g_tp_price[i]     = StrToDoubleOrZero(p[2]);
   g_tp_prob[i]      = StrToDoubleOrZero(p[3]);
   g_tp_src[i]       = (np > 4) ? p[4] : "";
   g_tp_tf[i]        = (np > 5) ? p[5] : "";
   g_tp_dist_pips[i] = (np > 6) ? StrToDoubleOrZero(p[6]) : 0.0;
   g_tp_n++;
  }

// AURION X 4.6 - SL|mode|price|dist_pips|dist_atr|stop_hunt_p|reason
void StoreSL(const string &p[], int np)
  {
   if(np < 6) return;
   string mode = p[1];
   double price = StrToDoubleOrZero(p[2]);
   double pips  = StrToDoubleOrZero(p[3]);
   double hunt  = StrToDoubleOrZero(p[5]);
   if(mode == "conservative")
     { g_has_sl_cons = true; g_sl_cons_price = price; g_sl_cons_pips = pips; g_sl_cons_hunt = hunt; }
   else if(mode == "balanced")
     { g_has_sl_bal  = true; g_sl_bal_price  = price; g_sl_bal_pips  = pips; g_sl_bal_hunt  = hunt; }
   else if(mode == "aggressive")
     { g_has_sl_agg  = true; g_sl_agg_price  = price; g_sl_agg_pips  = pips; g_sl_agg_hunt  = hunt; }
  }

// AURION X 4.6 - RR|sl_mode|tp1|tp2|tp3|tp4
void StoreRR(const string &p[], int np)
  {
   if(np < 6) return;
   string mode = p[1];
   double t1 = StrToDoubleOrZero(p[2]);
   double t2 = StrToDoubleOrZero(p[3]);
   double t3 = StrToDoubleOrZero(p[4]);
   double t4 = StrToDoubleOrZero(p[5]);
   if(mode == "conservative")
     { g_rr_cons[0]=t1; g_rr_cons[1]=t2; g_rr_cons[2]=t3; g_rr_cons[3]=t4; }
   else if(mode == "balanced")
     { g_rr_bal[0]=t1; g_rr_bal[1]=t2; g_rr_bal[2]=t3; g_rr_bal[3]=t4; }
   else if(mode == "aggressive")
     { g_rr_agg[0]=t1; g_rr_agg[1]=t2; g_rr_agg[2]=t3; g_rr_agg[3]=t4; }
  }

// AURION X 4.6 - SL_REC|mode
void StoreSLREC(const string &p[], int np)
  {
   if(np < 2) return;
   g_sl_recommended = p[1];
  }

// AURION X 4.6 - AMBIENT|ts|vol_class|entry  (sessions+volumen sin decision)
void StoreAMBIENT(const string &p[], int np)
  {
   if(np < 4) return;
   g_dec_ts        = p[1];
   if(StringLen(p[2]) > 0) g_dec_vol_class = p[2];
   if(StringLen(p[3]) > 0) g_dec_entry_px  = StrToDoubleOrZero(p[3]);
  }

// AURION X 4.6 - SESS|name|high|low|swept_hi|swept_lo|active
void StoreSESS(const string &p[], int np)
  {
   if(np < 7) return;
   if(g_sess_n >= 6) return;
   int i = g_sess_n;
   g_sess_name[i]   = p[1];
   g_sess_high[i]   = StrToDoubleOrZero(p[2]);
   g_sess_low[i]    = StrToDoubleOrZero(p[3]);
   g_sess_sw_hi[i]  = (p[4] == "1" || p[4] == "True" || p[4] == "true");
   g_sess_sw_lo[i]  = (p[5] == "1" || p[5] == "True" || p[5] == "true");
   g_sess_active[i] = (p[6] == "1" || p[6] == "True" || p[6] == "true");
   g_sess_n++;
  }

//+------------------------------------------------------------------+
//| Top-K selectors                                                   |
//+------------------------------------------------------------------+
void ApplyTopK_OB(bool &selected[])
  {
   ArrayResize(selected, g_ob_n);
   for(int i = 0; i < g_ob_n; i++) selected[i] = false;
   // Tomar los top-K por strength globalmente.
   int picked = 0;
   bool taken[]; ArrayResize(taken, g_ob_n);
   for(int i = 0; i < g_ob_n; i++) taken[i] = false;
   while(picked < TopOB_Global)
     {
      int best = -1;
      double bestSt = -1.0;
      for(int j = 0; j < g_ob_n; j++)
        {
         if(taken[j]) continue;
         if(g_ob_st[j] > bestSt) { bestSt = g_ob_st[j]; best = j; }
        }
      if(best < 0) break;
      taken[best] = true;
      selected[best] = true;
      picked++;
     }
  }

void ApplyTopK_FVG(bool &selected[])
  {
   ArrayResize(selected, g_fvg_n);
   for(int i = 0; i < g_fvg_n; i++) selected[i] = false;
   int picked = 0;
   bool taken[]; ArrayResize(taken, g_fvg_n);
   for(int i = 0; i < g_fvg_n; i++) taken[i] = false;
   while(picked < TopFVG_Global)
     {
      int best = -1;
      double bestSt = -1.0;
      for(int j = 0; j < g_fvg_n; j++)
        {
         if(taken[j]) continue;
         if(g_fvg_state[j] == "FILLED") continue; // ya filtrado en Store si DrawInvalidated=false
         if(g_fvg_st[j] > bestSt) { bestSt = g_fvg_st[j]; best = j; }
        }
      if(best < 0) break;
      taken[best] = true;
      selected[best] = true;
      picked++;
     }
  }

void ApplyTopK_LIQ(bool &selected[])
  {
   ArrayResize(selected, g_liq_n);
   for(int i = 0; i < g_liq_n; i++) selected[i] = false;
   int picked = 0;
   bool taken[]; ArrayResize(taken, g_liq_n);
   for(int i = 0; i < g_liq_n; i++) taken[i] = false;
   while(picked < TopLIQ_PrimaryTF)
     {
      int best = -1;
      double bestSc = -1.0;
      for(int j = 0; j < g_liq_n; j++)
        {
         if(taken[j]) continue;
         if(g_liq_tf[j] != PrimaryTF_Hint) continue;
         if(g_liq_sw[j]) continue; // no swept
         if(g_liq_sc[j] > bestSc) { bestSc = g_liq_sc[j]; best = j; }
        }
      if(best < 0) break;
      taken[best] = true;
      selected[best] = true;
      picked++;
     }
  }

// BOS/CHOCH: el más reciente (max timestamp) por TF, hasta K por TF.
void ApplyTopK_PerTF(int n, const string &tf_arr[], const datetime &t_arr[], int kPerTF, bool &selected[])
  {
   ArrayResize(selected, n);
   for(int i = 0; i < n; i++) selected[i] = false;
   // Recopilar TFs únicos
   string seen_tfs[]; int seen_n = 0;
   for(int i = 0; i < n; i++)
     {
      bool already = false;
      for(int k = 0; k < seen_n; k++) if(seen_tfs[k] == tf_arr[i]) { already=true; break; }
      if(already) continue;
      ArrayResize(seen_tfs, seen_n+1); seen_tfs[seen_n] = tf_arr[i]; seen_n++;
     }
   // Para cada TF, elegir los k más recientes
   for(int j = 0; j < seen_n; j++)
     {
      int picked = 0;
      bool taken[]; ArrayResize(taken, n);
      for(int i = 0; i < n; i++) taken[i] = false;
      while(picked < kPerTF)
        {
         int best = -1;
         datetime bestT = 0;
         for(int i = 0; i < n; i++)
           {
            if(taken[i]) continue;
            if(tf_arr[i] != seen_tfs[j]) continue;
            if(t_arr[i] > bestT) { bestT = t_arr[i]; best = i; }
           }
         if(best < 0) break;
         taken[best] = true;
         selected[best] = true;
         picked++;
        }
     }
  }

//+------------------------------------------------------------------+
//| Drawers (loops sobre seleccionados)                                |
//+------------------------------------------------------------------+
int DrawAllOBs(const bool &sel[], double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   int d = 0;
   for(int i = 0; i < g_ob_n; i++)
     {
      if(!sel[i]) continue;
      DrawOneOB(i, bucketsPx, bucketsN, bucketStep);
      d++;
     }
   return d;
  }
int DrawAllBOS(const bool &sel[], double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   int d = 0;
   for(int i = 0; i < g_bos_n; i++)
     {
      if(!sel[i]) continue;
      DrawOneBOS(i, bucketsPx, bucketsN, bucketStep);
      d++;
     }
   return d;
  }
int DrawAllCHOCH(const bool &sel[], double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   int d = 0;
   for(int i = 0; i < g_cho_n; i++)
     {
      if(!sel[i]) continue;
      DrawOneCHOCH(i, bucketsPx, bucketsN, bucketStep);
      d++;
     }
   return d;
  }
int DrawAllFVG(const bool &sel[], double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   int d = 0;
   for(int i = 0; i < g_fvg_n; i++)
     {
      if(!sel[i]) continue;
      DrawOneFVG(i, bucketsPx, bucketsN, bucketStep);
      d++;
     }
   return d;
  }
int DrawAllLIQ(const bool &sel[], double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   int d = 0;
   for(int i = 0; i < g_liq_n; i++)
     {
      if(!sel[i]) continue;
      DrawOneLIQ(i, bucketsPx, bucketsN, bucketStep);
      d++;
     }
   return d;
  }

//+------------------------------------------------------------------+
//| Drawers individuales                                              |
//+------------------------------------------------------------------+
void DrawOneOB(int i, double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   string id    = g_ob_id[i];
   string tf    = g_ob_tf[i];
   string dir   = g_ob_dir[i];
   datetime t0  = g_ob_t0[i];
   datetime tMit= g_ob_tm[i];
   double bot   = g_ob_bot[i];
   double top   = g_ob_top[i];
   string state = g_ob_state[i];
   double st    = g_ob_st[i];
   double qu    = g_ob_qu[i];
   double frsh  = g_ob_fr[i];

   if(top <= bot) { double tmp=bot; bot=top; top=tmp; }
   bool isMitigated= (state == "MITIGATED");
   bool isInvalid  = (state == "INVALIDATED");
   datetime t2 = (tMit > 0) ? tMit : ExtendForward(tf, ExtendBars);

   color col;
   if(isMitigated)
      col = (dir == "BULLISH") ? OBBullMitColor : OBBearMitColor;
   else
      col = (dir == "BULLISH") ? OBBullColor : OBBearColor;
   if(isInvalid) col = clrDimGray;

   string name = OBJ_PREFIX + "OB_" + id;
   RegisterId(name);
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, t0, top, t2, bot);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t0);
   ObjectSetDouble (0, name, OBJPROP_PRICE, 0, top);
   ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
   ObjectSetDouble (0, name, OBJPROP_PRICE, 1, bot);
   ObjectSetInteger(0, name, OBJPROP_COLOR, col);
   ObjectSetInteger(0, name, OBJPROP_STYLE, isMitigated ? STYLE_DOT : STYLE_SOLID);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, WidthForStrength(st));
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_FILL, !isInvalid);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);

   if(ShowLabels)
     {
      string label = ShortLabels
        ? StringFormat("OB %s %s %.2f", tf, (dir=="BULLISH"?"↑":"↓"), st)
        : StringFormat("OB %s %s S=%.2f Q=%.2f F=%.2f [%s]", tf, dir, st, qu, frsh, state);
      double labelPrice = AntiOverlapPrice((dir=="BULLISH")?top:bot, bucketsPx, bucketsN, bucketStep);
      // Label en el extremo derecho del rectángulo para no ensuciar el lado izquierdo
      datetime tLabel = (tMit > 0) ? tMit : ExtendForward(tf, ExtendBars / 3);
      PaintLabel(name + "_lbl", tLabel, labelPrice, col, label);
     }
  }

void DrawOneBOS(int i, double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   string id    = g_bos_id[i];
   string tf    = g_bos_tf[i];
   string dir   = g_bos_dir[i];
   datetime tSw = g_bos_tsw[i];
   datetime tBk = g_bos_tbk[i];
   double pSw   = g_bos_psw[i];
   double pBk   = g_bos_pbk[i];
   double st    = g_bos_st[i];
   double qu    = g_bos_qu[i];
   double rl    = g_bos_rl[i];

   string nameH = OBJ_PREFIX + "BOS_H_" + id;
   string nameB = OBJ_PREFIX + "BOS_B_" + id;
   RegisterId(nameH); RegisterId(nameB);
   datetime t2 = ExtendForward(tf, ExtendBars);

   if(ObjectFind(0, nameH) < 0)
      ObjectCreate(0, nameH, OBJ_TREND, 0, tSw, pSw, t2, pSw);
   ObjectSetInteger(0, nameH, OBJPROP_TIME, 0, tSw);
   ObjectSetDouble (0, nameH, OBJPROP_PRICE, 0, pSw);
   ObjectSetInteger(0, nameH, OBJPROP_TIME, 1, t2);
   ObjectSetDouble (0, nameH, OBJPROP_PRICE, 1, pSw);
   ObjectSetInteger(0, nameH, OBJPROP_COLOR, BOSColor);
   ObjectSetInteger(0, nameH, OBJPROP_STYLE, STYLE_DASH);
   ObjectSetInteger(0, nameH, OBJPROP_WIDTH, WidthForStrength(st));
   ObjectSetInteger(0, nameH, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, nameH, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, nameH, OBJPROP_SELECTABLE, false);

   if(ObjectFind(0, nameB) < 0)
      ObjectCreate(0, nameB, OBJ_ARROW, 0, tBk, pBk);
   ObjectSetInteger(0, nameB, OBJPROP_TIME, 0, tBk);
   ObjectSetDouble (0, nameB, OBJPROP_PRICE, 0, pBk);
   ObjectSetInteger(0, nameB, OBJPROP_ARROWCODE, (dir == "BULLISH") ? 233 : 234);
   ObjectSetInteger(0, nameB, OBJPROP_COLOR, BOSColor);
   ObjectSetInteger(0, nameB, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, nameB, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, nameB, OBJPROP_SELECTABLE, false);

   if(ShowLabels)
     {
      string label = ShortLabels
        ? StringFormat("BOS %s %s %.2f", tf, (dir=="BULLISH"?"↑":"↓"), st)
        : StringFormat("BOS %s %s S=%.2f Q=%.2f R=%.2f", tf, dir, st, qu, rl);
      double labelPrice = AntiOverlapPrice(pSw, bucketsPx, bucketsN, bucketStep);
      // Label justo a la derecha del break-arrow
      PaintLabel(nameH + "_lbl", tBk, labelPrice, BOSColor, label);
     }
  }

void DrawOneCHOCH(int i, double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   string id    = g_cho_id[i];
   string tf    = g_cho_tf[i];
   string dir   = g_cho_dir[i];
   string kind  = g_cho_kind[i];
   datetime tSw = g_cho_tsw[i];
   datetime tBk = g_cho_tbk[i];
   double pSw   = g_cho_psw[i];
   double pBk   = g_cho_pbk[i];
   double st    = g_cho_st[i];
   double qu    = g_cho_qu[i];

   string nameH = OBJ_PREFIX + "CHOCH_H_" + id;
   string nameB = OBJ_PREFIX + "CHOCH_B_" + id;
   RegisterId(nameH); RegisterId(nameB);
   datetime t2 = ExtendForward(tf, ExtendBars);

   if(ObjectFind(0, nameH) < 0)
      ObjectCreate(0, nameH, OBJ_TREND, 0, tSw, pSw, t2, pSw);
   ObjectSetInteger(0, nameH, OBJPROP_TIME, 0, tSw);
   ObjectSetDouble (0, nameH, OBJPROP_PRICE, 0, pSw);
   ObjectSetInteger(0, nameH, OBJPROP_TIME, 1, t2);
   ObjectSetDouble (0, nameH, OBJPROP_PRICE, 1, pSw);
   ObjectSetInteger(0, nameH, OBJPROP_COLOR, CHOCHColor);
   ObjectSetInteger(0, nameH, OBJPROP_STYLE, (kind == "MAJOR") ? STYLE_SOLID : STYLE_DASH);
   ObjectSetInteger(0, nameH, OBJPROP_WIDTH, WidthForStrength(st));
   ObjectSetInteger(0, nameH, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, nameH, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, nameH, OBJPROP_SELECTABLE, false);

   if(ObjectFind(0, nameB) < 0)
      ObjectCreate(0, nameB, OBJ_ARROW, 0, tBk, pBk);
   ObjectSetInteger(0, nameB, OBJPROP_TIME, 0, tBk);
   ObjectSetDouble (0, nameB, OBJPROP_PRICE, 0, pBk);
   ObjectSetInteger(0, nameB, OBJPROP_ARROWCODE, (dir == "BULLISH") ? 217 : 218);
   ObjectSetInteger(0, nameB, OBJPROP_COLOR, CHOCHColor);
   ObjectSetInteger(0, nameB, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, nameB, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, nameB, OBJPROP_SELECTABLE, false);

   if(ShowLabels)
     {
      string label = ShortLabels
        ? StringFormat("CHoCH %s %s %.2f", tf, (dir=="BULLISH"?"↑":"↓"), st)
        : StringFormat("CHoCH %s %s %s S=%.2f Q=%.2f", tf, dir, kind, st, qu);
      double labelPrice = AntiOverlapPrice(pSw, bucketsPx, bucketsN, bucketStep);
      PaintLabel(nameH + "_lbl", tBk, labelPrice, CHOCHColor, label);
     }
  }

void DrawOneFVG(int i, double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   string id    = g_fvg_id[i];
   string tf    = g_fvg_tf[i];
   string dir   = g_fvg_dir[i];
   datetime t0  = g_fvg_t0[i];
   datetime tInv= g_fvg_tinv[i];
   double bot   = g_fvg_bot[i];
   double top   = g_fvg_top[i];
   string state = g_fvg_state[i];
   double st    = g_fvg_st[i];
   double fp    = g_fvg_fp[i];

   if(top <= bot) { double tmp=bot; bot=top; top=tmp; }
   bool isFilled = (state == "FILLED");
   bool isInverted = (state == "INVERTED");
   bool isPartial = (state == "PARTIAL");
   datetime t2 = (tInv > 0) ? tInv : ExtendForward(tf, ExtendBars);
   color col;
   if(isInverted)
      col = (dir == "BULLISH") ? FVGBearMitColor : FVGBullMitColor;
   else if(isPartial)
      col = (dir == "BULLISH") ? FVGBullMitColor : FVGBearMitColor;
   else
      col = (dir == "BULLISH") ? FVGBullColor : FVGBearColor;

   string name = OBJ_PREFIX + "FVG_" + id;
   RegisterId(name);
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, t0, top, t2, bot);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t0);
   ObjectSetDouble (0, name, OBJPROP_PRICE, 0, top);
   ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
   ObjectSetDouble (0, name, OBJPROP_PRICE, 1, bot);
   ObjectSetInteger(0, name, OBJPROP_COLOR, col);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);

   if(ShowLabels)
     {
      string label = ShortLabels
        ? StringFormat("FVG %s %s %.0f%%", tf, (dir=="BULLISH"?"↑":"↓"), fp*100.0)
        : StringFormat("FVG %s %s S=%.2f fill=%.0f%% [%s]", tf, dir, st, fp * 100.0, state);
      double labelPrice = AntiOverlapPrice(top, bucketsPx, bucketsN, bucketStep);
      datetime tLabel = (tInv > 0) ? tInv : ExtendForward(tf, ExtendBars / 3);
      PaintLabel(name + "_lbl", tLabel, labelPrice, col, label);
     }
  }

void DrawOneLIQ(int i, double &bucketsPx[], int &bucketsN, const double bucketStep)
  {
   string id   = g_liq_id[i];
   string tf   = g_liq_tf[i];
   string kind = g_liq_kind[i];
   datetime t0 = g_liq_t0[i];
   datetime tSw= g_liq_tsw[i];
   double price= g_liq_p[i];
   double sc   = g_liq_sc[i];
   bool sw     = g_liq_sw[i];

   string name = OBJ_PREFIX + "LIQ_" + id;
   RegisterId(name);
   datetime t2 = (sw && tSw > 0) ? tSw : ExtendForward(tf, ExtendBars);
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_TREND, 0, t0, price, t2, price);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t0);
   ObjectSetDouble (0, name, OBJPROP_PRICE, 0, price);
   ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
   ObjectSetDouble (0, name, OBJPROP_PRICE, 1, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, sw ? clrGray : LIQColor);
   ObjectSetInteger(0, name, OBJPROP_STYLE, sw ? STYLE_DASHDOT : STYLE_DOT);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, WidthForStrength(sc));
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);

   if(ShowLabels)
     {
      string lbl = ShortLabels
        ? StringFormat("%s %s %.2f", kind, tf, sc)
        : StringFormat("%s %s s=%.2f%s", kind, tf, sc, sw ? " SWEPT" : "");
      double labelPrice = AntiOverlapPrice(price, bucketsPx, bucketsN, bucketStep);
      PaintLabel(name + "_lbl", t0, labelPrice, sw ? clrGray : LIQColor, lbl);
     }
  }

//+------------------------------------------------------------------+
//| Panel lateral resumen                                             |
//+------------------------------------------------------------------+
void RenderSummaryPanel()
  {
   // Conteos por TF
   string tfs[] = {"M15","M30","H1","H4","D1","W1"};
   int nTfs = ArraySize(tfs);
   int cntOB[6], cntBOS[6], cntCH[6], cntFVG[6], cntLIQ[6];
   for(int i = 0; i < nTfs; i++)
     { cntOB[i]=0; cntBOS[i]=0; cntCH[i]=0; cntFVG[i]=0; cntLIQ[i]=0; }
   for(int j = 0; j < g_ob_n;  j++) for(int t=0;t<nTfs;t++) if(g_ob_tf[j]==tfs[t])  cntOB[t]++;
   for(int j = 0; j < g_bos_n; j++) for(int t=0;t<nTfs;t++) if(g_bos_tf[j]==tfs[t]) cntBOS[t]++;
   for(int j = 0; j < g_cho_n; j++) for(int t=0;t<nTfs;t++) if(g_cho_tf[j]==tfs[t]) cntCH[t]++;
   for(int j = 0; j < g_fvg_n; j++) for(int t=0;t<nTfs;t++) if(g_fvg_tf[j]==tfs[t]) cntFVG[t]++;
   for(int j = 0; j < g_liq_n; j++) for(int t=0;t<nTfs;t++) if(g_liq_tf[j]==tfs[t]) cntLIQ[t]++;

   // Encontrar último BOS y CHOCH
   int lastBOS = -1; datetime lastBOST = 0;
   for(int j = 0; j < g_bos_n; j++) if(g_bos_tsw[j] > lastBOST) { lastBOST = g_bos_tsw[j]; lastBOS = j; }
   int lastCH = -1; datetime lastCHT = 0;
   for(int j = 0; j < g_cho_n; j++) if(g_cho_tsw[j] > lastCHT) { lastCHT = g_cho_tsw[j]; lastCH = j; }

   // Encontrar OB más fuerte (target)
   int strongestOB = -1; double bestObSt = -1.0;
   for(int j = 0; j < g_ob_n; j++) if(g_ob_st[j] > bestObSt) { bestObSt = g_ob_st[j]; strongestOB = j; }

   // Encontrar FVG más fuerte (target alt)
   int strongestFVG = -1; double bestFvgSt = -1.0;
   for(int j = 0; j < g_fvg_n; j++)
      if(g_fvg_state[j] != "FILLED" && g_fvg_st[j] > bestFvgSt)
        { bestFvgSt = g_fvg_st[j]; strongestFVG = j; }

   // Construir líneas del panel
   string title = StringFormat("◆  AURION SMC  |  %s", Symbol_);
   string bias  = g_has_conf
      ? StringFormat("bias    : %-8s   conf : %.2f", g_conf_dirBias, g_conf_score)
      : "bias    : n/a";
   string ctxMacro = g_has_conf ? StringFormat("macro   : %-8s", g_conf_macro)  : "";
   string ctxInst  = g_has_conf ? StringFormat("inst    : %-8s", g_conf_inst)   : "";
   string ctxOp    = g_has_conf ? StringFormat("op      : %-8s", g_conf_op)     : "";
   // AURION X 4.5.3 (11-jun-2026): mas espacio entre columnas para legibilidad.
   //  TF   = 5 char, cada columna numerica = 4 char + 3 espacios separadores.
   string hdr   = "TF      OB    BOS    CHO    FVG    LIQ";
   string rows[6];
   for(int i = 0; i < nTfs; i++)
      rows[i] = StringFormat("%-5s  %4d   %4d   %4d   %4d   %4d",
                             tfs[i], cntOB[i], cntBOS[i], cntCH[i], cntFVG[i], cntLIQ[i]);

   string lastEv = "";
   if(lastBOS >= 0)
      lastEv = StringFormat("BOS     : %-4s  %-7s  @ %.2f",
                            g_bos_tf[lastBOS], g_bos_dir[lastBOS], g_bos_psw[lastBOS]);
   string lastChStr = "";
   if(lastCH >= 0)
      lastChStr = StringFormat("CHoCH   : %-4s  %-7s  @ %.2f",
                               g_cho_tf[lastCH], g_cho_dir[lastCH], g_cho_psw[lastCH]);

   string obStr = "";
   if(strongestOB >= 0)
      obStr = StringFormat("OB tgt  : %-4s %s  %.2f - %.2f   S=%.2f",
                           g_ob_tf[strongestOB],
                           (g_ob_dir[strongestOB]=="BULLISH"?"↑":"↓"),
                           g_ob_bot[strongestOB], g_ob_top[strongestOB],
                           g_ob_st[strongestOB]);
   string fvgStr = "";
   if(strongestFVG >= 0)
      fvgStr = StringFormat("FVG tgt : %-4s %s  %.2f - %.2f   S=%.2f",
                            g_fvg_tf[strongestFVG],
                            (g_fvg_dir[strongestFVG]=="BULLISH"?"↑":"↓"),
                            g_fvg_bot[strongestFVG], g_fvg_top[strongestFVG],
                            g_fvg_st[strongestFVG]);

   string footer = StringFormat("drawn=%d    pruned=%d    refresh=%ds",
                                g_drawn, g_pruned, RefreshSeconds);

   // AURION X 4.5 - SMC DECISION block (6 lineas + separador si hay decision)
   string decSep    = g_has_decision ? "─── SMC DECISION ───" : "";
   string decScore  = "";
   string decMarket = "";
   string decQuality= "";
   string decSweep  = "";
   string decExec   = "";
   if(g_has_decision)
     {
      decScore   = StringFormat("SCORE   : %5.1f     GRADE : %s",
                                g_dec_score, g_dec_grade);
      decMarket  = StringFormat("MARKET  : %-10s  pct=%.2f",
                                g_dec_pd_state, g_dec_pd_pct);
      decQuality = StringFormat("OB Q    : %-3s       FVG Q : %s",
                                g_dec_ob_grade, g_dec_fvg_grade);
      // AURION X 4.6: ahora MTF es /5 (D1+H4+H1+M30+M15)
      decSweep   = StringFormat("SWEEP   : %-5s     MTF   : %d/5",
                                g_dec_sweep_kind, g_dec_mtf_aligned);
      if(g_dec_allow)
         decExec = StringFormat("EXEC    : ALLOW    [%s]", g_dec_status);
      else
         decExec = StringFormat("EXEC    : BLOCK (%s)",
                                StringSubstr(g_dec_block_reason, 0, 24));
     }

   // AURION X 4.4 (10-jun-2026): line-height generoso para legibilidad.
   color biasCol = (g_conf_dirBias == "BULLISH") ? PanelBullColor
                 : (g_conf_dirBias == "BEARISH") ? PanelBearColor
                 : PanelTextColor;
   // Color por grade SMC (consistencia visual con el setup_grade).
   color gradeCol = PanelTextColor;
   if(g_has_decision)
     {
      if(g_dec_grade == "A+")     gradeCol = PanelBullColor;
      else if(g_dec_grade == "A") gradeCol = clrLightGreen;
      else if(g_dec_grade == "B") gradeCol = clrGold;
      else if(g_dec_grade == "C") gradeCol = clrOrange;
      else if(g_dec_grade == "X") gradeCol = PanelBearColor;
     }
   color execCol = g_dec_allow ? PanelBullColor : PanelBearColor;
   // AURION X 4.5.3 (11-jun-2026): MINIMOS forzados para evitar layout apretado
   // aunque el chart tenga inputs guardados con valores bajos.
   int  fSize   = MathMax(PanelFontSize, 11);
   int  lSpace  = MathMax(PanelLineSpacing, 10);
   int  sGap    = MathMax(PanelSectionGap, 14);
   int  pPad    = MathMax(PanelPadding, 16);
   int lineH    = fSize + lSpace;
   int sectGap  = sGap;
   int yTop     = PanelYDistance;
   int y        = yTop + pPad;

   // 1) FONDO del panel ANTES de dibujar texto, para que las labels queden encima.
   //    Calculamos altura aproximada contando lineas activas.
   bool has_sl_block = (g_has_sl_cons || g_has_sl_bal || g_has_sl_agg);
   int sl_block_lines = 0;
   if(has_sl_block)
     {
      sl_block_lines = 2;                       // sep + hdr
      if(g_has_sl_cons) sl_block_lines++;
      if(g_has_sl_bal)  sl_block_lines++;
      if(g_has_sl_agg)  sl_block_lines++;
      sl_block_lines++;                          // REC line
     }
   int linesUsed = 1                                       // title
                 + 1                                       // bias
                 + (g_has_conf ? 3 : 0)                    // macro/inst/op
                 + 1                                       // hdr
                 + nTfs                                    // 6 rows
                 + (lastBOS >= 0 ? 1 : 0)
                 + (lastCH  >= 0 ? 1 : 0)
                 + (strongestOB  >= 0 ? 1 : 0)
                 + (strongestFVG >= 0 ? 1 : 0)
                 + (g_has_decision ? 7 : 0)                // SMC decision (6) + VOL/MODE (1)
                 + (g_tp_n > 0 ? (g_tp_n + 2) : 0)         // sep + hdr + N TPs
                 + sl_block_lines                          // sep + hdr + 1-3 SL + REC
                 + (g_sess_n > 0 ? (g_sess_n + 2) : 0)     // sep + hdr + N sessions
                 + 1;                                      // footer
   // Gaps reales: tras title, ctx, rows, targets, +1 por cada bloque extra (decision/tp/sl/sess).
   int extra_blocks = 0;
   if(g_has_decision) extra_blocks++;
   if(g_tp_n > 0)     extra_blocks++;
   if(has_sl_block)   extra_blocks++;
   if(g_sess_n > 0)   extra_blocks++;
   int gapsPx   = 4 + 3 * sectGap + extra_blocks * sectGap;
   int bgHeight = linesUsed * lineH + gapsPx + pPad * 2 + 6;
   int bgWidth  = MathMax(PanelWidth, 460);   // 4.6: ancho ampliado por nuevas columnas
   if(PanelShowBackground)
      PanelBackground("bg", yTop, bgWidth, bgHeight);

   // 2) Texto sobre el fondo.
   PanelLine("title", title, PanelTitleColor, y);                       y += lineH + 4;
   PanelLine("bias",  bias,  biasCol,         y);                       y += lineH;
   if(StringLen(ctxMacro) > 0)
     { PanelLine("ctx_macro", ctxMacro, PanelTextColor, y);             y += lineH; }
   if(StringLen(ctxInst) > 0)
     { PanelLine("ctx_inst",  ctxInst,  PanelTextColor, y);             y += lineH; }
   if(StringLen(ctxOp) > 0)
     { PanelLine("ctx_op",    ctxOp,    PanelTextColor, y);             y += lineH; }
   y += sectGap;
   PanelLine("hdr",   hdr,   PanelTitleColor, y);                       y += lineH;
   for(int i = 0; i < nTfs; i++)
     { PanelLine("row_" + tfs[i], rows[i], PanelTextColor, y);          y += lineH; }
   y += sectGap;
   if(StringLen(lastEv) > 0)
     { PanelLine("lastBOS", lastEv, PanelTextColor, y);                 y += lineH; }
   if(StringLen(lastChStr) > 0)
     { PanelLine("lastCH",  lastChStr, PanelTextColor, y);              y += lineH; }
   if(StringLen(obStr) > 0)
     { PanelLine("obT",     obStr, PanelTextColor, y);                  y += lineH; }
   if(StringLen(fvgStr) > 0)
     { PanelLine("fvgT",    fvgStr, PanelTextColor, y);                 y += lineH; }
   // AURION X 4.5 - Bloque SMC DECISION (resaltado por grade)
   if(g_has_decision)
     {
      y += sectGap;
      PanelLine("dec_sep",  decSep,    PanelTitleColor, y);             y += lineH;
      PanelLine("dec_score",decScore,  gradeCol,        y);             y += lineH;
      PanelLine("dec_mkt",  decMarket, PanelTextColor,  y);             y += lineH;
      PanelLine("dec_qual", decQuality,PanelTextColor,  y);             y += lineH;
      PanelLine("dec_sw",   decSweep,  PanelTextColor,  y);             y += lineH;
      PanelLine("dec_exec", decExec,   execCol,         y);             y += lineH;
      // AURION X 4.6 - Bloque vol class + qmode (linea compacta)
      string volQmode = StringFormat("VOL     : %-7s   MODE: %s",
                                     g_dec_vol_class, g_dec_qmode);
      PanelLine("dec_vol", volQmode, PanelTextColor, y);                y += lineH;
     }

   // AURION X 4.6 - Bloque SMART TARGETS (TP1..TP4)
   if(g_tp_n > 0)
     {
      y += sectGap;
      PanelLine("tp_sep", "─── SMART TARGETS ───", PanelTitleColor, y); y += lineH;
      string tpHdr = "TP   PRICE       PROB   SRC        DIST";
      PanelLine("tp_hdr", tpHdr, PanelTitleColor, y);                   y += lineH;
      for(int k = 0; k < g_tp_n; k++)
        {
         color tpCol = (g_tp_prob[k] >= 0.65) ? PanelBullColor
                     : (g_tp_prob[k] >= 0.50) ? PanelTextColor
                     : clrLightGray;
         string srcShort = g_tp_src[k];
         if(StringLen(srcShort) > 10) srcShort = StringSubstr(srcShort, 0, 10);
         string row = StringFormat("TP%d  %-10.2f  %4.0f%%  %-10s %.0fp",
                                   g_tp_rank[k], g_tp_price[k],
                                   g_tp_prob[k] * 100.0, srcShort,
                                   g_tp_dist_pips[k]);
         PanelLine("tp_" + IntegerToString(k), row, tpCol, y);          y += lineH;
        }
     }

   // AURION X 4.6 - Bloque SMART STOPS (3 modos + RR vs TP1)
   if(g_has_sl_cons || g_has_sl_bal || g_has_sl_agg)
     {
      y += sectGap;
      PanelLine("sl_sep", "─── SMART STOPS ───", PanelTitleColor, y);   y += lineH;
      string slHdr = "MODE     PRICE       PIPS    HUNT%  RR1";
      PanelLine("sl_hdr", slHdr, PanelTitleColor, y);                   y += lineH;
      if(g_has_sl_cons)
        {
         color c = (g_sl_recommended == "conservative") ? PanelBullColor : PanelTextColor;
         string row = StringFormat("CONS     %-10.2f  %5.0fp  %3.0f%%   %.2f",
                                   g_sl_cons_price, g_sl_cons_pips,
                                   g_sl_cons_hunt * 100.0, g_rr_cons[0]);
         PanelLine("sl_c", row, c, y);                                  y += lineH;
        }
      if(g_has_sl_bal)
        {
         color c = (g_sl_recommended == "balanced") ? PanelBullColor : PanelTextColor;
         string row = StringFormat("BAL      %-10.2f  %5.0fp  %3.0f%%   %.2f",
                                   g_sl_bal_price, g_sl_bal_pips,
                                   g_sl_bal_hunt * 100.0, g_rr_bal[0]);
         PanelLine("sl_b", row, c, y);                                  y += lineH;
        }
      if(g_has_sl_agg)
        {
         color c = (g_sl_recommended == "aggressive") ? PanelBullColor : PanelTextColor;
         string row = StringFormat("AGR      %-10.2f  %5.0fp  %3.0f%%   %.2f",
                                   g_sl_agg_price, g_sl_agg_pips,
                                   g_sl_agg_hunt * 100.0, g_rr_agg[0]);
         PanelLine("sl_a", row, c, y);                                  y += lineH;
        }
      string slRec = StringFormat("REC      %s", g_sl_recommended);
      PanelLine("sl_rec", slRec, clrGold, y);                           y += lineH;
     }

   // AURION X 4.6 - Bloque SESSIONS (Asia/London/NY ranges)
   if(g_sess_n > 0)
     {
      y += sectGap;
      PanelLine("sess_sep", "─── SESSIONS ───", PanelTitleColor, y);    y += lineH;
      string sessHdr = "SESS       HIGH       LOW       SWP";
      PanelLine("sess_hdr", sessHdr, PanelTitleColor, y);               y += lineH;
      for(int s = 0; s < g_sess_n; s++)
        {
         color sc = g_sess_active[s] ? PanelTitleColor : PanelTextColor;
         string sw = "";
         if(g_sess_sw_hi[s]) sw += "H";
         if(g_sess_sw_lo[s]) sw += "L";
         if(StringLen(sw) == 0) sw = "-";
         string row = StringFormat("%-9s  %-9.2f  %-9.2f  %s",
                                   g_sess_name[s], g_sess_high[s],
                                   g_sess_low[s], sw);
         PanelLine("sess_" + IntegerToString(s), row, sc, y);           y += lineH;
        }
     }

   y += sectGap;
   PanelLine("foot",  footer, clrSlateGray,    y);
  }

//+------------------------------------------------------------------+
//| PanelBackground: rectangulo opaco detras del texto                |
//+------------------------------------------------------------------+
void PanelBackground(const string suffix, int yTop, int width, int height)
  {
   string name = PANEL_PREFIX + suffix;
   RegisterId(name);
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, PanelCorner);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, PanelXDistance);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yTop);
   // AURION X 4.5.3: ancho minimo del fondo forzado.
   ObjectSetInteger(0, name, OBJPROP_XSIZE, MathMax(width, 410));
   ObjectSetInteger(0, name, OBJPROP_YSIZE, height);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, PanelBgColor);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_BORDER_COLOR, PanelBorderColor);
   ObjectSetInteger(0, name, OBJPROP_COLOR, PanelBorderColor);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);       // delante de velas pero detras de texto
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
  }

void PanelLine(const string suffix, const string text, color col, int yDist)
  {
   string name = PANEL_PREFIX + suffix;
   RegisterId(name);
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, PanelCorner);
   // AURION X 4.5.3: minimos seguros forzados aunque el chart tenga
   // inputs guardados con valores pequenos. Tambien forzamos fuente
   // monoespaciada ("Courier New" siempre disponible en Windows).
   int xPad   = MathMax(PanelPadding, 16);
   int fSize  = MathMax(PanelFontSize, 11);
   string fnt = (StringLen(PanelFont) > 0) ? PanelFont : "Courier New";
   // Si el usuario dejo una fuente no monoespaciada conocida, sustituimos.
   if(fnt == "Arial" || fnt == "Tahoma" || fnt == "Verdana" || fnt == "Segoe UI")
      fnt = "Courier New";
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, PanelXDistance + xPad);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDist);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fSize);
   ObjectSetInteger(0, name, OBJPROP_COLOR, col);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetString(0, name, OBJPROP_FONT, fnt);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   // ANCHOR para que XDistance crezca a la izquierda en CORNER_RIGHT_*
   if(PanelCorner == CORNER_RIGHT_UPPER || PanelCorner == CORNER_RIGHT_LOWER)
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
   else
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
  }

//+------------------------------------------------------------------+
//| Helpers                                                           |
//+------------------------------------------------------------------+
bool HttpGet(const string url, string &out_body)
  {
   char post[];
   char result[];
   string result_headers;
   string headers = "Accept: text/plain\r\n";
   ResetLastError();
   int res = WebRequest("GET", url, headers, RequestTimeoutMs, post, result, result_headers);
   if(res == -1)
     {
      PrintFormat("[AURION SMC v3] WebRequest err=%d url=%s", GetLastError(), url);
      return false;
     }
   if(res < 200 || res >= 300)
     {
      PrintFormat("[AURION SMC v3] WebRequest HTTP=%d url=%s", res, url);
      return false;
     }
   out_body = CharArrayToString(result, 0, ArraySize(result), CP_UTF8);
   return true;
  }

bool AgeOk(const string tf, const datetime t0)
  {
   if(t0 <= 0) return true;
   ENUM_TIMEFRAMES e = StrToTimeframe(tf);
   int secs = PeriodSeconds(e);
   if(secs <= 0) secs = PeriodSeconds(PERIOD_M15);
   int ageBars = (int)((TimeCurrent() - t0) / secs);
   return (ageBars <= MaxAgeBars);
  }

int WidthForStrength(const double s)
  {
   if(s >= 0.85) return 3;
   if(s >= 0.70) return 2;
   return 1;
  }

double AntiOverlapPrice(const double price, double &bucketsPx[], int &bucketsN, const double step)
  {
   double p = price;
   for(int k = 0; k < bucketsN; k++)
     {
      if(MathAbs(p - bucketsPx[k]) < step)
        {
         p += step * 0.6;
         k = -1;
        }
     }
   ArrayResize(bucketsPx, bucketsN + 1);
   bucketsPx[bucketsN] = p;
   bucketsN++;
   return p;
  }

double ChartATRApprox()
  {
   int handle = iATR(_Symbol, PERIOD_CURRENT, 14);
   if(handle == INVALID_HANDLE) return 0.0;
   double buf[];
   if(CopyBuffer(handle, 0, 0, 1, buf) <= 0) return 0.0;
   return buf[0];
  }

datetime IsoToTime(const string iso)
  {
   if(StringLen(iso) < 10) return 0;
   string s = iso;
   StringReplace(s, "T", " ");
   int pos = StringFind(s, "+");
   if(pos < 0) pos = StringFind(s, "Z");
   if(pos > 10) s = StringSubstr(s, 0, pos);
   int dot = StringFind(s, ".");
   if(dot > 10) s = StringSubstr(s, 0, dot);
   return StringToTime(s);
  }

datetime ExtendForward(const string tf, const int barsAhead)
  {
   ENUM_TIMEFRAMES e = StrToTimeframe(tf);
   int secs = PeriodSeconds(e);
   if(secs <= 0) secs = PeriodSeconds(PERIOD_M15);
   return TimeCurrent() + (datetime)(barsAhead * secs);
  }

ENUM_TIMEFRAMES StrToTimeframe(const string s)
  {
   if(s == "M1")  return PERIOD_M1;
   if(s == "M5")  return PERIOD_M5;
   if(s == "M15") return PERIOD_M15;
   if(s == "M30") return PERIOD_M30;
   if(s == "H1")  return PERIOD_H1;
   if(s == "H4")  return PERIOD_H4;
   if(s == "D1")  return PERIOD_D1;
   if(s == "W1")  return PERIOD_W1;
   if(s == "MN1") return PERIOD_MN1;
   return PERIOD_M15;
  }

double StrToDoubleOrZero(const string s)
  {
   if(StringLen(s) == 0) return 0.0;
   return StringToDouble(s);
  }

string TrimAll(const string s)
  {
   string r = s;
   StringReplace(r, "\r", "");
   StringTrimLeft(r);
   StringTrimRight(r);
   return r;
  }

void PaintLabel(const string name, datetime t, double price, color col, const string text)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_TEXT, 0, t, price);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t);
   ObjectSetDouble (0, name, OBJPROP_PRICE, 0, price);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, col);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, LabelFontSize);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER);
   // El label paralelo NO se registra como ID propio; se purga junto al objeto
   // padre cuando éste deja de venir del backend.
  }

//+------------------------------------------------------------------+
//| Active-IDs tracking + Prune                                       |
//+------------------------------------------------------------------+
void RegisterId(const string objName)
  {
   ArrayResize(g_active_ids, g_active_count + 1);
   g_active_ids[g_active_count] = objName;
   g_active_count++;
   // Asociar label paralelo si existe
   ArrayResize(g_active_ids, g_active_count + 1);
   g_active_ids[g_active_count] = objName + "_lbl";
   g_active_count++;
  }

bool IsRegisteredId(const string objName)
  {
   for(int i = 0; i < g_active_count; i++)
      if(g_active_ids[i] == objName) return true;
   return false;
  }

int PruneStaleObjects()
  {
   int pruned = 0;
   int total = ObjectsTotal(0, -1, -1);
   for(int i = total - 1; i >= 0; i--)
     {
      string name = ObjectName(0, i, -1, -1);
      if(StringFind(name, OBJ_PREFIX) != 0 && StringFind(name, PANEL_PREFIX) != 0) continue;
      if(IsRegisteredId(name)) continue;
      ObjectDelete(0, name);
      pruned++;
     }
   return pruned;
  }

void ClearAll()
  {
   int total = ObjectsTotal(0, -1, -1);
   for(int i = total - 1; i >= 0; i--)
     {
      string name = ObjectName(0, i, -1, -1);
      if(StringFind(name, OBJ_PREFIX) == 0 || StringFind(name, PANEL_PREFIX) == 0)
         ObjectDelete(0, name);
     }
  }
//+------------------------------------------------------------------+

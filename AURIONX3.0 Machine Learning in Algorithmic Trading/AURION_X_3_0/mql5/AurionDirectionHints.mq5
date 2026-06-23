//+------------------------------------------------------------------+
//| AurionDirectionHints.mq5                                       |
//| Flechas alineadas con la lógica M1 del ScalperEngine (Python).   |
//| Temprana: mom_score >= InpMomScoreEarly. Confirmada: >= 0.35.   |
//| Cooldown InpCooldownBars reduce señales pegadas; si InpConfSuppressesEarly |
//| solo se dibuja confirmada cuando ambos umbrales se cumplen (menos ruido).   |
//| Python excluye la vela en formación; aquí se usa la última       |
//| cerrada como barra "i" (índice n-2 si n-1 es la actual).         |
//+------------------------------------------------------------------+
#property copyright "AURION X 3.0"
#property version   "1.02"
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   4

#property indicator_label1  "Temprana COMPRA"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrDodgerBlue
#property indicator_width1  1

#property indicator_label2  "Temprana VENTA"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrTomato
#property indicator_width2  1

#property indicator_label3  "Confirmada COMPRA"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrDeepSkyBlue
#property indicator_width3  3

#property indicator_label4  "Confirmada VENTA"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrOrangeRed
#property indicator_width4  3

input ENUM_TIMEFRAMES InpSignalTF      = PERIOD_M1;
input int             InpEMAFast       = 8;
input int             InpEMASlow       = 21;
input int             InpMinBars        = 42;
input double          InpMinSepRatio   = 0.000065;
input double          InpMomPct          = 0.00018;
input double          InpMomScoreEarly = 0.18;
input double          InpMomScoreConf  = 0.35;
// Mínimo de velas del TF de señal entre dos flechas del mismo tipo (reduce diamantes apilados en chop).
input int             InpCooldownBars   = 4;
// Si true: en la misma barra solo dibuja confirmada (no temprana + confirmada a la vez).
input bool            InpConfSuppressesEarly = true;
input int             InpArrowEarlyUp  = 241;
input int             InpArrowEarlyDn    = 242;
input int             InpArrowConfUp     = 159;
input int             InpArrowConfDn     = 159;

double BufEarlyBuy[];
double BufEarlySell[];
double BufConfBuy[];
double BufConfSell[];

//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, BufEarlyBuy, INDICATOR_DATA);
   SetIndexBuffer(1, BufEarlySell, INDICATOR_DATA);
   SetIndexBuffer(2, BufConfBuy, INDICATOR_DATA);
   SetIndexBuffer(3, BufConfSell, INDICATOR_DATA);
   ArraySetAsSeries(BufEarlyBuy, true);
   ArraySetAsSeries(BufEarlySell, true);
   ArraySetAsSeries(BufConfBuy, true);
   ArraySetAsSeries(BufConfSell, true);
   PlotIndexSetInteger(0, PLOT_ARROW, InpArrowEarlyUp);
   PlotIndexSetInteger(1, PLOT_ARROW, InpArrowEarlyDn);
   PlotIndexSetInteger(2, PLOT_ARROW, InpArrowConfUp);
   PlotIndexSetInteger(3, PLOT_ARROW, InpArrowConfDn);
   IndicatorSetString(INDICATOR_SHORTNAME, "Aurion hints (" + EnumToString(InpSignalTF) + ")");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
// Serie temporal: índice 0 = más antiguo, total-1 = vela actual (en formación).
void BuildEma(const double &c[], int total, int period, double &ema[])
  {
   ArrayResize(ema, total);
   if(total < 1 || period < 1) return;
   double k = 2.0 / (period + 1.0);
   ema[0] = c[0];
   for(int i = 1; i < total; i++)
      ema[i] = c[i] * k + ema[i - 1] * (1.0 - k);
  }

//+------------------------------------------------------------------+
double MomentumScore(const double &h[], const double &l[], const double &c[], int total, int i)
  {
   if(i < 6) return 0.0;
   double last_range = MathMax(h[i] - l[i], 1e-12);
   double avg_range = 0.0;
   for(int j = i - 5; j < i; j++)
      avg_range += MathMax(h[j] - l[j], 1e-12);
   avg_range /= 5.0;
   double range_exp = MathMin(1.0, last_range / MathMax(avg_range, 1e-12));
   double vel = MathAbs(c[i] - c[i - 2]) / MathMax(MathAbs(c[i - 2]), 1e-12);
   double vel_score = MathMin(1.0, vel * 250.0);
   double ft = 0.0;
   if((c[i] > c[i - 1] && c[i - 1] > c[i - 2]) || (c[i] < c[i - 1] && c[i - 1] < c[i - 2]))
      ft = 1.0;
   double raw = 0.45 * range_exp + 0.35 * vel_score + 0.20 * ft;
   return MathMax(0.0, MathMin(1.0, raw));
  }

//+------------------------------------------------------------------+
void ClearBuffers(const int rates_total)
  {
   for(int b = 0; b < rates_total; b++)
     {
      BufEarlyBuy[b] = EMPTY_VALUE;
      BufEarlySell[b] = EMPTY_VALUE;
      BufConfBuy[b] = EMPTY_VALUE;
      BufConfSell[b] = EMPTY_VALUE;
     }
  }

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   (void)prev_calculated;
   if(rates_total < 10) return(0);

   ClearBuffers(rates_total);

   int need = MathMax(InpMinBars + 30, 200);
   double mc[], mh[], ml[];
   datetime mt[];
   int n = CopyClose(_Symbol, InpSignalTF, 0, need, mc);
   if(n < InpMinBars + 3) return(0);
   CopyHigh(_Symbol, InpSignalTF, 0, need, mh);
   CopyLow(_Symbol, InpSignalTF, 0, need, ml);
   CopyTime(_Symbol, InpSignalTF, 0, need, mt);

   double ema_f[], ema_s[];
   BuildEma(mc, n, InpEMAFast, ema_f);
   BuildEma(mc, n, InpEMASlow, ema_s);

   // Igual que Python: última barra procesada = última cerrada (no la en formación).
   int i_last_closed = n - 2;
   int i_start = MathMax(InpMinBars, 6);

   int cd = MathMax(0, InpCooldownBars);
   int last_early_buy_i = -1000000;
   int last_early_sell_i = -1000000;
   int last_conf_buy_i = -1000000;
   int last_conf_sell_i = -1000000;

   for(int i = i_start; i <= i_last_closed; i++)
     {
      int i1 = i - 1;
      int i2 = i - 2;
      if(i2 < 0) continue;

      double ef_i = ema_f[i];
      double es_i = ema_s[i];
      double ef_i1 = ema_f[i1];
      double es_i1 = ema_s[i1];
      double c = mc[i];
      double c1 = mc[i1];
      double l1 = ml[i1];

      double sep_ratio = MathAbs(ef_i - es_i) / (MathAbs(c) + 1e-12);
      if(sep_ratio < InpMinSepRatio) continue;

      double mom3 = 0.0;
      if(i >= 3 && MathAbs(mc[i - 3]) > 1e-12)
         mom3 = (c - mc[i - 3]) / mc[i - 3];

      bool bull_regime = (ef_i > es_i && ef_i1 >= es_i1 * 0.9999);
      bool bear_regime = (ef_i < es_i && ef_i1 <= es_i1 * 1.0001);

      bool pullback_buy = (l1 <= ef_i1 * 1.0002 || c1 < ef_i1);
      bool continuation_buy = (c > c1 && c >= ef_i && c > es_i);
      bool impulse_buy = (InpMomPct > 0 ? (mom3 > InpMomPct && c > c1) : (mom3 > 0 || c > c1));

      bool pullback_sell = (mh[i1] >= ef_i1 * 0.9998 || c1 > ef_i1);
      bool continuation_sell = (c < c1 && c <= ef_i && c < es_i);
      bool impulse_sell = (InpMomPct > 0 ? (mom3 < -InpMomPct && c < c1) : (mom3 < 0 || c < c1));

      bool buy_setup = bull_regime && pullback_buy && continuation_buy && impulse_buy;
      bool sell_setup = bear_regime && pullback_sell && continuation_sell && impulse_sell;

      string dir = "";
      if(buy_setup && !sell_setup) dir = "BUY";
      else if(sell_setup && !buy_setup) dir = "SELL";
      else if(buy_setup && sell_setup) dir = (c - es_i) >= (es_i - c) ? "BUY" : "SELL";
      else continue;

      double mom_score = MomentumScore(mh, ml, mc, n, i);
      datetime bar_time = mt[i];
      int sh = iBarShift(_Symbol, PERIOD_CURRENT, bar_time, true);
      if(sh < 0 || sh >= rates_total) continue;

      // time[]/high[]/low[] en OnCalculate: índice 0 = más antiguo; sh cuenta desde la vela actual.
      int idx = rates_total - 1 - sh;
      if(idx < 0 || idx >= rates_total) continue;

      double pad = 12 * _Point;
      double price_arrow = (dir == "BUY") ? low[idx] - pad : high[idx] + pad;

      if(mom_score >= InpMomScoreConf)
        {
         if(dir == "BUY")
           {
            if(i - last_conf_buy_i >= cd)
              {
               BufConfBuy[sh] = price_arrow;
               last_conf_buy_i = i;
              }
            if(!InpConfSuppressesEarly && (i - last_early_buy_i >= cd))
              {
               BufEarlyBuy[sh] = price_arrow;
               last_early_buy_i = i;
              }
           }
         else
           {
            if(i - last_conf_sell_i >= cd)
              {
               BufConfSell[sh] = price_arrow;
               last_conf_sell_i = i;
              }
            if(!InpConfSuppressesEarly && (i - last_early_sell_i >= cd))
              {
               BufEarlySell[sh] = price_arrow;
               last_early_sell_i = i;
              }
           }
        }
      else if(mom_score >= InpMomScoreEarly)
        {
         if(dir == "BUY")
           {
            if(i - last_early_buy_i >= cd)
              {
               BufEarlyBuy[sh] = price_arrow;
               last_early_buy_i = i;
              }
           }
         else
           {
            if(i - last_early_sell_i >= cd)
              {
               BufEarlySell[sh] = price_arrow;
               last_early_sell_i = i;
              }
           }
        }
     }

   return(rates_total);
  }
//+------------------------------------------------------------------+

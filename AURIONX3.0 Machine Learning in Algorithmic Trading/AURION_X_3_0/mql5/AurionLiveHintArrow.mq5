//+------------------------------------------------------------------+
//| AurionLiveHintArrow.mq5                                          |
//| Dibuja flecha BUY/SELL leyendo aurion_live_hints.csv             |
//+------------------------------------------------------------------+
#property copyright "AURION X 3.0"
#property version   "1.00"
#property strict
#property indicator_chart_window

input string InpHintsFileName = "aurion_live_hints.csv"; // ruta relativa a MQL5/Files
input int    InpArrowUpEarlyCode = 241;
input int    InpArrowDnEarlyCode = 242;
input int    InpArrowUpConfCode  = 233;
input int    InpArrowDnConfCode  = 234;
input color  InpBuyColor      = clrLime;
input color  InpSellColor     = clrTomato;
input int    InpArrowEarlySize = 1;
input int    InpArrowConfSize  = 3;
input int    InpMaxAgeMinutes = 20;

datetime g_last_hint_ts = 0;
string   g_last_hint_dir = "";
double   g_last_hint_conf = -1.0;
int      g_last_hint_score = -1;
double   g_last_hint_rr = -1.0;
string   g_last_hint_stage = "";

void ClearHintObjects()
{
   ObjectDelete(0, "AURION_HINT_EARLY_BUY");
   ObjectDelete(0, "AURION_HINT_EARLY_SELL");
   ObjectDelete(0, "AURION_HINT_CONF_BUY");
   ObjectDelete(0, "AURION_HINT_CONF_SELL");
   ObjectDelete(0, "AURION_HINT_LABEL");
}

bool ReadLatestHint(string symbol, string &dir, datetime &ts, double &conf, int &score, double &rr, string &stage)
{
   int h = FileOpen(InpHintsFileName, FILE_READ | FILE_CSV | FILE_ANSI);
   if(h == INVALID_HANDLE)
      return false;
   bool found = false;
   while(!FileIsEnding(h))
   {
      string s  = FileReadString(h);
      string d  = FileReadString(h);
      string cf = FileReadString(h);
      string sc = FileReadString(h);
      string rrs = FileReadString(h);
      string stg = FileReadString(h);
      string st  = FileReadString(h);
      string tt  = FileReadString(h);
      if(StringCompare(StringUpper(s), StringUpper(symbol)) == 0)
      {
         dir = StringUpper(d);
         ts = StringToTime(tt);
         conf = StringToDouble(cf);
         score = (int)StringToInteger(sc);
         rr = StringToDouble(rrs);
         stage = StringUpper(stg);
         found = true;
         break;
      }
   }
   FileClose(h);
   return found;
}

void DrawHint(string dir, datetime ts, double conf, int score, double rr, string stage)
{
   ClearHintObjects();
   int shift = iBarShift(_Symbol, PERIOD_CURRENT, ts, true);
   if(shift < 0) shift = 0;
   double hi = iHigh(_Symbol, PERIOD_CURRENT, shift);
   double lo = iLow(_Symbol, PERIOD_CURRENT, shift);
   double pad = 15 * _Point;
   datetime t = iTime(_Symbol, PERIOD_CURRENT, shift);
   bool confirmed = (stage == "CONFIRMED");
   if(dir == "BUY")
   {
      ObjectCreate(0, "AURION_HINT_EARLY_BUY", OBJ_ARROW, 0, t, lo - (pad * 0.8));
      ObjectSetInteger(0, "AURION_HINT_EARLY_BUY", OBJPROP_ARROWCODE, InpArrowUpEarlyCode);
      ObjectSetInteger(0, "AURION_HINT_EARLY_BUY", OBJPROP_COLOR, InpBuyColor);
      ObjectSetInteger(0, "AURION_HINT_EARLY_BUY", OBJPROP_WIDTH, InpArrowEarlySize);
      if(confirmed)
      {
         ObjectCreate(0, "AURION_HINT_CONF_BUY", OBJ_ARROW, 0, t, lo - (pad * 1.35));
         ObjectSetInteger(0, "AURION_HINT_CONF_BUY", OBJPROP_ARROWCODE, InpArrowUpConfCode);
         ObjectSetInteger(0, "AURION_HINT_CONF_BUY", OBJPROP_COLOR, InpBuyColor);
         ObjectSetInteger(0, "AURION_HINT_CONF_BUY", OBJPROP_WIDTH, InpArrowConfSize);
      }
   }
   else if(dir == "SELL")
   {
      ObjectCreate(0, "AURION_HINT_EARLY_SELL", OBJ_ARROW, 0, t, hi + (pad * 0.8));
      ObjectSetInteger(0, "AURION_HINT_EARLY_SELL", OBJPROP_ARROWCODE, InpArrowDnEarlyCode);
      ObjectSetInteger(0, "AURION_HINT_EARLY_SELL", OBJPROP_COLOR, InpSellColor);
      ObjectSetInteger(0, "AURION_HINT_EARLY_SELL", OBJPROP_WIDTH, InpArrowEarlySize);
      if(confirmed)
      {
         ObjectCreate(0, "AURION_HINT_CONF_SELL", OBJ_ARROW, 0, t, hi + (pad * 1.35));
         ObjectSetInteger(0, "AURION_HINT_CONF_SELL", OBJPROP_ARROWCODE, InpArrowDnConfCode);
         ObjectSetInteger(0, "AURION_HINT_CONF_SELL", OBJPROP_COLOR, InpSellColor);
         ObjectSetInteger(0, "AURION_HINT_CONF_SELL", OBJPROP_WIDTH, InpArrowConfSize);
      }
   }
   string txt = StringFormat("AURION %s | conf=%.2f | score=%d | RR=%.2f | %s", dir, conf, score, rr, stage);
   double lbl_price = (dir == "BUY") ? (lo - (pad * 2.1)) : (hi + (pad * 2.1));
   ObjectCreate(0, "AURION_HINT_LABEL", OBJ_TEXT, 0, t, lbl_price);
   ObjectSetString(0, "AURION_HINT_LABEL", OBJPROP_TEXT, txt);
   ObjectSetInteger(0, "AURION_HINT_LABEL", OBJPROP_COLOR, (dir == "BUY") ? InpBuyColor : InpSellColor);
   ObjectSetInteger(0, "AURION_HINT_LABEL", OBJPROP_FONTSIZE, 9);
}

int OnInit()
{
   IndicatorSetString(INDICATOR_SHORTNAME, "Aurion Live Hint Arrow");
   return(INIT_SUCCEEDED);
}

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
   (void)rates_total; (void)prev_calculated; (void)time; (void)open; (void)high; (void)low; (void)close; (void)tick_volume; (void)volume; (void)spread;
   string dir = "";
   datetime ts = 0;
   double conf = 0.0;
   int score = 0;
   double rr = 0.0;
   string stage = "EARLY";
   if(!ReadLatestHint(_Symbol, dir, ts, conf, score, rr, stage))
      return rates_total;
   if(ts <= 0)
      return rates_total;
   datetime now_utc = TimeGMT();
   if((now_utc - ts) > (InpMaxAgeMinutes * 60))
      return rates_total;
   if(
      ts != g_last_hint_ts ||
      dir != g_last_hint_dir ||
      MathAbs(conf - g_last_hint_conf) > 0.000001 ||
      score != g_last_hint_score ||
      MathAbs(rr - g_last_hint_rr) > 0.000001 ||
      stage != g_last_hint_stage
   )
   {
      DrawHint(dir, ts, conf, score, rr, stage);
      g_last_hint_ts = ts;
      g_last_hint_dir = dir;
      g_last_hint_conf = conf;
      g_last_hint_score = score;
      g_last_hint_rr = rr;
      g_last_hint_stage = stage;
   }
   return rates_total;
}

//+------------------------------------------------------------------+
//|                                            MaxTradingBridge.mq5  |
//|                                    Max Trading Agent v2 (MT5 EA) |
//| Zero-DLL Local HTTP/JSON Bridge + Independent Hard Risk Guard    |
//+------------------------------------------------------------------+
#property copyright "Max Trading Agent v2"
#property version   "2.10"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Bridge Connection Inputs
input string   BridgeUrl              = "http://127.0.0.1:8765/api/bridge/sync";
input string   BridgeAuthToken        = "";      // Filled in by setup_and_test.bat (never commit a real token)
input int      SyncIntervalMs         = 350;      // Fast poll interval (ms)
input int      WebTimeoutMs           = 1200;     // HTTP timeout (ms)
input string   LogicalSymbols         = "XAUUSD,EURUSD,GBPUSD,USDJPY"; // Watchlist (plain names). Missing ones are skipped
input bool     IncludeChartSymbol     = true;     // Also trade the symbol of the chart the EA is on
input string   SymbolSuffix           = "";       // Leave empty = auto-detect (e.g. ".m", "m", ".pro")
input string   SymbolPrefix           = "";       // Leave empty = auto-detect
input ulong    MagicNumber            = 202602;

//--- Independent MQL5 Hard Risk Guard (Safety Net)
input double   HardMaxDailyDDPct      = 3.50;     // Hardware daily drawdown kill-switch (%)
input double   HardMaxLot             = 0.50;     // Hard lot cap per trade
input int      HardMaxPositions       = 4;        // Hard concurrent position cap
input double   HardMaxSpreadPoints    = 80.0;     // Absolute emergency spread cap (points)
input int      HeartbeatTimeoutSec    = 15;       // Freeze new entries if Python silent
input int      DefaultOrderTTLMs      = 1500;     // Reject stale commands older than TTL
input bool     AutoFlattenOnHardDD    = false;    // Flatten all if HardMaxDailyDDPct hit
input bool     SendNewsCalendar       = true;     // Send high-impact news from the MT5 economic calendar

//--- Globals
CTrade         g_trade;
string         g_logical[];
string         g_broker[];
datetime       g_lastBarTime[];
datetime       g_dayStartDate         = 0;
double         g_dayStartEquity       = 0.0;
datetime       g_lastPythonHeartbeat  = 0;
datetime       g_lastSpecPush         = 0;
string         g_prefix               = "";
string         g_suffix               = "";
int            g_lastSyncCode         = 0;
datetime       g_lastErrorPrint       = 0;
bool           g_connected            = false;
datetime       g_lastCalendarPush     = 0;

string BrokerOf(const string logical);  // defined below (auto symbol mapping)

//--- Idempotency Ring Buffer (256 processed client_order_ids)
string         g_seenOrderIds[256];
int            g_seenCursor           = 0;

//--- Pending Acks Queue
struct AckItem {
   string client_order_id;
   string status;
   ulong  ticket;
   double fill_price;
   double slippage_points;
   double commission;
   int    error_code;
   string message;
};
AckItem        g_acks[];

//+------------------------------------------------------------------+
//| Helper: Check/Record Idempotent Order ID                         |
//+------------------------------------------------------------------+
bool IsOrderIdSeen(const string cid) {
   if(cid == "") return false;
   for(int i = 0; i < 256; i++) {
      if(g_seenOrderIds[i] == cid) return true;
   }
   return false;
}

void RecordOrderId(const string cid) {
   if(cid == "") return;
   g_seenOrderIds[g_seenCursor] = cid;
   g_seenCursor = (g_seenCursor + 1) % 256;
}

void PushAck(const string cid, const string status, ulong ticket, double fillPrice, double slipPts, double comm, int err, const string msg) {
   int n = ArraySize(g_acks);
   ArrayResize(g_acks, n + 1);
   g_acks[n].client_order_id = cid;
   g_acks[n].status          = status;
   g_acks[n].ticket          = ticket;
   g_acks[n].fill_price      = fillPrice;
   g_acks[n].slippage_points = slipPts;
   g_acks[n].commission      = comm;
   g_acks[n].error_code      = err;
   g_acks[n].message         = msg;
}

//+------------------------------------------------------------------+
//| Helper: Auto-Detect Broker Supported Filling Mode                |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING DetectFillingMode(const string sym) {
   long mode = SymbolInfoInteger(sym, SYMBOL_FILLING_MODE);
   if((mode & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   if((mode & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   return ORDER_FILLING_RETURN;
}

string FillingModeName(ENUM_ORDER_TYPE_FILLING fm) {
   if(fm == ORDER_FILLING_IOC) return "IOC";
   if(fm == ORDER_FILLING_FOK) return "FOK";
   return "RETURN";
}

string EscapeJson(string s) {
   StringReplace(s, "\\", "\\\\");
   StringReplace(s, "\"", "\\\"");
   return s;
}

//+------------------------------------------------------------------+
//| Helper: Daily Equity Reset & DD Calculation                      |
//+------------------------------------------------------------------+
double ComputeDailyDDPct() {
   MqlDateTime dtNow, dtDay;
   TimeToStruct(TimeGMT(), dtNow);
   TimeToStruct(g_dayStartDate, dtDay);
   if(g_dayStartEquity <= 0.0 || dtNow.day != dtDay.day || dtNow.mon != dtDay.mon) {
      g_dayStartDate   = TimeGMT();
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   }
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(g_dayStartEquity <= 0.0) return 0.0;
   double dd = ((g_dayStartEquity - eq) / g_dayStartEquity) * 100.0;
   return dd > 0.0 ? dd : 0.0;
}

int CountOurPositions() {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong tk = PositionGetTicket(i);
      if(tk > 0 && PositionGetInteger(POSITION_MAGIC) == (long)MagicNumber) count++;
   }
   return count;
}

void FlattenAllPositions() {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)MagicNumber) continue;
      string sym = PositionGetString(POSITION_SYMBOL);
      g_trade.SetTypeFilling(DetectFillingMode(sym));
      g_trade.PositionClose(tk, 30);
   }
}

//+------------------------------------------------------------------+
//| Lightweight JSON Extractors for OrderCommand Objects             |
//+------------------------------------------------------------------+
string JsonGetStr(const string obj, const string key) {
   string pat = "\"" + key + "\":";
   int p = StringFind(obj, pat);
   if(p < 0) return "";
   p += StringLen(pat);
   while(p < StringLen(obj) && (StringGetCharacter(obj, p) == ' ' || StringGetCharacter(obj, p) == '\t')) p++;
   if(p >= StringLen(obj) || StringGetCharacter(obj, p) != '\"') return "";
   p++;
   int end = StringFind(obj, "\"", p);
   if(end < 0) return "";
   return StringSubstr(obj, p, end - p);
}

double JsonGetNum(const string obj, const string key) {
   string pat = "\"" + key + "\":";
   int p = StringFind(obj, pat);
   if(p < 0) return 0.0;
   p += StringLen(pat);
   while(p < StringLen(obj) && StringGetCharacter(obj, p) == ' ') p++;
   int end = p;
   while(end < StringLen(obj)) {
      ushort ch = StringGetCharacter(obj, end);
      if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == 'e' || ch == 'E' || ch == '+') end++;
      else break;
   }
   if(end <= p) return 0.0;
   return StringToDouble(StringSubstr(obj, p, end - p));
}

//+------------------------------------------------------------------+
//| Execute Single Validated OrderCommand under MQL5 Hard Risk Guard |
//+------------------------------------------------------------------+
void ExecuteCommand(const string obj) {
   string cid       = JsonGetStr(obj, "client_order_id");
   string action    = JsonGetStr(obj, "action");
   string symLog    = JsonGetStr(obj, "symbol");
   string symBrok   = JsonGetStr(obj, "broker_symbol");
   string side      = JsonGetStr(obj, "side");
   double vol       = JsonGetNum(obj, "volume");
   double sl        = JsonGetNum(obj, "sl");
   double tp        = JsonGetNum(obj, "tp");
   ulong  ticket    = (ulong)JsonGetNum(obj, "ticket");
   double expPrice  = JsonGetNum(obj, "expected_price");
   long   expMs     = (long)JsonGetNum(obj, "expires_at_ms");

   if(symBrok == "") symBrok = BrokerOf(symLog);

   // 1. Idempotency Guard
   if(IsOrderIdSeen(cid)) {
      PushAck(cid, "DUPLICATE", 0, 0.0, 0.0, 0.0, 0, "Duplicate client_order_id ignored");
      return;
   }
   RecordOrderId(cid);

   // 2. Command Expiry TTL Guard
   long nowMs = (long)TimeGMT() * 1000;
   if(expMs > 0 && nowMs > expMs + 3000) {
      PushAck(cid, "EXPIRED", 0, 0.0, 0.0, 0.0, 0, "Order expired past TTL");
      return;
   }

   // 3. Emergency Flatten Always Allowed
   if(action == "FLATTEN_ALL") {
      FlattenAllPositions();
      PushAck(cid, "FILLED", 0, 0.0, 0.0, 0.0, 0, "All positions flattened");
      return;
   }

   if(action == "CLOSE") {
      if(PositionSelectByTicket(ticket)) {
         string psym = PositionGetString(POSITION_SYMBOL);
         g_trade.SetTypeFilling(DetectFillingMode(psym));
         bool ok = g_trade.PositionClose(ticket, 30);
         PushAck(cid, ok ? "FILLED" : "REJECTED", ticket, g_trade.ResultPrice(), 0.0, 0.0, (int)g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
      } else {
         PushAck(cid, "REJECTED", ticket, 0.0, 0.0, 0.0, 0, "Ticket not found");
      }
      return;
   }

   if(action == "PARTIAL_CLOSE") {
      if(PositionSelectByTicket(ticket)) {
         string psym = PositionGetString(POSITION_SYMBOL);
         double curVol = PositionGetDouble(POSITION_VOLUME);
         double minVol = SymbolInfoDouble(psym, SYMBOL_VOLUME_MIN);
         double stepVol = SymbolInfoDouble(psym, SYMBOL_VOLUME_STEP);
         double closeVol = MathFloor((curVol * 0.5) / stepVol) * stepVol;
         if(closeVol >= minVol && (curVol - closeVol) >= minVol) {
            g_trade.SetTypeFilling(DetectFillingMode(psym));
            bool ok = g_trade.PositionClosePartial(ticket, closeVol, 30);
            PushAck(cid, ok ? "FILLED" : "REJECTED", ticket, g_trade.ResultPrice(), 0.0, 0.0, (int)g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
         } else {
            PushAck(cid, "REJECTED", ticket, 0.0, 0.0, 0.0, 0, "Volume too small for partial close");
         }
      }
      return;
   }

   if(action == "MODIFY") {
      if(PositionSelectByTicket(ticket)) {
         string psym = PositionGetString(POSITION_SYMBOL);
         int digs = (int)SymbolInfoInteger(psym, SYMBOL_DIGITS);
         bool ok = g_trade.PositionModify(ticket, NormalizeDouble(sl, digs), NormalizeDouble(tp, digs));
         PushAck(cid, ok ? "FILLED" : "REJECTED", ticket, 0.0, 0.0, 0.0, (int)g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
      }
      return;
   }

   // 4. OPEN ORDER — Enforce Independent MQL5 Hard Risk Guard!
   if(action == "OPEN") {
      double ddPct = ComputeDailyDDPct();
      if(ddPct >= HardMaxDailyDDPct) {
         if(AutoFlattenOnHardDD) FlattenAllPositions();
         PushAck(cid, "REJECTED", 0, 0.0, 0.0, 0.0, 9001, "MQL5 Hard Daily DD Limit breached");
         return;
      }
      if(CountOurPositions() >= HardMaxPositions) {
         PushAck(cid, "REJECTED", 0, 0.0, 0.0, 0.0, 9002, "MQL5 HardMaxPositions reached");
         return;
      }
      if(vol <= 0.0 || vol > HardMaxLot) {
         PushAck(cid, "REJECTED", 0, 0.0, 0.0, 0.0, 9003, "Volume violates MQL5 HardMaxLot");
         return;
      }
      SymbolSelect(symBrok, true);
      double pt     = SymbolInfoDouble(symBrok, SYMBOL_POINT);
      int    digs   = (int)SymbolInfoInteger(symBrok, SYMBOL_DIGITS);
      double ask    = SymbolInfoDouble(symBrok, SYMBOL_ASK);
      double bid    = SymbolInfoDouble(symBrok, SYMBOL_BID);
      double spPts  = (pt > 0.0) ? (ask - bid) / pt : 0.0;
      if(spPts > HardMaxSpreadPoints) {
         PushAck(cid, "REJECTED", 0, 0.0, 0.0, 0.0, 9004, "MQL5 HardMaxSpreadPoints exceeded");
         return;
      }

      g_trade.SetExpertMagicNumber(MagicNumber);
      g_trade.SetDeviationInPoints(25);
      g_trade.SetTypeFilling(DetectFillingMode(symBrok));

      double normSl = (sl > 0.0) ? NormalizeDouble(sl, digs) : 0.0;
      double normTp = (tp > 0.0) ? NormalizeDouble(tp, digs) : 0.0;
      string cmt    = StringSubstr(cid, 0, 28);

      bool ok = false;
      if(side == "BUY")       ok = g_trade.Buy(vol, symBrok, ask, normSl, normTp, cmt);
      else if(side == "SELL") ok = g_trade.Sell(vol, symBrok, bid, normSl, normTp, cmt);

      double fillPx = g_trade.ResultPrice();
      double slipPts = (pt > 0.0 && expPrice > 0.0 && fillPx > 0.0) ? MathAbs(fillPx - expPrice) / pt : 0.0;
      ulong dealTk = g_trade.ResultOrder();
      PushAck(cid, ok ? "FILLED" : "REJECTED", dealTk, fillPx, slipPts, 0.0, (int)g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Broker server time minus UTC, rounded to 15 minutes              |
//+------------------------------------------------------------------+
long ServerUtcOffsetSec() {
   double diff = (double)(TimeTradeServer() - TimeGMT());
   return (long)(MathRound(diff / 900.0) * 900.0);
}

bool HasCurrency(string &list[], const string c) {
   for(int i = 0; i < ArraySize(list); i++) if(list[i] == c) return true;
   return false;
}

//+------------------------------------------------------------------+
//| High-impact news from the MT5 economic calendar (every 15 min)   |
//| Returns "" when not due or the calendar is unavailable.          |
//+------------------------------------------------------------------+
string BuildCalendarJson() {
   if(!SendNewsCalendar) return "";
   if(g_lastCalendarPush != 0 && TimeGMT() - g_lastCalendarPush < 900) return "";
   g_lastCalendarPush = TimeGMT();
   long offset = ServerUtcOffsetSec();
   string curr[];
   for(int i = 0; i < ArraySize(g_logical); i++) {
      string s = g_logical[i];
      if(StringLen(s) < 6) continue;
      string a = StringSubstr(s, 0, 3);
      string b = StringSubstr(s, 3, 3);
      if(a != "XAU" && a != "XAG" && !HasCurrency(curr, a)) { int n = ArraySize(curr); ArrayResize(curr, n + 1); curr[n] = a; }
      if(b != "XAU" && b != "XAG" && !HasCurrency(curr, b)) { int n = ArraySize(curr); ArrayResize(curr, n + 1); curr[n] = b; }
   }
   datetime from = TimeTradeServer() - 3600;
   datetime to   = TimeTradeServer() + 36 * 3600;
   string items = "";
   int count = 0;
   bool anyOk = false;
   for(int c = 0; c < ArraySize(curr); c++) {
      MqlCalendarValue vals[];
      ResetLastError();
      bool ok = CalendarValueHistory(vals, from, to, NULL, curr[c]);
      if(!ok) continue;
      anyOk = true;
      int n = ArraySize(vals);
      for(int k = 0; k < n && count < 60; k++) {
         MqlCalendarEvent ev;
         if(!CalendarEventById(vals[k].event_id, ev)) continue;
         if(ev.importance != CALENDAR_IMPORTANCE_HIGH) continue;
         if(count > 0) items += ",";
         items += "{\"title\":\"" + EscapeJson(ev.name) + "\",\"currency\":\"" + curr[c] +
                  "\",\"time_utc\":" + IntegerToString((long)vals[k].time - offset) + ",\"impact\":\"HIGH\"}";
         count++;
      }
   }
   if(!anyOk) return "";
   return "\"calendar\":[" + items + "],";
}

//+------------------------------------------------------------------+
//| Build BridgeSyncRequest JSON Payload                             |
//+------------------------------------------------------------------+
string BuildSyncJson() {
   double bal   = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq    = AccountInfoDouble(ACCOUNT_EQUITY);
   double marg  = AccountInfoDouble(ACCOUNT_MARGIN);
   double fmarg = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double ddPct = ComputeDailyDDPct();
   double dpnl  = eq - g_dayStartEquity;
   bool isDemo  = (AccountInfoInteger(ACCOUNT_TRADE_MODE) != ACCOUNT_TRADE_MODE_REAL);
   long nowMs   = (long)TimeGMT() * 1000;

   string j = "{";
   j += "\"timestamp_ms\":" + IntegerToString(nowMs) + ",";
   j += "\"account\":{";
   j += "\"balance\":" + DoubleToString(bal, 2) + ",";
   j += "\"equity\":" + DoubleToString(eq, 2) + ",";
   j += "\"margin\":" + DoubleToString(marg, 2) + ",";
   j += "\"free_margin\":" + DoubleToString(fmarg, 2) + ",";
   j += "\"day_start_equity\":" + DoubleToString(g_dayStartEquity, 2) + ",";
   j += "\"daily_pnl\":" + DoubleToString(dpnl, 2) + ",";
   j += "\"daily_dd_pct\":" + DoubleToString(ddPct, 4) + ",";
   j += "\"currency\":\"" + AccountInfoString(ACCOUNT_CURRENCY) + "\",";
   j += "\"is_demo\":" + (isDemo ? "true" : "false") + "},";

   // Hard Guard Telemetry
   j += "\"hard_guard\":{";
   j += "\"hard_max_daily_dd_pct\":" + DoubleToString(HardMaxDailyDDPct, 2) + ",";
   j += "\"hard_max_lot\":" + DoubleToString(HardMaxLot, 2) + ",";
   j += "\"hard_max_positions\":" + IntegerToString(HardMaxPositions) + ",";
   j += "\"heartbeat_timeout_sec\":" + IntegerToString(HeartbeatTimeoutSec) + ",";
   j += "\"order_ttl_ms\":" + IntegerToString(DefaultOrderTTLMs) + ",";
   j += "\"ea_version\":\"2.1.0\",\"synced\":true},";
   j += "\"server_utc_offset_sec\":" + IntegerToString(ServerUtcOffsetSec()) + ",";
   j += BuildCalendarJson();

   // Ticks
   j += "\"ticks\":[";
   int nSym = ArraySize(g_logical);
   for(int i = 0; i < nSym; i++) {
      string bs = g_broker[i];
      double bid = SymbolInfoDouble(bs, SYMBOL_BID);
      double ask = SymbolInfoDouble(bs, SYMBOL_ASK);
      double pt  = SymbolInfoDouble(bs, SYMBOL_POINT);
      int digs   = (int)SymbolInfoInteger(bs, SYMBOL_DIGITS);
      double sp  = (pt > 0.0) ? (ask - bid) / pt : 0.0;
      if(i > 0) j += ",";
      j += "{\"symbol\":\"" + g_logical[i] + "\",\"bid\":" + DoubleToString(bid, digs) +
           ",\"ask\":" + DoubleToString(ask, digs) + ",\"spread_points\":" + DoubleToString(sp, 2) +
           ",\"timestamp_ms\":" + IntegerToString(nowMs) + "}";
   }
   j += "],";

   // Closed M1 Bars (initial 80 bars on first push, then newly closed bars)
   j += "\"closed_bars\":[";
   bool firstBar = true;
   for(int i = 0; i < nSym; i++) {
      string bs = g_broker[i];
      int digs  = (int)SymbolInfoInteger(bs, SYMBOL_DIGITS);
      MqlRates rates[];
      int countReq = (g_lastBarTime[i] == 0) ? 80 : 3;
      int copied = CopyRates(bs, PERIOD_M1, 1, countReq, rates);
      if(copied <= 0) continue;
      ArraySetAsSeries(rates, false);
      for(int k = 0; k < copied; k++) {
         if(rates[k].time <= g_lastBarTime[i]) continue;
         if(!firstBar) j += ",";
         firstBar = false;
         j += "{\"symbol\":\"" + g_logical[i] + "\",\"timeframe\":\"M1\",\"timestamp\":" + IntegerToString((long)rates[k].time) +
              ",\"open\":" + DoubleToString(rates[k].open, digs) +
              ",\"high\":" + DoubleToString(rates[k].high, digs) +
              ",\"low\":" + DoubleToString(rates[k].low, digs) +
              ",\"close\":" + DoubleToString(rates[k].close, digs) +
              ",\"volume\":" + IntegerToString((long)rates[k].tick_volume) +
              ",\"spread_points\":" + IntegerToString(rates[k].spread) + "}";
         g_lastBarTime[i] = rates[k].time;
      }
   }
   j += "],";

   // Symbol Specs (sent on init and every 60s)
   j += "\"symbol_specs\":[";
   bool sendSpecs = (TimeGMT() - g_lastSpecPush >= 60 || g_lastSpecPush == 0);
   if(sendSpecs) {
      g_lastSpecPush = TimeGMT();
      for(int i = 0; i < nSym; i++) {
         string bs = g_broker[i];
         if(i > 0) j += ",";
         j += "{\"symbol\":\"" + g_logical[i] + "\",\"broker_symbol\":\"" + bs + "\"," +
              "\"digits\":" + IntegerToString((int)SymbolInfoInteger(bs, SYMBOL_DIGITS)) + "," +
              "\"point\":" + DoubleToString(SymbolInfoDouble(bs, SYMBOL_POINT), 8) + "," +
              "\"tick_size\":" + DoubleToString(SymbolInfoDouble(bs, SYMBOL_TRADE_TICK_SIZE), 8) + "," +
              "\"tick_value\":" + DoubleToString(SymbolInfoDouble(bs, SYMBOL_TRADE_TICK_VALUE), 4) + "," +
              "\"contract_size\":" + DoubleToString(SymbolInfoDouble(bs, SYMBOL_TRADE_CONTRACT_SIZE), 2) + "," +
              "\"volume_min\":" + DoubleToString(SymbolInfoDouble(bs, SYMBOL_VOLUME_MIN), 4) + "," +
              "\"volume_max\":" + DoubleToString(SymbolInfoDouble(bs, SYMBOL_VOLUME_MAX), 2) + "," +
              "\"volume_step\":" + DoubleToString(SymbolInfoDouble(bs, SYMBOL_VOLUME_STEP), 4) + "," +
              "\"stops_level\":" + IntegerToString((int)SymbolInfoInteger(bs, SYMBOL_TRADE_STOPS_LEVEL)) + "," +
              "\"freeze_level\":" + IntegerToString((int)SymbolInfoInteger(bs, SYMBOL_TRADE_FREEZE_LEVEL)) + "," +
              "\"filling_mode\":\"" + FillingModeName(DetectFillingMode(bs)) + "\"}";
      }
   }
   j += "],";

   // Open Positions (Single Source of Truth)
   j += "\"positions\":[";
   bool firstPos = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)MagicNumber) continue;
      string psym = PositionGetString(POSITION_SYMBOL);
      string logSym = psym;
      for(int s = 0; s < nSym; s++) { if(g_broker[s] == psym) { logSym = g_logical[s]; break; } }
      int digs = (int)SymbolInfoInteger(psym, SYMBOL_DIGITS);
      string pside = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      if(!firstPos) j += ",";
      firstPos = false;
      j += "{\"ticket\":" + IntegerToString((long)tk) +
           ",\"client_order_id\":\"" + EscapeJson(PositionGetString(POSITION_COMMENT)) + "\"" +
           ",\"symbol\":\"" + logSym + "\"" +
           ",\"side\":\"" + pside + "\"" +
           ",\"volume\":" + DoubleToString(PositionGetDouble(POSITION_VOLUME), 4) +
           ",\"open_price\":" + DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), digs) +
           ",\"current_price\":" + DoubleToString(PositionGetDouble(POSITION_PRICE_CURRENT), digs) +
           ",\"sl\":" + DoubleToString(PositionGetDouble(POSITION_SL), digs) +
           ",\"tp\":" + DoubleToString(PositionGetDouble(POSITION_TP), digs) +
           ",\"profit\":" + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2) +
           ",\"swap\":" + DoubleToString(PositionGetDouble(POSITION_SWAP), 2) +
           ",\"open_time\":" + IntegerToString((long)PositionGetInteger(POSITION_TIME)) + "}";
   }
   j += "],";

   // Order Acks
   j += "\"acks\":[";
   int nAcks = ArraySize(g_acks);
   for(int i = 0; i < nAcks; i++) {
      if(i > 0) j += ",";
      j += "{\"client_order_id\":\"" + EscapeJson(g_acks[i].client_order_id) + "\"," +
           "\"status\":\"" + g_acks[i].status + "\"," +
           "\"ticket\":" + IntegerToString((long)g_acks[i].ticket) + "," +
           "\"fill_price\":" + DoubleToString(g_acks[i].fill_price, 6) + "," +
           "\"slippage_points\":" + DoubleToString(g_acks[i].slippage_points, 2) + "," +
           "\"commission\":" + DoubleToString(g_acks[i].commission, 2) + "," +
           "\"error_code\":" + IntegerToString(g_acks[i].error_code) + "," +
           "\"message\":\"" + EscapeJson(g_acks[i].message) + "\"}";
   }
   ArrayResize(g_acks, 0);
   j += "]}";
   return j;
}

//+------------------------------------------------------------------+
//| Parse & Dispatch Commands Array from Response JSON               |
//+------------------------------------------------------------------+
void ProcessResponse(const string resp) {
   int cmdStart = StringFind(resp, "\"commands\":[");
   if(cmdStart < 0) return;
   int pos = cmdStart + StringLen("\"commands\":[");
   int len = StringLen(resp);
   while(pos < len) {
      int objStart = StringFind(resp, "{", pos);
      int arrEnd   = StringFind(resp, "]", pos);
      if(objStart < 0 || (arrEnd >= 0 && objStart > arrEnd)) break;
      int objEnd = StringFind(resp, "}", objStart);
      if(objEnd < 0) break;
      string cmdObj = StringSubstr(resp, objStart, objEnd - objStart + 1);
      ExecuteCommand(cmdObj);
      pos = objEnd + 1;
   }
}

//+------------------------------------------------------------------+
//| Expert Lifecycle                                                 |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Broker symbol auto-detection (prefix/suffix such as XAUUSD.m)    |
//+------------------------------------------------------------------+
bool SymbolOk(const string name) {
   if(name == "") return false;
   bool custom = false;
   return SymbolExist(name, custom);
}

string StripAffixes(const string broker) {
   string s = broker;
   if(g_prefix != "" && StringFind(s, g_prefix) == 0) s = StringSubstr(s, StringLen(g_prefix));
   int ls = StringLen(g_suffix);
   if(ls > 0 && StringLen(s) > ls && StringSubstr(s, StringLen(s) - ls) == g_suffix) s = StringSubstr(s, 0, StringLen(s) - ls);
   return s;
}

// Learn the broker's prefix/suffix from the chart symbol (e.g. chart XAUUSD.m -> suffix ".m").
void DetectAffixes(string &watch[]) {
   g_prefix = SymbolPrefix;
   g_suffix = SymbolSuffix;
   if(g_prefix != "" || g_suffix != "") return;
   string chart = _Symbol;
   for(int i = 0; i < ArraySize(watch); i++) {
      int at = StringFind(chart, watch[i]);
      if(watch[i] != "" && at >= 0) {
         g_prefix = StringSubstr(chart, 0, at);
         g_suffix = StringSubstr(chart, at + StringLen(watch[i]));
         return;
      }
   }
   // Chart symbol not in the watchlist: assume 6 letters + suffix (EURJPY.m, BTCUSDm ...)
   if(StringLen(chart) > 6) g_suffix = StringSubstr(chart, 6);
}

// Find the broker's name for a plain symbol. Returns "" when the broker does not offer it.
string ResolveBroker(const string logical) {
   string cand = g_prefix + logical + g_suffix;
   if(SymbolOk(cand)) return cand;
   if(SymbolOk(logical)) return logical;
   // Last resort: shortest broker symbol that contains the plain name (XAUUSDm, XAUUSD.pro, ...)
   string best = "";
   int total = SymbolsTotal(false);
   for(int k = 0; k < total; k++) {
      string nm = SymbolName(k, false);
      if(StringFind(nm, logical) >= 0 && (best == "" || StringLen(nm) < StringLen(best))) best = nm;
   }
   return best;
}

string BrokerOf(const string logical) {
   for(int i = 0; i < ArraySize(g_logical); i++) if(g_logical[i] == logical) return g_broker[i];
   return ResolveBroker(logical);
}

void AddSymbol(const string logical, const string broker) {
   for(int i = 0; i < ArraySize(g_logical); i++) if(g_logical[i] == logical || g_broker[i] == broker) return;
   int n = ArraySize(g_logical);
   ArrayResize(g_logical, n + 1);
   ArrayResize(g_broker, n + 1);
   ArrayResize(g_lastBarTime, n + 1);
   g_logical[n] = logical;
   g_broker[n] = broker;
   g_lastBarTime[n] = 0;
   SymbolSelect(broker, true);
}

string SyncErrorHint(const int code, const int err) {
   if(code == -1 && err == 4014) return "URL not allowed. Tools > Options > Expert Advisors > Allow WebRequest, add http://127.0.0.1:8765";
   if(code == -1 && (err == 5203 || err == 4060)) return "Cannot reach the robot (backend). Run start_all.bat first";
   if(code == 401 || code == 403) return "Wrong token. Set BridgeAuthToken = BRIDGE_AUTH_TOKEN from .env, or run start_all.bat again";
   if(code == 422) return "Backend rejected the data (422). Update the repo and the EA";
   if(code == 503) return "Backend not ready or weak server token. Run start_all.bat --setup";
   return "Check the robot window (start_all.bat) for details";
}

//+------------------------------------------------------------------+
//| Expert Lifecycle                                                 |
//+------------------------------------------------------------------+
int OnInit() {
   if(StringLen(BridgeAuthToken) < 24 || StringFind(BridgeAuthToken, "change_me") >= 0) {
      Print("MaxTradingBridge: BridgeAuthToken is empty or a placeholder. Copy BRIDGE_AUTH_TOKEN from .env into the EA inputs.");
      return(INIT_PARAMETERS_INCORRECT);
   }
   string watch[];
   int n = StringSplit(LogicalSymbols, ',', watch);
   for(int i = 0; i < n; i++) { StringTrimLeft(watch[i]); StringTrimRight(watch[i]); StringToUpper(watch[i]); }
   DetectAffixes(watch);
   ArrayResize(g_logical, 0);
   ArrayResize(g_broker, 0);
   ArrayResize(g_lastBarTime, 0);
   if(IncludeChartSymbol) AddSymbol(StripAffixes(_Symbol), _Symbol);
   for(int i = 0; i < n; i++) {
      if(watch[i] == "") continue;
      string b = ResolveBroker(watch[i]);
      if(b == "") { Print("MaxTradingBridge: broker has no ", watch[i], ", skipped"); continue; }
      AddSymbol(watch[i], b);
   }
   string mapping = "";
   for(int i = 0; i < ArraySize(g_logical); i++) mapping += (i > 0 ? ", " : "") + g_logical[i] + "=" + g_broker[i];
   Print("MaxTradingBridge: prefix='", g_prefix, "' suffix='", g_suffix, "' symbols: ", mapping);
   g_dayStartEquity      = AccountInfoDouble(ACCOUNT_EQUITY);
   g_dayStartDate        = TimeGMT();
   g_lastPythonHeartbeat = TimeGMT();
   EventSetMillisecondTimer(SyncIntervalMs);
   Print("MaxTradingBridge v2.10 initialized. Bridge URL: ", BridgeUrl);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   EventKillTimer();
}

void OnTimer() {
   // Independent Hard DD check even if Python is offline
   if(ComputeDailyDDPct() >= HardMaxDailyDDPct && AutoFlattenOnHardDD) {
      FlattenAllPositions();
   }

   string payload = BuildSyncJson();
   char postData[];
   int bytes = StringToCharArray(payload, postData, 0, WHOLE_ARRAY, CP_UTF8);
   if(bytes > 0) ArrayResize(postData, bytes - 1);

   char result[];
   string resultHeaders;
   string headers = "Content-Type: application/json\r\nX-Bridge-Token: " + BridgeAuthToken + "\r\n";

   ResetLastError();
   int code = WebRequest("POST", BridgeUrl, headers, WebTimeoutMs, postData, result, resultHeaders);
   int err = GetLastError();
   if(code == 200) {
      g_lastPythonHeartbeat = TimeGMT();
      if(!g_connected) {
         Print("MaxTradingBridge: CONNECTED to robot (", BridgeUrl, ")");
         Comment("MaxTradingBridge v2.10 | CONNECTED to robot | ", ArraySize(g_logical), " symbols");
      }
      g_connected = true;
      string respStr = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
      ProcessResponse(respStr);
   } else {
      // Always say why the sync failed (at most every 10 s), so the Experts tab is never silent.
      if(g_connected || code != g_lastSyncCode || TimeGMT() - g_lastErrorPrint >= 10) {
         string hint = SyncErrorHint(code, err);
         Print("MaxTradingBridge: sync GAGAL (http=", code, ", err=", err, "). ", hint);
         Comment("MaxTradingBridge v2.10 | NOT CONNECTED (http=", code, ", err=", err, ")\n", hint);
         g_lastErrorPrint = TimeGMT();
      }
      g_connected = false;
   }
   g_lastSyncCode = code;
}
//+------------------------------------------------------------------+

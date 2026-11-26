//+------------------------------------------------------------------+
//|                                            MaxTradingBridge.mq5  |
//|                                    Max Trading Agent v2 (MT5 EA) |
//| Zero-DLL Local HTTP/JSON Bridge + Independent Hard Risk Guard    |
//+------------------------------------------------------------------+
#property copyright "Max Trading Agent v2"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Bridge Connection Inputs
input string   BridgeUrl              = "http://127.0.0.1:8765/api/bridge/sync";
input string   BridgeAuthToken        = "mta2_dev_change_me_to_random_secret_32chars";
input int      SyncIntervalMs         = 350;      // Fast poll interval (ms)
input int      WebTimeoutMs           = 1200;     // HTTP timeout (ms)
input string   LogicalSymbols         = "XAUUSD,EURUSD,GBPUSD,USDJPY";
input string   SymbolSuffix           = "";       // e.g. "m" or ".pro" for broker suffix
input ulong    MagicNumber            = 202602;

//--- Independent MQL5 Hard Risk Guard (Safety Net)
input double   HardMaxDailyDDPct      = 3.50;     // Hardware daily drawdown kill-switch (%)
input double   HardMaxLot             = 0.50;     // Hard lot cap per trade
input int      HardMaxPositions       = 4;        // Hard concurrent position cap
input double   HardMaxSpreadPoints    = 80.0;     // Absolute emergency spread cap (points)
input int      HeartbeatTimeoutSec    = 15;       // Freeze new entries if Python silent
input int      DefaultOrderTTLMs      = 1500;     // Reject stale commands older than TTL
input bool     AutoFlattenOnHardDD    = false;    // Flatten all if HardMaxDailyDDPct hit

//--- Globals
CTrade         g_trade;
string         g_logical[];
string         g_broker[];
datetime       g_lastBarTime[];
datetime       g_dayStartDate         = 0;
double         g_dayStartEquity       = 0.0;
datetime       g_lastPythonHeartbeat  = 0;
datetime       g_lastSpecPush         = 0;

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

   if(symBrok == "") symBrok = symLog + SymbolSuffix;

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
   j += "\"ea_version\":\"2.0.0\",\"synced\":true},";

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
int OnInit() {
   int n = StringSplit(LogicalSymbols, ',', g_logical);
   ArrayResize(g_broker, n);
   ArrayResize(g_lastBarTime, n);
   for(int i = 0; i < n; i++) {
      StringTrimLeft(g_logical[i]);
      StringTrimRight(g_logical[i]);
      g_broker[i] = g_logical[i] + SymbolSuffix;
      g_lastBarTime[i] = 0;
      SymbolSelect(g_broker[i], true);
   }
   g_dayStartEquity      = AccountInfoDouble(ACCOUNT_EQUITY);
   g_dayStartDate        = TimeGMT();
   g_lastPythonHeartbeat = TimeGMT();
   EventSetMillisecondTimer(SyncIntervalMs);
   Print("MaxTradingBridge v2.00 initialized. Bridge URL: ", BridgeUrl);
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

   int code = WebRequest("POST", BridgeUrl, headers, WebTimeoutMs, postData, result, resultHeaders);
   if(code == 200) {
      g_lastPythonHeartbeat = TimeGMT();
      string respStr = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
      ProcessResponse(respStr);
   } else if(TimeGMT() - g_lastPythonHeartbeat > HeartbeatTimeoutSec) {
      Comment("MaxTradingBridge v2.00 | WATCHDOG: Python Brain offline (", code, ") — New entries frozen.");
   }
}
//+------------------------------------------------------------------+

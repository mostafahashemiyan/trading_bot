import time
import config
from exchange import ExchangeClient
from indicators import prepare_df
from strategy import trend_pullback_signal
from risk import position_size
from tracker import TradeTracker
from report import log_detailed_report # استفاده از لاگر جدید

class TradingBot:
    def __init__(self):
        self.exchange = ExchangeClient()
        self.tracker = TradeTracker()

    def run_once(self, symbol: str):
        print(f"\n--- Checking {symbol} ---")
        report = {"symbol": symbol, "step": "fetching_data", "status": "pending"}

        # ۱. دریافت داده‌ها
        ohlcv_1h = self.exchange.fetch_ohlcv(symbol, "1h", limit=300)
        ohlcv_5m = self.exchange.fetch_ohlcv(symbol, "5m", limit=300)
        
        df1h = prepare_df(ohlcv_1h)
        df5 = prepare_df(ohlcv_5m)

        if df1h.empty or df5.empty:
            print(f"[{symbol}] Data empty or insufficient.")
            return

        # ۲. بررسی سیگنال
        sig = trend_pullback_signal(df1h, df1h, df5) # استفاده از 1h برای هر دو جهت تست
        report.update({
            "step": "strategy_analysis",
            "setup_found": sig.get("setup"),
            "side": sig.get("side"),
            "entry": sig.get("entry"),
            "stop": sig.get("stop"),
            "tp": sig.get("tp"),
            "reasons": sig.get("reasons")
        })

        if not sig.get("setup"):
            print(f"[{symbol}] No Signal: {sig.get('reasons')}")
            log_detailed_report(symbol, report)
            return

        # ۳. ورود به معامله
        print(f"[{symbol}] SIGNAL FOUND! Side: {sig['side']} | Entry: {sig['entry']}")
        
        balance = self.exchange.get_balance_usdt()
        size = position_size(balance, sig['entry'], sig['stop'], config.RISK_PER_TRADE)
        
        if (size * sig['entry']) < config.MIN_NOTIONAL:
            print(f"[{symbol}] Size too small, skipping.")
            report["status"] = "skipped_small_size"
            log_detailed_report(symbol, report)
            return

        # اجرای دستور در صرافی
        side = "buy" if sig["side"] == "LONG" else "sell"
        result = self.exchange.execute_trade(symbol, side, size, sig['stop'], sig['tp'])
        
        report["status"] = "executed"
        report["order_result"] = str(result)
        log_detailed_report(symbol, report)
        print(f"[{symbol}] Trade Executed Successfully.")

    def run(self):
        print("Bot Started. Monitoring markets...")
        while True:
            for symbol in config.SYMBOLS:
                try:
                    self.run_once(symbol)
                except Exception as e:
                    print(f"CRITICAL ERROR for {symbol}: {e}")
            time.sleep(60)

if __name__ == "__main__":
    TradingBot().run()
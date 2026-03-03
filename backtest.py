import os
import sys
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی برای دسترسی به OpenAI
load_dotenv()

# تنظیم مسیر ماژول‌ها
BASE_DIR = os.path.dirname(__file__)
NEW_BOT_DIR = os.path.join(BASE_DIR, "new-bot")
if NEW_BOT_DIR not in sys.path:
    sys.path.append(NEW_BOT_DIR)

import config
from indicators import prepare_df
from strategy import trend_pullback_signal
from risk import position_size
from llm_gatekeeper import llm_decide # وارد کردن فیلتر هوشمند

class AIBacktester:
    def __init__(self, symbol="ETH/USDT"):
        self.symbol = symbol
        self.exchange = ccxt.kucoin()
        self.balance = 1000.0
        self.trades = []

    def run(self):
        # دریافت دیتای تاریخی
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, "5m", limit=1000)
        df = prepare_df(ohlcv)
        
        print(f"Starting AI-Filtered Simulation for {self.symbol}...")
        
        for i in range(200, len(df) - 10):
            current_df = df.iloc[:i+1]
            sig = trend_pullback_signal(current_df, current_df, current_df)

            # اگر استراتژی تایید کرد، حالا هوش مصنوعی باید نظر بدهد
            if sig.get("setup"):
                # استخراج ویژگی‌ها برای ارسال به LLM
                features = {
                    "symbol": self.symbol,
                    "trend": sig["trend"],
                    "rsi_15m": float(current_df["rsi"].iloc[-1]),
                    "rr_ratio": sig["rr"],
                    "side": sig["side"]
                }
                
                print(f"Signal at index {i}. Consulting AI...")
                # فیلتر هوش مصنوعی
                ai_result = llm_decide(features)
                
                if ai_result.get("decision") == "TRADE":
                    print("✅ AI Approved this trade!")
                    self.execute_trade_simulation(df, i, sig)
                else:
                    print(f"❌ AI Rejected: {ai_result.get('reason')}")

        self.report()

    def execute_trade_simulation(self, df, index, sig):
        entry = sig["entry"]
        sl = sig["stop"]
        tp = sig["tp"]
        
        # محاسبه حجم بر اساس ریسک در هر معامله
        size = position_size(self.balance, entry, sl, config.RISK_PER_TRADE)
        
        for j in range(index + 1, len(df)):
            low, high = df.iloc[j]["low"], df.iloc[j]["high"]
            if sig["side"] == "LONG":
                if low <= sl: self.record_trade("LOSS", entry, sl, size); break
                if high >= tp: self.record_trade("WIN", entry, tp, size); break
            else:
                if high <= sl: self.record_trade("LOSS", entry, sl, size); break
                if low >= tp: self.record_trade("WIN", entry, tp, size); break

    def record_trade(self, result, entry, exit, size):
        risk_amount = self.balance * config.RISK_PER_TRADE
        actual_pnl = risk_amount * 2.2 if result == "WIN" else -risk_amount
        self.balance += actual_pnl
        self.trades.append({"balance": self.balance})

    def report(self):
        df_trades = pd.DataFrame(self.trades)
        plt.figure(figsize=(10, 5))
        plt.plot(df_trades["balance"], label="AI Filtered Equity", color='blue')
        plt.axhline(y=1000, color='red', linestyle='--')
        plt.title("Backtest with AI Filtering")
        plt.show()

if __name__ == "__main__":
    AIBacktester().run()
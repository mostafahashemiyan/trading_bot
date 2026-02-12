# config.py

SYMBOLS = [
    "ETH/USDT",
]

TIMEFRAMES = ["1h", "15m", "5m"]

RISK_PER_TRADE = 0.01        # 1%
MIN_RR = 2.0
MAX_STOP_PCT = 1.2

DRY_RUN = True              # 🚨 keep TRUE until confident
COOLDOWN_AFTER_TRADE = 30   # candles

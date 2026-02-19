# config.py

SYMBOLS = [
    "BTC/USDT",
]

TIMEFRAMES = ["1h", "15m", "5m"]

# Execution / Risk
RISK_PER_TRADE = 0.01        # 1%
DRY_RUN = True              # 🚨 keep TRUE until confident

# --- Market type ---
# This strategy is SHORT-only. Spot markets cannot short. Use KuCoin Futures (swap) for real shorts.
USE_FUTURES = True

# Futures risk settings
LEVERAGE = 3                 # set leverage when supported
MARGIN_MODE = "cross"        # "cross" or "isolated" (best-effort; exchange may ignore)

# --- Strategy: 5M Trend + Structure + Pullback (Safe Short) ---
HTF_TIMEFRAME = "15m"        # Pine input.timeframe("15")
RR_RATIO = 1.8
PIVOT_LEN = 5
EMA_FAST = 20
EMA_SLOW = 50
MIN_GAP_BARS = 10            # Min bars between trades (on 5m bars)
MIN_RISK_ATR_MULT = 0.5      # Min risk must be > ATR * this
ATR_LEN = 14

# LLM gatekeeper rules (still used)
MIN_RR = 1.8

# Optional cooldown (not used by this strategy by default)
COOLDOWN_AFTER_TRADE = 30   # candles

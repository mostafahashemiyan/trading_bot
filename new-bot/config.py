"""Bot configuration.

Configured for **KuCoin Futures (USDT-margined)** in **One-Way** mode.

Key points:
- One-Way: per symbol you can hold either a LONG or a SHORT at a time.
  When an opposite signal arrives, the bot closes the current position first,
  then opens the new direction.
- In CCXT, KuCoin USDT-margined swap symbols are typically like: "ETH/USDT:USDT".
  If your market list differs, adjust SYMBOLS accordingly.
"""

# ----------------------
# Exchange / Market
# ----------------------
EXCHANGE_ID = "kucoinfutures"

SYMBOLS = ["ETH/USDT:USDT"]
TIMEFRAME = "5m"

# ----------------------
# Futures settings
# ----------------------
POSITION_MODE = "oneway"  # requested by user
MARGIN_MODE = "isolated"  # "isolated" or "cross"
LEVERAGE = 3

# ----------------------
# Strategy parameters
# ----------------------
FAST_EMA = 20
SLOW_EMA = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_PERIOD = 14
ADX_THRESHOLD = 18
ATR_PERIOD = 14

# ----------------------
# Risk parameters
# ----------------------
RISK_PER_TRADE = 0.01
SL_ATR_MULT = 1.2
TP_ATR_MULT = 2.0
MIN_NOTIONAL = 10

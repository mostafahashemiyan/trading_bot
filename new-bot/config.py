"""
CONFIG — Central configuration for the entire trading system.
Fully redesigned professional version (2026).
"""

# ─────────────────────────────────────────────────────────────
# Exchange & Symbols
# ─────────────────────────────────────────────────────────────
EXCHANGE_ID = "kucoinfutures"

# ✔ MULTI-SYMBOL SUPPORT
# Example:
# SYMBOLS = ["ETH/USDT:USDT", "BTC/USDT:USDT", "SOL/USDT:USDT"]
# You can add/remove symbols freely.
SYMBOLS = ["ETH/USDT:USDT"]   # ← You fill this manually before running the bot.

MAIN_TIMEFRAME = "1h"
HIGH_TF = "1h"
MEDIUM_TF = "15m"
LOW_TF = "5m"

# ─────────────────────────────────────────────────────────────
# Futures Parameters
# ─────────────────────────────────────────────────────────────
LEVERAGE = 3
MARGIN_MODE = "isolated"
POSITION_MODE = "oneway"

# ─────────────────────────────────────────────────────────────
# Indicator Settings
# ─────────────────────────────────────────────────────────────
FAST_EMA_PERIOD = 20
SLOW_EMA_PERIOD = 50

RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35

ATR_PERIOD = 14

ADX_PERIOD = 14
ADX_THRESHOLD = 25   # strict trend filter (conservative)

# Trend quality filters
EMA_SLOPE_LOOKBACK = 3
MIN_EMA_DISTANCE_ATR = 0.20   # ema distance must be at least 20% of ATR

# ─────────────────────────────────────────────────────────────
# Risk Management
# ─────────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.003    # 0.3% risk per trade
SL_ATR_MULT = 1.3
TP_ATR_MULT = 3.0         # R:R ≈ 2.3

# Notional safety limits
SAFETY_FACTOR = 0.15      # max notional = balance × leverage × 0.15
MIN_NOTIONAL_VALUE = 10

# ─────────────────────────────────────────────────────────────
# Quality Filters
# ─────────────────────────────────────────────────────────────
MIN_VOLUME_MULTIPLIER = 1.5     # must exceed ×1.5 of volume MA20
COOLDOWN_CANDLES = 12           # minimum candles between trades

# ─────────────────────────────────────────────────────────────
# Slippage & Fees
# ─────────────────────────────────────────────────────────────
COMMISSION = 0.0006
SLIPPAGE_PCT = 0.0005
MAX_SPREAD_PCT = 0.0015   # 0.15%

# ─────────────────────────────────────────────────────────────
# LLM Gatekeeper
# ─────────────────────────────────────────────────────────────
ENABLE_LLM = True

LLM_MODEL = "gpt-4o-mini"
LLM_MIN_RR = 2.2
LLM_MIN_ADX = 25
LLM_TEMPERATURE = 0.15
LLM_TIMEOUT_SECONDS = 2.0
LLM_FAILSAFE_DECISION = "REJECT"   # safest option for live trading
# ─────────────────────────────────────────────────────────────
# Bot Behavior
# ─────────────────────────────────────────────────────────────
DRY_RUN = True    # ← Change to False for REAL TRADING (live markets)

# Retry settings for network robustness
RETRY_ATTEMPTS = 6
RETRY_DELAY = 2.0
RETRY_MAX_DELAY = 60.0

# Wait time between scans
SCAN_DELAY_SECONDS = 300

# ─────────────────────────────────────────────────────────────
# Backtest Settings
# ─────────────────────────────────────────────────────────────
BACKTEST_INITIAL_BALANCE = 1000
BACKTEST_COMMISSION = COMMISSION
BACKTEST_SLIPPAGE = SLIPPAGE_PCT
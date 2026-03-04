"""
Central configuration file for the trading bot.
All important parameters are here → easy to tune.
Conservative settings for reliability (March 2026).
"""

# ── Exchange & Symbols ────────────────────────────────────────
EXCHANGE_ID = "kucoinfutures"
SYMBOLS = ["ETH/USDT:USDT"]           # can add more later
MAIN_TIMEFRAME = "1h"
HIGH_TF = "1h"
MEDIUM_TF = "15m"
LOW_TF = "5m"

# ── Futures parameters ────────────────────────────────────────
LEVERAGE = 1.5
MARGIN_MODE = "isolated"              # safer than cross
POSITION_MODE = "oneway"              # KuCoin default

# ── Indicator periods & thresholds ────────────────────────────
FAST_EMA_PERIOD = 20
SLOW_EMA_PERIOD = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
ADX_PERIOD = 14
ADX_THRESHOLD = 25                    # strict trend filter
ATR_PERIOD = 14
SAFETY_FACTOR = 0.10
# ── Risk & Money Management ───────────────────────────────────
RISK_PER_TRADE = 0.003               # 0.5% of balance per trade
SL_ATR_MULT = 1.3
TP_ATR_MULT = 3.0                     # target RR ≈ 2.3
SAFETY_FACTOR = 0.15                  # max notional = balance × lev × 0.15
MIN_NOTIONAL_VALUE = 10               # minimum trade size in USDT

# ── Quality filters ───────────────────────────────────────────
MIN_VOLUME_MULTIPLIER = 1.5           # volume > 1.5 × 20-period avg
COOLDOWN_CANDLES = 12                 # min candles between trades

# ── Backtest / Simulation settings ────────────────────────────
COMMISSION = 0.0006                   # KuCoin taker ≈ 0.06%
SLIPPAGE_PCT = 0.0005                 # 0.05% slippage
DRY_RUN = True                        # must be True for live safety


# ── LLM gatekeeper (future) ───────────────────────────────────
# Will be used in bot.py to filter signals
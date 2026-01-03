# 🧠 LLM-Guarded Crypto Trading Bot (Kraken)

A **deterministic, multi-timeframe crypto trading bot** enhanced with an **LLM decision gatekeeper** for conservative, high-probability trade approval.

This project blends **rule-based technical analysis** with **LLM-based risk filtering** to reduce false signals and overtrading.

---

## ✨ Features

- 📊 **Multi-Timeframe Strategy**
  - **1H** → Trend direction
  - **15M** → Pullback validation
  - **5M** → Precise entry timing

- 📐 **Deterministic Trading Logic**
  - EMA trend alignment
  - RSI pullback zone detection
  - Momentum or wick-rejection entries
  - Fixed risk-reward ratio

- 🧠 **LLM Trade Gatekeeper (OpenAI)**
  - Reviews structured trade data
  - Conservative approval logic
  - JSON-only, schema-validated responses

- 🔐 **Safe-by-Default**
  - `DRY_RUN = True` prevents live trading
  - All decisions logged for review

- ⚙️ **Kraken Exchange Integration**
  - Built on `ccxt`
  - Live trading ready when enabled

---

## 📂 Project Structure

trading_bot/
│
├── bot.py # Main execution loop
├── config.py # Configuration & constants
├── exchange.py # Kraken (ccxt) wrapper
├── indicators.py # EMA, RSI, feature engineering
├── strategy.py # Deterministic trade logic
├── llm_gatekeeper.py # OpenAI decision layer
├── risk.py # Position sizing
├── logger.py # JSON logging
├── requirements.txt
└── .env # API keys (not committed)





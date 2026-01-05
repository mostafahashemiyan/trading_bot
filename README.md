# 🧠 LLM-Gated Crypto Trading Bot

A **professional, risk-aware cryptocurrency trading system** that combines  
**deterministic technical analysis** with a **Large Language Model (LLM) acting as a conservative risk gatekeeper**.

This project is designed to be **safe-first**, **explainable**, and **production-oriented**, supporting both:
- 🤖 Automated execution (runs every 60 seconds)
- 🖥️ Manual execution via UI (Gradio, button-based)

---

## 📌 Core Philosophy

> The strategy finds trades.  
> The LLM decides whether the trade deserves to exist.

The LLM is **not a signal generator**.  
It only evaluates already-detected setups and can **approve or veto** them.

If anything is unclear → **NO_TRADE**.

---

## ✨ Key Features

- Multi-timeframe analysis (1H / 15M / 5M)
- Trend-pullback trading strategy
- LLM used strictly as a risk gatekeeper
- Conservative bias (NO_TRADE by default)
- DRY_RUN safety protection
- Structured JSON logging
- Gradio UI for manual execution & inspection
- Clean, modular architecture ready for extension

---

## 🧱 Project Structure

.
├── bot.py               # Main orchestration (auto mode)
├── ui.py                # Gradio UI (manual mode)
├── config.py            # Global configuration
├── exchange.py          # Exchange access (ccxt / Kraken)
├── indicators.py        # EMA / RSI indicators
├── strategy.py          # Trend pullback strategy
├── llm_gatekeeper.py    # LLM decision logic
├── risk.py              # Position sizing
├── logger.py            # JSON logger
├── bot_log.json         # Generated logs
└── README.md

---

## 🧠 Strategy Overview

### 1. Higher-Timeframe Trend (1H)
- EMA50 > EMA200
- Only bullish trend setups are allowed

### 2. Pullback Condition (15M)
- RSI between 40 and 60
- Filters out overextended price moves

### 3. Entry Confirmation (5M)
At least one of the following must occur:
- Strong bullish momentum candle
- Clear lower-wick rejection (liquidity grab)

### 4. Trade Levels
- Entry: Current close
- Stop: Recent swing low (with safety buffer)
- Take-Profit: ~2.2× Risk
- Minimum RR: ≥ 2.0

---

## 🧠 LLM Gatekeeper

The LLM acts as a **risk evaluator**, not a trader.

### What the LLM DOES
- Evaluates confluence
- Confirms trend alignment
- Validates risk–reward
- Approves or vetoes trades conservatively

### What the LLM DOES NOT DO
- Invent strategies
- Use generic market sentiment
- Ignore provided indicators
- Force trades

---

## 📐 LLM Output Schema

{
  "decision": "TRADE" | "NO_TRADE",
  "side": "LONG" | "SHORT" | null,
  "confidence": 0-100,
  "reason": "short explanation"
}

Rules:
- If decision is NO_TRADE → side must be null
- Confidence above 70 only for very strong setups
- JSON only (no markdown, no prose)

---

## 🔐 Safety & Risk Controls

- DRY_RUN = True by default
- Trades execute only if:
  1. Strategy setup is valid
  2. LLM approves the trade
  3. DRY_RUN is disabled

⚠️ Never disable DRY_RUN without extensive testing.

---

## 🧪 Installation

1. Clone the repository

git clone https://github.com/your-username/llm-gated-crypto-bot.git  
cd llm-gated-crypto-bot

2. Install dependencies

pip install ccxt pandas numpy gradio python-dotenv openai

3. Environment variables

Create a .env file:

OPENAI_API_KEY=your_openai_key  
OPENAI_MODEL=gpt-4o-mini  
KRAKEN_API_KEY=your_kraken_key  
KRAKEN_API_SECRET=your_kraken_secret

---

## 🚀 Running the Bot (Automated Mode)

python bot.py

Runs every 60 seconds.  
Use this mode for paper trading or live trading (after disabling DRY_RUN).

---

## 🖥️ Running the UI (Manual Mode)

python ui.py

Then open:
http://127.0.0.1:7860

This mode runs only when the button is clicked.

---

## 🔁 Automated Mode vs UI Mode

Automated Mode:
- Runs every 60 seconds
- Suitable for background execution

UI Mode:
- Manual execution only
- Suitable for debugging and inspection

Do not run both modes simultaneously.

---

## 📊 Logging

All activity is logged to bot_log.json.

Each log entry includes:
- Timestamp
- Strategy output
- LLM decision
- Execution status

Useful for debugging, analysis, and backtesting.

---

## 🛠️ Extensibility Roadmap

- Multi-symbol trading
- Cooldown & trade memory
- Backtesting engine
- Confidence calibration
- Telegram / Discord alerts
- Performance dashboards

---

## ⚠️ Disclaimer

This software is provided for educational and research purposes only.  
Cryptocurrency trading involves significant financial risk.

You are solely responsible for any trades executed using this system.

---

## 🙌 Final Note

This bot is intentionally conservative.

If you understand why it refuses trades,  
you are using it correctly.

Build safely. Trade responsibly.

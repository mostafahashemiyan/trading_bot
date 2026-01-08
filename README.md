
# 🧠 Multi-Symbol Crypto Trading Bot (LLM-Gated)

A **professional, risk-aware crypto trading bot** designed to scan **multiple symbols in parallel**, evaluate **high-probability setups**, and filter trades using a **Large Language Model (LLM) as a conservative risk gatekeeper**.

This project focuses on **decision quality, capital protection, and discipline** rather than overtrading.

> ⚠️ This repository contains **NO UI**.  
> It is a **pure backend / engine-level trading system**.

---

## ✨ Key Features

- ✅ **Multi-symbol scanning** (ETH, BTC, SOL, XRP, easily extensible)
- ⚡ **Parallel execution** using `asyncio`
- 🧠 **LLM-based trade approval** (conservative, risk-first)
- 📊 **Multi-timeframe strategy** (1H / 15M / 5M)
- 🧾 **Separate JSON logs per symbol**
- 🛡️ **Strong trade filtering** (avoids chop & fake breakouts)
- 🔁 **Automatic scan every 60 seconds**
- 🚨 **DRY_RUN mode enabled by default**

---

## 🏗 Architecture Overview

```
bot.py               → Async scheduler & multi-symbol runner
config.py            → Symbols & global configuration
exchange.py          → Exchange connection (CCXT)
indicators.py        → EMA / RSI calculations
strategy.py          → Trend + pullback strategy logic
llm_gatekeeper.py    → LLM-based trade approval
risk.py              → Position sizing utilities
logger.py            → Per-symbol JSON logging
results/             → Output folder (auto-created)
```

Each symbol is:
- Evaluated independently
- Logged independently
- Never shares state with others

This makes the system **safe, scalable, and debuggable**.

---

## 📈 Strategy Logic (High Level)

1️⃣ **Trend Detection (1H)**  
- EMA50 > EMA200 → bullish bias

2️⃣ **Pullback Validation (15M)**  
- RSI between 40–60

3️⃣ **Entry Confirmation (5M)**  
- Momentum candle **OR**
- Strong bullish wick rejection

4️⃣ **Risk Management**
- Stop below recent swing low
- Fixed RR ≈ 2.2

5️⃣ **LLM Gatekeeper**
- Final approval or rejection
- Prefers `NO_TRADE` over marginal setups

> 💡 The bot is designed to **skip low-quality trades**, even in bullish trends.

---

## 🧠 Role of the LLM

The LLM does **NOT** generate signals.

It acts as a **risk gatekeeper**, evaluating:
- Momentum quality
- Confluence
- Risk–reward sanity
- Indicator alignment

If anything is unclear → **NO_TRADE**.

This dramatically reduces:
- Overtrading
- Emotional bias
- False breakouts

---

## ⚙️ Configuration

### `config.py`
```python
SYMBOLS = [
    "ETH/USDT",
    "BTC/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

DRY_RUN = True  # 🚨 Keep TRUE until fully tested
```

You can add or remove symbols freely.

---

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install ccxt pandas numpy python-dotenv openai
```

### 2️⃣ Set environment variables
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

KRAKEN_API_KEY=your_kraken_key
KRAKEN_API_SECRET=your_kraken_secret
```

### 3️⃣ Run the bot
```bash
python bot.py
```

The bot will:
- Scan all symbols every **60 seconds**
- Print decisions in terminal
- Append results to JSON files

---

## 📂 Output Format

A `results/` folder is created automatically:

```
results/
├─ ETH_USDT.json
├─ BTC_USDT.json
├─ SOL_USDT.json
└─ XRP_USDT.json
```

Each file contains newline-delimited JSON entries:

```json
{
  "symbol": "ETH/USDT",
  "strategy_signal": {...},
  "decision": {
    "decision": "NO_TRADE",
    "confidence": 60,
    "reason": "No momentum or wick rejection on 5M"
  },
  "timestamp": "2026-01-08T16:02:11Z"
}
```

Perfect for:
- Backtesting
- Dashboards
- Performance analysis

---

## 🛡 Safety Notes

- 🚨 **DRY_RUN is ON by default**
- No orders are placed unless explicitly enabled
- LLM failures automatically result in `NO_TRADE`
- Each symbol is isolated (no cascading risk)

---

## 🚀 Future Improvements

This architecture is intentionally extensible:

- 🔢 Symbol ranking & best-trade selection
- 💰 Portfolio-level risk allocation
- 📩 Telegram / Discord alerts
- 📊 Performance analytics per symbol
- 🔄 Multiple strategies in parallel

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**.

Trading cryptocurrencies involves significant risk.  
The author assumes **no responsibility** for financial losses.

Always test thoroughly before live deployment.

---

## 🏁 Final Note

This project prioritizes **discipline over frequency**.

> *Missing bad trades is a feature, not a bug.*

If you value:
- Capital preservation
- Clean architecture
- Professional-grade logic

You are using the right system.

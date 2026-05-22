# trendv7.py — How to use

This is the consolidated final file. It replaces `trendv6.py`,
`trendv6_backtest_no_llm.py`, and `trendv6_backtest_with_llm.py` with a single
script that supports live analysis, backtests, and LLM-vs-no-LLM comparisons
through one CLI.

## Install

```bash
pip install ccxt pandas ta python-dotenv
# optional, only if you want the LLM layer:
pip install openai openai-agents
# put OPENAI_API_KEY in a .env file next to trendv7.py
```

## Three commands

### 1. Live analysis

```bash
# single-timeframe
python trendv7.py analyze --symbol ETH/USDT --timeframe 1h

# multi-timeframe (the new v7 feature: HTF strategy + LTF entry trigger)
python trendv7.py analyze --symbol ETH/USDT --timeframe 1h --ltf 15m

# no LLM, save report to disk, print every strategy engine
python trendv7.py analyze --symbol ETH/USDT --timeframe 1h --no-llm --save --run-all

# with position sizing (1% risk on a $5000 paper account)
python trendv7.py analyze --symbol ETH/USDT --timeframe 4h \
  --balance 5000 --risk-pct 1.0 --max-daily-loss-r -3 --max-open-trades 1
```

### 2. Backtest

```bash
# no-LLM walk-forward
python trendv7.py backtest --symbol ETH/USDT --timeframe 1h --limit 1500

# with LLM strategy selector
python trendv7.py backtest --symbol ETH/USDT --timeframe 1h --limit 1500 --use-llm

# control fees + slippage + targeting TP2
python trendv7.py backtest --symbol BTC/USDT --timeframe 4h \
  --target-mode tp2 --entry-mode next_open \
  --fee-pct 0.05 --slippage-pct 0.02 --diagnostic
```

### 3. Compare (the headline feature)

Runs LLM and no-LLM on the same window and prints the delta — the same
comparison the project notes describe at the end of section 14.

```bash
python trendv7.py compare --symbol ETH/USDT --timeframe 1h --limit 1500
```

## What changed vs v6

- **Strategy tuning from the documented backtest:**
  - Pullback (the winner) → candle confirmation + volume + S/R guard + higher min quality
  - Momentum → demoted to confirmation-only by default (`ALLOW_MOMENTUM_AS_EXECUTION=False`)
  - Range-bound → hard-gated to Sideways markets
  - Breakout-and-retest → relaxed (was 0 trades in v6) — wider lookback, ATR tolerance, volume check
  - Reversal → stricter (quality ≥ 0.75, candle confirm required, not allowed vs Strong trend)
- **Multi-timeframe**: HTF strategy, LTF entry trigger via `--ltf`
- **Position sizing**: account balance, % risk, max daily loss, max open trades
- **Fees + slippage** modelled in backtest results
- **New metrics**: Sharpe-like, Sortino-like, Calmar-like, expectancy
- **One file** with `analyze` / `backtest` / `compare` subcommands
- **MINIMUM_RR** raised from 1.20 to 1.50

## Interactive mode

Run with no arguments to use the prompt-style UX from the old files:

```bash
python trendv7.py
```
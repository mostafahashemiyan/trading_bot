"""
MTF BACKTEST + LLM FILTER + VISUAL OUTPUT (2026)
------------------------------------------------
Generates:
- Equity curve chart
- Drawdown chart
- Trades CSV + Equity CSV

Timeframes:
HTF = 1h
MTF = 15m
LTF = 5m
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import config
from indicators import prepare_df
from strategy import trend_pullback_signal
from risk import position_size


# ====== USER PARAMETERS (your predefined values) ======
LLM_REQUIRED_CONFIDENCE = 70
LLM_REQUIRED_RR = 2.2
VOLUME_MULTIPLIER = 1.5  


# ====== BASIC SETTINGS ======
SYMBOL = "ETH/USDT:USDT"
HTF = "1h"
MTF = "15m"
LTF = "5m"

INITIAL_BALANCE = 1000
COMMISSION = config.COMMISSION
SLIPPAGE = config.SLIPPAGE_PCT

START_DATE = "2025-01-01T00:00:00Z"


# ======================================================
# FETCH DATA
# ======================================================
def fetch_tf(exchange, tf):
    print(f"Fetching {tf} ...")
    since = exchange.parse8601(START_DATE)
    out = []
    while True:
        batch = exchange.fetch_ohlcv(SYMBOL, tf, since=since, limit=1000)
        if not batch:
            break
        out.extend(batch)
        since = batch[-1][0] + 1
        time.sleep(0.25)
    return out


exchange = ccxt.kucoinfutures({"enableRateLimit": True})
exchange.load_markets()

htf_raw = fetch_tf(exchange, HTF)
mtf_raw = fetch_tf(exchange, MTF)
ltf_raw = fetch_tf(exchange, LTF)

df_h = prepare_df(htf_raw)
df_m = prepare_df(mtf_raw)
df_l = prepare_df(ltf_raw)

df_h.index = pd.to_datetime([x[0] for x in htf_raw], unit="ms")
df_m.index = pd.to_datetime([x[0] for x in mtf_raw], unit="ms")
df_l.index = pd.to_datetime([x[0] for x in ltf_raw], unit="ms")

print(f"Loaded {len(df_h)} HTF candles")
print(f"Loaded {len(df_m)} MTF candles")
print(f"Loaded {len(df_l)} LTF candles")


# ======================================================
# FUTURES EQUITY FUNCTION
# ======================================================
def fut_equity(balance, pos, price, entry):
    return balance if pos == 0 else balance + pos * (price - entry)


# ======================================================
# LLM-LIKE FILTER
# ======================================================
def llm_filter(signal, df_h, df_m, df_l):

    if not signal["setup"]:
        return False, "setup failed"

    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]
    rr = abs((tp - entry) / (entry - sl))

    if rr < LLM_REQUIRED_RR:
        return False, f"RR too low ({rr:.2f})"

    if signal["confidence"] < LLM_REQUIRED_CONFIDENCE:
        return False, f"low confidence ({signal['confidence']})"

    if df_h["adx"].iloc[-1] < config.ADX_THRESHOLD:
        return False, "weak ADX"

    vol_ma = df_l["volume"].rolling(20).mean().iloc[-1]
    if df_l["volume"].iloc[-1] < vol_ma * VOLUME_MULTIPLIER:
        return False, "low volume"

    ema_dist = abs(df_h["ema_fast"].iloc[-1] - df_h["ema_slow"].iloc[-1])
    if ema_dist < df_h["atr"].iloc[-1] * 0.2:
        return False, "ema too close (choppy)"

    return True, "approved"


# ======================================================
# BACKTEST LOOP
# ======================================================
balance = INITIAL_BALANCE
position = 0.0
entry_price = 0.0
sl = 0.0
tp = 0.0

equity_curve = []
trades = []


for i in range(50, len(df_h)):

    h_time = df_h.index[i]
    row_h = df_h.iloc[i]
    o, h, l, c = row_h.open, row_h.high, row_h.low, row_h.close

    df_m_slice = df_m[df_m.index <= h_time].tail(200)
    df_l_slice = df_l[df_l.index <= h_time].tail(300)

    # EXIT
    if position != 0:
        exit_price = None
        exit_reason = None

        if position > 0:
            if l <= sl:
                exit_price = min(o, sl) * (1 - SLIPPAGE)
                exit_reason = "SL"
            elif h >= tp:
                exit_price = max(o, tp) * (1 - SLIPPAGE)
                exit_reason = "TP"
        else:
            if h >= sl:
                exit_price = max(o, sl) * (1 + SLIPPAGE)
                exit_reason = "SL"
            elif l <= tp:
                exit_price = min(o, tp) * (1 + SLIPPAGE)
                exit_reason = "TP"

        if exit_price:
            pnl = position * (exit_price - entry_price)
            fee = abs(position * exit_price) * COMMISSION
            pnl -= fee
            balance += pnl

            trades.append({
                "time": h_time,
                "side": "LONG" if position > 0 else "SHORT",
                "entry": entry_price,
                "exit": exit_price,
                "pnl": pnl,
                "reason": exit_reason
            })

            position = 0

    # ENTRY
    if position == 0 and not df_m_slice.empty and not df_l_slice.empty:

        signal = trend_pullback_signal(
            df_h.iloc[:i+1],
            df_m_slice,
            df_l_slice
        )

        approved, reason = llm_filter(signal, df_h.iloc[:i+1], df_m_slice, df_l_slice)

        if approved:
            side = signal["side"]
            entry_raw = o
            entry_price = entry_raw * (1 + SLIPPAGE if side == "LONG" else 1 - SLIPPAGE)

            size = position_size(balance, entry_price, signal["sl"])
            if size * entry_price >= config.MIN_NOTIONAL_VALUE:

                position = size if side == "LONG" else -size
                sl = signal["sl"]
                tp = signal["tp"]

                fee = abs(position * entry_price) * COMMISSION
                balance -= fee

                trades.append({
                    "time": h_time,
                    "side": side,
                    "entry": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "size": abs(size),
                    "llm_reason": reason
                })

    equity_curve.append(
        fut_equity(balance, position, c, entry_price)
    )


# ======================================================
# FINAL REPORT
# ======================================================
eq = pd.Series(equity_curve, index=df_h.index[:len(equity_curve)])
peak = eq.cummax()
dd = (eq - peak) / peak * 100

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  BACKTEST REPORT (LLM FILTER)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Initial balance : {INITIAL_BALANCE}")
print(f"Final balance   : {balance:.2f}")
print(f"Return %        : {(balance/INITIAL_BALANCE - 1)*100:.2f}%")
print(f"Max Drawdown    : {dd.min():.2f}%")
print(f"Trades executed : {len(trades)}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

pd.DataFrame(trades).to_csv("mtf_llm_trades.csv", index=False)
eq.to_csv("mtf_llm_equity.csv")
print("Saved mtf_llm_trades.csv and mtf_llm_equity.csv")


# ======================================================
# VISUALIZATION
# ======================================================
plt.style.use("seaborn-v0_8")

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Equity chart
ax[0].plot(eq, label="Equity Curve", linewidth=1.8)
ax[0].set_title("Equity Curve")
ax[0].set_ylabel("Balance (USDT)")
ax[0].grid(True)

# Drawdown chart
ax[1].fill_between(dd.index, dd, 0, color="red", alpha=0.4)
ax[1].set_title("Drawdown (%)")
ax[1].set_ylabel("DD %")
ax[1].set_xlabel("Date")
ax[1].grid(True)

plt.tight_layout()
plt.show()
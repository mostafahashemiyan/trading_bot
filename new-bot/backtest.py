"""
Final conservative backtest version (March 2026).
Ready to be used as baseline before LLM gatekeeper.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
import sys

# Local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import config
except ImportError:
    config = None

# Fallbacks if config module not found (e.g. when run from another directory)
if config is None:
    class _Config:
        COOLDOWN_CANDLES = 12
        LEVERAGE = 2
        SAFETY_FACTOR = 0.15
        MIN_NOTIONAL_VALUE = 10
    config = _Config()

from indicators import prepare_df
from risk import position_size, sl_tp_from_atr
from strategy import trend_pullback_signal

# ────────────────────────────────────────────────
# Backtest settings
# ────────────────────────────────────────────────
SYMBOL = "DOGE/USDT:USDT"
TIMEFRAME = "1h"

BACKTEST_START = "2025-01-01T00:00:00Z"
BACKTEST_END = None
INITIAL_BALANCE = 1000.0
COMMISSION = 0.0006
SLIPPAGE_PCT = 0.0005

# ────────────────────────────────────────────────
# Exchange & symbol check
# ────────────────────────────────────────────────
exchange = ccxt.kucoinfutures({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

print("Loading markets...")
markets = exchange.load_markets()

if SYMBOL not in markets:
    eth_syms = [s for s in markets if 'ETH' in s.upper() and 'USDT' in s.upper()]
    SYMBOL = eth_syms[0] if eth_syms else 'ETHUSDTM'
    print(f"-> Using symbol: {SYMBOL}")

# ────────────────────────────────────────────────
# Fetch OHLCV
# ────────────────────────────────────────────────
print(f"Fetching {SYMBOL} {TIMEFRAME} from {BACKTEST_START} ...")
since = exchange.parse8601(BACKTEST_START)
ohlcv = []

while True:
    try:
        data = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
    except Exception as e:
        print(f"Fetch error: {e}")
        break

    if not data:
        break

    ohlcv.extend(data)
    since = data[-1][0] + 1
    time.sleep(0.4)

if not ohlcv:
    print("No data received.")
    sys.exit(1)

df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df.set_index('ts', inplace=True)

print(f"Loaded {len(df)} candles | {df.index[0].date()} -> {df.index[-1].date()}")

# ────────────────────────────────────────────────
# Indicators & signals
# ────────────────────────────────────────────────
raw_list = df.reset_index().values.tolist()
df_ind = prepare_df(raw_list)
df = df.join(df_ind.set_index('ts')[['ema_fast', 'ema_slow', 'rsi', 'atr', 'adx']])
df.dropna(inplace=True)

# Single-timeframe signal (same logic as strategy.trend_pullback_signal on one TF)
vol_ma = df['volume'].rolling(20).mean()
df['trend_bull'] = df['ema_fast'] > df['ema_slow']
df['trend_bear'] = df['ema_fast'] < df['ema_slow']
df['adx_ok'] = df['adx'] >= config.ADX_THRESHOLD
df['vol_ok'] = df['volume'] >= (vol_ma * config.MIN_VOLUME_MULTIPLIER)
body = (df['close'] - df['open']).abs()
rng = df['high'] - df['low']
df['bull_candle'] = (df['close'] > df['open']) & (body > 0.6 * rng.replace(0, np.nan))
df['bear_candle'] = (df['close'] < df['open']) & (body > 0.6 * rng.replace(0, np.nan))
df['long_signal'] = df['trend_bull'] & df['adx_ok'] & df['vol_ok'] & df['bull_candle']
df['short_signal'] = df['trend_bear'] & df['adx_ok'] & df['vol_ok'] & df['bear_candle']

print(f"After cleaning: {len(df)} rows")

# ────────────────────────────────────────────────
# Simulation loop
# ────────────────────────────────────────────────
trades = []
balance = INITIAL_BALANCE
position = 0.0
entry_price = 0.0
stop_loss = 0.0
take_profit = 0.0
bankruptcy_price = 0.0
last_trade_i = -999
equity_curve = [balance]

for i in range(1, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i-1]
    o, h, l, c = row['open'], row['high'], row['low'], row['close']

    # Cooldown
    if last_trade_i >= 0 and (i - last_trade_i) < config.COOLDOWN_CANDLES:
        equity_curve.append(balance + position * c)
        continue

    # Exit / Liquidation
    if position != 0:
        exit_price = None
        exit_type = None

        if position > 0:
            if l <= stop_loss:
                exit_price = max(stop_loss, o) * (1 - SLIPPAGE_PCT)
                exit_type = 'SL'
            elif h >= take_profit:
                exit_price = min(take_profit, o) * (1 - SLIPPAGE_PCT)
                exit_type = 'TP'
            elif l <= bankruptcy_price:
                exit_price = max(bankruptcy_price, l) * (1 - SLIPPAGE_PCT)
                exit_type = 'LIQ'
        else:
            if h >= stop_loss:
                exit_price = min(stop_loss, o) * (1 + SLIPPAGE_PCT)
                exit_type = 'SL'
            elif l <= take_profit:
                exit_price = max(take_profit, o) * (1 + SLIPPAGE_PCT)
                exit_type = 'TP'
            elif h >= bankruptcy_price:
                exit_price = min(bankruptcy_price, h) * (1 + SLIPPAGE_PCT)
                exit_type = 'LIQ'

        if exit_price is not None:
            pnl_raw = position * (exit_price - entry_price)
            comm = abs(position * exit_price) * COMMISSION
            pnl_net = pnl_raw - comm
            balance += pnl_net

            trades.append({
                'exit_time': row.name,
                'exit_type': exit_type,
                'pnl': pnl_net,
                'balance': balance
            })

            position = 0.0

            if balance <= 0:
                balance = 0.0
                print(f"Bankruptcy at {row.name}")
                break

    # Entry
    if position == 0 and balance > 0:
        side = None
        if row['long_signal'] and not prev['long_signal']:
            side = 'LONG'
        elif row['short_signal'] and not prev['short_signal']:
            side = 'SHORT'

        if side:
            entry_raw = o
            atr = row['atr']
            sl, tp = sl_tp_from_atr(side, entry_raw, atr)

            risk_dist = abs(entry_raw - sl)
            if risk_dist <= 0:
                equity_curve.append(balance + position * c)
                continue

            size_base = position_size(balance, entry_raw, sl)

            max_not = balance * config.LEVERAGE * config.SAFETY_FACTOR
            size_base = min(size_base, max_not / entry_raw)

            notional = size_base * entry_raw
            if notional < config.MIN_NOTIONAL_VALUE:
                equity_curve.append(balance + position * c)
                continue

            entry_price = entry_raw * (1 + SLIPPAGE_PCT if side == 'LONG' else 1 - SLIPPAGE_PCT)
            position = size_base if side == 'LONG' else -size_base
            stop_loss = sl
            take_profit = tp

            imm = 1.0 / config.LEVERAGE
            mmr = 0.005
            if side == 'LONG':
                bankruptcy_price = entry_price * (1 - imm - mmr)
            else:
                bankruptcy_price = entry_price * (1 + imm + mmr)

            comm = notional * COMMISSION
            balance -= comm

            trades.append({
                'entry_time': row.name,
                'side': side,
                'entry_price': entry_price,
                'size': abs(size_base),
                'sl': sl,
                'tp': tp,
                'balance_after_entry': balance
            })

            last_trade_i = i

    equity_curve.append(balance + position * c)

# ────────────────────────────────────────────────
# Report
# ────────────────────────────────────────────────
df_trades = pd.DataFrame(trades)

print("\n" + "=" * 70)
print(f"BACKTEST REPORT - {SYMBOL} {TIMEFRAME}")
print(f"Period          : {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Initial         : ${INITIAL_BALANCE:,.2f}")
print(f"Final           : ${balance:,.2f}")
print(f"Return          : {(balance/INITIAL_BALANCE-1)*100:.2f}%")
print(f"Closed trades   : {len(df_trades[df_trades.get('exit_type').notna()])}")
print(f"Longs           : {len(df_trades[df_trades['side']=='LONG'])}")
print(f"Shorts          : {len(df_trades[df_trades['side']=='SHORT'])}")

if 'pnl' in df_trades.columns and not df_trades.empty:
    closed = df_trades[df_trades['exit_type'].notna()]
    win_rate = (closed['pnl'] > 0).mean() * 100
    avg_pnl = closed['pnl'].mean()
    liq_count = len(closed[closed['exit_type'] == 'LIQ'])
    print(f"Win Rate        : {win_rate:.1f}%")
    print(f"Avg PnL/trade   : ${avg_pnl:,.2f}")
    print(f"Liquidations    : {liq_count}")

equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
peak = equity_series.cummax()
dd = (equity_series - peak) / peak * 100
print(f"Max Drawdown    : {dd.min():.2f}%")

print("\nLast 10 equity values:")
print(equity_series.tail(10))

df_trades.to_csv("backtest_trades.csv", index=False)
equity_series.to_csv("equity_curve.csv")
print("\nResults saved.")
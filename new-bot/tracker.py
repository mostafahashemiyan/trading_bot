# tracker.py
import json
import os
import pandas as pd
from datetime import datetime

TRADE_FILE = "trades_log.json"


def load_trades():
    if not os.path.exists(TRADE_FILE):
        return []

    try:
        with open(TRADE_FILE, "r") as f:
            return json.load(f)

    except:
        return []


def save_trade(trade_data):
    trades = load_trades()
    trades.append(trade_data)
    with open(TRADE_FILE, "w") as f:
        json.dump(trades, f, indent=4)
    print(f"Trade saved to {TRADE_FILE}")


def generate_report():
    trades = load_trades()
    if not trades:
        print("⚠ No trades found in log.")
        return

    df = pd.DataFrame(trades)

    # Simple simulation of results (since we don't have a live feedback loop in this script yet)
    # In a real scenario, you would update the 'outcome' field based on fetch_my_trades()

    total_trades = len(df)
    print(f"\n--- PERFORMANCE REPORT ({total_trades} Trades) ---")
    print(df[["timestamp", "symbol", "side", "entry", "size"]].tail())
    print("------------------------------------------------------")

    # If you implement PnL tracking later, you can calc win rate here:
    # wins = df[df['pnl'] > 0]
    # win_rate = len(wins) / total_trades * 100
    # print(f"Win Rate: {win_rate}%")

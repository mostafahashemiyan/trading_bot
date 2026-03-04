"""
REPORT MODULE (Professional 2026)
---------------------------------
Generates performance summaries using trade_history.json
"""

import json
import os
import pandas as pd

DIR = "trade_logs"
HISTORY_FILE = os.path.join(DIR, "trades_history.json")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        print("⚠ No history file found.")
        return pd.DataFrame()
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            return pd.DataFrame(data)
    except Exception as e:
        print(f"Failed to load history: {e}")
        return pd.DataFrame()


def generate_report():
    df = load_history()
    if df.empty:
        print("⚠ No trades found.")
        return

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊  TRADE PERFORMANCE REPORT")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    total = len(df)
    print(f"Total trades recorded : {total}")

    if "side" in df.columns:
        print(f"Long trades           : {len(df[df.side == 'LONG'])}")
        print(f"Short trades          : {len(df[df.side == 'SHORT'])}")

    if "outcome" in df.columns:
        sl_hits = len(df[df.outcome == "SL"])
        tp_hits = len(df[df.outcome == "TP"])
        print(f"SL hits               : {sl_hits}")
        print(f"TP hits               : {tp_hits}")

    if "pnl" in df.columns:
        wins = df[df.pnl > 0]
        win_rate = len(wins) / len(df) * 100
        avg_pnl = df.pnl.mean()

        print(f"Win rate              : {win_rate:.2f}%")
        print(f"Avg PnL per trade     : {avg_pnl:.2f}")

    print("\nLast 5 trades:")
    print(df.tail(5).to_string(index=False))

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
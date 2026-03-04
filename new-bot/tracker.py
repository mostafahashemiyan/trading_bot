"""
TRADE TRACKER (Professional 2026)
---------------------------------
Tracks:
- Open trades
- Closed trades
- Trade history persistence

Used by bot.py to store full records of trading activity.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


DIR = "trade_logs"
os.makedirs(DIR, exist_ok=True)

OPEN_FILE = os.path.join(DIR, "open_trades.json")
HISTORY_FILE = os.path.join(DIR, "trades_history.json")


# ───────────────────────────────────────────────
# JSON Helpers
# ───────────────────────────────────────────────
def _read(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _write(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ───────────────────────────────────────────────
# TRADE TRACKER CLASS
# ───────────────────────────────────────────────
class TradeTracker:

    def __init__(self):
        self.open_trades: List[Dict[str, Any]] = _read(OPEN_FILE, [])

    def _save_open(self):
        _write(OPEN_FILE, self.open_trades)

    def _append_history(self, trade: Dict[str, Any]):
        hist = _read(HISTORY_FILE, [])
        hist.append(trade)
        _write(HISTORY_FILE, hist)

    # ───────────────────────────────────────────
    # Opening Trades
    # ───────────────────────────────────────────
    def open_trade(
        self,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        size: float,
        entry_ts: float,
        tp_id: Optional[str] = None,
        sl_id: Optional[str] = None,
    ):
        ts = datetime.utcfromtimestamp(entry_ts).isoformat() + "Z"

        trade = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": size,
            "tp_id": tp_id,
            "sl_id": sl_id,
            "status": "OPEN",
        }

        self.open_trades.append(trade)
        self._save_open()
        self._append_history(trade)

    # ───────────────────────────────────────────
    # Closing Trades
    # ───────────────────────────────────────────
    def close_trade(self, symbol: str, outcome: str):
        new_list = []

        for t in self.open_trades:
            if t["symbol"] == symbol and t["status"] == "OPEN":
                t["status"] = "CLOSED"
                t["outcome"] = outcome
                self._append_history(t)
            else:
                new_list.append(t)

        self.open_trades = new_list
        self._save_open()

    # ───────────────────────────────────────────
    # Getters
    # ───────────────────────────────────────────
    def get_open(self) -> List[Dict[str, Any]]:
        self.open_trades = _read(OPEN_FILE, [])
        return self.open_trades

    def get_history(self) -> List[Dict[str, Any]]:
        return _read(HISTORY_FILE, [])
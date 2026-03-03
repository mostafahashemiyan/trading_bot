import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

TRADE_FILE = "trades_log.json"
OPEN_TRADES_FILE = "open_trades.json"


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_trades() -> List[Dict[str, Any]]:
    return _read_json(TRADE_FILE, [])


def save_trade(trade_data: Dict[str, Any]) -> None:
    trades = load_trades()
    trades.append(trade_data)
    _write_json(TRADE_FILE, trades)
    print(f"Trade saved to {TRADE_FILE}")


def generate_report() -> None:
    trades = load_trades()
    if not trades:
        print("⚠ No trades found in log.")
        return

    df = pd.DataFrame(trades)
    total_trades = len(df)
    print(f"\n--- PERFORMANCE REPORT ({total_trades} Trades) ---")
    cols = [c for c in ["timestamp", "symbol", "side", "entry", "stop", "tp", "size", "status", "outcome"] if c in df.columns]
    print(df[cols].tail())
    print("------------------------------------------------------")


class TradeTracker:

    def __init__(self):
        self._open: List[Dict[str, Any]] = _read_json(OPEN_TRADES_FILE, [])

    def _persist_open(self):
        _write_json(OPEN_TRADES_FILE, self._open)

    def open_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        entry_time: float,
        sl_id: Optional[str] = None,
        tp_id: Optional[str] = None,
    ):
        ts = datetime.utcfromtimestamp(entry_time).isoformat() + "Z"
        trade = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "entry": entry_price,
            "stop": stop_loss,
            "tp": take_profit,
            "size": size,
            "sl_id": sl_id,
            "tp_id": tp_id,
            "status": "OPEN",
        }
        self._open.append(trade)
        self._persist_open()
        save_trade({**trade})

    def get_open_trades(self) -> List[Dict[str, Any]]:
        self._open = _read_json(OPEN_TRADES_FILE, [])
        return list(self._open)

    def mark_closed(self, symbol: str, sl_or_tp: str):
        updated = []
        for t in self._open:
            if t.get("symbol") == symbol and t.get("status") == "OPEN":
                t["status"] = "CLOSED"
                t["outcome"] = sl_or_tp
                save_trade(t)  # Append to history
            updated.append(t)
        self._open = [t for t in updated if t["status"] != "CLOSED"]
        self._persist_open()
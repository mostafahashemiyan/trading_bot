"""
TRADE TRACKER (Professional 2026)
---------------------------------
Tracks:
- Open trades
- Closed trades
- Trade history persistence
- Per-run decision reports

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
REPORT_FILE = os.path.join(DIR, "decision_reports.json")


# ───────────────────────────────────────────────
# JSON Helpers
# ───────────────────────────────────────────────
def _read(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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

    def _append_report(self, report: Dict[str, Any]):
        reports = _read(REPORT_FILE, [])
        reports.append(report)
        _write(REPORT_FILE, reports)

    # ───────────────────────────────────────────
    # Decision Reports
    # ───────────────────────────────────────────
    def save_decision_report(
        self,
        symbol: str,
        strategy_signal: Dict[str, Any],
        decision: Dict[str, Any],
        trade_result: Optional[Dict[str, Any]] = None,
    ):
        entry = strategy_signal.get("entry")
        sl = strategy_signal.get("sl")
        tp = strategy_signal.get("tp")
        side = strategy_signal.get("side")

        trend = None
        if side == "LONG":
            trend = "bullish"
        elif side == "SHORT":
            trend = "bearish"

        rr = None
        try:
            if entry is not None and sl is not None and tp is not None and entry != sl:
                rr = abs((tp - entry) / (entry - sl))
        except Exception:
            rr = None

        now_iso = datetime.utcnow().isoformat() + "Z"

        report = {
            "symbol": symbol,
            "strategy_signal": {
                "trend": trend,
                "setup": strategy_signal.get("setup"),
                "side": side,
                "entry": entry,
                "stop": sl,
                "tp": tp,
                "rr": rr,
                "confidence": strategy_signal.get("confidence", 0),
                "reasons": strategy_signal.get("reasons", []),
            },
            "decision": {
                "decision": decision.get("decision"),
                "side": decision.get("side"),
                "confidence": decision.get("confidence", 0),
                "reason": decision.get("reason"),
            },
            "trade_result": trade_result,
            "timestamp": now_iso,
            "time": now_iso,
        }

        self._append_report(report)

    def get_reports(self) -> List[Dict[str, Any]]:
        return _read(REPORT_FILE, [])

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
    def close_trade(
        self,
        symbol: str,
        outcome: str,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
    ):
        new_list = []
        closed_any = False
        now_iso = datetime.utcnow().isoformat() + "Z"

        for t in self.open_trades:
            if t["symbol"] == symbol and t["status"] == "OPEN":
                t["status"] = "CLOSED"
                t["outcome"] = outcome
                t["closed_at"] = now_iso

                if exit_price is not None:
                    t["exit_price"] = exit_price
                if pnl is not None:
                    t["pnl"] = pnl

                self._append_history(t)
                closed_any = True
            else:
                new_list.append(t)

        self.open_trades = new_list
        self._save_open()
        return closed_any

    # ───────────────────────────────────────────
    # Getters
    # ───────────────────────────────────────────
    def get_open(self) -> List[Dict[str, Any]]:
        self.open_trades = _read(OPEN_FILE, [])
        return self.open_trades

    def get_history(self) -> List[Dict[str, Any]]:
        return _read(HISTORY_FILE, [])
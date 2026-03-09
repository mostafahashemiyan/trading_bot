"""
ADVANCED LOGGER (2026)
Structured JSON logs for every step of the bot.
Each log entry is a single JSON object in one line.
"""

import json
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def log_event(symbol: str, strategy_signal: dict, decision: dict):
    """
    Creates a full structured log entry like:

    {
      "symbol": "ETH/USDT",
      "strategy_signal": {...},
      "decision": {...},
      "timestamp": "...",
      "time": "..."
    }
    """

    entry = {
        "symbol": symbol,
        "strategy_signal": strategy_signal,
        "decision": decision,
        "timestamp": datetime.utcnow().isoformat(),
        "time": datetime.utcnow().isoformat()
    }

    filename = f"{symbol.replace('/', '_')}.json"
    path = os.path.join(LOG_DIR, filename)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def log_system(message: str):
    entry = {
        "event": "system",
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }

    filename = "system_log.json"
    path = os.path.join(LOG_DIR, filename)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log(event_type, data, symbol=None):
    entry = {
        "event": event_type,
        "symbol": symbol,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }

    if symbol:
        filename = f"{symbol.replace('/', '_')}.json"
    else:
        filename = "SYSTEM.json"

    path = os.path.join(LOG_DIR, filename)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry
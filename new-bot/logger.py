"""
LOGGER MODULE (Professional 2026)
---------------------------------
Creates structured JSON logs for:
- Signals
- LLM decisions
- Trades
- Errors
"""

import json
import os
from datetime import datetime


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def log(event_type: str, data: dict, symbol: str = "GLOBAL"):
    """
    event_type examples:
      - "signal"
      - "llm_decision"
      - "trade_open"
      - "trade_close"
      - "error"
      - "system"

    Writes one JSON object per line.
    """

    data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    data["event"] = event_type
    data["symbol"] = symbol

    filename = f"{symbol.replace('/', '_')}.json"
    file_path = os.path.join(LOG_DIR, filename)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def log_system(message: str):
    log("system", {"msg": message}, symbol="SYSTEM")
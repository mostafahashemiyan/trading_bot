# logger.py

import json
import os
from datetime import datetime

LOG_DIR = "results"
os.makedirs(LOG_DIR, exist_ok=True)

def log(symbol: str, data: dict):
    data["time"] = datetime.utcnow().isoformat()
    filename = f"{symbol.replace('/', '_')}.json"
    path = os.path.join(LOG_DIR, filename)

    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")

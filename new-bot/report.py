import json
import os
from datetime import datetime

LOG_DIR = "reports"
os.makedirs(LOG_DIR, exist_ok=True)

def log_detailed_report(symbol: str, report_data: dict):
    report_data["time"] = datetime.utcnow().isoformat()
    filename = f"{symbol.replace('/', '_')}_logs.json"
    path = os.path.join(LOG_DIR, filename)
    
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(report_data, ensure_ascii=False) + "\n")
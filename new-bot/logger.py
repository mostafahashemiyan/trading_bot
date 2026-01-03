import json
from datetime import datetime

def log(data):
    data["time"] = datetime.utcnow().isoformat()
    with open("bot_log.json", "a") as f:
        f.write(json.dumps(data) + "\n")

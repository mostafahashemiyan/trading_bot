# logger.py
import json
import os
from datetime import datetime

LOG_DIR = "reports" # تغییر نام پوشه برای نظم بیشتر
os.makedirs(LOG_DIR, exist_ok=True)

def log_report(symbol: str, data: dict):
    """ذخیره گزارش کامل هر اجرای برنامه برای یک نماد خاص"""
    data["timestamp"] = datetime.utcnow().isoformat()
    # نام فایل بر اساس تاریخ روز ساخته می‌شود تا گزارش‌ها خیلی حجیم نشوند
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"{symbol.replace('/', '_')}_{date_str}.json"
    path = os.path.join(LOG_DIR, filename)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
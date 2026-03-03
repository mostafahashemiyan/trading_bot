# llm_gatekeeper.py
"""
LLM Gatekeeper – تصمیم نهایی ورود به معامله
این فایل سیگنال خام استراتژی رو می‌گیره و با کمک مدل LLM تصمیم می‌گیره که وارد معامله بشیم یا نه.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    # اگر از مدل‌های دیگر استفاده می‌کنی، base_url رو تنظیم کن
    # base_url="https://api.deepseek.com/v1"   # مثلاً برای deepseek
)

DEFAULT_MODEL = "gpt-4o-mini"   # یا "o1-mini" یا "deepseek-chat" یا هر مدل دیگه‌ای


def llm_decide(signal: dict, symbol: str, current_price: float = None) -> dict:
    """
    ورودی: دیکشنری سیگنال تولید شده توسط strategy
    خروجی: دیکشنری تصمیم نهایی LLM
    """

    # اطلاعات مهم برای LLM
    context = {
        "symbol": symbol,
        "side": signal.get("side"),
        "entry": signal.get("entry"),
        "stop_loss": signal.get("sl"),
        "take_profit": signal.get("tp"),
        "risk_reward": round((signal.get("tp", 0) - signal.get("entry", 0)) / (signal.get("entry", 0) - signal.get("sl", 0)), 2) if signal.get("side") == "LONG" else round((signal.get("entry", 0) - signal.get("tp", 0)) / (signal.get("sl", 0) - signal.get("entry", 0)), 2),
        "adx": signal.get("adx", "نامشخص"),
        "rsi": signal.get("rsi", "نامشخص"),
        "volume_status": signal.get("volume_status", "نامشخص"),
        "reasons": signal.get("reasons", []),
        "strategy_confidence": signal.get("confidence", 50),
        "current_price": current_price or signal.get("entry"),
    }

    # Prompt سیستم – خیلی مهم است
    system_prompt = f"""You are a very strict and conservative trading gatekeeper.
Your job is to PROTECT the capital and ONLY approve HIGH-CONFIDENCE setups.

Rules you MUST follow:
- Approve ONLY if RR ≥ 2.3
- Approve ONLY if ADX ≥ 25 (strong trend)
- Reject if setup looks like choppy/range market
- Reject if volume is low or momentum weak
- Reject if risk/reward is poor or stop is too wide
- You can suggest tighter SL/TP if needed

Respond ONLY with valid JSON, no extra text:
{{
  "decision": "APPROVE" or "REJECT",
  "reason": "short explanation in 1 sentence",
  "confidence": 0-100,
  "suggested_sl": number or null,
  "suggested_tp": number or null
}}
"""

    user_prompt = f"""Current signal for {symbol}:

{json.dumps(context, indent=2, ensure_ascii=False)}

Analyze carefully and decide."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.15,          # کم برای تصمیم‌گیری محافظه‌کارانه
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # اطمینان از وجود کلیدهای اصلی
        result.setdefault("decision", "REJECT")
        result.setdefault("reason", "No response from LLM")
        result.setdefault("confidence", 0)

        return result

    except Exception as e:
        print(f"LLM Error: {e}")
        return {
            "decision": "REJECT",
            "reason": f"LLM call failed: {str(e)}",
            "confidence": 0
        }
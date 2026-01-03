import json
import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

def llm_decide(features: dict) -> dict:
    prompt = f"""
You are a professional crypto trader.

You must reply with ONLY valid JSON.
No markdown.
No explanation outside JSON.

Rules:
- Approve TRADE only if conditions are high probability
- Conservative bias
- Risk-reward must be acceptable

Input (JSON):
{json.dumps(features, indent=2)}

Return EXACTLY this schema:
{{
  "decision": "TRADE" | "NO_TRADE",
  "confidence": 0-100,
  "reason": "string"
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()

        # 🧹 Remove markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        # 🔎 Extract JSON safely
        match = JSON_PATTERN.search(raw)
        if not match:
            raise ValueError(f"No JSON found in response: {raw}")

        data = json.loads(match.group())

        # ✅ Final validation
        if data.get("decision") not in ["TRADE", "NO_TRADE"]:
            raise ValueError("Invalid decision field")

        return data

    except Exception as e:
        return {
            "decision": "NO_TRADE",
            "confidence": 0,
            "reason": f"LLM parsing failure: {str(e)}"
        }

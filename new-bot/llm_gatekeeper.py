import json
import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

def llm_decide(features: dict) -> dict:

    prompt = f"""
    You are a professional crypto trading risk gatekeeper.

    Your task is NOT to find trades.
    Your task is to EVALUATE the provided setup and decide whether it should be traded.

    You must be conservative.
    If anything is unclear, risky, or misaligned → choose NO_TRADE.

    You are given REAL indicator values and a PREDEFINED strategy signal.
    Do NOT rely on general crypto knowledge alone.

    Rules:
    - Only approve trades with clear confluence
    - Risk-reward must be acceptable (RR ≥ 2)
    - Trend alignment must be respected
    - Avoid overconfidence
    - Prefer NO_TRADE over marginal trades

    Input data (JSON):
    {json.dumps(features, indent=2)}

    Return ONLY valid JSON in this EXACT schema:

    {{
    "decision": "TRADE" | "NO_TRADE",
    "side": "LONG" | "SHORT" | null,
    "confidence": 0-100,
    "reason": "short, precise explanation"
    }}

    Constraints:
    - If decision is NO_TRADE → side MUST be null
    - Confidence above 70 only for very strong setups
    - Do not include markdown or extra text
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

import json
import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

def llm_decide(features: dict) -> dict:
    rubric = """
    RUBRIC:
    1. Trend (40%): Alignment with 1H EMA200.
    2. Pullback (30%): 15M RSI health (Target 40-60).
    3. Confirmation (30%): 5M Momentum/Wicks.
    SCORE < 60 = NO_TRADE.
    """

    prompt = f"{rubric}\nAnalyze this data and return JSON:\n{json.dumps(features)}"

    try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250,
            )
            
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            
            match = JSON_PATTERN.search(raw)
            if match:
                # Successfully found JSON
                return json.loads(match.group())
            else:
                # Could not find JSON in the LLM text
                print(f"⚠️ LLM did not return JSON. Raw output: {raw}")
                return {
                    "decision": "NO_TRADE",
                    "side": None,
                    "confidence": 0,
                    "reason": "LLM failed to provide a valid JSON decision"
                }

    except Exception as e:
            print(f"❌ LLM Gateway Error: {str(e)}")
            return {
                "decision": "NO_TRADE",
                "side": None,
                "confidence": 0,
                "reason": f"System error: {str(e)}"
            }
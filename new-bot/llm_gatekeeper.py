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

            # 1. Strip potential Markdown code blocks
            raw = raw.replace("```json", "").replace("```", "").strip()

            # 2. Use the Regex pattern to find the JSON block inside the text
            match = JSON_PATTERN.search(raw)
            if match:
                return json.loads(match.group())
            else:
                print(f"⚠️ LLM sent non-JSON text for {features['symbol']}")
                return {"decision": "NO_TRADE", "reason": "Non-JSON response"}
                
    except Exception as e:
        return {"decision": "NO_TRADE", "reason": f"LLM Error: {str(e)}"}
"""
LLM GATEKEEPER (Professional 2026)
----------------------------------
A strict risk-protection layer.
The strategy passes raw signals here, and the LLM decides:
APPROVE or REJECT.

Outputs ALWAYS valid JSON.
"""

import os
import json
from pathlib import Path
from openai import OpenAI
import config


def _load_env_from_file() -> None:
    """
    Minimal .env loader so we don't depend on python-dotenv.
    Looks for a .env file in the project root (parent of this folder)
    and injects values into os.environ if they are not already set.
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if not env_path.is_file():
            return

        with env_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Strip optional surrounding quotes
                if (value.startswith("'") and value.endswith("'")) or (
                    value.startswith('"') and value.endswith('"')
                ):
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Fail silently – downstream code will surface missing vars clearly
        pass


_load_env_from_file()

# Load API key from environment (after .env injection)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_decide(signal: dict, symbol: str) -> dict:
    """
    Input: raw signal from strategy
    Output:
        {
          "decision": "APPROVE" / "REJECT",
          "reason": "...",
          "confidence": int,
          "suggested_sl": float or null,
          "suggested_tp": float or null
        }
    """

    if not config.ENABLE_LLM:
        return {
            "decision": "APPROVE",
            "reason": "LLM disabled by config",
            "confidence": 100,
            "suggested_sl": None,
            "suggested_tp": None
        }

    data = {
        "symbol": symbol,
        "side": signal.get("side"),
        "entry": signal.get("entry"),
        "stop_loss": signal.get("sl"),
        "take_profit": signal.get("tp"),
        "risk_reward": abs((signal.get("tp") - signal.get("entry")) /
                           (signal.get("entry") - signal.get("sl"))) if signal.get("sl") else None,
        "confidence_raw": signal.get("confidence", 0),
        "reasons": signal.get("reasons", [])
    }

    # System rules
    system_prompt = f"""
You are a hyper-conservative trading gatekeeper.

RULES YOU MUST ENFORCE:
- Approve ONLY if risk/reward >= {config.LLM_MIN_RR}
- Approve ONLY if ADX >= {config.LLM_MIN_ADX}
- Reject if volume is weak
- Reject if candle momentum is weak
- Reject if trend structure unclear
- Reject if reasons contain anything suggesting uncertainty
- Approve only high-confidence opportunities
- You may propose tighter SL/TP if risk management can improve

Respond ONLY in this JSON format:
{{
  "decision": "APPROVE" or "REJECT",
  "reason": "one short sentence",
  "confidence": 0–100,
  "suggested_sl": null or number,
  "suggested_tp": null or number
}}
"""

    user_prompt = f"Evaluate this trading signal:\n{json.dumps(data, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        result = json.loads(response.choices[0].message.content)
        # Ensure fields exist
        result.setdefault("decision", "REJECT")
        result.setdefault("reason", "Model returned incomplete response")
        result.setdefault("confidence", 0)
        result.setdefault("suggested_sl", None)
        result.setdefault("suggested_tp", None)

        return result

    except Exception as e:
        return {
            "decision": "REJECT",
            "reason": f"LLM failed: {str(e)}",
            "confidence": 0,
            "suggested_sl": None,
            "suggested_tp": None
        }
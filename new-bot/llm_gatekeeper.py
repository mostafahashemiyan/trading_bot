import os
import json
import time
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

                if (value.startswith("'") and value.endswith("'")) or (
                    value.startswith('"') and value.endswith('"')
                ):
                    value = value[1:-1]

                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


_load_env_from_file()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _failsafe_response(reason: str, elapsed_ms: int = 0) -> dict:
    return {
        "decision": config.LLM_FAILSAFE_DECISION,
        "reason": reason,
        "confidence": 0,
        "suggested_sl": None,
        "suggested_tp": None,
        "latency_ms": elapsed_ms
    }


def llm_decide(signal: dict, symbol: str) -> dict:
    """
    Input: raw signal from strategy
    Output:
        {
          "decision": "APPROVE" / "REJECT",
          "reason": "...",
          "confidence": int,
          "suggested_sl": float or null,
          "suggested_tp": float or null,
          "latency_ms": int
        }
    """

    if not config.ENABLE_LLM:
        return {
            "decision": "APPROVE",
            "reason": "LLM disabled by config",
            "confidence": 100,
            "suggested_sl": None,
            "suggested_tp": None,
            "latency_ms": 0
        }

    ctx = signal.get("market_context", {})

    risk_reward = None
    try:
        entry = signal.get("entry")
        sl = signal.get("sl")
        tp = signal.get("tp")
        if entry is not None and sl is not None and tp is not None and entry != sl:
            risk_reward = abs((tp - entry) / (entry - sl))
    except Exception:
        risk_reward = None

    data = {
        "symbol": symbol,
        "side": signal.get("side"),
        "entry": signal.get("entry"),
        "stop_loss": signal.get("sl"),
        "take_profit": signal.get("tp"),
        "risk_reward": risk_reward,
        "confidence_raw": signal.get("confidence", 0),
        "reasons": signal.get("reasons", []),
        "market_context": {
            "adx": ctx.get("adx"),
            "ema_fast": ctx.get("ema_fast"),
            "ema_slow": ctx.get("ema_slow"),
            "ema_distance": ctx.get("ema_distance"),
            "ema_fast_slope": ctx.get("ema_fast_slope"),
            "htf_close": ctx.get("htf_close"),
            "atr": ctx.get("atr"),
            "volume_current": ctx.get("volume_current"),
            "volume_ma20": ctx.get("volume_ma20"),
            "volume_ratio": ctx.get("volume_ratio"),
            "candle_body_ratio": ctx.get("candle_body_ratio"),
            "trend_quality_pass": ctx.get("trend_quality_pass")
        }
    }

    system_prompt = f"""
You are a hyper-conservative trading gatekeeper.

RULES YOU MUST ENFORCE:
- Approve ONLY if risk/reward >= {config.LLM_MIN_RR}
- Approve ONLY if ADX >= {config.LLM_MIN_ADX}
- Reject if volume is weak
- Reject if candle momentum is weak
- Reject if trend structure unclear
- Reject if reasons contain anything suggesting uncertainty
- Evaluate ADX strength using market_context.adx
- Evaluate trend using ema_fast vs ema_slow, ema_distance, ema_fast_slope, htf_close, and trend_quality_pass
- Evaluate momentum using candle_body_ratio
- Evaluate volume using volume_ratio
- Approve only high-confidence opportunities
- You may propose tighter SL/TP if risk management can improve

Respond ONLY in this JSON format:
{{
  "decision": "APPROVE" or "REJECT",
  "reason": "one short sentence",
  "confidence": 0,
  "suggested_sl": null,
  "suggested_tp": null
}}
"""

    user_prompt = f"Evaluate this trading signal:\n{json.dumps(data, indent=2)}"

    started = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            timeout=config.LLM_TIMEOUT_SECONDS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        result = json.loads(response.choices[0].message.content)
        result.setdefault("decision", "REJECT")
        result.setdefault("reason", "Model returned incomplete response")
        result.setdefault("confidence", 0)
        result.setdefault("suggested_sl", None)
        result.setdefault("suggested_tp", None)
        result["latency_ms"] = elapsed_ms

        if result["decision"] not in {"APPROVE", "REJECT"}:
            return _failsafe_response(
                f"LLM returned invalid decision: {result['decision']}",
                elapsed_ms
            )

        return result

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return _failsafe_response(f"LLM failed or timed out: {str(e)}", elapsed_ms)
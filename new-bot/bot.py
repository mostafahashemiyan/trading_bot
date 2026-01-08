# bot.py

import asyncio
from exchange import fetch_ohlcv
from indicators import prepare_df
from strategy import trend_pullback_signal
from llm_gatekeeper import llm_decide
from logger import log
from config import SYMBOLS, DRY_RUN
from datetime import datetime


async def analyze_symbol(symbol: str) -> dict:
    # --------------------------------------------------
    # Fetch market data
    # --------------------------------------------------
    ohlcv_1h = fetch_ohlcv(symbol, "1h")
    ohlcv_15m = fetch_ohlcv(symbol, "15m")
    ohlcv_5m = fetch_ohlcv(symbol, "5m")

    df_1h = prepare_df(ohlcv_1h)
    df_15m = prepare_df(ohlcv_15m)
    df_5m = prepare_df(ohlcv_5m)

    # --------------------------------------------------
    # Strategy
    # --------------------------------------------------
    signal = trend_pullback_signal(df_1h, df_15m, df_5m)

    # --------------------------------------------------
    # No setup → log and return
    # --------------------------------------------------
    if not signal["setup"]:
        result = {
            "symbol": symbol,
            "strategy_signal": signal,
            "decision": {
                "decision": "NO_TRADE",
                "side": None,
                "confidence": 0,
                "reason": "Strategy conditions not met"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        log(symbol, result)
        return result

    # --------------------------------------------------
    # LLM features
    # --------------------------------------------------
    features = {
        "symbol": symbol,
        "setup_detected": signal["setup"],
        "strategy_reasons": signal["reasons"],
        "trend": signal["trend"],
        "entry": signal["entry"],
        "stop": signal["stop"],
        "tp": signal["tp"],
        "rr": signal["rr"],
        "timeframes": {
            "1h": {
                "ema50": round(df_1h["ema50"].iloc[-1], 2),
                "ema200": round(df_1h["ema200"].iloc[-1], 2),
            },
            "15m": {
                "rsi": round(df_15m["rsi"].iloc[-1], 2)
            },
            "5m": {
                "close": round(df_5m["close"].iloc[-1], 2),
                "ema20": round(df_5m["ema20"].iloc[-1], 2)
            }
        }
    }

    # --------------------------------------------------
    # LLM decision
    # --------------------------------------------------
    decision = llm_decide(features)

    result = {
        "symbol": symbol,
        "strategy_signal": signal,
        "decision": decision,
        "timestamp": datetime.utcnow().isoformat()
    }

    log(symbol, result)

    # --------------------------------------------------
    # Execution (still gated)
    # --------------------------------------------------
    if decision["decision"] == "TRADE" and not DRY_RUN:
        print(f"🚀 EXECUTE {decision['side']} on {symbol}")

    return result


async def run_loop():
    print("🟢 Multi-symbol bot started (60s interval)")

    while True:
        start = datetime.utcnow().isoformat()
        print(f"\n⏱ Scan started at {start}")

        tasks = [analyze_symbol(symbol) for symbol in SYMBOLS]
        results = await asyncio.gather(*tasks)

        for r in results:
            print(f"{r['symbol']} → {r['decision']['decision']}")

        # --------------------------------------------------
        # Wait 60 seconds before next scan
        # --------------------------------------------------
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(run_loop())

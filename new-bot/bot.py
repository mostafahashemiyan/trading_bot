from exchange import fetch_ohlcv
from indicators import prepare_df
from strategy import trend_pullback_signal
from llm_gatekeeper import llm_decide
from logger import log
from config import SYMBOL, DRY_RUN
import time


def run(symbol=SYMBOL):
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
        log({
            "strategy_setup": False,
            "strategy_reasons": signal["reasons"],
            "entry": signal["entry"],
            "stop": signal["stop"],
            "rr": signal["rr"]
        })

        return {
            "symbol": SYMBOL,
            "signal": signal,
            "decision": {
                "decision": "NO_TRADE",
                "side": None,
                "confidence": 0,
                "reason": "Strategy conditions not met"
            }
        }

    # --------------------------------------------------
    # LLM features
    # --------------------------------------------------
    features = {
        "symbol": SYMBOL,
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

    log({
        "strategy_signal": signal,
        "llm_decision": decision
    })

    # --------------------------------------------------
    # Execution (still gated)
    # --------------------------------------------------
    if decision["decision"] == "TRADE":
        if decision["side"] == "LONG":
            order_side = "buy"
        elif decision["side"] == "SHORT":
            order_side = "sell"
        else:
            order_side = None

        if order_side and not DRY_RUN:
            print(f"🚀 EXECUTE {order_side.upper()} TRADE")

    # --------------------------------------------------
    # Return result (important for UI / API)
    # --------------------------------------------------
    return {
        "symbol": symbol,
        "signal": signal,
        "decision": decision
    }


def run_once(symbol):
    return run(symbol)


if __name__ == "__main__":
    while True:
        run()
        time.sleep(60)

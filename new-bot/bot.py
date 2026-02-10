# bot.py

import asyncio
from exchange import fetch_ohlcv, execute_trade
from indicators import prepare_df
from strategy import trend_pullback_signal
from llm_gatekeeper import llm_decide
from tracker import save_trade, generate_report
from logger import log
from config import SYMBOLS, DRY_RUN, RISK_PER_TRADE
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
    if decision["decision"] == "TRADE":
        side = decision.get("side", "buy").lower()

        if not side:
            side = "buy"  # Fallback

        if DRY_RUN:
            print(f"[DRY RUN] Would execute {side.upper()} on {symbol}")
            print(f"Entry: {signal['entry']} | SL: {signal['stop']} | TP: {signal['tp']}")

            # Save "Paper Trade" to tracker
            save_trade({
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "type": "PAPER",
                "entry": signal["entry"],
                "stop": signal["stop"],
                "tp": signal["tp"],
                "size": "N/A"
            })

        else:
            # LIVE EXECUTION
            print(f"[LIVE] Initiating Trade on {symbol}...")

            execution_data = execute_trade(
                symbol=symbol,
                side=side,
                entry_price=signal["entry"],
                stop_loss=signal["stop"],
                take_profit=signal["tp"],
                risk_per_trade=RISK_PER_TRADE
            )

            if execution_data:
                # Log success to tracker
                save_trade({
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "type": "LIVE",
                    "entry": signal["entry"],
                    "stop": signal["stop"],
                    "tp": signal["tp"],
                    "size": execution_data["amount"],
                    "order_ids": execution_data
                })

        return result


async def run_loop():
    print("🟢 Multi-symbol bot started (60s interval)")

    # Startup report
    generate_report()

    while True:
        start = datetime.utcnow().isoformat()
        print(f"\n⏱ Scan started at {start}")

        tasks = [analyze_symbol(symbol) for symbol in SYMBOLS]
        results = await asyncio.gather(*tasks)

        for r in results:
            dec = r.get('decision', {}).get('decision', 'N/A')
            print(f"{r['symbol']} → {dec}")

        # --------------------------------------------------
        # Wait 60 seconds before next scan
        # --------------------------------------------------
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(run_loop())
